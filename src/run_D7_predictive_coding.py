"""
run_D7_predictive_coding.py -- D7: routing po bledzie rekonstrukcji (multi-seed).

Hipoteza (D7_PLAN.md):
  Blad rekonstrukcji podow niesie informacje, ktorej NIE MA w logitach routera
  -> routing_pc podnosi system_acc na Fashion, mimo ze D4 (ensemble na tych
  samych cechach) nie dal nic.

Plan:
  Dla N_SEEDS seedow x {Fashion, MNIST}:
    1. Trening modelu bazowego train_phased (jak D6) -> baseline per-seed
       (routing po logitach) mierzony na TYM SAMYM modelu.
    2. Faza 1.5: dekodery D7a (target=features) i D7b (target=x) przy
       zamrozonym modelu.
    3. Ewaluacja wariantow: hard | fuse (sweep lambda) | iter (sweep k),
       osobno dla D7a i D7b.
  Porownanie CZYSTE: baseline i PC to ten sam model, rozni je tylko kanal
  routingu (per-seed parowanie idealne).

Kryterium werdyktu (D7_PLAN.md sekcja 5), Fashion, delta system vs baseline:
  delta > +(std_base + std_D7)   -> SYGNAL+ (dialog bije ensemble)
  |delta| <= prog                -> SZUM (sufit potwierdzony takze dla PC)
  delta < -(prog)                -> SYGNAL- (rekonstrukcja myli routing)
UWAGA uczciwosciowa: sweep lambda/k liczony na tescie (precedens D4) --
raportujemy CALY sweep, nie tylko najlepszy punkt.

Backbone (D6b: decyzja):
  --backbone cnn (domyslny) : pelny CNN D6 (32,64) -- najbogatsze cechy
  --backbone s3             : slim stride (16,32)  -- stress-test (luka 8.9pp)

Tryb szybki:  python src/run_D7_predictive_coding.py --smoke
Pelny:        python src/run_D7_predictive_coding.py
S3:           python src/run_D7_predictive_coding.py --backbone s3
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import train_phased, evaluate
from mars_v2_cnn import MarsV2CNNSystem
from mars_v2_slim import MarsV2SlimSystem
from mars_v2_pc import (PCDecoders, train_decoders, evaluate_pc, mac_pc)
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BB_H, EMB, POD_H = 128, 32, 24
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]   # sweep dla fuse
KS = [2, 3]                            # sweep dla iter
EPOCHS_DEC = 10                        # faza 1.5 (dekodery male, zbiegaja szybko)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def build_model(backbone, device):
    if backbone == "cnn":
        return MarsV2CNNSystem(backbone_hidden=BB_H, emb_dim=EMB,
                               pod_hidden=POD_H, channels=(32, 64)).to(device)
    if backbone == "s3":
        return MarsV2SlimSystem(backbone_hidden=BB_H, emb_dim=EMB,
                                pod_hidden=POD_H, channels=(16, 32),
                                downsample="stride", depthwise=False).to(device)
    raise ValueError(backbone)


def variant_grid():
    """Lista (target_mode, variant, param, nazwa) -- wspolna dla runu i agregacji."""
    grid = []
    for tm, tname in (("features", "D7a"), ("input", "D7b")):
        grid.append((tm, "hard", None, f"{tname}_hard"))
        for lam in LAMBDAS:
            grid.append((tm, "fuse", lam, f"{tname}_fuse_l{lam}"))
        for k in KS:
            grid.append((tm, "iter", k, f"{tname}_iter_k{k}"))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, 8+4 epok")
    ap.add_argument("--backbone", choices=["cnn", "s3"], default="cnn")
    ap.add_argument("--datasets", nargs="+", default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 8 if args.smoke else 30
    epochs_dec = 4 if args.smoke else EPOCHS_DEC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = variant_grid()

    print("=" * 72)
    print(f"D7 -- predictive coding routing  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | backbone={args.backbone} | seeds={n_seeds} | "
          f"epochs={epochs}+{epochs_dec}(dec) | fuse lambda={LAMBDAS} | iter k={KS}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "D7_predictive_coding", "backbone": args.backbone,
           "device": device, "n_seeds": n_seeds, "epochs": epochs,
           "epochs_dec": epochs_dec, "lambdas": LAMBDAS, "ks": KS,
           "datasets": {}}

    for ds_name in args.datasets:
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)

        per_seed = []
        for seed in range(n_seeds):
            # --- model bazowy + baseline (routing po logitach) ---
            torch.manual_seed(seed)
            model = build_model(args.backbone, device)
            train_phased(model, Xtr, ytr, epochs=epochs, device=device)
            r0, s0, o0 = evaluate(model, Xte, yte)
            row = {"baseline": {"routing": r0, "system": s0, "oracle": o0},
                   "pc": {}}

            # --- faza 1.5: dekodery (model zamrozony w train_decoders) ---
            torch.manual_seed(seed)  # powtarzalna inicjalizacja dekoderow
            decs = {}
            for tm, tdim in (("features", BB_H), ("input", 784)):
                dec = PCDecoders(pod_hidden=POD_H, target_dim=tdim).to(device)
                train_decoders(model, dec, Xtr, tm, epochs=epochs_dec,
                               device=device)
                decs[tm] = dec

            # --- ewaluacja wariantow PC ---
            for tm, variant, param, name in grid:
                row["pc"][name] = evaluate_pc(model, decs[tm], Xte, yte,
                                              tm, variant, param)
            per_seed.append(row)

            best = max(row["pc"], key=lambda k: row["pc"][k]["system"])
            print(f"[{ds_name}] seed {seed}: base sys={s0*100:.2f}% "
                  f"(rout {r0*100:.2f}%, orac {o0*100:.2f}%) | "
                  f"best PC: {best} sys={row['pc'][best]['system']*100:.2f}% "
                  f"(d {(row['pc'][best]['system']-s0)*100:+.2f}pp)")

        # --- agregacja ---
        agg = {"baseline": {m: stats([p["baseline"][m] for p in per_seed])
                            for m in ("routing", "system", "oracle")}}
        base_sys = [p["baseline"]["system"] for p in per_seed]
        std_base_pp = stats([s * 100 for s in base_sys])["std"]

        for _, _, _, name in grid:
            v = {m: stats([p["pc"][name][m] for p in per_seed])
                 for m in ("routing", "system")}
            v["recon_mse"] = stats([p["pc"][name]["recon_mse"]
                                    for p in per_seed])
            deltas = [(p["pc"][name]["system"] - p["baseline"]["system"]) * 100
                      for p in per_seed]
            v["delta_system_pp"] = stats(deltas)
            agg[name] = v

        # --- werdykt (najlepszy wariant; sweep na tescie -- patrz naglowek) ---
        names = [g[3] for g in grid]
        best = max(names, key=lambda n: agg[n]["delta_system_pp"]["mean"])
        d = agg[best]["delta_system_pp"]
        noise = std_base_pp + d["std"]
        if d["mean"] > noise:
            verdict = "SYGNAL+"
        elif d["mean"] < -noise:
            verdict = "SYGNAL-"
        else:
            verdict = "SZUM"
        agg["verdict"] = {"best_variant": best, "delta_pp": d,
                          "noise_pp": round(noise, 4), "verdict": verdict}

        # --- raport ---
        b = agg["baseline"]
        print(f"\n--- {ds_name} (n={n_seeds}, backbone={args.backbone}) ---")
        print(f"  baseline: sys {b['system']['mean']*100:.2f}"
              f"+/-{b['system']['std']*100:.2f}% | "
              f"rout {b['routing']['mean']*100:.2f}% | "
              f"orac {b['oracle']['mean']*100:.2f}%")
        for name in names:
            v = agg[name]
            d_ = v["delta_system_pp"]
            print(f"  {name:16s}: sys {v['system']['mean']*100:.2f}"
                  f"+/-{v['system']['std']*100:.2f}% | "
                  f"rout {v['routing']['mean']*100:.2f}% | "
                  f"dSys {d_['mean']:+.2f}+/-{d_['std']:.2f}pp | "
                  f"mse {v['recon_mse']['mean']:.4f}")
        vd = agg["verdict"]
        print(f"  WERDYKT ({vd['best_variant']}, prog {vd['noise_pp']:.2f}pp): "
              f"{vd['verdict']}")

        # --- MAC (koszt dodatkowy kanalu PC, per wariant uzycia) ---
        model_mac = build_model(args.backbone, "cpu")
        mac_info = {
            "top1_baseline": model_mac.mac_per_sample_top1()["total_top1"],
            "pc_all_pods_D7a": mac_pc(model_mac, BB_H, 10)["total"],
            "pc_all_pods_D7b": mac_pc(model_mac, 784, 10)["total"],
            "pc_iter_k3_D7b": mac_pc(model_mac, 784, 3)["total"],
        }
        print(f"  MAC: top1 {mac_info['top1_baseline']:,} | "
              f"PC-all D7b {mac_info['pc_all_pods_D7b']:,} | "
              f"PC-iter3 D7b {mac_info['pc_iter_k3_D7b']:,}\n")

        out["datasets"][ds_name] = {"per_seed": per_seed, "agg": agg,
                                    "mac": mac_info}

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.backbone}" if args.backbone != "cnn" else ""
    fname = (f"D7_predictive_coding{suffix}"
             f"{'_smoke' if args.smoke else ''}.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
