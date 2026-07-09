"""
run_D6b_slim_cnn.py -- D6b: odchudzone warianty CNN backbone (multi-seed).

Pytanie badawcze (D6B_PLAN.md):
  Ile z +2.38pp (Fashion, D6) przetrwa przy MAC bliskim MLP (215.6k)?

Plan:
  Dla N_SEEDS seedow x {Fashion, MNIST} trenujemy warianty S1-S4
  (MarsV2SlimSystem, train_phased REUZYTY -- identyczny protokol jak D6).
  Baseline'y MLP i pelny CNN NIE sa retrenowane -- czytamy per-seed wyniki
  z results/D6_cnn_backbone.json (te same seedy 0..4, te same epoki,
  ten sam kod treningu => porownanie parami jest uczciwe).

Kryterium werdyktu (D6B_PLAN.md sekcja 4), Fashion, najlepszy wariant
z MAC <= 2.2x MLP, delta vs MLP:
  delta >= +1.0pp i > prog szumu (std_MLP + std_wariantu)  -> SYGNAL+ efekt.
  delta <  0                                               -> SYGNAL-
  inaczej                                                  -> SZUM
Dodatkowo "retention" = delta_wariantu / delta_D6 (ile zysku przetrwalo).

Tryb szybki:  python src/run_D6b_slim_cnn.py --smoke   (1 seed, 8 epok)
Pelny:        python src/run_D6b_slim_cnn.py           (5 seedow, 30 epok)
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
from mars_v2_slim import MarsV2SlimSystem
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
D6_PATH = os.path.join(RESULTS_DIR, "D6_cnn_backbone.json")

BB_H = 128           # jak CNN w D6
EMB, POD_H = 32, 24  # jak cala seria D
MAC_BUDGET = 2.2     # kryterium "efektywnosciowe": MAC <= 2.2x MLP

VARIANTS = {
    "S1_half":      dict(channels=(16, 32), downsample="maxpool", depthwise=False),
    "S2_quarter":   dict(channels=(8, 16),  downsample="maxpool", depthwise=False),
    "S3_stride":    dict(channels=(16, 32), downsample="stride",  depthwise=False),
    "S4_depthwise": dict(channels=(16, 32), downsample="maxpool", depthwise=True),
}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def load_d6_baselines():
    """Per-seed wyniki MLP i pelnego CNN z D6 (seedy 0..4, 30 epok)."""
    if not os.path.exists(D6_PATH):
        return None
    with open(D6_PATH, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 8 epok -- sanity check (bez werdyktu)")
    ap.add_argument("--datasets", nargs="+", default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 8 if args.smoke else 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d6 = load_d6_baselines()
    if d6 is None and not args.smoke:
        sys.exit("BLAD: brak results/D6_cnn_backbone.json -- pelny run D6b "
                 "wymaga baseline'ow per-seed z D6.")

    print("=" * 72)
    print(f"D6b -- odchudzone CNN backbone'y vs MLP/pelny-CNN z D6  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epochs={epochs} | "
          f"bb_h={BB_H} | budzet MAC <= {MAC_BUDGET}x MLP")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "D6b_slim_cnn", "device": device,
           "n_seeds": n_seeds, "epochs": epochs, "mac_budget": MAC_BUDGET,
           "variants": {k: {**v, "channels": list(v["channels"])}
                        for k, v in VARIANTS.items()},
           "datasets": {}}

    for ds_name in args.datasets:
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)

        per_seed = []  # per_seed[i][variant] = {routing, system, oracle}
        macs = {}
        for seed in range(n_seeds):
            row = {}
            for vname, cfg in VARIANTS.items():
                torch.manual_seed(seed)
                m = MarsV2SlimSystem(backbone_hidden=BB_H, emb_dim=EMB,
                                     pod_hidden=POD_H, **cfg).to(device)
                train_phased(m, Xtr, ytr, epochs=epochs, device=device)
                r, s, o = evaluate(m, Xte, yte)
                row[vname] = {"routing": r, "system": s, "oracle": o}
                if vname not in macs:
                    macs[vname] = m.mac_per_sample_top1()
                print(f"[{ds_name}] seed {seed} {vname:14s}: "
                      f"sys={s*100:.2f}% rout={r*100:.2f}% orac={o*100:.2f}%")
            per_seed.append(row)

        # --- agregacja ---
        agg = {}
        mlp_mac = d6["datasets"][ds_name]["agg"]["mlp_mac_total"] if d6 else 215600
        d6_ds = d6["datasets"][ds_name] if d6 else None
        mlp_sys_seed = ([p["mlp"]["system"] for p in d6_ds["per_seed"]]
                        if d6_ds else None)
        d6_delta = (d6_ds["agg"]["delta_system_pp"]["mean"] if d6_ds else None)

        for vname in VARIANTS:
            v = {}
            for metric in ("routing", "system", "oracle"):
                v[metric] = stats([p[vname][metric] for p in per_seed])
            v["mac"] = macs[vname]
            v["mac_ratio_vs_mlp"] = round(macs[vname]["total_top1"] / mlp_mac, 2)
            if mlp_sys_seed and len(mlp_sys_seed) >= n_seeds:
                deltas = [(per_seed[i][vname]["system"] - mlp_sys_seed[i]) * 100
                          for i in range(n_seeds)]
                v["delta_vs_mlp_pp"] = stats(deltas)
                if d6_delta:
                    v["retention_vs_d6"] = round(
                        v["delta_vs_mlp_pp"]["mean"] / d6_delta, 3)
            agg[vname] = v

        # --- werdykt (tylko full, tylko warianty w budzecie MAC) ---
        verdict = None
        if not args.smoke and mlp_sys_seed:
            mlp_std_pp = stats([s * 100 for s in mlp_sys_seed])["std"]
            in_budget = {k: v for k, v in agg.items()
                         if v["mac_ratio_vs_mlp"] <= MAC_BUDGET}
            if in_budget:
                best = max(in_budget,
                           key=lambda k: in_budget[k]["delta_vs_mlp_pp"]["mean"])
                d = agg[best]["delta_vs_mlp_pp"]
                noise = mlp_std_pp + d["std"]
                if d["mean"] >= 1.0 and d["mean"] > noise:
                    v_str = "SYGNAL+ (efektywnosciowy)"
                elif d["mean"] < 0:
                    v_str = "SYGNAL-"
                else:
                    v_str = "SZUM"
                verdict = {"best_in_budget": best, "delta_pp": d,
                           "noise_pp": round(noise, 4), "verdict": v_str}

        # --- raport ---
        print(f"\n--- {ds_name} (n={n_seeds}) ---")
        if d6_ds:
            a = d6_ds["agg"]
            print(f"  ref MLP     : sys {a['mlp_system']['mean']*100:.2f}% | "
                  f"MAC {a['mlp_mac_total']:>9,} (1.00x)")
            print(f"  ref CNN(D6) : sys {a['cnn_system']['mean']*100:.2f}% | "
                  f"MAC {a['cnn_mac_total']:>9,} "
                  f"({a['cnn_mac_total']/mlp_mac:.1f}x)")
        for vname in VARIANTS:
            v = agg[vname]
            extra = ""
            if "delta_vs_mlp_pp" in v:
                extra = (f" | dSys {v['delta_vs_mlp_pp']['mean']:+.2f}"
                         f"+/-{v['delta_vs_mlp_pp']['std']:.2f}pp")
                if "retention_vs_d6" in v:
                    extra += f" | retention {v['retention_vs_d6']*100:.0f}%"
            print(f"  {vname:14s}: sys {v['system']['mean']*100:.2f}"
                  f"+/-{v['system']['std']*100:.2f}% | "
                  f"MAC {v['mac']['total_top1']:>9,} "
                  f"({v['mac_ratio_vs_mlp']:.2f}x){extra}")
        if verdict:
            print(f"  WERDYKT ({verdict['best_in_budget']}, "
                  f"prog szumu {verdict['noise_pp']:.2f}pp): "
                  f"{verdict['verdict']}")
        print()

        out["datasets"][ds_name] = {"per_seed": per_seed, "agg": agg,
                                    "verdict": verdict}

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "D6b_slim_cnn_smoke.json" if args.smoke else "D6b_slim_cnn.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
