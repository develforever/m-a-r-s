"""
run_D6_cnn_backbone.py -- D6: CNN backbone v2 vs MLP backbone v2 (multi-seed).

Pytanie badawcze (z DROGA_D_NOTATKI.md):
  Czy bogatszy (CNN) shared backbone podnosi sufit routera, ktorego D4+D5 nie
  ruszyly modyfikacjami samego routingu? Seria B sugeruje +3.5pp na Fashion.

Plan:
  Dla N_SEEDS seedow x {MNIST, Fashion} trenujemy i ewaluujemy:
    - v2 MLP  (D1, phased)  -- baseline (klasa MarsV2System)
    - v2 CNN  (D6, phased)  -- nowy backbone (klasa MarsV2CNNSystem)
  Trening: train_phased z mars_v2.py (REUZYTY -- ten sam kod dla obu modeli,
  rozni je TYLKO backbone). Metryki per seed: routing_acc, system_acc, oracle.
  Raport: mean +/- std (Bessel n-1), delta CNN-MLP, interpretacja sygnal/szum.

Tryb szybki:  python src/run_D6_cnn_backbone.py --smoke   (1 seed, mniej epok)
Pelny:        python src/run_D6_cnn_backbone.py           (N_SEEDS, EPOCHS)
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import MarsV2System, train_phased, evaluate, N_IN, N_CLASSES
from mars_v2_cnn import MarsV2CNNSystem
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Baseline MLP v2: backbone_hidden ze sweepu A5 (256 daje routing ~98% na MNIST).
MLP_BB_H = 256
# CNN: kompresja do mniejszego wspolnego wektora (CNN niesie cechy w kanalach).
CNN_BB_H = 128
EMB, POD_H = 32, 24
CNN_CHANNELS = (32, 64)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_models(ds_name, Xtr, ytr, Xte, yte, seed, epochs, device):
    """Trenuje v2 MLP i v2 CNN dla jednego seeda. Zwraca dict metryk."""
    # --- v2 MLP (baseline) ---
    torch.manual_seed(seed)
    mlp = MarsV2System(backbone_hidden=MLP_BB_H, emb_dim=EMB,
                       pod_hidden=POD_H).to(device)
    train_phased(mlp, Xtr, ytr, epochs=epochs, device=device)
    r_mlp, s_mlp, o_mlp = evaluate(mlp, Xte, yte)

    # --- v2 CNN (D6) ---
    torch.manual_seed(seed)
    cnn = MarsV2CNNSystem(backbone_hidden=CNN_BB_H, emb_dim=EMB,
                          pod_hidden=POD_H, channels=CNN_CHANNELS).to(device)
    train_phased(cnn, Xtr, ytr, epochs=epochs, device=device)
    r_cnn, s_cnn, o_cnn = evaluate(cnn, Xte, yte)

    return {
        "mlp": {"routing": r_mlp, "system": s_mlp, "oracle": o_mlp},
        "cnn": {"routing": r_cnn, "system": s_cnn, "oracle": o_cnn},
        "mlp_mac": mlp.mac_per_sample_top1(),
        "cnn_mac": cnn.mac_per_sample_top1(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 8 epok -- szybki sanity check")
    ap.add_argument("--datasets", nargs="+", default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 8 if args.smoke else 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"D6 -- CNN backbone v2 vs MLP backbone v2  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epochs={epochs}")
    print(f"MLP bb_h={MLP_BB_H} | CNN bb_h={CNN_BB_H} ch={CNN_CHANNELS}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "D6_cnn_backbone", "device": device,
           "n_seeds": n_seeds, "epochs": epochs, "datasets": {}}

    for ds_name in args.datasets:
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        per_seed = []
        for seed in range(n_seeds):
            r = run_models(ds_name, Xtr, ytr, Xte, yte, seed, epochs, device)
            per_seed.append(r)
            print(f"[{ds_name}] seed {seed}: "
                  f"MLP sys={r['mlp']['system']*100:.2f}% "
                  f"(rout {r['mlp']['routing']*100:.2f}%, orac {r['mlp']['oracle']*100:.2f}%)  |  "
                  f"CNN sys={r['cnn']['system']*100:.2f}% "
                  f"(rout {r['cnn']['routing']*100:.2f}%, orac {r['cnn']['oracle']*100:.2f}%)  "
                  f"=> dSys {(r['cnn']['system']-r['mlp']['system'])*100:+.2f}pp")

        agg = {}
        for arch in ("mlp", "cnn"):
            for metric in ("routing", "system", "oracle"):
                agg[f"{arch}_{metric}"] = stats(
                    [p[arch][metric] for p in per_seed])
        delta_sys = [p["cnn"]["system"] - p["mlp"]["system"] for p in per_seed]
        agg["delta_system_pp"] = stats([d * 100 for d in delta_sys])
        agg["mlp_mac_total"] = per_seed[0]["mlp_mac"]["total_top1"]
        agg["cnn_mac_total"] = per_seed[0]["cnn_mac"]["total_top1"]

        d = agg["delta_system_pp"]
        verdict = "SYGNAL+" if d["mean"] > abs(agg["cnn_system"]["std"]) + abs(agg["mlp_system"]["std"]) \
            else ("SYGNAL-" if d["mean"] < -(abs(agg["cnn_system"]["std"]) + abs(agg["mlp_system"]["std"])) else "SZUM")
        agg["verdict"] = verdict

        print(f"\n--- {ds_name} (n={n_seeds}) ---")
        print(f"  MLP system: {agg['mlp_system']['mean']*100:.2f} +/- {agg['mlp_system']['std']*100:.2f}pp"
              f"  (routing {agg['mlp_routing']['mean']*100:.2f}%)")
        print(f"  CNN system: {agg['cnn_system']['mean']*100:.2f} +/- {agg['cnn_system']['std']*100:.2f}pp"
              f"  (routing {agg['cnn_routing']['mean']*100:.2f}%)")
        print(f"  delta system (CNN-MLP): {d['mean']:+.2f} +/- {d['std']:.2f}pp  -> {verdict}")
        print(f"  MAC: MLP {agg['mlp_mac_total']:,}  CNN {agg['cnn_mac_total']:,}\n")
        out["datasets"][ds_name] = {"per_seed": per_seed, "agg": agg}

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "D6_cnn_backbone_smoke.json" if args.smoke else "D6_cnn_backbone.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
