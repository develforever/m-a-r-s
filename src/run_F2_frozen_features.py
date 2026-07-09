"""
run_F2_frozen_features.py -- F2: jakosc ZAMROZONEJ reprezentacji ogolnej.

Diagnoza z F1 (DROGA_F_NOTATKI.md):
  Architektura tlumi zapominanie (F ~10-16pp, MAC staly), ale absolutny
  poziom ogranicza jakosc zamrozonych cech: losowy backbone (F1d) bije
  trenowany na task0 o 24-32pp, a do replay-200 brakuje ~17-23pp.
  Cala gra toczy sie o lepsza reprezentacje OGOLNA, zdobyta uczciwie
  (bez etykiet przyszlych zadan).

Warianty (pre-rejestrowane; proto=mean, reszta jak F1):
  rand_8_16  : losowy S2 (= F1d, wewnetrzna referencja, te same seedy)
  rand_16_32 : losowy, 2x szerszy -- losowosc jest DARMOWA, placimy MAC
  rand_32_64 : losowy, 4x szerszy
  ae0_8_16   : autoenkoder na OBRAZACH task0 (bez etykiet), S2
  ae0_16_32  : jw., szerszy

Kryterium werdyktu (Z GORY, class-IL Fashion, vs F0 per-seed):
  najlepszy wariant vs replay-200: ACC >= replay - prog szumu => SYGNAL+
  ("architektura zamiast pamieci" uratowana lepszymi cechami);
  inaczej: raport ile brakuje + krzywa ACC vs MAC (Pareto cech).
  Dodatkowo: kazdy wariant vs rand_8_16 (czy szerokosc/AE w ogole pomaga;
  raportowac min per-seed -- lekcja E4).

Wymaga: results/F0_cl_baselines.json.

Tryb szybki:  python src/run_F2_frozen_features.py --smoke
Pelny:        python src/run_F2_frozen_features.py
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import make_task_data, eval_protocols, cl_metrics
from mars_cl import MarsCLSystem
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
F0_PATH = os.path.join(RESULTS_DIR, "F0_cl_baselines.json")

VARIANTS = {
    "rand_8_16":  dict(backbone_source="random", channels=(8, 16)),
    "rand_16_32": dict(backbone_source="random", channels=(16, 32)),
    "rand_32_64": dict(backbone_source="random", channels=(32, 64)),
    "ae0_8_16":   dict(backbone_source="ae0",    channels=(8, 16)),
    "ae0_16_32":  dict(backbone_source="ae0",    channels=(16, 32)),
}
LR = 0.001


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(cfg, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSystem(proto_mode="mean", **cfg)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    R_c, R_t = [], []
    seen = []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row_c, row_t = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row_c)
        R_t.append(row_t)
    return R_c, R_t, m.mac_per_sample()["total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="+",
                    default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f0 = None
    if os.path.exists(F0_PATH):
        with open(F0_PATH, encoding="utf-8") as f:
            f0 = json.load(f)
    elif not args.smoke:
        sys.exit("BLAD: brak results/F0_cl_baselines.json.")

    print("=" * 72)
    print(f"F2 -- zamrozone cechy: szerokosc x zrodlo  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "F2_frozen_features", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "variants": {k: {**v, "channels": list(v["channels"])}
                        for k, v in VARIANTS.items()},
           "datasets": {}}

    for ds_name in args.datasets:
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)
        res = {"variants": {}}

        for vname, cfg in VARIANTS.items():
            per_seed = []
            mac = None
            for seed in range(n_seeds):
                R_c, R_t, mac = run_sequence(cfg, task_data, seed, epochs,
                                             device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "R_task_il": R_t,
                                 "class_il": m_c, "task_il": m_t})
                print(f"[{ds_name}] {vname:11s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp | "
                      f"task-IL ACC={m_t['ACC']*100:.2f}%")
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            agg["mac_total"] = mac
            res["variants"][vname] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt vs replay ----------
        verdict = None
        if not args.smoke and f0:
            f0m = f0["datasets"][ds_name]["methods"]
            rep = [p["class_il"]["ACC"] for p
                   in f0m["replay"]["per_seed"]][:n_seeds]
            best = max(VARIANTS, key=lambda v:
                       res["variants"][v]["agg"]["class_il_ACC"]["mean"])
            mars = [p["class_il"]["ACC"] for p
                    in res["variants"][best]["per_seed"]]
            d_rep = stats([(a - b) * 100 for a, b in zip(mars, rep)])
            ref = [p["class_il"]["ACC"] for p
                   in res["variants"]["rand_8_16"]["per_seed"]]
            d_ref = stats([(a - b) * 100 for a, b in zip(mars, ref)])
            noise = (stats([r * 100 for r in rep])["std"]
                     + stats([m_ * 100 for m_ in mars])["std"])
            v_str = ("SYGNAL+ (cechy uratowaly teze)"
                     if d_rep["mean"] >= -noise
                     else "PONIZEJ REPLAY (raport Pareto cech)")
            verdict = {"best_variant": best, "delta_vs_replay_pp": d_rep,
                       "delta_vs_rand_8_16_pp": d_ref,
                       "noise_pp": round(noise, 4), "verdict": v_str}

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if f0:
            f0m = f0["datasets"][ds_name]["methods"]
            print(f"  [F0] replay : ACC "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%"
                  f" | [F0] joint: "
                  f"{f0m['joint']['agg']['class_il_ACC']['mean']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            print(f"  {vname:11s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp | "
                  f"MAC {a['mac_total']:,}")
        if verdict:
            print(f"  WERDYKT ({verdict['best_variant']} vs replay, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            print(f"    d vs replay: {verdict['delta_vs_replay_pp']['mean']:+.2f}pp"
                  f" | d vs rand_8_16: "
                  f"{verdict['delta_vs_rand_8_16_pp']['mean']:+.2f}pp")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("F2_frozen_features_smoke.json" if args.smoke
             else "F2_frozen_features.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
