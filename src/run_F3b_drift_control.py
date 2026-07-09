"""
run_F3b_drift_control.py -- F3b: kontrola dryfu projekcji semantycznej.

Diagnoza z F3 (DROGA_F_NOTATKI.md): f3_sem = 70.8% Fashion, resztkowa luka
do replay (6.2pp) i sufitu g1_all (9.7pp) to DRYF projekcji podczas
douczania (dowod: 4 epoki < dryf < 15 epok; F 21.4pp vs 28.8pp).
F3b krecimy trzema pre-rejestrowanymi pokretlami (DROGA_F_NOTATKI, F3):

  ep4      : epochs_proj=4                (mniej czasu na dryf)
  ep8      : epochs_proj=8
  l2sp_0.1 : epochs_proj=15, l2sp=0.1     (kotwica do wag sprzed zadania)
  l2sp_1.0 : epochs_proj=15, l2sp=1.0
  k4       : epochs_proj=15, stats_k=4    (bogatszy sen: 4 centroidy/klase)
  combo    : epochs_proj=8, l2sp=0.1, stats_k=4

Kryterium werdyktu (Z GORY, class-IL Fashion, per-seed vs F0 replay-200):
  najlepszy wariant: d >= -prog szumu => SYGNAL+ ("architektura + statystyki
  + semantyka >= experience replay przy zerowym buforze") -- glowna teza F.
  Pomocniczo: vs f3_sem z F3 (ile daly pokretla) i vs g1_all (dystans do
  sufitu). Min per-seed raportowany.

Wymaga: results/F0_cl_baselines.json, results/F3_feature_replay.json,
        results/G1_semantic.json, data/glove.6B.50d.txt.

Tryb szybki:  python src/run_F3b_drift_control.py --smoke
Pelny:        python src/run_F3b_drift_control.py
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
from mars_cl_f3 import MarsCLSemanticF3
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REFS = {"f0": "F0_cl_baselines.json", "f3": "F3_feature_replay.json",
        "g1": "G1_semantic.json"}
LR = 0.001

VARIANTS = {
    "ep4":      dict(epochs_proj=4),
    "ep8":      dict(epochs_proj=8),
    "l2sp_0.1": dict(epochs_proj=15, l2sp=0.1),
    "l2sp_1.0": dict(epochs_proj=15, l2sp=1.0),
    "k4":       dict(epochs_proj=15, stats_k=4),
    "combo":    dict(epochs_proj=8, l2sp=0.1, stats_k=4),
}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(cfg, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemanticF3(wv, **cfg)
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
    return R_c, R_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--glove", default=None)
    ap.add_argument("--datasets", nargs="+",
                    default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    refs = {}
    for k, fname in REFS.items():
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                refs[k] = json.load(f)
        elif not args.smoke and k == "f0":
            sys.exit(f"BLAD: brak {p}.")

    print("=" * 72)
    print(f"F3b -- kontrola dryfu projekcji  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok(pody)={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "F3b_drift_control", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "variants": VARIANTS, "datasets": {}}

    for ds_name in args.datasets:
        kw = {"glove_path": args.glove} if args.glove else {}
        wv = load_word_vectors(ds_name, device=device, **kw)
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)
        res = {"variants": {}}

        for vname, cfg in VARIANTS.items():
            per_seed = []
            for seed in range(n_seeds):
                R_c, R_t = run_sequence(cfg, wv, task_data, seed, epochs,
                                        device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t})
                print(f"[{ds_name}] {vname:9s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp")
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            res["variants"][vname] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt ----------
        verdict = None
        if not args.smoke and "f0" in refs:
            rep = [p["class_il"]["ACC"] for p in
                   refs["f0"]["datasets"][ds_name]["methods"]["replay"]
                   ["per_seed"]][:n_seeds]
            best = max(VARIANTS, key=lambda v:
                       res["variants"][v]["agg"]["class_il_ACC"]["mean"])
            mars = [p["class_il"]["ACC"] for p
                    in res["variants"][best]["per_seed"]]
            d_rep = stats([(a - b) * 100 for a, b in zip(mars, rep)])
            noise = (stats([r * 100 for r in rep])["std"]
                     + stats([m_ * 100 for m_ in mars])["std"])
            verdict = {"best_variant": best, "delta_vs_replay_pp": d_rep,
                       "noise_pp": round(noise, 4),
                       "verdict": ("SYGNAL+ (bez bufora >= replay)"
                                   if d_rep["mean"] >= -noise
                                   else "PONIZEJ REPLAY")}
            if "f3" in refs:
                f3s = refs["f3"]["datasets"][ds_name]["variants"]["f3_sem"] \
                    ["per_seed"][:n_seeds]
                verdict["delta_vs_f3_sem_pp"] = stats(
                    [(a - p["class_il"]["ACC"]) * 100
                     for a, p in zip(mars, f3s)])
            if "g1" in refs:
                verdict["g1_all_upper_ref"] = \
                    refs["g1"]["datasets"][ds_name]["variants"]["g1_all"] \
                    ["agg"]["class_il_ACC"]["mean"]

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if "f0" in refs:
            f0m = refs["f0"]["datasets"][ds_name]["methods"]
            print(f"  [F0] replay: "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            print(f"  {vname:9s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
        if verdict:
            print(f"  WERDYKT ({verdict['best_variant']} vs replay, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            extra = ""
            if "delta_vs_f3_sem_pp" in verdict:
                extra += (f" | d vs f3_sem: "
                          f"{verdict['delta_vs_f3_sem_pp']['mean']:+.2f}pp")
            if "g1_all_upper_ref" in verdict:
                extra += (f" | sufit g1_all: "
                          f"{verdict['g1_all_upper_ref']*100:.2f}%")
            print(f"    d vs replay: "
                  f"{verdict['delta_vs_replay_pp']['mean']:+.2f}pp{extra}")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("F3b_drift_control_smoke.json" if args.smoke
             else "F3b_drift_control.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
