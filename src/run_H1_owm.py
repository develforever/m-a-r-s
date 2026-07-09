"""
run_H1_owm.py -- H1: OWM na projekcji semantycznej (multi-seed).

Hipoteza (DROGA_H_PLAN.md):
  Rzutnik ortogonalny eliminuje resztkowy dryf projekcji, ktorego sen
  parametryczny nie domknal (F3b: F=15.8pp przy 27pp replay). Z OWM dlugi
  trening projekcji (15 epok) przestaje szkodzic -- ochrona matematyczna,
  nie czasowa.

Warianty (pre-rejestrowane; wszystkie stats_k=4, epochs_proj=15, l2sp=0):
  owm_a1        : OWM alpha=1.0, BEZ snu przy fit_proj (czysty test OWM;
                  sen zostaje dla podow)
  owm_dream_a1  : OWM alpha=1.0 + sen (glowny kandydat: zbroja + negatywy)
  owm_dream_a01 : jw., alpha=0.1 (ostrzejszy rzutnik)
  owm_dream_a10 : jw., alpha=10  (lagodniejszy)

Kryteria werdyktu (Z GORY, class-IL Fashion, per-seed):
  GLOWNY: najlepszy wariant vs f3b_combo (75.68%): d > prog szumu => SYGNAL+
  AMBITNY: > replay-200 (76.97%) => "bez bufora POKONUJE replay"
  KONTROLA PLASTYCZNOSCI (ryzyko OWM): krzywa R[t][t] po zadaniach --
  spadek acc swiezych zadan z numerem zadania = kurczacy sie null-space.
  Min per-seed raportowany.

Wymaga: results/F0_cl_baselines.json, results/F3b_drift_control.json,
        results/G1_semantic.json, data/glove.6B.50d.txt.

Tryb szybki:  python src/run_H1_owm.py --smoke
Pelny:        python src/run_H1_owm.py
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
from mars_cl_owm import MarsCLSemanticOWM
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REFS = {"f0": "F0_cl_baselines.json", "f3b": "F3b_drift_control.json",
        "g1": "G1_semantic.json"}
LR = 0.001

VARIANTS = {
    "owm_a1":        dict(owm_alpha=1.0, use_dream=False),
    "owm_dream_a1":  dict(owm_alpha=1.0, use_dream=True),
    "owm_dream_a01": dict(owm_alpha=0.1, use_dream=True),
    "owm_dream_a10": dict(owm_alpha=10.0, use_dream=True),
}
COMMON = dict(stats_k=4, epochs_proj=15, l2sp=0.0)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(cfg, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemanticOWM(wv, **cfg, **COMMON)
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
    plasticity = [R_c[t][t] for t in range(len(R_c))]   # acc tuz po nauce
    return R_c, R_t, plasticity


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
        elif not args.smoke and k in ("f0", "f3b"):
            sys.exit(f"BLAD: brak {p}.")

    print("=" * 72)
    print(f"H1 -- OWM na projekcji semantycznej  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "H1_owm", "device": device, "n_seeds": n_seeds,
           "epochs_per_task": epochs, "variants": VARIANTS,
           "common": COMMON, "datasets": {}}

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
                R_c, R_t, plast = run_sequence(cfg, wv, task_data, seed,
                                               epochs, device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t, "plasticity": plast})
                print(f"[{ds_name}] {vname:14s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp | "
                      f"plastycznosc R[t][t]: "
                      + " ".join(f"{p*100:.0f}" for p in plast))
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            # plastycznosc per pozycja zadania (srednia po seedach)
            T = len(per_seed[0]["plasticity"])
            agg["plasticity_by_task"] = [
                round(sum(p["plasticity"][t] for p in per_seed)
                      / len(per_seed), 4) for t in range(T)]
            res["variants"][vname] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt ----------
        verdict = None
        if not args.smoke and "f0" in refs and "f3b" in refs:
            rep = [p["class_il"]["ACC"] for p in
                   refs["f0"]["datasets"][ds_name]["methods"]["replay"]
                   ["per_seed"]][:n_seeds]
            f3b = [p["class_il"]["ACC"] for p in
                   refs["f3b"]["datasets"][ds_name]["variants"]["combo"]
                   ["per_seed"]][:n_seeds]
            best = max(VARIANTS, key=lambda v:
                       res["variants"][v]["agg"]["class_il_ACC"]["mean"])
            mars = [p["class_il"]["ACC"] for p
                    in res["variants"][best]["per_seed"]]
            d_f3b = stats([(a - b) * 100 for a, b in zip(mars, f3b)])
            d_rep = stats([(a - b) * 100 for a, b in zip(mars, rep)])
            noise = (stats([r * 100 for r in f3b])["std"]
                     + stats([m_ * 100 for m_ in mars])["std"])
            noise_rep = (stats([r * 100 for r in rep])["std"]
                         + stats([m_ * 100 for m_ in mars])["std"])
            if d_rep["mean"] > noise_rep and d_rep["min"] > 0:
                v_str = "SYGNAL++ (bez bufora POKONUJE replay)"
            elif d_f3b["mean"] > noise:
                v_str = "SYGNAL+ (OWM > sen sam)"
            elif abs(d_f3b["mean"]) <= noise:
                v_str = "SZUM vs F3b"
            else:
                v_str = "SYGNAL-"
            verdict = {"best_variant": best, "delta_vs_f3b_combo_pp": d_f3b,
                       "delta_vs_replay_pp": d_rep,
                       "noise_pp": round(noise, 4), "verdict": v_str}
            if "g1" in refs:
                verdict["g1_all_upper_ref"] = \
                    refs["g1"]["datasets"][ds_name]["variants"]["g1_all"] \
                    ["agg"]["class_il_ACC"]["mean"]

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if "f0" in refs:
            f0m = refs["f0"]["datasets"][ds_name]["methods"]
            print(f"  [F0] replay   : "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%")
        if "f3b" in refs:
            f3a = refs["f3b"]["datasets"][ds_name]["variants"]["combo"]["agg"]
            print(f"  [F3b] combo   : "
                  f"{f3a['class_il_ACC']['mean']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            print(f"  {vname:14s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp | "
                  f"plast: " + "->".join(
                      f"{p*100:.0f}" for p in a["plasticity_by_task"]))
        if verdict:
            print(f"  WERDYKT ({verdict['best_variant']}, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            extra = (f" | sufit g1_all: {verdict['g1_all_upper_ref']*100:.2f}%"
                     if "g1_all_upper_ref" in verdict else "")
            print(f"    d vs f3b_combo: "
                  f"{verdict['delta_vs_f3b_combo_pp']['mean']:+.2f}pp | "
                  f"d vs replay: {verdict['delta_vs_replay_pp']['mean']:+.2f}pp"
                  f"{extra}")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "H1_owm_smoke.json" if args.smoke else "H1_owm.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
