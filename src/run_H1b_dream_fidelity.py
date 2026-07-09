"""
run_H1b_dream_fidelity.py -- H1b: wiernosc snu (ostatni strzal mechanizmowy).

Diagnoza z H1 (DROGA_H_NOTATKI.md): OWM wyeliminowal dryf jako wyjasnienie
-- resztkowa luka do sufitu g1_all (76.2 -> 80.45 Fashion) to WIERNOSC SNU
(rozjazd wyśnionych rozkladow z realnymi cechami decyduje, gdzie laduja
nowe slowa wzgledem starych klas). H1b podnosi rozdzielczosc snu.

Warianty (pre-rejestrowane; baza = f3b k4: epochs_proj=15, l2sp=0, bez OWM):
  k8    : 8 centroidow diagonalnych / klase   (~8 KB/klase)
  k16   : 16 centroidow diagonalnych / klase  (~16 KB/klase)
  full1 : 1 Gaussian z PELNA kowariancja      (~66 KB/klase)
  full4 : 4 klastry z pelna kowariancja       (~262 KB/klase)
Wszystko: zero przechowywanych probek (statystyki cech).

Kryteria werdyktu (Z GORY, class-IL Fashion, per-seed vs F0 replay-200):
  SYGNAL++ : d > prog szumu ORAZ min per-seed > 0
             ("bez bufora POKONUJE experience replay")
  ROWNOWAZNOSC : |d| <= prog (utrzymany wynik F3b)
  Pomocniczo: vs f3b k4 (75.78) -- ile dala rozdzielczosc snu;
  sufit: g1_all 80.45. Min per-seed raportowany.

Wymaga: results/F0_cl_baselines.json, results/F3b_drift_control.json,
        results/G1_semantic.json, data/glove.6B.50d.txt.

Tryb szybki:  python src/run_H1b_dream_fidelity.py --smoke
Pelny:        python src/run_H1b_dream_fidelity.py
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
REFS = {"f0": "F0_cl_baselines.json", "f3b": "F3b_drift_control.json",
        "g1": "G1_semantic.json"}
LR = 0.001

VARIANTS = {
    "k8":    dict(dream_model="diag", stats_k=8),
    "k16":   dict(dream_model="diag", stats_k=16),
    "full1": dict(dream_model="full", stats_k=1),
    "full4": dict(dream_model="full", stats_k=4),
}
COMMON = dict(epochs_proj=15, l2sp=0.0)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(cfg, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemanticF3(wv, **cfg, **COMMON)
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
        elif not args.smoke and k in ("f0", "f3b"):
            sys.exit(f"BLAD: brak {p}.")

    print("=" * 72)
    print(f"H1b -- wiernosc snu  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "H1b_dream_fidelity", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "variants": {k: dict(v) for k, v in VARIANTS.items()},
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
                R_c, R_t = run_sequence(cfg, wv, task_data, seed, epochs,
                                        device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t})
                print(f"[{ds_name}] {vname:6s} seed {seed}: "
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
        if not args.smoke and "f0" in refs and "f3b" in refs:
            rep = [p["class_il"]["ACC"] for p in
                   refs["f0"]["datasets"][ds_name]["methods"]["replay"]
                   ["per_seed"]][:n_seeds]
            f3k4 = [p["class_il"]["ACC"] for p in
                    refs["f3b"]["datasets"][ds_name]["variants"]["k4"]
                    ["per_seed"]][:n_seeds]
            best = max(VARIANTS, key=lambda v:
                       res["variants"][v]["agg"]["class_il_ACC"]["mean"])
            mars = [p["class_il"]["ACC"] for p
                    in res["variants"][best]["per_seed"]]
            d_rep = stats([(a - b) * 100 for a, b in zip(mars, rep)])
            d_k4 = stats([(a - b) * 100 for a, b in zip(mars, f3k4)])
            noise_rep = (stats([r * 100 for r in rep])["std"]
                         + stats([m_ * 100 for m_ in mars])["std"])
            if d_rep["mean"] > noise_rep and d_rep["min"] > 0:
                v_str = "SYGNAL++ (bez bufora POKONUJE replay)"
            elif abs(d_rep["mean"]) <= noise_rep:
                v_str = "ROWNOWAZNOSC z replay (jak F3b)"
            else:
                v_str = "PONIZEJ REPLAY"
            verdict = {"best_variant": best, "delta_vs_replay_pp": d_rep,
                       "delta_vs_f3b_k4_pp": d_k4,
                       "noise_vs_replay_pp": round(noise_rep, 4),
                       "verdict": v_str}
            if "g1" in refs:
                verdict["g1_all_upper_ref"] = \
                    refs["g1"]["datasets"][ds_name]["variants"]["g1_all"] \
                    ["agg"]["class_il_ACC"]["mean"]

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if "f0" in refs:
            f0m = refs["f0"]["datasets"][ds_name]["methods"]
            print(f"  [F0] replay : "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%")
        if "f3b" in refs:
            f3a = refs["f3b"]["datasets"][ds_name]["variants"]["k4"]["agg"]
            print(f"  [F3b] k4    : {f3a['class_il_ACC']['mean']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            print(f"  {vname:6s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
        if verdict:
            print(f"  WERDYKT ({verdict['best_variant']}, "
                  f"prog {verdict['noise_vs_replay_pp']:.2f}pp): "
                  f"{verdict['verdict']}")
            extra = (f" | sufit g1_all: {verdict['g1_all_upper_ref']*100:.2f}%"
                     if "g1_all_upper_ref" in verdict else "")
            print(f"    d vs replay: {verdict['delta_vs_replay_pp']['mean']:+.2f}pp"
                  f" (min {verdict['delta_vs_replay_pp']['min']:+.2f}) | "
                  f"d vs f3b_k4: {verdict['delta_vs_f3b_k4_pp']['mean']:+.2f}pp"
                  f"{extra}")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("H1b_dream_fidelity_smoke.json" if args.smoke
             else "H1b_dream_fidelity.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
