"""
run_F3_feature_replay.py -- F3: parametryczny feature replay (multi-seed).

Hipoteza (DROGA_F_PLAN.md, F3):
  Statystyki cech (Gaussiany per klasa, ~1 KB/klase, zero danych) zastepuja
  bufor replay: (a) pseudo-negatywy kalibruja pody miedzy zadaniami,
  (b) "sen" cech starych klas pozwala douczac projekcje semantyczna bez
  dryfu (naprawa g1_seq: forgetting 98pp).

Warianty (pre-rejestrowane):
  f3_ncm : F1d + pseudo-negatywy dla podow (routing NCM bez zmian)
  f3_sem : projekcja slowna douczana per zadanie Z replayem + pody j.w.
           (glowny kandydat; cel = poziom g1_all bez podgladania danych)

Kryteria werdyktu (Z GORY, class-IL Fashion, per-seed):
  GLOWNY: najlepszy wariant vs replay-200 (F0): d >= -prog szumu => SYGNAL+
     ("statystyki zamiast danych" -- rownowaznosc replay przy zerowym buforze)
  POMOCNICZE: f3_ncm vs F1d (ile daje sama kalibracja podow);
     f3_sem vs g1_all z G1 (czy replay domyka luke do gornej granicy).
  Min per-seed raportowany (lekcja E4).

Wymaga: results/F0_cl_baselines.json, results/F1_mars_cl.json,
        results/G1_semantic.json (pelny), data/glove.6B.50d.txt.

Tryb szybki:  python src/run_F3_feature_replay.py --smoke
Pelny:        python src/run_F3_feature_replay.py
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
from mars_cl_f3 import MarsCLF3, MarsCLSemanticF3
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REFS = {"f0": "F0_cl_baselines.json", "f1": "F1_mars_cl.json",
        "g1": "G1_semantic.json"}
LR = 0.001
REPLAY_PER_CLASS = 256


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(vname, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    if vname == "f3_ncm":
        m = MarsCLF3(replay_per_class=REPLAY_PER_CLASS)
    else:
        m = MarsCLSemanticF3(wv, replay_per_class=REPLAY_PER_CLASS)
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
        elif not args.smoke and k in ("f0", "f1"):
            sys.exit(f"BLAD: brak {p}.")

    print("=" * 72)
    print(f"F3 -- parametryczny feature replay  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs} | "
          f"replay/klase={REPLAY_PER_CLASS} (statystyki, nie dane)")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "F3_feature_replay", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "replay_per_class": REPLAY_PER_CLASS, "datasets": {}}

    for ds_name in args.datasets:
        kw = {"glove_path": args.glove} if args.glove else {}
        wv = load_word_vectors(ds_name, device=device, **kw)
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)
        res = {"variants": {}}

        for vname in ("f3_ncm", "f3_sem"):
            per_seed = []
            for seed in range(n_seeds):
                R_c, R_t = run_sequence(vname, wv, task_data, seed,
                                        epochs, device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t})
                print(f"[{ds_name}] {vname:7s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp | "
                      f"task-IL ACC={m_t['ACC']*100:.2f}%")
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            res["variants"][vname] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt ----------
        verdict = None
        if not args.smoke and "f0" in refs and "f1" in refs:
            rep = [p["class_il"]["ACC"] for p in
                   refs["f0"]["datasets"][ds_name]["methods"]["replay"]
                   ["per_seed"]][:n_seeds]
            f1d = [p["class_il"]["ACC"] for p in
                   refs["f1"]["datasets"][ds_name]["variants"]["F1d"]
                   ["per_seed"]][:n_seeds]
            best = max(("f3_ncm", "f3_sem"), key=lambda v:
                       res["variants"][v]["agg"]["class_il_ACC"]["mean"])
            mars = [p["class_il"]["ACC"] for p
                    in res["variants"][best]["per_seed"]]
            d_rep = stats([(a - b) * 100 for a, b in zip(mars, rep)])
            d_f1d = stats([(a - b) * 100 for a, b in zip(mars, f1d)])
            noise = (stats([r * 100 for r in rep])["std"]
                     + stats([m_ * 100 for m_ in mars])["std"])
            v_str = ("SYGNAL+ (statystyki zamiast danych)"
                     if d_rep["mean"] >= -noise
                     else "PONIZEJ REPLAY")
            verdict = {"best_variant": best, "delta_vs_replay_pp": d_rep,
                       "delta_vs_F1d_pp": d_f1d,
                       "noise_pp": round(noise, 4), "verdict": v_str}
            if "g1" in refs:
                try:
                    g1all = refs["g1"]["datasets"][ds_name]["variants"] \
                        ["g1_all"]["agg"]["class_il_ACC"]["mean"]
                    verdict["g1_all_upper_ref"] = g1all
                except KeyError:
                    pass

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if "f0" in refs:
            f0m = refs["f0"]["datasets"][ds_name]["methods"]
            print(f"  [F0] replay: "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%")
        if "f1" in refs:
            f1a = refs["f1"]["datasets"][ds_name]["variants"]["F1d"]["agg"]
            print(f"  [F1] F1d   : {f1a['class_il_ACC']['mean']*100:.2f}%")
        for vname in ("f3_ncm", "f3_sem"):
            a = res["variants"][vname]["agg"]
            print(f"  {vname:7s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
        if verdict:
            print(f"  WERDYKT ({verdict['best_variant']} vs replay, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            extra = (f" | g1_all(ref): {verdict['g1_all_upper_ref']*100:.2f}%"
                     if "g1_all_upper_ref" in verdict else "")
            print(f"    d vs replay: {verdict['delta_vs_replay_pp']['mean']:+.2f}pp"
                  f" | d vs F1d: {verdict['delta_vs_F1d_pp']['mean']:+.2f}pp"
                  f"{extra}")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("F3_feature_replay_smoke.json" if args.smoke
             else "F3_feature_replay.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
