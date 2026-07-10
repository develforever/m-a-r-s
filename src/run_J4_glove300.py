"""
run_J4_glove300.py -- J4 (OPCJONALNY): GloVe 300d jako bogatsza geometria
slow (DROGA_J_PLAN.md, sekcja J4).

Audyt 2026-07-10: caly mechanizm semantyczny (G1/F3b/H1b) uzywa GloVe
50d -- najubozszej geometrii slow z pakietu 6B. 300d to ten sam,
publiczny, statyczny zasob (os zasobow bez zmian). Pytanie: czy bogatsza
geometria slow podnosi (a) diagnostyczny sufit g1_all, (b) uczciwy
wynik sekwencyjny k16?

Warianty (Fashion; 5 seedow):
  all_50  : proj_train="all", GloVe 50d  (reprodukcja sufitu G1: 80.45)
  all_300 : proj_train="all", GloVe 300d (nowy sufit?)
  k16_50  : sekwencyjny F3b/H1b k16, 50d (reprodukcja 77.57)
  k16_300 : sekwencyjny k16, 300d

Kryteria werdyktu (Z GORY, class-IL Fashion):
  Glowne (uczciwy CL): k16_300 vs k16_50 (pary per-seed): SYGNAL+ jesli
    sr. d > prog szumu ORAZ min > 0; SYGNAL- symetrycznie; inaczej SZUM.
  Diagnostyka: all_300 vs all_50 (czy sufit w ogole rosnie).
  Ryzyko pre-rejestrowane: projekcja 128->300 ma ~6x wiecej parametrow
    (38.4k vs 6.4k) = wieksza powierzchnia dryfu miedzy zadaniami;
    SYGNAL- jest realny i tez jest wynikiem.

Wymaga: data/glove.6B.50d.txt ORAZ data/glove.6B.300d.txt
        (python scripts/download_glove_300d.py),
        results/F0_cl_baselines.json (kontekst).

Tryb szybki:  python src/run_J4_glove300.py --smoke
Pelny:        python src/run_J4_glove300.py
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import make_task_data, cl_metrics, eval_protocols
from mars_cl_f3 import MarsCLSemanticF3
from mars_cl_semantic import MarsCLSemantic, load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE50 = os.path.join(DATA_DIR, "glove.6B.50d.txt")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
F0_REF = os.path.join(RESULTS_DIR, "F0_cl_baselines.json")
LR = 0.001

VARIANTS = {  # (tryb, wymiar)
    "all_50":   ("all", 50),
    "all_300":  ("all", 300),
    "k16_50":   ("seq", 50),
    "k16_300":  ("seq", 300),
}
SEQ_CFG = dict(dream_model="diag", stats_k=16, epochs_proj=15, l2sp=0.0)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(mode, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    if mode == "all":   # diagnostyczny sufit (projekcja widzi wszystko)
        m = MarsCLSemantic(wv, proj_train="all")
    else:               # uczciwy CL: sekwencyjny + sen k16
        m = MarsCLSemanticF3(wv, **SEQ_CFG)
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
    ap.add_argument("--datasets", nargs="+", default=["Fashion-MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p in (GLOVE50, GLOVE300):
        if not os.path.exists(p):
            sys.exit(f"BLAD: brak {p} "
                     f"(300d: python scripts/download_glove_300d.py)")
    f0_ref = None
    if os.path.exists(F0_REF):
        with open(F0_REF, encoding="utf-8") as f:
            f0_ref = json.load(f)

    print("=" * 72)
    print(f"J4 -- GloVe 300d  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "J4_glove300", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "seq_cfg": SEQ_CFG, "datasets": {}}

    for ds_name in args.datasets:
        wv50 = load_word_vectors(ds_name, glove_path=GLOVE50, device=device)
        wv300 = load_word_vectors(ds_name, glove_path=GLOVE300, device=device)
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)
        res = {"variants": {}}

        for vname, (mode, dim) in VARIANTS.items():
            wv = wv50 if dim == 50 else wv300
            per_seed = []
            for seed in range(n_seeds):
                R_c, R_t = run_sequence(mode, wv, task_data, seed, epochs,
                                        device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t})
                print(f"[{ds_name}] {vname:8s} seed {seed}: "
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
        if not args.smoke:
            def accs(v):
                return [p["class_il"]["ACC"] for p
                        in res["variants"][v]["per_seed"]]
            seq50, seq300 = accs("k16_50"), accs("k16_300")
            all50, all300 = accs("all_50"), accs("all_300")
            d_seq = stats([(a - b) * 100 for a, b in zip(seq300, seq50)])
            d_all = stats([(a - b) * 100 for a, b in zip(all300, all50)])
            noise = (stats([b * 100 for b in seq50])["std"]
                     + stats([a * 100 for a in seq300])["std"])
            if d_seq["mean"] > noise and d_seq["min"] > 0:
                v_str = "SYGNAL+ (300d podnosi uczciwy wynik)"
            elif d_seq["mean"] < -noise:
                v_str = "SYGNAL- (dryf wiekszej projekcji wygrywa z geometria)"
            else:
                v_str = "SZUM"
            verdict = {"delta_seq_300v50_pp": d_seq,
                       "delta_all_300v50_pp": d_all,
                       "noise_pp": round(noise, 4), "verdict": v_str}

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if f0_ref and ds_name in f0_ref["datasets"]:
            f0m = f0_ref["datasets"][ds_name]["methods"]
            print(f"  [F0] replay : "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            print(f"  {vname:8s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
        if verdict:
            print(f"  WERDYKT (k16_300 vs k16_50, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            print(f"    d(seq): {verdict['delta_seq_300v50_pp']['mean']:+.2f}pp"
                  f" (min {verdict['delta_seq_300v50_pp']['min']:+.2f}) | "
                  f"d(sufit all): "
                  f"{verdict['delta_all_300v50_pp']['mean']:+.2f}pp")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "J4_glove300_smoke.json" if args.smoke else "J4_glove300.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
