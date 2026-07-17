"""
run_K1_sparse300.py -- K1: zlozenie dzwigni sen sparse_k16 x GloVe 300d
(DROGA_K_PLAN.md, sekcja K1).

Motywacja: J3/J2b -- rzadkosc snu to jedyna dzwignia SYGNAL+ projektu;
J4 -- 300d podnosi sufit (+0.72, 5/5 par), ale seq z diag_k16 to null.
Pytanie: czy wierniejszy sen (sparse) pozwala projekcji 128->300
skonsumowac bogatsza geometrie, ktorej diag nie skonsumowal?

Warianty (epochs_proj=15, l2sp=0, bez kondycjonowania):
  fashion_sp16_300 : Fashion, sparse_k16 x 300d
  cifar_sp16_300   : CIFAR-n, sparse_k16 x 300d

Bazy par (TE SAME seedy, bez re-runu; konstrukcja odtwarza kolejnosc
konsumpcji RNG z run_J3/run_J2b):
  Fashion: results/J3_sparse_dreams.json  sparse_k16 (78.49 +/- 0.91)
  CIFAR:   results/J2b_cifar_sparse.json  sparse_k16 (37.51 +/- 1.35)

Kryteria werdyktu (Z GORY, class-IL, per dataset):
  SYGNAL+ : sr. d > prog szumu (std+std) ORAZ min per-seed > 0;
  SYGNAL- : sr. d < -prog;
  SYGNAL-parowy+/- (NOWE od K): wszystkie pary jednego znaku ORAZ
    |sr. delt| > 2*std(delt) -- sprawdzane tylko gdy klasyczny = SZUM;
  inaczej SZUM.
  Ryzyko pre-rejestrowane (z J4): projekcja 128->300 = 6x parametrow =
  wiekszy dryf; sen sparse moze go nie utrzymac -> SYGNAL- realny.

Wymaga: data/glove.6B.50d.txt, data/glove.6B.300d.txt,
        results/J3_sparse_dreams.json, results/J2b_cifar_sparse.json,
        opcjonalnie results/J4_glove300.json, K0 (kontekst sufitow).

Tryb szybki:  python src/run_K1_sparse300.py --smoke
Pelny:        python src/run_K1_sparse300.py
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
from cifar_cl import CifarBackbone
from mars_cl_j import MarsCLSemanticF3J, load_cifar10_norm
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
J3_REF = os.path.join(RESULTS_DIR, "J3_sparse_dreams.json")
J2B_REF = os.path.join(RESULTS_DIR, "J2b_cifar_sparse.json")
LR = 0.001
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
              bn_calib=False, feat_signorm=False)

VARIANTS = {  # nazwa -> dataset
    "fashion_sp16_300": "Fashion-MNIST",
    "cifar_sp16_300": "CIFAR-10",
}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def verdict_paired(deltas_pp, noise_pp):
    """Werdykt wg DROGA_K_PLAN: SYGNAL+/- (std+std), potem parowy."""
    d = stats(deltas_pp)
    if d["mean"] > noise_pp and d["min"] > 0:
        v = "SYGNAL+"
    elif d["mean"] < -noise_pp:
        v = "SYGNAL-"
    elif all(x > 0 for x in deltas_pp) and d["mean"] > 2 * d["std"]:
        v = "SYGNAL-parowy+"
    elif all(x < 0 for x in deltas_pp) and -d["mean"] > 2 * d["std"]:
        v = "SYGNAL-parowy-"
    else:
        v = "SZUM"
    return v, d


def run_sequence(ds_name, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    if ds_name == "CIFAR-10":   # konstrukcja jak run_J2b
        m = MarsCLSemanticF3J(wv, backbone_module=CifarBackbone(),
                              **COMMON)
    else:                       # konstrukcja jak run_J3
        m = MarsCLSemanticF3J(wv, **COMMON)
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


def load_ref(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def base_accs(ds_name, j3, j2b, n):
    if ds_name == "CIFAR-10":
        ps = j2b["systems"]["sparse_k16"]["per_seed"]
    else:
        ps = j3["datasets"][ds_name]["variants"]["sparse_k16"]["per_seed"]
    return [p["class_il"]["ACC"] for p in ps][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(GLOVE300):
        sys.exit(f"BLAD: brak {GLOVE300} "
                 f"(python scripts/download_glove_300d.py)")
    j3, j2b = load_ref(J3_REF), load_ref(J2B_REF)
    if not args.smoke and (j3 is None or j2b is None):
        sys.exit("BLAD: brak referencji J3/J2b (bazy par per-seed).")

    print("=" * 72)
    print(f"K1 -- sen sparse_k16 x GloVe 300d "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "K1_sparse300", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "common": COMMON, "systems": {}, "verdicts": {}}

    for name, ds_name in VARIANTS.items():
        wv = load_word_vectors(ds_name, glove_path=GLOVE300, device=device)
        if ds_name == "CIFAR-10":
            Xtr, ytr, Xte, yte = load_cifar10_norm(device)
        else:
            Xtr, ytr, Xte, yte = load_dataset(ds_name, device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)

        per_seed = []
        for seed in range(n_seeds):
            R_c, R_t = run_sequence(ds_name, wv, task_data, seed, epochs,
                                    device)
            m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
            per_seed.append({"R_class_il": R_c, "class_il": m_c,
                             "task_il": m_t})
            print(f"[{ds_name}] {name:16s} seed {seed}: "
                  f"class-IL ACC={m_c['ACC']*100:.2f}% "
                  f"F={m_c['forgetting']*100:.1f}pp")
        agg = {"class_il_ACC": stats([p["class_il"]["ACC"]
                                      for p in per_seed]),
               "class_il_forgetting": stats(
                   [p["class_il"]["forgetting"] for p in per_seed])}
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt per dataset ----------
        if not args.smoke and j3 and j2b:
            base = base_accs(ds_name, j3, j2b, n_seeds)
            new = [p["class_il"]["ACC"] for p in per_seed]
            deltas = [(a - b) * 100 for a, b in zip(new, base)]
            noise = (stats([b * 100 for b in base])["std"]
                     + stats([a * 100 for a in new])["std"])
            v, d = verdict_paired(deltas, noise)
            out["verdicts"][name] = {
                "base": "sparse_k16 x 50d (J3/J2b)",
                "delta_pp": d, "noise_pp": round(noise, 4),
                "pairs": [round(x, 2) for x in deltas], "verdict": v}

    # ---------- raport ----------
    print(f"\n--- K1 (n={n_seeds}) -- class-IL ---")
    for name in VARIANTS:
        a = out["systems"][name]["agg"]
        line = (f"  {name:16s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
        print(line)
        if name in out["verdicts"]:
            vd = out["verdicts"][name]
            print(f"    WERDYKT vs {vd['base']} "
                  f"(prog {vd['noise_pp']:.2f}pp): {vd['verdict']} | "
                  f"d={vd['delta_pp']['mean']:+.2f}pp "
                  f"(min {vd['delta_pp']['min']:+.2f}) | "
                  f"pary {vd['pairs']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("K1_sparse300_smoke.json" if args.smoke
             else "K1_sparse300.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
