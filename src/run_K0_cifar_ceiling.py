"""
run_K0_cifar_ceiling.py -- K0: brakujacy sufit zamrozonych cech na CIFAR
(DROGA_K_PLAN.md, sekcja K0 -- diagnostyka, bez werdyktu SYGNAL/SZUM).

Motywacja: na Fashion sufit mechanizmu to g1_all (80.45 przy 50d,
81.16 przy 300d); na CIFAR nigdy go nie zmierzono -- F4/J2 raportuja
tylko joint 70.24, ktory jest TRENOWALNY, wiec nie jest sufitem
zamrozonych cech. K0 rozstrzyga, ile z luki 37.51 -> 70.24 jest
mechanizmowe (do wziecia w K/I), a ile reprezentacyjne (Etap L).

Warianty (wejscie znormalizowane jak J2/J2b; proj_train="all"):
  all_50  : GloVe 50d
  all_300 : GloVe 300d

Raport (Z GORY): sufit_50, sufit_300,
  gap_mech = max(sufit) - 37.51 (J2b sparse_k16),
  gap_repr = 70.24 (J2 joint) - max(sufit).
Interpretacja pre-rejestrowana: gap_mech < 3pp -> mechanizm na CIFAR
praktycznie domkniety; > 5pp -> jest przestrzen dla K2/I; 3-5pp ->
strefa szara, decyzja po K2.

Wymaga: data/glove.6B.50d.txt, data/glove.6B.300d.txt,
        results/J2_cifar_normalized.json, results/J2b_cifar_sparse.json.

Tryb szybki:  python src/run_K0_cifar_ceiling.py --smoke
Pelny:        python src/run_K0_cifar_ceiling.py
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
from mars_cl_j import load_cifar10_norm
from mars_cl_semantic import MarsCLSemantic, load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE = {50: os.path.join(DATA_DIR, "glove.6B.50d.txt"),
         300: os.path.join(DATA_DIR, "glove.6B.300d.txt")}
J2_REF = os.path.join(RESULTS_DIR, "J2_cifar_normalized.json")
J2B_REF = os.path.join(RESULTS_DIR, "J2b_cifar_sparse.json")
LR = 0.001

VARIANTS = {"all_50": 50, "all_300": 300}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_ceiling(wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemantic(wv, proj_train="all",
                       backbone_module=CifarBackbone())
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p in GLOVE.values():
        if not os.path.exists(p):
            sys.exit(f"BLAD: brak {p} "
                     f"(300d: python scripts/download_glove_300d.py)")
    j2, j2b = load_ref(J2_REF), load_ref(J2B_REF)
    if not args.smoke and (j2 is None or j2b is None):
        sys.exit("BLAD: brak referencji J2/J2b (kontekst luk).")

    print("=" * 72)
    print(f"K0 -- sufit zamrozonych cech CIFAR "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    Xtr, ytr, Xte, yte = load_cifar10_norm(device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "K0_cifar_ceiling", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "variants": {}, "systems": {}}

    for name, dim in VARIANTS.items():
        wv = load_word_vectors("CIFAR-10", glove_path=GLOVE[dim],
                               device=device)
        per_seed = []
        for seed in range(n_seeds):
            R_c, R_t = run_ceiling(wv, task_data, seed, epochs, device)
            m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
            per_seed.append({"R_class_il": R_c, "class_il": m_c,
                             "task_il": m_t})
            print(f"[CIFAR-10n] {name:8s} seed {seed}: "
                  f"class-IL ACC={m_c['ACC']*100:.2f}% "
                  f"F={m_c['forgetting']*100:.1f}pp")
        agg = {"class_il_ACC": stats([p["class_il"]["ACC"]
                                      for p in per_seed]),
               "class_il_forgetting": stats(
                   [p["class_il"]["forgetting"] for p in per_seed])}
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}
        out["variants"][name] = {"glove_dim": dim}

    # ---------- diagnostyka luk (bez werdyktu -- K0 to pomiar) ----------
    diag = None
    if not args.smoke and j2 and j2b:
        best = max(out["systems"][n]["agg"]["class_il_ACC"]["mean"]
                   for n in VARIANTS)
        mars = j2b["systems"]["sparse_k16"]["agg"]["class_il_ACC"]["mean"]
        joint = j2["systems"]["joint"]["agg"]["class_il_ACC"]["mean"]
        gap_mech = (best - mars) * 100
        gap_repr = (joint - best) * 100
        if gap_mech < 3.0:
            interp = ("mechanizm na CIFAR praktycznie domkniety "
                      "(dalszy wzrost tylko przez Etap L)")
        elif gap_mech > 5.0:
            interp = "jest przestrzen dla dzwigni mechanizmowych (K2, I)"
        else:
            interp = "strefa szara 3-5pp -- decyzja po K2"
        diag = {"ceiling_best": round(best, 4),
                "mars_sparse_k16_ref": round(mars, 4),
                "joint_ref": round(joint, 4),
                "gap_mech_pp": round(gap_mech, 2),
                "gap_repr_pp": round(gap_repr, 2),
                "interpretation": interp}

    # ---------- raport ----------
    print(f"\n--- K0: sufit zamrozonych cech, CIFAR-10n "
          f"(n={n_seeds}) -- class-IL ---")
    for name in VARIANTS:
        a = out["systems"][name]["agg"]
        print(f"  {name:8s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
              f"+/-{a['class_il_ACC']['std']*100:.2f}% "
              f"(min {a['class_il_ACC']['min']*100:.2f}%)")
    if diag:
        print(f"  gap_mech = {diag['gap_mech_pp']:+.2f}pp "
              f"(sufit {diag['ceiling_best']*100:.2f} vs "
              f"mars 37.51) | gap_repr = {diag['gap_repr_pp']:+.2f}pp "
              f"(joint 70.24)")
        print(f"  INTERPRETACJA: {diag['interpretation']}")

    out["diagnosis"] = diag
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("K0_cifar_ceiling_smoke.json" if args.smoke
             else "K0_cifar_ceiling.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
