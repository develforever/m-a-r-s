"""
run_M1_long_horizon.py -- M1: dlugi horyzont Split-CIFAR-100, 20 zadan
x 5 klas na zamrozonym pretrained (DROGA_M_PLAN.md).

Pytanie: czy mechanizm utrzymuje zdolnosc uczenia NOWYCH klas po 20
zadaniach (loss-of-plasticity, Dohare et al. 2024)?

Warianty:
  m1_seq_300 : sekwencyjny, sparse_k16, kotwice 300d (GLOWNY)
  m1_all_300 : sufit zamrozonych cech (proj_train="all")
  m1_seq_50  : obserwacja -- stloczenie 100 kotwic w 50d

Kryteria (Z GORY):
  GLOWNE (plastycznosc, m1_seq_300): pary per-seed
  spadek = sr. R[t][t] zadan 1-5 - sr. R[t][t] zadan 16-20.
  LOSS-OF-PLASTICITY jesli sr. > prog szumu (std+std) ORAZ wszystkie
  pary dodatnie; parowy analogicznie; inaczej PLASTYCZNOSC UTRZYMANA.
  Mitygacja confoundu trudnosci: raport R[t][t] / acc_all(t)
  (sufit per zadanie jako odniesienie trudnosci).
  Obserwacje: mech% sufitu przy T=20 (kontekst L1: 96.7% przy T=5);
  seq_300 vs seq_50 (pary); pamiec 100 x 24.1 KB = 2.41 MB.

Wymaga: data/glove.6B.50d.txt, data/glove.6B.300d.txt; pierwszy run
pobiera CIFAR-100 i ekstrahuje cechy do cache (kilka minut).

Tryb szybki:  python src/run_M1_long_horizon.py --smoke
Pelny:        python src/run_M1_long_horizon.py
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
from mars_cl_l import ReducedBackbone
from mars_cl_m import (MarsCLSemanticAllM, MarsCollectiveM, TASKS20,
                       extract_or_load_cifar100_feats)
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE = {50: os.path.join(DATA_DIR, "glove.6B.50d.txt"),
         300: os.path.join(DATA_DIR, "glove.6B.300d.txt")}
LR = 0.001
SEQ_CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
               bn_calib=False, feat_signorm=False)

VARIANTS = {  # nazwa -> (tryb, wymiar kotwic)
    "m1_seq_300": ("seq", 300),
    "m1_all_300": ("all", 300),
    "m1_seq_50":  ("seq", 50),
}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def verdict_paired(deltas_pp, noise_pp):
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


def run_one(mode, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    if mode == "all":
        m = MarsCLSemanticAllM(wv, proj_train="all",
                               backbone_module=ReducedBackbone())
    else:
        m = MarsCollectiveM(wv, backbone_module=ReducedBackbone(),
                            **SEQ_CFG)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    R_c = []
    seen = []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row_c, _ = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row_c)
    return R_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p in GLOVE.values():
        if not os.path.exists(p):
            sys.exit(f"BLAD: brak {p}")

    print("=" * 72)
    print(f"M1 -- dlugi horyzont CIFAR-100, 20 zadan "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    Ftr, ytr, Fte, yte = extract_or_load_cifar100_feats(device)
    task_data = make_task_data(Ftr, ytr, Fte, yte, tasks=TASKS20)
    wv_cache = {d: load_word_vectors("CIFAR-100", glove_path=GLOVE[d],
                                     device=device) for d in (50, 300)}

    t0 = time.perf_counter()
    out = {"experiment": "M1_long_horizon", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_tasks": len(TASKS20), "seq_cfg": SEQ_CFG,
           "memory_mb_100_classes": round(100 * 24.1 / 1024, 2),
           "systems": {}, "verdicts": {}}

    for name, (mode, dim) in VARIANTS.items():
        per_seed = []
        for seed in range(n_seeds):
            R_c = run_one(mode, wv_cache[dim], task_data, seed, epochs,
                          device)
            m_c = cl_metrics(R_c)
            rtt = [R_c[t][t] for t in range(len(R_c))]
            early = sum(rtt[:5]) / 5
            late = sum(rtt[15:20]) / 5
            per_seed.append({"R_class_il": R_c, "class_il": m_c,
                             "Rtt": [round(v, 4) for v in rtt],
                             "plast_early": round(early, 4),
                             "plast_late": round(late, 4)})
            print(f"[CIFAR-100] {name:10s} seed {seed}: "
                  f"ACC={m_c['ACC']*100:.2f}% "
                  f"F={m_c['forgetting']*100:.1f}pp | "
                  f"R[t][t] 1-5: {early*100:.1f}% | 16-20: {late*100:.1f}%")
        agg = {"class_il_ACC": stats([p["class_il"]["ACC"]
                                      for p in per_seed]),
               "class_il_forgetting": stats(
                   [p["class_il"]["forgetting"] for p in per_seed]),
               "plast_early": stats([p["plast_early"] for p in per_seed]),
               "plast_late": stats([p["plast_late"] for p in per_seed])}
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykty ----------
    if not args.smoke:
        main_ps = out["systems"]["m1_seq_300"]["per_seed"]
        drop = [(p["plast_early"] - p["plast_late"]) * 100
                for p in main_ps]
        noise = (stats([p["plast_early"] * 100 for p in main_ps])["std"]
                 + stats([p["plast_late"] * 100 for p in main_ps])["std"])
        v, d = verdict_paired(drop, noise)
        label = {"SYGNAL+": "LOSS-OF-PLASTICITY",
                 "SYGNAL-parowy+": "LOSS-OF-PLASTICITY (parowy)",
                 }.get(v, "PLASTYCZNOSC UTRZYMANA"
                       if v in ("SZUM",) else f"ODWROTNY KIERUNEK ({v})")
        # mitygacja confoundu: plastycznosc wzgledem sufitu per zadanie
        allp = out["systems"]["m1_all_300"]["per_seed"]
        ratio_early, ratio_late = [], []
        for ps, pa in zip(main_ps, allp):
            all_final = pa["R_class_il"][-1]
            r = [ps["Rtt"][t] / max(all_final[t], 1e-9)
                 for t in range(len(ps["Rtt"]))]
            ratio_early.append(sum(r[:5]) / 5)
            ratio_late.append(sum(r[15:20]) / 5)
        out["verdicts"]["plastycznosc"] = {
            "drop_pp_pairs": [round(x, 2) for x in drop],
            "delta": d, "noise_pp": round(noise, 4),
            "verdict_raw": v, "verdict": label,
            "ratio_to_ceiling_early": round(
                sum(ratio_early) / len(ratio_early), 4),
            "ratio_to_ceiling_late": round(
                sum(ratio_late) / len(ratio_late), 4)}

        # obserwacje
        seqA = [p["class_il"]["ACC"] for p in main_ps]
        allA = [p["class_il"]["ACC"] for p in allp]
        seq50 = [p["class_il"]["ACC"]
                 for p in out["systems"]["m1_seq_50"]["per_seed"]]
        d50 = [(a - b) * 100 for a, b in zip(seqA, seq50)]
        n50 = (stats([x * 100 for x in seq50])["std"]
               + stats([x * 100 for x in seqA])["std"])
        v50, ds50 = verdict_paired(d50, n50)
        out["verdicts"]["mech_pct_of_ceiling_T20"] = round(
            stats(seqA)["mean"] / stats(allA)["mean"] * 100, 1)
        out["verdicts"]["anchors_300_vs_50"] = {
            "pairs_pp": [round(x, 2) for x in d50], "delta": ds50,
            "noise_pp": round(n50, 4), "verdict": f"[obserwacja] {v50}"}

    # ---------- raport ----------
    print(f"\n--- M1 (n={n_seeds}) -- CIFAR-100, 20 zadan, class-IL ---")
    for name in VARIANTS:
        a = out["systems"][name]["agg"]
        print(f"  {name:10s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
              f"+/-{a['class_il_ACC']['std']*100:.2f}% | "
              f"F {a['class_il_forgetting']['mean']*100:.1f}pp | "
              f"R[t][t] {a['plast_early']['mean']*100:.1f}"
              f"->{a['plast_late']['mean']*100:.1f}%")
    if "plastycznosc" in out.get("verdicts", {}):
        vd = out["verdicts"]["plastycznosc"]
        print(f"  WERDYKT plastycznosci (prog {vd['noise_pp']:.2f}pp): "
              f"{vd['verdict']} | spadek={vd['delta']['mean']:+.2f}pp "
              f"| pary {vd['drop_pp_pairs']}")
        print(f"    wzgledem sufitu per zadanie: "
              f"{vd['ratio_to_ceiling_early']*100:.1f}% -> "
              f"{vd['ratio_to_ceiling_late']*100:.1f}%")
        print(f"  mech% sufitu @T=20: "
              f"{out['verdicts']['mech_pct_of_ceiling_T20']}% "
              f"(kontekst L1 @T=5: 96.7%)")
        va = out["verdicts"]["anchors_300_vs_50"]
        print(f"  kotwice 300 vs 50: {va['verdict']} "
              f"| d={va['delta']['mean']:+.2f}pp | pary {va['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("M1_long_horizon_smoke.json" if args.smoke
             else "M1_long_horizon.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
