"""
run_M1c_mid_budget.py -- M1c: punkt posredni frontu stability-plasticity
(DROGA_M_PLAN.md, sekcja M1c -- dopisana PRZED runem, 20.07.2026).

M1 (51 snow/klase) i M1b (~5) zmierzyly dwa konce frontu. M1c: punkt
posredni wybrany Z GORY (srednie geometryczne zaokraglone): sny 16,
negatywy podow 32 na stara klase.

Kryteria (Z GORY): glowne ACC vs m1_seq_300 (pary): SYGNAL+ = punkt
posredni bije oba konce (nowa domyslna konfiguracja dlugiego
horyzontu); SYGNAL-/SZUM = front ostry, M1 zostaje. Obserwacje:
F, late R[t][t], pozycja na 3-punktowej krzywej frontu.

Wymaga: results/M1_long_horizon.json, cache cech CIFAR-100.

Tryb szybki:  python src/run_M1c_mid_budget.py --smoke
Pelny:        python src/run_M1c_mid_budget.py
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
from mars_cl_m import (MarsCollectiveMBalanced, TASKS20,
                       extract_or_load_cifar100_feats)
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
M1_REF = os.path.join(RESULTS_DIR, "M1_long_horizon.json")
LR = 0.001
SEQ_CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
               bn_calib=False, feat_signorm=False)


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


def run_one(wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCollectiveMBalanced(wv, backbone_module=ReducedBackbone(),
                                dream_per_old=16, neg_per_old=32,
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

    m1 = None
    if os.path.exists(M1_REF):
        with open(M1_REF, encoding="utf-8") as f:
            m1 = json.load(f)
    elif not args.smoke:
        sys.exit(f"BLAD: brak {M1_REF} (baza par).")

    print("=" * 72)
    print(f"M1c -- punkt posredni frontu (sny 16, negatywy 32 na klase) "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs}")
    print("=" * 72)

    Ftr, ytr, Fte, yte = extract_or_load_cifar100_feats(device)
    task_data = make_task_data(Ftr, ytr, Fte, yte, tasks=TASKS20)
    wv = load_word_vectors("CIFAR-100", glove_path=GLOVE300, device=device)

    t0 = time.perf_counter()
    out = {"experiment": "M1c_mid_budget", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "seq_cfg": SEQ_CFG, "systems": {}, "verdicts": {}}

    per_seed = []
    for seed in range(n_seeds):
        R_c = run_one(wv, task_data, seed, epochs, device)
        m_c = cl_metrics(R_c)
        rtt = [R_c[t][t] for t in range(len(R_c))]
        early = sum(rtt[:5]) / 5
        late = sum(rtt[15:20]) / 5
        per_seed.append({"R_class_il": R_c, "class_il": m_c,
                         "Rtt": [round(v, 4) for v in rtt],
                         "plast_early": round(early, 4),
                         "plast_late": round(late, 4)})
        print(f"[CIFAR-100] m1c_seq_300 seed {seed}: "
              f"ACC={m_c['ACC']*100:.2f}% F={m_c['forgetting']*100:.1f}pp"
              f" | R[t][t] 1-5: {early*100:.1f}% | 16-20: {late*100:.1f}%")
    agg = {"class_il_ACC": stats([p["class_il"]["ACC"] for p in per_seed]),
           "class_il_forgetting": stats(
               [p["class_il"]["forgetting"] for p in per_seed]),
           "plast_early": stats([p["plast_early"] for p in per_seed]),
           "plast_late": stats([p["plast_late"] for p in per_seed])}
    out["systems"]["m1c_seq_300"] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykty ----------
    if not args.smoke and m1:
        base = m1["systems"]["m1_seq_300"]["per_seed"][:n_seeds]
        allp = m1["systems"]["m1_all_300"]["per_seed"][:n_seeds]
        bA = [p["class_il"]["ACC"] for p in base]
        nA = [p["class_il"]["ACC"] for p in per_seed]
        dA = [(a - b) * 100 for a, b in zip(nA, bA)]
        noiseA = (stats([x * 100 for x in bA])["std"]
                  + stats([x * 100 for x in nA])["std"])
        vA, dsA = verdict_paired(dA, noiseA)
        out["verdicts"]["ACC_vs_m1"] = {
            "pairs_pp": [round(x, 2) for x in dA], "delta": dsA,
            "noise_pp": round(noiseA, 4), "verdict": vA}

        bL = [p["plast_late"] for p in base]
        nL = [p["plast_late"] for p in per_seed]
        dL = [(a - b) * 100 for a, b in zip(nL, bL)]
        noiseL = (stats([x * 100 for x in bL])["std"]
                  + stats([x * 100 for x in nL])["std"])
        vL, dsL = verdict_paired(dL, noiseL)
        out["verdicts"]["late_Rtt_vs_m1"] = {
            "pairs_pp": [round(x, 2) for x in dL], "delta": dsL,
            "noise_pp": round(noiseL, 4), "verdict": vL}

        aA = [p["class_il"]["ACC"] for p in allp]
        out["verdicts"]["mech_pct_of_ceiling_T20"] = round(
            stats(nA)["mean"] / stats(aA)["mean"] * 100, 1)
        out["verdicts"]["late_Rtt_vs_all"] = round(
            (stats(nL)["mean"]
             - stats([p["plast_late"] for p in allp])["mean"]) * 100, 2)

    # ---------- raport ----------
    print(f"\n--- M1b (n={n_seeds}) -- CIFAR-100, 20 zadan ---")
    if m1:
        b = m1["systems"]["m1_seq_300"]["agg"]
        print(f"  [M1] seq_300 : ACC {b['class_il_ACC']['mean']*100:.2f}%"
              f" | late R[t][t] {b['plast_late']['mean']*100:.1f}%")
    print(f"  m1c_seq_300  : ACC {agg['class_il_ACC']['mean']*100:.2f}"
          f"+/-{agg['class_il_ACC']['std']*100:.2f}% | "
          f"F {agg['class_il_forgetting']['mean']*100:.1f}pp | "
          f"late R[t][t] {agg['plast_late']['mean']*100:.1f}%")
    for key in ("ACC_vs_m1", "late_Rtt_vs_m1"):
        if key in out["verdicts"]:
            vd = out["verdicts"][key]
            print(f"  {key} (prog {vd['noise_pp']:.2f}pp): {vd['verdict']}"
                  f" | d={vd['delta']['mean']:+.2f}pp | pary {vd['pairs_pp']}")
    if "mech_pct_of_ceiling_T20" in out.get("verdicts", {}):
        print(f"  mech% sufitu @T=20: "
              f"{out['verdicts']['mech_pct_of_ceiling_T20']}% "
              f"(M1: 85.8%; L1 @T=5: 96.7%) | late R[t][t] vs sufit: "
              f"{out['verdicts']['late_Rtt_vs_all']:+.2f}pp (M1: -7.77)")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("M1c_mid_budget_smoke.json" if args.smoke
             else "M1c_mid_budget.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
