"""
run_Q1_collective_horizon.py -- Q1: kolektyw na dlugim horyzoncie --
CIFAR-100, 20 agentow, 95 klas z wiadomosci 24 KB (DROGA_Q_PLAN.md).

NOWY plik (branch droga-q) -- istniejacy kod NIETKNIETY
(MarsCollectiveM z serii M uzywany bez zmian).

Setup = M1 (TASKS20, pretrained cache, kotwice 300d, sparse k16)
x protokol I: kolektor uczy task 0 lokalnie, adoptuje taski 1..19
od 19 nadawcow (kazdy zna tylko swoj task; n_dream=500 -- parytet
licznosci klasy CIFAR-100).

Kryteria (Z GORY, DROGA_Q_PLAN.md):
  GLOWNE: ACC koncowe kolektywu vs m1_seq_300 (M1, TE SAME seedy,
  pary): SZUM = protokol przenosi sie na 100 klas; SYGNAL-/parowy- =
  zmierzony koszt skali (obserwacja: |d| vs 0.56pp z L2); SYGNAL+ =
  wymaga replikacji.
  Q1b (obserwacja): R[t][t] adopcji early(1-5) vs late(16-20) na tle
  sufitu m1_all_300 per zadanie; pary kolektyw-vs-seq na spadku.

Wymaga: data/glove.6B.300d.txt, cache cech CIFAR-100 (powstaje przy
M1), results/M1_long_horizon.json.

Tryb szybki:  python src/run_Q1_collective_horizon.py --smoke
Pelny:        python src/run_Q1_collective_horizon.py  (~15 min)
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
from mars_cl_m import (MarsCollectiveM, TASKS20,
                       extract_or_load_cifar100_feats)
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
M1_REF = os.path.join(RESULTS_DIR, "M1_long_horizon.json")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
N_DREAM_ADOPT = 500  # parytet licznosci klasy CIFAR-100 (jak w M)


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


def build(wv, seed, device):
    torch.manual_seed(seed)
    m = MarsCollectiveM(wv, backbone_module=ReducedBackbone(), **CFG)
    m.to(device)
    return m


def run_collective(wv, task_data, seed, epochs, device):
    """Kolektor: task 0 lokalnie + 19 adopcji od jednozadaniowych
    nadawcow. Zwraca pelna macierz R (class-IL) jak M1."""
    col = build(wv, seed, device)
    col.init_representation([task_data[0]], epochs=epochs, lr=LR,
                            device=device)
    col.learn_task(task_data[0], epochs=epochs, lr=LR, device=device)
    seen = list(task_data[0]["classes"])
    row, _ = eval_protocols(col.forward, task_data, 0, seen)
    R_c = [row]
    for t in range(1, len(task_data)):
        td = task_data[t]
        A = build(wv, seed, device)
        A.init_representation([td], epochs=epochs, lr=LR, device=device)
        A.learn_task(td, epochs=epochs, lr=LR, device=device)
        payloads = {c: A.export_class_stats(c, int((td["ytr"] == c).sum()))
                    for c in td["classes"]}
        col.adopt_classes(td["classes"], payloads, epochs=epochs, lr=LR,
                          device=device, n_dream=N_DREAM_ADOPT)
        seen = seen + list(td["classes"])
        row, _ = eval_protocols(col.forward, task_data, t, seen)
        R_c.append(row)
    return R_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(GLOVE300):
        sys.exit(f"BLAD: brak {GLOVE300}")
    m1 = None
    if os.path.exists(M1_REF):
        with open(M1_REF, encoding="utf-8") as f:
            m1 = json.load(f)
    elif not args.smoke:
        sys.exit(f"BLAD: brak {M1_REF} (baza par -- najpierw FULL M1).")

    print("=" * 72)
    print(f"Q1 -- kolektyw na dlugim horyzoncie, CIFAR-100 "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream_adopt={N_DREAM_ADOPT} | paczki=19x5 klas")
    print("=" * 72)

    Ftr, ytr, Fte, yte = extract_or_load_cifar100_feats(device)
    task_data = make_task_data(Ftr, ytr, Fte, yte, tasks=TASKS20)
    wv = load_word_vectors("CIFAR-100", glove_path=GLOVE300,
                           device=device)

    t0 = time.perf_counter()
    out = {"experiment": "Q1_collective_horizon", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_dream_adopt": N_DREAM_ADOPT, "cfg": CFG,
           "encoder": "resnet18_IMAGENET1K_V1 -> random frozen 512->128",
           "memory_mb_100_classes": round(100 * 24.1 / 1024, 2),
           "systems": {}, "verdicts": {}}

    per_seed = []
    for seed in range(n_seeds):
        R_c = run_collective(wv, task_data, seed, epochs, device)
        m_c = cl_metrics(R_c)
        rtt = [R_c[t][t] for t in range(len(R_c))]
        early = sum(rtt[:5]) / 5
        late = sum(rtt[15:20]) / 5 if len(rtt) >= 20 else float("nan")
        per_seed.append({"R_class_il": R_c, "class_il": m_c,
                         "Rtt": [round(v, 4) for v in rtt],
                         "adopt_early": round(early, 4),
                         "adopt_late": round(late, 4)})
        print(f"[CIFAR-100] kolektyw seed {seed}: "
              f"ACC={m_c['ACC']*100:.2f}% "
              f"F={m_c['forgetting']*100:.1f}pp | "
              f"R[t][t] 1-5: {early*100:.1f}% | 16-20: {late*100:.1f}%")

    agg = {"class_il_ACC": stats([p["class_il"]["ACC"] for p in per_seed]),
           "class_il_forgetting": stats([p["class_il"]["forgetting"]
                                         for p in per_seed])}
    out["systems"]["collective"] = {"per_seed": per_seed, "agg": agg}

    if not args.smoke and m1:
        seq = m1["systems"]["m1_seq_300"]["per_seed"][:n_seeds]
        allp = m1["systems"]["m1_all_300"]["per_seed"][:n_seeds]
        # ---------- GLOWNE: koszt protokolu ----------
        base = [p["class_il"]["ACC"] for p in seq]
        col = [p["class_il"]["ACC"] for p in per_seed]
        d = [(a - b) * 100 for a, b in zip(col, base)]
        noise = (stats([x * 100 for x in base])["std"]
                 + stats([x * 100 for x in col])["std"])
        v, ds = verdict_paired(d, noise)
        allm = m1["systems"]["m1_all_300"]["agg"]["class_il_ACC"]["mean"]
        out["verdicts"]["collective_vs_seq"] = {
            "base": "M1 m1_seq_300 (agent sekwencyjny, te same seedy)",
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(noise, 4), "verdict": v,
            "referencja_L2_pp": -0.56,
            "gap_to_all300_pp": round(
                (allm - stats(col)["mean"]) * 100, 2)}
        # ---------- Q1b: deficyt pozny adopcji (obserwacja) ----------
        drop_col = [(p["adopt_early"] - p["adopt_late"]) * 100
                    for p in per_seed]
        drop_seq = [(p["plast_early"] - p["plast_late"]) * 100
                    for p in seq]
        pairs = [round(c - s, 2) for c, s in zip(drop_col, drop_seq)]
        ratio_e, ratio_l = [], []
        for pc, pa in zip(per_seed, allp):
            all_final = pa["R_class_il"][-1]
            r = [pc["Rtt"][t] / max(all_final[t], 1e-9)
                 for t in range(len(pc["Rtt"]))]
            ratio_e.append(sum(r[:5]) / 5)
            ratio_l.append(sum(r[15:20]) / 5)
        out["verdicts"]["obs_deficyt_adopcji"] = {
            "drop_early_late_col_pp": [round(x, 2) for x in drop_col],
            "drop_early_late_seq_pp": [round(x, 2) for x in drop_seq],
            "pairs_col_minus_seq_pp": pairs,
            "ratio_to_ceiling_early": round(
                sum(ratio_e) / len(ratio_e), 4),
            "ratio_to_ceiling_late": round(
                sum(ratio_l) / len(ratio_l), 4),
            "ranga": "obserwacja (pierwszy pomiar osi)"}

    print(f"\n--- Q1 (n={n_seeds}) ---")
    if m1:
        a = m1["systems"]["m1_seq_300"]["agg"]["class_il_ACC"]
        print(f"  [M1] agent seq : {a['mean']*100:.2f}"
              f"+/-{a['std']*100:.2f}%")
    print(f"  kolektyw (N=20): {agg['class_il_ACC']['mean']*100:.2f}"
          f"+/-{agg['class_il_ACC']['std']*100:.2f}% "
          f"(min {agg['class_il_ACC']['min']*100:.2f}%)")
    for key in ("collective_vs_seq", "obs_deficyt_adopcji"):
        if key in out["verdicts"]:
            vd = out["verdicts"][key]
            print(f"  {key}: {json.dumps(vd, ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("Q1_collective_horizon_smoke.json" if args.smoke
             else "Q1_collective_horizon.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
