"""
run_Q2_early_repair.py -- Q2: naprawa wczesnego deficytu adopcji --
re-adopcja paczek 1-5 i budzet snu (DROGA_Q2_PLAN.md).

NOWY plik (branch droga-q) -- istniejacy kod NIETKNIETY.

Warianty (kazdy pelna sciezka Q1, te same seedy 0-4):
  q2a_readopt   : sciezka Q1 (n_dream=500) + po 19. adopcji naprawa
                  paczek 1-5: unlearn_class(light) x5 + adopt_classes
                  z payloadow PRZECHOWANYCH (1. generacja -- zero
                  rekursji snu; maszyneria I4b).
  q2b_dream2500 : sciezka Q1 z n_dream=2500 przy kazdej adopcji.

Metryka glowna: ACC = srednia finalnego wiersza class-IL po wszystkim
(identycznie jak ACC Q1).

Kryteria (Z GORY, DROGA_Q2_PLAN.md):
  per wariant pary vs kolektyw Q1: SYGNAL+ = dzwignia dziala (raport
  odzysku bariery d/6.67); SZUM = nie dziala; SYGNAL- = szkodzi.
  Obs.: acc zadan 1-5 przed/po naprawie (q2a); najlepszy wariant vs
  m1_seq_300 (czy bariera zamknieta); forgetting.

Wymaga: cache cech CIFAR-100, data/glove.6B.300d.txt,
results/Q1_collective_horizon.json, results/M1_long_horizon.json.

Tryb szybki:  python src/run_Q2_early_repair.py --smoke
Pelny:        python src/run_Q2_early_repair.py  (~25-30 min)
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import make_task_data, eval_protocols
from mars_cl_l import ReducedBackbone
from mars_cl_m import (MarsCollectiveM, TASKS20,
                       extract_or_load_cifar100_feats)
from mars_cl_n import unlearn_class
from mars_cl_semantic import load_word_vectors
from mars_collective import MarsCollective  # noqa: F401 (rejestr klas)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
Q1_REF = os.path.join(RESULTS_DIR, "Q1_collective_horizon.json")
M1_REF = os.path.join(RESULTS_DIR, "M1_long_horizon.json")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
EARLY_BATCHES = (1, 2, 3, 4, 5)
VARIANTS = {"q2a_readopt": 500, "q2b_dream2500": 2500}
Q1_BARRIER_PP = 6.67


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


def run_variant(variant, wv, task_data, seed, epochs, device):
    n_dream = VARIANTS[variant]
    col = build(wv, seed, device)
    col.init_representation([task_data[0]], epochs=epochs, lr=LR,
                            device=device)
    col.learn_task(task_data[0], epochs=epochs, lr=LR, device=device)
    stored = {}
    for t in range(1, len(task_data)):
        td = task_data[t]
        A = build(wv, seed, device)
        A.init_representation([td], epochs=epochs, lr=LR, device=device)
        A.learn_task(td, epochs=epochs, lr=LR, device=device)
        stored[t] = {c: A.export_class_stats(
            c, int((td["ytr"] == c).sum())) for c in td["classes"]}
        col.adopt_classes(td["classes"], stored[t], epochs=epochs, lr=LR,
                          device=device, n_dream=n_dream)
    seen = [c for td in task_data for c in td["classes"]]
    row_pre, _ = eval_protocols(col.forward, task_data,
                                len(task_data) - 1, seen)
    rec = {"final_row_pre": [round(v, 4) for v in row_pre],
           "ACC_pre": round(sum(row_pre) / len(row_pre), 4)}
    if variant == "q2a_readopt":
        for t in EARLY_BATCHES:
            td = task_data[t]
            for c in td["classes"]:
                unlearn_class(col, c, scrub=False, device=device)
            col.adopt_classes(td["classes"], stored[t], epochs=epochs,
                              lr=LR, device=device, n_dream=n_dream)
        row, _ = eval_protocols(col.forward, task_data,
                                len(task_data) - 1, seen)
        rec["final_row"] = [round(v, 4) for v in row]
        rec["ACC"] = round(sum(row) / len(row), 4)
    else:
        rec["final_row"] = rec["final_row_pre"]
        rec["ACC"] = rec["ACC_pre"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q1 = m1 = None
    for ref, name in ((Q1_REF, "Q1"), (M1_REF, "M1")):
        if not os.path.exists(ref) and not args.smoke:
            sys.exit(f"BLAD: brak {ref} (baza par -- najpierw FULL {name}).")
    if os.path.exists(Q1_REF):
        q1 = json.load(open(Q1_REF, encoding="utf-8"))
    if os.path.exists(M1_REF):
        m1 = json.load(open(M1_REF, encoding="utf-8"))

    print("=" * 72)
    print(f"Q2 -- naprawa wczesnego deficytu adopcji "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={dict(VARIANTS)} | naprawa paczek={EARLY_BATCHES}")
    print("=" * 72)

    Ftr, ytr, Fte, yte = extract_or_load_cifar100_feats(device)
    task_data = make_task_data(Ftr, ytr, Fte, yte, tasks=TASKS20)
    wv = load_word_vectors("CIFAR-100", glove_path=GLOVE300,
                           device=device)

    t0 = time.perf_counter()
    out = {"experiment": "Q2_early_repair", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "variants": {k: {"n_dream": v} for k, v in VARIANTS.items()},
           "early_batches": list(EARLY_BATCHES),
           "systems": {}, "verdicts": {}}

    for variant in VARIANTS:
        per_seed = []
        for seed in range(n_seeds):
            rec = run_variant(variant, wv, task_data, seed, epochs,
                              device)
            per_seed.append(rec)
            extra = (f" (pre-naprawa {rec['ACC_pre']*100:.2f}%)"
                     if variant == "q2a_readopt" else "")
            print(f"[{variant}] seed {seed}: ACC={rec['ACC']*100:.2f}%"
                  + extra)
        out["systems"][variant] = {
            "per_seed": per_seed,
            "agg": {"ACC": stats([p["ACC"] for p in per_seed])}}

    if not args.smoke and q1:
        base = [p["class_il"]["ACC"] for p in
                q1["systems"]["collective"]["per_seed"]][:n_seeds]
        best_name, best_mean = None, -1.0
        for variant in VARIANTS:
            col = [p["ACC"] for p in out["systems"][variant]["per_seed"]]
            d = [(a - b) * 100 for a, b in zip(col, base)]
            noise = (stats([x * 100 for x in base])["std"]
                     + stats([x * 100 for x in col])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"][f"{variant}_vs_Q1"] = {
                "pairs_pp": [round(x, 2) for x in d], "delta": ds,
                "noise_pp": round(noise, 4), "verdict": v,
                "odzysk_bariery": round(ds["mean"] / Q1_BARRIER_PP, 3)}
            if stats(col)["mean"] > best_mean:
                best_mean, best_name = stats(col)["mean"], variant
        # mechanizm q2a: zadania 1-5 przed/po naprawie
        ps = out["systems"]["q2a_readopt"]["per_seed"]
        pre = [sum(p["final_row_pre"][t] for t in EARLY_BATCHES) / 5
               for p in ps]
        post = [sum(p["final_row"][t] for t in EARLY_BATCHES) / 5
                for p in ps]
        out["verdicts"]["obs_q2a_zadania_1_5"] = {
            "pre_pp": [round(x * 100, 2) for x in pre],
            "post_pp": [round(x * 100, 2) for x in post],
            "pairs_pp": [round((b - a) * 100, 2)
                         for a, b in zip(pre, post)],
            "ranga": "obserwacja"}
        # najlepszy wariant vs seq (czy bariera zamknieta)
        if m1:
            seqb = [p["class_il"]["ACC"] for p in
                    m1["systems"]["m1_seq_300"]["per_seed"]][:n_seeds]
            col = [p["ACC"] for p in
                   out["systems"][best_name]["per_seed"]]
            d = [(a - b) * 100 for a, b in zip(col, seqb)]
            noise = (stats([x * 100 for x in seqb])["std"]
                     + stats([x * 100 for x in col])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"]["obs_best_vs_seq"] = {
                "wariant": best_name,
                "pairs_pp": [round(x, 2) for x in d], "delta": ds,
                "noise_pp": round(noise, 4), "verdict": v,
                "ranga": "obserwacja (bariera zamknieta gdy SZUM)"}

    print(f"\n--- Q2 (n={n_seeds}) ---")
    for key, vd in out.get("verdicts", {}).items():
        print(f"  {key}: {json.dumps(vd, ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("Q2_early_repair_smoke.json" if args.smoke
             else "Q2_early_repair.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
