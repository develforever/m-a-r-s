"""
run_Q2d_combo.py -- Q2d: kombinacja dzwigni -- budzet snu 2500 +
re-adopcja wczesnych paczek (DROGA_Q2D_PLAN.md).

NOWY plik (branch droga-q) -- istniejacy kod NIETKNIETY.

Wariant q2d_combo: sciezka q2b (n_dream=2500 przy kazdej adopcji)
+ po 19. adopcji naprawa paczek 1-5 (unlearn light + re-adopcja
z payloadow przechowanych, n_dream=2500).

Kryteria (Z GORY, DROGA_Q2D_PLAN.md):
  GLOWNE: pary vs q2b (Q2_early_repair.json): SYGNAL+ = dzwignie
  addytywne; SZUM = budzet subsumuje naprawe; SYGNAL- = naprawa
  przy 2500 szkodzi.
  Obs.: pary vs seq_selfdream2500 (luka domknieta gdy SZUM);
  acc zadan 1-5 przed/po naprawie; luka do sufitu 47.41.

Wymaga: cache cech CIFAR-100, data/glove.6B.300d.txt,
results/Q2_early_repair.json, results/Q2c_seq_selfdream.json,
results/M1_long_horizon.json.

Tryb szybki:  python src/run_Q2d_combo.py --smoke
Pelny:        python src/run_Q2d_combo.py  (~20 min)
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
Q2_REF = os.path.join(RESULTS_DIR, "Q2_early_repair.json")
Q2C_REF = os.path.join(RESULTS_DIR, "Q2c_seq_selfdream.json")
M1_REF = os.path.join(RESULTS_DIR, "M1_long_horizon.json")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
N_DREAM = 2500
EARLY_BATCHES = (1, 2, 3, 4, 5)


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


def run_combo(wv, task_data, seed, epochs, device):
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
                          device=device, n_dream=N_DREAM)
    seen = [c for td in task_data for c in td["classes"]]
    row_pre, _ = eval_protocols(col.forward, task_data,
                                len(task_data) - 1, seen)
    for t in EARLY_BATCHES:
        td = task_data[t]
        for c in td["classes"]:
            unlearn_class(col, c, scrub=False, device=device)
        col.adopt_classes(td["classes"], stored[t], epochs=epochs,
                          lr=LR, device=device, n_dream=N_DREAM)
    row, _ = eval_protocols(col.forward, task_data,
                            len(task_data) - 1, seen)
    return {"final_row_pre": [round(v, 4) for v in row_pre],
            "ACC_pre": round(sum(row_pre) / len(row_pre), 4),
            "final_row": [round(v, 4) for v in row],
            "ACC": round(sum(row) / len(row), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    refs = {}
    for ref, name in ((Q2_REF, "Q2"), (Q2C_REF, "Q2c"), (M1_REF, "M1")):
        if os.path.exists(ref):
            refs[name] = json.load(open(ref, encoding="utf-8"))
        elif not args.smoke:
            sys.exit(f"BLAD: brak {ref} (najpierw FULL {name}).")

    print("=" * 72)
    print(f"Q2d -- kombinacja: budzet 2500 + re-adopcja "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={N_DREAM} | naprawa paczek={EARLY_BATCHES}")
    print("=" * 72)

    Ftr, ytr, Fte, yte = extract_or_load_cifar100_feats(device)
    task_data = make_task_data(Ftr, ytr, Fte, yte, tasks=TASKS20)
    wv = load_word_vectors("CIFAR-100", glove_path=GLOVE300,
                           device=device)

    t0 = time.perf_counter()
    out = {"experiment": "Q2d_combo", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "n_dream": N_DREAM, "early_batches": list(EARLY_BATCHES),
           "systems": {}, "verdicts": {}}

    per_seed = []
    for seed in range(n_seeds):
        rec = run_combo(wv, task_data, seed, epochs, device)
        per_seed.append(rec)
        print(f"[q2d_combo] seed {seed}: ACC={rec['ACC']*100:.2f}% "
              f"(pre-naprawa {rec['ACC_pre']*100:.2f}%)")
    out["systems"]["q2d_combo"] = {
        "per_seed": per_seed,
        "agg": {"ACC": stats([p["ACC"] for p in per_seed])}}

    if not args.smoke and refs:
        combo = [p["ACC"] for p in per_seed]
        # GLOWNE: vs q2b
        q2b = [p["ACC"] for p in
               refs["Q2"]["systems"]["q2b_dream2500"]["per_seed"]][:n_seeds]
        d = [(a - b) * 100 for a, b in zip(combo, q2b)]
        noise = (stats([x * 100 for x in q2b])["std"]
                 + stats([x * 100 for x in combo])["std"])
        v, ds = verdict_paired(d, noise)
        out["verdicts"]["combo_vs_q2b"] = {
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(noise, 4), "verdict": v}
        # obs: vs seq_selfdream
        sd = [p["class_il"]["ACC"] for p in
              refs["Q2c"]["systems"]["seq_selfdream2500"]["per_seed"]
              ][:n_seeds]
        d = [(a - b) * 100 for a, b in zip(combo, sd)]
        noise = (stats([x * 100 for x in sd])["std"]
                 + stats([x * 100 for x in combo])["std"])
        v, ds = verdict_paired(d, noise)
        allm = refs["M1"]["systems"]["m1_all_300"]["agg"][
            "class_il_ACC"]["mean"]
        out["verdicts"]["obs_combo_vs_selfdream"] = {
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(noise, 4), "verdict": v,
            "gap_to_all300_pp": round(
                (allm - stats(combo)["mean"]) * 100, 2),
            "ranga": "obserwacja (luka domknieta gdy SZUM)"}
        # obs: mechanizm -- zadania 1-5 przed/po
        pre = [sum(p["final_row_pre"][t] for t in EARLY_BATCHES) / 5
               for p in per_seed]
        post = [sum(p["final_row"][t] for t in EARLY_BATCHES) / 5
                for p in per_seed]
        out["verdicts"]["obs_zadania_1_5"] = {
            "pre_pp": [round(x * 100, 2) for x in pre],
            "post_pp": [round(x * 100, 2) for x in post],
            "pairs_pp": [round((b - a) * 100, 2)
                         for a, b in zip(pre, post)],
            "ranga": "obserwacja"}

    print(f"\n--- Q2d (n={n_seeds}) ---")
    for key, vd in out.get("verdicts", {}).items():
        print(f"  {key}: {json.dumps(vd, ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "Q2d_combo_smoke.json" if args.smoke else "Q2d_combo.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
