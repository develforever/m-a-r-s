"""
run_N1c_reinit.py -- N1c: pelne wymazanie przez reinicjalizacje
projekcji (DROGA_N_PLAN.md, sekcja N1c -- dopisana PRZED runem).

unlearn_reinit = light + reinit projekcji (deterministycznie z seeda)
+ nauka od zera na snach pozostalych 9 klas. Werdykty (pary per-seed):
  GLOWNE 4: relearn(4, n=100) po reinit vs never (z N1b, TE SAME
            seedy/probki): SZUM = PELNA gwarancja wymazania.
  GLOWNE 5: sr. acc pozostalych 9 klas po reinit vs pelny system:
            SZUM = wymazanie darmowe; SYGNAL- = cena gwarancji.
  Obserwacja: relearn po reinit vs scrub (N1b) -- taksonomia
  light/scrub/reinit.

Wymaga: data/glove.6B.300d.txt, results/N1b_relearn_balanced.json.

Tryb szybki:  python src/run_N1c_reinit.py --smoke
Pelny:        python src/run_N1c_reinit.py  (~5 min)
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import make_task_data
from mars_cl_n import class_accs, relearn_small, unlearn_reinit
from mars_collective import MarsCollective
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
N1B_REF = os.path.join(RESULTS_DIR, "N1b_relearn_balanced.json")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
C_STAR = 4
ALLOWED9 = [0, 1, 2, 3, 4, 6, 7, 8, 9]
N_RELEARN = 100


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_scrub = 256 if args.smoke else 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n1b = None
    if os.path.exists(N1B_REF):
        with open(N1B_REF, encoding="utf-8") as f:
            n1b = json.load(f)
    elif not args.smoke:
        sys.exit(f"BLAD: brak {N1B_REF} (kontrola never).")

    print("=" * 72)
    print(f"N1c -- reinit projekcji ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | c*={C_STAR}")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "N1c_reinit", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "per_seed": [], "verdicts": {}}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        full = MarsCollective(wv, **CFG)
        full.to(device)
        full.init_representation(task_data, epochs=epochs, lr=LR,
                                 device=device)
        for td in task_data:
            full.learn_task(td, epochs=epochs, lr=LR, device=device)

        before = class_accs(full, task_data, allowed=full.seen_classes)
        rest_before = sum(before[r] for r in range(10) if r != C_STAR) / 9

        mr = copy.deepcopy(full)
        unlearn_reinit(mr, C_STAR, epochs=epochs, lr=LR, device=device,
                       n_dream_scrub=n_scrub, seed=1000 + seed)
        after = class_accs(mr, task_data, allowed=mr.seen_classes)
        rest_after = sum(after[r] for r in range(10) if r != C_STAR) / 9

        td2 = task_data[2]
        idx_c = (td2["ytr"] == C_STAR).nonzero(as_tuple=True)[0]
        g = torch.Generator().manual_seed(seed)
        perm = idx_c[torch.randperm(len(idx_c), generator=g)][:N_RELEARN]
        relearn_small(mr, C_STAR, td2["Xtr"][perm], td2["ytr"][perm],
                      epochs=epochs, lr=LR, device=device,
                      balanced_negatives=True)
        acc_reinit = class_accs(mr, task_data, ALLOWED9)[C_STAR]

        out["per_seed"].append({
            "rest_before": round(rest_before, 4),
            "rest_after_reinit": round(rest_after, 4),
            "acc_relearn_reinit": round(acc_reinit, 4)})
        print(f"[Fashion] seed {seed}: reszta {rest_before*100:.1f}->"
              f"{rest_after*100:.1f}% | relearn(4) po reinit: "
              f"{acc_reinit*100:.1f}%")

    ps = out["per_seed"]
    if not args.smoke and n1b:
        nb = n1b["per_seed"][:n_seeds]
        never = [p["acc_relearn_never"] * 100 for p in nb]
        scrub = [p["acc_relearn_scrub"] * 100 for p in nb]
        rein = [p["acc_relearn_reinit"] * 100 for p in ps]
        d = [a - b for a, b in zip(rein, never)]
        noise = stats(never)["std"] + stats(rein)["std"]
        v, ds = verdict_paired(d, max(noise, 0.5))
        label = ("PELNA GWARANCJA WYMAZANIA" if v == "SZUM" else v)
        out["verdicts"]["reinit_vs_never"] = {
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(max(noise, 0.5), 4), "verdict": label}

        rb = [p["rest_before"] * 100 for p in ps]
        ra = [p["rest_after_reinit"] * 100 for p in ps]
        d2 = [a - b for a, b in zip(ra, rb)]
        noise2 = stats(rb)["std"] + stats(ra)["std"]
        v2, ds2 = verdict_paired(d2, noise2)
        out["verdicts"]["koszt_reszty"] = {
            "pairs_pp": [round(x, 2) for x in d2], "delta": ds2,
            "noise_pp": round(noise2, 4), "verdict": v2}
        out["verdicts"]["obs_reinit_vs_scrub"] = {
            "pairs_pp": [round(a - b, 2) for a, b in zip(rein, scrub)],
            "delta": stats([a - b for a, b in zip(rein, scrub)])}

    print(f"\n--- N1c (n={n_seeds}) ---")
    for k in ("rest_before", "rest_after_reinit", "acc_relearn_reinit"):
        vals = [p[k] * 100 for p in ps]
        print(f"  {k:20s}: {stats(vals)['mean']:.2f}"
              f"+/-{stats(vals)['std']:.2f}%")
    for key, vd in out.get("verdicts", {}).items():
        extra = (f" (prog {vd['noise_pp']:.2f}pp)"
                 if "noise_pp" in vd else "")
        print(f"  {key}{extra}: {vd.get('verdict', '')} "
              f"| d={vd['delta']['mean']:+.2f}pp | pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "N1c_reinit_smoke.json" if args.smoke else "N1c_reinit.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
