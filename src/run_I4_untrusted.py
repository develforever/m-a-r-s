"""
run_I4_untrusted.py -- I4: kolektyw niezaufany -- szkoda / detekcja /
naprawa (DROGA_I4_PLAN.md).

Setup jak I1: B uczy taski 0-3, adoptuje {8,9}; payload 8 w wariantach
clean / swap (payload 9 jako 8) / noise (smiec o realnych momentach).

P1 SZKODA: acc(8), acc(9), sr. acc 0-7 -- pary vs clean.
P2 DETEKCJA: D1 rank_consistency (bez adopcji), D2 canary (probna
   adopcja na kopii); kryterium: pelna separacja clean od OBU atakow
   w 5/5 seedow (min-max bez przeciecia).
P3 NAPRAWA (swap): unlearn_light(8) + re-adopcja clean -> vs clean
   (pary): SZUM = pelna naprawa lightem.

Wymaga: data/glove.6B.300d.txt.

Tryb szybki:  python src/run_I4_untrusted.py --smoke
Pelny:        python src/run_I4_untrusted.py  (~10 min)
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
from mars_cl_i4 import (canary_probe, forge_noise, forge_swap,
                        rank_consistency)
from mars_cl_n import class_accs, unlearn_class
from mars_collective import MarsCollective
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
VARIANTS = ("clean", "swap", "noise")


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
    m = MarsCollective(wv, **CFG)
    m.to(device)
    return m


def separation(clean_vals, attack_vals_list, higher_is_clean):
    """Pelna separacja: kazdy clean po wlasciwej stronie kazdego ataku."""
    attacks = [v for vals in attack_vals_list for v in vals]
    if higher_is_clean:
        return min(clean_vals) > max(attacks)
    return max(clean_vals) < min(attacks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else 6000
    n_probe = 128 if args.smoke else 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(GLOVE300):
        sys.exit(f"BLAD: brak {GLOVE300}")

    print("=" * 72)
    print(f"I4 -- kolektyw niezaufany ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)
    td_own = task_data[:4]
    td4 = task_data[4]

    t0 = time.perf_counter()
    out = {"experiment": "I4_untrusted", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "per_seed": [], "verdicts": {}}

    for seed in range(n_seeds):
        # odbiorca B: taski 0-3 (przed nadawca -- higiena RNG jak I1)
        B = build(wv, seed, device)
        B.init_representation(task_data, epochs=epochs, lr=LR,
                              device=device)
        for t in range(4):
            B.learn_task(task_data[t], epochs=epochs, lr=LR,
                         device=device)
        # nadawca A: task 4
        A = build(wv, seed, device)
        A.init_representation([td4], epochs=epochs, lr=LR, device=device)
        A.learn_task(td4, epochs=epochs, lr=LR, device=device)
        n8 = int((td4["ytr"] == 8).sum())
        n9 = int((td4["ytr"] == 9).sum())
        clean8 = A.export_class_stats(8, n8)
        clean9 = A.export_class_stats(9, n9)
        with torch.no_grad():
            pool = B.feats_batched(torch.cat(
                [td["Xtr"] for td in task_data]))
        payload8 = {"clean": clean8,
                    "swap": forge_swap(A.export_class_stats(9, n9)),
                    "noise": forge_noise(pool, k=CFG["stats_k"],
                                         n=6000, seed=seed)}

        rec = {}
        for var in VARIANTS:
            # P2: detekcja PRZED pelna adopcja
            d1 = rank_consistency(B, payload8[var], 8, device=device)
            d2 = canary_probe(B, [8, 9],
                              {8: payload8[var], 9: clean9},
                              td_own, epochs=epochs, lr=LR,
                              device=device, n_dream=n_probe)
            # P1: pelna adopcja
            Bv = copy.deepcopy(B)
            Bv.adopt_classes([8, 9], {8: payload8[var], 9: clean9},
                             epochs=epochs, lr=LR, device=device,
                             n_dream=n_dream)
            a = class_accs(Bv, task_data, allowed=Bv.seen_classes)
            rec[var] = {
                "D1_rank": round(d1, 4), "D2_canary_pp": round(d2, 4),
                "acc8": round(a[8], 4), "acc9": round(a[9], 4),
                "acc_own": round(sum(a[c] for c in range(8)) / 8, 4)}
            # P3: naprawa po swap
            if var == "swap":
                unlearn_class(Bv, 8, scrub=False, device=device)
                Bv.adopt_classes([8], {8: clean8}, epochs=epochs,
                                 lr=LR, device=device, n_dream=n_dream)
                ar = class_accs(Bv, task_data, allowed=Bv.seen_classes)
                rec["repair"] = {
                    "acc8": round(ar[8], 4), "acc9": round(ar[9], 4),
                    "acc_own": round(
                        sum(ar[c] for c in range(8)) / 8, 4)}
        out["per_seed"].append(rec)
        print(f"[Fashion] seed {seed}: "
              + " | ".join(f"{v}: acc8={rec[v]['acc8']*100:.1f} "
                           f"D1={rec[v]['D1_rank']:+.2f} "
                           f"D2={rec[v]['D2_canary_pp']:+.2f}pp"
                           for v in VARIANTS)
              + f" | repair acc8={rec['repair']['acc8']*100:.1f}")

    ps = out["per_seed"]
    if not args.smoke:
        # P1: szkoda (pary vs clean)
        for metric in ("acc8", "acc9", "acc_own"):
            for var in ("swap", "noise"):
                d = [(p[var][metric] - p["clean"][metric]) * 100
                     for p in ps]
                base = [p["clean"][metric] * 100 for p in ps]
                new = [p[var][metric] * 100 for p in ps]
                noise_t = stats(base)["std"] + stats(new)["std"]
                v, ds = verdict_paired(d, noise_t)
                out["verdicts"][f"szkoda_{var}_{metric}"] = {
                    "pairs_pp": [round(x, 2) for x in d], "delta": ds,
                    "noise_pp": round(noise_t, 4), "verdict": v}
        # P2: separacja detektorow
        for det, hic in (("D1_rank", True), ("D2_canary_pp", False)):
            cl = [p["clean"][det] for p in ps]
            at = [[p[v][det] for p in ps] for v in ("swap", "noise")]
            out["verdicts"][f"detekcja_{det}"] = {
                "clean": [round(x, 3) for x in cl],
                "swap": [round(x, 3) for x in at[0]],
                "noise": [round(x, 3) for x in at[1]],
                "pelna_separacja": separation(cl, at, hic)}
        # P3: naprawa vs clean
        for metric in ("acc8", "acc9", "acc_own"):
            d = [(p["repair"][metric] - p["clean"][metric]) * 100
                 for p in ps]
            base = [p["clean"][metric] * 100 for p in ps]
            new = [p["repair"][metric] * 100 for p in ps]
            noise_t = stats(base)["std"] + stats(new)["std"]
            v, ds = verdict_paired(d, noise_t)
            label = ("PELNA NAPRAWA" if v == "SZUM" else v)
            out["verdicts"][f"naprawa_{metric}"] = {
                "pairs_pp": [round(x, 2) for x in d], "delta": ds,
                "noise_pp": round(noise_t, 4), "verdict": label}

    print(f"\n--- I4 (n={n_seeds}) ---")
    for key, vd in out.get("verdicts", {}).items():
        if key.startswith("detekcja"):
            print(f"  {key}: pelna separacja = {vd['pelna_separacja']} "
                  f"| clean {vd['clean']} | swap {vd['swap']} | "
                  f"noise {vd['noise']}")
        else:
            print(f"  {key} (prog {vd['noise_pp']:.2f}pp): "
                  f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
                  f"| pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("I4_untrusted_smoke.json" if args.smoke
             else "I4_untrusted.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
