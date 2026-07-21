"""
run_I4b_full_repair.py -- I4b: naprawa pelnej paczki adopcyjnej
(DROGA_I4_PLAN.md, sekcja I4b -- dopisana PRZED runem).

Po ataku swap (payload 9-as-8 + uczciwy 9): unlearn_light(8) i (9),
potem ponowna adopcja {8: clean, 9: clean} razem. Pary vs sciezka
clean: acc8/acc9/wlasne -- SZUM x3 = PELNA NAPRAWA ZASIEGOWA.

Wymaga: data/glove.6B.300d.txt.

Tryb szybki:  python src/run_I4b_full_repair.py --smoke
Pelny:        python src/run_I4b_full_repair.py  (~6 min)
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
from mars_cl_i4 import forge_swap
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else 6000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(GLOVE300):
        sys.exit(f"BLAD: brak {GLOVE300}")

    print("=" * 72)
    print(f"I4b -- naprawa pelnej paczki "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs}")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)
    td4 = task_data[4]

    t0 = time.perf_counter()
    out = {"experiment": "I4b_full_repair", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "per_seed": [], "verdicts": {}}

    for seed in range(n_seeds):
        B = build(wv, seed, device)
        B.init_representation(task_data, epochs=epochs, lr=LR,
                              device=device)
        for t in range(4):
            B.learn_task(task_data[t], epochs=epochs, lr=LR,
                         device=device)
        A = build(wv, seed, device)
        A.init_representation([td4], epochs=epochs, lr=LR, device=device)
        A.learn_task(td4, epochs=epochs, lr=LR, device=device)
        n8 = int((td4["ytr"] == 8).sum())
        n9 = int((td4["ytr"] == 9).sum())
        clean8 = A.export_class_stats(8, n8)
        clean9 = A.export_class_stats(9, n9)
        swap8 = forge_swap(A.export_class_stats(9, n9))

        # sciezka clean (referencja)
        Bc = copy.deepcopy(B)
        Bc.adopt_classes([8, 9], {8: clean8, 9: clean9}, epochs=epochs,
                         lr=LR, device=device, n_dream=n_dream)
        ac = class_accs(Bc, task_data, allowed=Bc.seen_classes)

        # atak swap -> naprawa PELNEJ paczki
        Ba = copy.deepcopy(B)
        Ba.adopt_classes([8, 9], {8: swap8, 9: clean9}, epochs=epochs,
                         lr=LR, device=device, n_dream=n_dream)
        unlearn_class(Ba, 8, scrub=False, device=device)
        unlearn_class(Ba, 9, scrub=False, device=device)
        Ba.adopt_classes([8, 9], {8: clean8, 9: clean9}, epochs=epochs,
                         lr=LR, device=device, n_dream=n_dream)
        ar = class_accs(Ba, task_data, allowed=Ba.seen_classes)

        rec = {"clean": {"acc8": round(ac[8], 4), "acc9": round(ac[9], 4),
                         "acc_own": round(
                             sum(ac[c] for c in range(8)) / 8, 4)},
               "repaired": {"acc8": round(ar[8], 4),
                            "acc9": round(ar[9], 4),
                            "acc_own": round(
                                sum(ar[c] for c in range(8)) / 8, 4)}}
        out["per_seed"].append(rec)
        print(f"[Fashion] seed {seed}: clean acc8={ac[8]*100:.1f} "
              f"acc9={ac[9]*100:.1f} | repaired acc8={ar[8]*100:.1f} "
              f"acc9={ar[9]*100:.1f}")

    ps = out["per_seed"]
    if not args.smoke:
        full_ok = True
        for metric in ("acc8", "acc9", "acc_own"):
            d = [(p["repaired"][metric] - p["clean"][metric]) * 100
                 for p in ps]
            base = [p["clean"][metric] * 100 for p in ps]
            new = [p["repaired"][metric] * 100 for p in ps]
            noise_t = stats(base)["std"] + stats(new)["std"]
            v, ds = verdict_paired(d, max(noise_t, 0.5))
            if v not in ("SZUM",):
                full_ok = False
            out["verdicts"][f"naprawa_{metric}"] = {
                "pairs_pp": [round(x, 2) for x in d], "delta": ds,
                "noise_pp": round(max(noise_t, 0.5), 4), "verdict": v}
        out["verdicts"]["PELNA_NAPRAWA_ZASIEGOWA"] = bool(full_ok)

    print(f"\n--- I4b (n={n_seeds}) ---")
    for key, vd in out.get("verdicts", {}).items():
        if key == "PELNA_NAPRAWA_ZASIEGOWA":
            print(f"  PELNA NAPRAWA ZASIEGOWA: {vd}")
        else:
            print(f"  {key} (prog {vd['noise_pp']:.2f}pp): "
                  f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
                  f"| pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("I4b_full_repair_smoke.json" if args.smoke
             else "I4b_full_repair.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
