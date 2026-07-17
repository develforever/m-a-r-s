"""
run_I1_transplant.py -- I1: przeszczep klasy przez wymiane snow
(DROGA_I_PLAN.md, sekcja I1 + I1b).

Agent B uczy taski 0-3 Fashion z danych; agent A (ten sam seed =
ten sam backbone) uczy task 4 u siebie i wysyla payload statystyk
klas {8,9}; B adoptuje je ze snow. Warianty: adopcja na koncu
(transplant_end) vs po tasku 1 (transplant_mid, I1b).

Baza par (TE SAME seedy, bez re-runu): local = K1 fashion_sp16_300
(results/K1_sparse300.json): acc task4 = R_class_il[-1][4], ACC.
Higiena RNG: B konstruowany i uczony PRZED konstrukcja A -- strumien
RNG B identyczny z run_K1 do konca taska 3.

Kryteria (Z GORY, zatwierdzone 2026-07-17):
  strata przeszczepu = acc(task4)_local - acc(task4)_transplant_end
  (pary per-seed):
    SUKCES MOCNY : |sr.| < prog szumu (std+std)  [rownowaznosc]
    (jesli sr. < -prog: PRZESZCZEP LEPSZY -- tez raportowane)
    SUKCES SLABY : sr. < 3pp
    PORAZKA      : sr. >= 3pp  -> BRAMKA: I2/I3 wstrzymane.
  Kierunkowo na pelnym ACC: transplant_end vs local --
  SYGNAL+/-, SYGNAL-parowy+/- (nowe kryterium K), SZUM.
  I1b (obserwacja): strata mid vs end (pary) + acc klas 0-7.

Konfiguracja: zwyciezca K1 (sparse_k16 x GloVe 300d), n_dream=6000.
Wymaga: data/glove.6B.300d.txt, results/K1_sparse300.json.

Tryb szybki:  python src/run_I1_transplant.py --smoke
Pelny:        python src/run_I1_transplant.py
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
from mars_collective import MarsCollective
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
K1_REF = os.path.join(RESULTS_DIR, "K1_sparse300.json")
LR = 0.001
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
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


def build_agent(wv, seed, device):
    torch.manual_seed(seed)
    m = MarsCollective(wv, **COMMON)
    m.to(device)
    return m


def sender_payloads(wv, td4, seed, epochs, device):
    """Agent A: uczy task 4 u siebie, eksportuje payloady klas."""
    A = build_agent(wv, seed, device)
    A.init_representation([td4], epochs=epochs, lr=LR, device=device)
    A.learn_task(td4, epochs=epochs, lr=LR, device=device)
    return {c: A.export_class_stats(c, int((td4["ytr"] == c).sum()))
            for c in td4["classes"]}


def final_eval(m, task_data):
    seen = [c for td in task_data for c in td["classes"]]
    row_c, _ = eval_protocols(m.forward, task_data,
                              len(task_data) - 1, seen)
    return row_c


def run_variant(mode, wv, task_data, seed, epochs, n_dream, device):
    """mode: 'end' (adopcja po tasku 3) | 'mid' (po tasku 1, I1b)."""
    B = build_agent(wv, seed, device)
    B.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    learn_before = [0, 1, 2, 3] if mode == "end" else [0, 1]
    learn_after = [] if mode == "end" else [2, 3]
    for t in learn_before:
        B.learn_task(task_data[t], epochs=epochs, lr=LR, device=device)
    payloads = sender_payloads(wv, task_data[4], seed, epochs, device)
    B.adopt_classes(task_data[4]["classes"], payloads, epochs=epochs,
                    lr=LR, device=device, n_dream=n_dream)
    for t in learn_after:
        B.learn_task(task_data[t], epochs=epochs, lr=LR, device=device)
    row = final_eval(B, task_data)
    return {"final_row_class_il": row,
            "acc_task4": row[4],
            "acc_old_mean": sum(row[:4]) / 4,
            "ACC": sum(row) / len(row)}


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
    k1 = None
    if os.path.exists(K1_REF):
        with open(K1_REF, encoding="utf-8") as f:
            k1 = json.load(f)
    elif not args.smoke:
        sys.exit(f"BLAD: brak {K1_REF} (baza par local).")

    print("=" * 72)
    print(f"I1 -- przeszczep klasy przez wymiane snow "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream} | warianty=['transplant_end', "
          f"'transplant_mid']")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "I1_transplant", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_dream": n_dream, "common": COMMON, "systems": {},
           "verdicts": {}}

    for mode in ("end", "mid"):
        name = f"transplant_{mode}"
        per_seed = []
        for seed in range(n_seeds):
            r = run_variant(mode, wv, task_data, seed, epochs, n_dream,
                            device)
            per_seed.append(r)
            print(f"[Fashion] {name:14s} seed {seed}: "
                  f"ACC={r['ACC']*100:.2f}% | przeszczep(task4)="
                  f"{r['acc_task4']*100:.2f}% | stare(0-3)="
                  f"{r['acc_old_mean']*100:.2f}%")
        agg = {k: stats([p[k] for p in per_seed])
               for k in ("ACC", "acc_task4", "acc_old_mean")}
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykty ----------
    if not args.smoke and k1:
        lps = k1["systems"]["fashion_sp16_300"]["per_seed"]
        loc4 = [p["R_class_il"][-1][4] for p in lps][:n_seeds]
        locA = [p["class_il"]["ACC"] for p in lps][:n_seeds]

        end = out["systems"]["transplant_end"]["per_seed"]
        mid = out["systems"]["transplant_mid"]["per_seed"]
        e4 = [p["acc_task4"] for p in end]
        eA = [p["ACC"] for p in end]
        m4 = [p["acc_task4"] for p in mid]

        # glowny: strata przeszczepu (local - transplant_end)
        loss = [(a - b) * 100 for a, b in zip(loc4, e4)]
        noise4 = (stats([x * 100 for x in loc4])["std"]
                  + stats([x * 100 for x in e4])["std"])
        dl = stats(loss)
        if abs(dl["mean"]) < noise4:
            main_v = "SUKCES MOCNY (rownowaznosc przeszczepu z nauka lokalna)"
        elif dl["mean"] < -noise4:
            main_v = "PRZESZCZEP LEPSZY (strata ujemna ponad prog)"
        elif dl["mean"] < 3.0:
            main_v = "SUKCES SLABY (strata < 3pp)"
        else:
            main_v = "PORAZKA (strata >= 3pp) -> BRAMKA: I2/I3 wstrzymane"
        out["verdicts"]["strata_przeszczepu"] = {
            "pairs_pp": [round(x, 2) for x in loss], "delta": dl,
            "noise_pp": round(noise4, 4), "verdict": main_v}

        # kierunkowy: pelny ACC transplant_end vs local
        dA = [(a - b) * 100 for a, b in zip(eA, locA)]
        noiseA = (stats([x * 100 for x in locA])["std"]
                  + stats([x * 100 for x in eA])["std"])
        v, d = verdict_paired(dA, noiseA)
        out["verdicts"]["ACC_end_vs_local"] = {
            "pairs_pp": [round(x, 2) for x in dA], "delta": d,
            "noise_pp": round(noiseA, 4), "verdict": v}

        # I1b: mid vs end (obserwacja)
        dm = [(a - b) * 100 for a, b in zip(e4, m4)]  # ile mid traci vs end
        out["verdicts"]["I1b_mid_vs_end"] = {
            "pairs_pp_end_minus_mid": [round(x, 2) for x in dm],
            "delta": stats(dm), "ranga": "obserwacja"}

    # ---------- raport ----------
    print(f"\n--- I1 (n={n_seeds}) -- Fashion, class-IL ---")
    if k1:
        a = k1["systems"]["fashion_sp16_300"]["agg"]["class_il_ACC"]
        print(f"  [K1] local seq : ACC {a['mean']*100:.2f}"
              f"+/-{a['std']*100:.2f}%")
    for name in ("transplant_end", "transplant_mid"):
        a = out["systems"][name]["agg"]
        print(f"  {name:14s}: ACC {a['ACC']['mean']*100:.2f}"
              f"+/-{a['ACC']['std']*100:.2f}% | przeszczep "
              f"{a['acc_task4']['mean']*100:.2f}% | stare "
              f"{a['acc_old_mean']['mean']*100:.2f}%")
    for key, vd in out.get("verdicts", {}).items():
        print(f"  {key}: {vd.get('verdict', vd.get('delta'))} "
              f"| pary {vd.get('pairs_pp', vd.get('pairs_pp_end_minus_mid'))}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("I1_transplant_smoke.json" if args.smoke
             else "I1_transplant.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
