"""
run_I2b_fusion_lowdata.py -- I2b: fuzja statystyk w rezimie malych probek
(DROGA_I_PLAN.md, sekcja I2b -- dopisana PRZED runem, 17.07.2026).

I2 dal SZUM z powodem: payload nasyca sie przy 3000 obrazow/klase
(half ~ full). Tu payloady licza sie z n obrazow/klase na agenta
(proby rozlaczne, generator=seed), n in {100, 500}. Pytanie: czy przy
nienasyconych payloadach fuzja dwoch czesciowych widokow bije
pojedynczy widok?

Warianty: half_100 / fusion_100 / half_500 / fusion_500.
Kontekst z I2 (bez re-runu): half3000 94.38, full6000 94.17.

Kryteria (Z GORY, per n, acc task4, pary per-seed):
  fusion_cat(n) vs half(n): SYGNAL+/-, SYGNAL-parowy+/-, SZUM.
  Obserwacja: krzywa nasycenia half(100/500/3000).
  Ryzyko: k-means k=16 na ~100 wektorach = szumne centroidy.

Wymaga: data/glove.6B.300d.txt. Konfiguracja K1 (sparse_k16 x 300d),
n_dream=6000.

Tryb szybki:  python src/run_I2b_fusion_lowdata.py --smoke
Pelny:        python src/run_I2b_fusion_lowdata.py
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
from mars_cl_j import FeatureStatsKSparse
from mars_collective import MarsCollective, fuse_payloads_cat
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
LR = 0.001
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
              bn_calib=False, feat_signorm=False)
NS = (100, 500)


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


def payload_of(s, c, n):
    return {"p": s.p[c].detach().cpu(), "mean": s.mean[c].detach().cpu(),
            "var": s.var[c].detach().cpu(), "w": s.w[c].detach().cpu(),
            "n": int(n)}


def build_agent(wv, seed, device):
    torch.manual_seed(seed)
    m = MarsCollective(wv, **COMMON)
    m.to(device)
    return m


def make_payload_sets(wv, td4, seed, device):
    """Payloady half/fusion dla kazdego n (proby rozlaczne per klasa)."""
    O = build_agent(wv, seed, device)
    O.init_representation([td4], epochs=0, lr=LR, device=device)
    with torch.no_grad():
        feats4 = O.feats_batched(td4["Xtr"])
    y4 = td4["ytr"]
    g = torch.Generator().manual_seed(seed)
    sets = {}
    for n in NS:
        iA, iB = [], []
        for c in td4["classes"]:
            idx_c = (y4 == c).nonzero(as_tuple=True)[0]
            perm = idx_c[torch.randperm(len(idx_c), generator=g)]
            iA.append(perm[:n])
            iB.append(perm[n:2 * n])
        iA, iB = torch.cat(iA), torch.cat(iB)
        sA = FeatureStatsKSparse(k=COMMON["stats_k"])
        sB = FeatureStatsKSparse(k=COMMON["stats_k"])
        sA.update(feats4[iA], y4[iA], td4["classes"])
        sB.update(feats4[iB], y4[iB], td4["classes"])
        half, fus = {}, {}
        for c in td4["classes"]:
            pA, pB = payload_of(sA, c, n), payload_of(sB, c, n)
            half[c] = pA
            fus[c] = fuse_payloads_cat(pA, pB)
        sets[f"half_{n}"] = half
        sets[f"fusion_{n}"] = fus
    return sets


def run_recipient(wv, task_data, payloads, seed, epochs, n_dream, device):
    C = build_agent(wv, seed, device)
    C.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    for t in range(4):
        C.learn_task(task_data[t], epochs=epochs, lr=LR, device=device)
    C.adopt_classes(task_data[4]["classes"], payloads, epochs=epochs,
                    lr=LR, device=device, n_dream=n_dream)
    seen = [c for td in task_data for c in td["classes"]]
    row, _ = eval_protocols(C.forward, task_data, len(task_data) - 1, seen)
    return {"final_row_class_il": row, "acc_task4": row[4],
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

    variants = [f"{kind}_{n}" for n in NS for kind in ("half", "fusion")]
    print("=" * 72)
    print(f"I2b -- fuzja low-data ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream} | warianty={variants}")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "I2b_fusion_lowdata", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_dream": n_dream, "common": COMMON, "ns": list(NS),
           "systems": {}, "verdicts": {}}
    per = {v: [] for v in variants}

    for seed in range(n_seeds):
        sets = make_payload_sets(wv, task_data[4], seed, device)
        for vname in variants:
            r = run_recipient(wv, task_data, sets[vname], seed, epochs,
                              n_dream, device)
            per[vname].append(r)
            print(f"[Fashion] {vname:10s} seed {seed}: "
                  f"przeszczep(task4)={r['acc_task4']*100:.2f}% "
                  f"ACC={r['ACC']*100:.2f}%")

    for vname in variants:
        agg = {k: stats([p[k] for p in per[vname]])
               for k in ("ACC", "acc_task4")}
        out["systems"][vname] = {"per_seed": per[vname], "agg": agg}

    # ---------- werdykty ----------
    if not args.smoke:
        def t4(v):
            return [p["acc_task4"] for p in per[v]]
        for n in NS:
            d = [(a - b) * 100 for a, b
                 in zip(t4(f"fusion_{n}"), t4(f"half_{n}"))]
            noise = (stats([x * 100 for x in t4(f"half_{n}")])["std"]
                     + stats([x * 100 for x in t4(f"fusion_{n}")])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"][f"fusion_vs_half_n{n}"] = {
                "pairs_pp": [round(x, 2) for x in d], "delta": ds,
                "noise_pp": round(noise, 4), "verdict": v}

    # ---------- raport ----------
    print(f"\n--- I2b (n={n_seeds}) -- acc klas przeszczepionych ---")
    print("  kontekst I2: half3000 94.38 | full6000 94.17")
    for vname in variants:
        a = out["systems"][vname]["agg"]
        print(f"  {vname:10s}: przeszczep {a['acc_task4']['mean']*100:.2f}"
              f"+/-{a['acc_task4']['std']*100:.2f}% | "
              f"ACC {a['ACC']['mean']*100:.2f}%")
    for key, vd in out.get("verdicts", {}).items():
        print(f"  {key}: {vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
              f"(prog {vd['noise_pp']:.2f}) | pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("I2b_fusion_lowdata_smoke.json" if args.smoke
             else "I2b_fusion_lowdata.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
