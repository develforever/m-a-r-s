"""
run_I3_collective.py -- I3: N=5 agentow x 2 klasy -- headline rewolucji
(DROGA_I_PLAN.md, sekcja I3).

Setup: agent_i (i=0..4, wspolny seed = ten sam backbone) uczy sie
WYLACZNIE taska i Fashion jako swojego jedynego zadania. Agent 0 =
kolektor: po nauce taska 0 adoptuje kolejno payloady od agentow 1-4
(4 adopcje x 2 klasy). Finalnie: 10-klasowy class-IL agenta 0.

Kryteria (Z GORY):
  Glowne: kolektyw (agent 0) vs pojedynczy agent sekwencyjny
  (K1 fashion_sp16_300, 79.23 +/- 0.73, TE SAME seedy, pary per-seed):
  SYGNAL+/-, SYGNAL-parowy+/-, SZUM. SYGNAL+ = "5 agentow, zero
  wymienionych obrazow, wynik >= scentralizowanego uczenia
  sekwencyjnego".
  Obserwacje: luka do sufitu g1_all_300 (81.16, J4); kontekst
  replay-200 (F0, 76.97); krzywa ACC po kazdej adopcji.
  Ryzyko pre-rejestrowane: 4 adopcje = projekcja douczana 4x na samych
  snach; kumulacja dryfu -> SYGNAL- realny i tez jest wynikiem.

Wymaga: data/glove.6B.300d.txt, results/K1_sparse300.json,
opcjonalnie results/J4_glove300.json, results/F0_cl_baselines.json.

Tryb szybki:  python src/run_I3_collective.py --smoke
Pelny:        python src/run_I3_collective.py
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
J4_REF = os.path.join(RESULTS_DIR, "J4_glove300.json")
F0_REF = os.path.join(RESULTS_DIR, "F0_cl_baselines.json")
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


def load_ref(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_collective(wv, task_data, seed, epochs, n_dream, device):
    # kolektor: uczy sie tylko taska 0
    col = build_agent(wv, seed, device)
    col.init_representation([task_data[0]], epochs=epochs, lr=LR,
                            device=device)
    col.learn_task(task_data[0], epochs=epochs, lr=LR, device=device)
    curve = []
    for i in range(1, len(task_data)):
        # nadawca i: wlasny agent, ten sam seed (protokol), uczy task i
        Ai = build_agent(wv, seed, device)
        Ai.init_representation([task_data[i]], epochs=epochs, lr=LR,
                               device=device)
        Ai.learn_task(task_data[i], epochs=epochs, lr=LR, device=device)
        td = task_data[i]
        payloads = {c: Ai.export_class_stats(c, int((td["ytr"] == c).sum()))
                    for c in td["classes"]}
        col.adopt_classes(td["classes"], payloads, epochs=epochs, lr=LR,
                          device=device, n_dream=n_dream)
        row, _ = eval_protocols(col.forward, task_data, i,
                                list(col.seen_classes))
        curve.append(sum(row) / len(row))
    final_row = row
    return {"acc_curve_after_adoptions": [round(c, 4) for c in curve],
            "final_row_class_il": final_row,
            "acc_task0_final": final_row[0],
            "ACC": sum(final_row) / len(final_row)}


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
    k1, j4, f0 = load_ref(K1_REF), load_ref(J4_REF), load_ref(F0_REF)
    if not args.smoke and k1 is None:
        sys.exit(f"BLAD: brak {K1_REF} (baza par).")

    print("=" * 72)
    print(f"I3 -- kolektyw 5 agentow x 2 klasy "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream}")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "I3_collective", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_dream": n_dream, "common": COMMON, "systems": {},
           "verdicts": {}}

    per_seed = []
    for seed in range(n_seeds):
        r = run_collective(wv, task_data, seed, epochs, n_dream, device)
        per_seed.append(r)
        print(f"[Fashion] kolektyw seed {seed}: ACC={r['ACC']*100:.2f}% | "
              f"krzywa po adopcjach: "
              f"{[round(c*100, 1) for c in r['acc_curve_after_adoptions']]}")
    agg = {"ACC": stats([p["ACC"] for p in per_seed]),
           "acc_task0_final": stats([p["acc_task0_final"]
                                     for p in per_seed])}
    out["systems"]["collective"] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykt ----------
    if not args.smoke and k1:
        loc = [p["class_il"]["ACC"] for p in
               k1["systems"]["fashion_sp16_300"]["per_seed"]][:n_seeds]
        col = [p["ACC"] for p in per_seed]
        d = [(a - b) * 100 for a, b in zip(col, loc)]
        noise = (stats([x * 100 for x in loc])["std"]
                 + stats([x * 100 for x in col])["std"])
        v, ds = verdict_paired(d, noise)
        vd = {"base": "K1 fashion_sp16_300 (agent sekwencyjny)",
              "pairs_pp": [round(x, 2) for x in d], "delta": ds,
              "noise_pp": round(noise, 4), "verdict": v}
        if j4:
            ceil = j4["datasets"]["Fashion-MNIST"]["variants"]["all_300"][
                "agg"]["class_il_ACC"]["mean"]
            vd["gap_to_g1_all_300_pp"] = round(
                (ceil - stats(col)["mean"]) * 100, 2)
        out["verdicts"]["collective_vs_seq"] = vd

    # ---------- raport ----------
    print(f"\n--- I3 (n={n_seeds}) -- Fashion, class-IL ---")
    if k1:
        a = k1["systems"]["fashion_sp16_300"]["agg"]["class_il_ACC"]
        print(f"  [K1] agent seq : {a['mean']*100:.2f}+/-{a['std']*100:.2f}%")
    if f0 and "Fashion-MNIST" in f0.get("datasets", {}):
        rep = f0["datasets"]["Fashion-MNIST"]["methods"]["replay"][
            "agg"]["class_il_ACC"]["mean"]
        print(f"  [F0] replay-200: {rep*100:.2f}%")
    print(f"  kolektyw (N=5) : {agg['ACC']['mean']*100:.2f}"
          f"+/-{agg['ACC']['std']*100:.2f}% "
          f"(min {agg['ACC']['min']*100:.2f}%)")
    if "collective_vs_seq" in out["verdicts"]:
        vd = out["verdicts"]["collective_vs_seq"]
        extra = (f" | luka do g1_all_300: {vd['gap_to_g1_all_300_pp']}pp"
                 if "gap_to_g1_all_300_pp" in vd else "")
        print(f"  WERDYKT vs agent seq (prog {vd['noise_pp']:.2f}pp): "
              f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
              f"(min {vd['delta']['min']:+.2f}) | pary {vd['pairs_pp']}"
              + extra)

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("I3_collective_smoke.json" if args.smoke
             else "I3_collective.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
