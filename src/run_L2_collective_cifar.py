"""
run_L2_collective_cifar.py -- L2: kolektyw na mocnych cechach --
protokol I bez zmian, nowy backbone (DROGA_L_PLAN.md, sekcja L2).

Setup = run_I3 przeniesiony na Split-CIFAR-10n z PretrainedBackbone:
5 agentow x 2 klasy (wspolny seed: pretrained czesc identyczna
z definicji, losowa projekcja 512->128 z seeda), kolektor = agent 0,
4 adopcje po 2 klasy, payload 24.1 KB/klase, n_dream=5000 (parytet
licznosci klasy CIFAR).

Kryteria (Z GORY, class-IL):
  Glowne: kolektyw vs l1_seq (results/L1_pretrained.json, TE SAME
  seedy, pary): SZUM = rownowaznosc -> protokol przenosi sie na nowa
  reprezentacje i trudniejsze dane; SYGNAL- = limit protokolu
  (tez wynik). SYGNAL+/parowy symetrycznie.
  Obserwacje: krzywa po adopcjach; luka do l1_all.

Wymaga: data/glove.6B.50d.txt, results/L1_pretrained.json.

Tryb szybki:  python src/run_L2_collective_cifar.py --smoke
Pelny:        python src/run_L2_collective_cifar.py
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
from mars_cl_j import load_cifar10_norm
from mars_cl_l import PretrainedBackbone
from mars_collective import MarsCollective
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
L1_REF = os.path.join(RESULTS_DIR, "L1_pretrained.json")
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
    m = MarsCollective(wv, backbone_module=PretrainedBackbone(), **COMMON)
    m.to(device)
    return m


def run_collective(wv, task_data, seed, epochs, n_dream, device):
    col = build_agent(wv, seed, device)
    col.init_representation([task_data[0]], epochs=epochs, lr=LR,
                            device=device)
    col.learn_task(task_data[0], epochs=epochs, lr=LR, device=device)
    curve = []
    for i in range(1, len(task_data)):
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
    return {"acc_curve_after_adoptions": [round(c, 4) for c in curve],
            "final_row_class_il": row,
            "ACC": sum(row) / len(row)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else 5000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    l1 = None
    if os.path.exists(L1_REF):
        with open(L1_REF, encoding="utf-8") as f:
            l1 = json.load(f)
    elif not args.smoke:
        sys.exit(f"BLAD: brak {L1_REF} (baza par -- najpierw FULL L1).")

    print("=" * 72)
    print(f"L2 -- kolektyw na pretrained, CIFAR "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream}")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    Xtr, ytr, Xte, yte = load_cifar10_norm(device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "L2_collective_cifar", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_dream": n_dream, "common": COMMON,
           "encoder": "resnet18_IMAGENET1K_V1 -> random frozen 512->128",
           "systems": {}, "verdicts": {}}

    per_seed = []
    for seed in range(n_seeds):
        r = run_collective(wv, task_data, seed, epochs, n_dream, device)
        per_seed.append(r)
        print(f"[CIFAR-10n] kolektyw seed {seed}: ACC={r['ACC']*100:.2f}% "
              f"| krzywa: "
              f"{[round(c*100, 1) for c in r['acc_curve_after_adoptions']]}")
    agg = {"ACC": stats([p["ACC"] for p in per_seed])}
    out["systems"]["collective"] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykt ----------
    if not args.smoke and l1:
        base = [p["class_il"]["ACC"] for p in
                l1["systems"]["l1_seq"]["per_seed"]][:n_seeds]
        col = [p["ACC"] for p in per_seed]
        d = [(a - b) * 100 for a, b in zip(col, base)]
        noise = (stats([x * 100 for x in base])["std"]
                 + stats([x * 100 for x in col])["std"])
        v, ds = verdict_paired(d, noise)
        vd = {"base": "L1 l1_seq (agent sekwencyjny, pretrained)",
              "pairs_pp": [round(x, 2) for x in d], "delta": ds,
              "noise_pp": round(noise, 4), "verdict": v}
        allm = l1["systems"]["l1_all"]["agg"]["class_il_ACC"]["mean"]
        vd["gap_to_l1_all_pp"] = round((allm - stats(col)["mean"]) * 100, 2)
        out["verdicts"]["collective_vs_seq"] = vd

    # ---------- raport ----------
    print(f"\n--- L2 (n={n_seeds}) -- CIFAR-n, class-IL ---")
    if l1:
        a = l1["systems"]["l1_seq"]["agg"]["class_il_ACC"]
        print(f"  [L1] agent seq : {a['mean']*100:.2f}"
              f"+/-{a['std']*100:.2f}%")
    print(f"  kolektyw (N=5) : {agg['ACC']['mean']*100:.2f}"
          f"+/-{agg['ACC']['std']*100:.2f}% "
          f"(min {agg['ACC']['min']*100:.2f}%)")
    if "collective_vs_seq" in out["verdicts"]:
        vd = out["verdicts"]["collective_vs_seq"]
        print(f"  WERDYKT vs agent seq (prog {vd['noise_pp']:.2f}pp): "
              f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
              f"(min {vd['delta']['min']:+.2f}) | pary {vd['pairs_pp']} "
              f"| luka do l1_all: {vd['gap_to_l1_all_pp']}pp")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("L2_collective_cifar_smoke.json" if args.smoke
             else "L2_collective_cifar.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
