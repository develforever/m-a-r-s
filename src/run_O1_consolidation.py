"""
run_O1_consolidation.py -- O1: konsolidacja snem (DROGA_O_PLAN.md).

Po pelnej sekwencji jeden "gleboki sen": nauka projekcji na snach
wszystkich widzianych klas (2000/klase, epochs_proj=15). Tryby:
o1_reinit (GLOWNY, od zera) i o1_finetune (douczenie). Pody nietkniete.

Benchmarki i bazy par (TE SAME seedy; sanity: baza odtworzona <=0.01pp):
  Fashion: K1 fashion_sp16_300 (79.23 +/- 0.73; sufit 81.16)
  CIFAR-n: L1 l1_seq (74.69 +/- 0.69; sufit 77.23)

Kryteria (Z GORY, pary per-seed vs baza, per dataset):
  o1_reinit i o1_finetune: SYGNAL+/-, SYGNAL-parowy+/-, SZUM.
  Obserwacje: reinit vs finetune; dystans do sufitow.

Wymaga: data/glove.6B.50d/300d.txt, results/K1_sparse300.json,
results/L1_pretrained.json, cache cech CIFAR (mars_cl_l).

Tryb szybki:  python src/run_O1_consolidation.py --smoke
Pelny:        python src/run_O1_consolidation.py  (~5 min)
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
from cl_common import make_task_data, eval_protocols
from mars_cl_j import MarsCLSemanticF3J
from mars_cl_l import ReducedBackbone, extract_or_load_cifar_feats
from mars_cl_o import consolidate
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
K1_REF = os.path.join(RESULTS_DIR, "K1_sparse300.json")
L1_REF = os.path.join(RESULTS_DIR, "L1_pretrained.json")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
N_DREAM = 2000
CEILS = {"Fashion-MNIST": 81.16, "CIFAR-10": 77.23}


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


def final_acc(m, task_data):
    seen = [c for td in task_data for c in td["classes"]]
    row, _ = eval_protocols(m.forward, task_data, len(task_data) - 1,
                            seen)
    return sum(row) / len(row), row


def train_seq(ds_name, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    if ds_name == "CIFAR-10":
        m = MarsCLSemanticF3J(wv, backbone_module=ReducedBackbone(),
                              **CFG)
    else:
        m = MarsCLSemanticF3J(wv, **CFG)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    for td in task_data:
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
    return m


def load_ref(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else N_DREAM
    device = "cuda" if torch.cuda.is_available() else "cpu"

    k1, l1 = load_ref(K1_REF), load_ref(L1_REF)
    if not args.smoke and (k1 is None or l1 is None):
        sys.exit("BLAD: brak referencji K1/L1 (bazy par).")

    print("=" * 72)
    print(f"O1 -- konsolidacja snem ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "O1_consolidation", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_dream": n_dream, "cfg": CFG, "datasets": {}}

    for ds_name in ("Fashion-MNIST", "CIFAR-10"):
        if ds_name == "CIFAR-10":
            Ftr, ytr, Fte, yte = extract_or_load_cifar_feats(device)
            task_data = make_task_data(Ftr, ytr, Fte, yte)
            wv = load_word_vectors("CIFAR-10", device=device)
            base_ps = (l1["systems"]["l1_seq"]["per_seed"]
                       if l1 else None)
        else:
            Xtr, ytr, Xte, yte = load_dataset(ds_name, device)
            task_data = make_task_data(Xtr, ytr, Xte, yte)
            wv = load_word_vectors(ds_name, glove_path=GLOVE300,
                                   device=device)
            base_ps = (k1["systems"]["fashion_sp16_300"]["per_seed"]
                       if k1 else None)

        per_seed = []
        sanity_max = 0.0
        for seed in range(n_seeds):
            m = train_seq(ds_name, wv, task_data, seed, epochs, device)
            base_acc, base_row = final_acc(m, task_data)
            if base_ps is not None and not args.smoke:
                ref_row = base_ps[seed]["R_class_il"][-1]
                sanity_max = max(sanity_max,
                                 max(abs(a - b) for a, b
                                     in zip(base_row, ref_row)))
            mr = copy.deepcopy(m)
            consolidate(mr, mode="reinit", n_dream=n_dream,
                        epochs=epochs, lr=LR, device=device,
                        seed=2000 + seed)
            acc_r, _ = final_acc(mr, task_data)
            mf = copy.deepcopy(m)
            consolidate(mf, mode="finetune", n_dream=n_dream,
                        epochs=epochs, lr=LR, device=device)
            acc_f, _ = final_acc(mf, task_data)
            per_seed.append({"base_ACC": round(base_acc, 4),
                             "reinit_ACC": round(acc_r, 4),
                             "finetune_ACC": round(acc_f, 4)})
            print(f"[{ds_name}] seed {seed}: baza={base_acc*100:.2f} "
                  f"reinit={acc_r*100:.2f} finetune={acc_f*100:.2f}")

        res = {"per_seed": per_seed,
               "sanity_max_diff_pp": round(sanity_max * 100, 3),
               "verdicts": {}}
        if not args.smoke:
            base = [p["base_ACC"] for p in per_seed]
            for key, field in (("reinit", "reinit_ACC"),
                               ("finetune", "finetune_ACC")):
                new = [p[field] for p in per_seed]
                d = [(a - b) * 100 for a, b in zip(new, base)]
                noise = (stats([x * 100 for x in base])["std"]
                         + stats([x * 100 for x in new])["std"])
                v, ds_ = verdict_paired(d, noise)
                res["verdicts"][key] = {
                    "pairs_pp": [round(x, 2) for x in d], "delta": ds_,
                    "noise_pp": round(noise, 4), "verdict": v,
                    "mean_ACC": stats([x * 100 for x in new])["mean"],
                    "gap_to_ceiling_pp": round(
                        CEILS[ds_name]
                        - stats([x * 100 for x in new])["mean"], 2)}
            dd = [(p["reinit_ACC"] - p["finetune_ACC"]) * 100
                  for p in per_seed]
            res["verdicts"]["obs_reinit_vs_finetune"] = {
                "pairs_pp": [round(x, 2) for x in dd],
                "delta": stats(dd)}
        out["datasets"][ds_name] = res

        print(f"\n--- O1 {ds_name} (n={n_seeds}) ---")
        if not args.smoke:
            print(f"  sanity (baza vs ref): max "
                  f"{res['sanity_max_diff_pp']:.3f}pp")
            for key in ("reinit", "finetune"):
                vd = res["verdicts"][key]
                print(f"  {key:8s} (prog {vd['noise_pp']:.2f}pp): "
                      f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
                      f"| ACC {vd['mean_ACC']:.2f}% (luka do sufitu "
                      f"{vd['gap_to_ceiling_pp']:+.2f}pp) | "
                      f"pary {vd['pairs_pp']}")
        print()

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("O1_consolidation_smoke.json" if args.smoke
             else "O1_consolidation.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
