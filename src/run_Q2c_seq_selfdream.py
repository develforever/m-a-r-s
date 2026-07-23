"""
run_Q2c_seq_selfdream.py -- Q2c: kontrola uczciwosci -- sekwencyjny
z symetrycznym budzetem snu (DROGA_Q2C_PLAN.md).

NOWY plik (branch droga-q) -- istniejacy kod NIETKNIETY.
MarsSelfDream.learn_task = wierna kopia MarsCollectiveM.learn_task
z jedna roznica: material nowych klas dosniewany do N_SELF/klase
(500 realnych + 2000 snow z wlasnych statystyk 1. generacji).

Kryteria (Z GORY, DROGA_Q2C_PLAN.md):
  1) seq_selfdream2500 vs m1_seq_300 (pary): czy budzet dziala tez
     dla uczenia.
  2) ROZSTRZYGNIECIE: q2b (Q2_early_repair.json) vs seq_selfdream2500
     (pary): SZUM = rownowaznosc przy symetrycznych budzetach;
     SYGNAL+ = przewaga kolektywu realna; SYGNAL- = seq odzyskuje.
  3) Obs.: luka do sufitu 47.41; forgetting.

Wymaga: cache cech CIFAR-100, data/glove.6B.300d.txt,
results/M1_long_horizon.json, results/Q2_early_repair.json.

Tryb szybki:  python src/run_Q2c_seq_selfdream.py --smoke
Pelny:        python src/run_Q2c_seq_selfdream.py  (~15 min)
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import make_task_data, cl_metrics, eval_protocols
from mars_cl_l import ReducedBackbone
from mars_cl_m import (MarsCollectiveM, TASKS20, _train_pods_negatives_n,
                       extract_or_load_cifar100_feats)
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
M1_REF = os.path.join(RESULTS_DIR, "M1_long_horizon.json")
Q2_REF = os.path.join(RESULTS_DIR, "Q2_early_repair.json")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
N_SELF = 2500  # docelowa licznosc materialu nowej klasy (symetria z q2b)


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


class MarsSelfDream(MarsCollectiveM):
    """learn_task = wierna kopia MarsCollectiveM.learn_task; jedyna
    roznica: material nowych klas dosniewany do n_self/klase."""

    def __init__(self, word_vecs, n_self=N_SELF, **kw):
        super().__init__(word_vecs, **kw)
        self.n_self = n_self

    def learn_task(self, td, epochs, lr, device):
        classes = td["classes"]
        X, y = td["Xtr"], td["ytr"]
        with torch.no_grad():
            feats_real = self.feats_batched(X)
        old = list(self.seen_classes)
        self.stats.update(feats_real, y, classes)
        # augmentacja: 500 realnych + sen do n_self (1. generacja)
        aug_f, aug_y = [feats_real], [y]
        for c in classes:
            n_extra = self.n_self - int((y == c).sum())
            if n_extra > 0:
                aug_f.append(self.stats.sample(c, n_extra, device))
                aug_y.append(torch.full((n_extra,), c, dtype=torch.long,
                                        device=device))
        feats = torch.cat(aug_f)
        y_aug = torch.cat(aug_y)
        self._fit_proj_feats(feats, y_aug, old, old + list(classes),
                             epochs, lr, device)
        neg_f, neg_y = self.stats.replay_batch(old, self.replay_per_class,
                                               device)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        _train_pods_negatives_n(self, classes, feats, y_aug, routed,
                                neg_f, neg_y, epochs, lr, device)


def run_seq(wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsSelfDream(wv, backbone_module=ReducedBackbone(), **CFG)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    R_c = []
    seen = []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row, _ = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row)
    return R_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    m1 = q2 = None
    for ref, name in ((M1_REF, "M1"), (Q2_REF, "Q2")):
        if not os.path.exists(ref) and not args.smoke:
            sys.exit(f"BLAD: brak {ref} (najpierw FULL {name}).")
    if os.path.exists(M1_REF):
        m1 = json.load(open(M1_REF, encoding="utf-8"))
    if os.path.exists(Q2_REF):
        q2 = json.load(open(Q2_REF, encoding="utf-8"))

    print("=" * 72)
    print(f"Q2c -- seq z symetrycznym budzetem snu "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_self={N_SELF}")
    print("=" * 72)

    Ftr, ytr, Fte, yte = extract_or_load_cifar100_feats(device)
    task_data = make_task_data(Ftr, ytr, Fte, yte, tasks=TASKS20)
    wv = load_word_vectors("CIFAR-100", glove_path=GLOVE300,
                           device=device)

    t0 = time.perf_counter()
    out = {"experiment": "Q2c_seq_selfdream", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "n_self": N_SELF, "systems": {}, "verdicts": {}}

    per_seed = []
    for seed in range(n_seeds):
        R_c = run_seq(wv, task_data, seed, epochs, device)
        m_c = cl_metrics(R_c)
        per_seed.append({"R_class_il": R_c, "class_il": m_c})
        print(f"[seq_selfdream2500] seed {seed}: ACC={m_c['ACC']*100:.2f}% "
              f"F={m_c['forgetting']*100:.1f}pp")
    agg = {"class_il_ACC": stats([p["class_il"]["ACC"] for p in per_seed]),
           "class_il_forgetting": stats([p["class_il"]["forgetting"]
                                         for p in per_seed])}
    out["systems"]["seq_selfdream2500"] = {"per_seed": per_seed,
                                           "agg": agg}

    if not args.smoke and m1 and q2:
        sd = [p["class_il"]["ACC"] for p in per_seed]
        # 1) vs m1_seq_300
        base = [p["class_il"]["ACC"] for p in
                m1["systems"]["m1_seq_300"]["per_seed"]][:n_seeds]
        d = [(a - b) * 100 for a, b in zip(sd, base)]
        noise = (stats([x * 100 for x in base])["std"]
                 + stats([x * 100 for x in sd])["std"])
        v, ds = verdict_paired(d, noise)
        out["verdicts"]["selfdream_vs_seq"] = {
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(noise, 4), "verdict": v}
        # 2) ROZSTRZYGNIECIE: q2b vs seq_selfdream
        q2b = [p["ACC"] for p in
               q2["systems"]["q2b_dream2500"]["per_seed"]][:n_seeds]
        d = [(a - b) * 100 for a, b in zip(q2b, sd)]
        noise = (stats([x * 100 for x in sd])["std"]
                 + stats([x * 100 for x in q2b])["std"])
        v, ds = verdict_paired(d, noise)
        label = {"SZUM": "ROWNOWAZNOSC przy symetrycznych budzetach",
                 "SYGNAL+": "PRZEWAGA KOLEKTYWU REALNA",
                 "SYGNAL-parowy+": "PRZEWAGA KOLEKTYWU (parowa)",
                 "SYGNAL-": "SEQ ODZYSKUJE PRZEWAGE",
                 "SYGNAL-parowy-": "SEQ ODZYSKUJE (parowo)"}[v]
        allm = m1["systems"]["m1_all_300"]["agg"]["class_il_ACC"]["mean"]
        out["verdicts"]["ROZSTRZYGNIECIE_q2b_vs_selfdream"] = {
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(noise, 4), "verdict": v, "label": label,
            "gap_selfdream_do_all300_pp": round(
                (allm - stats(sd)["mean"]) * 100, 2)}

    print(f"\n--- Q2c (n={n_seeds}) ---")
    for key, vd in out.get("verdicts", {}).items():
        print(f"  {key}: {json.dumps(vd, ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("Q2c_seq_selfdream_smoke.json" if args.smoke
             else "Q2c_seq_selfdream.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
