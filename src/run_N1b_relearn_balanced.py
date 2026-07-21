"""
run_N1b_relearn_balanced.py -- N1b: naprawiony instrument poziomu 2
(DROGA_N_PLAN.md, sekcja N1b -- dopisana PRZED runem, 20.07.2026).

N1 poziom 2 uniewazniony (podloga ~0% we wszystkich sciezkach --
negatywy 2304 vs 100 pozytywow). N1b: relearn ze ZBALANSOWANYM
budzetem negatywow (lacznie ~= pozytywom). Sciezki, c*=4, n=100,
maska 9 klas i kryteria BEZ ZMIAN:
  GLOWNE 2: light vs never (SYGNAL+ = resztkowa informacja projekcji)
  GLOWNE 3: scrub vs never (SZUM = empiryczna gwarancja wymazania)
  obserwacja: scrub vs light; referencja gorna: pelny system.

Wymaga: data/glove.6B.300d.txt.

Tryb szybki:  python src/run_N1b_relearn_balanced.py --smoke
Pelny:        python src/run_N1b_relearn_balanced.py  (~8-10 min)
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
from mars_cl_n import class_accs, relearn_small, unlearn_class
from mars_collective import MarsCollective
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GLOVE300 = os.path.join(DATA_DIR, "glove.6B.300d.txt")
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


def train_seq(wv, tds, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCollective(wv, **CFG)
    m.to(device)
    m.init_representation(tds, epochs=epochs, lr=LR, device=device)
    for td in tds:
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
    return m


def pick_relearn_sample(task_data, seed):
    td = task_data[2]
    idx_c = (td["ytr"] == C_STAR).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    perm = idx_c[torch.randperm(len(idx_c), generator=g)][:N_RELEARN]
    return td["Xtr"][perm], td["ytr"][perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_scrub = 256 if args.smoke else 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(GLOVE300):
        sys.exit(f"BLAD: brak {GLOVE300}")

    print("=" * 72)
    print(f"N1b -- relearn zbalansowany "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"c*={C_STAR} | n_relearn={N_RELEARN} | negatywy lacznie~100")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)
    td_never = [task_data[t] for t in (0, 1, 3, 4)]

    t0 = time.perf_counter()
    out = {"experiment": "N1b_relearn_balanced", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "c_star": C_STAR, "n_relearn": N_RELEARN,
           "per_seed": [], "verdicts": {}}

    for seed in range(n_seeds):
        full = train_seq(wv, task_data, seed, epochs, device)
        X100, y100 = pick_relearn_sample(task_data, seed)
        ref_full = class_accs(full, task_data, ALLOWED9)[C_STAR]

        ml = copy.deepcopy(full)
        unlearn_class(ml, C_STAR, scrub=False, device=device)
        relearn_small(ml, C_STAR, X100, y100, epochs=epochs, lr=LR,
                      device=device, balanced_negatives=True)
        acc_light = class_accs(ml, task_data, ALLOWED9)[C_STAR]

        ms = copy.deepcopy(full)
        unlearn_class(ms, C_STAR, scrub=True, epochs=epochs, lr=LR,
                      device=device, n_dream_scrub=n_scrub)
        relearn_small(ms, C_STAR, X100, y100, epochs=epochs, lr=LR,
                      device=device, balanced_negatives=True)
        acc_scrub = class_accs(ms, task_data, ALLOWED9)[C_STAR]

        nv = train_seq(wv, td_never, seed, epochs, device)
        relearn_small(nv, C_STAR, X100, y100, epochs=epochs, lr=LR,
                      device=device, balanced_negatives=True)
        acc_never = class_accs(nv, task_data, ALLOWED9)[C_STAR]

        out["per_seed"].append({
            "acc_cstar_full": round(ref_full, 4),
            "acc_relearn_light": round(acc_light, 4),
            "acc_relearn_scrub": round(acc_scrub, 4),
            "acc_relearn_never": round(acc_never, 4)})
        print(f"[Fashion] seed {seed}: full={ref_full*100:.1f} "
              f"light={acc_light*100:.1f} scrub={acc_scrub*100:.1f} "
              f"never={acc_never*100:.1f}")

    ps = out["per_seed"]
    if not args.smoke:
        for key, fa in (("poziom2_light_vs_never", "acc_relearn_light"),
                        ("poziom2_scrub_vs_never", "acc_relearn_scrub")):
            d = [(p[fa] - p["acc_relearn_never"]) * 100 for p in ps]
            base = [p["acc_relearn_never"] * 100 for p in ps]
            new = [p[fa] * 100 for p in ps]
            noise = stats(base)["std"] + stats(new)["std"]
            v, ds = verdict_paired(d, noise)
            out["verdicts"][key] = {"pairs_pp": [round(x, 2) for x in d],
                                    "delta": ds,
                                    "noise_pp": round(noise, 4),
                                    "verdict": v}
        d = [(p["acc_relearn_scrub"] - p["acc_relearn_light"]) * 100
             for p in ps]
        out["verdicts"]["obs_scrub_vs_light"] = {
            "pairs_pp": [round(x, 2) for x in d], "delta": stats(d)}

    print(f"\n--- N1b (n={n_seeds}) -- Fashion, relearn zbalansowany ---")
    for k in ("acc_cstar_full", "acc_relearn_light", "acc_relearn_scrub",
              "acc_relearn_never"):
        vals = [p[k] * 100 for p in ps]
        print(f"  {k:18s}: {stats(vals)['mean']:.2f}"
              f"+/-{stats(vals)['std']:.2f}%")
    for key, vd in out.get("verdicts", {}).items():
        extra = (f" (prog {vd['noise_pp']:.2f}pp)"
                 if "noise_pp" in vd else "")
        print(f"  {key}{extra}: {vd.get('verdict', '')} "
              f"| d={vd['delta']['mean']:+.2f}pp | pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("N1b_relearn_balanced_smoke.json" if args.smoke
             else "N1b_relearn_balanced.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
