"""
run_N1_unlearning.py -- N1: selektywne zapominanie z gwarancja
(DROGA_N_PLAN.md). Fashion, konfiguracja K1 (sparse_k16 x 300d).

Poziom 1 (funkcjonalny): macierz 10 usuniec (light/scrub na kopii
pelnego modelu); acc klasy usunietej = 0 z konstrukcji (odnotowanie);
werdykt: sr. delta acc pozostalych 9 klas (pary per-seed).

Poziom 2 (informacyjny): klasa c*=4 (Z GORY). Trzy sciezki per seed:
  relearn_light : pelny trening -> unlearn_light(4) -> relearn(n=100)
  relearn_scrub : pelny trening -> unlearn_scrub(4) -> relearn(n=100)
  relearn_never : trening BEZ taska 2 (klasy 4,5) -> relearn(n=100)
acc(4) we wszystkich trzech z maska WSPOLNYCH 9 klas (bez 5).
Werdykty: light vs never (SYGNAL+ = resztkowa informacja w projekcji);
scrub vs never (SZUM = wymazanie do poziomu never = gwarancja
empiryczna). Referencja gorna: acc(4) pelnego systemu.

Wymaga: data/glove.6B.300d.txt.

Tryb szybki:  python src/run_N1_unlearning.py --smoke
Pelny:        python src/run_N1_unlearning.py  (~15-20 min)
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
C_STAR = 4                                   # klasa poziomu 2 (Z GORY)
ALLOWED9 = [0, 1, 2, 3, 4, 6, 7, 8, 9]       # wspolna maska (bez 5)
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


def train_full(wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCollective(wv, **CFG)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    for td in task_data:
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
    return m


def pick_relearn_sample(task_data, seed, device):
    td = task_data[2]                       # task 2 = klasy (4,5)
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
    print(f"N1 -- selektywne zapominanie "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"c*={C_STAR} | n_relearn={N_RELEARN}")
    print("=" * 72)

    wv = load_word_vectors("Fashion-MNIST", glove_path=GLOVE300,
                           device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)
    td_never = [task_data[t] for t in (0, 1, 3, 4)]   # bez taska 2

    t0 = time.perf_counter()
    out = {"experiment": "N1_unlearning", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "cfg": CFG,
           "c_star": C_STAR, "n_relearn": N_RELEARN,
           "per_seed": [], "verdicts": {}}

    for seed in range(n_seeds):
        full = train_full(wv, task_data, seed, epochs, device)
        before = class_accs(full, task_data, allowed=full.seen_classes)

        # ---------- poziom 1: macierz 10 usuniec ----------
        d_light, d_scrub, removed_zero = [], [], True
        for c in range(10):
            for scrub, acc_list in ((False, d_light), (True, d_scrub)):
                mc = copy.deepcopy(full)
                unlearn_class(mc, c, scrub=scrub, epochs=epochs, lr=LR,
                              device=device, n_dream_scrub=n_scrub)
                a = class_accs(mc, task_data, allowed=mc.seen_classes)
                if a[c] != 0.0:
                    removed_zero = False
                rest = [a[r] - before[r] for r in range(10) if r != c]
                acc_list.append(sum(rest) / len(rest))
        mean_dl = sum(d_light) / len(d_light)
        mean_ds = sum(d_scrub) / len(d_scrub)

        # ---------- poziom 2: relearn vs never (maska 9 klas) ----------
        X100, y100 = pick_relearn_sample(task_data, seed, device)
        ref_full = class_accs(full, task_data, ALLOWED9)[C_STAR]

        ml = copy.deepcopy(full)
        unlearn_class(ml, C_STAR, scrub=False, device=device)
        relearn_small(ml, C_STAR, X100, y100, epochs=epochs, lr=LR,
                      device=device)
        acc_light = class_accs(ml, task_data, ALLOWED9)[C_STAR]

        ms = copy.deepcopy(full)
        unlearn_class(ms, C_STAR, scrub=True, epochs=epochs, lr=LR,
                      device=device, n_dream_scrub=n_scrub)
        relearn_small(ms, C_STAR, X100, y100, epochs=epochs, lr=LR,
                      device=device)
        acc_scrub = class_accs(ms, task_data, ALLOWED9)[C_STAR]

        torch.manual_seed(seed)
        nv = MarsCollective(wv, **CFG)
        nv.to(device)
        nv.init_representation(td_never, epochs=epochs, lr=LR,
                               device=device)
        for td in td_never:
            nv.learn_task(td, epochs=epochs, lr=LR, device=device)
        relearn_small(nv, C_STAR, X100, y100, epochs=epochs, lr=LR,
                      device=device)
        acc_never = class_accs(nv, task_data, ALLOWED9)[C_STAR]

        out["per_seed"].append({
            "rest_delta_light": round(mean_dl, 4),
            "rest_delta_scrub": round(mean_ds, 4),
            "removed_acc_zero": removed_zero,
            "acc_cstar_full": round(ref_full, 4),
            "acc_relearn_light": round(acc_light, 4),
            "acc_relearn_scrub": round(acc_scrub, 4),
            "acc_relearn_never": round(acc_never, 4)})
        print(f"[Fashion] seed {seed}: rest d(light)={mean_dl*100:+.2f}pp"
              f" d(scrub)={mean_ds*100:+.2f}pp | usuniete=0: {removed_zero}"
              f" | c*=4: full={ref_full*100:.1f} light={acc_light*100:.1f}"
              f" scrub={acc_scrub*100:.1f} never={acc_never*100:.1f}")

    # ---------- werdykty ----------
    ps = out["per_seed"]
    if not args.smoke:
        for key, field in (("poziom1_light", "rest_delta_light"),
                           ("poziom1_scrub", "rest_delta_scrub")):
            d = [p[field] * 100 for p in ps]
            noise = 2 * stats(d)["std"] if len(d) > 1 else 1.0
            v, ds = verdict_paired(d, max(noise, 0.5))
            label = ("NIETKNIETE (sukces poziomu 1)" if v == "SZUM"
                     else v)
            out["verdicts"][key] = {"pairs_pp": [round(x, 2) for x in d],
                                    "delta": ds, "verdict": label}
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

    # ---------- raport ----------
    print(f"\n--- N1 (n={n_seeds}) -- Fashion, unlearning ---")
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
    fname = ("N1_unlearning_smoke.json" if args.smoke
             else "N1_unlearning.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
