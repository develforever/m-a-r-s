"""
run_K2_owm_sparse.py -- K2: OWM x sen sparse, tam gdzie boli
(DROGA_K_PLAN.md, sekcja K2).

Motywacja: H1 (OWM przy snie diag sprzed serii J): Fashion SZUM +
eliminacja (resztkowa luka != dryf), MNIST SYGNAL+ (+5.0pp przy a10).
OWM nigdy nie biegal na CIFAR (forgetting 32.7pp -- najwiekszy
w projekcie) ani ze snem sparse. Sen strzeze decyzji, OWM geometrii --
zlozenie nietestowane. Kotwice 50d (izolacja dzwigni OWM od K1).

Warianty (stats_k=16 sparse; epochs_proj=15, l2sp=0; use_dream=True,
owm_samples=2000 -- dokladnie H1):
  mnist_owm_a10   : MNIST,   alpha=10  (GLOWNY -- H1 wskazal a10)
  mnist_owm_a1    : MNIST,   alpha=1   (obserwacja)
  cifar_owm_a10   : CIFAR-n, alpha=10  (GLOWNY)
  cifar_owm_a1    : CIFAR-n, alpha=1   (obserwacja)
  fashion_owm_a1  : Fashion, alpha=1   (kontrola eliminacji H1;
                    oczekiwany SZUM, SYGNAL- tez informacja)

Bazy par (TE SAME seedy, bez re-runu; konstrukcja OWMSparse nie
konsumuje RNG poza sciezka F3J -- wagi startowe identyczne z baza):
  MNIST/Fashion: results/J3_sparse_dreams.json  sparse_k16
  CIFAR:         results/J2b_cifar_sparse.json  sparse_k16

Kryteria werdyktu (Z GORY, class-IL):
  Glowne (osobno MNIST i CIFAR): a10 vs baza sparse_k16 --
  SYGNAL+ (sr > prog std+std ORAZ min > 0) / SYGNAL- (sr < -prog) /
  SYGNAL-parowy+/- (wszystkie pary jednego znaku ORAZ |sr| > 2*std(delt),
  tylko gdy klasyczny = SZUM) / SZUM.
  a1 = obserwacja bez rangi werdyktu (przeciw grzebaniu w alfach).
  Fashion = kontrola. Plastycznosc: srednie R[t][t] (jak H1).

Wymaga: data/glove.6B.50d.txt, results/J3_sparse_dreams.json,
        results/J2b_cifar_sparse.json.

Tryb szybki:  python src/run_K2_owm_sparse.py --smoke
Pelny:        python src/run_K2_owm_sparse.py
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
from cifar_cl import CifarBackbone
from mars_cl_j import load_cifar10_norm
from mars_cl_k import MarsCLSemanticOWMSparse
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
J3_REF = os.path.join(RESULTS_DIR, "J3_sparse_dreams.json")
J2B_REF = os.path.join(RESULTS_DIR, "J2b_cifar_sparse.json")
LR = 0.001
COMMON = dict(stats_k=16, epochs_proj=15, l2sp=0.0,
              use_dream=True, owm_samples=2000)

VARIANTS = {  # nazwa -> (dataset, owm_alpha, ranga)
    "mnist_owm_a10":  ("MNIST", 10.0, "GLOWNY"),
    "mnist_owm_a1":   ("MNIST", 1.0, "obserwacja"),
    "cifar_owm_a10":  ("CIFAR-10", 10.0, "GLOWNY"),
    "cifar_owm_a1":   ("CIFAR-10", 1.0, "obserwacja"),
    "fashion_owm_a1": ("Fashion-MNIST", 1.0, "kontrola"),
}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def verdict_paired(deltas_pp, noise_pp):
    """Werdykt wg DROGA_K_PLAN: SYGNAL+/- (std+std), potem parowy."""
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


def run_sequence(ds_name, alpha, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    if ds_name == "CIFAR-10":   # konstrukcja jak run_J2b
        m = MarsCLSemanticOWMSparse(wv, backbone_module=CifarBackbone(),
                                    owm_alpha=alpha, **COMMON)
    else:                       # konstrukcja jak run_J3
        m = MarsCLSemanticOWMSparse(wv, owm_alpha=alpha, **COMMON)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    R_c, R_t = [], []
    seen = []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row_c, row_t = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row_c)
        R_t.append(row_t)
    return R_c, R_t


def load_ref(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def base_accs(ds_name, j3, j2b, n):
    if ds_name == "CIFAR-10":
        ps = j2b["systems"]["sparse_k16"]["per_seed"]
    else:
        ps = j3["datasets"][ds_name]["variants"]["sparse_k16"]["per_seed"]
    return [p["class_il"]["ACC"] for p in ps][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    j3, j2b = load_ref(J3_REF), load_ref(J2B_REF)
    if not args.smoke and (j3 is None or j2b is None):
        sys.exit("BLAD: brak referencji J3/J2b (bazy par per-seed).")

    print("=" * 72)
    print(f"K2 -- OWM x sen sparse ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    # cache danych i slow per dataset
    data_cache, wv_cache = {}, {}

    def get_ds(ds_name):
        if ds_name not in data_cache:
            if ds_name == "CIFAR-10":
                Xtr, ytr, Xte, yte = load_cifar10_norm(device)
            else:
                Xtr, ytr, Xte, yte = load_dataset(ds_name, device)
            data_cache[ds_name] = make_task_data(Xtr, ytr, Xte, yte)
            wv_cache[ds_name] = load_word_vectors(ds_name, device=device)
        return data_cache[ds_name], wv_cache[ds_name]

    t0 = time.perf_counter()
    out = {"experiment": "K2_owm_sparse", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "common": COMMON,
           "variants": {k: {"dataset": v[0], "owm_alpha": v[1],
                            "ranga": v[2]} for k, v in VARIANTS.items()},
           "systems": {}, "verdicts": {}}

    for name, (ds_name, alpha, rank) in VARIANTS.items():
        task_data, wv = get_ds(ds_name)
        per_seed = []
        for seed in range(n_seeds):
            R_c, R_t = run_sequence(ds_name, alpha, wv, task_data, seed,
                                    epochs, device)
            m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
            plast = sum(R_c[t][t] for t in range(len(R_c))) / len(R_c)
            per_seed.append({"R_class_il": R_c, "class_il": m_c,
                             "task_il": m_t,
                             "plasticity_Rtt": round(plast, 4)})
            print(f"[{ds_name}] {name:15s} seed {seed}: "
                  f"class-IL ACC={m_c['ACC']*100:.2f}% "
                  f"F={m_c['forgetting']*100:.1f}pp "
                  f"R[t][t]={plast*100:.1f}%")
        agg = {"class_il_ACC": stats([p["class_il"]["ACC"]
                                      for p in per_seed]),
               "class_il_forgetting": stats(
                   [p["class_il"]["forgetting"] for p in per_seed]),
               "plasticity_Rtt": stats([p["plasticity_Rtt"]
                                        for p in per_seed])}
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt / obserwacja ----------
        if not args.smoke and j3 and j2b:
            base = base_accs(ds_name, j3, j2b, n_seeds)
            new = [p["class_il"]["ACC"] for p in per_seed]
            deltas = [(a - b) * 100 for a, b in zip(new, base)]
            noise = (stats([b * 100 for b in base])["std"]
                     + stats([a * 100 for a in new])["std"])
            v, d = verdict_paired(deltas, noise)
            label = v if rank == "GLOWNY" else f"[{rank}] {v}"
            out["verdicts"][name] = {
                "ranga": rank, "base": "sparse_k16 (J3/J2b)",
                "delta_pp": d, "noise_pp": round(noise, 4),
                "pairs": [round(x, 2) for x in deltas], "verdict": label}

    # ---------- raport ----------
    print(f"\n--- K2 (n={n_seeds}) -- class-IL ---")
    for name in VARIANTS:
        a = out["systems"][name]["agg"]
        print(f"  {name:15s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
              f"+/-{a['class_il_ACC']['std']*100:.2f}% "
              f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
              f"F {a['class_il_forgetting']['mean']*100:.1f}pp | "
              f"R[t][t] {a['plasticity_Rtt']['mean']*100:.1f}%")
        if name in out["verdicts"]:
            vd = out["verdicts"][name]
            print(f"    WERDYKT vs {vd['base']} "
                  f"(prog {vd['noise_pp']:.2f}pp): {vd['verdict']} | "
                  f"d={vd['delta_pp']['mean']:+.2f}pp "
                  f"(min {vd['delta_pp']['min']:+.2f}) | "
                  f"pary {vd['pairs']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("K2_owm_sparse_smoke.json" if args.smoke
             else "K2_owm_sparse.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
