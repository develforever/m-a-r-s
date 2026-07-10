"""
run_J3_sparse_dreams.py -- J3: sen spike-and-slab -- atak na wiernosc snu
(DROGA_J_PLAN.md, sekcja J3).

Diagnoza H1/H1b: resztkowa luka do sufitu g1_all (77.57 -> 80.45 Fashion)
to WIERNOSC SNU. Cechy po ReLU maja dokladne zera z duzym
prawdopodobienstwem; sen diagonalny z clamp_min(0) generuje tam male
dodatnie wartosci (gestosc poza rozmaitoscia danych). Spike-and-slab
(mars_cl_j.FeatureStatsKSparse) przechowuje per wymiar P(cecha>0) +
momenty warunkowe i sni PRAWDZIWE zera. To kontynuacja drabiny wiernosci
z H1b (1 Gaussian 70.8 -> k4 75.8 -> k16 77.6) innym mechanizmem niz
zwiekszanie k (H1b: pelna kowariancja byla SLEPA ulica -- lokalnosc >
struktura; rzadkosc to inna os lokalnosci).

Warianty (baza wspolna: epochs_proj=15, l2sp=0, jak H1b):
  diag_k16   : baza (= H1b k16, ~16 KB/klase)
  sparse_k4  : ~6 KB/klase
  sparse_k8  : ~12 KB/klase (MNIEJ pamieci niz baza)
  sparse_k16 : ~24 KB/klase

Flaga --conditioning {none,cond} stosuje sie do WSZYSTKICH wariantow
naraz (porownania sparowane w obrebie rezimu). Wg planu: najpierw none;
run cond tylko jesli J1 dalo SYGNAL+.

Kryteria werdyktu (Z GORY, class-IL Fashion glowne; MNIST obserwacja):
  SYGNAL+ : sparse_k16 vs diag_k16 (pary per-seed): sr. d > prog szumu
            ORAZ min per-seed > 0; SYGNAL- symetrycznie; inaczej SZUM.
  Obserwacja rownopamieciowa: sparse_k8 vs diag_k16.
  Kontekst: replay-200 (F0), sufit g1_all (G1).

Wymaga: results/F0_cl_baselines.json, data/glove.6B.50d.txt.
Opcjonalnie: results/G1_semantic.json, results/H1b_dream_fidelity.json.

Tryb szybki:  python src/run_J3_sparse_dreams.py --smoke
Pelny:        python src/run_J3_sparse_dreams.py [--conditioning cond]
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
from mars_cl_j import MarsCLSemanticF3J
from mars_cl_semantic import load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REFS = {"f0": "F0_cl_baselines.json", "g1": "G1_semantic.json",
        "h1b": "H1b_dream_fidelity.json"}
LR = 0.001

VARIANTS = {
    "diag_k16":   dict(dream_model="diag",   stats_k=16),
    "sparse_k4":  dict(dream_model="sparse", stats_k=4),
    "sparse_k8":  dict(dream_model="sparse", stats_k=8),
    "sparse_k16": dict(dream_model="sparse", stats_k=16),
}
COMMON = dict(epochs_proj=15, l2sp=0.0)
# pamiec: diag = k*(2D+1), sparse = k*(3D+1) liczb (D=128)
MEM_KB = {"diag_k16": 16.1, "sparse_k4": 6.0, "sparse_k8": 12.0,
          "sparse_k16": 24.1}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(cfg, cond, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemanticF3J(wv, bn_calib=cond, feat_signorm=cond,
                          **cfg, **COMMON)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--glove", default=None)
    ap.add_argument("--conditioning", choices=["none", "cond"],
                    default="none")
    ap.add_argument("--datasets", nargs="+",
                    default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    cond = args.conditioning == "cond"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    refs = {}
    for k, fname in REFS.items():
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                refs[k] = json.load(f)
        elif not args.smoke and k == "f0":
            sys.exit(f"BLAD: brak {p}.")

    print("=" * 72)
    print(f"J3 -- sen spike-and-slab  ({'SMOKE' if args.smoke else 'FULL'}, "
          f"conditioning={args.conditioning})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "J3_sparse_dreams", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "conditioning": args.conditioning,
           "variants": {k: dict(v) for k, v in VARIANTS.items()},
           "memory_kb_per_class": MEM_KB,
           "common": COMMON, "datasets": {}}

    for ds_name in args.datasets:
        kw = {"glove_path": args.glove} if args.glove else {}
        wv = load_word_vectors(ds_name, device=device, **kw)
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)
        res = {"variants": {}}

        for vname, cfg in VARIANTS.items():
            per_seed = []
            for seed in range(n_seeds):
                R_c, R_t = run_sequence(cfg, cond, wv, task_data, seed,
                                        epochs, device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t})
                print(f"[{ds_name}] {vname:10s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp")
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            res["variants"][vname] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt ----------
        verdict = None
        if not args.smoke:
            base = [p["class_il"]["ACC"] for p
                    in res["variants"]["diag_k16"]["per_seed"]]
            sp16 = [p["class_il"]["ACC"] for p
                    in res["variants"]["sparse_k16"]["per_seed"]]
            sp8 = [p["class_il"]["ACC"] for p
                   in res["variants"]["sparse_k8"]["per_seed"]]
            d16 = stats([(a - b) * 100 for a, b in zip(sp16, base)])
            d8 = stats([(a - b) * 100 for a, b in zip(sp8, base)])
            noise = (stats([b * 100 for b in base])["std"]
                     + stats([a * 100 for a in sp16])["std"])
            if d16["mean"] > noise and d16["min"] > 0:
                v_str = "SYGNAL+ (rzadkosc snu podnosi wiernosc)"
            elif d16["mean"] < -noise:
                v_str = "SYGNAL- (rzadkosc snu szkodzi)"
            else:
                v_str = "SZUM"
            verdict = {"delta_sparse16_vs_diag16_pp": d16,
                       "delta_sparse8_vs_diag16_pp": d8,
                       "noise_pp": round(noise, 4), "verdict": v_str}

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if "f0" in refs:
            f0m = refs["f0"]["datasets"][ds_name]["methods"]
            print(f"  [F0] replay : "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%")
        if "g1" in refs:
            g1a = refs["g1"]["datasets"][ds_name]["variants"]["g1_all"]["agg"]
            print(f"  [G1] sufit  : {g1a['class_il_ACC']['mean']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            print(f"  {vname:10s} ({MEM_KB[vname]:.0f} KB): "
                  f"ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
        if verdict:
            print(f"  WERDYKT (sparse_k16 vs diag_k16, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            print(f"    d(sparse16): "
                  f"{verdict['delta_sparse16_vs_diag16_pp']['mean']:+.2f}pp "
                  f"(min {verdict['delta_sparse16_vs_diag16_pp']['min']:+.2f})"
                  f" | rownopamieciowo d(sparse8): "
                  f"{verdict['delta_sparse8_vs_diag16_pp']['mean']:+.2f}pp")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if args.conditioning == "none" else f"_{args.conditioning}"
    fname = (f"J3_sparse_dreams{suffix}_smoke.json" if args.smoke
             else f"J3_sparse_dreams{suffix}.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
