"""
run_J2b_cifar_sparse.py -- J2b: sen spike-and-slab na Split-CIFAR-10
(DROGA_J_PLAN.md, sekcja J2b -- dopisana PRZED runem, 10.07.2026).

Motywacja: J3 dal spojny kierunkowy zysk rzadkosci snu (Fashion 10/10
par dodatnich, MNIST +3.05pp), a J2 potwierdzil przenoszenie dzwigni
snu na obrazy naturalne (k4->k16: +1.21/+1.42pp). Pytanie: czy rzadkosc
snu dziala tez na cechach losowego backbone'u CIFAR?

Warianty (wejscie znormalizowane jak J2; bez kondycjonowania -- J1/J2
wykluczyly cond; epochs_proj=15, l2sp=0):
  sparse_k8  (~12 KB/klase)
  sparse_k16 (~24 KB/klase)
Baza porownawcza: mars_k16_raw z results/J2_cifar_normalized.json
(33.03 +/- 1.16), TE SAME seedy -- pary legalne bez re-runu
(determinizm sciezki potwierdzony sanity J1/J3: 0.00pp; kolejnosc
konstrukcji identyczna z run_J2: CifarBackbone() przed init klasy).

Kryteria werdyktu (Z GORY, class-IL):
  SYGNAL+ : sparse_k16 vs mars_k16_raw (pary per-seed): sr. d > prog
            szumu (std+std) ORAZ min per-seed > 0;
            SYGNAL- symetrycznie; inaczej SZUM.
  Obserwacje: czy srednia przekracza prog naprawczy z J2 (33.05);
  rownopamieciowo sparse_k8 (~12 KB) vs diag k16 (~16 KB);
  dystans do joint 70.24.

Wymaga: results/J2_cifar_normalized.json (baza par), data/glove.6B.50d.txt.

Tryb szybki:  python src/run_J2b_cifar_sparse.py --smoke
Pelny:        python src/run_J2b_cifar_sparse.py
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import make_task_data, eval_protocols, cl_metrics
from cifar_cl import CifarBackbone
from mars_cl_j import MarsCLSemanticF3J, load_cifar10_norm
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
J2_REF = os.path.join(RESULTS_DIR, "J2_cifar_normalized.json")
LR = 0.001

VARIANTS = {
    "sparse_k8":  dict(dream_model="sparse", stats_k=8),
    "sparse_k16": dict(dream_model="sparse", stats_k=16),
}
COMMON = dict(epochs_proj=15, l2sp=0.0, bn_calib=False, feat_signorm=False)
MEM_KB = {"sparse_k8": 12.0, "sparse_k16": 24.1}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_mars(cfg, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemanticF3J(wv, backbone_module=CifarBackbone(),
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
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    j2 = None
    if os.path.exists(J2_REF):
        with open(J2_REF, encoding="utf-8") as f:
            j2 = json.load(f)
    elif not args.smoke:
        sys.exit(f"BLAD: brak {J2_REF} (baza par per-seed).")

    print("=" * 72)
    print(f"J2b -- sen sparse na CIFAR  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    kw = {"glove_path": args.glove} if args.glove else {}
    wv = load_word_vectors("CIFAR-10", device=device, **kw)
    Xtr, ytr, Xte, yte = load_cifar10_norm(device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "J2b_cifar_sparse", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "variants": {k: dict(v) for k, v in VARIANTS.items()},
           "common": COMMON, "memory_kb_per_class": MEM_KB, "systems": {}}

    for name, cfg in VARIANTS.items():
        per_seed = []
        for seed in range(n_seeds):
            R_c, R_t = run_mars(cfg, wv, task_data, seed, epochs, device)
            m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
            per_seed.append({"R_class_il": R_c, "class_il": m_c,
                             "task_il": m_t})
            print(f"[CIFAR-10n] {name:10s} seed {seed}: "
                  f"class-IL ACC={m_c['ACC']*100:.2f}% "
                  f"F={m_c['forgetting']*100:.1f}pp")
        agg = {"class_il_ACC": stats([p["class_il"]["ACC"]
                                      for p in per_seed]),
               "class_il_forgetting": stats(
                   [p["class_il"]["forgetting"] for p in per_seed])}
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykt ----------
    verdict = None
    if not args.smoke and j2:
        base = [p["class_il"]["ACC"] for p in
                j2["systems"]["mars_k16_raw"]["per_seed"]][:n_seeds]
        sp16 = [p["class_il"]["ACC"] for p
                in out["systems"]["sparse_k16"]["per_seed"]]
        sp8 = [p["class_il"]["ACC"] for p
               in out["systems"]["sparse_k8"]["per_seed"]]
        d16 = stats([(a - b) * 100 for a, b in zip(sp16, base)])
        d8 = stats([(a - b) * 100 for a, b in zip(sp8, base)])
        noise = (stats([b * 100 for b in base])["std"]
                 + stats([a * 100 for a in sp16])["std"])
        if d16["mean"] > noise and d16["min"] > 0:
            v_str = "SYGNAL+ (rzadkosc snu dziala na CIFAR)"
        elif d16["mean"] < -noise:
            v_str = "SYGNAL- (rzadkosc snu szkodzi na CIFAR)"
        else:
            v_str = "SZUM"
        # obserwacja: prog naprawczy z J2 (stary combo + std)
        repair_thr = None
        if "verdict" in j2 and j2["verdict"] and \
                "repair_question" in j2["verdict"]:
            repair_thr = j2["verdict"]["repair_question"]["threshold"]
        verdict = {"delta_sparse16_vs_j2_k16raw_pp": d16,
                   "delta_sparse8_vs_j2_k16raw_pp": d8,
                   "noise_pp": round(noise, 4), "verdict": v_str,
                   "j2_base_mean": stats([b for b in base])["mean"],
                   "repair_threshold_ref": repair_thr}

    # ---------- raport ----------
    print(f"\n--- Split-CIFAR-10 znormalizowany, sen sparse "
          f"(n={n_seeds}) -- class-IL ---")
    if j2:
        b = j2["systems"]["mars_k16_raw"]["agg"]["class_il_ACC"]
        jt = j2["systems"]["joint"]["agg"]["class_il_ACC"]["mean"]
        print(f"  [J2] diag k16_raw : {b['mean']*100:.2f}"
              f"+/-{b['std']*100:.2f}% (min {b['min']*100:.2f}%) | "
              f"joint {jt*100:.2f}%")
    for name in VARIANTS:
        a = out["systems"][name]["agg"]
        print(f"  {name:10s} ({MEM_KB[name]:.0f} KB): "
              f"ACC {a['class_il_ACC']['mean']*100:.2f}"
              f"+/-{a['class_il_ACC']['std']*100:.2f}% "
              f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
              f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
    if verdict:
        print(f"  WERDYKT (sparse_k16 vs J2 k16_raw, "
              f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
        print(f"    d(sparse16): "
              f"{verdict['delta_sparse16_vs_j2_k16raw_pp']['mean']:+.2f}pp "
              f"(min {verdict['delta_sparse16_vs_j2_k16raw_pp']['min']:+.2f})"
              f" | d(sparse8): "
              f"{verdict['delta_sparse8_vs_j2_k16raw_pp']['mean']:+.2f}pp"
              + (f" | prog naprawczy J2: "
                 f"{verdict['repair_threshold_ref']*100:.2f}%"
                 if verdict.get("repair_threshold_ref") else ""))

    out["verdict"] = verdict
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("J2b_cifar_sparse_smoke.json" if args.smoke
             else "J2b_cifar_sparse.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
