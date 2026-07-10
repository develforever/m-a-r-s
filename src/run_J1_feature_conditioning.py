"""
run_J1_feature_conditioning.py -- J1: kondycjonowanie cech losowego
zamrozonego backbone'u (DROGA_J_PLAN.md, sekcja J1).

Audyt 2026-07-10: przy backbone_source="random" BatchNorm nigdy nie byl
kalibrowany (running stats (0,1) = identycznosc) i cechy nie mialy
wyrownanych skal per wymiar. NCM (Euklides) i k-means snu sa czule na
skale. Oba kondycjonowania uzywaja WYLACZNIE nieetykietowanych obrazow
zadania 0, raz, przed nauka (uczciwosc CL jak ae0/F2).

Warianty (baza wspolna = H1b k16: diag, epochs_proj=15, l2sp=0):
  k16_raw     : bez kondycjonowania (reprodukcja H1b k16 -- sanity)
  k16_bncal   : kalibracja BN na task0
  k16_signorm : cechy / per-wymiarowe std z task0
  k16_cond    : oba

Kryteria werdyktu (Z GORY, class-IL Fashion glowne; MNIST obserwacja):
  SYGNAL+ : najlepszy kondycjonowany vs k16_raw (pary per-seed):
            sr. d > prog szumu (std+std) ORAZ min per-seed > 0
  SYGNAL- : symetrycznie; inaczej SZUM.
  Sanity: k16_raw ma odtworzyc H1b k16 (77.57 +/- 1.02) w granicach
          progu szumu -- inaczej STOP (niedeterminizm).

Wymaga: results/F0_cl_baselines.json (kontekst replay),
        results/H1b_dream_fidelity.json (sanity), data/glove.6B.50d.txt.
Opcjonalnie: results/G1_semantic.json (sufit g1_all).

Tryb szybki:  python src/run_J1_feature_conditioning.py --smoke
Pelny:        python src/run_J1_feature_conditioning.py
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
REFS = {"f0": "F0_cl_baselines.json", "h1b": "H1b_dream_fidelity.json",
        "g1": "G1_semantic.json"}
LR = 0.001

VARIANTS = {
    "k16_raw":     dict(bn_calib=False, feat_signorm=False),
    "k16_bncal":   dict(bn_calib=True,  feat_signorm=False),
    "k16_signorm": dict(bn_calib=False, feat_signorm=True),
    "k16_cond":    dict(bn_calib=True,  feat_signorm=True),
}
COMMON = dict(dream_model="diag", stats_k=16, epochs_proj=15, l2sp=0.0)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(cfg, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemanticF3J(wv, **cfg, **COMMON)
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
    ap.add_argument("--datasets", nargs="+",
                    default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    refs = {}
    for k, fname in REFS.items():
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                refs[k] = json.load(f)
        elif not args.smoke and k in ("f0", "h1b"):
            sys.exit(f"BLAD: brak {p}.")

    print("=" * 72)
    print(f"J1 -- kondycjonowanie cech  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "J1_feature_conditioning", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "variants": {k: dict(v) for k, v in VARIANTS.items()},
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
                R_c, R_t = run_sequence(cfg, wv, task_data, seed, epochs,
                                        device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t})
                print(f"[{ds_name}] {vname:11s} seed {seed}: "
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
                    in res["variants"]["k16_raw"]["per_seed"]]
            conded = {v: [p["class_il"]["ACC"] for p
                          in res["variants"][v]["per_seed"]]
                      for v in VARIANTS if v != "k16_raw"}
            best = max(conded, key=lambda v: sum(conded[v]))
            d = stats([(a - b) * 100 for a, b in zip(conded[best], base)])
            noise = (stats([b * 100 for b in base])["std"]
                     + stats([a * 100 for a in conded[best]])["std"])
            if d["mean"] > noise and d["min"] > 0:
                v_str = "SYGNAL+ (kondycjonowanie podnosi wynik)"
            elif d["mean"] < -noise:
                v_str = "SYGNAL- (kondycjonowanie szkodzi)"
            else:
                v_str = "SZUM"
            verdict = {"best_conditioned": best, "delta_vs_raw_pp": d,
                       "noise_pp": round(noise, 4), "verdict": v_str}
            # sanity: reprodukcja H1b k16
            if "h1b" in refs and ds_name in refs["h1b"]["datasets"]:
                h_ref = [p["class_il"]["ACC"] for p in
                         refs["h1b"]["datasets"][ds_name]["variants"]["k16"]
                         ["per_seed"]][:n_seeds]
                d_s = stats([(a - b) * 100 for a, b in zip(base, h_ref)])
                n_s = (stats([b * 100 for b in base])["std"]
                       + stats([h * 100 for h in h_ref])["std"])
                verdict["sanity_raw_vs_h1b_k16_pp"] = d_s
                verdict["sanity_ok"] = bool(abs(d_s["mean"]) <= n_s)
                if not verdict["sanity_ok"]:
                    verdict["verdict"] += " [UWAGA: sanity FAIL -- k16_raw" \
                                          " nie odtwarza H1b; STOP]"

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
            print(f"  {vname:11s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
        if verdict:
            print(f"  WERDYKT ({verdict['best_conditioned']} vs k16_raw, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            print(f"    d: {verdict['delta_vs_raw_pp']['mean']:+.2f}pp "
                  f"(min {verdict['delta_vs_raw_pp']['min']:+.2f})"
                  + (f" | sanity vs H1b: "
                     f"{verdict['sanity_raw_vs_h1b_k16_pp']['mean']:+.2f}pp"
                     if "sanity_raw_vs_h1b_k16_pp" in verdict else ""))
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("J1_feature_conditioning_smoke.json" if args.smoke
             else "J1_feature_conditioning.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
