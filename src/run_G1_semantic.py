"""
run_G1_semantic.py -- G1: prototypy semantyczne vs class-mean (multi-seed).

Hipoteza (DROGA_G_PLAN.md):
  Slowa jako kotwice prototypow wnosza informacje SPOZA obrazow (strukture
  znaczen) i przelamuja plateau zamrozonych cech z F2 (~65% class-IL).
  Bonus: zero-shot routing -- prototypy wszystkich klas istnieja a priori.

Warianty (pre-rejestrowane; backbone losowy zamrozony (8,16) jak F1d):
  g1_t0  : projekcja obraz->slowa uczona TYLKO na zadaniu 0 (czysty CL)
  g1_seq : projekcja douczana na kazdym zadaniu (dryf mierzony forgettingiem)
  g1_all : projekcja na wszystkich klasach -- DIAGNOSTYKA, poza werdyktem

Kryteria werdyktu (Z GORY, class-IL Fashion, per-seed):
  GLOWNY: najlepszy uczciwy wariant (t0/seq) vs F1d (rand_8_16 z F1):
     delta > prog szumu => SYGNAL+ (semantyka > srednie obrazow)
  DODATKOWY: vs replay-200 z F0 (czy teza "bez bufora" wrocila do gry).
  ZERO-SHOT: routing na klasach niewidzianych przez projekcje (g1_t0,
     po zadaniu 0) > 2x losowe (10%) = sygnal groundingu.

Wymaga: results/F0_cl_baselines.json, results/F1_mars_cl.json,
        data/glove.6B.50d.txt.

Tryb szybki:  python src/run_G1_semantic.py --smoke
Pelny:        python src/run_G1_semantic.py
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
from mars_cl_semantic import MarsCLSemantic, load_word_vectors
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
F0_PATH = os.path.join(RESULTS_DIR, "F0_cl_baselines.json")
F1_PATH = os.path.join(RESULTS_DIR, "F1_mars_cl.json")

VARIANTS = ["g1_t0", "g1_seq", "g1_all"]
FAIR = ["g1_t0", "g1_seq"]
LR = 0.001


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(proj_train, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemantic(wv, proj_train=proj_train)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    zs = None
    if proj_train == "task0":
        zs = m.zero_shot_routing(task_data,
                                 trained_classes=task_data[0]["classes"],
                                 device=device)
    R_c, R_t = [], []
    seen = []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row_c, row_t = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row_c)
        R_t.append(row_t)
    return R_c, R_t, zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--glove", default=None, help="sciezka do glove.6B.50d.txt")
    ap.add_argument("--datasets", nargs="+",
                    default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    refs = {}
    for name, path in (("f0", F0_PATH), ("f1", F1_PATH)):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                refs[name] = json.load(f)
        elif not args.smoke:
            sys.exit(f"BLAD: brak {path}.")

    print("=" * 72)
    print(f"G1 -- prototypy semantyczne (slowa jako kotwice)  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "G1_semantic", "device": device, "n_seeds": n_seeds,
           "epochs_per_task": epochs, "datasets": {}}

    for ds_name in args.datasets:
        kw = {"glove_path": args.glove} if args.glove else {}
        wv = load_word_vectors(ds_name, device=device, **kw)
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)
        res = {"variants": {}}

        for vname in VARIANTS:
            proj_train = vname.split("_", 1)[1].replace("t0", "task0")
            per_seed = []
            for seed in range(n_seeds):
                R_c, R_t, zs = run_sequence(proj_train, wv, task_data,
                                            seed, epochs, device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                rec = {"R_class_il": R_c, "class_il": m_c, "task_il": m_t}
                if zs:
                    rec["zero_shot"] = zs
                per_seed.append(rec)
                zs_str = ""
                if zs:
                    unseen = [v["routing_acc"] for v in zs.values()
                              if v["kind"] == "unseen"]
                    zs_str = (f" | ZS routing unseen: "
                              f"{sum(unseen)/len(unseen)*100:.1f}%")
                print(f"[{ds_name}] {vname:7s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp{zs_str}")
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            if any("zero_shot" in p for p in per_seed):
                unseen_all = [v["routing_acc"]
                              for p in per_seed if "zero_shot" in p
                              for v in p["zero_shot"].values()
                              if v["kind"] == "unseen"]
                agg["zero_shot_unseen_routing"] = stats(unseen_all)
            res["variants"][vname] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt ----------
        verdict = None
        if not args.smoke and "f1" in refs and "f0" in refs:
            f1d = [p["class_il"]["ACC"] for p in
                   refs["f1"]["datasets"][ds_name]["variants"]["F1d"]
                   ["per_seed"]][:n_seeds]
            rep = [p["class_il"]["ACC"] for p in
                   refs["f0"]["datasets"][ds_name]["methods"]["replay"]
                   ["per_seed"]][:n_seeds]
            best = max(FAIR, key=lambda v:
                       res["variants"][v]["agg"]["class_il_ACC"]["mean"])
            mars = [p["class_il"]["ACC"] for p
                    in res["variants"][best]["per_seed"]]
            d_f1d = stats([(a - b) * 100 for a, b in zip(mars, f1d)])
            d_rep = stats([(a - b) * 100 for a, b in zip(mars, rep)])
            noise = (stats([r * 100 for r in f1d])["std"]
                     + stats([m_ * 100 for m_ in mars])["std"])
            v_str = ("SYGNAL+ (semantyka > srednie obrazow)"
                     if d_f1d["mean"] > noise and d_f1d["min"] > 0
                     else ("SZUM vs F1d" if abs(d_f1d["mean"]) <= noise
                           else "SYGNAL-"))
            verdict = {"best_fair_variant": best,
                       "delta_vs_F1d_pp": d_f1d,
                       "delta_vs_replay_pp": d_rep,
                       "noise_pp": round(noise, 4), "verdict": v_str}

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if "f0" in refs:
            f0m = refs["f0"]["datasets"][ds_name]["methods"]
            print(f"  [F0] replay: "
                  f"{f0m['replay']['agg']['class_il_ACC']['mean']*100:.2f}%")
        if "f1" in refs:
            f1a = refs["f1"]["datasets"][ds_name]["variants"]["F1d"]["agg"]
            print(f"  [F1] F1d   : {f1a['class_il_ACC']['mean']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            tag = "" if vname in FAIR else "  [diagnostyka]"
            zs_str = ""
            if "zero_shot_unseen_routing" in a:
                zs_str = (f" | ZS unseen "
                          f"{a['zero_shot_unseen_routing']['mean']*100:.1f}%")
            print(f"  {vname:7s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp"
                  f"{zs_str}{tag}")
        if verdict:
            print(f"  WERDYKT ({verdict['best_fair_variant']} vs F1d, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            print(f"    d vs F1d: {verdict['delta_vs_F1d_pp']['mean']:+.2f}pp"
                  f" | d vs replay: "
                  f"{verdict['delta_vs_replay_pp']['mean']:+.2f}pp")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "G1_semantic_smoke.json" if args.smoke else "G1_semantic.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
