"""
run_F1_mars_cl.py -- F1: MARS-CL vs baseline'y F0 (multi-seed).

Hipoteza (DROGA_F_PLAN.md):
  Modularnosc z nietykalna reprezentacja (prawo D5) daje w class-IL
  retencje bez bufora danych -- "architektura zamiast pamieci".

Warianty (pre-rejestrowane):
  F1a: backbone_source=task0,  proto=mean     (czysta teza)
  F1a-l: backbone_source=task0, proto=learned (czy uczone protos > NCM)
  F1c: backbone_source=task01, proto=mean     (diagnostyka transferu cech;
       uzywa danych zadania 1 -- NIE jest uczciwym CL, tylko pomiarem
       kosztu ubostwa cech z task0)
  F1d: backbone_source=random, proto=mean     (reservoir -- backbone nigdy
       nie trenowany; rozbraja ryzyko transferu z definicji)

Kryteria werdyktu (Z GORY, Split-Fashion, class-IL, vs F0 per-seed):
  GLOWNY: najlepszy UCZCIWY wariant (F1a/F1a-l/F1d -- bez F1c!) vs replay-200:
     ACC >= ACC(replay) - prog szumu  => SYGNAL+ ("architektura zamiast
     pamieci": retencja replay bez zadnego bufora danych)
     ACC < replay o wiecej niz prog   => granica podejscia (tez wynik)
  SANITY: kazdy wariant vs finetune: ACC wyzsza o >10pp.
  KOSZT: MAC(po 5 zadaniach)/MAC(po 1 zadaniu) <= 1.05.
  Dodatkowo (lekcja E4): raportowac min per-seed, nie tylko srednia.

Wymaga: results/F0_cl_baselines.json (pelny run F0 -- te same seedy).

Tryb szybki:  python src/run_F1_mars_cl.py --smoke
Pelny:        python src/run_F1_mars_cl.py
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
from mars_cl import MarsCLSystem
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
F0_PATH = os.path.join(RESULTS_DIR, "F0_cl_baselines.json")

VARIANTS = {
    "F1a":   dict(backbone_source="task0",  proto_mode="mean"),
    "F1a-l": dict(backbone_source="task0",  proto_mode="learned"),
    "F1c":   dict(backbone_source="task01", proto_mode="mean"),
    "F1d":   dict(backbone_source="random", proto_mode="mean"),
}
FAIR = ["F1a", "F1a-l", "F1d"]   # F1c wylaczone z werdyktu (podglada dane)
LR = 0.001


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_sequence(cfg, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSystem(**cfg)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    R_c, R_t = [], []
    seen = []
    mac_after = []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row_c, row_t = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row_c)
        R_t.append(row_t)
        mac_after.append(m.mac_per_sample()["total"])
    return R_c, R_t, mac_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="+",
                    default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f0 = None
    if os.path.exists(F0_PATH):
        with open(F0_PATH, encoding="utf-8") as f:
            f0 = json.load(f)
    elif not args.smoke:
        sys.exit("BLAD: brak results/F0_cl_baselines.json (pelny F0 najpierw).")

    print("=" * 72)
    print(f"F1 -- MARS-CL vs baseline'y F0  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs} | "
          f"warianty={list(VARIANTS)}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "F1_mars_cl", "device": device, "n_seeds": n_seeds,
           "epochs_per_task": epochs, "variants": VARIANTS, "datasets": {}}

    for ds_name in args.datasets:
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)
        res = {"variants": {}}

        for vname, cfg in VARIANTS.items():
            per_seed = []
            for seed in range(n_seeds):
                R_c, R_t, mac_after = run_sequence(cfg, task_data, seed,
                                                   epochs, device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "R_task_il": R_t,
                                 "class_il": m_c, "task_il": m_t,
                                 "mac_after": mac_after})
                print(f"[{ds_name}] {vname:6s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp | "
                      f"task-IL ACC={m_t['ACC']*100:.2f}% | "
                      f"MAC T1->T5: {mac_after[0]:,}->{mac_after[-1]:,}")
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            agg["mac_growth"] = round(
                per_seed[0]["mac_after"][-1] / per_seed[0]["mac_after"][0], 4)
            res["variants"][vname] = {"per_seed": per_seed, "agg": agg}

        # ---------- werdykt vs F0 ----------
        verdict = None
        if not args.smoke and f0:
            f0m = f0["datasets"][ds_name]["methods"]
            rep = [p["class_il"]["ACC"] for p
                   in f0m["replay"]["per_seed"]][:n_seeds]
            fin = [p["class_il"]["ACC"] for p
                   in f0m["finetune"]["per_seed"]][:n_seeds]
            best = max(FAIR, key=lambda v:
                       res["variants"][v]["agg"]["class_il_ACC"]["mean"])
            mars = [p["class_il"]["ACC"] for p
                    in res["variants"][best]["per_seed"]]
            d_rep = [(a - b) * 100 for a, b in zip(mars, rep)]
            d_fin = [(a - b) * 100 for a, b in zip(mars, fin)]
            d_rep_s, d_fin_s = stats(d_rep), stats(d_fin)
            noise = (stats([r * 100 for r in rep])["std"]
                     + stats([m_ * 100 for m_ in mars])["std"])
            mac_ok = res["variants"][best]["agg"]["mac_growth"] <= 1.05
            if d_rep_s["mean"] >= -noise:
                v_str = "SYGNAL+ (architektura zamiast pamieci)"
            else:
                v_str = "PONIZEJ REPLAY (granica podejscia -- tez wynik)"
            verdict = {"best_fair_variant": best,
                       "delta_vs_replay_pp": d_rep_s,
                       "delta_vs_finetune_pp": d_fin_s,
                       "noise_pp": round(noise, 4),
                       "sanity_vs_finetune_gt10pp": d_fin_s["min"] > 10,
                       "mac_growth_le_1.05": mac_ok,
                       "verdict": v_str}

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL ---")
        if f0:
            f0m = f0["datasets"][ds_name]["methods"]
            for b in ("finetune", "replay", "joint"):
                a = f0m[b]["agg"]
                key = "class_il_ACC"
                print(f"  [F0] {b:9s}: ACC {a[key]['mean']*100:.2f}"
                      f"+/-{a[key]['std']*100:.2f}%")
        for vname in VARIANTS:
            a = res["variants"][vname]["agg"]
            tag = "" if vname in FAIR else "  [diagnostyka, poza werdyktem]"
            print(f"  {vname:6s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                  f"+/-{a['class_il_ACC']['std']*100:.2f}% "
                  f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
                  f"F {a['class_il_forgetting']['mean']*100:.1f}pp | "
                  f"MAC wzrost x{a['mac_growth']}{tag}")
        if verdict:
            print(f"  WERDYKT ({verdict['best_fair_variant']} vs replay, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
            print(f"    d vs replay: {verdict['delta_vs_replay_pp']['mean']:+.2f}pp | "
                  f"d vs finetune: {verdict['delta_vs_finetune_pp']['mean']:+.2f}pp | "
                  f"MAC<=1.05: {verdict['mac_growth_le_1.05']}")
        print()
        res["verdict"] = verdict
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "F1_mars_cl_smoke.json" if args.smoke else "F1_mars_cl.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
