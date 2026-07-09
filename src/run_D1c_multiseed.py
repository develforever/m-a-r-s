"""
run_D1c_multiseed.py -- Droga D, D1 / DROGA 1: multi-seed walidacja.

Cel: rozstrzygnac, czy roznice v1<->v2 z Etapu 2/3 (rzedu 0.1-0.8pp przy
JEDNYM seedzie) to SYGNAL czy SZUM. Bez tego zaden claim "v2 vs v1" nie jest
rzetelny -- w zadna strone.

Dla kazdego z N_SEEDS seedow trenujemy i ewaluujemy trzy modele:
  - v1 Separate           (baseline, osobne encodery)
  - v2 D1a phased         (shared backbone, trening 2-fazowy)
  - v2 D1b end-to-end      (shared backbone, laczony loss)
Wszystkie ZROWNANE PARAMETRYCZNIE (~408.9k).

Metryki per seed: routing_acc, system_acc, oracle_acc.
Raport: mean +/- std (Bessel n-1), min, max -- konwencja z run_A9_multiseed.py.

INTERPRETACJA (ustalona z gory):
  Jesli |mean(v2) - mean(v1)| < std obu -> roznica to SZUM, v2 ~= v1.
  Jesli przekracza std i jest spojna miedzy seedami -> SYGNAL.
  (Heurystyka pierwszego rzutu; przy granicznych roznicach policzyc parowany
   t-test po seedach jako test formalny.)

Uruchom:
    .venv\\Scripts\\python.exe src\\run_D1c_multiseed.py
"""

import json, os, sys, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import (MarsV2System, train_phased, train_end_to_end, evaluate,
                     N_IN, N_CLASSES)
from run_D1_mars_v2_baseline import (SeparateV1, train_separate_v1,
                                     eval_separate_v1, load_dataset)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_SEEDS = 5
EPOCHS = 30
BB_H, EMB, POD_H = 384, 32, 24   # zrownane parametrycznie z v1


def stats(vals):
    """mean / std (Bessel n-1) / min / max -- jak w run_A9_multiseed.py."""
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0
    std = math.sqrt(var)
    return {"mean": round(mean, 4), "std": round(std, 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_one_seed(ds_name, Xtr, ytr, Xte, yte, seed, device):
    """Trenuje i ewaluuje v1, v2a, v2b dla jednego seeda. Zwraca dict metryk."""
    # --- v1 Separate ---
    torch.manual_seed(seed)
    v1 = SeparateV1(N_IN, N_CLASSES, router_enc_hidden=256, router_emb=64,
                    pod_hidden=24).to(device)
    train_separate_v1(v1, Xtr, ytr, epochs=EPOCHS, device=device)
    r1, s1, o1 = eval_separate_v1(v1, Xte, yte)

    # --- v2 D1a phased ---
    torch.manual_seed(seed)
    v2a = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES).to(device)
    train_phased(v2a, Xtr, ytr, epochs=EPOCHS, device=device)
    ra, sa, oa = evaluate(v2a, Xte, yte)

    # --- v2 D1b end-to-end ---
    torch.manual_seed(seed)
    v2b = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES).to(device)
    train_end_to_end(v2b, Xtr, ytr, epochs=EPOCHS, alpha=1.0, device=device)
    rb, sb, ob = evaluate(v2b, Xte, yte)

    return {
        "v1":  {"routing": r1, "system": s1, "oracle": o1},
        "v2a": {"routing": ra, "system": sa, "oracle": oa},
        "v2b": {"routing": rb, "system": sb, "oracle": ob},
    }


def run_dataset(ds_name, device):
    print(f"\n{'='*72}\nDataset: {ds_name}  ({N_SEEDS} seeds)\n{'='*72}")
    Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

    per_seed = []
    print(f"\n{'seed':>4} | {'v1 sys':>8} {'v2a sys':>8} {'v2b sys':>8} "
          f"| {'v1 rout':>8} {'v2a rout':>9} {'v2b rout':>9}")
    print("-" * 72)
    for seed in range(N_SEEDS):
        m = run_one_seed(ds_name, Xtr, ytr, Xte, yte, seed, device)
        per_seed.append(m)
        print(f"{seed:>4} | {m['v1']['system']*100:>7.2f}% "
              f"{m['v2a']['system']*100:>7.2f}% {m['v2b']['system']*100:>7.2f}% "
              f"| {m['v1']['routing']*100:>7.2f}% {m['v2a']['routing']*100:>8.2f}% "
              f"{m['v2b']['routing']*100:>8.2f}%")

    # agregacja
    agg = {}
    for model in ["v1", "v2a", "v2b"]:
        agg[model] = {
            metric: stats([per_seed[s][model][metric] for s in range(N_SEEDS)])
            for metric in ["routing", "system", "oracle"]
        }

    # --- raport mean +/- std ---
    print(f"\n  {'model':<6} {'system':>18} {'routing':>18} {'oracle':>18}")
    print("  " + "-" * 62)
    for model, label in [("v1", "v1 Separate"), ("v2a", "v2 phased"),
                         ("v2b", "v2 end2end")]:
        s, r, o = agg[model]["system"], agg[model]["routing"], agg[model]["oracle"]
        print(f"  {label:<12} {s['mean']*100:>6.2f}+/-{s['std']*100:.2f}%   "
              f"{r['mean']*100:>6.2f}+/-{r['std']*100:.2f}%   "
              f"{o['mean']*100:>6.2f}+/-{o['std']*100:.2f}%")

    # --- werdykt: sygnal czy szum (system acc, v2a vs v1) ---
    d_a = agg["v2a"]["system"]["mean"] - agg["v1"]["system"]["mean"]
    d_b = agg["v2b"]["system"]["mean"] - agg["v1"]["system"]["mean"]
    pooled = max(agg["v1"]["system"]["std"], agg["v2a"]["system"]["std"],
                 agg["v2b"]["system"]["std"])
    print(f"\n  v2a - v1 (system): {d_a*100:+.2f}pp | v2b - v1: {d_b*100:+.2f}pp")
    print(f"  max std miedzy modelami: {pooled*100:.2f}pp")
    if abs(d_a) < pooled and abs(d_b) < pooled:
        verdict = "SZUM -- v2 ~= v1 (roznica mniejsza niz std seedow)"
    else:
        verdict = "SYGNAL -- roznica przekracza std, warto badac dalej"
    print(f"  [WERDYKT] {verdict}")

    return {"dataset": ds_name, "n_seeds": N_SEEDS,
            "per_seed": per_seed, "stats": agg,
            "delta_v2a_v1_pp": round(d_a * 100, 2),
            "delta_v2b_v1_pp": round(d_b * 100, 2),
            "max_std_pp": round(pooled * 100, 2),
            "verdict": verdict}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA D -- D1 / DROGA 1: multi-seed walidacja (v1 vs v2a vs v2b)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print(f"Seeds: {N_SEEDS}  |  Epochs: {EPOCHS}  |  params ~408.9k (zrownane)")
    print("=" * 72)

    results = {}
    for ds_name in ["MNIST", "Fashion-MNIST"]:
        results[ds_name] = run_dataset(ds_name, device)

    print("\n" + "=" * 72)
    print("PODSUMOWANIE")
    print("=" * 72)
    for ds_name in ["MNIST", "Fashion-MNIST"]:
        r = results[ds_name]
        print(f"  {ds_name:<15}: {r['verdict']}")
        print(f"  {'':<15}  v2a-v1={r['delta_v2a_v1_pp']:+.2f}pp "
              f"v2b-v1={r['delta_v2b_v1_pp']:+.2f}pp (std {r['max_std_pp']:.2f}pp)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "D1c_multiseed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")
    print("\nDroga 1 zakonczona. Werdykt sygnal/szum -> decyduje o ksztalcie Drogi 2.")


if __name__ == "__main__":
    main()
