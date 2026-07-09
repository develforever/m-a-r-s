"""
run_D1b_pareto_sweep.py -- Droga D, krok D1 / ETAP 3: Adaptive Compute + Pareto.

Po potwierdzeniu fundamentu (Etap 2: router v2 ~98.3% = C4), tutaj wlaczamy
trojpoziomowe wnioskowanie (Early Exit / Top-1 / Selective Top-2) i mierzymy
KRZYWA PARETO: Accuracy vs sredni MAC per probka.

Pytanie centralne (z analizy Etapu 2): v2 w trybie top-1 ma WYZSZY MAC niz v1
(323k vs 237k), bo wiekszy wspolny backbone. Czy Early Exit sciagnie dosc
latwych probek (zwlaszcza na MNIST), by sredni MAC v2 spadl PONIZEJ v1 przy
zachowaniu accuracy? To rozstrzyga sweep.

KRYTERIUM SUKCESU (ustalone Z GORY, zgodnie z planem):
  v2 wygrywa, jesli DOMINUJE v1 na krzywej Accuracy-MAC w co najmniej jednym
  rezimie -- tj. istnieje prog v2 dajacy >= accuracy v1 przy < MAC v1
  (lub > accuracy przy <= MAC). Przy zrownanych parametrach.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_D1b_pareto_sweep.py
"""

import json, os, sys
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import (MarsV2System, train_phased, train_end_to_end, evaluate,
                     adaptive_sweep, pareto_front, N_IN, N_CLASSES)
from run_D1_mars_v2_baseline import (SeparateV1, train_separate_v1,
                                     eval_separate_v1, load_dataset)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EPOCHS = 30
BB_H, EMB, POD_H = 384, 32, 24   # zrownane parametrycznie z v1 (Etap 2)


def v1_top1_point(model, Xte, yte):
    """Punkt odniesienia v1: pojedynczy (acc, MAC) w trybie top-1."""
    r, s, o = eval_separate_v1(model, Xte, yte)
    mac = model.mac_per_sample()["total_top1"]
    return {"system_acc": s, "mac": mac}


def check_v2_dominates_v1(sweep, v1_point):
    """
    Kryterium sukcesu: czy istnieje prog v2 dominujacy v1?
    Zwraca liste punktow v2, ktore daja >= acc v1 przy < MAC v1
    (lub > acc przy <= MAC). Posortowane po MAC rosnaco.
    """
    va, vm = v1_point["system_acc"], v1_point["mac"]
    wins = [p for p in sweep
            if (p["acc"] >= va and p["avg_mac"] < vm)
            or (p["acc"] > va and p["avg_mac"] <= vm)]
    return sorted(wins, key=lambda r: r["avg_mac"])


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA D -- D1 / ETAP 3: Adaptive Compute + krzywa Pareto")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    all_results = {}

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
        Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

        # --- trening v2 (phased -- z Etapu 2 wiemy, ze ~= end2end) ---
        torch.manual_seed(42)
        v2 = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES).to(device)
        train_phased(v2, Xtr, ytr, epochs=EPOCHS, device=device)
        r2, s2, o2 = evaluate(v2, Xte, yte)
        print(f"v2 top-1: routing={r2*100:.1f}% system={s2*100:.1f}% oracle={o2*100:.1f}%")

        # --- baseline v1 (punkt odniesienia) ---
        torch.manual_seed(42)
        v1 = SeparateV1(N_IN, N_CLASSES, router_enc_hidden=256, router_emb=64,
                        pod_hidden=24).to(device)
        train_separate_v1(v1, Xtr, ytr, epochs=EPOCHS, device=device)
        v1p = v1_top1_point(v1, Xte, yte)
        print(f"v1 top-1: system={v1p['system_acc']*100:.1f}% MAC={v1p['mac']:,}")

        # --- sweep progow v2 -> krzywa Pareto ---
        sweep = adaptive_sweep(v2, Xte, yte)
        print(f"\n{'config':<22}{'acc':>7}{'avgMAC':>9}{'early%':>8}"
              f"{'top1%':>7}{'top2%':>7}{'vs_v1MAC':>9}")
        print("-" * 69)
        for r in sweep:
            mac_ratio = r["avg_mac"] / v1p["mac"]
            print(f"{r['config']:<22}{r['acc']*100:>6.1f}%{r['avg_mac']:>9,}"
                  f"{r['pct_early']:>7.1f}%{r['pct_top1']:>6.1f}%"
                  f"{r['pct_top2']:>6.1f}%{mac_ratio:>8.2f}x")

        front = pareto_front(sweep)
        wins = check_v2_dominates_v1(sweep, v1p)

        print(f"\n  Front Pareto: {len(front)} punktow")
        if wins:
            best = wins[0]  # najnizszy MAC wsrod wygrywajacych
            print(f"  [KRYTERIUM SUKCESU] v2 DOMINUJE v1: prog '{best['config']}' "
                  f"acc={best['acc']*100:.1f}% (v1={v1p['system_acc']*100:.1f}%) "
                  f"MAC={best['avg_mac']:,} (v1={v1p['mac']:,}, "
                  f"{best['avg_mac']/v1p['mac']:.2f}x) -> WIN")
        else:
            print(f"  [KRYTERIUM SUKCESU] v2 NIE dominuje v1 w zadnym rezimie. "
                  f"v2 ma przewage accuracy, ale nie MAC (wiekszy backbone). "
                  f"-> do interpretacji w paperze")

        all_results[ds_name] = {
            "v2_top1": {"routing_acc": round(r2, 4), "system_acc": round(s2, 4),
                        "oracle_acc": round(o2, 4)},
            "v1_top1": {"system_acc": round(v1p["system_acc"], 4), "mac": v1p["mac"]},
            "sweep": sweep,
            "pareto_front": front,
            "v2_dominates_v1": [w["config"] for w in wins],
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "D1b_pareto_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")
    print("\nEtap 3 zakonczony. Krzywa Pareto gotowa do whitepapera.")


if __name__ == "__main__":
    main()
