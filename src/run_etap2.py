"""
run_etap2.py — Etap 2: cykl snu i ochrona wiedzy (catastrophic forgetting).

Porównuje dwa podejścia na sekwencji zadań A (XOR) -> B (AND):
  - BASELINE: jedna współdzielona sieć, uczona kolejno A potem B.
  - M.A.R.S.: modularna kapsuła z osobnymi pulami neuronów + router,
    który kieruje każde zadanie do jego własnej puli. "Sen" = zamrożenie
    puli A przy nauce B.

Mierzymy RETENCJĘ A (ile zostało po nauce B) ORAZ naukę B jednocześnie.
Uczciwy test: przewaga liczy się tylko, gdy A zachowane I B nauczone —
inaczej można by "oszukać" zamrażając wszystko i nie ucząc B.

Uruchom:
    cd src
    python run_etap2.py
"""

import json
import os
from datetime import datetime
import numpy as np

from dataset import task_A_dataset, task_B_dataset
from capsule_modular import CapsuleModular
from capsule_sleep import CapsuleSleep

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_SEEDS = 5


def run_baseline(seed):
    XA, yA = task_A_dataset()
    XB, yB = task_B_dataset()
    cap = CapsuleSleep(n_in=2, n_hidden=16, n_out=1, seed=seed)
    cap.train(XA, yA, epochs=5000, lr=0.5, use_plasticity=False)
    a_before = cap.accuracy(XA, yA)
    cap.train(XB, yB, epochs=5000, lr=0.5, use_plasticity=False)
    return a_before, cap.accuracy(XA, yA), cap.accuracy(XB, yB)


def run_mars(seed):
    XA, yA = task_A_dataset()
    XB, yB = task_B_dataset()
    cap = CapsuleModular(n_in=2, pod_size=8, n_pods=2, seed=seed)
    cap.set_active_pod(0)
    cap.train(XA, yA, epochs=5000, lr=0.5)
    a_before = cap.accuracy(XA, yA, pod_idx=0)
    cap.set_active_pod(1)               # "SEN": zamrożenie puli A
    cap.train(XB, yB, epochs=5000, lr=0.5)
    # router: zadanie A -> pula 0, zadanie B -> pula 1
    return a_before, cap.accuracy(XA, yA, pod_idx=0), cap.accuracy(XB, yB, pod_idx=1)


def aggregate(fn):
    before, after, b = [], [], []
    for s in range(N_SEEDS):
        x0, x1, x2 = fn(s)
        before.append(x0); after.append(x1); b.append(x2)
    return {
        "A_before_mean": float(np.mean(before)),
        "A_after_mean": float(np.mean(after)),
        "B_mean": float(np.mean(b)),
        "A_after_min": float(np.min(after)),
        "seeds": N_SEEDS,
    }


def ascii_bar(label, value, width=30):
    filled = int(round(value * width))
    return f"  {label:<22} |{'#'*filled}{'.'*(width-filled)}| {value*100:.0f}%"


def main():
    print("=" * 64)
    print("ETAP 2 — Cykl snu i ochrona wiedzy (catastrophic forgetting)")
    print("Sekwencja: zadanie A (XOR) -> zadanie B (AND)")
    print("=" * 64)

    base = aggregate(run_baseline)
    mars = aggregate(run_mars)

    print(f"\nUśrednione po {N_SEEDS} seedach.\n")
    print("BASELINE (współdzielona sieć):")
    print(ascii_bar("A przed nauką B", base["A_before_mean"]))
    print(ascii_bar("A PO nauce B", base["A_after_mean"]))
    print(ascii_bar("B (nowe zadanie)", base["B_mean"]))

    print("\nM.A.R.S. (modularny + router):")
    print(ascii_bar("A przed nauką B", mars["A_before_mean"]))
    print(ascii_bar("A PO nauce B", mars["A_after_mean"]))
    print(ascii_bar("B (nowe zadanie)", mars["B_mean"]))

    retention_gain = mars["A_after_mean"] - base["A_after_mean"]
    print("\n--- WNIOSEK ---")
    print(f"Retencja A: baseline {base['A_after_mean']*100:.0f}% "
          f"vs M.A.R.S. {mars['A_after_mean']*100:.0f}% "
          f"(+{retention_gain*100:.0f} pkt proc.)")
    print(f"Nauka B:    baseline {base['B_mean']*100:.0f}% "
          f"vs M.A.R.S. {mars['B_mean']*100:.0f}%")
    if retention_gain > 0.2 and mars["B_mean"] >= base["B_mean"]:
        print("M.A.R.S. zachowuje starą wiedzę BEZ poświęcania nowej —")
        print("to realna przewaga modularności nad współdzieloną siecią.")
    print("\nUwaga: A przed nauką B u M.A.R.S. może być nieco niższe")
    print("(mniejsza pula: 8 neuronów zamiast 16). To uczciwy koszt modularności.")

    out = {
        "stage": "etap2",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task_sequence": "XOR -> AND",
        "baseline": base,
        "mars_modular": mars,
        "retention_gain_pp": retention_gain * 100,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "etap2_forgetting.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
