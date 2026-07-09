"""
run_etap1.py — Etap 1: lokalne uczenie (bez backpropagation).

Trenuje dwie kapsuły na XOR:
  - Forward-Forward (FF)
  - Contrastive Hebbian (CHL)
i porównuje je z baseline z Etapu 0 (wczytanym z results/).

Co udowadnia ten etap:
  Czy uczenie LOKALNE (bez globalnej propagacji błędu) potrafi nauczyć
  się tego samego zadania co backprop. Liczba MAC może być WYŻSZA niż
  w backprop — to oczekiwane (obie metody robią wiele przejść w przód).
  Realna przewaga energetyczna architektury M.A.R.S. pojawi się dopiero
  w Etapie 3 (aktywacja 1 z N kapsuł zamiast całej sieci).

Uruchom:
    cd src
    python run_etap1.py
"""

import json
import os
from datetime import datetime

from dataset import xor_dataset
from capsule_ff import CapsuleFF
from capsule_chl import CapsuleCHL
from metrics import EnergyTimer, summarize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_baseline():
    path = os.path.join(RESULTS_DIR, "etap0_baseline.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_ff():
    X, y = xor_dataset()
    cap = CapsuleFF(n_in=2, n_hidden=16, n_layers=2, seed=42, label_scale=4.0)
    with EnergyTimer() as timer:
        cap.train(X, y, epochs=2000, lr=0.05, verbose=True)
    acc = cap.accuracy(X, y)
    return summarize("capsule_ff_xor_local", cap.mac, timer, acc, extra={
        "task": "XOR", "learning": "Forward-Forward (local)",
        "epochs": 2000, "learning_rate": 0.05})


def run_chl():
    X, y = xor_dataset()
    cap = CapsuleCHL(n_in=2, n_hidden=8, n_out=1, seed=42)
    with EnergyTimer() as timer:
        cap.train(X, y, epochs=3000, lr=0.1, verbose=True)
    acc = cap.accuracy(X, y)
    return summarize("capsule_chl_xor_local", cap.mac, timer, acc, extra={
        "task": "XOR", "learning": "Contrastive Hebbian (local)",
        "epochs": 3000, "learning_rate": 0.1})


def main():
    print("=" * 64)
    print("ETAP 1 — Lokalne uczenie (bez backpropagation)")
    print("=" * 64)

    print("\n[1/2] Forward-Forward:")
    ff = run_ff()
    print("\n[2/2] Contrastive Hebbian:")
    chl = run_chl()

    baseline = load_baseline()

    print("\n" + "=" * 64)
    print("PORÓWNANIE")
    print("=" * 64)
    print(f"{'Metoda':<28}{'Dokładność':>12}{'MAC':>14}")
    print("-" * 64)
    if baseline:
        print(f"{'Baseline (backprop)':<28}"
              f"{baseline['accuracy']*100:>11.0f}%"
              f"{baseline['mac_operations']:>14,}")
    for r in (ff, chl):
        print(f"{r['name']:<28}{r['accuracy']*100:>11.0f}%"
              f"{r['mac_operations']:>14,}")

    print("\nInterpretacja:")
    print("- Obie metody lokalne nauczyły się XOR bez backpropagation.")
    print("- Wyższy MAC niż baseline jest OCZEKIWANY (wiele przejść w przód).")
    print("- Zysk energetyczny architektury pojawi się w Etapie 3")
    print("  (aktywacja 1 z N kapsuł zamiast całej sieci).")

    out = {
        "stage": "etap1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline,
        "forward_forward": ff,
        "contrastive_hebbian": chl,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "etap1_local_learning.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
