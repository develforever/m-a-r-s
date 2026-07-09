"""
run_etap0.py — uruchamia baseline, mierzy koszt, zapisuje wynik.

To jest skrypt wykonawczy Etapu 0. Po jego uruchomieniu masz:
  - potwierdzenie, że baseline uczy się XOR,
  - zmierzoną liczbę operacji MAC (koszt obliczeniowy),
  - czas wykonania (proxy energii),
  - zapisany rekord JSON w results/, który będzie PUNKTEM ODNIESIENIA
    dla wszystkich przyszłych etapów M.A.R.S.

Uruchom:
    cd src
    python run_etap0.py
"""

import json
import os
from datetime import datetime

from dataset import xor_dataset
from baseline_mlp import BaselineMLP
from metrics import EnergyTimer, summarize


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    print("=" * 60)
    print("ETAP 0 — Baseline (backpropagation) na zadaniu XOR")
    print("=" * 60)

    X, y = xor_dataset()

    EPOCHS = 5000
    LR = 0.5

    model = BaselineMLP(n_in=2, n_hidden=4, n_out=1, seed=42)

    print(f"\nTrening: {EPOCHS} epok, learning_rate={LR}")
    with EnergyTimer() as timer:
        model.train(X, y, epochs=EPOCHS, lr=LR, verbose=True)

    acc = model.accuracy(X, y)

    print("\n--- WYNIKI ---")
    print(f"Dokładność:        {acc * 100:.1f}%")
    print(f"Operacje MAC:      {model.mac.mac:,}")
    print(f"Czas (wall):       {timer.wall_time * 1000:.2f} ms")
    print(f"Czas (CPU):        {timer.cpu_time * 1000:.2f} ms")

    # sprawdzenie predykcji
    print("\nPredykcje XOR:")
    preds = model.predict(X)
    for xi, yi, pi in zip(X, y, preds):
        ok = "OK" if yi[0] == pi[0] else "BŁĄD"
        print(f"  {xi} -> oczek. {yi[0]:.0f}, model {pi[0]:.0f}  [{ok}]")

    record = summarize(
        name="baseline_mlp_xor_backprop",
        mac_counter=model.mac,
        energy_timer=timer,
        accuracy=acc,
        extra={
            "task": "XOR",
            "architecture": "2-4-1 MLP, tanh+sigmoid",
            "learning": "full backpropagation",
            "epochs": EPOCHS,
            "learning_rate": LR,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "etap0_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\nWynik zapisany: {os.path.abspath(out_path)}")
    print("\nTo jest TWÓJ PUNKT ODNIESIENIA. Każdy przyszły etap M.A.R.S.")
    print("porównujemy dokładnie z tymi liczbami.")


if __name__ == "__main__":
    main()
