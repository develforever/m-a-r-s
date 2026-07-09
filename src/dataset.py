"""
dataset.py — zadania testowe dla Etapu 0 (baseline M.A.R.S.)

Zasada: zadania muszą być MAŁE i DETERMINISTYCZNE, żeby porównania
między baseline (backpropagation) a przyszłymi kapsułami M.A.R.S.
były powtarzalne i uczciwe.

Na start dajemy XOR — klasyczny, nieliniowo separowalny problem,
którego pojedynczy perceptron NIE potrafi rozwiązać, a mała sieć
z jedną warstwą ukrytą potrafi. To minimalny, ale niebanalny test.
"""

import numpy as np


def xor_dataset():
    """
    Klasyczny problem XOR.

    Zwraca:
        X: wejścia, kształt (4, 2)
        y: etykiety, kształt (4, 1)
    """
    X = np.array(
        [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]],
        dtype=np.float64,
    )
    y = np.array(
        [[0.0],
         [1.0],
         [1.0],
         [0.0]],
        dtype=np.float64,
    )
    return X, y


def task_A_dataset():
    """
    Zadanie A — dla testu catastrophic forgetting (przyda się w Etapie 2).
    Tu: XOR.
    """
    return xor_dataset()


def task_B_dataset():
    """
    Zadanie B — dla testu catastrophic forgetting (Etap 2).
    Tu: AND (inny rozkład niż XOR, żeby sprawdzić, czy nauka B
    niszczy wiedzę o A).

    Zwraca:
        X: wejścia, kształt (4, 2)
        y: etykiety, kształt (4, 1)
    """
    X = np.array(
        [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]],
        dtype=np.float64,
    )
    y = np.array(
        [[0.0],
         [0.0],
         [0.0],
         [1.0]],
        dtype=np.float64,
    )
    return X, y


if __name__ == "__main__":
    X, y = xor_dataset()
    print("XOR dataset:")
    for xi, yi in zip(X, y):
        print(f"  {xi} -> {yi[0]}")
