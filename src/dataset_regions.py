"""
dataset_regions.py — Zadania w różnych, ROZRÓŻNIALNYCH regionach wejścia.

Po to, by router miał co rozpoznawać. Kluczowa lekcja z poprzednich etapów:
XOR i AND mają identyczne wejścia, więc router nie ma jak ich rozróżnić.
Tutaj każda kapsuła-specjalista obsługuje inny obszar przestrzeni 2D
(jak różne dziedziny w dokumentach M.A.R.S. — np. "Kapsuła Fizyczna"
vs inna). Różne pytania wyglądają różnie — router może je rozpoznać.

3 regiony (klastry) w przestrzeni [0,1]x[0,1]:
  - Region 0: lewy dół   (specjalista 0)
  - Region 1: prawy góra (specjalista 1)
  - Region 2: lewa góra  (specjalista 2)

W każdym regionie jest lokalne zadanie klasyfikacji binarnej (linia
decyzyjna), więc kapsuła faktycznie coś liczy, a nie tylko zwraca
stały region.
"""

import numpy as np


def make_regions(n_per_region=60, seed=0, n_regions=3, sigma=0.07):
    rng = np.random.default_rng(seed)
    if n_regions == 3:
        centers = np.array([[0.2, 0.2], [0.8, 0.8], [0.2, 0.8]])
    else:
        # Generuj centra rozmieszczone na okręgu w [0.2, 0.8]
        angles = np.linspace(0, 2 * np.pi, n_regions, endpoint=False)
        radius = 0.3
        centers = np.column_stack([
            0.5 + radius * np.cos(angles),
            0.5 + radius * np.sin(angles)
        ])
    X_list, region_list, y_list = [], [], []
    for r, c in enumerate(centers):
        pts = c + rng.normal(0, sigma, size=(n_per_region, 2))
        pts = np.clip(pts, 0, 1)
        # lokalne zadanie binarne: po której stronie lokalnej przekątnej
        local = (pts[:, 0] - c[0]) + (pts[:, 1] - c[1])
        y = (local > 0).astype(np.float64).reshape(-1, 1)
        X_list.append(pts)
        region_list.append(np.full(n_per_region, r))
        y_list.append(y)
    X = np.vstack(X_list)
    region = np.concatenate(region_list)
    y = np.vstack(y_list)
    idx = rng.permutation(len(X))
    return X[idx], region[idx], y[idx]


if __name__ == "__main__":
    X, region, y = make_regions()
    print("Kształt X:", X.shape, "regiony:", np.bincount(region))
