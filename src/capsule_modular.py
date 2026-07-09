"""
capsule_modular.py — Modularna kapsuła z osobnymi pulami neuronów per zadanie.

Sedno M.A.R.S. ("Specialist Pods"): rozłączne grupy neuronów ukrytych
dla różnych zadań, przy WSPÓLNYM wejściu (realna modularność, a nie dwie
osobne sieci w przebraniu).

Kluczowa lekcja z debugowania tego etapu:
  1. Sama ochrona wag (EWC) nie wystarcza, gdy zadania współdzielą
     pojemność jednej małej sieci — XOR i AND walczą o te same neurony.
  2. POJEDYNCZY współdzielony parametr (bias wyjścia) wystarczy, by
     zniszczyć modularność — przesuwa próg decyzyjny dla wszystkich
     zadań naraz. Dlatego KAŻDA pula ma swój własny bias wyjścia,
     a ROUTER wybiera, która pula odpowiada za dane zadanie.

Mechanizm "snu":
  Po nauce zadania A zamrażamy jego pulę (nie aktywujemy jej do nauki).
  Nauka B rusza tylko pulę B i jej własny bias. Pula A jest nietykalna
  -> zero zapominania A, a B uczy się we własnej przestrzeni.

To minimalny działający prototyp Engine Core (router) + Specialist Pods
z dokumentacji M.A.R.S.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CapsuleModular:
    def __init__(self, n_in=2, pod_size=8, n_pods=2, seed=42):
        rng = np.random.default_rng(seed)
        self.pod_size = pod_size
        self.n_pods = n_pods
        n_hidden = pod_size * n_pods
        self.W1 = rng.normal(0, 1.0, size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 1.0, size=(n_hidden, 1))
        # KAŻDA pula ma swój własny bias wyjścia (router-zależnie)
        self.b2_per_pod = np.zeros(n_pods)
        self.active_pod = 0

    def set_active_pod(self, pod_idx):
        self.active_pod = pod_idx

    def _pod_slice(self, pod_idx):
        start = pod_idx * self.pod_size
        return slice(start, start + self.pod_size)

    def forward(self, X, pod_idx=None):
        """
        Liczy wyjście używając TYLKO neuronów danej puli (router).
        Pozostałe pule nie uczestniczą — jak "uśpione" kapsuły w M.A.R.S.
        """
        if pod_idx is None:
            pod_idx = self.active_pod
        sl = self._pod_slice(pod_idx)
        z1 = X @ self.W1[:, sl] + self.b1[:, sl]
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2[sl] + self.b2_per_pod[pod_idx]
        a2 = sigmoid(z2)
        self._cache = (X, sl, a1, a2, pod_idx)
        return a2

    def train(self, X, y, epochs=5000, lr=0.5):
        pod = self.active_pod
        sl = self._pod_slice(pod)
        for _ in range(epochs):
            a2 = self.forward(X, pod)
            _, _, a1, _, _ = self._cache
            m = X.shape[0]
            d_z2 = (a2 - y) * (a2 * (1 - a2))
            dW2 = a1.T @ d_z2 / m
            db2 = float(d_z2.mean())
            d_a1 = d_z2 @ self.W2[sl].T
            d_z1 = d_a1 * (1 - a1 ** 2)
            dW1 = X.T @ d_z1 / m
            db1 = d_z1.mean(axis=0, keepdims=True)

            # aktualizujemy TYLKO aktywną pulę + jej własny bias wyjścia
            self.W1[:, sl] -= lr * dW1
            self.b1[:, sl] -= lr * db1
            self.W2[sl] -= lr * dW2
            self.b2_per_pod[pod] -= lr * db2

    def predict(self, X, pod_idx=None):
        return (self.forward(X, pod_idx) > 0.5).astype(np.float64)

    def accuracy(self, X, y, pod_idx=None):
        return float(np.mean(self.predict(X, pod_idx) == y))
