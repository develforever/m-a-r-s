"""
capsule_sleep.py — Kapsuła ze wspólną siecią (baseline dla Etapu 2).

Reprezentuje "stary sposób": jedna współdzielona sieć uczona kolejno
zadania A, potem B. Pokazuje problem catastrophic forgetting — nauka B
nadpisuje wiedzę o A.

Zawiera też (wyłączony domyślnie) mechanizm ochrony wag w stylu EWC
(Elastic Weight Consolidation) sterowany flagą use_plasticity. W Etapie 2
udowodniliśmy eksperymentalnie, że na tak małej, w pełni współdzielonej
sieci sam EWC NIE wystarcza — właściwym rozwiązaniem jest modularność
(patrz capsule_modular.py). Kod EWC zostawiamy jako udokumentowany
punkt odniesienia.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CapsuleSleep:
    def __init__(self, n_in=2, n_hidden=8, n_out=1, seed=42, protect=0.95):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0, size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 1.0, size=(n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))
        # plastyczność: 1.0 = w pełni uczący się, 0.0 = zamrożony
        self.plast_W1 = np.ones_like(self.W1)
        self.plast_W2 = np.ones_like(self.W2)
        self.protect = protect

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def _grads(self, X, y):
        m = X.shape[0]
        d_z2 = (self.a2 - y) * (self.a2 * (1 - self.a2))
        dW2 = self.a1.T @ d_z2 / m
        db2 = d_z2.mean(axis=0, keepdims=True)
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (1 - self.a1 ** 2)
        dW1 = X.T @ d_z1 / m
        db1 = d_z1.mean(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def train(self, X, y, epochs=5000, lr=0.5, use_plasticity=True):
        for _ in range(epochs):
            self.forward(X)
            dW1, db1, dW2, db2 = self._grads(X, y)
            if use_plasticity:
                self.W2 -= lr * dW2 * self.plast_W2
                self.W1 -= lr * dW1 * self.plast_W1
            else:
                self.W2 -= lr * dW2
                self.W1 -= lr * dW1
            self.b2 -= lr * db2
            self.b1 -= lr * db1

    def sleep_consolidate(self):
        """
        "SEN" w stylu EWC: oznacz ważne wagi (duże = ważne) i obniż
        ich plastyczność. Udowodniliśmy, że to za słabe na współdzieloną
        sieć — zostawione jako udokumentowany punkt odniesienia.
        """
        imp_W1 = np.abs(self.W1)
        imp_W2 = np.abs(self.W2)
        imp_W1 = imp_W1 / (imp_W1.max() + 1e-9)
        imp_W2 = imp_W2 / (imp_W2.max() + 1e-9)
        self.plast_W1 = np.minimum(self.plast_W1, 1.0 - self.protect * imp_W1)
        self.plast_W2 = np.minimum(self.plast_W2, 1.0 - self.protect * imp_W2)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(np.float64)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))
