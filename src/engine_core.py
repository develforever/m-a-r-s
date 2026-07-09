"""
engine_core.py — Engine Core (router) + Specialist Pods z usypianiem.

Router: mała sieć, która z punktu wejściowego przewiduje, KTÓRĄ kapsułę
obudzić. Uczy się sam (z etykiet regionu podczas treningu), bez jawnej
podpowiedzi na wejściu w czasie inferencji. To uczciwy test Engine Core
z dokumentów M.A.R.S.: rozpoznanie, a nie gotowy przełącznik.

Usypianie: przy inferencji liczy tylko router + 1 wybrana kapsuła.
Pozostałe kapsuły śpią (nie wykonują żadnych operacji) -> oszczędność MAC.

Mierzymy MAC dwóch trybów:
  - DENSE (baseline): wejście idzie przez WSZYSTKIE kapsuły, uśrednienie.
  - ROUTED (M.A.R.S.): router + tylko 1 kapsuła.

Kluczowy wniosek z Etapu 3: oszczędność z usypiania rośnie z liczbą
kapsuł N. Przy N=2 routing jest droższy (narzut routera), opłaca się
dopiero przy wielu specjalistach. To realny próg skali architektury.
"""

import numpy as np
from metrics import MACCounter


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Pod:
    """Pojedyncza kapsuła-specjalista (mały klasyfikator binarny)."""
    def __init__(self, n_in, n_hidden, seed, mac):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0, size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 1.0, size=(n_hidden, 1))
        self.b2 = np.zeros((1, 1))
        self.mac = mac

    def forward(self, X, count=True):
        if count:
            self.mac.add_matmul(X.shape, self.W1.shape)
        a1 = np.tanh(X @ self.W1 + self.b1)
        if count:
            self.mac.add_matmul(a1.shape, self.W2.shape)
        a2 = sigmoid(a1 @ self.W2 + self.b2)
        self._cache = (X, a1, a2)
        return a2

    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            a2 = self.forward(X, count=False)
            _, a1, _ = self._cache
            m = X.shape[0]
            d_z2 = (a2 - y) * (a2 * (1 - a2))
            dW2 = a1.T @ d_z2 / m
            db2 = d_z2.mean(axis=0, keepdims=True)
            d_a1 = (d_z2 @ self.W2.T) * (1 - a1 ** 2)
            dW1 = X.T @ d_a1 / m
            db1 = d_a1.mean(axis=0, keepdims=True)
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1


class Router:
    """Mała sieć wybierająca kapsułę (klasyfikacja regionu wejścia)."""
    def __init__(self, n_in, n_hidden, n_pods, seed, mac):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0, size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 1.0, size=(n_hidden, n_pods))
        self.b2 = np.zeros((1, n_pods))
        self.mac = mac

    def forward(self, X, count=True):
        if count:
            self.mac.add_matmul(X.shape, self.W1.shape)
        a1 = np.tanh(X @ self.W1 + self.b1)
        if count:
            self.mac.add_matmul(a1.shape, self.W2.shape)
        logits = a1 @ self.W2 + self.b2
        self._cache = (X, a1, logits)
        return softmax(logits)

    def train(self, X, region, epochs, lr):
        n_pods = self.W2.shape[1]
        Y = np.eye(n_pods)[region]
        for _ in range(epochs):
            p = self.forward(X, count=False)
            _, a1, _ = self._cache
            m = X.shape[0]
            d_logits = (p - Y) / m
            dW2 = a1.T @ d_logits
            db2 = d_logits.sum(axis=0, keepdims=True)
            d_a1 = (d_logits @ self.W2.T) * (1 - a1 ** 2)
            dW1 = X.T @ d_a1
            db1 = d_a1.sum(axis=0, keepdims=True)
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1

    def predict_pod(self, X):
        return self.forward(X, count=True).argmax(axis=1)


class EngineCore:
    def __init__(self, n_in=2, n_pods=3, pod_hidden=8, router_hidden=8, seed=42):
        self.mac = MACCounter()
        self.n_pods = n_pods
        self.router = Router(n_in, router_hidden, n_pods, seed, self.mac)
        self.pods = [Pod(n_in, pod_hidden, seed + 100 + i, self.mac)
                     for i in range(n_pods)]

    def train(self, X, region, y, epochs=3000, lr=0.3):
        # 1. ucz router rozpoznawać region
        self.router.train(X, region, epochs=epochs, lr=lr)
        # 2. ucz każdą kapsułę na JEJ regionie
        for r in range(self.n_pods):
            mask = region == r
            self.pods[r].train(X[mask], y[mask], epochs=epochs, lr=lr)

    def infer_routed(self, X):
        """M.A.R.S.: router + tylko wybrana kapsuła (reszta śpi)."""
        self.mac.reset()
        pod_idx = self.router.predict_pod(X)
        out = np.zeros((X.shape[0], 1))
        for r in range(self.n_pods):
            mask = pod_idx == r
            if mask.any():
                out[mask] = self.pods[r].forward(X[mask], count=True)
        return out, pod_idx, self.mac.mac

    def infer_dense(self, X):
        """Baseline: wejście przez WSZYSTKIE kapsuły, uśrednienie."""
        self.mac.reset()
        outs = [pod.forward(X, count=True) for pod in self.pods]
        return np.mean(outs, axis=0), self.mac.mac
