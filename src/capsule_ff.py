"""
capsule_ff.py — Kapsuła ucząca się metodą Forward-Forward (Hinton, 2022).

Idea Forward-Forward (FF):
  Zamiast jednego przejścia w przód + przejścia wstecz (backprop),
  robimy DWA przejścia w przód:
    - "pozytywne": prawdziwe dane (poprawna para wejście + etykieta)
    - "negatywne": dane fałszywe (zła etykieta wstrzyknięta do wejścia)
  Każda warstwa ma LOKALNY cel: dla danych pozytywnych jej aktywność
  (suma kwadratów = "goodness") ma być WYSOKA, dla negatywnych NISKA.
  Uczenie jest lokalne — każda warstwa optymalizuje swój własny cel,
  bez globalnej propagacji błędu wstecz.

Kodowanie etykiety dla XOR:
  Doklejamy one-hot etykiety pomnożony przez współczynnik (label_scale),
  żeby sygnał klasy był wyraźny (słaby sygnał etykiety = FF utyka na 50%).
  Klasyfikacja: dla każdej z 2 klas liczymy łączną goodness, wybieramy
  klasę o wyższej.

Parametry sprawdzone eksperymentalnie (5/5 sukcesów na różnych seedach):
  n_hidden=16, label_scale=4.0, lr=0.05, epochs=2000.
"""

import numpy as np
from metrics import MACCounter


def relu(x):
    return np.maximum(0.0, x)


class FFLayer:
    def __init__(self, n_in, n_out, mac, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))
        self.b = np.zeros((1, n_out))
        self.mac = mac
        self.threshold = 2.0  # próg goodness

    def forward(self, x):
        self.mac.add_matmul(x.shape, self.W.shape)
        pre = x @ self.W + self.b
        return relu(pre)

    def goodness(self, out):
        # goodness = średnia z kwadratów aktywacji
        return (out ** 2).mean(axis=1, keepdims=True)

    def train_step(self, x_pos, x_neg, lr):
        # przejście w przód dla pozytywnych i negatywnych
        out_pos = self.forward(x_pos)
        out_neg = self.forward(x_neg)
        g_pos = self.goodness(out_pos)   # chcemy wysokie
        g_neg = self.goodness(out_neg)   # chcemy niskie

        # logistyczny cel: goodness pozytywnych > próg, negatywnych < próg
        p_pos = 1.0 / (1.0 + np.exp(-(g_pos - self.threshold)))
        p_neg = 1.0 / (1.0 + np.exp(-(g_neg - self.threshold)))

        # gradient LOKALNY (bez propagacji przez inne warstwy)
        n_out = out_pos.shape[1]
        grad_pos = (p_pos - 1.0) * (2.0 * out_pos / n_out)  # pchnij w górę
        grad_neg = (p_neg - 0.0) * (2.0 * out_neg / n_out)  # pchnij w dół

        self.mac.add_matmul((x_pos.shape[1], x_pos.shape[0]), grad_pos.shape)
        dW_pos = x_pos.T @ grad_pos / x_pos.shape[0]
        self.mac.add_matmul((x_neg.shape[1], x_neg.shape[0]), grad_neg.shape)
        dW_neg = x_neg.T @ grad_neg / x_neg.shape[0]

        self.W -= lr * (dW_pos + dW_neg)
        self.b -= lr * (grad_pos.mean(axis=0, keepdims=True)
                        + grad_neg.mean(axis=0, keepdims=True))

        # znormalizowane aktywacje jako wejście do następnej warstwy
        return self._normalize(out_pos), self._normalize(out_neg)

    def _normalize(self, out):
        # normalizacja długości — klasyczny trik FF, by następna warstwa
        # czytała kierunek aktywacji, a nie samą "goodness"
        norm = np.sqrt((out ** 2).sum(axis=1, keepdims=True)) + 1e-9
        return out / norm


class CapsuleFF:
    def __init__(self, n_in=2, n_hidden=16, n_layers=2, seed=42, label_scale=4.0):
        self.mac = MACCounter()
        self.n_in = n_in
        self.label_scale = label_scale
        # one-hot dla 2 klas -> +2 wejścia
        dims = [n_in + 2] + [n_hidden] * n_layers
        self.layers = [
            FFLayer(dims[i], dims[i + 1], self.mac, seed=seed + i)
            for i in range(len(dims) - 1)
        ]

    def _embed(self, X, label):
        onehot = np.zeros((X.shape[0], 2))
        onehot[:, label] = self.label_scale
        return np.concatenate([X, onehot], axis=1)

    def train(self, X, y, epochs=2000, lr=0.05, verbose=False):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                xi = X[i:i+1]
                yi = int(y[i, 0])
                x_pos = self._embed(xi, yi)
                x_neg = self._embed(xi, 1 - yi)
                for layer in self.layers:
                    x_pos, x_neg = layer.train_step(x_pos, x_neg, lr)
            if verbose and epoch % 200 == 0:
                acc = self.accuracy(X, y)
                print(f"  [FF] epoch {epoch:5d}  acc={acc*100:.1f}%")

    def _total_goodness(self, X, label):
        x = self._embed(X, label)
        total = np.zeros((X.shape[0], 1))
        for layer in self.layers:
            out = layer.forward(x)
            total += layer.goodness(out)
            x = layer._normalize(out)
        return total

    def predict(self, X):
        g0 = self._total_goodness(X, 0)
        g1 = self._total_goodness(X, 1)
        return (g1 > g0).astype(np.float64)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))
