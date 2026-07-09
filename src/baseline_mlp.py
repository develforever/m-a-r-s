"""
baseline_mlp.py — klasyczna mini-sieć (MLP) z backpropagation w czystym NumPy.

To jest PUNKT ODNIESIENIA (baseline) dla całego projektu M.A.R.S.
Reprezentuje "stary paradygmat": pełna wsteczna propagacja błędu,
globalna aktualizacja wszystkich wag.

Każde przyszłe rozwiązanie M.A.R.S. (uczenie lokalne, kapsuły, sen)
będzie porównywane DOKŁADNIE z tymi liczbami: dokładność, liczba MAC,
czas. Dlatego kod jest celowo prosty i jawnie liczy każdą operację.

Architektura: 2 wejścia -> warstwa ukryta (tanh) -> 1 wyjście (sigmoid).
Uczenie: gradient descent z pełną backpropagation.
"""

import numpy as np
from metrics import MACCounter


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv_from_output(out):
    # pochodna sigmoidy wyrażona przez jej wyjście
    return out * (1.0 - out)


def tanh_deriv_from_output(out):
    # pochodna tanh wyrażona przez jej wyjście
    return 1.0 - out ** 2


class BaselineMLP:
    def __init__(self, n_in=2, n_hidden=4, n_out=1, seed=42):
        rng = np.random.default_rng(seed)
        # inicjalizacja wag — mała, losowa, deterministyczna (seed)
        self.W1 = rng.normal(0, 1.0, size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 1.0, size=(n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))
        self.mac = MACCounter()

    def forward(self, X):
        # warstwa 1
        self.mac.add_matmul(X.shape, self.W1.shape)
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        # warstwa 2
        self.mac.add_matmul(self.a1.shape, self.W2.shape)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, lr):
        m = X.shape[0]

        # gradient na wyjściu (MSE + sigmoid)
        d_a2 = (self.a2 - y)                       # (m, n_out)
        d_z2 = d_a2 * sigmoid_deriv_from_output(self.a2)

        # gradienty warstwy 2
        self.mac.add_matmul((self.a1.shape[1], m), (m, d_z2.shape[1]))
        dW2 = self.a1.T @ d_z2 / m
        db2 = d_z2.mean(axis=0, keepdims=True)

        # propagacja błędu do warstwy ukrytej
        self.mac.add_matmul(d_z2.shape, (self.W2.shape[1], self.W2.shape[0]))
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * tanh_deriv_from_output(self.a1)

        # gradienty warstwy 1
        self.mac.add_matmul((X.shape[1], m), (m, d_z1.shape[1]))
        dW1 = X.T @ d_z1 / m
        db1 = d_z1.mean(axis=0, keepdims=True)

        # aktualizacja wag (gradient descent)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=5000, lr=0.5, verbose=False):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, lr)
            if verbose and epoch % 500 == 0:
                loss = np.mean((self.a2 - y) ** 2)
                print(f"  epoch {epoch:5d}  loss={loss:.5f}")

    def predict(self, X):
        out = self.forward(X)
        return (out > 0.5).astype(np.float64)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))
