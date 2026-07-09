"""
mars_specialists.py — M.A.R.S. Droga A: prawdziwa specjalizacja (krok A2).

Łączy ProtoRouter (z A1, ~94% na MNIST) z WĄSKIMI specjalistami.
W przeciwieństwie do starego systemu (gdzie każdy pod był redundantnym
pełnym klasyfikatorem 10 klas, hidden=64), tutaj:

  - pody są WĘŻSZE (mniejszy hidden) → mniej MAC,
  - uczone z przewagą swoich danych (specjalizacja),
  - router faktycznie ma znaczenie (jego trafność ~= jakość systemu).

To realizuje "Specialist Pods" z dokumentów M.A.R.S. — nie ensemble
redundantnych kopii, lecz prawdziwie wyspecjalizowane jednostki.

Architektura specjalisty: dla odporności na błędy routera pody zachowują
pełne wyjście (10 klas) i są uczone z domieszką cudzych danych (70/30),
żeby przy błędnym routingu móc skorygować odpowiedź. To kompromis między
czystą specjalizacją a robustnością.
"""
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import ProtoRouter


class NarrowPod(nn.Module):
    """Wąski specjalista — mniejszy MLP niż redundantny pod."""
    def __init__(self, n_in, n_out, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
        self.n_in = n_in
        self.hidden = hidden
        self.n_out = n_out

    def forward(self, x):
        return self.net(x)

    def mac_per_sample(self):
        return self.n_in * self.hidden + self.hidden * self.n_out


class SpecialistSystem(nn.Module):
    """
    ProtoRouter + wąscy specjaliści. Prawdziwa modularność.
    """
    def __init__(self, n_in, n_pods, pod_hidden=24, router_emb=8, router_enc=16):
        super().__init__()
        self.n_in = n_in
        self.n_pods = n_pods
        self.pod_hidden = pod_hidden
        self.router = ProtoRouter(n_in, n_pods, enc_hidden=router_enc, emb=router_emb)
        self.pods = nn.ModuleList([
            NarrowPod(n_in, n_pods, hidden=pod_hidden) for _ in range(n_pods)
        ])

    def forward(self, x):
        """Router wybiera specjalistę, aktywuje tylko jego (reszta śpi)."""
        capsule_ids = self.router.route(x)
        out = torch.zeros(x.shape[0], self.n_pods, device=x.device)
        for pid in range(self.n_pods):
            mask = capsule_ids == pid
            if mask.any():
                out[mask] = self.pods[pid](x[mask])
        return out, capsule_ids

    def train_router(self, X, y, epochs=15, lr=0.003):
        self.router.train()
        opt = torch.optim.Adam(self.router.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 512):
                idx = perm[s:s+512]
                loss = crit(self.router(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    def train_specialists(self, X, y, epochs=15, lr=0.001, own_ratio=0.7):
        """
        Każdy specjalista uczony z PRZEWAGĄ swoich danych (own_ratio),
        z domieszką cudzych — dla odporności na błędy routera.
        """
        crit = nn.CrossEntropyLoss()
        for c in range(self.n_pods):
            opt = torch.optim.Adam(self.pods[c].parameters(), lr=lr)
            mask = y == c
            own_X, own_y = X[mask], y[mask]
            n_own = len(own_X)
            n_other = int(n_own * (1 - own_ratio) / own_ratio)
            other_X, other_y = X[~mask][:n_other], y[~mask][:n_other]
            X_pod = torch.cat([own_X, other_X])
            y_pod = torch.cat([own_y, other_y])
            for _ in range(epochs):
                perm = torch.randperm(len(X_pod))
                for s in range(0, len(X_pod), 256):
                    idx = perm[s:s+256]
                    loss = crit(self.pods[c](X_pod[idx]), y_pod[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def mac_routed(self):
        return self.router.mac_per_sample() + self.pods[0].mac_per_sample()
