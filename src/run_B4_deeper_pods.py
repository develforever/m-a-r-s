"""
run_B4_deeper_pods.py — B4: sweep głębszych podów na Fashion-MNIST.

Hipoteza: NarrowPod (1 warstwa, h=24) daje ORACLE=98.56%. Ale routing cap
to ~89-90%. Jeśli głębsze pody dadzą wyższy ORACLE → to raczej nie pomoże
systemowi (bo i tak wąskim gardłem jest routing). ALE — głębsze pody mogą
lepiej kompensować błędy routera (lepsza generalizacja per pod).

Testujemy:
  1L: 784→24→10, 784→32→10, 784→48→10, 784→64→10
  2L: 784→32→16→10, 784→48→24→10, 784→64→32→10

Mierzymy: ORACLE acc, system acc (z ProtoRouter A8 best), MAC per pod.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_B4_deeper_pods.py
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import ProtoRouter
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_PODS = 10
N_IN = 784
N_OUT = 10


def load_fashion_mnist(device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    Xtr = torch.stack([train[i][0].view(-1) for i in range(len(train))]).to(device)
    ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(device)
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))]).to(device)
    yte = torch.tensor([test[i][1] for i in range(len(test))]).to(device)
    return Xtr, ytr, Xte, yte


class FastPods2L(nn.Module):
    """
    FastPods z 2 warstwami ukrytymi: n_in → h1 → h2 → n_out.
    Wagi trzymane jako stacked tensory [N_pods, ...].
    """
    def __init__(self, n_pods, n_in, h1, h2, n_out):
        super().__init__()
        self.n_pods, self.n_in, self.h1, self.h2, self.n_out = n_pods, n_in, h1, h2, n_out
        self.W1 = nn.Parameter(torch.randn(n_pods, n_in, h1) / (n_in ** 0.5))
        self.b1 = nn.Parameter(torch.zeros(n_pods, h1))
        self.W2 = nn.Parameter(torch.randn(n_pods, h1, h2) / (h1 ** 0.5))
        self.b2 = nn.Parameter(torch.zeros(n_pods, h2))
        self.W3 = nn.Parameter(torch.randn(n_pods, h2, n_out) / (h2 ** 0.5))
        self.b3 = nn.Parameter(torch.zeros(n_pods, n_out))

    def forward_auto(self, x, ids):
        """Grouped forward (V2) — best for N=10."""
        out = torch.zeros(x.shape[0], self.n_out, device=x.device, dtype=x.dtype)
        order = torch.argsort(ids)
        x_s = x[order]
        counts = torch.bincount(ids, minlength=self.n_pods)
        start = 0
        for pid in range(self.n_pods):
            c = int(counts[pid].item())
            if c > 0:
                h = torch.relu(x_s[start:start+c] @ self.W1[pid] + self.b1[pid])
                h = torch.relu(h @ self.W2[pid] + self.b2[pid])
                out[order[start:start+c]] = h @ self.W3[pid] + self.b3[pid]
                start += c
        return out

    def mac_per_sample(self):
        return self.n_in * self.h1 + self.h1 * self.h2 + self.h2 * self.n_out


def train_1L_pods(Xtr, ytr, device, hidden, own_ratio=0.7, epochs=12):
    """1-layer pods → FastPods."""
    fast = FastPods(N_PODS, N_IN, hidden, N_OUT).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(nn.Linear(N_IN, hidden), nn.ReLU(), nn.Linear(hidden, N_OUT)).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        mask = ytr == c
        own_X, own_y = Xtr[mask], ytr[mask]
        n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
        X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
        y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        for _ in range(epochs):
            perm = torch.randperm(len(X_pod), device=device)
            for s in range(0, len(X_pod), 256):
                idx = perm[s:s+256]
                loss = crit(pod(X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fast.W1.data[c] = pod[0].weight.data.T
            fast.b1.data[c] = pod[0].bias.data
            fast.W2.data[c] = pod[2].weight.data.T
            fast.b2.data[c] = pod[2].bias.data
    return fast


def train_2L_pods(Xtr, ytr, device, h1, h2, own_ratio=0.7, epochs=12):
    """2-layer pods → FastPods2L."""
    fast = FastPods2L(N_PODS, N_IN, h1, h2, N_OUT).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(
            nn.Linear(N_IN, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, N_OUT)
        ).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        mask = ytr == c
        own_X, own_y = Xtr[mask], ytr[mask]
        n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
        X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
        y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        for _ in range(epochs):
            perm = torch.randperm(len(X_pod), device=device)
            for s in range(0, len(X_pod), 256):
                idx = perm[s:s+256]
                loss = crit(pod(X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fast.W1.data[c] = pod[0].weight.data.T
            fast.b1.data[c] = pod[0].bias.data
            fast.W2.data[c] = pod[2].weight.data.T
            fast.b2.data[c] = pod[2].bias.data
            fast.W3.data[c] = pod[4].weight.data.T
            fast.b3.data[c] = pod[4].bias.data
    return fast


def train_router(Xtr, ytr, device, enc_hidden=256, emb=32, epochs=30, lr=0.003):
    """Trenuje ProtoRouter (A8 config)."""
    router = ProtoRouter(N_IN, N_PODS, enc_hidden=enc_hidden, emb=emb).to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return router


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("B4 — Głębsze pody na Fashion-MNIST")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    t0 = time.perf_counter()

    print("\nŁadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    # Trenuj router RAZ (współdzielony)
    print("Trening routera (enc_h=256, emb=32, ep=30)...")
    router = train_router(Xtr, ytr, device)
    router.eval()
    with torch.no_grad():
        route_ids = router.route(Xte)
        routing_acc = (route_ids == yte).float().mean().item()
    print(f"Routing acc: {routing_acc*100:.1f}%\n")

    # ================================================================
    # SWEEP podów
    # ================================================================
    pod_configs = [
        # (label, type, h1, h2)
        ("1L h=24",    "1L", 24,  None),
        ("1L h=32",    "1L", 32,  None),
        ("1L h=48",    "1L", 48,  None),
        ("1L h=64",    "1L", 64,  None),
        ("1L h=128",   "1L", 128, None),
        ("2L 32→16",   "2L", 32,  16),
        ("2L 48→24",   "2L", 48,  24),
        ("2L 64→32",   "2L", 64,  32),
        ("2L 128→64",  "2L", 128, 64),
    ]

    print(f"{'config':>12} {'oracle':>8} {'sys_acc':>9} {'pod_MAC':>9}")
    print("-" * 42)

    results_list = []
    for (label, ptype, h1, h2) in pod_configs:
        if ptype == "1L":
            fast = train_1L_pods(Xtr, ytr, device, hidden=h1)
            mac = N_IN * h1 + h1 * N_OUT
        else:
            fast = train_2L_pods(Xtr, ytr, device, h1=h1, h2=h2)
            mac = N_IN * h1 + h1 * h2 + h2 * N_OUT
        fast.eval()

        with torch.no_grad():
            oracle_out = fast.forward_auto(Xte, yte)
            oracle_acc = (oracle_out.argmax(1) == yte).float().mean().item()
            sys_out = fast.forward_auto(Xte, route_ids)
            sys_acc = (sys_out.argmax(1) == yte).float().mean().item()

        print(f"{label:>12} {oracle_acc*100:>7.1f}% {sys_acc*100:>8.1f}% {mac:>9,}")
        results_list.append({
            "config": label, "type": ptype, "h1": h1, "h2": h2,
            "oracle_acc": round(oracle_acc, 4),
            "system_acc": round(sys_acc, 4),
            "pod_mac": mac,
        })

    # ================================================================
    # WNIOSEK
    # ================================================================
    elapsed = time.perf_counter() - t0
    best = max(results_list, key=lambda r: r["system_acc"])

    print("\n" + "=" * 72)
    print("WNIOSEK — B4: Głębsze pody")
    print("=" * 72)
    print(f"Routing acc (stały):  {routing_acc*100:.1f}%")
    print(f"Baseline (1L h=24):   oracle={results_list[0]['oracle_acc']*100:.1f}%  "
          f"system={results_list[0]['system_acc']*100:.1f}%")
    print(f"Best ({best['config']}): oracle={best['oracle_acc']*100:.1f}%  "
          f"system={best['system_acc']*100:.1f}%  MAC={best['pod_mac']:,}")
    print(f"Czas: {elapsed:.0f}s")

    # JSON
    results = {
        "experiment": "B4_deeper_pods_fashion",
        "dataset": "Fashion-MNIST",
        "device": device,
        "routing_acc": round(routing_acc, 4),
        "configs": results_list,
        "best": best,
        "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "B4_deeper_pods_fashion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
