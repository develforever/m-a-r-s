"""
run_B5_own_ratio.py — B5: sweep own_ratio na Fashion-MNIST.

Hipoteza: own_ratio=0.7 z A2 było heurystyką, nigdy sweepowane.
Na F-MNIST, gdzie routing acc=89%, pod dostaje ~11% "obcych" próbek.
Niższe own_ratio = pod lepiej radzi z cudzymi danymi (robustniejszy),
ale traci specjalizację. Wyższe = mocniejszy na swoich, ale kruchy.

Kluczowe pytanie: jaki own_ratio minimalizuje impact błędów routera?

Uruchom:
    .venv\\Scripts\\python.exe src\\run_B5_own_ratio.py
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
HIDDEN = 24


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


def train_pods(Xtr, ytr, device, hidden=HIDDEN, own_ratio=0.7, epochs=12):
    """Trenuje specjalistów z danym own_ratio."""
    fast = FastPods(N_PODS, N_IN, hidden, N_OUT).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(nn.Linear(N_IN, hidden), nn.ReLU(), nn.Linear(hidden, N_OUT)).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        mask = ytr == c
        own_X, own_y = Xtr[mask], ytr[mask]
        if own_ratio < 1.0:
            n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
            X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
            y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        else:
            X_pod, y_pod = own_X, own_y
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("B5 — Sweep own_ratio na Fashion-MNIST")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    t0 = time.perf_counter()

    print("\nŁadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    # Trenuj router RAZ
    print("Trening routera (enc_h=256, emb=32, ep=30)...")
    router = ProtoRouter(N_IN, N_PODS, enc_hidden=256, emb=32).to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    for _ in range(30):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    router.eval()
    with torch.no_grad():
        route_ids = router.route(Xte)
        routing_acc = (route_ids == yte).float().mean().item()
    print(f"Routing acc: {routing_acc*100:.1f}%\n")

    # ================================================================
    # Sweep own_ratio
    # ================================================================
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"{'own_ratio':>10} {'oracle':>8} {'sys_acc':>9} {'gap':>6}")
    print("-" * 36)

    results_list = []
    for ratio in ratios:
        fast = train_pods(Xtr, ytr, device, own_ratio=ratio)
        fast.eval()
        with torch.no_grad():
            oracle_out = fast.forward_auto(Xte, yte)
            oracle_acc = (oracle_out.argmax(1) == yte).float().mean().item()
            sys_out = fast.forward_auto(Xte, route_ids)
            sys_acc = (sys_out.argmax(1) == yte).float().mean().item()
        gap = oracle_acc - sys_acc
        print(f"{ratio:>10.1f} {oracle_acc*100:>7.1f}% {sys_acc*100:>8.1f}% {gap*100:>5.1f}")
        results_list.append({
            "own_ratio": ratio,
            "oracle_acc": round(oracle_acc, 4),
            "system_acc": round(sys_acc, 4),
            "routing_oracle_gap_pp": round(gap * 100, 2),
        })

    # ================================================================
    # WNIOSEK
    # ================================================================
    elapsed = time.perf_counter() - t0
    best = max(results_list, key=lambda r: r["system_acc"])
    baseline = next(r for r in results_list if r["own_ratio"] == 0.7)

    print("\n" + "=" * 72)
    print("WNIOSEK — B5: own_ratio sweep")
    print("=" * 72)
    print(f"Routing acc: {routing_acc*100:.1f}%")
    print(f"Baseline (own_ratio=0.7): oracle={baseline['oracle_acc']*100:.1f}%  "
          f"system={baseline['system_acc']*100:.1f}%")
    print(f"Best (own_ratio={best['own_ratio']}): oracle={best['oracle_acc']*100:.1f}%  "
          f"system={best['system_acc']*100:.1f}%")
    print(f"Poprawa vs baseline: {(best['system_acc'] - baseline['system_acc'])*100:+.2f}pp")
    print(f"Czas: {elapsed:.0f}s")

    # JSON
    results = {
        "experiment": "B5_own_ratio_fashion",
        "dataset": "Fashion-MNIST",
        "device": device,
        "routing_acc": round(routing_acc, 4),
        "configs": results_list,
        "best": best,
        "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "B5_own_ratio_fashion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
