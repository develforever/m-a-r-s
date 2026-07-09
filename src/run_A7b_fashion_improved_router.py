"""
run_A7b_fashion_improved_router.py -- szybki test: lepszy router na Fashion-MNIST.

Z A7 wiemy: routing acc na F-MNIST = 87.7% (luka 10.7pp do ORACLE 98.4%).
Z A5 wiemy: wiekszy encoder domyka luke na MNIST (96.7% -> 98.2%).

Pytanie: ile domknie na Fashion-MNIST?
Test: enc_hidden=128,emb=16 (sweet spot z A5) i enc_hidden=256,emb=64 (best z A5).

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A7b_fashion_improved_router.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import ProtoRouter
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_IN = 784
N_PODS = 10
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


def train_specialists(Xtr, ytr, device, hidden=HIDDEN, own_ratio=0.7, epochs=12):
    fast = FastPods(N_PODS, N_IN, hidden, N_PODS).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, N_PODS)
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
    return fast


def train_and_eval_router(Xtr, ytr, Xte, yte, fast, enc_hidden, emb, epochs, lr):
    device = Xtr.device
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
    router.eval()
    with torch.no_grad():
        ids = router.route(Xte)
        routing_acc = (ids == yte).float().mean().item()
        out = fast.forward_auto(Xte, ids)
        system_acc = (out.argmax(1) == yte).float().mean().item()
    return routing_acc, system_acc, router.mac_per_sample()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("A7b: lepszy router na Fashion-MNIST")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    print("\nLadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    print("Trening specjalistow (h=24, raz)...")
    fast = train_specialists(Xtr, ytr, device)
    fast.eval()

    with torch.no_grad():
        oracle_acc = (fast.forward_auto(Xte, yte).argmax(1) == yte).float().mean().item()
    print(f"ORACLE sufit: {oracle_acc*100:.1f}%\n")

    configs = [
        ("A7 baseline", 32, 16, 15, 0.003),
        ("A5 sweet-spot", 128, 16, 30, 0.001),
        ("A5 best", 256, 64, 50, 0.001),
    ]

    print(f"{'config':<16} {'enc_h':>6} {'emb':>5} {'ep':>4} "
          f"{'rout_acc':>9} {'sys_acc':>9} {'rMAC':>8}")
    print("-" * 65)

    rows = []
    for name, enc_h, emb, ep, lr in configs:
        r_acc, s_acc, r_mac = train_and_eval_router(
            Xtr, ytr, Xte, yte, fast, enc_h, emb, ep, lr)
        print(f"{name:<16} {enc_h:>6} {emb:>5} {ep:>4} "
              f"{r_acc*100:>8.1f}% {s_acc*100:>8.1f}% {r_mac:>8,}")
        rows.append({
            "name": name, "enc_hidden": enc_h, "emb": emb,
            "epochs": ep, "lr": lr,
            "routing_acc": round(r_acc, 4),
            "system_acc": round(s_acc, 4),
            "router_mac": r_mac,
        })

    best = max(rows, key=lambda r: r["system_acc"])
    baseline = rows[0]
    delta = best["system_acc"] - baseline["system_acc"]

    print(f"\n--- WNIOSEK ---")
    print(f"Baseline (enc_h=32): {baseline['system_acc']*100:.1f}%")
    print(f"Best ({best['name']}): {best['system_acc']*100:.1f}%")
    print(f"ORACLE: {oracle_acc*100:.1f}%")
    print(f"Poprawa: {delta*100:+.1f}pp")
    print(f"Luka do ORACLE: {(oracle_acc - best['system_acc'])*100:.1f}pp")

    results = {
        "device": device,
        "oracle_acc": round(oracle_acc, 4),
        "configs": rows,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A7b_fashion_improved_router.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
