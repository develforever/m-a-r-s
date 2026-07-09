"""
run_A1_router.py — Droga A, krok A1: lepszy router (zastąpienie tekstury 2D).

Mierzy routing accuracy na PRAWDZIWYM MNIST dla:
  - MLPRouter (bottleneck 4D)
  - ProtoRouter (embedding 8D)
  - punkt odniesienia: stary router teksturowy dawał ~44%

Cel: przebić 85% routing accuracy. To warunek konieczny dla kroku A2
(wąskie pody), bo przy specjalizacji system accuracy = router accuracy.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A1_router.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import MLPRouter, ProtoRouter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_mnist(batch_size=512):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size, shuffle=False))


def train_router(router, train_loader, device, epochs=15, lr=0.003):
    router.to(device).train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for X, y in train_loader:
            X = X.view(-1, 784).to(device)
            y = y.to(device)
            loss = crit(router(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()


def eval_router(router, test_loader, device):
    router.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.view(-1, 784).to(device)
            y = y.to(device)
            pred = router.route(X)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 64)
    print("DROGA A — krok A1: lepszy router (zastąpienie tekstury 2D)")
    print(f"Device: {device}")
    print("=" * 64)
    print("\nStary router teksturowy (2D-UV): ~44% routing acc (punkt odniesienia)")
    print("Cel: przebić 85%, by umożliwić specjalizację podów (krok A2).\n")

    train_loader, test_loader = load_mnist()
    results = {"device": device, "baseline_texture_router_acc": 0.44, "routers": []}

    candidates = [
        ("MLPRouter (bottleneck=4)", MLPRouter(784, 10, enc_hidden=16, bottleneck=4)),
        ("MLPRouter (bottleneck=8)", MLPRouter(784, 10, enc_hidden=32, bottleneck=8)),
        ("ProtoRouter (emb=8)", ProtoRouter(784, 10, enc_hidden=16, emb=8)),
        ("ProtoRouter (emb=16)", ProtoRouter(784, 10, enc_hidden=32, emb=16)),
    ]

    print(f"{'Router':<28} {'routing_acc':>12} {'MAC':>10} {'status':>14}")
    print("-" * 66)
    for name, router in candidates:
        train_router(router, train_loader, device, epochs=15, lr=0.003)
        acc = eval_router(router, test_loader, device)
        mac = router.mac_per_sample()
        status = "PRZEBIJA 85%" if acc > 0.85 else "poniżej progu"
        print(f"{name:<28} {acc*100:>11.1f}% {mac:>10,} {status:>14}")
        results["routers"].append({
            "name": name, "routing_acc": round(acc, 4),
            "mac": mac, "passes_85": acc > 0.85,
        })

    best = max(results["routers"], key=lambda r: r["routing_acc"])
    print("\n--- WNIOSEK ---")
    print(f"Najlepszy router: {best['name']} — {best['routing_acc']*100:.1f}% "
          f"(vs 44% tekstura)")
    if best["routing_acc"] > 0.85:
        print("Próg 85% przebity → krok A2 (wąskie pody) ma sens.")
        print("Router przestaje być atrapą — staje się sercem specjalizacji.")
    else:
        print("Próg 85% NIE przebity → trzeba mocniejszego routera przed A2.")
    results["best"] = best

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A1_router.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
