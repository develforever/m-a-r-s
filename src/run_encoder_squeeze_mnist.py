"""
run_encoder_squeeze_mnist.py — wyciśnięcie: odchudzenie encodera routera.

Odkrycie z analizy tekstur: tekstura (LUT nearest) już jest darmowa.
Wąskim gardłem routera jest ENCODER MLP (784→64→2 = 50,304 MAC).
Test syntetyczny pokazał, że encoder można zmniejszyć ~8× bez straty
jakości routingu. Ten skrypt weryfikuje to na PRAWDZIWYM MNIST.

Dla każdego rozmiaru encodera mierzymy:
  - routing accuracy (czy router trafia w dobry pod)
  - system accuracy (końcowa dokładność M.A.R.S.)
  - MAC routera

Cel: znaleźć najmniejszy encoder, który nie psuje system accuracy.
Każdy zaoszczędzony MAC w routerze to czysty zysk (liczy się per próbka).

Uruchom:
    .venv\\Scripts\\python.exe src\\run_encoder_squeeze_mnist.py
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
from mars_torch import MARSystem

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


def eval_system(mars, test_loader, device):
    mars.eval()
    correct = routing_correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.view(-1, 784).to(device)
            y = y.to(device)
            logits, capsule_ids, _ = mars(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            routing_correct += (capsule_ids == y).sum().item()
            total += len(y)
    return correct / total, routing_correct / total


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 64)
    print("WYCIŚNIĘCIE — odchudzenie encodera routera (MNIST)")
    print(f"Device: {device}")
    print("=" * 64)

    train_loader, test_loader = load_mnist()
    results = {"device": device, "by_encoder_hidden": []}

    print(f"\n{'enc_hidden':>10} {'router_MAC':>11} {'routing_acc':>12} {'system_acc':>11}")
    print("-" * 50)

    for enc_h in [64, 32, 16, 8, 4]:
        mars = MARSystem(n_in=784, n_pods=10, pod_hidden=64, grid_size=64,
                         encoder_hidden=enc_h).to(device)
        mars.train()
        mars.train_system(train_loader, device, epochs_proj=60, epochs_pods=5,
                          lr_proj=0.003, lr_pods=0.001)
        sys_acc, routing_acc = eval_system(mars, test_loader, device)
        router_mac = mars.router.mac_per_sample()

        print(f"{enc_h:>10} {router_mac:>11,} {routing_acc*100:>11.1f}% {sys_acc*100:>10.1f}%")
        results["by_encoder_hidden"].append({
            "encoder_hidden": enc_h,
            "router_mac": router_mac,
            "routing_acc": round(routing_acc, 4),
            "system_acc": round(sys_acc, 4),
        })

    # znajdź najmniejszy encoder z system_acc w granicach 1pp od najlepszego
    best_acc = max(r["system_acc"] for r in results["by_encoder_hidden"])
    acceptable = [r for r in results["by_encoder_hidden"]
                  if r["system_acc"] >= best_acc - 0.01]
    smallest = min(acceptable, key=lambda r: r["encoder_hidden"])

    print("\n--- WNIOSEK ---")
    print(f"Najmniejszy encoder bez straty jakości: hidden={smallest['encoder_hidden']}")
    print(f"Router MAC: {smallest['router_mac']:,} "
          f"(vs 50,304 dla hidden=64 → "
          f"{(1-smallest['router_mac']/50304)*100:.0f}% mniej)")
    print(f"System accuracy: {smallest['system_acc']*100:.1f}%")
    print("\nTekstura pozostaje darmowa (LUT nearest). Oszczędność pochodzi")
    print("z odchudzenia encodera — bez ruszania tekstur ani jakości.")

    results["recommended"] = smallest

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "encoder_squeeze_mnist.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
