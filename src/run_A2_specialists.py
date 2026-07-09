"""
run_A2_specialists.py — Droga A, krok A2: prawdziwa specjalizacja na MNIST.

Test prawdy modularności. Porównuje na PRAWDZIWYM MNIST:
  - REDUNDANTNY (stary): ProtoRouter + pełne pody (hidden=64, wszystkie dane)
  - SPECJALIŚCI (nowy):   ProtoRouter + wąskie pody (hidden=24, swoje dane 70/30)

Mierzy: system accuracy, MAC poda, całkowita oszczędność.

Pytanie rozstrzygające: czy specjaliści dają porównywalną accuracy przy
MNIEJSZYCH podach? Jeśli tak — M.A.R.S. jest wreszcie prawdziwie modularny.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A2_specialists.py
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
from mars_specialists import SpecialistSystem, NarrowPod
from routers_v2 import ProtoRouter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_mnist_tensors(device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    Xtr = torch.stack([train[i][0].view(-1) for i in range(len(train))]).to(device)
    ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(device)
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))]).to(device)
    yte = torch.tensor([test[i][1] for i in range(len(test))]).to(device)
    return Xtr, ytr, Xte, yte


def eval_specialist_system(system, Xte, yte):
    system.eval()
    with torch.no_grad():
        out, capsule_ids = system(Xte)
        pred = out.argmax(dim=1)
        acc = (pred == yte).float().mean().item()
        routing_acc = (capsule_ids == yte).float().mean().item()
    return acc, routing_acc


def build_redundant(n_in, n_pods, Xtr, ytr, device):
    """Stary system: ProtoRouter + pełne pody (hidden=64, wszystkie dane)."""
    router = ProtoRouter(n_in, n_pods, enc_hidden=16, emb=8).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    for _ in range(15):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    pods = nn.ModuleList([NarrowPod(n_in, n_pods, hidden=64) for _ in range(n_pods)]).to(device)
    for c in range(n_pods):
        opt = torch.optim.Adam(pods[c].parameters(), lr=0.001)
        for _ in range(5):
            perm = torch.randperm(len(Xtr), device=device)
            for s in range(0, len(Xtr), 512):
                idx = perm[s:s+512]
                loss = crit(pods[c](Xtr[idx]), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()

    def evaluate(Xte, yte):
        with torch.no_grad():
            ids = router.route(Xte)
            out = torch.zeros(len(Xte), n_pods, device=device)
            for pid in range(n_pods):
                m = ids == pid
                if m.any():
                    out[m] = pods[pid](Xte[m])
            return (out.argmax(1) == yte).float().mean().item()

    pod_mac = n_in * 64 + 64 * n_pods
    return evaluate, router.mac_per_sample(), pod_mac


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 64)
    print("DROGA A — krok A2: prawdziwa specjalizacja (MNIST)")
    print(f"Device: {device}")
    print("=" * 64)

    print("\nŁadowanie MNIST...")
    Xtr, ytr, Xte, yte = load_mnist_tensors(device)

    # ─── SPECJALIŚCI (nowy) ───
    print("\n[SPECJALIŚCI] ProtoRouter + wąskie pody (hidden=24, dane 70/30)")
    spec = SpecialistSystem(784, 10, pod_hidden=24, router_emb=8).to(device)
    print("  trening routera...")
    spec.train_router(Xtr, ytr, epochs=15)
    print("  trening specjalistów...")
    spec.train_specialists(Xtr, ytr, epochs=15, own_ratio=0.7)
    spec_acc, spec_routing = eval_specialist_system(spec, Xte, yte)
    spec_pod_mac = spec.pods[0].mac_per_sample()
    spec_router_mac = spec.router.mac_per_sample()
    print(f"  system acc={spec_acc*100:.1f}%  routing acc={spec_routing*100:.1f}%")
    print(f"  pod MAC={spec_pod_mac:,}  router MAC={spec_router_mac:,}")

    # ─── REDUNDANTNY (stary) ───
    print("\n[REDUNDANTNY] ProtoRouter + pełne pody (hidden=64, wszystkie dane)")
    redundant_eval, red_router_mac, red_pod_mac = build_redundant(784, 10, Xtr, ytr, device)
    red_acc = redundant_eval(Xte, yte)
    print(f"  system acc={red_acc*100:.1f}%")
    print(f"  pod MAC={red_pod_mac:,}  router MAC={red_router_mac:,}")

    # ─── PORÓWNANIE ───
    baseline_mlp = 234752
    spec_total = spec_router_mac + spec_pod_mac
    red_total = red_router_mac + red_pod_mac

    print("\n" + "=" * 64)
    print("PORÓWNANIE")
    print("=" * 64)
    print(f"{'System':<22} {'acc':>7} {'pod MAC':>10} {'total MAC':>11} {'oszczędność':>12}")
    print("-" * 64)
    print(f"{'Redundantny (stary)':<22} {red_acc*100:>6.1f}% {red_pod_mac:>10,}"
          f" {red_total:>11,} {(1-red_total/baseline_mlp)*100:>11.1f}%")
    print(f"{'Specjaliści (nowy)':<22} {spec_acc*100:>6.1f}% {spec_pod_mac:>10,}"
          f" {spec_total:>11,} {(1-spec_total/baseline_mlp)*100:>11.1f}%")

    pod_reduction = (1 - spec_pod_mac / red_pod_mac) * 100
    acc_diff = (spec_acc - red_acc) * 100

    print("\n--- WNIOSEK ---")
    print(f"Pod specjalisty jest {pod_reduction:.0f}% mniejszy od redundantnego.")
    print(f"Różnica accuracy: {acc_diff:+.1f} pp")
    if acc_diff > -2.0 and pod_reduction > 30:
        print("SPECJALIZACJA DZIAŁA: mniejsze pody, porównywalna jakość.")
        print("M.A.R.S. jest wreszcie prawdziwie modularny (nie ensemble kopii).")
    elif acc_diff <= -2.0:
        print("Specjalizacja kosztuje accuracy — router (94%) ścina jakość.")
        print("Opcja: mocniejszy router (ProtoRouter 16D = 96.5%) lub top-2 routing.")
    else:
        print("Wynik niejednoznaczny — wymaga strojenia.")

    results = {
        "device": device,
        "specialists": {
            "system_acc": round(spec_acc, 4), "routing_acc": round(spec_routing, 4),
            "pod_mac": spec_pod_mac, "router_mac": spec_router_mac, "total_mac": spec_total,
        },
        "redundant": {
            "system_acc": round(red_acc, 4),
            "pod_mac": red_pod_mac, "router_mac": red_router_mac, "total_mac": red_total,
        },
        "pod_size_reduction_pct": round(pod_reduction, 1),
        "acc_diff_pp": round(acc_diff, 2),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A2_specialists.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
