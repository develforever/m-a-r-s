"""
run_A2b_stronger_router.py — Droga A, krok A2b: odzyskać accuracy.

A2 pokazało: system specjalistów osiąga 94.3%, ograniczony routerem (94%).
Hipoteza: mocniejszy router (ProtoRouter 16D = 96.5%) podniesie sufit.

Ten skrypt mierzy TRZY scenariusze na tym samym zestawie specjalistów:
  - Router 8D  (94%, obecny)
  - Router 16D (96.5%, mocniejszy)
  - ORACLE     (router idealny — górny limit samych specjalistów)

ORACLE jest kluczowy: rozdziela błąd ROUTERA od błędu SPECJALISTY.
  - ORACLE wysoki → router jest sufitem, 16D pomoże.
  - ORACLE niski  → problem w specjalistach, mocniejszy router nie pomoże.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A2b_stronger_router.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from mars_specialists import NarrowPod
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


def train_router(router, Xtr, ytr, epochs=15, lr=0.003):
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def train_specialists(n_pods, Xtr, ytr, device, hidden=24, own_ratio=0.7, epochs=15):
    pods = nn.ModuleList([NarrowPod(784, n_pods, hidden=hidden) for _ in range(n_pods)]).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(n_pods):
        opt = torch.optim.Adam(pods[c].parameters(), lr=0.001)
        mask = ytr == c
        own_X, own_y = Xtr[mask], ytr[mask]
        n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
        X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
        y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        for _ in range(epochs):
            perm = torch.randperm(len(X_pod), device=device)
            for s in range(0, len(X_pod), 256):
                idx = perm[s:s+256]
                loss = crit(pods[c](X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
    return pods


def eval_system(pods, n_pods, capsule_ids, Xte, yte, device):
    with torch.no_grad():
        out = torch.zeros(len(Xte), n_pods, device=device)
        for pid in range(n_pods):
            m = capsule_ids == pid
            if m.any():
                out[m] = pods[pid](Xte[m])
        return (out.argmax(1) == yte).float().mean().item()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_pods = 10
    print("=" * 64)
    print("DROGA A — krok A2b: odzyskać accuracy (mocniejszy router)")
    print(f"Device: {device}")
    print("=" * 64)

    print("\nŁadowanie MNIST...")
    Xtr, ytr, Xte, yte = load_mnist_tensors(device)

    print("Trening wąskich specjalistów (hidden=24, dane 70/30)...")
    pods = train_specialists(n_pods, Xtr, ytr, device)

    results = {"device": device, "scenarios": []}

    # Router 8D
    print("\nTrening routera 8D...")
    r8 = ProtoRouter(784, n_pods, enc_hidden=16, emb=8).to(device)
    train_router(r8, Xtr, ytr)
    with torch.no_grad():
        ids8 = r8.route(Xte)
        racc8 = (ids8 == yte).float().mean().item()
    sacc8 = eval_system(pods, n_pods, ids8, Xte, yte, device)

    # Router 16D
    print("Trening routera 16D...")
    r16 = ProtoRouter(784, n_pods, enc_hidden=32, emb=16).to(device)
    train_router(r16, Xtr, ytr)
    with torch.no_grad():
        ids16 = r16.route(Xte)
        racc16 = (ids16 == yte).float().mean().item()
    sacc16 = eval_system(pods, n_pods, ids16, Xte, yte, device)

    # ORACLE (router idealny)
    oracle_ids = yte.clone()
    sacc_oracle = eval_system(pods, n_pods, oracle_ids, Xte, yte, device)

    pod_mac = pods[0].mac_per_sample()
    baseline = 234752

    print("\n" + "=" * 64)
    print("PORÓWNANIE (te same specjaliści, różne routery)")
    print("=" * 64)
    print(f"{'Scenariusz':<22} {'routing':>9} {'system':>8} {'total MAC':>11} {'oszcz.':>8}")
    print("-" * 60)
    for name, racc, sacc, rmac in [
        ("Router 8D (obecny)", racc8, sacc8, r8.mac_per_sample()),
        ("Router 16D", racc16, sacc16, r16.mac_per_sample()),
        ("ORACLE (idealny)", 1.00, sacc_oracle, 0),
    ]:
        total = rmac + pod_mac
        sav = (1 - total / baseline) * 100 if rmac > 0 else 0
        sav_str = f"{sav:.1f}%" if rmac > 0 else "—"
        print(f"{name:<22} {racc*100:>8.1f}% {sacc*100:>7.1f}% {total:>11,} {sav_str:>8}")
        results["scenarios"].append({
            "name": name, "routing_acc": round(racc, 4),
            "system_acc": round(sacc, 4), "router_mac": rmac, "total_mac": rmac + pod_mac,
        })

    print("\n--- WNIOSEK ---")
    print(f"ORACLE (górny limit specjalistów): {sacc_oracle*100:.1f}%")
    if sacc_oracle > 0.96:
        print("Specjaliści są DOBRZY — router jest sufitem.")
        gain = (sacc16 - sacc8) * 100
        print(f"Router 16D podniósł system o {gain:+.1f}pp ({sacc8*100:.1f}% → {sacc16*100:.1f}%)")
        if sacc16 >= 0.96:
            print("ODZYSKANE: router 16D + specjaliści dorównuje redundantnemu (96%)")
            print("przy 62% mniejszych podach. Najlepszy z obu światów.")
    else:
        print(f"Specjaliści sami ograniczają się do {sacc_oracle*100:.1f}% —")
        print("mocniejszy router nie wystarczy, trzeba lepszych specjalistów.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A2b_stronger_router.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
