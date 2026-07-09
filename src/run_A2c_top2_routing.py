"""
run_A2c_top2_routing.py — Droga A: top-2 routing (wyciśnięcie ku sufitowi 99%).

A2b pokazało: specjaliści mają sufit ORACLE 99%, ale top-1 routing daje 96%.
Luka 3pp wynika z błędów routera (zły 1 pod = zła odpowiedź).

Top-2 routing: router wskazuje 2 najbliższe pody, system łączy ich odpowiedzi.
Gdy router myli top-1, prawidłowy pod często jest w top-2 → ratunek.

Trzy strategie łączenia (mierzymy którą najlepsza):
  - confidence: weź pod o wyższej pewności (max softmax)
  - agregacja: zsumuj softmax obu podów (ensemble 2)
  - top-1 (odniesienie)

KOSZT: top-2 aktywuje 2 pody = 2× pod MAC. Trade-off: ile accuracy za 2× MAC.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A2c_top2_routing.py
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
N_PODS = 10


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


def train_specialists(Xtr, ytr, device, hidden=24, own_ratio=0.7, epochs=15):
    pods = nn.ModuleList([NarrowPod(784, N_PODS, hidden=hidden) for _ in range(N_PODS)]).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
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


def all_pod_outputs(pods, X):
    """Predykcje wszystkich podów dla wszystkich próbek: [N, n_pods, n_classes]."""
    with torch.no_grad():
        return torch.stack([pods[p](X) for p in range(N_PODS)], dim=1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 64)
    print("DROGA A — top-2 routing (wyciśnięcie ku sufitowi 99%)")
    print(f"Device: {device}")
    print("=" * 64)

    print("\nŁadowanie MNIST...")
    Xtr, ytr, Xte, yte = load_mnist_tensors(device)

    print("Trening routera 16D + specjalistów...")
    router = ProtoRouter(784, N_PODS, enc_hidden=32, emb=16).to(device)
    train_router(router, Xtr, ytr)
    pods = train_specialists(Xtr, ytr, device)

    with torch.no_grad():
        scores = router(Xte)  # [N, n_pods] (im wyżej tym bliżej)
        top2 = scores.topk(2, dim=1).indices  # [N, 2]
        pod_out = all_pod_outputs(pods, Xte)  # [N, n_pods, n_classes]
        soft = torch.softmax(pod_out, dim=2)  # softmax per pod

        N = len(Xte)
        ar = torch.arange(N, device=device)

        # TOP-1
        ids1 = scores.argmax(1)
        out1 = pod_out[ar, ids1]
        acc1 = (out1.argmax(1) == yte).float().mean().item()

        # TOP-2 confidence
        p0, p1 = top2[:, 0], top2[:, 1]
        soft0, soft1 = soft[ar, p0], soft[ar, p1]
        conf0, conf1 = soft0.max(1).values, soft1.max(1).values
        pick0 = conf0 >= conf1
        out_conf = torch.where(pick0.unsqueeze(1), pod_out[ar, p0], pod_out[ar, p1])
        acc_conf = (out_conf.argmax(1) == yte).float().mean().item()

        # TOP-2 agregacja
        out_agg = soft0 + soft1
        acc_agg = (out_agg.argmax(1) == yte).float().mean().item()

        # ORACLE
        out_o = pod_out[ar, yte]
        acc_oracle = (out_o.argmax(1) == yte).float().mean().item()

    router_mac = router.mac_per_sample()
    pod_mac = pods[0].mac_per_sample()
    baseline = 234752

    print("\n" + "=" * 64)
    print("PORÓWNANIE")
    print("=" * 64)
    print(f"{'Strategia':<22} {'accuracy':>9} {'pod MAC':>9} {'total':>9} {'oszczędn.':>10}")
    print("-" * 62)
    configs = [
        ("top-1 (1 pod)", acc1, pod_mac),
        ("top-2 confidence", acc_conf, pod_mac * 2),
        ("top-2 agregacja", acc_agg, pod_mac * 2),
        ("ORACLE (sufit)", acc_oracle, pod_mac),
    ]
    results = {"device": device, "strategies": []}
    for name, acc, pmac in configs:
        total = router_mac + pmac
        sav = (1 - total / baseline) * 100
        sav_str = f"{sav:.1f}%" if "ORACLE" not in name else "—"
        print(f"{name:<22} {acc*100:>8.1f}% {pmac:>9,} {total:>9,} {sav_str:>10}")
        results["strategies"].append({
            "name": name, "accuracy": round(acc, 4),
            "pod_mac": pmac, "total_mac": total,
        })

    best2 = max(acc_conf, acc_agg)
    gain = (best2 - acc1) * 100
    print("\n--- WNIOSEK ---")
    print(f"Top-2 podnosi accuracy o {gain:+.1f}pp ({acc1*100:.1f}% → {best2*100:.1f}%)")
    print(f"Koszt: 2× pod MAC. Sufit ORACLE: {acc_oracle*100:.1f}%")
    if gain > 1.0:
        print(f"Top-2 warty rozważenia: +{gain:.1f}pp za 2× pod MAC.")
        print("Decyzja: czy accuracy warta podwojenia kosztu podów — zależy od celu.")
    else:
        print("Top-2 daje mało — router top-1 jest już blisko swojego limitu.")
        print("Lepiej inwestować w mocniejszy router niż w drugi pod.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A2c_top2_routing.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
