"""
run_A9_multiseed.py -- Multi-seed walidacja najlepszych konfiguracji routera.

Cel: confidence intervals dla najlepszych configow z A5 (MNIST) i A8 (F-MNIST).
Kazdy config uruchamiany 5 razy z roznymi seedami (trening specjalistow + router).

Metryki per seed: routing_acc, system_acc, oracle_acc.
Raport: mean, std, min, max.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A9_multiseed.py
"""

import json
import os
import sys
import math

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
N_SEEDS = 5


def load_mnist(device):
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


def train_specialists(Xtr, ytr, device, seed, hidden=HIDDEN, own_ratio=0.7, epochs=12):
    torch.manual_seed(seed)
    fast = FastPods(N_PODS, N_IN, hidden, N_OUT).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, N_OUT)
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


def train_and_eval_router(Xtr, ytr, Xte, yte, fast, cfg, seed):
    device = Xtr.device
    torch.manual_seed(seed + 1000)
    router = ProtoRouter(N_IN, N_PODS,
                         enc_hidden=cfg["enc_hidden"],
                         emb=cfg["emb"]).to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    router.eval()
    with torch.no_grad():
        ids = router.route(Xte)
        routing_acc = (ids == yte).float().mean().item()
        system_acc = (fast.forward_auto(Xte, ids).argmax(1) == yte).float().mean().item()
        oracle_acc = (fast.forward_auto(Xte, yte).argmax(1) == yte).float().mean().item()
    return routing_acc, system_acc, oracle_acc


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0
    std = math.sqrt(var)
    return {"mean": round(mean, 4), "std": round(std, 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_dataset(name, loader, cfg, device):
    print(f"\n{'='*72}")
    print(f"  {name} -- config: enc_h={cfg['enc_hidden']}, emb={cfg['emb']}, "
          f"ep={cfg['epochs']}, lr={cfg['lr']}")
    print(f"{'='*72}")

    print(f"\nLadowanie {name}...")
    Xtr, ytr, Xte, yte = loader(device)

    print(f"{'seed':>6} {'rout_acc':>9} {'sys_acc':>9} {'oracle':>9}")
    print("-" * 38)

    rout_accs, sys_accs, oracle_accs = [], [], []

    for seed in range(N_SEEDS):
        fast = train_specialists(Xtr, ytr, device, seed)
        fast.eval()
        r_acc, s_acc, o_acc = train_and_eval_router(Xtr, ytr, Xte, yte, fast, cfg, seed)
        rout_accs.append(r_acc)
        sys_accs.append(s_acc)
        oracle_accs.append(o_acc)
        print(f"{seed:>6} {r_acc*100:>8.2f}% {s_acc*100:>8.2f}% {o_acc*100:>8.2f}%")

    r_stats = stats(rout_accs)
    s_stats = stats(sys_accs)
    o_stats = stats(oracle_accs)

    print(f"\n  routing:  {r_stats['mean']*100:.2f}% +/- {r_stats['std']*100:.2f}%")
    print(f"  system:   {s_stats['mean']*100:.2f}% +/- {s_stats['std']*100:.2f}%")
    print(f"  oracle:   {o_stats['mean']*100:.2f}% +/- {o_stats['std']*100:.2f}%")
    print(f"  luka do ORACLE: {(o_stats['mean'] - s_stats['mean'])*100:.2f}pp")

    return {
        "dataset": name,
        "config": cfg,
        "n_seeds": N_SEEDS,
        "per_seed": [
            {"seed": s, "routing_acc": round(rout_accs[s], 4),
             "system_acc": round(sys_accs[s], 4),
             "oracle_acc": round(oracle_accs[s], 4)}
            for s in range(N_SEEDS)
        ],
        "stats": {
            "routing_acc": r_stats,
            "system_acc": s_stats,
            "oracle_acc": o_stats,
            "gap_to_oracle_pp": round((o_stats["mean"] - s_stats["mean"]) * 100, 2),
        }
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("A9 -- Multi-seed walidacja najlepszych konfiguracji routera")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print(f"Seeds: {N_SEEDS}")
    print("=" * 72)

    # Best configs from A5 (MNIST) and A8 (F-MNIST)
    mnist_cfg = {"enc_hidden": 256, "emb": 64, "epochs": 50, "lr": 0.001}
    fmnist_cfg = {"enc_hidden": 256, "emb": 32, "epochs": 30, "lr": 0.003}

    mnist_result = run_dataset("MNIST", load_mnist, mnist_cfg, device)
    fmnist_result = run_dataset("Fashion-MNIST", load_fashion_mnist, fmnist_cfg, device)

    # Podsumowanie
    print("\n" + "=" * 72)
    print("PODSUMOWANIE")
    print("=" * 72)
    for r in [mnist_result, fmnist_result]:
        s = r["stats"]
        print(f"  {r['dataset']:<15}: system {s['system_acc']['mean']*100:.2f}% "
              f"+/- {s['system_acc']['std']*100:.2f}%  "
              f"(luka do ORACLE: {s['gap_to_oracle_pp']:.2f}pp)")

    # JSON
    results = {
        "device": device,
        "n_seeds": N_SEEDS,
        "datasets": [mnist_result, fmnist_result],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A9_multiseed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
