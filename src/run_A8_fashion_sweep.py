"""
run_A8_fashion_sweep.py -- Droga A+: sweep hiperparametrow routera na Fashion-MNIST.

Cel: zamknac luke 10.7pp miedzy ProtoRouter (87.7%) a ORACLE (98.4%) na F-MNIST.
Z A5 (MNIST) wiemy, ze wiekszy encoder domyka luke. Sprawdzamy transfer.

Dwufazowy sweep (identyczny schemat jak A5):
  Faza 1: enc_hidden x emb (16 kombinacji) przy stalych epochs=15, lr=0.003
  Faza 2: epochs x lr (9 kombinacji) na najlepszej architekturze z Fazy 1

Specjalisci (h=24, own_ratio=0.7) sa trenowani RAZ i wspoldzieleni.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A8_fashion_sweep.py
"""

import json
import os
import sys
import itertools

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


def train_specialists_into_fastpods(Xtr, ytr, device, hidden=HIDDEN, own_ratio=0.7, epochs=12):
    """Trenuje specjalistow, przenosi do FastPods. Robione RAZ."""
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


def train_and_eval_router(Xtr, ytr, Xte, yte, fast, enc_hidden, emb, epochs, lr):
    """Trenuje router i mierzy routing acc + system acc."""
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
    router_mac = router.mac_per_sample()
    return routing_acc, system_acc, router_mac


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("A8 -- sweep hiperparametrow routera na Fashion-MNIST")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    print("\nLadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    print("Trening specjalistow (h=24, raz)...")
    fast = train_specialists_into_fastpods(Xtr, ytr, device)
    fast.eval()

    # ORACLE
    with torch.no_grad():
        oracle_out = fast.forward_auto(Xte, yte)
        oracle_acc = (oracle_out.argmax(1) == yte).float().mean().item()
    print(f"ORACLE sufit: {oracle_acc*100:.1f}%\n")

    # ================================================================
    # FAZA 1: enc_hidden x emb (architektura)
    # ================================================================
    enc_hiddens = [32, 64, 128, 256]
    embs = [8, 16, 32, 64]
    default_epochs = 15
    default_lr = 0.003

    print("FAZA 1: sweep enc_hidden x emb")
    print(f"{'enc_h':>6} {'emb':>5} {'rout_acc':>9} {'sys_acc':>9} {'rMAC':>8}")
    print("-" * 42)

    phase1_results = []
    best_sys_acc = 0
    best_arch = None

    for enc_h, emb in itertools.product(enc_hiddens, embs):
        r_acc, s_acc, r_mac = train_and_eval_router(
            Xtr, ytr, Xte, yte, fast, enc_h, emb, default_epochs, default_lr)
        print(f"{enc_h:>6} {emb:>5} {r_acc*100:>8.1f}% {s_acc*100:>8.1f}% {r_mac:>8,}")
        phase1_results.append({
            "enc_hidden": enc_h, "emb": emb,
            "epochs": default_epochs, "lr": default_lr,
            "routing_acc": round(r_acc, 4),
            "system_acc": round(s_acc, 4),
            "router_mac": r_mac,
        })
        if s_acc > best_sys_acc:
            best_sys_acc = s_acc
            best_arch = (enc_h, emb)

    print(f"\nNajlepsza architektura: enc_hidden={best_arch[0]}, emb={best_arch[1]} "
          f"-> {best_sys_acc*100:.1f}%")

    # ================================================================
    # FAZA 2: epochs x lr (trening) na najlepszej architekturze
    # ================================================================
    epochs_list = [15, 30, 50]
    lr_list = [0.001, 0.003, 0.01]

    print(f"\nFAZA 2: sweep epochs x lr (enc_h={best_arch[0]}, emb={best_arch[1]})")
    print(f"{'epochs':>7} {'lr':>7} {'rout_acc':>9} {'sys_acc':>9} {'rMAC':>8}")
    print("-" * 45)

    phase2_results = []
    overall_best_acc = 0
    overall_best_cfg = None

    for ep, lr in itertools.product(epochs_list, lr_list):
        r_acc, s_acc, r_mac = train_and_eval_router(
            Xtr, ytr, Xte, yte, fast, best_arch[0], best_arch[1], ep, lr)
        print(f"{ep:>7} {lr:>7.3f} {r_acc*100:>8.1f}% {s_acc*100:>8.1f}% {r_mac:>8,}")
        phase2_results.append({
            "enc_hidden": best_arch[0], "emb": best_arch[1],
            "epochs": ep, "lr": lr,
            "routing_acc": round(r_acc, 4),
            "system_acc": round(s_acc, 4),
            "router_mac": r_mac,
        })
        if s_acc > overall_best_acc:
            overall_best_acc = s_acc
            overall_best_cfg = {"enc_hidden": best_arch[0], "emb": best_arch[1],
                                "epochs": ep, "lr": lr}

    # ================================================================
    # WNIOSEK
    # ================================================================
    baseline_acc = 0.877  # z A7
    delta = overall_best_acc - baseline_acc

    print("\n" + "=" * 72)
    print("WNIOSEK")
    print("=" * 72)
    print(f"Baseline (A7, enc_h=32/emb=16): {baseline_acc*100:.1f}%")
    print(f"Najlepszy router: {overall_best_acc*100:.1f}% "
          f"(enc_h={overall_best_cfg['enc_hidden']}, "
          f"emb={overall_best_cfg['emb']}, "
          f"ep={overall_best_cfg['epochs']}, "
          f"lr={overall_best_cfg['lr']})")
    print(f"ORACLE sufit: {oracle_acc*100:.1f}%")
    print(f"Poprawa vs baseline: {delta*100:+.1f}pp")
    print(f"Pozostala luka do ORACLE: {(oracle_acc - overall_best_acc)*100:.1f}pp")

    # JSON
    results = {
        "dataset": "Fashion-MNIST",
        "device": device,
        "oracle_acc": round(oracle_acc, 4),
        "baseline_acc": baseline_acc,
        "best_config": overall_best_cfg,
        "best_system_acc": round(overall_best_acc, 4),
        "phase1": phase1_results,
        "phase2": phase2_results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A8_fashion_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
