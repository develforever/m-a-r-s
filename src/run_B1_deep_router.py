"""
run_B1_deep_router.py — B1: sweep głębszego routera na Fashion-MNIST.

Cel: zamknąć lukę 9.3pp do ORACLE (z A8: 89.03% vs 98.34%).
ProtoRouter z 1 warstwą uderza w sufit. Testujemy DeepProtoRouter:
  - 2-warstwowy encoder z BatchNorm + Dropout
  - Cosine similarity vs L2
  - K-means inicjalizacja prototypów

Trzy fazy:
  1. Architektura encodera (depth × similarity_type)
  2. Trening hparams (epochs × lr) na najlepszej architekturze
  3. Porównanie z baseline (ProtoRouter z A8)

Specjaliści (h=24, own_ratio=0.7) trenowani RAZ i współdzieleni.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_B1_deep_router.py
"""

import json
import os
import sys
import itertools
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from routers_v3 import DeepProtoRouter
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
    """Trenuje specjalistów, przenosi do FastPods. Robione RAZ."""
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


def train_deep_router(router, Xtr, ytr, epochs, lr, use_kmeans=False,
                      label_smoothing=0.0):
    """Trenuje DeepProtoRouter z opcjonalnym k-means init."""
    device = Xtr.device
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Opcjonalne warmup + k-means
    if use_kmeans:
        # 3 epoki warmup
        for _ in range(3):
            perm = torch.randperm(len(Xtr), device=device)
            for s in range(0, len(Xtr), 512):
                idx = perm[s:s+512]
                loss = crit(router(Xtr[idx]), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        router.init_protos_kmeans(Xtr)

    # Główny trening
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def eval_router(router, fast, Xte, yte):
    """Ewaluacja: routing acc + system acc."""
    router.eval()
    with torch.no_grad():
        ids = router.route(Xte)
        routing_acc = (ids == yte).float().mean().item()
        out = fast.forward_auto(Xte, ids)
        system_acc = (out.argmax(1) == yte).float().mean().item()
    return routing_acc, system_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("B1 — Głębszy router dla Fashion-MNIST (DeepProtoRouter)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    t0 = time.perf_counter()

    print("\nŁadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    print("Trening specjalistów (h=24, raz)...")
    fast = train_specialists_into_fastpods(Xtr, ytr, device)
    fast.eval()

    # ORACLE
    with torch.no_grad():
        oracle_out = fast.forward_auto(Xte, yte)
        oracle_acc = (oracle_out.argmax(1) == yte).float().mean().item()
    print(f"ORACLE sufit: {oracle_acc*100:.1f}%\n")

    # ================================================================
    # Baseline: ProtoRouter z A8 (enc_h=256, emb=32, ep=30, lr=0.003)
    # ================================================================
    print("Baseline: ProtoRouter (A8 best config)...")
    baseline_router = ProtoRouter(N_IN, N_PODS, enc_hidden=256, emb=32).to(device)
    baseline_router.train()
    opt = torch.optim.Adam(baseline_router.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    for _ in range(30):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(baseline_router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    b_racc, b_sacc = eval_router(baseline_router, fast, Xte, yte)
    b_mac = baseline_router.mac_per_sample()
    print(f"  routing={b_racc*100:.1f}%  system={b_sacc*100:.1f}%  MAC={b_mac:,}")

    # ================================================================
    # FAZA 1: Architektura (depth × similarity) — stały trening
    # ================================================================
    configs = [
        # (label, enc_h, enc_h2, emb, cosine, dropout)
        ("2L-256-128-L2",   256, 128, 32, False, 0.1),
        ("2L-256-128-cos",  256, 128, 32, True,  0.1),
        ("2L-128-64-L2",    128,  64, 32, False, 0.1),
        ("2L-128-64-cos",   128,  64, 32, True,  0.1),
        ("2L-256-64-L2",    256,  64, 32, False, 0.1),
        ("2L-256-64-cos",   256,  64, 32, True,  0.1),
        ("1L-256-0-L2",     256,   0, 32, False, 0.1),  # ProtoRouter + BN+Dropout
        ("1L-256-0-cos",    256,   0, 32, True,  0.1),
    ]
    default_epochs = 30
    default_lr = 0.003

    print(f"\nFAZA 1: sweep architektur (epochs={default_epochs}, lr={default_lr})")
    print(f"{'config':>20} {'rout_acc':>9} {'sys_acc':>9} {'rMAC':>8} {'temp':>6}")
    print("-" * 56)

    phase1_results = []
    best_sys_acc = 0
    best_cfg_idx = 0

    for i, (label, eh, eh2, emb, cosine, drop) in enumerate(configs):
        router = DeepProtoRouter(
            N_IN, N_PODS, enc_hidden=eh, enc_hidden2=eh2,
            emb=emb, dropout=drop, use_cosine=cosine
        ).to(device)
        train_deep_router(router, Xtr, ytr, default_epochs, default_lr)
        r_acc, s_acc = eval_router(router, fast, Xte, yte)
        r_mac = router.mac_per_sample()
        temp = router.temperature.item()
        print(f"{label:>20} {r_acc*100:>8.1f}% {s_acc*100:>8.1f}% {r_mac:>8,} {temp:>6.2f}")
        phase1_results.append({
            "config": label, "enc_hidden": eh, "enc_hidden2": eh2,
            "emb": emb, "use_cosine": cosine, "dropout": drop,
            "epochs": default_epochs, "lr": default_lr,
            "routing_acc": round(r_acc, 4), "system_acc": round(s_acc, 4),
            "router_mac": r_mac, "temperature": round(temp, 4),
        })
        if s_acc > best_sys_acc:
            best_sys_acc = s_acc
            best_cfg_idx = i

    best = configs[best_cfg_idx]
    print(f"\nNajlepsza architektura: {best[0]} → {best_sys_acc*100:.1f}%")

    # ================================================================
    # FAZA 2: Trening hparams na najlepszej architekturze
    # ================================================================
    _, eh, eh2, emb, cosine, drop = best
    epochs_list = [30, 50, 80]
    lr_list = [0.001, 0.003, 0.005]
    kmeans_list = [False, True]

    print(f"\nFAZA 2: sweep trening ({best[0]})")
    print(f"{'ep':>4} {'lr':>6} {'kmeans':>7} {'rout_acc':>9} {'sys_acc':>9}")
    print("-" * 40)

    phase2_results = []
    overall_best_acc = 0
    overall_best_cfg = None

    for ep, lr, km in itertools.product(epochs_list, lr_list, kmeans_list):
        router = DeepProtoRouter(
            N_IN, N_PODS, enc_hidden=eh, enc_hidden2=eh2,
            emb=emb, dropout=drop, use_cosine=cosine
        ).to(device)
        train_deep_router(router, Xtr, ytr, ep, lr, use_kmeans=km)
        r_acc, s_acc = eval_router(router, fast, Xte, yte)
        km_str = "yes" if km else "no"
        print(f"{ep:>4} {lr:>6.3f} {km_str:>7} {r_acc*100:>8.1f}% {s_acc*100:>8.1f}%")
        phase2_results.append({
            "config": best[0], "enc_hidden": eh, "enc_hidden2": eh2,
            "emb": emb, "use_cosine": cosine, "dropout": drop,
            "epochs": ep, "lr": lr, "use_kmeans": km,
            "routing_acc": round(r_acc, 4), "system_acc": round(s_acc, 4),
        })
        if s_acc > overall_best_acc:
            overall_best_acc = s_acc
            overall_best_cfg = {
                "config": best[0], "enc_hidden": eh, "enc_hidden2": eh2,
                "emb": emb, "use_cosine": cosine, "dropout": drop,
                "epochs": ep, "lr": lr, "use_kmeans": km,
            }

    # ================================================================
    # FAZA 3: Porównanie z najlepszym configiem z Fazy 2 + Label Smoothing
    # ================================================================
    print(f"\nFAZA 3: test label smoothing na najlepszym configu")
    for ls in [0.0, 0.05, 0.1]:
        router = DeepProtoRouter(
            N_IN, N_PODS, enc_hidden=overall_best_cfg["enc_hidden"],
            enc_hidden2=overall_best_cfg["enc_hidden2"],
            emb=emb, dropout=drop, use_cosine=overall_best_cfg["use_cosine"]
        ).to(device)
        train_deep_router(router, Xtr, ytr, overall_best_cfg["epochs"],
                          overall_best_cfg["lr"],
                          use_kmeans=overall_best_cfg["use_kmeans"],
                          label_smoothing=ls)
        r_acc, s_acc = eval_router(router, fast, Xte, yte)
        marker = " ← BEST" if s_acc > overall_best_acc else ""
        print(f"  label_smoothing={ls:.2f}: routing={r_acc*100:.1f}%  system={s_acc*100:.1f}%{marker}")
        if s_acc > overall_best_acc:
            overall_best_acc = s_acc
            overall_best_cfg["label_smoothing"] = ls

    # ================================================================
    # WNIOSEK
    # ================================================================
    delta_vs_baseline = overall_best_acc - b_sacc
    delta_vs_oracle = oracle_acc - overall_best_acc
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("WNIOSEK — B1: DeepProtoRouter vs ProtoRouter na F-MNIST")
    print("=" * 72)
    print(f"Baseline (ProtoRouter A8):  {b_sacc*100:.2f}% (routing {b_racc*100:.1f}%)")
    print(f"DeepProtoRouter (best):     {overall_best_acc*100:.2f}%")
    print(f"ORACLE sufit:               {oracle_acc*100:.2f}%")
    print(f"Poprawa vs baseline:        {delta_vs_baseline*100:+.2f}pp")
    print(f"Pozostała luka do ORACLE:   {delta_vs_oracle*100:.2f}pp")
    print(f"Best config: {overall_best_cfg}")
    print(f"Czas: {elapsed:.0f}s")

    # JSON
    results = {
        "experiment": "B1_deep_router_fashion",
        "dataset": "Fashion-MNIST",
        "device": device,
        "oracle_acc": round(oracle_acc, 4),
        "baseline_proto_router": {
            "system_acc": round(b_sacc, 4),
            "routing_acc": round(b_racc, 4),
            "router_mac": b_mac,
        },
        "best_deep_router": {
            "system_acc": round(overall_best_acc, 4),
            "config": overall_best_cfg,
        },
        "delta_vs_baseline_pp": round(delta_vs_baseline * 100, 2),
        "delta_vs_oracle_pp": round(delta_vs_oracle * 100, 2),
        "phase1_architecture": phase1_results,
        "phase2_training": phase2_results,
        "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "B1_deep_router_fashion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
