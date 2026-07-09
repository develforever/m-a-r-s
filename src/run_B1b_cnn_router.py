"""
run_B1b_cnn_router.py — CNN-based router for Fashion-MNIST.

DIAGNOZA z B1-B6: routing accuracy capped at ~89% regardless of:
  - encoder depth (B1: +0.68pp)
  - augmentation (B2: HURTS)
  - pod depth (B4: 0pp)
  - own_ratio (B5: 0pp)
  - joint training (B6: +0.45pp)

HIPOTEZA: Bottleneck to flat MLP na 784-dim wektorze. F-MNIST to obrazki
28×28 — spatial structure matters (T-shirt vs Shirt różnią się lokalnymi
wzorami: kołnierzyk, rękawy, guziki). CNN wyłapie te lokalne features.

Testujemy: lekki CNN encoder (2-3 conv layers) → prototype routing.
Cel: przebić 90% routing accuracy.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_B1b_cnn_router.py
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_PODS = 10
N_IN = 784
N_OUT = 10
HIDDEN = 24


class CNNProtoRouter(nn.Module):
    """
    CNN-based prototype router.
    Conv encoder extracts spatial features → embedding → prototype matching.
    """
    def __init__(self, n_pods=10, emb=32, channels=(16, 32)):
        super().__init__()
        self.n_pods = n_pods
        self.emb = emb
        c1, c2 = channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),  # 28×28
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 14×14
            nn.Conv2d(c1, c2, 3, padding=1),  # 14×14
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 7×7
        )
        # 7×7×c2 → emb
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * c2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, emb),
        )
        self.protos = nn.Parameter(torch.randn(n_pods, emb))

    def encode(self, x):
        """x: [B, 784] → [B, emb]"""
        x = x.view(-1, 1, 28, 28)
        features = self.conv(x)
        features = features.view(features.shape[0], -1)
        return self.fc(features)

    def forward(self, x):
        """Logits = -distance to prototypes."""
        e = self.encode(x)
        d = torch.cdist(e, self.protos)
        return -d

    def route(self, x):
        return self.forward(x).argmax(dim=1)

    def mac_per_sample(self):
        # Approximate: conv1 + conv2 + fc
        # Conv1: 1×3×3×16×28×28 = 112,896
        # Conv2: 16×3×3×32×14×14 = 903,168
        # FC: 7*7*32*128 + 128*emb
        c1, c2 = 16, 32
        conv1_mac = 1 * 3 * 3 * c1 * 28 * 28
        conv2_mac = c1 * 3 * 3 * c2 * 14 * 14
        fc_mac = 7 * 7 * c2 * 128 + 128 * self.emb
        proto_mac = self.emb * self.n_pods
        return conv1_mac + conv2_mac + fc_mac + proto_mac


class LightCNNProtoRouter(nn.Module):
    """
    Lighter CNN router — fewer channels, strided conv instead of maxpool.
    """
    def __init__(self, n_pods=10, emb=32, channels=(8, 16)):
        super().__init__()
        self.n_pods = n_pods
        self.emb = emb
        c1, c2 = channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1),  # 14×14
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),  # 7×7
            nn.BatchNorm2d(c2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * c2, emb),
        )
        self.protos = nn.Parameter(torch.randn(n_pods, emb))

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        features = self.conv(x)
        features = features.view(features.shape[0], -1)
        return self.fc(features)

    def forward(self, x):
        e = self.encode(x)
        d = torch.cdist(e, self.protos)
        return -d

    def route(self, x):
        return self.forward(x).argmax(dim=1)

    def mac_per_sample(self):
        c1, c2 = 8, 16
        conv1_mac = 1 * 3 * 3 * c1 * 14 * 14
        conv2_mac = c1 * 3 * 3 * c2 * 7 * 7
        fc_mac = 7 * 7 * c2 * self.emb
        proto_mac = self.emb * self.n_pods
        return conv1_mac + conv2_mac + fc_mac + proto_mac


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
    fast = FastPods(N_PODS, N_IN, hidden, N_OUT).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(nn.Linear(N_IN, hidden), nn.ReLU(), nn.Linear(hidden, N_OUT)).to(device)
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


def train_router(router, Xtr, ytr, epochs=30, lr=0.003, label_smoothing=0.0):
    device = Xtr.device
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def eval_router(router, fast, Xte, yte):
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
    print("B1b — CNN Router dla Fashion-MNIST")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    t0 = time.perf_counter()
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    print("Trening specjalistów...")
    fast = train_specialists(Xtr, ytr, device)
    fast.eval()

    with torch.no_grad():
        oracle_acc = (fast.forward_auto(Xte, yte).argmax(1) == yte).float().mean().item()
    print(f"ORACLE: {oracle_acc*100:.1f}%\n")

    # ================================================================
    # Sweep CNN routerów
    # ================================================================
    configs = [
        # (label, RouterClass, kwargs, epochs, lr, ls)
        ("CNN(16,32) emb32 ep30",   CNNProtoRouter, {"channels": (16, 32), "emb": 32}, 30, 0.003, 0.0),
        ("CNN(16,32) emb32 ep50",   CNNProtoRouter, {"channels": (16, 32), "emb": 32}, 50, 0.003, 0.0),
        ("CNN(16,32) emb64 ep50",   CNNProtoRouter, {"channels": (16, 32), "emb": 64}, 50, 0.003, 0.0),
        ("CNN(16,32) emb32 ls01",   CNNProtoRouter, {"channels": (16, 32), "emb": 32}, 50, 0.003, 0.1),
        ("CNN(32,64) emb32 ep50",   CNNProtoRouter, {"channels": (32, 64), "emb": 32}, 50, 0.003, 0.0),
        ("CNN(32,64) emb64 ep50",   CNNProtoRouter, {"channels": (32, 64), "emb": 64}, 50, 0.003, 0.0),
        ("LightCNN(8,16) emb32",    LightCNNProtoRouter, {"channels": (8, 16), "emb": 32}, 50, 0.003, 0.0),
        ("LightCNN(16,32) emb32",   LightCNNProtoRouter, {"channels": (16, 32), "emb": 32}, 50, 0.003, 0.0),
    ]

    print(f"{'config':>28} {'rout':>6} {'sys':>6} {'MAC':>9}")
    print("-" * 55)

    results_list = []
    best_sys_acc = 0
    best_name = ""

    for (label, RClass, kwargs, epochs, lr, ls) in configs:
        router = RClass(n_pods=N_PODS, **kwargs).to(device)
        train_router(router, Xtr, ytr, epochs=epochs, lr=lr, label_smoothing=ls)
        r_acc, s_acc = eval_router(router, fast, Xte, yte)
        mac = router.mac_per_sample()
        print(f"{label:>28} {r_acc*100:>5.1f}% {s_acc*100:>5.1f}% {mac:>9,}")
        results_list.append({
            "config": label, "routing_acc": round(r_acc, 4),
            "system_acc": round(s_acc, 4), "router_mac": mac,
        })
        if s_acc > best_sys_acc:
            best_sys_acc = s_acc
            best_name = label

    # ================================================================
    # WNIOSEK
    # ================================================================
    elapsed = time.perf_counter() - t0
    baseline_acc = 0.8903  # A9 multi-seed

    print("\n" + "=" * 72)
    print("WNIOSEK — B1b: CNN Router")
    print("=" * 72)
    print(f"Baseline (ProtoRouter MLP, A9): {baseline_acc*100:.2f}%")
    print(f"Best CNN ({best_name}): {best_sys_acc*100:.2f}%")
    print(f"ORACLE: {oracle_acc*100:.2f}%")
    print(f"Poprawa vs baseline: {(best_sys_acc - baseline_acc)*100:+.2f}pp")
    print(f"Luka do ORACLE: {(oracle_acc - best_sys_acc)*100:.2f}pp")
    print(f"Czas: {elapsed:.0f}s")

    results = {
        "experiment": "B1b_cnn_router_fashion",
        "dataset": "Fashion-MNIST",
        "device": device,
        "oracle_acc": round(oracle_acc, 4),
        "baseline_acc": baseline_acc,
        "best_config": best_name,
        "best_system_acc": round(best_sys_acc, 4),
        "delta_vs_baseline_pp": round((best_sys_acc - baseline_acc) * 100, 2),
        "configs": results_list,
        "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "B1b_cnn_router_fashion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
