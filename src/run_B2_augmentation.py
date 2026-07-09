"""
run_B2_augmentation.py — B2: augmentacja + CosineAnnealing na Fashion-MNIST.

Problem z B1: głębszy encoder dał tylko +0.68pp. Wnioskujemy, że bottleneck
to NIE encoder capacity, lecz trudność dystrybucji F-MNIST (T-shirt ≈ Shirt ≈ Coat).
Augmentacja poszerza rozkład treningowy, scheduler stabilizuje końcówkę treningu.

Testujemy osobno i łącznie:
  A) Data augmentation (RandomAffine: rotation, translation, scale)
  B) CosineAnnealingLR (z warmup)
  C) Label smoothing (0.1)
  D) Większe emb (64, 128) — może 32 to za mało na F-MNIST

Używamy ProtoRouter (A8 config, szybszy) — jeśli augmentacja pomoże tutaj,
będzie działać też z DeepProtoRouter.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_B2_augmentation.py
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
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import ProtoRouter
from routers_v3 import DeepProtoRouter
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


def augment_batch(X, device):
    """
    Augmentacja na GPU: random affine na 28×28 obrazkach.
    X: [B, 784] → reshape → augment → reshape back.
    """
    B = X.shape[0]
    imgs = X.view(B, 1, 28, 28)

    # Random affine: rotation ±15°, translate ±10%, scale 0.9-1.1
    angle = (torch.rand(B, device=device) - 0.5) * 30  # ±15°
    translate_x = (torch.rand(B, device=device) - 0.5) * 0.2  # ±10%
    translate_y = (torch.rand(B, device=device) - 0.5) * 0.2
    scale = 0.9 + torch.rand(B, device=device) * 0.2  # 0.9-1.1

    # Build affine matrices
    cos_a = torch.cos(angle * 3.14159 / 180)
    sin_a = torch.sin(angle * 3.14159 / 180)

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos_a * scale
    theta[:, 0, 1] = -sin_a * scale
    theta[:, 0, 2] = translate_x
    theta[:, 1, 0] = sin_a * scale
    theta[:, 1, 1] = cos_a * scale
    theta[:, 1, 2] = translate_y

    grid = F.affine_grid(theta, imgs.shape, align_corners=False)
    augmented = F.grid_sample(imgs, grid, align_corners=False, padding_mode='zeros')
    return augmented.view(B, 784)


def train_specialists_into_fastpods(Xtr, ytr, device, hidden=HIDDEN, own_ratio=0.7,
                                     epochs=12, use_augment=False):
    """Trenuje specjalistów z opcjonalną augmentacją."""
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
                xb = X_pod[idx]
                if use_augment:
                    xb = augment_batch(xb, device)
                loss = crit(pod(xb), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fast.W1.data[c] = pod[0].weight.data.T
            fast.b1.data[c] = pod[0].bias.data
            fast.W2.data[c] = pod[2].weight.data.T
            fast.b2.data[c] = pod[2].bias.data
    return fast


def train_router_augmented(router, Xtr, ytr, epochs, lr, use_augment=False,
                           use_scheduler=False, label_smoothing=0.0):
    """Trenuje router z opcjonalną augmentacją i schedulerem."""
    device = Xtr.device
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            xb = Xtr[idx]
            if use_augment:
                xb = augment_batch(xb, device)
            loss = crit(router(xb), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if scheduler is not None:
            scheduler.step()


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
    print("B2 — Augmentacja + Scheduler na Fashion-MNIST")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    t0 = time.perf_counter()

    print("\nŁadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    # ================================================================
    # Trening specjalistów: bez augment (baseline) i z augment
    # ================================================================
    print("Trening specjalistów BEZ augmentacji...")
    fast_plain = train_specialists_into_fastpods(Xtr, ytr, device, use_augment=False)
    fast_plain.eval()

    print("Trening specjalistów Z augmentacją...")
    fast_aug = train_specialists_into_fastpods(Xtr, ytr, device, use_augment=True)
    fast_aug.eval()

    # ORACLE
    with torch.no_grad():
        oracle_plain = (fast_plain.forward_auto(Xte, yte).argmax(1) == yte).float().mean().item()
        oracle_aug = (fast_aug.forward_auto(Xte, yte).argmax(1) == yte).float().mean().item()
    print(f"ORACLE (plain pods):  {oracle_plain*100:.1f}%")
    print(f"ORACLE (aug pods):    {oracle_aug*100:.1f}%")

    # ================================================================
    # Sweep: router training configs
    # ================================================================
    configs = [
        # (label, enc_h, emb, epochs, lr, augR, augP, sched, ls)
        # augR = augment router, augP = augment pods (which fast to use)
        ("baseline",              256, 32, 30, 0.003, False, False, False, 0.0),
        ("aug_router",            256, 32, 30, 0.003, True,  False, False, 0.0),
        ("aug_pods",              256, 32, 30, 0.003, False, True,  False, 0.0),
        ("aug_both",              256, 32, 30, 0.003, True,  True,  False, 0.0),
        ("cosine_sched",          256, 32, 50, 0.005, False, False, True,  0.0),
        ("aug+sched",             256, 32, 50, 0.005, True,  False, True,  0.0),
        ("aug+sched+ls",          256, 32, 50, 0.005, True,  False, True,  0.1),
        ("aug+sched+ls+augpods",  256, 32, 50, 0.005, True,  True,  True,  0.1),
        ("aug+sched_emb64",       256, 64, 50, 0.005, True,  False, True,  0.1),
        ("aug+sched_emb128",      256, 128, 50, 0.005, True,  False, True,  0.1),
        # Also try longer
        ("aug+sched_ep80",        256, 32, 80, 0.005, True,  False, True,  0.1),
        ("aug+sched_lr001",       256, 32, 80, 0.001, True,  False, True,  0.1),
    ]

    print(f"\n{'config':>25} {'rout_acc':>9} {'sys_acc':>9} {'oracle':>7}")
    print("-" * 55)

    results_list = []
    best_sys_acc = 0
    best_cfg_name = ""

    for (label, enc_h, emb, epochs, lr, augR, augP, sched, ls) in configs:
        router = ProtoRouter(N_IN, N_PODS, enc_hidden=enc_h, emb=emb).to(device)
        train_router_augmented(router, Xtr, ytr, epochs, lr,
                               use_augment=augR, use_scheduler=sched,
                               label_smoothing=ls)
        fast_use = fast_aug if augP else fast_plain
        orc = oracle_aug if augP else oracle_plain
        r_acc, s_acc = eval_router(router, fast_use, Xte, yte)
        print(f"{label:>25} {r_acc*100:>8.1f}% {s_acc*100:>8.1f}% {orc*100:>6.1f}%")
        results_list.append({
            "config": label, "enc_hidden": enc_h, "emb": emb,
            "epochs": epochs, "lr": lr,
            "augment_router": augR, "augment_pods": augP,
            "scheduler": sched, "label_smoothing": ls,
            "routing_acc": round(r_acc, 4), "system_acc": round(s_acc, 4),
            "oracle_acc": round(orc, 4),
        })
        if s_acc > best_sys_acc:
            best_sys_acc = s_acc
            best_cfg_name = label

    # ================================================================
    # WNIOSEK
    # ================================================================
    baseline_acc = results_list[0]["system_acc"]
    delta = best_sys_acc - baseline_acc
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("WNIOSEK — B2: Augmentacja + Scheduler")
    print("=" * 72)
    print(f"Baseline (no aug, no sched): {baseline_acc*100:.2f}%")
    print(f"Best config ({best_cfg_name}): {best_sys_acc*100:.2f}%")
    print(f"Poprawa: {delta*100:+.2f}pp")
    print(f"Czas: {elapsed:.0f}s")

    # JSON
    results = {
        "experiment": "B2_augmentation_fashion",
        "dataset": "Fashion-MNIST",
        "device": device,
        "oracle_plain": round(oracle_plain, 4),
        "oracle_augmented_pods": round(oracle_aug, 4),
        "best_config": best_cfg_name,
        "best_system_acc": round(best_sys_acc, 4),
        "delta_vs_baseline_pp": round(delta * 100, 2),
        "configs": results_list,
        "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "B2_augmentation_fashion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
