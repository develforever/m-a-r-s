"""
run_C1_adaptive_compute.py -- Droga C, krok 1: Adaptive Compute.

Laczy dwa pominiete pomysly:
  - CBAR (Confidence-Based Adaptive Routing): selektywny top-2 dla niepewnych
  - Early Exit: pominiecie poda dla pewnych probek

3-tier system:
  confidence > theta_high  ->  EARLY EXIT  (predykcja routera, 0 podow)
  theta_low < conf < theta_high  ->  TOP-1  (normalny routing, 1 pod)
  confidence < theta_low  ->  TOP-2  (agregacja 2 podow)

Testujemy na MNIST i Fashion-MNIST, rozne progi, mierzymy:
  - accuracy vs baseline top-1
  - sredni koszt MAC per sample
  - % probek w kazdym tierze

Uruchom:
    .venv\\Scripts\\python.exe src\\run_C1_adaptive_compute.py
"""

import json, os, sys
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import ProtoRouter
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_IN, N_CLASSES, HIDDEN = 784, 10, 24


def load_dataset(name, device):
    if name == "MNIST":
        ds_cls = torchvision.datasets.MNIST
        mean, std = 0.1307, 0.3081
    else:
        ds_cls = torchvision.datasets.FashionMNIST
        mean, std = 0.2860, 0.3530
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    train = ds_cls(root=DATA_DIR, train=True, download=True, transform=transform)
    test = ds_cls(root=DATA_DIR, train=False, download=True, transform=transform)
    Xtr = torch.stack([train[i][0].view(-1) for i in range(len(train))]).to(device)
    ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(device)
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))]).to(device)
    yte = torch.tensor([test[i][1] for i in range(len(test))]).to(device)
    return Xtr, ytr, Xte, yte


def train_system(Xtr, ytr, device, enc_hidden=256, emb=64, router_epochs=50, router_lr=0.001):
    """Trenuje router + specjalistow. Zwraca (router, fast)."""
    router = ProtoRouter(N_IN, N_CLASSES, enc_hidden=enc_hidden, emb=emb).to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=router_lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(router_epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    fast = FastPods(N_CLASSES, N_IN, HIDDEN, N_CLASSES).to(device)
    crit2 = nn.CrossEntropyLoss()
    for c in range(N_CLASSES):
        pod = nn.Sequential(nn.Linear(N_IN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, N_CLASSES)).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        mask = ytr == c
        own_X, own_y = Xtr[mask], ytr[mask]
        n_other = int(len(own_X) * 0.3 / 0.7)
        X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
        y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        for _ in range(12):
            perm = torch.randperm(len(X_pod), device=device)
            for s in range(0, len(X_pod), 256):
                idx = perm[s:s+256]
                loss = crit2(pod(X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fast.W1.data[c] = pod[0].weight.data.T
            fast.b1.data[c] = pod[0].bias.data
            fast.W2.data[c] = pod[2].weight.data.T
            fast.b2.data[c] = pod[2].bias.data
    return router, fast


def adaptive_inference(router, fast, X, y, theta_high, theta_low):
    """3-tier adaptive compute inference."""
    router.eval(); fast.eval()
    with torch.no_grad():
        logits = router(X)
        probs = torch.softmax(logits, dim=1)
        conf, pred_class = probs.max(dim=1)

        # Tier masks
        early_mask = conf >= theta_high
        top1_mask = (conf >= theta_low) & (conf < theta_high)
        top2_mask = conf < theta_low

        predictions = torch.zeros(len(X), dtype=torch.long, device=X.device)

        # Tier 1: Early exit (router prediction only)
        if early_mask.any():
            predictions[early_mask] = pred_class[early_mask]

        # Tier 2: Top-1 (normal routing)
        if top1_mask.any():
            ids = pred_class[top1_mask]
            out = fast.forward_auto(X[top1_mask], ids)
            predictions[top1_mask] = out.argmax(1)

        # Tier 3: Top-2 (aggregate 2 pods)
        if top2_mask.any():
            sorted_probs, sorted_ids = probs[top2_mask].topk(2, dim=1)
            p0 = sorted_ids[:, 0]
            p1 = sorted_ids[:, 1]
            w0 = sorted_probs[:, 0].unsqueeze(1)
            w1 = sorted_probs[:, 1].unsqueeze(1)
            out0 = fast.forward_auto(X[top2_mask], p0)
            out1 = fast.forward_auto(X[top2_mask], p1)
            # Confidence-weighted aggregation
            combined = w0 * out0 + w1 * out1
            predictions[top2_mask] = combined.argmax(1)

        acc = (predictions == y).float().mean().item()
        n = len(X)
        n_early = early_mask.sum().item()
        n_top1 = top1_mask.sum().item()
        n_top2 = top2_mask.sum().item()

        router_mac = router.mac_per_sample()
        pod_mac = N_IN * HIDDEN + HIDDEN * N_CLASSES

        # Average MAC: early=router only, top1=router+1pod, top2=router+2pods
        avg_mac = router_mac + pod_mac * (n_top1 + 2 * n_top2) / n

    return {
        "acc": acc,
        "avg_mac": round(avg_mac),
        "pct_early": round(n_early / n * 100, 1),
        "pct_top1": round(n_top1 / n * 100, 1),
        "pct_top2": round(n_top2 / n * 100, 1),
        "n_early": n_early,
        "n_top1": n_top1,
        "n_top2": n_top2,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA C -- krok C1: Adaptive Compute (CBAR + Early Exit)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    # Progi do testowania
    theta_configs = [
        # (theta_high, theta_low, name)
        (1.01, 0.0, "baseline (top-1 only)"),    # no early exit, no top-2
        (0.999, 0.0, "early exit >99.9%"),
        (0.99, 0.0, "early exit >99%"),
        (0.95, 0.0, "early exit >95%"),
        (1.01, 0.5, "top-2 <50%"),
        (1.01, 0.7, "top-2 <70%"),
        (1.01, 0.9, "top-2 <90%"),
        (0.99, 0.5, "full adaptive (EE>99%, T2<50%)"),
        (0.99, 0.7, "full adaptive (EE>99%, T2<70%)"),
        (0.95, 0.5, "full adaptive (EE>95%, T2<50%)"),
        (0.95, 0.7, "full adaptive (EE>95%, T2<70%)"),
    ]

    all_results = {}

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\n{'='*40}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*40}")

        Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

        enc_h = 256 if ds_name == "MNIST" else 256
        emb = 64 if ds_name == "MNIST" else 32
        ep = 50 if ds_name == "MNIST" else 30

        print("Trening systemu...")
        router, fast = train_system(Xtr, ytr, device, enc_hidden=enc_h, emb=emb,
                                     router_epochs=ep)

        router_mac = router.mac_per_sample()
        pod_mac = N_IN * HIDDEN + HIDDEN * N_CLASSES
        full_mac = router_mac + pod_mac

        print(f"Router MAC: {router_mac:,}, Pod MAC: {pod_mac:,}, "
              f"Full MAC: {full_mac:,}")

        print(f"\n{'config':<36} {'acc':>7} {'avgMAC':>8} "
              f"{'early%':>7} {'top1%':>7} {'top2%':>7} {'save%':>7}")
        print("-" * 82)

        ds_results = []
        for th, tl, name in theta_configs:
            r = adaptive_inference(router, fast, Xte, yte, th, tl)
            mac_saving = (1 - r["avg_mac"] / full_mac) * 100
            print(f"{name:<36} {r['acc']*100:>6.2f}% {r['avg_mac']:>8,} "
                  f"{r['pct_early']:>6.1f}% {r['pct_top1']:>6.1f}% "
                  f"{r['pct_top2']:>6.1f}% {mac_saving:>6.1f}%")
            ds_results.append({
                "config": name,
                "theta_high": th, "theta_low": tl,
                "acc": round(r["acc"], 4),
                "avg_mac": r["avg_mac"],
                "mac_saving_pct": round(mac_saving, 1),
                **r,
            })

        # Najlepszy z poprawiona accuracy
        baseline = ds_results[0]
        better = [r for r in ds_results if r["acc"] >= baseline["acc"]]
        if len(better) > 1:
            best = max(better, key=lambda r: r["mac_saving_pct"])
            print(f"\n  Najlepszy (acc >= baseline): {best['config']}")
            print(f"  acc={best['acc']*100:.2f}% "
                  f"(baseline={baseline['acc']*100:.2f}%), "
                  f"MAC saving={best['mac_saving_pct']:.1f}%")
        else:
            best_acc = max(ds_results, key=lambda r: r["acc"])
            print(f"\n  Najlepszy accuracy: {best_acc['config']} "
                  f"= {best_acc['acc']*100:.2f}%")

        all_results[ds_name] = ds_results

    # JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "C1_adaptive_compute.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
