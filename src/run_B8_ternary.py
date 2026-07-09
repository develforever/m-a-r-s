"""
run_B8_ternary.py — B8: Ternary kwantyzacja na FastPods + ProtoRouter.

Port mechanizmu z sleep_ternary.py na obecną architekturę (ProtoRouter + FastPods).
Testujemy na obu datasetach (MNIST i F-MNIST).

Mierzymy:
  - Accuracy drop po kwantyzacji (post-training)
  - Sparsity (% wag = 0)
  - Compression ratio (16× theoretical)
  - Throughput (czy ternary jest szybszy na GPU)

Uruchom:
    .venv\\Scripts\\python.exe src\\run_B8_ternary.py
"""

import json
import os
import sys
import time
import copy

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


def load_dataset(name, device):
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
        ])
        train = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, transform=transform)
    Xtr = torch.stack([train[i][0].view(-1) for i in range(len(train))]).to(device)
    ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(device)
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))]).to(device)
    yte = torch.tensor([test[i][1] for i in range(len(test))]).to(device)
    return Xtr, ytr, Xte, yte


def train_system(Xtr, ytr, device, enc_h=256, emb=32, router_epochs=30,
                 pod_epochs=12, own_ratio=0.7):
    """Trenuje pełny system (router + pody)."""
    router = ProtoRouter(N_IN, N_PODS, enc_hidden=enc_h, emb=emb).to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    for _ in range(router_epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    fast = FastPods(N_PODS, N_IN, HIDDEN, N_OUT).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(nn.Linear(N_IN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, N_OUT)).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        mask = ytr == c
        own_X, own_y = Xtr[mask], ytr[mask]
        n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
        X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
        y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        for _ in range(pod_epochs):
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
    return router, fast


def eval_system(router, fast, Xte, yte):
    router.eval()
    fast.eval()
    with torch.no_grad():
        ids = router.route(Xte)
        out = fast.forward_auto(Xte, ids)
        acc = (out.argmax(1) == yte).float().mean().item()
    return acc


def ternary_quantize_tensor(w, threshold_factor=0.7):
    """Kwantyzuj tensor do [-alpha, 0, +alpha]."""
    threshold = threshold_factor * w.abs().mean()
    ternary = torch.zeros_like(w)
    pos_mask = w > threshold
    neg_mask = w < -threshold
    ternary[pos_mask] = 1.0
    ternary[neg_mask] = -1.0

    n_nonzero = pos_mask.sum() + neg_mask.sum()
    alpha = 0.0
    if n_nonzero > 0:
        alpha = (w[pos_mask].sum() - w[neg_mask].sum()) / n_nonzero
    return ternary * alpha.abs(), (ternary == 0).float().mean().item()


def quantize_fastpods(fast, threshold_factor=0.7):
    """Kwantyzuj wagi FastPods do ternary."""
    fast_q = copy.deepcopy(fast)
    total_params = 0
    nonzero_params = 0
    sparsities = []

    with torch.no_grad():
        # W1: [N_PODS, N_IN, HIDDEN]
        for pid in range(fast_q.n_pods):
            w1_q, sp1 = ternary_quantize_tensor(fast_q.W1.data[pid], threshold_factor)
            fast_q.W1.data[pid] = w1_q
            sparsities.append(sp1)
            total_params += fast_q.W1.data[pid].numel()
            nonzero_params += (fast_q.W1.data[pid] != 0).sum().item()

            w2_q, sp2 = ternary_quantize_tensor(fast_q.W2.data[pid], threshold_factor)
            fast_q.W2.data[pid] = w2_q
            sparsities.append(sp2)
            total_params += fast_q.W2.data[pid].numel()
            nonzero_params += (fast_q.W2.data[pid] != 0).sum().item()

    avg_sparsity = sum(sparsities) / len(sparsities)
    return fast_q, {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity": avg_sparsity,
        "compression_ratio": 32 / 2,  # FP32 → 2-bit ternary
    }


def quantize_router(router, threshold_factor=0.7):
    """Kwantyzuj wagi encodera routera do ternary."""
    router_q = copy.deepcopy(router)
    with torch.no_grad():
        for name, param in router_q.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                q, _ = ternary_quantize_tensor(param.data, threshold_factor)
                param.data.copy_(q)
    return router_q


def throughput_test(router, fast, device, batch_size=4096, n_runs=50):
    """Zmierz throughput systemu."""
    x = torch.randn(batch_size, N_IN, device=device)
    router.eval()
    fast.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            ids = router.route(x)
            fast.forward_auto(x, ids)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            ids = router.route(x)
            fast.forward_auto(x, ids)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return batch_size * n_runs / elapsed


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("B8 — Ternary Kwantyzacja (FastPods + ProtoRouter)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    t0 = time.perf_counter()
    all_results = {}

    for dataset_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\n{'='*40}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*40}")

        Xtr, ytr, Xte, yte = load_dataset(dataset_name, device)
        router, fast = train_system(Xtr, ytr, device)

        acc_full = eval_system(router, fast, Xte, yte)
        print(f"  Full precision: {acc_full*100:.2f}%")

        # Kwantyzuj TYLKO pody
        fast_q, pod_metrics = quantize_fastpods(fast)
        acc_pods_q = eval_system(router, fast_q, Xte, yte)
        print(f"  Ternary pods:   {acc_pods_q*100:.2f}% (drop {(acc_full-acc_pods_q)*100:.2f}pp)")
        print(f"    sparsity: {pod_metrics['sparsity']*100:.1f}%")

        # Kwantyzuj pody + router
        router_q = quantize_router(router)
        acc_both_q = eval_system(router_q, fast_q, Xte, yte)
        print(f"  Ternary all:    {acc_both_q*100:.2f}% (drop {(acc_full-acc_both_q)*100:.2f}pp)")

        # Sweep threshold
        print(f"\n  Threshold sweep (pods only):")
        print(f"  {'threshold':>10} {'acc':>7} {'drop':>7} {'sparsity':>9}")
        best_thresh_acc = 0
        best_thresh = 0.7
        for tf in [0.5, 0.6, 0.7, 0.8, 0.9]:
            fq, m = quantize_fastpods(fast, threshold_factor=tf)
            a = eval_system(router, fq, Xte, yte)
            drop = (acc_full - a) * 100
            print(f"  {tf:>10.1f} {a*100:>6.2f}% {drop:>+6.2f}pp {m['sparsity']*100:>8.1f}%")
            if a > best_thresh_acc:
                best_thresh_acc = a
                best_thresh = tf

        # Throughput
        print(f"\n  Throughput (batch=4096, 50 runs):")
        tp_full = throughput_test(router, fast, device)
        tp_q = throughput_test(router, fast_q, device)
        print(f"    Full precision: {tp_full:,.0f} samples/s")
        print(f"    Ternary pods:   {tp_q:,.0f} samples/s")
        print(f"    Speedup:        {tp_q/tp_full:.2f}×")

        all_results[dataset_name] = {
            "acc_full": round(acc_full, 4),
            "acc_ternary_pods": round(acc_pods_q, 4),
            "acc_ternary_all": round(acc_both_q, 4),
            "drop_pods_pp": round((acc_full - acc_pods_q) * 100, 2),
            "drop_all_pp": round((acc_full - acc_both_q) * 100, 2),
            "pod_sparsity": round(pod_metrics['sparsity'], 4),
            "compression_ratio": pod_metrics['compression_ratio'],
            "throughput_full": round(tp_full),
            "throughput_ternary": round(tp_q),
            "speedup": round(tp_q / tp_full, 2),
            "best_threshold": best_thresh,
        }

    # ================================================================
    # WNIOSEK
    # ================================================================
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("WNIOSEK — B8: Ternary Kwantyzacja")
    print("=" * 72)
    for ds, r in all_results.items():
        print(f"\n  {ds}:")
        print(f"    Full: {r['acc_full']*100:.2f}% → Ternary pods: {r['acc_ternary_pods']*100:.2f}% "
              f"(drop {r['drop_pods_pp']:.2f}pp)")
        print(f"    Sparsity: {r['pod_sparsity']*100:.1f}%, Compression: {r['compression_ratio']:.0f}×")
        print(f"    Throughput: {r['speedup']:.2f}× speedup")
    print(f"\n  Czas: {elapsed:.0f}s")

    # JSON
    results = {
        "experiment": "B8_ternary_quantization",
        "device": device,
        "datasets": all_results,
        "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "B8_ternary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
