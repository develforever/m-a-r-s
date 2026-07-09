"""
run_A3_benchmark.py — Droga A, krok A3: pełny pomiar zbiorczy.

Mierzy na MNIST trzy systemy na wszystkich wymiarach naraz:
  1. Baseline monolit (784→256→128→10)
  2. Redundantny M.A.R.S. (router + pody hidden=64, wszystkie dane)
  3. Specjaliści M.A.R.S. (router 16D + pody hidden=24, dane 70/30)

Metryki: accuracy, total MAC, throughput (samples/s), oszczędność MAC.

Throughput liczony wektoryzowanym forward (FastPods.forward_auto) — uczciwy
pomiar bez narzutu pętli. Pytanie A3: czy 80.9% mniej MAC = przewaga CZASOWA?
Lekcja z Etapu B: na małych podach niekoniecznie — monolit to gładki matmul.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A3_benchmark.py
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from mars_specialists import NarrowPod
from routers_v2 import ProtoRouter
from mars_fast_forward import FastPods

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


def bench_throughput(fn, x, device, n_warmup=15, n_runs=100):
    for _ in range(n_warmup):
        fn(x)
    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_runs):
            fn(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    return x.shape[0] / ((elapsed_ms / n_runs) / 1000)


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


def train_pods_into_fastpods(Xtr, ytr, device, hidden, own_ratio, epochs=12):
    """Trenuje pody osobno, potem przenosi wagi do FastPods (wektoryzacja)."""
    fast = FastPods(N_PODS, 784, hidden, N_PODS).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = NarrowPod(784, N_PODS, hidden=hidden).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        if own_ratio < 1.0:
            mask = ytr == c
            own_X, own_y = Xtr[mask], ytr[mask]
            n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
            X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
            y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        else:
            X_pod, y_pod = Xtr, ytr
        for _ in range(epochs):
            perm = torch.randperm(len(X_pod), device=device)
            for s in range(0, len(X_pod), 256):
                idx = perm[s:s+256]
                loss = crit(pod(X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        # przenieś wagi do FastPods (net[0]=Linear1, net[2]=Linear2)
        with torch.no_grad():
            fast.W1.data[c] = pod.net[0].weight.data.T
            fast.b1.data[c] = pod.net[0].bias.data
            fast.W2.data[c] = pod.net[2].weight.data.T
            fast.b2.data[c] = pod.net[2].bias.data
    return fast


def eval_acc(router, fast, Xte, yte, device):
    with torch.no_grad():
        ids = router.route(Xte)
        out = fast.forward_auto(Xte, ids)
        return (out.argmax(1) == yte).float().mean().item()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("DROGA A — krok A3: pełny pomiar zbiorczy (MNIST)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 70)

    print("\nŁadowanie MNIST...")
    Xtr, ytr, Xte, yte = load_mnist_tensors(device)
    baseline_mac = 234752

    rows = []

    # 1. Baseline monolit
    print("\n[1/3] Baseline monolit...")
    base = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    opt = torch.optim.Adam(base.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(8):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(base(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        base_acc = (base(Xte).argmax(1) == yte).float().mean().item()
    base_tput = bench_throughput(lambda x: base(x), Xte, device)
    rows.append(("Baseline monolit", base_acc, baseline_mac, base_tput, 0.0))

    # 2. Redundantny
    print("[2/3] Redundantny M.A.R.S. (hidden=64)...")
    r_red = ProtoRouter(784, N_PODS, enc_hidden=16, emb=8).to(device)
    train_router(r_red, Xtr, ytr)
    fast_red = train_pods_into_fastpods(Xtr, ytr, device, hidden=64, own_ratio=1.0)
    red_acc = eval_acc(r_red, fast_red, Xte, yte, device)
    red_mac = r_red.mac_per_sample() + (784 * 64 + 64 * 10)
    red_tput = bench_throughput(
        lambda x: fast_red.forward_auto(x, r_red.route(x)), Xte, device)
    rows.append(("Redundantny (h=64)", red_acc, red_mac, red_tput,
                 (1 - red_mac / baseline_mac) * 100))

    # 3. Specjaliści
    print("[3/3] Specjaliści M.A.R.S. (router 16D, hidden=24)...")
    r_spec = ProtoRouter(784, N_PODS, enc_hidden=32, emb=16).to(device)
    train_router(r_spec, Xtr, ytr)
    fast_spec = train_pods_into_fastpods(Xtr, ytr, device, hidden=24, own_ratio=0.7)
    spec_acc = eval_acc(r_spec, fast_spec, Xte, yte, device)
    spec_mac = r_spec.mac_per_sample() + (784 * 24 + 24 * 10)
    spec_tput = bench_throughput(
        lambda x: fast_spec.forward_auto(x, r_spec.route(x)), Xte, device)
    rows.append(("Specjaliści (h=24)", spec_acc, spec_mac, spec_tput,
                 (1 - spec_mac / baseline_mac) * 100))

    # Tabela
    print("\n" + "=" * 70)
    print("TABELA ZBIORCZA")
    print("=" * 70)
    print(f"{'System':<22} {'acc':>7} {'MAC':>9} {'samples/s':>12} {'oszczędność':>12}")
    print("-" * 66)
    for name, acc, mac, tput, sav in rows:
        sav_str = f"{sav:.1f}%" if sav > 0 else "—"
        print(f"{name:<22} {acc*100:>6.1f}% {mac:>9,} {tput:>12,.0f} {sav_str:>12}")

    base_tput_val = rows[0][3]
    spec_tput_val = rows[2][3]
    print("\n--- WNIOSEK ---")
    print(f"Specjaliści: {rows[2][1]*100:.1f}% accuracy, {rows[2][4]:.1f}% mniej MAC")
    print(f"Throughput specjalistów vs baseline: {spec_tput_val/base_tput_val:.2f}×")
    if spec_tput_val >= base_tput_val:
        print("Specjaliści są też SZYBSI — oszczędność MAC przekłada się na czas.")
    else:
        print("Specjaliści mają mniej MAC, ale NIE są szybsi (narzut routingu na")
        print("małych podach, lekcja z Etapu B). Przewaga czasowa wymaga większych podów.")
        print("MAC oszczędność (80.9%) pozostaje realna i ważna dla energii.")

    results = {
        "device": device,
        "systems": [
            {"name": n, "accuracy": round(a, 4), "total_mac": m,
             "throughput_sps": round(t), "mac_saving_pct": round(s, 1)}
            for n, a, m, t, s in rows
        ],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A3_benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
