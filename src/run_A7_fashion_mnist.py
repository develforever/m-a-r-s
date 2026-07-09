"""
run_A7_fashion_mnist.py -- Droga A+, blok 3: walidacja na Fashion-MNIST.

Cel: potwierdzic ze wyniki M.A.R.S. nie sa specyficzne dla MNIST.
Fashion-MNIST ma identyczny format (28x28, 10 klas) ale inny rozklad
(ubrania zamiast cyfr) -- trudniejszy problem.

Powtarzamy kluczowe pomiary:
  1. System accuracy (specjalisci vs monolit)
  2. MAC savings
  3. Catastrophic forgetting (Split F-MNIST: klasy 0-4 vs 5-9)

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A7_fashion_mnist.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import ProtoRouter
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_IN = 784
N_CLASSES = 10
N_PODS = 10
HIDDEN = 24
TASK_A = list(range(5))
TASK_B = list(range(5, 10))


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


def split_by_classes(X, y, classes):
    mask = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    for c in classes:
        mask |= (y == c)
    return X[mask], y[mask]


def train_router(router, X, y, epochs=15, lr=0.003):
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        for s in range(0, len(X), 512):
            idx = perm[s:s+512]
            loss = crit(router(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def train_specialists(Xtr, ytr, device, hidden=HIDDEN, own_ratio=0.7, epochs=12):
    fast = FastPods(N_PODS, N_IN, hidden, N_PODS).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, N_PODS)
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


def build_specialist_data(Xtr, ytr, cls, own_ratio=0.7):
    mask = ytr == cls
    own_X, own_y = Xtr[mask], ytr[mask]
    n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
    X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
    y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
    return X_pod, y_pod


def train_pod_specialist(X_pod, y_pod, device, hidden=HIDDEN, epochs=12):
    pod = nn.Sequential(
        nn.Linear(N_IN, hidden), nn.ReLU(),
        nn.Linear(hidden, N_CLASSES)
    ).to(device)
    opt = torch.optim.Adam(pod.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X_pod), device=device)
        for s in range(0, len(X_pod), 256):
            idx = perm[s:s+256]
            loss = crit(pod(X_pod[idx]), y_pod[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return (pod[0].weight.data.T.clone(), pod[0].bias.data.clone(),
            pod[2].weight.data.T.clone(), pod[2].bias.data.clone())


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA A+ -- blok 3: Fashion-MNIST (walidacja transferu)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    print("\nLadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    results = {"device": device}

    # ================================================================
    # TEST 1: System accuracy + MAC
    # ================================================================
    print("\n[1/2] System accuracy + MAC")

    # Baseline monolit
    mono = nn.Sequential(
        nn.Linear(N_IN, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, N_CLASSES)
    ).to(device)
    opt = torch.optim.Adam(mono.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(12):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(mono(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mono_acc = (mono(Xte).argmax(1) == yte).float().mean().item()
    mono_mac = 784*256 + 256*128 + 128*10
    print(f"  Monolit: {mono_acc*100:.1f}% acc, {mono_mac:,} MAC")

    # M.A.R.S. specjalisci
    torch.manual_seed(42)
    router = ProtoRouter(N_IN, N_PODS, enc_hidden=32, emb=16).to(device)
    train_router(router, Xtr, ytr)
    fast = train_specialists(Xtr, ytr, device)
    router.eval(); fast.eval()
    with torch.no_grad():
        ids = router.route(Xte)
        rout_acc = (ids == yte).float().mean().item()
        mars_acc = (fast.forward_auto(Xte, ids).argmax(1) == yte).float().mean().item()
        oracle_acc = (fast.forward_auto(Xte, yte).argmax(1) == yte).float().mean().item()
    mars_mac = router.mac_per_sample() + (N_IN * HIDDEN + HIDDEN * N_PODS)
    mac_saving = (1 - mars_mac / mono_mac) * 100
    print(f"  M.A.R.S.: {mars_acc*100:.1f}% acc, {mars_mac:,} MAC "
          f"({mac_saving:.1f}% oszczednosci)")
    print(f"  Router: {rout_acc*100:.1f}% routing acc")
    print(f"  ORACLE: {oracle_acc*100:.1f}%")

    results["accuracy"] = {
        "monolit": {"acc": round(mono_acc, 4), "mac": mono_mac},
        "mars": {"acc": round(mars_acc, 4), "mac": mars_mac,
                 "mac_saving_pct": round(mac_saving, 1),
                 "routing_acc": round(rout_acc, 4)},
        "oracle": round(oracle_acc, 4),
    }

    # ================================================================
    # TEST 2: Catastrophic forgetting (Split F-MNIST)
    # ================================================================
    print("\n[2/2] Catastrophic forgetting (Split Fashion-MNIST)")
    print(f"  Task A: klasy {TASK_A}, Task B: klasy {TASK_B}")
    n_seeds = 3

    Xtr_A, ytr_A = split_by_classes(Xtr, ytr, TASK_A)
    Xtr_B, ytr_B = split_by_classes(Xtr, ytr, TASK_B)
    Xte_A, yte_A = split_by_classes(Xte, yte, TASK_A)
    Xte_B, yte_B = split_by_classes(Xte, yte, TASK_B)

    forget_results = []

    for scenario_name, use_mars in [("Monolith", False), ("M.A.R.S. replay", True)]:
        A_bef, A_aft, Bs_list = [], [], []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            if not use_mars:
                # Monolith: train A then B
                m = nn.Sequential(
                    nn.Linear(N_IN, 256), nn.ReLU(),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, N_CLASSES)
                ).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=0.001)
                for _ in range(12):
                    perm = torch.randperm(len(Xtr_A), device=device)
                    for s in range(0, len(Xtr_A), 512):
                        idx = perm[s:s+512]
                        loss = crit(m(Xtr_A[idx]), ytr_A[idx])
                        opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    ab = (m(Xte_A).argmax(1) == yte_A).float().mean().item()
                for _ in range(12):
                    perm = torch.randperm(len(Xtr_B), device=device)
                    for s in range(0, len(Xtr_B), 512):
                        idx = perm[s:s+512]
                        loss = crit(m(Xtr_B[idx]), ytr_B[idx])
                        opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    aa = (m(Xte_A).argmax(1) == yte_A).float().mean().item()
                    b_acc = (m(Xte_B).argmax(1) == yte_B).float().mean().item()
            else:
                # M.A.R.S.: freeze pods A, train pods B, retrain router A+B
                fp = FastPods(N_CLASSES, N_IN, HIDDEN, N_CLASSES).to(device)
                rt = ProtoRouter(N_IN, N_CLASSES, enc_hidden=32, emb=16).to(device)
                train_router(rt, Xtr_A, ytr_A)
                for c in TASK_A:
                    X_pod, y_pod = build_specialist_data(Xtr_A, ytr_A, c)
                    W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device)
                    with torch.no_grad():
                        fp.W1.data[c]=W1; fp.b1.data[c]=b1
                        fp.W2.data[c]=W2; fp.b2.data[c]=b2
                rt.eval(); fp.eval()
                with torch.no_grad():
                    ab = (fp.forward_auto(Xte_A, rt.route(Xte_A)).argmax(1)==yte_A).float().mean().item()
                saved = (fp.W1.data[:5].clone(), fp.b1.data[:5].clone(),
                         fp.W2.data[:5].clone(), fp.b2.data[:5].clone())
                for c in TASK_B:
                    X_pod, y_pod = build_specialist_data(Xtr_B, ytr_B, c)
                    W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device)
                    with torch.no_grad():
                        fp.W1.data[c]=W1; fp.b1.data[c]=b1
                        fp.W2.data[c]=W2; fp.b2.data[c]=b2
                with torch.no_grad():
                    fp.W1.data[:5]=saved[0]; fp.b1.data[:5]=saved[1]
                    fp.W2.data[:5]=saved[2]; fp.b2.data[:5]=saved[3]
                train_router(rt, Xtr, ytr, epochs=15)
                rt.eval(); fp.eval()
                with torch.no_grad():
                    aa = (fp.forward_auto(Xte_A, rt.route(Xte_A)).argmax(1)==yte_A).float().mean().item()
                    b_acc = (fp.forward_auto(Xte_B, rt.route(Xte_B)).argmax(1)==yte_B).float().mean().item()

            A_bef.append(ab); A_aft.append(aa); Bs_list.append(b_acc)

        avg = lambda lst: sum(lst)/len(lst)
        ab_avg, aa_avg, b_avg = avg(A_bef), avg(A_aft), avg(Bs_list)
        ret = aa_avg / ab_avg * 100 if ab_avg > 0 else 0
        print(f"  {scenario_name:<20}: A_przed={ab_avg*100:.1f}%  "
              f"A_po={aa_avg*100:.1f}%  B={b_avg*100:.1f}%  retencja={ret:.1f}%")
        forget_results.append({
            "name": scenario_name,
            "A_before": round(ab_avg, 4), "A_after": round(aa_avg, 4),
            "B": round(b_avg, 4), "retencja_pct": round(ret, 1),
        })

    results["forgetting"] = forget_results

    # Wniosek
    print("\n--- WNIOSEK ---")
    mono_ret = forget_results[0]["retencja_pct"]
    mars_ret = forget_results[1]["retencja_pct"]
    print(f"Fashion-MNIST: monolit retencja {mono_ret:.1f}%, "
          f"M.A.R.S. retencja {mars_ret:.1f}% (delta +{mars_ret-mono_ret:.1f}pp)")
    print(f"System accuracy: monolit {results['accuracy']['monolit']['acc']*100:.1f}% "
          f"vs M.A.R.S. {results['accuracy']['mars']['acc']*100:.1f}%")
    print(f"MAC saving: {results['accuracy']['mars']['mac_saving_pct']:.1f}%")

    # JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A7_fashion_mnist.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
