"""
run_A6_expand_fix.py -- Droga A+, blok 2: naprawa expand (continual learning bez replay).

Problem: M.A.R.S. expand ma retencje 95%, ale nauka B = 63.2%.
Przyczyna: zamrozony encoder nie generalizuje na nowe klasy.

Rozwiazanie: EWC (Elastic Weight Consolidation) na encoderze routera.
Po Phase 1 wyliczamy Fisher Information Matrix encodera.
W Phase 2 encoder moze sie zmieniac, ale z kara za odchylenie od starych wag.
Prototypy 0-4 zamrozone (jak dotad).

Testujemy rozne sily EWC (lambda): 0 (brak), 100, 1000, 10000.
lambda=0 to stary expand (baseline), reszta to EWC z rozna sila.

Metryki: retencja A, accuracy B, accuracy all.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A6_expand_fix.py
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
TASK_A = list(range(5))
TASK_B = list(range(5, 10))
HIDDEN = 24


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


def split_by_classes(X, y, classes):
    mask = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    for c in classes:
        mask |= (y == c)
    return X[mask], y[mask]


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


def build_specialist_data(Xtr, ytr, cls, own_ratio=0.7):
    mask = ytr == cls
    own_X, own_y = Xtr[mask], ytr[mask]
    n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
    X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
    y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
    return X_pod, y_pod


def compute_fisher(router, X, y, n_samples=2000):
    """Estymacja diagonalnej Fisher Information Matrix dla encodera."""
    router.train()
    fisher = {}
    for name, param in router.enc.named_parameters():
        fisher[name] = torch.zeros_like(param)

    crit = nn.CrossEntropyLoss()
    indices = torch.randperm(len(X))[:n_samples]
    for i in indices:
        router.zero_grad()
        loss = crit(router(X[i:i+1]), y[i:i+1])
        loss.backward()
        for name, param in router.enc.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

    for name in fisher:
        fisher[name] /= n_samples

    return fisher


def run_expand_ewc(Xtr, ytr, Xte, yte, seed, ewc_lambda=0):
    """M.A.R.S. expand z EWC na encoderze."""
    device = Xtr.device
    torch.manual_seed(seed)

    Xtr_A, ytr_A = split_by_classes(Xtr, ytr, TASK_A)
    Xtr_B, ytr_B = split_by_classes(Xtr, ytr, TASK_B)
    Xte_A, yte_A = split_by_classes(Xte, yte, TASK_A)
    Xte_B, yte_B = split_by_classes(Xte, yte, TASK_B)

    fast = FastPods(N_CLASSES, N_IN, HIDDEN, N_CLASSES).to(device)
    router = ProtoRouter(N_IN, N_CLASSES, enc_hidden=32, emb=16).to(device)

    # Phase 1: Train on Task A
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    for _ in range(15):
        perm = torch.randperm(len(Xtr_A), device=device)
        for s in range(0, len(Xtr_A), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr_A[idx]), ytr_A[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    for c in TASK_A:
        X_pod, y_pod = build_specialist_data(Xtr_A, ytr_A, c)
        W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device)
        with torch.no_grad():
            fast.W1.data[c] = W1; fast.b1.data[c] = b1
            fast.W2.data[c] = W2; fast.b2.data[c] = b2

    # Measure A before Phase 2
    router.eval(); fast.eval()
    with torch.no_grad():
        ids = router.route(Xte_A)
        acc_A_before = (fast.forward_auto(Xte_A, ids).argmax(1) == yte_A).float().mean().item()

    # Save pods A + compute Fisher
    pods_A = {k: fast.__dict__['_parameters'][k].data[:5].clone()
              if k in fast.__dict__.get('_parameters', {}) else None
              for k in []}
    pods_A_W1 = fast.W1.data[:5].clone()
    pods_A_b1 = fast.b1.data[:5].clone()
    pods_A_W2 = fast.W2.data[:5].clone()
    pods_A_b2 = fast.b2.data[:5].clone()

    # Fisher + old encoder weights (for EWC)
    fisher = compute_fisher(router, Xtr_A, ytr_A) if ewc_lambda > 0 else {}
    old_enc_params = {name: param.data.clone()
                      for name, param in router.enc.named_parameters()}

    # Phase 2: Train pods B
    for c in TASK_B:
        X_pod, y_pod = build_specialist_data(Xtr_B, ytr_B, c)
        W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device)
        with torch.no_grad():
            fast.W1.data[c] = W1; fast.b1.data[c] = b1
            fast.W2.data[c] = W2; fast.b2.data[c] = b2

    # Restore pods A
    with torch.no_grad():
        fast.W1.data[:5] = pods_A_W1; fast.b1.data[:5] = pods_A_b1
        fast.W2.data[:5] = pods_A_W2; fast.b2.data[:5] = pods_A_b2

    # Train router on Task B with EWC on encoder
    old_protos = router.protos.data[:5].clone()

    # Encoder: trainable (with EWC penalty if lambda > 0)
    # Protos 0-4: manually frozen after each step
    # Protos 5-9: trainable
    router.train()
    all_params = list(router.parameters())
    opt = torch.optim.Adam(all_params, lr=0.005)

    for _ in range(20):
        perm = torch.randperm(len(Xtr_B), device=device)
        for s in range(0, len(Xtr_B), 512):
            idx = perm[s:s+512]
            ce_loss = crit(router(Xtr_B[idx]), ytr_B[idx])

            # EWC penalty
            ewc_loss = torch.tensor(0.0, device=device)
            if ewc_lambda > 0:
                for name, param in router.enc.named_parameters():
                    if name in fisher:
                        ewc_loss += (fisher[name] * (param - old_enc_params[name]) ** 2).sum()

            loss = ce_loss + ewc_lambda * ewc_loss
            opt.zero_grad(); loss.backward(); opt.step()

            # Freeze old protos
            with torch.no_grad():
                router.protos.data[:5] = old_protos

    # Measure
    router.eval(); fast.eval()
    with torch.no_grad():
        ids_A = router.route(Xte_A)
        acc_A_after = (fast.forward_auto(Xte_A, ids_A).argmax(1) == yte_A).float().mean().item()
        ids_B = router.route(Xte_B)
        acc_B = (fast.forward_auto(Xte_B, ids_B).argmax(1) == yte_B).float().mean().item()
        ids_all = router.route(Xte)
        acc_all = (fast.forward_auto(Xte, ids_all).argmax(1) == yte).float().mean().item()

    # Routing accuracy breakdown
    with torch.no_grad():
        rout_A = (router.route(Xte_A) == yte_A).float().mean().item()
        rout_B = (router.route(Xte_B) == yte_B).float().mean().item()

    return acc_A_before, acc_A_after, acc_B, acc_all, rout_A, rout_B


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA A+ -- blok 2: fix expand (EWC na encoderze)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    print("\nLadowanie MNIST...")
    Xtr, ytr, Xte, yte = load_mnist_tensors(device)

    lambdas = [0, 10, 100, 1000, 10000]
    n_seeds = 3

    print(f"\n{'lambda':>8} {'A przed':>9} {'A po':>7} {'B':>7} "
          f"{'all':>7} {'retencja':>9} {'routA':>7} {'routB':>7}")
    print("-" * 72)

    all_results = []

    for lam in lambdas:
        A_bef, A_aft, Bs, ALLs, rAs, rBs = [], [], [], [], [], []
        for seed in range(n_seeds):
            ab, aa, b, a, ra, rb = run_expand_ewc(Xtr, ytr, Xte, yte, seed, lam)
            A_bef.append(ab); A_aft.append(aa); Bs.append(b)
            ALLs.append(a); rAs.append(ra); rBs.append(rb)

        avg = lambda lst: sum(lst) / len(lst)
        ab_avg = avg(A_bef); aa_avg = avg(A_aft); b_avg = avg(Bs)
        all_avg = avg(ALLs); ra_avg = avg(rAs); rb_avg = avg(rBs)
        ret = aa_avg / ab_avg * 100 if ab_avg > 0 else 0

        print(f"{lam:>8} {ab_avg*100:>8.1f}% {aa_avg*100:>6.1f}% {b_avg*100:>6.1f}% "
              f"{all_avg*100:>6.1f}% {ret:>8.1f}% {ra_avg*100:>6.1f}% {rb_avg*100:>6.1f}%")

        all_results.append({
            "ewc_lambda": lam,
            "A_before": round(ab_avg, 4), "A_after": round(aa_avg, 4),
            "B": round(b_avg, 4), "all": round(all_avg, 4),
            "retencja_pct": round(ret, 1),
            "routing_A": round(ra_avg, 4), "routing_B": round(rb_avg, 4),
        })

    # Wniosek
    baseline = all_results[0]  # lambda=0
    best = max(all_results, key=lambda r: r["all"])

    print("\n--- WNIOSEK ---")
    print(f"Baseline expand (lambda=0): retencja={baseline['retencja_pct']:.1f}%, "
          f"B={baseline['B']*100:.1f}%, all={baseline['all']*100:.1f}%")
    print(f"Najlepszy EWC (lambda={best['ewc_lambda']}): retencja={best['retencja_pct']:.1f}%, "
          f"B={best['B']*100:.1f}%, all={best['all']*100:.1f}%")
    delta_B = best['B'] - baseline['B']
    delta_ret = best['retencja_pct'] - baseline['retencja_pct']
    print(f"Poprawa B: {delta_B*100:+.1f}pp, poprawa retencji: {delta_ret:+.1f}pp")

    # Replay reference
    print(f"\nDla porownania: M.A.R.S. replay = retencja 98.4%, B=95.3%, all=96.4%")

    # JSON
    results = {
        "device": device,
        "n_seeds": n_seeds,
        "scenarios": all_results,
        "replay_reference": {"retencja_pct": 98.4, "B": 0.953, "all": 0.964},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A6_expand_fix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
