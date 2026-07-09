"""
run_A4_forgetting.py -- Droga A, krok A4: catastrophic forgetting na MNIST.

Split-MNIST: Task A = digits 0-4, Task B = digits 5-9.
Pytanie: ile wiedzy o Task A zostaje po nauce Task B?

Trzy scenariusze:
  1. Monolith   -- train A, then B (no protection) -> forgetting baseline
  2. M.A.R.S. (replay)  -- freeze pods A, train pods B, retrain router on A+B
                           Upper bound: router widzi stare dane (cheap replay)
  3. M.A.R.S. (expand)  -- freeze pods A + encoder, train pods B + new protos
                           Realistyczny: ZERO starych danych w Phase 2

Metryki:
  - retencja_A = accuracy_A_after / accuracy_A_before (ile zostalo)
  - accuracy_B = jak dobrze nauczyl sie B
  - accuracy_all = laczone na digits 0-9

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A4_forgetting.py
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
N_IN = 784
N_CLASSES = 10
TASK_A_CLASSES = list(range(5))   # digits 0-4
TASK_B_CLASSES = list(range(5, 10))  # digits 5-9
N_SEEDS = 3


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
    """Wydziela probki nalezace do podanych klas."""
    mask = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    for c in classes:
        mask |= (y == c)
    return X[mask], y[mask]


def train_model(model, X, y, epochs=12, lr=0.001, bs=512):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        for s in range(0, len(X), bs):
            idx = perm[s:s+bs]
            loss = crit(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def eval_acc(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


# =====================================================================
# SCENARIUSZ 1: MONOLITH (catastrophic forgetting baseline)
# =====================================================================
def run_monolith(Xtr, ytr, Xte, yte, seed):
    """Train on A, then B. Classic forgetting."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(N_IN, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, N_CLASSES)
    ).to(Xtr.device)

    Xtr_A, ytr_A = split_by_classes(Xtr, ytr, TASK_A_CLASSES)
    Xtr_B, ytr_B = split_by_classes(Xtr, ytr, TASK_B_CLASSES)
    Xte_A, yte_A = split_by_classes(Xte, yte, TASK_A_CLASSES)
    Xte_B, yte_B = split_by_classes(Xte, yte, TASK_B_CLASSES)

    # Phase 1: Task A
    train_model(model, Xtr_A, ytr_A, epochs=12)
    acc_A_before = eval_acc(model, Xte_A, yte_A)

    # Phase 2: Task B (no protection)
    train_model(model, Xtr_B, ytr_B, epochs=12)
    acc_A_after = eval_acc(model, Xte_A, yte_A)
    acc_B = eval_acc(model, Xte_B, yte_B)
    acc_all = eval_acc(model, Xte, yte)

    return acc_A_before, acc_A_after, acc_B, acc_all


# =====================================================================
# SCENARIUSZ 2: M.A.R.S. REPLAY (freeze pods, retrain router on A+B)
# =====================================================================
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


def train_pod_specialist(X_pod, y_pod, device, hidden=24, epochs=12):
    """Trenuje pojedynczego specjaliste i zwraca wagi (W1,b1,W2,b2)."""
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
    """Dane specjalisty: 70% swoich, 30% cudzych."""
    mask = ytr == cls
    own_X, own_y = Xtr[mask], ytr[mask]
    n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
    other_mask = ~mask
    X_pod = torch.cat([own_X, Xtr[other_mask][:n_other]])
    y_pod = torch.cat([own_y, ytr[other_mask][:n_other]])
    return X_pod, y_pod


def run_mars_replay(Xtr, ytr, Xte, yte, seed, hidden=24):
    """Freeze pods A, train pods B, retrain router on A+B."""
    device = Xtr.device
    torch.manual_seed(seed)

    Xtr_A, ytr_A = split_by_classes(Xtr, ytr, TASK_A_CLASSES)
    Xtr_B, ytr_B = split_by_classes(Xtr, ytr, TASK_B_CLASSES)
    Xte_A, yte_A = split_by_classes(Xte, yte, TASK_A_CLASSES)
    Xte_B, yte_B = split_by_classes(Xte, yte, TASK_B_CLASSES)

    fast = FastPods(N_CLASSES, N_IN, hidden, N_CLASSES).to(device)
    router = ProtoRouter(N_IN, N_CLASSES, enc_hidden=32, emb=16).to(device)

    # Phase 1: Train on Task A (digits 0-4)
    # Train router on A labels
    train_router(router, Xtr_A, ytr_A)

    # Train pods 0-4 as specialists
    for c in TASK_A_CLASSES:
        X_pod, y_pod = build_specialist_data(Xtr_A, ytr_A, c)
        W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device, hidden)
        with torch.no_grad():
            fast.W1.data[c] = W1; fast.b1.data[c] = b1
            fast.W2.data[c] = W2; fast.b2.data[c] = b2

    # Measure A before Phase 2
    router.eval(); fast.eval()
    with torch.no_grad():
        ids = router.route(Xte_A)
        acc_A_before = (fast.forward_auto(Xte_A, ids).argmax(1) == yte_A).float().mean().item()

    # Save pods A weights (freeze = just don't train them)
    pods_A_W1 = fast.W1.data[:5].clone()
    pods_A_b1 = fast.b1.data[:5].clone()
    pods_A_W2 = fast.W2.data[:5].clone()
    pods_A_b2 = fast.b2.data[:5].clone()

    # Phase 2: Train pods 5-9 on Task B (pods 0-4 FROZEN)
    for c in TASK_B_CLASSES:
        X_pod, y_pod = build_specialist_data(Xtr_B, ytr_B, c)
        W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device, hidden)
        with torch.no_grad():
            fast.W1.data[c] = W1; fast.b1.data[c] = b1
            fast.W2.data[c] = W2; fast.b2.data[c] = b2

    # Restore pods A (in case FastPods was somehow modified)
    with torch.no_grad():
        fast.W1.data[:5] = pods_A_W1
        fast.b1.data[:5] = pods_A_b1
        fast.W2.data[:5] = pods_A_W2
        fast.b2.data[:5] = pods_A_b2

    # Retrain router on A+B (REPLAY: has access to old data for routing)
    train_router(router, Xtr, ytr, epochs=15)

    # Measure
    router.eval(); fast.eval()
    with torch.no_grad():
        ids_A = router.route(Xte_A)
        acc_A_after = (fast.forward_auto(Xte_A, ids_A).argmax(1) == yte_A).float().mean().item()
        ids_B = router.route(Xte_B)
        acc_B = (fast.forward_auto(Xte_B, ids_B).argmax(1) == yte_B).float().mean().item()
        ids_all = router.route(Xte)
        acc_all = (fast.forward_auto(Xte, ids_all).argmax(1) == yte).float().mean().item()

    return acc_A_before, acc_A_after, acc_B, acc_all


# =====================================================================
# SCENARIUSZ 3: M.A.R.S. EXPAND (freeze pods A + encoder, no old data)
# =====================================================================
def run_mars_expand(Xtr, ytr, Xte, yte, seed, hidden=24):
    """Freeze pods A + encoder. Train pods B + new protos on B data ONLY."""
    device = Xtr.device
    torch.manual_seed(seed)

    Xtr_A, ytr_A = split_by_classes(Xtr, ytr, TASK_A_CLASSES)
    Xtr_B, ytr_B = split_by_classes(Xtr, ytr, TASK_B_CLASSES)
    Xte_A, yte_A = split_by_classes(Xte, yte, TASK_A_CLASSES)
    Xte_B, yte_B = split_by_classes(Xte, yte, TASK_B_CLASSES)

    fast = FastPods(N_CLASSES, N_IN, hidden, N_CLASSES).to(device)
    router = ProtoRouter(N_IN, N_CLASSES, enc_hidden=32, emb=16).to(device)

    # Phase 1: Train on Task A (digits 0-4) -- identical to replay
    train_router(router, Xtr_A, ytr_A)
    for c in TASK_A_CLASSES:
        X_pod, y_pod = build_specialist_data(Xtr_A, ytr_A, c)
        W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device, hidden)
        with torch.no_grad():
            fast.W1.data[c] = W1; fast.b1.data[c] = b1
            fast.W2.data[c] = W2; fast.b2.data[c] = b2

    # Measure A before Phase 2
    router.eval(); fast.eval()
    with torch.no_grad():
        ids = router.route(Xte_A)
        acc_A_before = (fast.forward_auto(Xte_A, ids).argmax(1) == yte_A).float().mean().item()

    # Save pods A weights
    pods_A_W1 = fast.W1.data[:5].clone()
    pods_A_b1 = fast.b1.data[:5].clone()
    pods_A_W2 = fast.W2.data[:5].clone()
    pods_A_b2 = fast.b2.data[:5].clone()

    # Phase 2: Train pods B (pods A FROZEN)
    for c in TASK_B_CLASSES:
        X_pod, y_pod = build_specialist_data(Xtr_B, ytr_B, c)
        W1, b1, W2, b2 = train_pod_specialist(X_pod, y_pod, device, hidden)
        with torch.no_grad():
            fast.W1.data[c] = W1; fast.b1.data[c] = b1
            fast.W2.data[c] = W2; fast.b2.data[c] = b2

    # Restore pods A
    with torch.no_grad():
        fast.W1.data[:5] = pods_A_W1; fast.b1.data[:5] = pods_A_b1
        fast.W2.data[:5] = pods_A_W2; fast.b2.data[:5] = pods_A_b2

    # EXPAND: freeze encoder, train ONLY prototypes 5-9 on Task B data
    # Freeze encoder
    for p in router.enc.parameters():
        p.requires_grad = False
    # Freeze old prototypes by training with mask
    old_protos = router.protos.data[:5].clone()

    opt = torch.optim.Adam([router.protos], lr=0.01)
    crit = nn.CrossEntropyLoss()
    for _ in range(20):
        perm = torch.randperm(len(Xtr_B), device=device)
        for s in range(0, len(Xtr_B), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr_B[idx]), ytr_B[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            # Restore old prototypes after each step (manual freeze)
            with torch.no_grad():
                router.protos.data[:5] = old_protos

    # Unfreeze encoder for eval (doesn't matter, no training)
    for p in router.enc.parameters():
        p.requires_grad = True

    # Measure
    router.eval(); fast.eval()
    with torch.no_grad():
        ids_A = router.route(Xte_A)
        acc_A_after = (fast.forward_auto(Xte_A, ids_A).argmax(1) == yte_A).float().mean().item()
        ids_B = router.route(Xte_B)
        acc_B = (fast.forward_auto(Xte_B, ids_B).argmax(1) == yte_B).float().mean().item()
        ids_all = router.route(Xte)
        acc_all = (fast.forward_auto(Xte, ids_all).argmax(1) == yte).float().mean().item()

    return acc_A_before, acc_A_after, acc_B, acc_all


# =====================================================================
# MAIN
# =====================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA A -- krok A4: catastrophic forgetting (Split-MNIST)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)
    print(f"Task A: digits {TASK_A_CLASSES}")
    print(f"Task B: digits {TASK_B_CLASSES}")
    print(f"Sekwencja: train A -> train B -> mierz retencje A")
    print(f"Usrednione po {N_SEEDS} seedach.\n")

    print("Ladowanie MNIST...")
    Xtr, ytr, Xte, yte = load_mnist_tensors(device)

    scenarios = [
        ("Monolith (baseline)", run_monolith),
        ("M.A.R.S. replay (router A+B)", run_mars_replay),
        ("M.A.R.S. expand (bez starych danych)", run_mars_expand),
    ]

    all_results = []

    for name, fn in scenarios:
        print(f"\n[{name}]")
        A_befores, A_afters, Bs, ALLs = [], [], [], []
        for seed in range(N_SEEDS):
            ab, aa, b, a = fn(Xtr, ytr, Xte, yte, seed)
            A_befores.append(ab); A_afters.append(aa)
            Bs.append(b); ALLs.append(a)
            print(f"  seed {seed}: A_before={ab*100:.1f}%  "
                  f"A_after={aa*100:.1f}%  B={b*100:.1f}%  all={a*100:.1f}%")

        avg = lambda lst: sum(lst) / len(lst)
        ab_avg = avg(A_befores)
        aa_avg = avg(A_afters)
        b_avg = avg(Bs)
        all_avg = avg(ALLs)
        retencja = aa_avg / ab_avg * 100 if ab_avg > 0 else 0

        print(f"  SREDNIA: A_before={ab_avg*100:.1f}%  A_after={aa_avg*100:.1f}%  "
              f"B={b_avg*100:.1f}%  all={all_avg*100:.1f}%  "
              f"retencja={retencja:.1f}%")

        all_results.append({
            "name": name,
            "A_before": round(ab_avg, 4),
            "A_after": round(aa_avg, 4),
            "B": round(b_avg, 4),
            "all": round(all_avg, 4),
            "retencja_pct": round(retencja, 1),
        })

    # Tabela zbiorcza
    print("\n" + "=" * 72)
    print("TABELA ZBIORCZA")
    print("=" * 72)
    print(f"{'Scenariusz':<38} {'A przed':>8} {'A po':>8} "
          f"{'B':>8} {'retencja':>10}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['name']:<38} {r['A_before']*100:>7.1f}% "
              f"{r['A_after']*100:>7.1f}% {r['B']*100:>7.1f}% "
              f"{r['retencja_pct']:>9.1f}%")

    # Wniosek
    mono = all_results[0]
    best_mars = max(all_results[1:], key=lambda r: r["retencja_pct"])
    delta = best_mars["retencja_pct"] - mono["retencja_pct"]

    print("\n--- WNIOSEK ---")
    print(f"Monolith retencja: {mono['retencja_pct']:.1f}%")
    print(f"Najlepsza M.A.R.S. retencja: {best_mars['retencja_pct']:.1f}% "
          f"({best_mars['name']})")
    print(f"Delta: +{delta:.1f} pkt proc.")
    if delta > 20:
        print("M.A.R.S. ZNACZACO chroni stara wiedze przy nauce nowej.")
    elif delta > 5:
        print("M.A.R.S. umiarkowanie chroni stara wiedze.")
    else:
        print("Roznica niewielka -- modularnosc nie daje tu duzej przewagi.")

    # JSON
    results = {
        "device": device,
        "task_A": TASK_A_CLASSES,
        "task_B": TASK_B_CLASSES,
        "n_seeds": N_SEEDS,
        "scenarios": all_results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A4_forgetting.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
