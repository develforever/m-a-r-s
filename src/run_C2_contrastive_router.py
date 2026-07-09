"""
run_C2_contrastive_router.py -- Droga C, krok 2: Contrastive Learning dla routera.

Hipoteza: ProtoRouter trenowany cross-entropy optymalizuje granice decyzyjne,
nie klastry. SupCon (Supervised Contrastive Loss) wymusza separacje klastrow
-> lepsze routing accuracy, szczegolnie na Fashion-MNIST.

Testujemy:
  - CE baseline (obecny router)
  - SupCon pre-training + CE fine-tuning
  - SupCon + ProtoRouter (contrastive embeddings -> prototypy)

Na MNIST i Fashion-MNIST.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_C2_contrastive_router.py
"""

import json, os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def train_specialists(Xtr, ytr, device):
    fast = FastPods(N_CLASSES, N_IN, HIDDEN, N_CLASSES).to(device)
    crit = nn.CrossEntropyLoss()
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
                loss = crit(pod(X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fast.W1.data[c] = pod[0].weight.data.T
            fast.b1.data[c] = pod[0].bias.data
            fast.W2.data[c] = pod[2].weight.data.T
            fast.b2.data[c] = pod[2].bias.data
    return fast


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D], L2-normalized
        device = features.device
        B = features.shape[0]
        features = F.normalize(features, dim=1)

        # Similarity matrix
        sim = torch.mm(features, features.T) / self.temperature  # [B, B]

        # Mask: same class = positive
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_pos.fill_diagonal_(0)  # exclude self

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # Exclude self from denominator
        mask_self = torch.eye(B, device=device)
        sim = sim * (1 - mask_self)

        exp_sim = torch.exp(sim) * (1 - mask_self)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of positives
        n_pos = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)

        # Only samples with at least 1 positive
        valid = n_pos > 0
        loss = -mean_log_prob[valid].mean()
        return loss


def train_router_ce(router, X, y, epochs=50, lr=0.001):
    """Standard CE training (baseline)."""
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        for s in range(0, len(X), 512):
            idx = perm[s:s+512]
            loss = crit(router(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def train_router_supcon(router, X, y, supcon_epochs=30, ce_epochs=20,
                         lr_supcon=0.001, lr_ce=0.001, temperature=0.1):
    """SupCon pre-train encoder, then CE fine-tune full router."""
    device = X.device
    supcon_loss_fn = SupConLoss(temperature=temperature)

    # Phase 1: SupCon on encoder (embeddings)
    router.train()
    # Only train encoder
    opt = torch.optim.Adam(router.enc.parameters(), lr=lr_supcon)
    for ep in range(supcon_epochs):
        perm = torch.randperm(len(X), device=device)
        for s in range(0, len(X), 256):
            idx = perm[s:s+256]
            emb = router.enc(X[idx])
            loss = supcon_loss_fn(emb, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # Initialize prototypes as class centroids from learned embeddings
    with torch.no_grad():
        emb_all = router.enc(X)
        for c in range(N_CLASSES):
            mask = y == c
            if mask.any():
                router.protos.data[c] = emb_all[mask].mean(0)

    # Phase 2: CE fine-tune (full router)
    train_router_ce(router, X, y, epochs=ce_epochs, lr=lr_ce)


def eval_system(router, fast, X, y):
    router.eval(); fast.eval()
    with torch.no_grad():
        ids = router.route(X)
        routing_acc = (ids == y).float().mean().item()
        out = fast.forward_auto(X, ids)
        system_acc = (out.argmax(1) == y).float().mean().item()
    return routing_acc, system_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA C -- krok C2: Contrastive Router (SupCon)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    all_results = {}

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\n{'='*40}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*40}")

        Xtr, ytr, Xte, yte = load_dataset(ds_name, device)
        print("Trening specjalistow (raz)...")
        fast = train_specialists(Xtr, ytr, device)

        enc_h = 256
        emb = 64 if ds_name == "MNIST" else 32

        configs = [
            ("CE baseline", "ce", {}),
            ("SupCon T=0.07 + CE", "supcon", {"temperature": 0.07,
                "supcon_epochs": 30, "ce_epochs": 20}),
            ("SupCon T=0.1 + CE", "supcon", {"temperature": 0.1,
                "supcon_epochs": 30, "ce_epochs": 20}),
            ("SupCon T=0.2 + CE", "supcon", {"temperature": 0.2,
                "supcon_epochs": 30, "ce_epochs": 20}),
            ("SupCon T=0.1 long", "supcon", {"temperature": 0.1,
                "supcon_epochs": 50, "ce_epochs": 30}),
        ]

        print(f"\n{'config':<28} {'rout_acc':>9} {'sys_acc':>9}")
        print("-" * 50)

        ds_results = []
        for name, method, kwargs in configs:
            torch.manual_seed(42)
            router = ProtoRouter(N_IN, N_CLASSES, enc_hidden=enc_h, emb=emb).to(device)
            if method == "ce":
                ep = 50 if ds_name == "MNIST" else 30
                train_router_ce(router, Xtr, ytr, epochs=ep)
            else:
                train_router_supcon(router, Xtr, ytr, **kwargs)

            r_acc, s_acc = eval_system(router, fast, Xte, yte)
            print(f"{name:<28} {r_acc*100:>8.2f}% {s_acc*100:>8.2f}%")
            ds_results.append({
                "config": name, "method": method,
                "routing_acc": round(r_acc, 4),
                "system_acc": round(s_acc, 4),
                **kwargs,
            })

        baseline = ds_results[0]
        best = max(ds_results, key=lambda r: r["system_acc"])
        delta = best["system_acc"] - baseline["system_acc"]
        print(f"\n  Baseline: {baseline['system_acc']*100:.2f}%")
        print(f"  Best: {best['config']} = {best['system_acc']*100:.2f}% "
              f"(delta={delta*100:+.2f}pp)")

        all_results[ds_name] = ds_results

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "C2_contrastive_router.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
