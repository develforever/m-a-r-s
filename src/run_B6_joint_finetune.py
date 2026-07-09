"""
run_B6_joint_finetune.py — B6: end-to-end joint fine-tuning na F-MNIST.

KLUCZOWY INSIGHT z B1/B2: bottleneck to nie encoder depth ani augmentacja,
ale fakt że router i pody trenowane OSOBNO. Router uczy się "jaka to klasa"
(routing_acc=89%), ale nie wie które pody lepiej radzą z konkretnymi próbkami.

Joint fine-tuning: po osobnym treningu (router + pody), kilka epok end-to-end:
  1. Router wybiera pod (hard routing)
  2. Pod generuje logity
  3. Loss: CE na finalnym output
  4. Gradient: Gumbel-Softmax (differentiable routing) lub REINFORCE

To pozwala routerowi nauczyć się "wyślij próbkę tam, gdzie pod da dobry wynik"
zamiast "wyślij próbkę do poda z jej klasą".

Uruchom:
    .venv\\Scripts\\python.exe src\\run_B6_joint_finetune.py
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


def train_pods_and_router(Xtr, ytr, device, hidden=HIDDEN, own_ratio=0.7,
                          pod_epochs=12, router_epochs=30):
    """Faza 1: trening osobny (baseline)."""
    # Router
    router = ProtoRouter(N_IN, N_PODS, enc_hidden=256, emb=32).to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    for _ in range(router_epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # Pods
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


def gumbel_softmax_routing(router_logits, temperature=1.0, hard=True):
    """Gumbel-Softmax: differentiable approximation of argmax."""
    return F.gumbel_softmax(router_logits, tau=temperature, hard=hard)


def joint_forward_gumbel(router, fast, x, tau=1.0):
    """
    End-to-end forward z Gumbel-Softmax routing.
    Returns logits [B, N_OUT].
    """
    B = x.shape[0]
    router_logits = router(x)  # [B, N_PODS]
    routing_weights = gumbel_softmax_routing(router_logits, temperature=tau, hard=True)
    # routing_weights: [B, N_PODS] — one-hot (hard) z gradientem (straight-through)

    # Forward przez WSZYSTKIE pody (potrzebne do gradientu)
    # x: [B, N_IN], W1: [N_PODS, N_IN, H] → all_h: [N_PODS, B, H]
    # Oblicz output każdego poda
    all_logits = []
    for pid in range(fast.n_pods):
        h = torch.relu(x @ fast.W1[pid] + fast.b1[pid])
        o = h @ fast.W2[pid] + fast.b2[pid]
        all_logits.append(o)  # [B, N_OUT]
    all_logits = torch.stack(all_logits, dim=1)  # [B, N_PODS, N_OUT]

    # Ważona suma (ale hard=True → efektywnie wybiera 1 pod)
    out = (routing_weights.unsqueeze(-1) * all_logits).sum(dim=1)  # [B, N_OUT]
    return out


def joint_forward_reinforce(router, fast, x):
    """
    Forward z hard routing + REINFORCE gradient estimation.
    Zwraca logits i log_prob wybranej akcji.
    """
    router_logits = router(x)  # [B, N_PODS]
    probs = F.softmax(router_logits, dim=1)
    ids = probs.argmax(dim=1)  # hard routing (deterministic at eval)

    # Przy treningu: sample
    if router.training:
        dist = torch.distributions.Categorical(probs)
        ids = dist.sample()
        log_prob = dist.log_prob(ids)
    else:
        log_prob = None

    out = fast.forward_auto(x, ids)
    return out, ids, log_prob


def joint_finetune_gumbel(router, fast, Xtr, ytr, epochs=5, lr=0.0003, tau_start=2.0, tau_end=0.5):
    """Joint fine-tuning z Gumbel-Softmax."""
    device = Xtr.device
    # Optymalizator na WSZYSTKO (router + pods)
    params = list(router.parameters()) + list(fast.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss()

    router.train()
    fast.train()

    for ep in range(epochs):
        tau = tau_start - (tau_start - tau_end) * ep / max(epochs - 1, 1)
        perm = torch.randperm(len(Xtr), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            out = joint_forward_gumbel(router, fast, Xtr[idx], tau=tau)
            loss = crit(out, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

    return epoch_loss / max(n_batches, 1)


def joint_finetune_reinforce(router, fast, Xtr, ytr, epochs=5, lr_router=0.0003, lr_pods=0.0001):
    """Joint fine-tuning z REINFORCE (policy gradient)."""
    device = Xtr.device
    opt_router = torch.optim.Adam(router.parameters(), lr=lr_router)
    opt_pods = torch.optim.Adam(fast.parameters(), lr=lr_pods)
    crit = nn.CrossEntropyLoss(reduction='none')

    router.train()
    fast.train()

    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            x, y = Xtr[idx], ytr[idx]

            # Forward z sampling
            out, ids, log_prob = joint_forward_reinforce(router, fast, x)

            # Pod loss (standard CE, direct gradient)
            pod_loss = crit(out, y).mean()
            opt_pods.zero_grad()
            pod_loss.backward(retain_graph=True)
            opt_pods.step()

            # Router loss (REINFORCE: reward = correct prediction)
            with torch.no_grad():
                reward = (out.argmax(1) == y).float() - 0.5  # baseline=0.5
            router_loss = -(log_prob * reward).mean()
            opt_router.zero_grad()
            router_loss.backward()
            opt_router.step()


def eval_system(router, fast, Xte, yte):
    router.eval()
    fast.eval()
    with torch.no_grad():
        ids = router.route(Xte)
        routing_acc = (ids == yte).float().mean().item()
        out = fast.forward_auto(Xte, ids)
        system_acc = (out.argmax(1) == yte).float().mean().item()
    return routing_acc, system_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("B6 — Joint Fine-Tuning (end-to-end) na Fashion-MNIST")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    t0 = time.perf_counter()

    print("\nŁadowanie Fashion-MNIST...")
    Xtr, ytr, Xte, yte = load_fashion_mnist(device)

    # ================================================================
    # Faza 1: trening osobny (baseline)
    # ================================================================
    print("\nFaza 1: trening osobny...")
    router, fast = train_pods_and_router(Xtr, ytr, device)
    r_acc_before, s_acc_before = eval_system(router, fast, Xte, yte)
    print(f"  Przed joint: routing={r_acc_before*100:.1f}%  system={s_acc_before*100:.1f}%")

    # ================================================================
    # Faza 2: Joint fine-tuning — Gumbel-Softmax
    # ================================================================
    import copy
    configs = [
        # (label, method, epochs, lr/lr_r, lr_pods, tau_start, tau_end)
        ("Gumbel ep=3 lr=3e-4",    "gumbel", 3,  0.0003, None, 2.0, 0.5),
        ("Gumbel ep=5 lr=3e-4",    "gumbel", 5,  0.0003, None, 2.0, 0.5),
        ("Gumbel ep=10 lr=1e-4",   "gumbel", 10, 0.0001, None, 2.0, 0.3),
        ("Gumbel ep=10 lr=3e-4",   "gumbel", 10, 0.0003, None, 2.0, 0.3),
        ("REINFORCE ep=3",         "reinforce", 3,  0.0003, 0.0001, None, None),
        ("REINFORCE ep=5",         "reinforce", 5,  0.0003, 0.0001, None, None),
        ("REINFORCE ep=10",        "reinforce", 10, 0.0001, 0.00005, None, None),
    ]

    print(f"\n{'config':>25} {'rout_acc':>9} {'sys_acc':>9} {'delta':>7}")
    print("-" * 55)

    results_list = []
    best_sys_acc = s_acc_before
    best_cfg_name = "baseline (no joint)"

    for (label, method, epochs, lr, lr_p, tau_s, tau_e) in configs:
        # Deep copy — każdy test startuje od tego samego punktu
        r_copy = copy.deepcopy(router)
        f_copy = copy.deepcopy(fast)

        if method == "gumbel":
            joint_finetune_gumbel(r_copy, f_copy, Xtr, ytr,
                                  epochs=epochs, lr=lr, tau_start=tau_s, tau_end=tau_e)
        else:
            joint_finetune_reinforce(r_copy, f_copy, Xtr, ytr,
                                    epochs=epochs, lr_router=lr, lr_pods=lr_p)

        r_acc, s_acc = eval_system(r_copy, f_copy, Xte, yte)
        delta = s_acc - s_acc_before
        print(f"{label:>25} {r_acc*100:>8.1f}% {s_acc*100:>8.1f}% {delta*100:>+6.2f}")
        results_list.append({
            "config": label, "method": method, "epochs": epochs,
            "lr": lr, "lr_pods": lr_p,
            "routing_acc": round(r_acc, 4), "system_acc": round(s_acc, 4),
            "delta_pp": round(delta * 100, 2),
        })
        if s_acc > best_sys_acc:
            best_sys_acc = s_acc
            best_cfg_name = label

    # ================================================================
    # WNIOSEK
    # ================================================================
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("WNIOSEK — B6: Joint Fine-Tuning")
    print("=" * 72)
    print(f"Przed joint training:  system={s_acc_before*100:.2f}%")
    print(f"Best ({best_cfg_name}): system={best_sys_acc*100:.2f}%")
    print(f"Poprawa: {(best_sys_acc - s_acc_before)*100:+.2f}pp")
    print(f"Czas: {elapsed:.0f}s")

    # JSON
    results = {
        "experiment": "B6_joint_finetune_fashion",
        "dataset": "Fashion-MNIST",
        "device": device,
        "before_joint": {
            "routing_acc": round(r_acc_before, 4),
            "system_acc": round(s_acc_before, 4),
        },
        "best_config": best_cfg_name,
        "best_system_acc": round(best_sys_acc, 4),
        "delta_pp": round((best_sys_acc - s_acc_before) * 100, 2),
        "configs": results_list,
        "elapsed_s": round(elapsed, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "B6_joint_finetune_fashion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
