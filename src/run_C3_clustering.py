"""
run_C3_clustering.py -- Droga C, krok 3: Learned Clustering.

Hipoteza: 10 podow (1 per klasa) nie jest optymalne. Moze 5 podow
(grupy klas) da lepszy routing, bo router musi odroznic 5 grup zamiast 10.

Testujemy:
  - K=10 (baseline, 1 pod per klasa)
  - K=5 (grupowanie k-means w przestrzeni embeddingu)
  - K=3 (wieksze grupy)
  - K=7 (posredni)
  - K=5 semantyczne (reczne grupy dla Fashion-MNIST)

Kazdy pod klasyfikuje WSZYSTKIE 10 klas, ale specjalizuje sie w swoich.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_C3_clustering.py
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
N_IN, N_OUT, HIDDEN = 784, 10, 24


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


def kmeans_cluster_classes(Xtr, ytr, n_clusters, device):
    """Grupuje 10 klas w n_clusters grup na podstawie centroidow klas."""
    # Compute class centroids
    centroids = []
    for c in range(10):
        centroids.append(Xtr[ytr == c].mean(0))
    centroids = torch.stack(centroids)  # [10, 784]

    # Simple k-means on centroids
    torch.manual_seed(42)
    centers = centroids[torch.randperm(10)[:n_clusters]]  # [K, 784]

    for _ in range(50):
        dists = torch.cdist(centroids, centers)  # [10, K]
        assignments = dists.argmin(1)  # [10]
        for k in range(n_clusters):
            mask = assignments == k
            if mask.any():
                centers[k] = centroids[mask].mean(0)

    # Map: class -> cluster
    class_to_cluster = assignments.tolist()
    return class_to_cluster


def build_cluster_system(Xtr, ytr, Xte, yte, n_pods, class_to_cluster, device,
                          enc_hidden=256, emb=32, router_epochs=30):
    """Buduje system z K podami (grupy klas)."""
    # Convert class labels to cluster labels
    c2c = torch.tensor(class_to_cluster, device=device)
    ytr_cluster = c2c[ytr]
    yte_cluster = c2c[yte]

    # Train router (routes to clusters)
    router = ProtoRouter(N_IN, n_pods, enc_hidden=enc_hidden, emb=emb).to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    for _ in range(router_epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr_cluster[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # Train specialist pods (each pod classifies all 10 classes, but
    # specializes in its cluster's classes)
    fast = FastPods(n_pods, N_IN, HIDDEN, N_OUT).to(device)
    crit2 = nn.CrossEntropyLoss()
    for k in range(n_pods):
        pod = nn.Sequential(nn.Linear(N_IN, HIDDEN), nn.ReLU(),
                            nn.Linear(HIDDEN, N_OUT)).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        # This pod's classes
        my_classes = [c for c in range(10) if class_to_cluster[c] == k]
        my_mask = torch.zeros(len(ytr), dtype=torch.bool, device=device)
        for c in my_classes:
            my_mask |= (ytr == c)
        own_X, own_y = Xtr[my_mask], ytr[my_mask]
        n_other = int(len(own_X) * 0.3 / 0.7)
        other_mask = ~my_mask
        if other_mask.sum() > 0:
            X_pod = torch.cat([own_X, Xtr[other_mask][:n_other]])
            y_pod = torch.cat([own_y, ytr[other_mask][:n_other]])
        else:
            X_pod, y_pod = own_X, own_y
        for _ in range(12):
            perm = torch.randperm(len(X_pod), device=device)
            for s in range(0, len(X_pod), 256):
                idx = perm[s:s+256]
                loss = crit2(pod(X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fast.W1.data[k] = pod[0].weight.data.T
            fast.b1.data[k] = pod[0].bias.data
            fast.W2.data[k] = pod[2].weight.data.T
            fast.b2.data[k] = pod[2].bias.data

    # Eval
    router.eval(); fast.eval()
    with torch.no_grad():
        ids = router.route(Xte)
        routing_acc = (ids == yte_cluster).float().mean().item()
        out = fast.forward_auto(Xte, ids)
        system_acc = (out.argmax(1) == yte).float().mean().item()
        # ORACLE (perfect routing to cluster)
        oracle_out = fast.forward_auto(Xte, yte_cluster)
        oracle_acc = (oracle_out.argmax(1) == yte).float().mean().item()

    router_mac = router.mac_per_sample()
    pod_mac = N_IN * HIDDEN + HIDDEN * N_OUT
    total_mac = router_mac + pod_mac

    return routing_acc, system_acc, oracle_acc, total_mac, router_mac


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA C -- krok C3: Learned Clustering")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    all_results = {}

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\n{'='*40}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*40}")

        Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

        configs = []

        # K=10 baseline (1:1 mapping)
        configs.append(("K=10 (baseline)", 10, list(range(10))))

        # K-means clustered
        for K in [7, 5, 3]:
            c2c = kmeans_cluster_classes(Xtr, ytr, K, device)
            configs.append((f"K={K} (k-means)", K, c2c))

        # Semantic groups for Fashion-MNIST
        if ds_name == "Fashion-MNIST":
            # 0:T-shirt, 1:Trouser, 2:Pullover, 3:Dress, 4:Coat,
            # 5:Sandal, 6:Shirt, 7:Sneaker, 8:Bag, 9:Ankle boot
            semantic_5 = [0, 1, 0, 2, 0, 3, 0, 3, 4, 3]
            # tops(0,2,4,6), pants(1), dress(3), shoes(5,7,9), bag(8)
            configs.append(("K=5 (semantic)", 5, semantic_5))

        print(f"\n{'config':<22} {'K':>3} {'groups':<36} "
              f"{'rout':>6} {'sys':>6} {'oracle':>7} {'MAC':>7}")
        print("-" * 96)

        ds_results = []
        for name, K, c2c in configs:
            # Format groups
            groups = {}
            for c, g in enumerate(c2c):
                groups.setdefault(g, []).append(c)
            groups_str = str(dict(sorted(groups.items())))
            if len(groups_str) > 35:
                groups_str = groups_str[:32] + "..."

            r_acc, s_acc, o_acc, total_mac, r_mac = build_cluster_system(
                Xtr, ytr, Xte, yte, K, c2c, device)

            print(f"{name:<22} {K:>3} {groups_str:<36} "
                  f"{r_acc*100:>5.1f}% {s_acc*100:>5.1f}% "
                  f"{o_acc*100:>6.1f}% {total_mac:>7,}")

            ds_results.append({
                "config": name, "n_pods": K,
                "class_to_cluster": c2c,
                "groups": groups,
                "routing_acc": round(r_acc, 4),
                "system_acc": round(s_acc, 4),
                "oracle_acc": round(o_acc, 4),
                "total_mac": total_mac,
                "router_mac": r_mac,
            })

        best = max(ds_results, key=lambda r: r["system_acc"])
        baseline = ds_results[0]
        delta = best["system_acc"] - baseline["system_acc"]
        print(f"\n  Baseline (K=10): {baseline['system_acc']*100:.1f}%")
        print(f"  Best: {best['config']} = {best['system_acc']*100:.1f}% "
              f"(delta={delta*100:+.1f}pp)")

        all_results[ds_name] = ds_results

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "C3_clustering.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
