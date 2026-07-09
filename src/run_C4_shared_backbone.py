"""
run_C4_shared_backbone.py -- Droga C, krok 4: Shared Backbone.

Hipoteza: router i pody maja osobne encodery. Jesli wspoldzielimy encoder
(784->256), to:
  - router = encoder + mala glowica routingu (256->prototypy)
  - pod = encoder output -> mala glowica klasyfikacji (256->10)
  - mniej parametrow, potencjalnie lepszy routing (silniejszy encoder)

Porownanie:
  - Separate (baseline): router(784->256->emb) + pod(784->24->10)
  - Shared: encoder(784->256) + routing_head(256->emb) + pod_head(256->10)

Uruchom:
    .venv\\Scripts\\python.exe src\\run_C4_shared_backbone.py
"""

import json, os, sys
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_IN, N_CLASSES = 784, 10


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


class SharedBackboneSystem(nn.Module):
    """Wspolny encoder + routing head + N pod heads."""
    def __init__(self, n_in, backbone_hidden, n_pods, emb_dim, pod_hidden, n_out):
        super().__init__()
        self.n_pods = n_pods

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_in, backbone_hidden),
            nn.ReLU(),
        )

        # Routing head: backbone output -> embedding -> prototypes
        self.routing_head = nn.Linear(backbone_hidden, emb_dim)
        self.protos = nn.Parameter(torch.randn(n_pods, emb_dim) * 0.1)

        # Pod heads: small classifiers on top of backbone output
        # Each pod: backbone_hidden -> pod_hidden -> n_out
        self.pod_W1 = nn.Parameter(torch.randn(n_pods, backbone_hidden, pod_hidden) * 0.01)
        self.pod_b1 = nn.Parameter(torch.zeros(n_pods, pod_hidden))
        self.pod_W2 = nn.Parameter(torch.randn(n_pods, pod_hidden, n_out) * 0.01)
        self.pod_b2 = nn.Parameter(torch.zeros(n_pods, n_out))

    def route(self, features):
        """Route based on backbone features."""
        emb = self.routing_head(features)
        dists = torch.cdist(emb.unsqueeze(0), self.protos.unsqueeze(0)).squeeze(0)
        return dists.argmin(dim=1)

    def route_logits(self, features):
        emb = self.routing_head(features)
        dists = torch.cdist(emb.unsqueeze(0), self.protos.unsqueeze(0)).squeeze(0)
        return -dists

    def pod_forward(self, features, pod_ids):
        """Forward through selected pod heads."""
        B = features.shape[0]
        W1 = self.pod_W1[pod_ids]  # [B, backbone_h, pod_h]
        b1 = self.pod_b1[pod_ids]  # [B, pod_h]
        W2 = self.pod_W2[pod_ids]  # [B, pod_h, n_out]
        b2 = self.pod_b2[pod_ids]  # [B, n_out]
        h = torch.relu(torch.bmm(features.unsqueeze(1), W1).squeeze(1) + b1)
        return torch.bmm(h.unsqueeze(1), W2).squeeze(1) + b2

    def forward(self, x):
        """Full forward: backbone -> route -> pod."""
        features = self.backbone(x)
        pod_ids = self.route(features)
        return self.pod_forward(features, pod_ids)

    def mac_per_sample(self, backbone_hidden, pod_hidden, emb_dim):
        backbone_mac = N_IN * backbone_hidden
        routing_mac = backbone_hidden * emb_dim + emb_dim * self.n_pods
        pod_mac = backbone_hidden * pod_hidden + pod_hidden * N_CLASSES
        return backbone_mac + routing_mac + pod_mac


def train_shared_system(model, Xtr, ytr, epochs=30, lr=0.001):
    """End-to-end training of shared backbone system."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # Phase 1: Train routing (backbone + routing head)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            features = model.backbone(Xtr[idx])
            logits = model.route_logits(features)
            loss = crit(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # Phase 2: Train pod heads (backbone frozen for stability, pods learn)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in [model.routing_head.weight, model.routing_head.bias, model.protos]:
        p.requires_grad = False

    opt2 = torch.optim.Adam([model.pod_W1, model.pod_b1, model.pod_W2, model.pod_b2], lr=0.001)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            with torch.no_grad():
                features = model.backbone(Xtr[idx])
            # Train each sample through its correct pod (oracle routing for pod training)
            out = model.pod_forward(features, ytr[idx])
            loss = crit(out, ytr[idx])
            opt2.zero_grad(); loss.backward(); opt2.step()

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True


def eval_shared(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        features = model.backbone(Xte)
        ids = model.route(features)
        routing_acc = (ids == yte).float().mean().item()
        out = model.pod_forward(features, ids)
        system_acc = (out.argmax(1) == yte).float().mean().item()
        # ORACLE
        oracle_out = model.pod_forward(features, yte)
        oracle_acc = (oracle_out.argmax(1) == yte).float().mean().item()
    return routing_acc, system_acc, oracle_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA C -- krok C4: Shared Backbone")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    all_results = {}

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\n{'='*40}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*40}")

        Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

        configs = [
            # (name, backbone_h, pod_h, emb, epochs, lr)
            ("shared 256->16->10", 256, 16, 32, 30, 0.001),
            ("shared 256->24->10", 256, 24, 32, 30, 0.001),
            ("shared 256->32->10", 256, 32, 32, 30, 0.001),
            ("shared 128->24->10", 128, 24, 16, 30, 0.001),
            ("shared 256->24 ep50", 256, 24, 32, 50, 0.001),
        ]

        print(f"\n{'config':<24} {'rout':>7} {'sys':>7} {'oracle':>7} {'MAC':>8}")
        print("-" * 56)

        ds_results = []
        for name, bb_h, pod_h, emb, epochs, lr in configs:
            torch.manual_seed(42)
            model = SharedBackboneSystem(N_IN, bb_h, N_CLASSES, emb, pod_h, N_CLASSES).to(device)
            train_shared_system(model, Xtr, ytr, epochs=epochs, lr=lr)
            r_acc, s_acc, o_acc = eval_shared(model, Xte, yte)
            mac = model.mac_per_sample(bb_h, pod_h, emb)
            print(f"{name:<24} {r_acc*100:>6.1f}% {s_acc*100:>6.1f}% "
                  f"{o_acc*100:>6.1f}% {mac:>8,}")
            ds_results.append({
                "config": name,
                "backbone_hidden": bb_h, "pod_hidden": pod_h, "emb": emb,
                "routing_acc": round(r_acc, 4),
                "system_acc": round(s_acc, 4),
                "oracle_acc": round(o_acc, 4),
                "mac": mac,
            })

        best = max(ds_results, key=lambda r: r["system_acc"])
        print(f"\n  Best: {best['config']} = {best['system_acc']*100:.1f}% "
              f"(MAC={best['mac']:,})")

        all_results[ds_name] = ds_results

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "C4_shared_backbone.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
