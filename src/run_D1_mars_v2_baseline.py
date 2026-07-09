"""
run_D1_mars_v2_baseline.py -- Droga D, krok D1: benchmark M.A.R.S. v2.

ETAP 2 (ten plik na teraz): trening i PUNKT KONTROLNY.
  - Trenujemy MarsV2System 2-fazowo (D1a) na MNIST.
  - Sprawdzamy, czy szkielet w trybie top-1 odtwarza wynik z C4
    (router ~98.2%). To brama: jak routing sie zgadza, fundament jest OK.
  - Dodatkowo trenujemy wariant end-to-end (D1b) i uczciwy baseline v1
    (Separate, zrownany po liczbie parametrow) -- dla pelnego porownania.

UWAGA METODOLOGICZNA (zgodnie z planem):
  Pody w v2 ucza sie na REALNYM przypisaniu routera (argmax), nie na ORACLE.
  Dlatego system_acc moze wyjsc nieco nizej niz "idealne" liczby z C4, gdzie
  pody trenowano na ytr. Punktem kontrolnym zgodnosci jest ROUTING accuracy
  (Faza 1 identyczna jak C4), nie zawyzone ORACLE. To poprawne zachowanie.

  ORACLE raportujemy WYLACZNIE jako sufit/diagnostyke -- nie jest metryka do paperu.

Etap 3 (pozniej): dopisanie forward_adaptive (progi EE/top-1/top-2) i sweep
krzywej Pareto. Tu jeszcze go nie ma -- najpierw potwierdzamy fundament.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_D1_mars_v2_baseline.py
"""

import json, os, sys
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import (MarsV2System, train_phased, train_end_to_end, evaluate,
                     N_IN, N_CLASSES)
from routers_v2 import ProtoRouter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Punkt kontrolny: router z C4/A5 na MNIST osiagal ~98.2%.
C4_ROUTER_REF = 0.982
CHECKPOINT_TOL = 0.010   # tolerancja +-1.0pp (rozne seedy/inicjalizacje)


def load_dataset(name, device):
    if name == "MNIST":
        ds_cls = torchvision.datasets.MNIST
        mean, std = 0.1307, 0.3081
    else:
        ds_cls = torchvision.datasets.FashionMNIST
        mean, std = 0.2860, 0.3530
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))])
    train = ds_cls(root=DATA_DIR, train=True, download=True, transform=transform)
    test = ds_cls(root=DATA_DIR, train=False, download=True, transform=transform)
    Xtr = torch.stack([train[i][0].view(-1) for i in range(len(train))]).to(device)
    ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(device)
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))]).to(device)
    yte = torch.tensor([test[i][1] for i in range(len(test))]).to(device)
    return Xtr, ytr, Xte, yte


# =================================================================== baseline v1
class SeparateV1(nn.Module):
    """
    Uczciwy baseline v1 (Separate): router i pody maja OSOBNE ekstraktory z 784.
    Brak wspoldzielonego backbone -- to jest architektura, ktora v2 ma pobic.
    Routing prototypowy (ProtoRouter), pody = wektoryzowane glowice z 784.
    """
    def __init__(self, n_in=N_IN, n_pods=N_CLASSES,
                 router_enc_hidden=256, router_emb=64, pod_hidden=24, n_out=N_CLASSES):
        super().__init__()
        self.n_in, self.n_pods = n_in, n_pods
        self.pod_hidden, self.n_out = pod_hidden, n_out
        self.router = ProtoRouter(n_in, n_pods, enc_hidden=router_enc_hidden, emb=router_emb)
        # osobne pody operujace na surowym 784 (jak w C1)
        self.pod_W1 = nn.Parameter(torch.randn(n_pods, n_in, pod_hidden) / (n_in ** 0.5))
        self.pod_b1 = nn.Parameter(torch.zeros(n_pods, pod_hidden))
        self.pod_W2 = nn.Parameter(torch.randn(n_pods, pod_hidden, n_out) / (pod_hidden ** 0.5))
        self.pod_b2 = nn.Parameter(torch.zeros(n_pods, n_out))

    def route(self, x):
        return self.router(x).argmax(dim=1)

    def pod_forward(self, x, pod_ids):
        W1, b1 = self.pod_W1[pod_ids], self.pod_b1[pod_ids]
        W2, b2 = self.pod_W2[pod_ids], self.pod_b2[pod_ids]
        h = torch.relu(torch.bmm(x.unsqueeze(1), W1).squeeze(1) + b1)
        return torch.bmm(h.unsqueeze(1), W2).squeeze(1) + b2

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def mac_per_sample(self):
        router_mac = self.router.mac_per_sample()
        pod_mac = self.n_in * self.pod_hidden + self.pod_hidden * self.n_out
        return {"router": router_mac, "pod": pod_mac, "total_top1": router_mac + pod_mac}


def train_separate_v1(model, Xtr, ytr, epochs=30, lr=0.001, batch=512, device="cpu"):
    """Trening baseline v1: router, potem pody na realnym routingu (argmax)."""
    crit = nn.CrossEntropyLoss()
    model.train()
    # Faza 1: router
    opt1 = torch.optim.Adam(model.router.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            loss = crit(model.router(Xtr[idx]), ytr[idx])
            opt1.zero_grad(); loss.backward(); opt1.step()
    # Faza 2: pody na realnym routingu (spojnie z v2 -- uczciwe porownanie)
    opt2 = torch.optim.Adam([model.pod_W1, model.pod_b1, model.pod_W2, model.pod_b2], lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            with torch.no_grad():
                pod_ids = model.route(Xtr[idx])
            out = model.pod_forward(Xtr[idx], pod_ids)
            loss = crit(out, ytr[idx])
            opt2.zero_grad(); loss.backward(); opt2.step()
    return model


def eval_separate_v1(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        ids = model.route(Xte)
        routing_acc = (ids == yte).float().mean().item()
        system_acc = (model.pod_forward(Xte, ids).argmax(1) == yte).float().mean().item()
        oracle_acc = (model.pod_forward(Xte, yte).argmax(1) == yte).float().mean().item()
    return routing_acc, system_acc, oracle_acc


# ======================================================================== main
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA D -- krok D1 (Etap 2): trening + punkt kontrolny")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    EPOCHS = 30
    # Konfiguracja v2 ZROWNANA PARAMETRYCZNIE z baseline v1 (Separate).
    # v1 (reh=256,remb=64,ph=24) = 408,948 params.
    # v2 (bb=384,emb=32,ph=24)   = 408,980 params -> diff 32 (0.008%).
    # Parametry to os KONTROLOWANA; MAC raportujemy jako wynik (zgodnie z planem).
    BB_H, EMB, POD_H = 384, 32, 24

    all_results = {}

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\n{'='*52}\nDataset: {ds_name}\n{'='*52}")
        Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

        # --- v2 D1a: phased ---
        torch.manual_seed(42)
        v2a = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES).to(device)
        train_phased(v2a, Xtr, ytr, epochs=EPOCHS, device=device)
        r_a, s_a, o_a = evaluate(v2a, Xte, yte)

        # --- v2 D1b: end-to-end ---
        torch.manual_seed(42)
        v2b = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES).to(device)
        train_end_to_end(v2b, Xtr, ytr, epochs=EPOCHS, alpha=1.0, device=device)
        r_b, s_b, o_b = evaluate(v2b, Xte, yte)

        # --- baseline v1: Separate (zrownany po parametrach) ---
        torch.manual_seed(42)
        v1 = SeparateV1(N_IN, N_CLASSES, router_enc_hidden=256, router_emb=64,
                        pod_hidden=24).to(device)
        train_separate_v1(v1, Xtr, ytr, epochs=EPOCHS, device=device)
        r_1, s_1, o_1 = eval_separate_v1(v1, Xte, yte)

        mac_v2 = v2a.mac_per_sample_top1()
        mac_v1 = v1.mac_per_sample()

        # --- raport ---
        print(f"\n{'model':<22} {'params':>9} {'rout':>7} {'sys':>7} "
              f"{'oracle':>7} {'MAC(t1)':>9}")
        print("-" * 66)
        print(f"{'v1 Separate':<22} {v1.n_params():>9,} {r_1*100:>6.1f}% "
              f"{s_1*100:>6.1f}% {o_1*100:>6.1f}% {mac_v1['total_top1']:>9,}")
        print(f"{'v2 D1a phased':<22} {v2a.n_params():>9,} {r_a*100:>6.1f}% "
              f"{s_a*100:>6.1f}% {o_a*100:>6.1f}% {mac_v2['total_top1']:>9,}")
        print(f"{'v2 D1b end2end':<22} {v2b.n_params():>9,} {r_b*100:>6.1f}% "
              f"{s_b*100:>6.1f}% {o_b*100:>6.1f}% {mac_v2['total_top1']:>9,}")

        # --- kontrola zrownania parametrow (uczciwy baseline) ---
        pdiff = v2a.n_params() - v1.n_params()
        ppct = 100 * pdiff / v1.n_params()
        print(f"\n  [ZROWNANIE PARAMETROW] v2 - v1 = {pdiff:+,} "
              f"({ppct:+.2f}%) -> {'OK (<2%)' if abs(ppct) < 2 else 'NIEZROWNANE'}")

        # --- punkt kontrolny (tylko MNIST) ---
        if ds_name == "MNIST":
            best_router = max(r_a, r_b)
            diff = best_router - C4_ROUTER_REF
            ok = abs(diff) <= CHECKPOINT_TOL
            print(f"\n  [PUNKT KONTROLNY] router v2 = {best_router*100:.1f}% "
                  f"vs C4 ref {C4_ROUTER_REF*100:.1f}% "
                  f"(diff {diff*100:+.1f}pp) -> {'OK' if ok else 'SPRAWDZ'}")
            if not ok:
                print("  Uwaga: routing odbiega od C4. Sprawdz seed/epoki/konfiguracje "
                      "zanim przejdziemy do Etapu 3.")

        all_results[ds_name] = {
            "config": {"backbone_hidden": BB_H, "emb": EMB, "pod_hidden": POD_H,
                       "epochs": EPOCHS},
            "v1_separate": {"params": v1.n_params(),
                            "routing_acc": round(r_1, 4), "system_acc": round(s_1, 4),
                            "oracle_acc": round(o_1, 4), "mac": mac_v1},
            "v2_d1a_phased": {"params": v2a.n_params(),
                              "routing_acc": round(r_a, 4), "system_acc": round(s_a, 4),
                              "oracle_acc": round(o_a, 4), "mac": mac_v2},
            "v2_d1b_end2end": {"params": v2b.n_params(),
                               "routing_acc": round(r_b, 4), "system_acc": round(s_b, 4),
                               "oracle_acc": round(o_b, 4), "mac": mac_v2},
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "D1_mars_v2_baseline.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")
    print("\nEtap 2 zakonczony. Jesli punkt kontrolny OK -> Etap 3 (forward_adaptive + sweep).")


if __name__ == "__main__":
    main()
