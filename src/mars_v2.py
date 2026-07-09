"""
mars_v2.py -- Rdzen architektury M.A.R.S. v2.

Laczy dwa przelomy z Drogi C w jednym, czystym pliku:
  - Shared Backbone (z C4): wspolny encoder 784->backbone_hidden dla routera i podow.
  - Adaptive Compute (z C1): trojpoziomowe wnioskowanie -- forward_adaptive()
    (Etap 3). W trybie domyslnym forward() = TOP-1 (= C4).

ROZNICE WZGLEDEM C4 (swiadome, zgodne z planem D1):
  1. Pody trenowane na REALNYM przypisaniu routera (argmax), NIE na ORACLE.
     W C4 (train_shared_system, Faza 2) pody uczyly sie na ytr (prawdziwa
     etykieta = idealne przypisanie). To dawalo ORACLE=100%, ale tworzylo
     train/test mismatch: pod nigdy nie widzial probek, ktore router skieruje
     blednie. Tutaj pod uczy sie na danych, ktore REALNIE do niego trafia.
  2. Dwa tryby treningu (oba mierzone w D1):
     - "phased"     (D1a): Faza1 backbone+router -> freeze -> Faza2 pody.
     - "end_to_end" (D1b): laczony loss L = L_router + alpha * L_pods.

Konwencje (kształt klasy, wzory MAC, routing prototypowy) zachowane z
run_C4_shared_backbone.py i routers_v2.py.

Etap 1 = ten plik w trybie top-1. Etap 2 = run_D1 (trening + punkt kontrolny
zgodnosci z C4). Etap 3 = forward_adaptive (progi EE/top-1/top-2) + sweep Pareto.
"""
import torch
import torch.nn as nn

N_IN, N_CLASSES = 784, 10


class MarsV2System(nn.Module):
    """
    Wspolny backbone + routing head (prototypowy) + N pod heads.

    Pod heads trzymane jako stacked tensory [n_pods, in, out] -- wektoryzacja
    w duchu FastPods/SharedBackboneSystem (bmm zamiast petli po podach).
    """

    def __init__(self, n_in=N_IN, backbone_hidden=256, n_pods=N_CLASSES,
                 emb_dim=32, pod_hidden=24, n_out=N_CLASSES):
        super().__init__()
        self.n_in = n_in
        self.n_pods = n_pods
        self.backbone_hidden = backbone_hidden
        self.emb_dim = emb_dim
        self.pod_hidden = pod_hidden
        self.n_out = n_out

        # --- Shared backbone: wspolny ekstraktor cech (zawsze aktywny) ---
        self.backbone = nn.Sequential(
            nn.Linear(n_in, backbone_hidden),
            nn.ReLU(),
        )

        # --- Routing head: features -> embedding -> dystans do prototypow ---
        self.routing_head = nn.Linear(backbone_hidden, emb_dim)
        self.protos = nn.Parameter(torch.randn(n_pods, emb_dim) * 0.1)

        # --- Pod heads: male klasyfikatory na wyjsciu backbone ---
        # kazdy pod: backbone_hidden -> pod_hidden -> n_out
        self.pod_W1 = nn.Parameter(torch.randn(n_pods, backbone_hidden, pod_hidden) * 0.01)
        self.pod_b1 = nn.Parameter(torch.zeros(n_pods, pod_hidden))
        self.pod_W2 = nn.Parameter(torch.randn(n_pods, pod_hidden, n_out) * 0.01)
        self.pod_b2 = nn.Parameter(torch.zeros(n_pods, n_out))

    # ---------------------------------------------------------------- routing
    def features(self, x):
        """Wspolne cechy backbone (liczone RAZ, reuzywane przez router i pody)."""
        return self.backbone(x)

    def route_logits(self, features):
        """Logity routingu = -dystans do prototypow [B, n_pods]."""
        emb = self.routing_head(features)
        dists = torch.cdist(emb.unsqueeze(0), self.protos.unsqueeze(0)).squeeze(0)
        return -dists

    def route(self, features):
        """Twarde przypisanie (argmax) -- wybrany pod per probka [B]."""
        return self.route_logits(features).argmax(dim=1)

    # ------------------------------------------------------------------- pods
    def pod_forward(self, features, pod_ids):
        """Forward przez wybrane pod heads (wektoryzowany bmm). [B, n_out]."""
        W1 = self.pod_W1[pod_ids]   # [B, backbone_h, pod_h]
        b1 = self.pod_b1[pod_ids]   # [B, pod_h]
        W2 = self.pod_W2[pod_ids]   # [B, pod_h, n_out]
        b2 = self.pod_b2[pod_ids]   # [B, n_out]
        h = torch.relu(torch.bmm(features.unsqueeze(1), W1).squeeze(1) + b1)
        return torch.bmm(h.unsqueeze(1), W2).squeeze(1) + b2

    # --------------------------------------------------------------- forward
    def forward(self, x):
        """
        Pelny forward w trybie TOP-1 (Etap 1, odpowiednik C4):
        backbone -> route (argmax) -> wybrany pod.
        """
        feats = self.features(x)
        pod_ids = self.route(feats)
        return self.pod_forward(feats, pod_ids)

    # ------------------------------------------------------------------- MAC
    def mac_per_sample_top1(self):
        """
        Sredni MAC per probka w trybie top-1 (1 pod na probke).
        JAWNIE rozbity na skladniki -- backbone jest STALYM kosztem, placonym
        zawsze (kluczowe dla uczciwego porownania v1<->v2 i dla Etapu 3).
        """
        backbone_mac = self.n_in * self.backbone_hidden
        routing_mac = self.backbone_hidden * self.emb_dim + self.emb_dim * self.n_pods
        pod_mac = self.backbone_hidden * self.pod_hidden + self.pod_hidden * self.n_out
        return {
            "backbone": backbone_mac,
            "routing": routing_mac,
            "pod": pod_mac,
            "total_top1": backbone_mac + routing_mac + pod_mac,
        }

    def n_params(self):
        """Liczba parametrow (do zrownania budzetu z baseline v1)."""
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------- ETAP 3: adaptive compute
    def forward_adaptive(self, x, theta_high, theta_low):
        """
        Trojpoziomowe wnioskowanie (adaptacja logiki z C1, na shared backbone):

          conf > theta_high          -> EARLY EXIT (predykcja routera, 0 podow)
          theta_low <= conf <= t_high -> TOP-1      (1 pod)
          conf < theta_low           -> TOP-2      (2 pody, agregacja wazona)

        KLUCZOWE dla MAC: backbone + routing licza sie ZAWSZE (raz), bo
        confidence pochodzi z routera, ktory siedzi na backbonie. Pody to
        koszt DODATKOWY: 0 / 1 / 2 razy pod_mac. features liczone RAZ i
        reuzywane (zrodlo oszczednosci) -- przy top-2 backbone NIE liczy sie 2x.

        Zwraca (predictions[B], stats dict).
        """
        feats = self.features(x)
        logits = self.route_logits(feats)
        probs = torch.softmax(logits, dim=1)
        conf, router_pred = probs.max(dim=1)

        early_mask = conf >= theta_high
        top2_mask = conf < theta_low
        top1_mask = ~early_mask & ~top2_mask

        preds = torch.zeros(len(x), dtype=torch.long, device=x.device)

        # Tier 1: Early Exit -- predykcja routera, bez podow
        if early_mask.any():
            preds[early_mask] = router_pred[early_mask]

        # Tier 2: Top-1 -- jeden pod (wskazany przez router)
        if top1_mask.any():
            ids = router_pred[top1_mask]
            out = self.pod_forward(feats[top1_mask], ids)
            preds[top1_mask] = out.argmax(1)

        # Tier 3: Top-2 -- dwa pody, agregacja wazona confidence
        if top2_mask.any():
            top2_probs, top2_ids = probs[top2_mask].topk(2, dim=1)
            p0, p1 = top2_ids[:, 0], top2_ids[:, 1]
            w0, w1 = top2_probs[:, 0].unsqueeze(1), top2_probs[:, 1].unsqueeze(1)
            f2 = feats[top2_mask]
            out0 = self.pod_forward(f2, p0)
            out1 = self.pod_forward(f2, p1)
            preds[top2_mask] = (w0 * out0 + w1 * out1).argmax(1)

        n = len(x)
        n_early = int(early_mask.sum())
        n_top1 = int(top1_mask.sum())
        n_top2 = int(top2_mask.sum())

        mac = self.mac_per_sample_top1()
        fixed = mac["backbone"] + mac["routing"]   # placone zawsze
        pod = mac["pod"]
        # sredni MAC: fixed (zawsze) + pod * (1*top1 + 2*top2) / n
        avg_mac = fixed + pod * (n_top1 + 2 * n_top2) / n

        stats = {
            "avg_mac": round(avg_mac),
            "pct_early": round(n_early / n * 100, 1),
            "pct_top1": round(n_top1 / n * 100, 1),
            "pct_top2": round(n_top2 / n * 100, 1),
            "n_early": n_early, "n_top1": n_top1, "n_top2": n_top2,
        }
        return preds, stats


# ====================================================================== train
def train_phased(model, Xtr, ytr, epochs=30, lr=0.001, batch=512, device="cpu"):
    """
    D1a -- trening 2-fazowy.
      Faza 1: backbone + routing_head (klasyfikacja 10 klas).
      Faza 2: ZAMROZENIE backbone + router, trening podow na TWARDYM
              przypisaniu routera (argmax) -- NIE na ORACLE.
    """
    crit = nn.CrossEntropyLoss()

    # --- Faza 1: routing ---
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    opt1 = torch.optim.Adam(
        list(model.backbone.parameters())
        + list(model.routing_head.parameters()) + [model.protos], lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            feats = model.backbone(Xtr[idx])
            loss = crit(model.route_logits(feats), ytr[idx])
            opt1.zero_grad(); loss.backward(); opt1.step()

    # --- Faza 2: pody na realnym routingu, backbone+router zamrozone ---
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in [model.routing_head.weight, model.routing_head.bias, model.protos]:
        p.requires_grad = False
    opt2 = torch.optim.Adam(
        [model.pod_W1, model.pod_b1, model.pod_W2, model.pod_b2], lr=lr)

    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            with torch.no_grad():
                feats = model.backbone(Xtr[idx])
                pod_ids = model.route(feats)          # <-- ROUTER, nie ytr
            out = model.pod_forward(feats, pod_ids)
            loss = crit(out, ytr[idx])
            opt2.zero_grad(); loss.backward(); opt2.step()

    for p in model.parameters():
        p.requires_grad = True
    return model


def train_end_to_end(model, Xtr, ytr, epochs=30, lr=0.001, batch=512,
                     alpha=1.0, device="cpu"):
    """
    D1b -- trening end-to-end (jednofazowy).
      Laczony loss: L = L_router + alpha * L_pods.
      L_router: routing_head ma trafiac w prawdziwa klase.
      L_pods:   pod wskazany przez router (argmax, detach) ma klasyfikowac
                poprawnie -- pody ucza sie na realnym przypisaniu, ale gradient
                routingu plynie przez L_router (nie przez argmax, ktory jest
                nierozniczkowalny).
    """
    crit = nn.CrossEntropyLoss()
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            feats = model.backbone(Xtr[idx])
            route_logits = model.route_logits(feats)
            loss_router = crit(route_logits, ytr[idx])
            # twarde przypisanie routera (detach -- nieroz. argmax nie niesie gradientu)
            with torch.no_grad():
                pod_ids = route_logits.argmax(dim=1)
            out = model.pod_forward(feats, pod_ids)
            loss_pods = crit(out, ytr[idx])
            loss = loss_router + alpha * loss_pods
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ======================================================================= eval
def evaluate(model, Xte, yte):
    """
    Zwraca routing_acc, system_acc (top-1), oracle_acc.
    ORACLE liczony WYLACZNIE jako sufit/diagnostyka -- NIE jest metryka do paperu.
    """
    model.eval()
    with torch.no_grad():
        feats = model.features(Xte)
        ids = model.route(feats)
        routing_acc = (ids == yte).float().mean().item()
        system_acc = (model.pod_forward(feats, ids).argmax(1) == yte).float().mean().item()
        oracle_acc = (model.pod_forward(feats, yte).argmax(1) == yte).float().mean().item()
    return routing_acc, system_acc, oracle_acc


# ====================================================== ETAP 3: sweep + Pareto
# Domyslna siatka progow. theta_high steruje Early Exit (ile latwych probek
# omija pody), theta_low steruje Selective Top-2 (ile niepewnych dostaje 2 pody).
DEFAULT_THETA_GRID = [
    # (theta_high, theta_low, name)
    (1.01, 0.0,  "baseline top-1"),          # bez EE, bez top-2 (= czysty top-1)
    (0.999, 0.0, "EE>99.9%"),
    (0.99, 0.0,  "EE>99%"),
    (0.95, 0.0,  "EE>95%"),
    (0.90, 0.0,  "EE>90%"),
    (1.01, 0.5,  "T2<50%"),
    (1.01, 0.7,  "T2<70%"),
    (0.99, 0.5,  "EE>99% + T2<50%"),
    (0.99, 0.7,  "EE>99% + T2<70%"),
    (0.95, 0.5,  "EE>95% + T2<50%"),
    (0.95, 0.7,  "EE>95% + T2<70%"),
]


def adaptive_sweep(model, Xte, yte, grid=None):
    """
    Przebiega siatke progow, zwraca liste wynikow {name, theta_high, theta_low,
    acc, avg_mac, pct_early/top1/top2}. Baza do krzywej Pareto (Accuracy vs MAC).
    """
    grid = grid or DEFAULT_THETA_GRID
    model.eval()
    results = []
    with torch.no_grad():
        for th, tl, name in grid:
            preds, stats = model.forward_adaptive(Xte, th, tl)
            acc = (preds == yte).float().mean().item()
            results.append({"config": name, "theta_high": th, "theta_low": tl,
                            "acc": round(acc, 4), **stats})
    return results


def pareto_front(points, acc_key="acc", mac_key="avg_mac"):
    """
    Zwraca podzbior punktow na froncie Pareto: nie istnieje inny punkt o
    >= acc I <= mac (czyli nie jest zdominowany). Posortowany rosnaco po MAC.
    """
    front = []
    for p in points:
        dominated = any(
            (q[acc_key] >= p[acc_key] and q[mac_key] <= p[mac_key]
             and (q[acc_key] > p[acc_key] or q[mac_key] < p[mac_key]))
            for q in points)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda r: r[mac_key])


if __name__ == "__main__":
    # Szybki smoke test poprawnosci ksztaltow (bez danych, losowe wejscie).
    torch.manual_seed(0)
    m = MarsV2System(backbone_hidden=256, emb_dim=32, pod_hidden=24)
    x = torch.randn(64, N_IN)
    feats = m.features(x)
    assert feats.shape == (64, 256)
    assert m.route(feats).shape == (64,)
    assert m.pod_forward(feats, m.route(feats)).shape == (64, 10)
    assert m.forward(x).shape == (64, 10)
    # smoke test adaptive
    preds, stats = m.forward_adaptive(x, 0.95, 0.5)
    assert preds.shape == (64,)
    mac = m.mac_per_sample_top1()
    print("Smoke test OK.")
    print(f"  Params: {m.n_params():,}")
    print(f"  MAC top-1: {mac['total_top1']:,} "
          f"(backbone={mac['backbone']:,}, routing={mac['routing']:,}, pod={mac['pod']:,})")
