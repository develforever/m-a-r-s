"""
mars_v2_hier.py -- E2: routing hierarchiczny (grupa -> klasa) na CNN backbone.

MOTYWACJA (z DROGA_E_PLAN.md + E1):
  E1: ~60% bledow routera na Fashion w 4 parach klastra upper-body, 86%
  probek odzyskiwalnych w tym klastrze, 86% w najnizszym kwartylu pewnosci.
  Sufit z D4/D5/D7 dotyczyl ALGORYTMOW na plaskiej decyzji 10-way. E2 zmienia
  STRUKTURE decyzji: najpierw grupa (grubo = latwo), potem klasa w grupie.

LEKCJA ZE SMOKE'A v1 (06.07.2026, wazna):
  Wersja 1 uczyla backbone w fazie 1 na etykietach GRUP (4-way CE). Wynik:
  routing grupowy 99.4%, ale ORACLE zapadl sie do 68% -- gruboziarnisty
  trening NISZCZY drobnoziarniste cechy (backbone rozdziela grupy i gubi
  informacje wewnatrzgrupowa; zamrozone cechy nie niosa tego, co pod ma
  rozrozniac). Obserwacja odnotowana w DROGA_E_NOTATKI.md.

KONSTRUKCJA v2 (ta wersja) -- trening TROJFAZOWY:
  Faza 1: backbone + glowica KLASOWA (10-way CE) -- DOKLADNIE jak plaski v2.
          Reprezentacja identyczna z baseline'em => test czysto struktury.
  Faza 2: backbone zamrozony; glowica GRUPOWA (nowe emb+prototypy) uczy sie
          etykiet grup na tych samych cechach.
  Faza 3: router zamrozony; pody grup (10-way, pelna glowica -- moga ratowac
          zle zroutowane) na probkach z REALNEGO routingu grupowego.
  Inferencja: backbone -> glowica grupowa -> pod grupy. Glowica klasowa
  NIE jest uzywana w inferencji (koszt MAC bez zmian, uczciwie).
"""
import torch
import torch.nn as nn

from mars_v2 import N_IN, N_CLASSES
from mars_v2_cnn import MarsV2CNNSystem


class MarsV2HierSystem(MarsV2CNNSystem):
    """
    v2+CNN z routingiem do grup klas.
      groups: lista list, np. [[0,2,3,4,6],[5,7,9],[1],[8]] -- partycja 0..9.
    Dziedziczone routing_head/protos (rozmiar n_grup) = glowica GRUPOWA.
    Dodatkowa glowica KLASOWA (cls_head/cls_protos) sluzy WYLACZNIE do
    treningu backbone'u w fazie 1 (10-way, jak plaski v2).
    """
    def __init__(self, groups, n_in=N_IN, backbone_hidden=128,
                 emb_dim=32, pod_hidden=24, n_out=N_CLASSES,
                 channels=(32, 64)):
        flat = sorted(c for g in groups for c in g)
        assert flat == list(range(N_CLASSES)), f"grupy nie sa partycja: {flat}"
        super().__init__(backbone_hidden=backbone_hidden,
                         n_pods=len(groups), emb_dim=emb_dim,
                         pod_hidden=pod_hidden, n_out=n_out,
                         channels=channels)
        self.groups = [list(g) for g in groups]
        c2g = torch.zeros(N_CLASSES, dtype=torch.long)
        for gi, g in enumerate(groups):
            for c in g:
                c2g[c] = gi
        self.register_buffer("class_to_group", c2g)

        # Glowica klasowa (tylko trening fazy 1; nieobecna w inferencji).
        self.cls_head = nn.Linear(backbone_hidden, emb_dim)
        self.cls_protos = nn.Parameter(torch.randn(N_CLASSES, emb_dim) * 0.1)

    def group_labels(self, y):
        """Etykiety grupowe dla etykiet klasowych. [B] -> [B]."""
        return self.class_to_group[y]

    def cls_logits(self, features):
        """Logity klasowe 10-way (jak route_logits plaskiego v2)."""
        emb = self.cls_head(features)
        dists = torch.cdist(emb.unsqueeze(0),
                            self.cls_protos.unsqueeze(0)).squeeze(0)
        return -dists


def train_phased_hier(model, Xtr, ytr, epochs=30, lr=0.001, batch=512,
                      device="cpu"):
    """
    Trening trojfazowy (patrz naglowek pliku).
    Faza 1 = lustro fazy 1 plaskiego train_phased (10-way CE) -> te same cechy.
    """
    crit = nn.CrossEntropyLoss()
    gtr = model.group_labels(ytr)

    # --- Faza 1: backbone + glowica klasowa (10-way, jak plaski v2) ---
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    opt1 = torch.optim.Adam(
        list(model.backbone.parameters())
        + list(model.cls_head.parameters()) + [model.cls_protos], lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            feats = model.backbone(Xtr[idx])
            loss = crit(model.cls_logits(feats), ytr[idx])
            opt1.zero_grad(); loss.backward(); opt1.step()

    # --- Faza 2: glowica grupowa na ZAMROZONYCH cechach ---
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt2 = torch.optim.Adam(
        list(model.routing_head.parameters()) + [model.protos], lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            with torch.no_grad():
                feats = model.backbone(Xtr[idx])
            loss = crit(model.route_logits(feats), gtr[idx])
            opt2.zero_grad(); loss.backward(); opt2.step()

    # --- Faza 3: pody grup na realnym routingu grupowym ---
    for p in [model.routing_head.weight, model.routing_head.bias, model.protos]:
        p.requires_grad = False
    opt3 = torch.optim.Adam(
        [model.pod_W1, model.pod_b1, model.pod_W2, model.pod_b2], lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            with torch.no_grad():
                feats = model.backbone(Xtr[idx])
                pod_ids = model.route(feats)          # realny routing grupowy
            out = model.pod_forward(feats, pod_ids)
            loss = crit(out, ytr[idx])
            opt3.zero_grad(); loss.backward(); opt3.step()

    for p in model.parameters():
        p.requires_grad = True
    return model


def evaluate_hier(model, Xte, yte):
    """
    Zwraca (group_routing_acc, system_acc, oracle_acc).
    group_routing_acc -- router trafia w GRUPE prawdziwej klasy.
    oracle_acc -- sufit: probka idzie do wlasciwej grupy z definicji.
    """
    model.eval()
    with torch.no_grad():
        feats = model.features(Xte)
        gte = model.group_labels(yte)
        ids = model.route(feats)
        group_routing_acc = (ids == gte).float().mean().item()
        system_acc = (model.pod_forward(feats, ids).argmax(1)
                      == yte).float().mean().item()
        oracle_acc = (model.pod_forward(feats, gte).argmax(1)
                      == yte).float().mean().item()
    return group_routing_acc, system_acc, oracle_acc


if __name__ == "__main__":
    # Smoke: partycja, ksztalty, MAC.
    groups = [[0, 2, 3, 4, 6], [5, 7, 9], [1], [8]]
    torch.manual_seed(0)
    m = MarsV2HierSystem(groups, backbone_hidden=128, emb_dim=32,
                         pod_hidden=24, channels=(32, 64))
    x = torch.randn(8, N_IN)
    y = torch.randint(0, 10, (8,))
    feats = m.features(x)
    assert feats.shape == (8, 128)
    assert m.route(feats).max() < len(groups)
    assert m.cls_logits(feats).shape == (8, 10)
    assert m.pod_forward(feats, m.route(feats)).shape == (8, 10)
    assert m.group_labels(y).shape == (8,)
    mac = m.mac_per_sample_top1()
    print("Smoke test OK.")
    print(f"  n_grup={len(groups)} | MAC top-1: {mac['total_top1']:,} "
          f"(routing={mac['routing']:,} -- glowica klasowa poza inferencja)")
