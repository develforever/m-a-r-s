"""
mars_cl.py -- F1: MARS-CL, modularny system continual learning.

ZASADA KONSTRUKCYJNA (prawo D5 podniesione do rangi architektury):
  Wspolna reprezentacja (backbone + projekcja embeddingu) jest NIETYKALNA
  po inicjalizacji. Cala plastycznosc zyje w prototypach i podach --
  a te sa per klasa i po nauczeniu ZAMROZONE. Catastrophic forgetting
  to sekwencyjna wersja zjawiska z D5/E2-v1 (dotkniecie wspolnych cech
  psuje skalibrowane glowice) -- wiec go konstrukcyjnie wykluczamy.

ARCHITEKTURA:
  backbone S2 (zamrozony) -> proj 128->32 (zamrozona) -> emb
  routing class-IL: najblizszy prototyp sposrod WSZYSTKICH widzianych klas
  pod klasy c: 128->24->10 (pelna glowica, jak v2), trenowany na probkach
  realnie routowanych do c w ramach zadania, potem zamrozony.

ZRODLO REPREZENTACJI (backbone_source -- pre-rejestrowane warianty F1):
  "task0"  (F1a): backbone+proj uczone na klasach zadania 0, potem freeze.
  "task01" (F1c): jw. na klasach zadan 0+1 -- DIAGNOSTYKA transferu cech
           (uzywa danych przyszlego zadania; nie jest to uczciwy CL,
           sluzy TYLKO do pomiaru, ile kosztuje ubostwo cech z task0).
  "random" (F1d): backbone NIGDY nie trenowany (reservoir / random features)
           -- rozbraja ryzyko transferu z definicji.

PROTOTYPY (proto_mode):
  "mean"    : prototyp = srednia embeddingow klasy (NCM; zero parametrow,
              zero interferencji, one-shot -- naturalnie continual).
  "learned" : prototypy nowych klas uczone CE (-dist, jak v2) przy
              ZAMROZONYCH starych prototypach.
"""
import torch
import torch.nn as nn

from mars_v2_slim import SlimCNNBackbone
from cl_common import BB_H, S2

EMB, POD_H, N_OUT = 32, 24, 10


class MarsCLSystem(nn.Module):
    def __init__(self, backbone_source="task0", proto_mode="mean",
                 emb_dim=EMB, pod_hidden=POD_H, channels=None,
                 backbone_module=None):
        super().__init__()
        # "ae0" (F2): backbone inicjalizowany autoenkoderem na OBRAZACH
        # zadania 0 -- bez etykiet (uczciwy CL), ogolnosc bez waskiej
        # supervizji. Projekcja emb zostaje LOSOWA (odnotowane; JL).
        # backbone_module (F4): wstrzykniety backbone (np. CIFAR) --
        # kontrakt: [B, n_in] -> [B, BB_H].
        assert backbone_source in ("task0", "task01", "random", "ae0")
        assert proto_mode in ("mean", "learned")
        self.backbone_source = backbone_source
        self.proto_mode = proto_mode
        self.emb_dim = emb_dim
        self.pod_hidden = pod_hidden
        self.channels = tuple(channels) if channels else S2["channels"]

        self.backbone = (backbone_module if backbone_module is not None
                         else SlimCNNBackbone(backbone_hidden=BB_H,
                                              channels=self.channels,
                                              downsample="maxpool",
                                              depthwise=False))
        self.proj = nn.Linear(BB_H, emb_dim)

        # Stan rosnacy (zwykle tensory, nie Parameters -- po nauce zamrozone):
        self.seen_classes = []            # kolejnosc dodawania
        self.protos = {}                  # klasa -> [emb_dim]
        self.pods = {}                    # klasa -> dict W1,b1,W2,b2

    def feats_batched(self, X, batch=2048):
        """Cechy backbone liczone porcjami (CIFAR: pelny forward na 12k
        probek 3x32x32 przekracza VRAM 1050 Ti)."""
        with torch.no_grad():
            return torch.cat([self.backbone(X[s:s + batch])
                              for s in range(0, len(X), batch)])

    # ------------------------------------------------------------ reprezentacja
    def embed_from_feats(self, feats):
        """Punkt przeciazenia dla wariantow (G1: normalizacja kosinusowa)."""
        return self.proj(feats)

    def embed(self, x):
        with torch.no_grad():
            return self.embed_from_feats(self.feats_batched(x))

    def init_representation(self, task_data, epochs, lr, device):
        """
        Jednorazowa inicjalizacja backbone+proj wg backbone_source,
        potem TRWALE zamrozenie. Dla "random": nic nie trenujemy.
        Dla "ae0": autoenkoder na obrazach zadania 0 (BEZ etykiet).
        """
        if self.backbone_source == "ae0":
            X = task_data[0]["Xtr"]
            dec = nn.Linear(BB_H, 784, device=device)
            crit = nn.MSELoss()
            opt = torch.optim.Adam(
                list(self.backbone.parameters()) + list(dec.parameters()),
                lr=lr)
            self.train()
            for _ in range(epochs):
                perm = torch.randperm(len(X), device=device)
                for s in range(0, len(X), 512):
                    idx = perm[s:s + 512]
                    x = X[idx]
                    loss = crit(dec(self.backbone(x)), x)
                    opt.zero_grad(); loss.backward(); opt.step()
            # dekoder odrzucany; proj zostaje losowa (zamrozona nizej)
        elif self.backbone_source != "random":
            n_tasks = 1 if self.backbone_source == "task0" else 2
            X = torch.cat([task_data[t]["Xtr"] for t in range(n_tasks)])
            y = torch.cat([task_data[t]["ytr"] for t in range(n_tasks)])
            classes = sorted(set(
                c for t in range(n_tasks) for c in task_data[t]["classes"]))
            # lustro fazy 1 v2: CE na -dist do tymczasowych prototypow
            tmp_protos = nn.Parameter(
                torch.randn(len(classes), self.emb_dim, device=device) * 0.1)
            c2i = {c: i for i, c in enumerate(classes)}
            yi = torch.tensor([c2i[int(c)] for c in y.tolist()], device=device)
            crit = nn.CrossEntropyLoss()
            opt = torch.optim.Adam(
                list(self.backbone.parameters())
                + list(self.proj.parameters()) + [tmp_protos], lr=lr)
            self.train()
            for _ in range(epochs):
                perm = torch.randperm(len(X), device=device)
                for s in range(0, len(X), 512):
                    idx = perm[s:s + 512]
                    emb = self.proj(self.backbone(X[idx]))
                    d = torch.cdist(emb.unsqueeze(0),
                                    tmp_protos.unsqueeze(0)).squeeze(0)
                    loss = crit(-d, yi[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            # tmp_protos sa ODRZUCANE -- prototypy klas powstaja w learn_task
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # ------------------------------------------------------------ routing
    def _proto_stack(self):
        return torch.stack([self.protos[c] for c in self.seen_classes])

    def route(self, emb):
        """Najblizszy prototyp sposrod widzianych klas -> etykieta klasy [B]."""
        d = torch.cdist(emb.unsqueeze(0),
                        self._proto_stack().unsqueeze(0)).squeeze(0)
        idx = d.argmin(dim=1)
        lut = torch.tensor(self.seen_classes, device=emb.device)
        return lut[idx]

    # ------------------------------------------------------------ pody
    def _pod_forward_class(self, feats, c):
        p = self.pods[c]
        h = torch.relu(feats @ p["W1"] + p["b1"])
        return h @ p["W2"] + p["b2"]

    def forward(self, x):
        """Logity [B, 10]: backbone -> routing -> pod wybranej klasy."""
        with torch.no_grad():
            feats = self.backbone(x)
            emb = self.embed_from_feats(feats)
            routed = self.route(emb)
            out = torch.zeros(len(x), N_OUT, device=x.device)
            for c in self.seen_classes:
                m = routed == c
                if m.any():
                    out[m] = self._pod_forward_class(feats[m], c)
        return out

    # ------------------------------------------------------------ nauka zadania
    def learn_task(self, td, epochs, lr, device):
        """
        Dodaje klasy zadania: prototypy (mean/learned) + pody trenowane na
        probkach REALNIE routowanych (w obrebie zadania). Stare prototypy
        i pody NIETYKANE. Zwraca nic; stan rosnie w miejscu.
        """
        classes = td["classes"]
        X, y = td["Xtr"], td["ytr"]
        emb = self.embed(X)   # batched (feats_batched w srodku)
        feats = None  # liczone nizej raz dla podow

        # --- prototypy nowych klas ---
        if self.proto_mode == "mean":
            for c in classes:
                self.protos[c] = emb[y == c].mean(dim=0).detach()
        else:  # learned: CE po WSZYSTKICH widzianych, stare protos = staly
            new_p = nn.Parameter(torch.stack(
                [emb[y == c].mean(dim=0) for c in classes]).clone())
            old = ([self.protos[c] for c in self.seen_classes]
                   if self.seen_classes else [])
            all_classes = self.seen_classes + list(classes)
            c2i = {c: i for i, c in enumerate(all_classes)}
            yi = torch.tensor([c2i[int(v)] for v in y.tolist()], device=device)
            crit = nn.CrossEntropyLoss()
            opt = torch.optim.Adam([new_p], lr=lr)
            for _ in range(epochs):
                perm = torch.randperm(len(X), device=device)
                for s in range(0, len(X), 512):
                    idx = perm[s:s + 512]
                    stack = (torch.cat([torch.stack(old), new_p])
                             if old else new_p)
                    d = torch.cdist(emb[idx].unsqueeze(0),
                                    stack.unsqueeze(0)).squeeze(0)
                    loss = crit(-d, yi[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            for i, c in enumerate(classes):
                self.protos[c] = new_p.detach()[i].clone()

        self.seen_classes = self.seen_classes + list(classes)
        self._train_pods(classes, X, y, epochs, lr, device)

    def _train_pods(self, classes, X, y, epochs, lr, device):
        """Pody nowych klas: trening na realnym routingu w zadaniu.
        Wydzielone, zeby warianty (np. G1 semantic) mogly reuzyc."""
        with torch.no_grad():
            feats = self.feats_batched(X)
            routed = self.route(self.embed_from_feats(feats))
        crit = nn.CrossEntropyLoss()
        for c in classes:
            m = routed == c
            if m.sum() < 10:      # degeneracja routingu -- fallback: wlasna klasa
                m = y == c
            Xi, yi_c = feats[m], y[m]
            W1 = torch.randn(BB_H, self.pod_hidden, device=device) * 0.01
            b1 = torch.zeros(self.pod_hidden, device=device)
            W2 = torch.randn(self.pod_hidden, N_OUT, device=device) * 0.01
            b2 = torch.zeros(N_OUT, device=device)
            for t_ in (W1, b1, W2, b2):
                t_.requires_grad = True
            opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
            for _ in range(epochs):
                perm = torch.randperm(len(Xi), device=device)
                for s in range(0, len(Xi), 512):
                    idx = perm[s:s + 512]
                    h = torch.relu(Xi[idx] @ W1 + b1)
                    loss = crit(h @ W2 + b2, yi_c[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            self.pods[c] = {"W1": W1.detach(), "b1": b1.detach(),
                            "W2": W2.detach(), "b2": b2.detach()}

    # ------------------------------------------------------------ koszt
    def mac_per_sample(self):
        """MAC inferencji przy T widzianych klasach (top-1 pod)."""
        if hasattr(self.backbone, "mac_backbone"):
            bb = self.backbone.mac_backbone()   # wstrzykniety (np. CIFAR)
        else:
            c1, c2 = self.channels
            bb = (1 * c1 * 9 * 28 * 28 + c1 * c2 * 9 * 14 * 14
                  + 49 * c2 * BB_H)
        proj = BB_H * self.emb_dim
        routing = self.emb_dim * max(len(self.seen_classes), 1)
        pod = BB_H * self.pod_hidden + self.pod_hidden * N_OUT
        return {"backbone": bb, "proj": proj, "routing": routing, "pod": pod,
                "total": bb + proj + routing + pod}


if __name__ == "__main__":
    # Smoke: 2 zadania na losowych danych, ksztalty + wzrost stanu + MAC.
    device = "cpu"
    torch.manual_seed(0)
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    td0 = {"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200]}
    td1 = {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:]}
    m = MarsCLSystem(backbone_source="random", proto_mode="mean")
    m.init_representation([td0, td1], epochs=0, lr=1e-3, device=device)
    m.learn_task(td0, epochs=1, lr=1e-3, device=device)
    assert m.seen_classes == [0, 1] and set(m.pods) == {0, 1}
    mac1 = m.mac_per_sample()["total"]
    m.learn_task(td1, epochs=1, lr=1e-3, device=device)
    assert m.seen_classes == [0, 1, 2, 3]
    out = m.forward(X[:32])
    assert out.shape == (32, 10)
    mac2 = m.mac_per_sample()["total"]
    print(f"Smoke test OK. MAC T=2 klasy: {mac1:,} -> T=4 klasy: {mac2:,} "
          f"(wzrost {mac2-mac1} = tylko routing)")
