"""
mars_collective.py -- Droga I: kolektywne uczenie przez wymiane snow
(DROGA_I_PLAN.md).

NOWY plik -- kod v0.5 pozostaje NIETKNIETY; podklasa na branchu droga-i.

Protokol: agenci maja TEN SAM zamrozony losowy backbone (wspolny seed)
i ta sama przestrzen slow. Wiadomosc o klasie c to WYLACZNIE statystyki
spike-and-slab + licznosc proby:

    payload(c) = {p:[k,D], mean:[k,D], var:[k,D], w:[k], n:int}

(~24.1 KB dla k=16, D=128, fp32). Zero obrazow, gradientow, wag.

adopt_classes() = WIERNA kopia MarsCLSemanticF3.learn_task, z jedna
roznica: cechy zadania sa WYSNIONE z payloadu zamiast policzone
z obrazow (stats.update pominiete -- statystyki SA wiadomoscia,
zostaja w pamieci odbiorcy i chronia klase przeszczepiona w przyszlych
zadaniach dokladnie tak, jak wlasne).

Fuzja (I2): fuse_payloads_cat -- unia komponentow [2k] z wagami
wazonymi licznosciami (bezstratna, 2x pamiec); redream_payload --
sen z payloadu + ponowny k-means do k (kompresja z powrotem).
"""
import torch
import torch.nn.functional as F

from mars_cl_f3 import _train_pods_with_negatives
from mars_cl_j import FeatureStatsKSparse, MarsCLSemanticF3J


class MarsCollective(MarsCLSemanticF3J):
    """F3J (sen sparse) + eksport/adopcja klas przez payload statystyk."""

    # ------------------------------------------------------------ eksport
    def export_class_stats(self, c, n_samples):
        """Payload klasy c (wiadomosc do innych agentow)."""
        s = self.stats
        return {"p": s.p[c].detach().cpu(),
                "mean": s.mean[c].detach().cpu(),
                "var": s.var[c].detach().cpu(),
                "w": s.w[c].detach().cpu(),
                "n": int(n_samples)}

    # ------------------------------------------------------------ adopcja
    def adopt_classes(self, classes, payloads, epochs, lr, device,
                      n_dream=6000):
        """
        Nauka klas z payloadow: sciezka learn_task (projekcja + pody)
        na cechach WYSNIONYCH z otrzymanych statystyk.
        """
        for c in classes:
            pl = payloads[c]
            self.stats.p[c] = pl["p"].to(device)
            self.stats.mean[c] = pl["mean"].to(device)
            self.stats.var[c] = pl["var"].to(device)
            self.stats.w[c] = pl["w"].to(device)
        feats = torch.cat([self.stats.sample(c, n_dream, device)
                           for c in classes])
        y = torch.cat([torch.full((n_dream,), c, dtype=torch.long,
                                  device=device) for c in classes])
        old = list(self.seen_classes)
        # dalej DOKLADNIE jak MarsCLSemanticF3.learn_task (bez
        # stats.update -- payload juz w pamieci):
        self._fit_proj_feats(feats, y, old, old + list(classes),
                             epochs, lr, device)
        neg_f, neg_y = self.stats.replay_batch(old, self.replay_per_class,
                                               device)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        _train_pods_with_negatives(self, classes, feats, y, routed,
                                   neg_f, neg_y, epochs, lr, device)


# ------------------------------------------------------------------ fuzja
def fuse_payloads_cat(pa, pb):
    """I2: unia komponentow [2k]; wagi wazone licznosciami (bezstratna)."""
    n = pa["n"] + pb["n"]
    return {"p": torch.cat([pa["p"], pb["p"]]),
            "mean": torch.cat([pa["mean"], pb["mean"]]),
            "var": torch.cat([pa["var"], pb["var"]]),
            "w": torch.cat([pa["w"] * pa["n"], pb["w"] * pb["n"]]) / n,
            "n": n}


def redream_payload(payload, c, k, n_dream, device):
    """
    I2: kompresja payloadu (np. po fuzji [2k]) z powrotem do k
    komponentow: sen n_dream probek -> ponowny k-means (stratna).
    """
    src = FeatureStatsKSparse(k=k)
    src.p[c], src.mean[c] = payload["p"].to(device), payload["mean"].to(device)
    src.var[c], src.w[c] = payload["var"].to(device), payload["w"].to(device)
    dreams = src.sample(c, n_dream, device)
    dst = FeatureStatsKSparse(k=k)
    dst.update(dreams, torch.full((n_dream,), c, dtype=torch.long,
                                  device=device), [c])
    return {"p": dst.p[c].cpu(), "mean": dst.mean[c].cpu(),
            "var": dst.var[c].cpu(), "w": dst.w[c].cpu(),
            "n": payload["n"]}


if __name__ == "__main__":
    # Smoke (CPU, syntetyczne): A uczy klasy 2,3 -> payload -> B adoptuje.
    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    td0 = {"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200]}
    td1 = {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:]}

    def build():
        torch.manual_seed(7)     # wspolny seed = wspolny backbone
        return MarsCollective(wv, dream_model="sparse", stats_k=4,
                              replay_per_class=16)

    A, B = build(), build()
    assert torch.allclose(next(A.backbone.parameters()),
                          next(B.backbone.parameters())), \
        "rozne backbone'y przy wspolnym seedzie"

    A.init_representation([td1], epochs=1, lr=1e-3, device="cpu")
    A.learn_task(td1, epochs=1, lr=1e-3, device="cpu")
    payloads = {c: A.export_class_stats(c, 100) for c in (2, 3)}
    assert payloads[2]["p"].shape == (4, 128) and payloads[2]["n"] == 100

    B.init_representation([td0], epochs=1, lr=1e-3, device="cpu")
    B.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    B.adopt_classes([2, 3], payloads, epochs=1, lr=1e-3, device="cpu",
                    n_dream=64)
    assert B.seen_classes == [0, 1, 2, 3]
    assert 2 in B.pods and 3 in B.pods and 2 in B.stats.p
    assert B.forward(X[:16]).shape == (16, 10)

    fused = fuse_payloads_cat(payloads[2], payloads[3])   # test ksztaltu
    assert fused["p"].shape == (8, 128) and fused["n"] == 200
    red = redream_payload(fused, 2, 4, 256, "cpu")
    assert red["p"].shape == (4, 128)
    print("Smoke OK: eksport -> adopcja -> forward; fuzja cat[2k] "
          "i redream[k] dzialaja.")
