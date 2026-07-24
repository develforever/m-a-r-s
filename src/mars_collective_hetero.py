"""
mars_collective_hetero.py -- Droga R: kolektyw HETEROGENICZNY
(DROGA_R_PLAN.md). Protokol representation-agnostic.

NOWY plik -- kod serii I/L (mars_collective.py, mars_cl_*.py) NIETKNIETY.
Branch: droga-r.

Problem (z kodu I): payload klasy to statystyki spike-and-slab w
PRZESTRZENI CECH nadawcy (D=BB_H). Homogenicznie feature_A = feature_B
(wspolny seed backbone'u) -> sen A jest poprawnym materialem dla B.
Heterogenicznie feature_A != feature_B -> sen A jest szumem dla proj_B.

Mechanizm R1 (kotwica-interlingua + dekoder per-agent):
  1. Nadawca liczy statystyki w PRZESTRZENI KOTWIC (po proj), nie cech:
     rozklad Gaussa (mean/var) po embeddingu anchorowym klasy. Wspolny
     uklad odniesienia z definicji (obie projekcje celuja w te same
     wektory slow GloVe).
  2. Odbiorca uczy dekodera `anchor -> feature_B` (glowa ReLU -> cechy
     nieujemne) WYLACZNIE na wlasnych widzianych klasach (pary
     (embed_from_feats(f), f)). Prywatna, przyblizona inwersja projekcji.
  3. Re-materializacja: sampluj Gaussa anchorowego -> dekoduj do
     pseudo-cech feature_B -> zbuduj feature-payload (FeatureStatsKSparse)
     -> WYWOLAJ ISTNIEJACE adopt_classes bez zmian.

R0 (podloga, kontrola "czy sama kotwica wystarczy"): adopcja samym
prototypem (proto_c = word_vec_c) + staly pod ufajacy kotwicy; bez
dekodera i bez douczenia projekcji. Mierzy czyste routowanie anchorowe
(przewidziana porazka wg G1: projekcja nietrenowana przyciaga niewidziane
do znanych slow, ZS 5.9% < 10%).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_cl_j import FeatureStatsKSparse
from mars_collective import MarsCollective

AnchorPayload = Dict[str, object]   # {"mean":[E], "var":[E], "n":int}
FeaturePayload = Dict[str, object]  # {"p","mean","var","w":[k,..], "n":int}
_R0_POD_LOGIT = 30.0                 # duzy logit "ufam kotwicy" dla podlogi


class AnchorDecoder(nn.Module):
    """Dekoder anchor(E) -> feature_B(D); glowa ReLU (cechy nieujemne,
    jak po backbone ReLU). Maly MLP, trenowany per-agent na jego wlasnych
    klasach."""

    def __init__(self, emb_dim: int, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(emb))


class MarsCollectiveHetero(MarsCollective):
    """MarsCollective + eksport anchorowy, dekoder anchor->feature,
    adopcja heterogeniczna. Rdzen (export_class_stats, adopt_classes)
    nietkniety -- adopt_classes_hetero uzywa go wewnetrznie."""

    def __init__(self, *args, dec_hidden: int = 256, **kw):
        super().__init__(*args, **kw)
        self.dec_hidden = dec_hidden
        self.decoder: AnchorDecoder | None = None

    # --------------------------------------------------- nadawca (anchor)
    def export_anchor_payload(self, c: int, n_dream: int,
                              device: str) -> AnchorPayload:
        """Payload klasy c w przestrzeni kotwic: Gauss (mean/var) po
        embeddingu anchorowym snionych cech nadawcy. Zero cech nadawcy
        w sieci -- tylko statystyka w wymiarze slow."""
        if c not in self.stats.p:
            raise KeyError(f"eksport klasy {c}: brak statystyk cech "
                           f"(agent nie nauczyl sie klasy)")
        with torch.no_grad():
            feats = self.stats.sample(c, n_dream, device)
            emb = self.embed_from_feats(feats)            # [n, E], znormaliz.
            mean = emb.mean(dim=0)
            var = emb.var(dim=0, unbiased=True).clamp_min(1e-8)
        return {"mean": mean.detach().cpu(), "var": var.detach().cpu(),
                "n": int(n_dream)}

    # --------------------------------------------------- dekoder (odbiorca)
    def train_decoder_on(self, emb: torch.Tensor, feats: torch.Tensor,
                         epochs: int, lr: float, device: str,
                         batch: int = 512) -> float:
        """Uczy dekoder MLP na PODANYCH parach (emb, feats). Zwraca
        koncowy MSE. (Uzywane tez przez kontrole oracle: pary z 10
        klas realnych.)"""
        emb_dim, feat_dim = emb.shape[1], feats.shape[1]
        self.decoder = AnchorDecoder(emb_dim, feat_dim,
                                     self.dec_hidden).to(device)
        opt = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        crit = nn.MSELoss()
        last = float("nan")
        for _ in range(epochs):
            perm = torch.randperm(len(feats), device=device)
            for s in range(0, len(feats), batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                loss = crit(self.decoder(emb[idx]), feats[idx])
                loss.backward()
                opt.step()
                last = float(loss.detach())
        for p in self.decoder.parameters():
            p.requires_grad = False
        return last

    def train_decoder(self, own_classes: Sequence[int], n_per: int,
                      epochs: int, lr: float, device: str,
                      batch: int = 512) -> float:
        """Uczy anchor->feature_B na WLASNYCH klasach odbiorcy (pary
        (embed_from_feats(f), f) z sennych cech). Zwraca koncowy MSE."""
        own = [c for c in own_classes if c in self.stats.p]
        if not own:
            raise ValueError("train_decoder: brak wlasnych klas ze "
                             "statystykami cech")
        with torch.no_grad():
            feats = torch.cat([self.stats.sample(c, n_per, device)
                               for c in own])
            emb = self.embed_from_feats(feats)
        return self.train_decoder_on(emb, feats, epochs, lr, device, batch)

    @staticmethod
    def _sample_anchor(pl: AnchorPayload, n_dream: int,
                       device: str) -> torch.Tensor:
        """Sampluj Gaussa diag z payloadu anchorowego -> [n_dream, E]."""
        mean = pl["mean"].to(device)
        var = pl["var"].to(device)
        return mean + var.sqrt() * torch.randn(n_dream, mean.shape[0],
                                               device=device)

    def _materialize_payload(self, c: int, pl: AnchorPayload,
                             materialize_fn, n_dream: int, stats_k: int,
                             device: str) -> FeaturePayload:
        """Sampluj Gaussa anchorowego -> materializuj cechy (materialize_fn:
        emb->feats, >=0) -> feature-payload spike-and-slab k komponentow."""
        with torch.no_grad():
            z = self._sample_anchor(pl, n_dream, device)
            pseudo = materialize_fn(z)                      # [n, D], >=0
        tmp = FeatureStatsKSparse(k=stats_k)
        y = torch.full((n_dream,), c, dtype=torch.long, device=device)
        tmp.update(pseudo, y, [c])
        return {"p": tmp.p[c].detach().cpu(), "mean": tmp.mean[c].detach().cpu(),
                "var": tmp.var[c].detach().cpu(), "w": tmp.w[c].detach().cpu(),
                "n": int(pl["n"])}

    def _adopt_via(self, classes: Sequence[int],
                   anchor_payloads: Dict[int, AnchorPayload], materialize_fn,
                   epochs: int, lr: float, device: str, n_dream: int,
                   stats_k: int) -> None:
        """Wspolna sciezka: materializuj feature-payloady i wywolaj
        istniejace adopt_classes (projekcja + pody) BEZ ZMIAN."""
        feat_payloads = {c: self._materialize_payload(
            c, anchor_payloads[c], materialize_fn, n_dream, stats_k, device)
            for c in classes}
        self.adopt_classes(list(classes), feat_payloads, epochs=epochs,
                            lr=lr, device=device, n_dream=n_dream)

    # --------------------------------------------------- adopcja (dekoder MLP)
    def adopt_classes_hetero(self, classes: Sequence[int],
                             anchor_payloads: Dict[int, AnchorPayload],
                             epochs: int, lr: float, device: str,
                             n_dream: int, stats_k: int) -> None:
        """R-mild v1 / oracle: materializacja dekoderem MLP (self.decoder)."""
        if self.decoder is None:
            raise RuntimeError("brak dekodera -- najpierw train_decoder()")
        self._adopt_via(classes, anchor_payloads, self.decoder, epochs, lr,
                        device, n_dream, stats_k)

    # --------------------------------------------------- adopcja (R2: mapa cech)
    def adopt_classes_maptransform(self, classes: Sequence[int],
                                   payloads_a: Dict[int, FeaturePayload],
                                   map_fn, epochs: int, lr: float,
                                   device: str, n_dream: int,
                                   stats_k: int) -> None:
        """R2: payload to statystyki cech w H_A (jak w I). Śnij próbki w
        H_A → `map_fn` (Ω Procrustesa lub mapa liniowa: H_A→H_B, >=0) →
        zbuduj statystyki H_B → istniejące `adopt_classes` BEZ ZMIAN."""
        feat_payloads_b: Dict[int, FeaturePayload] = {}
        for c in classes:
            pa = payloads_a[c]
            k = pa["p"].shape[0]
            src = FeatureStatsKSparse(k=k)
            src.p[c] = pa["p"].to(device)
            src.mean[c] = pa["mean"].to(device)
            src.var[c] = pa["var"].to(device)
            src.w[c] = pa["w"].to(device)
            with torch.no_grad():
                h_a = src.sample(c, n_dream, device)       # [n, D_A]
                h_b = map_fn(h_a)                          # [n, D_B], >=0
            tmp = FeatureStatsKSparse(k=stats_k)
            y = torch.full((n_dream,), c, dtype=torch.long, device=device)
            tmp.update(h_b, y, [c])
            feat_payloads_b[c] = {
                "p": tmp.p[c].detach().cpu(), "mean": tmp.mean[c].detach().cpu(),
                "var": tmp.var[c].detach().cpu(), "w": tmp.w[c].detach().cpu(),
                "n": int(pa["n"])}
        self.adopt_classes(list(classes), feat_payloads_b, epochs=epochs,
                           lr=lr, device=device, n_dream=n_dream)

    # --------------------------------------------------- adopcja (translator)
    def adopt_classes_translate(self, classes: Sequence[int],
                                anchor_payloads: Dict[int, AnchorPayload],
                                translator, epochs: int, lr: float,
                                device: str, n_dream: int,
                                stats_k: int) -> None:
        """R1b: materializacja uregularyzowanym translatorem
        (Ridge/KernelRidge, .predict: emb->feats >=0)."""
        self._adopt_via(classes, anchor_payloads, translator.predict, epochs,
                        lr, device, n_dream, stats_k)

    # --------------------------------------------------- adopcja R0 (floor)
    def adopt_classes_anchor_only(self, classes: Sequence[int],
                                  device: str) -> None:
        """Podloga: proto_c = word_vec_c + staly pod ufajacy kotwicy;
        bez re-materializacji, bez douczenia projekcji. Mierzy czyste
        routowanie anchorowe (nie moze crashowac forward)."""
        if not self.pods:
            raise RuntimeError("adopt_classes_anchor_only: brak wlasnego "
                               "poda-wzorca (naucz najpierw task0)")
        ref = next(iter(self.pods.values()))
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
            b2 = torch.zeros_like(ref["b2"])
            b2[c] = _R0_POD_LOGIT
            self.pods[c] = {"W1": torch.zeros_like(ref["W1"]),
                            "b1": torch.zeros_like(ref["b1"]),
                            "W2": torch.zeros_like(ref["W2"]),
                            "b2": b2}
        self.seen_classes = self.seen_classes + list(classes)


# ==================================================== R-hard: funkcje pomocnicze
# Droga R-hard (DROGA_R_HARD_PLAN.md): transfer między RÓŻNYMI reprezentacjami
# (różne fronty sensoryczne, różne wymiary cech). Rdzeń I/L i `adopt_classes`
# NIETKNIĘTE -- poniższe funkcje działają OBOK niego:
#   - nadawca o wymiarze != 128 (foundation-512) eksportuje payload k-sparse
#     WPROST z cech, z pominięciem proj/podów (rdzeń hardkoduje BB_H=128),
#   - kalibracja per-próbka przez DWA fronty na tych samych obrazach,
#   - prostokątną mapę H_A(D_A)->H_B(D_B) realizuje RidgeTranslator
#     (już prostokątny: W=[D_A+1, D_B], zamknięta forma) -- bez zmian.


def _identity(t: torch.Tensor) -> torch.Tensor:
    """Front tożsamościowy: cechy foundation 512-d = surowy cache (bez redukcji)."""
    return t


def feature_payload_from_feats(feats: torch.Tensor, c: int, stats_k: int,
                               device: str, n: int | None = None
                               ) -> FeaturePayload:
    """Payload k-sparse (format jak w I: p/mean/var/w/n) WPROST z cech klasy c
    w przestrzeni NADAWCY -- z pominięciem proj/podów (rdzeń nietknięty). Do
    R-hard, gdy nadawca ma wymiar != 128 (foundation-512) i nie może być
    pełnym agentem. feats: [N, D] (>=0, po ReLU backbone'u/ResNet)."""
    if feats.ndim != 2:
        raise ValueError("feature_payload_from_feats: feats musi być [N, D]")
    if len(feats) == 0:
        raise ValueError(f"feature_payload_from_feats: brak próbek klasy {c}")
    feats = feats.to(device)
    tmp = FeatureStatsKSparse(k=stats_k)
    y = torch.full((len(feats),), c, dtype=torch.long, device=device)
    tmp.update(feats, y, [c])
    return {"p": tmp.p[c].detach().cpu(), "mean": tmp.mean[c].detach().cpu(),
            "var": tmp.var[c].detach().cpu(), "w": tmp.w[c].detach().cpu(),
            "n": int(n if n is not None else len(feats))}


def class_indices(y: torch.Tensor, c: int, n_max: int) -> torch.Tensor:
    """Pierwsze n_max indeksów próbek klasy c (spójna kolejność datasetu --
    umożliwia parowanie tych samych obrazów przez dwa fronty)."""
    return (y == c).nonzero(as_tuple=True)[0][:n_max]


@torch.no_grad()
def feats_through_front(X: torch.Tensor, front, device: str,
                        batch: int = 2048) -> torch.Tensor:
    """Cechy X przez `front` (callable: input->cechy) liczone porcjami
    (VRAM 1050 Ti). front = agent.backbone (CNN) albo _identity (cache 512)."""
    if len(X) == 0:
        raise ValueError("feats_through_front: puste X")
    return torch.cat([front(X[s:s + batch].to(device))
                      for s in range(0, len(X), batch)])


def paired_calib_feats(cal_classes: Sequence[int], n_per: int,
                       y: torch.Tensor, src_a, src_b, device: str,
                       batch: int = 2048):
    """Pary (H_A, H_B): cechy TYCH SAMYCH obrazów klas kalibracyjnych przez
    dwa fronty sensoryczne. src_a, src_b = (input_tensor, front). Wybiera te
    same indeksy klas z `y` i przepuszcza input_tensor[idx] przez odpowiedni
    front. Zwraca (H_A[N, D_A], H_B[N, D_B]) -- podstawa fitu prostokątnej
    mapy H_A->H_B."""
    idx = torch.cat([class_indices(y, c, n_per) for c in cal_classes])
    xa, fa = src_a
    xb, fb = src_b
    ha = feats_through_front(xa[idx], fa, device, batch)
    hb = feats_through_front(xb[idx], fb, device, batch)
    if len(ha) != len(hb):
        raise RuntimeError("paired_calib_feats: niespójne parowanie H_A/H_B")
    return ha, hb


if __name__ == "__main__":
    # Smoke (CPU, syntetyczne): dwa RoZNE backbone'y (rozny seed) ->
    # eksport anchorowy A -> dekoder B -> adopcja R1; oraz podloga R0.
    # Wiring only (nie prawdziwy eksperyment).
    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    td_a = {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:]}
    td_b = {"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200]}

    def build(seed: int) -> MarsCollectiveHetero:
        torch.manual_seed(seed)               # rozny seed = rozny backbone
        return MarsCollectiveHetero(wv, dream_model="sparse", stats_k=4,
                                    replay_per_class=16)

    A, B = build(7), build(9)   # HETEROGENICZNI
    assert not torch.allclose(next(A.backbone.parameters()),
                              next(B.backbone.parameters())), \
        "backbone'y powinny sie roznic (rozny seed)"

    A.init_representation([td_a], epochs=1, lr=1e-3, device="cpu")
    A.learn_task(td_a, epochs=1, lr=1e-3, device="cpu")
    anchor_pl = {c: A.export_anchor_payload(c, 128, "cpu") for c in (2, 3)}
    assert anchor_pl[2]["mean"].shape == (50,)

    B.init_representation([td_b], epochs=1, lr=1e-3, device="cpu")
    B.learn_task(td_b, epochs=1, lr=1e-3, device="cpu")
    mse = B.train_decoder([0, 1], n_per=128, epochs=2, lr=1e-3, device="cpu")
    assert mse == mse, "MSE NaN"    # nie NaN
    B.adopt_classes_hetero([2, 3], anchor_pl, epochs=1, lr=1e-3,
                           device="cpu", n_dream=64, stats_k=4)
    assert B.seen_classes == [0, 1, 2, 3]
    assert 2 in B.pods and 3 in B.pods and 2 in B.stats.p
    assert B.forward(X[:16]).shape == (16, 10)

    # R0 na swiezym odbiorcy
    C = build(11)
    C.init_representation([td_b], epochs=1, lr=1e-3, device="cpu")
    C.learn_task(td_b, epochs=1, lr=1e-3, device="cpu")
    C.adopt_classes_anchor_only([2, 3], device="cpu")
    assert C.seen_classes == [0, 1, 2, 3] and 2 in C.pods
    assert C.forward(X[:16]).shape == (16, 10)
    print(f"Smoke OK: eksport anchorowy -> dekoder (MSE {mse:.4f}) -> "
          f"adopcja R1; podloga R0 nie crashuje forward.")
