"""
mars_cl_k.py -- Droga K: zlozenia dzwigni (DROGA_K_PLAN.md).

NOWY plik -- kod v0.4 (mars_cl*.py, mars_cl_j.py, mars_cl_owm.py)
pozostaje NIETKNIETY; wszystko tu jest podklasa na branchu droga-k.

MarsCLSemanticOWMSparse: OWM (H1, mars_cl_owm) x sen spike-and-slab
(J3, mars_cl_j.FeatureStatsKSparse). Motywacja K2: sen strzeze granic
decyzyjnych (negatywy dla nowych slow), OWM geometrii starych mapowan
(rzutnik na gradientach projekcji). H1 testowal OWM wylacznie ze snem
diagonalnym sprzed serii J i wylacznie na Fashion/MNIST; zlozenie
OWM x sparse oraz OWM na CIFAR sa nietestowane.

Konstrukcja NIE konsumuje RNG poza sciezka rodzica (bufor P = eye,
statystyki bez inicjalizacji losowej) -- wagi startowe identyczne
z MarsCLSemanticF3J przy tym samym seedzie => pary per-seed z bazami
J3/J2b legalne bez re-runu (precedens J2b vs J2).
"""
import torch
import torch.nn.functional as F

from mars_cl_j import FeatureStatsKSparse
from mars_cl_owm import MarsCLSemanticOWM


class MarsCLSemanticOWMSparse(MarsCLSemanticOWM):
    """OWM na projekcji + sen sparse (spike-and-slab) w statystykach."""

    def __init__(self, word_vecs, stats_k=16, **kw):
        super().__init__(word_vecs, dream_model="diag",
                         stats_k=stats_k, **kw)
        self.stats = FeatureStatsKSparse(k=stats_k)


if __name__ == "__main__":
    # Smoke (CPU, dane syntetyczne, sekundy): OWM aktywny + sen z zerami.
    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    m = MarsCLSemanticOWMSparse(wv, owm_alpha=1.0, stats_k=4,
                                replay_per_class=16)
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    td0 = {"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200]}
    td1 = {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:]}
    m.init_representation([td0, td1], epochs=1, lr=1e-3, device="cpu")
    m.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    m.learn_task(td1, epochs=1, lr=1e-3, device="cpu")

    # 1) sen sparse: ksztalt, nieujemnosc, prawdziwe zera
    s = m.stats.sample(0, 256, "cpu")
    assert s.shape == (256, 128) and (s >= 0).all()
    frac_zero = (s == 0).float().mean().item()
    assert frac_zero > 0.01, f"sen bez zer? frac={frac_zero}"

    # 2) rzutnik OWM zaktualizowany i chroni cechy task0
    assert m._tasks_seen == 2
    assert not torch.allclose(m.P, torch.eye(m.P.shape[0])), \
        "P nie zaktualizowane"
    feats = m.feats_batched(td0["Xtr"])
    resid = (feats @ m.P).norm(dim=1) / feats.norm(dim=1).clamp_min(1e-9)

    # 3) forward
    assert m.forward(X[:16]).shape == (16, 10)
    print(f"Smoke OK. residual P@x (task0): {resid.mean():.4f} "
          f"(blisko 0 = rzutnik chroni); frakcja zer snu: {frac_zero:.2f}")
