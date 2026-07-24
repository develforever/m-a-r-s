"""
mars_translate.py -- Droga R1b: uregularyzowane translatory
anchor(E) -> feature(D) na klasach dzielonych (DROGA_R1B_PLAN.md).

NOWY plik -- rdzen I/L NIETKNIETY. Zastepuje niestabilny dekoder MLP
z R-mild v1 (nie generalizowal poza 2 klasy) gladka, uregularyzowana
interpolacja o zamknietej formie:

  RidgeTranslator        -- mapa liniowa anchor->feature z regularyzacja
                            L2 (najstabilniejsza; dla R-mild liniowa mapa
                            istnieje z konstrukcji: feature_A, feature_B
                            to liniowe obrazy tej samej 512).
  KernelRidgeTranslator  -- RBF kernel ridge (gladka interpolacja
                            nieliniowa; szerokosc jadra = heurystyka
                            mediany, wsparcie kapowane).

Wyjscie clamp_min(0): cechy sa nieujemne (jak po ReLU backbone),
parytet z FeatureStatsKSparse.
"""
from __future__ import annotations

import torch


def _add_bias(a: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)
    return torch.cat([a, ones], dim=1)


class RidgeTranslator:
    """Liniowa mapa anchor(E)->feature(D) z regularyzacja L2 (zamknieta
    forma). W: [E+1, D] (z biasem). predict clamp_min(0)."""

    def __init__(self, lam: float = 0.1):
        self.lam = float(lam)
        self.W: torch.Tensor | None = None

    def fit(self, anchor: torch.Tensor, feats: torch.Tensor
            ) -> "RidgeTranslator":
        if anchor.ndim != 2 or feats.ndim != 2 or len(anchor) != len(feats):
            raise ValueError("fit: anchor[N,E], feats[N,D], rowne N")
        aa = _add_bias(anchor)                       # [N, E+1]
        eye = torch.eye(aa.shape[1], dtype=aa.dtype, device=aa.device)
        gram = aa.T @ aa + self.lam * eye            # [E+1, E+1]
        self.W = torch.linalg.solve(gram, aa.T @ feats)   # [E+1, D]
        return self

    def predict(self, anchor: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            raise RuntimeError("RidgeTranslator: najpierw fit()")
        return (_add_bias(anchor) @ self.W).clamp_min(0.0)


def _rbf(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    return torch.exp(-gamma * torch.cdist(x, y) ** 2)


class KernelRidgeTranslator:
    """RBF kernel ridge anchor->feature. gamma = 1/median(pairwise sqdist)
    (heurystyka mediany, zamrozona z danych treningowych). Wsparcie
    kapowane do max_support losowych punktow (koszt O(N^3))."""

    def __init__(self, lam: float = 0.1, max_support: int = 400,
                 seed: int = 0):
        self.lam = float(lam)
        self.max_support = int(max_support)
        self.seed = int(seed)
        self.support: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None
        self.gamma: float | None = None

    def fit(self, anchor: torch.Tensor, feats: torch.Tensor
            ) -> "KernelRidgeTranslator":
        if anchor.ndim != 2 or feats.ndim != 2 or len(anchor) != len(feats):
            raise ValueError("fit: anchor[N,E], feats[N,D], rowne N")
        n = len(anchor)
        if n > self.max_support:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            idx = torch.randperm(n, generator=g)[:self.max_support]
            anchor, feats = anchor[idx], feats[idx]
        self.support = anchor
        with torch.no_grad():
            d2 = torch.cdist(anchor, anchor) ** 2
            med = d2[d2 > 0].median() if (d2 > 0).any() else torch.tensor(1.0)
            self.gamma = float(1.0 / med.clamp_min(1e-8))
            k = _rbf(anchor, anchor, self.gamma)     # [M, M]
            eye = torch.eye(len(anchor), dtype=k.dtype, device=k.device)
            self.alpha = torch.linalg.solve(k + self.lam * eye, feats)  # [M,D]
        return self

    def predict(self, anchor: torch.Tensor) -> torch.Tensor:
        if self.alpha is None or self.support is None or self.gamma is None:
            raise RuntimeError("KernelRidgeTranslator: najpierw fit()")
        return (_rbf(anchor, self.support, self.gamma) @ self.alpha
                ).clamp_min(0.0)


class ProcrustesAlign:
    """Ortogonalne wyrownanie H_A -> H_B (Droga R2). Minimalizuje
    ||H_A @ Omega - H_B||_F po ortogonalnych Omega: Omega = U Vt, gdzie
    U S Vt = SVD(H_Aᵀ H_B). Zachowuje normy i katy (izometria)."""

    def __init__(self):
        self.omega: torch.Tensor | None = None

    def fit(self, h_a: torch.Tensor, h_b: torch.Tensor) -> "ProcrustesAlign":
        if h_a.shape != h_b.shape or h_a.ndim != 2:
            raise ValueError("fit: H_A, H_B musza byc [N,D] o tym samym "
                             "ksztalcie (R-mild: rowne wymiary)")
        u, _, vt = torch.linalg.svd(h_a.T @ h_b, full_matrices=False)
        self.omega = u @ vt                          # [D, D] ortogonalna
        return self

    def transform(self, h_a: torch.Tensor) -> torch.Tensor:
        if self.omega is None:
            raise RuntimeError("ProcrustesAlign: najpierw fit()")
        return h_a @ self.omega

    def disparity(self, h_a: torch.Tensor, h_b: torch.Tensor) -> float:
        """Rezyduum wyrownania (obserwacja jakosci)."""
        return float(((self.transform(h_a) - h_b) ** 2).mean())


if __name__ == "__main__":
    # Smoke (CPU): odtworzenie liniowej mapy + nieujemnosc + ksztalty.
    torch.manual_seed(0)
    E, D, N = 8, 16, 500
    A = torch.randn(N, E)
    Wtrue = torch.randn(E, D)
    F = torch.relu(A @ Wtrue + 0.01 * torch.randn(N, D))

    rid = RidgeTranslator(lam=1e-2).fit(A, F)
    pr = rid.predict(A)
    assert pr.shape == (N, D) and (pr >= 0).all()
    mse_r = torch.mean((pr - F) ** 2).item()

    kr = KernelRidgeTranslator(lam=1e-2, max_support=300).fit(A, F)
    pk = kr.predict(A[:32])
    assert pk.shape == (32, D) and (pk >= 0).all()
    mse_k = torch.mean((kr.predict(A) - F) ** 2).item()

    # Procrustes: HB = HA @ Q (Q ortogonalna) -> odzysk near-exact
    Q, _ = torch.linalg.qr(torch.randn(D, D))
    HA = torch.relu(torch.randn(N, D))
    HB = HA @ Q
    pa = ProcrustesAlign().fit(HA, HB)
    assert pa.omega.shape == (D, D)
    disp = pa.disparity(HA, HB)
    assert disp < 1e-6, f"Procrustes nie odzyskal ortogonalnej mapy: {disp}"
    print(f"Smoke OK: Ridge MSE {mse_r:.4f} | KernelRidge MSE {mse_k:.4f} "
          f"| gamma={kr.gamma:.4f} | Procrustes disp {disp:.2e}")
