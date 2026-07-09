"""
mars_cl_f3.py -- F3: parametryczny feature replay ("sen" bez danych).

IDEA (DROGA_F_PLAN.md, sekcja F3; inspiracja: generative replay z analizy
zewnetrznego agenta, uproszczona dzieki naszej architekturze):
  Backbone jest ZAMROZONY => cechy klas sa STACJONARNE => zamiast bufora
  obrazow (replay-200) albo modulu generatywnego (VAE) wystarczy per klasa
  SREDNIA i WARIANCJA cech (diagonalny Gaussian, ~1 KB/klase). Podczas
  nauki zadania t "snimy" cechy starych klas próbkami z Gaussianow.

Dwa zastosowania (dwa warianty systemu):
  MarsCLF3 (f3_ncm)      : pseudo-negatywy DLA PODOW -- pod 10-way widzi
    cechy starych klas jako negatywy => kalibracja logitow miedzy zadaniami.
    Routing bez zmian (NCM w losowej projekcji, jak F1d).
  MarsCLSemanticF3 (f3_sem): jw. + replay DLA PROJEKCJI semantycznej --
    douczanie projekcji obraz->slowa na kazdym zadaniu z domieszka
    wygenerowanych cech starych klas (targety = ich slowa). Cel: zbiec do
    poziomu g1_all (projekcja "widzi" wszystkie klasy) bez podgladania
    danych. Naprawia katastrofe g1_seq (forgetting 98pp -> ?).

Prywatnosc/pamiec: zero przechowywanych probek; tylko statystyki cech.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_cl import MarsCLSystem, BB_H
from mars_cl_semantic import MarsCLSemantic, TEMP


class FeatureStats:
    """Diagonalne Gaussiany cech per klasa (backbone zamrozony)."""
    def __init__(self):
        self.mean = {}
        self.var = {}

    def update(self, feats, y, classes):
        for c in classes:
            fc = feats[y == c]
            self.mean[c] = fc.mean(dim=0).detach()
            self.var[c] = fc.var(dim=0, unbiased=True).clamp_min(1e-6).detach()

    def sample(self, c, n, device):
        m, v = self.mean[c].to(device), self.var[c].to(device)
        return (m + v.sqrt() * torch.randn(n, len(m), device=device)).clamp_min(0.0)
        # clamp_min(0): cechy sa po ReLU (nieujemne) -- Gaussian tego nie wie

    def replay_batch(self, classes, n_per, device):
        """Cechy + etykiety wygenerowane dla podanych klas."""
        if not classes:
            return None, None
        Xs = torch.cat([self.sample(c, n_per, device) for c in classes])
        ys = torch.cat([torch.full((n_per,), c, dtype=torch.long,
                                   device=device) for c in classes])
        return Xs, ys


class FeatureStatsK:
    """
    F3b: bogatszy sen -- k centroidow per klasa (mini k-means w cechach),
    kazdy z diagonalna wariancja i waga (licznosc klastra). k=1 == FeatureStats.
    Pamiec: k * (2*128 + 1) liczb na klase (~kilka KB przy k=4).
    """
    def __init__(self, k=4, iters=10):
        self.k = k
        self.iters = iters
        self.mean = {}   # klasa -> [k, D]
        self.var = {}    # klasa -> [k, D]
        self.w = {}      # klasa -> [k] (wagi klastrow)

    def _kmeans(self, fc):
        k = min(self.k, len(fc))
        idx = torch.randperm(len(fc), device=fc.device)[:k]
        cent = fc[idx].clone()
        for _ in range(self.iters):
            d = torch.cdist(fc, cent)
            assign = d.argmin(dim=1)
            for j in range(k):
                m = assign == j
                if m.any():
                    cent[j] = fc[m].mean(dim=0)
        return cent, assign, k

    def update(self, feats, y, classes):
        for c in classes:
            fc = feats[y == c].detach()
            cent, assign, k = self._kmeans(fc)
            means, vars_, ws = [], [], []
            for j in range(k):
                m = assign == j
                sub = fc[m] if m.any() else fc
                means.append(sub.mean(dim=0))
                vars_.append(sub.var(dim=0, unbiased=True).clamp_min(1e-6)
                             if len(sub) > 1 else
                             torch.full_like(sub.mean(dim=0), 1e-3))
                ws.append(float(m.sum()))
            self.mean[c] = torch.stack(means)
            self.var[c] = torch.stack(vars_)
            self.w[c] = torch.tensor(ws, device=feats.device) / sum(ws)

    def sample(self, c, n, device):
        m, v, w = (self.mean[c].to(device), self.var[c].to(device),
                   self.w[c].to(device))
        comp = torch.multinomial(w, n, replacement=True)
        return (m[comp] + v[comp].sqrt()
                * torch.randn(n, m.shape[1], device=device)).clamp_min(0.0)

    def replay_batch(self, classes, n_per, device):
        if not classes:
            return None, None
        Xs = torch.cat([self.sample(c, n_per, device) for c in classes])
        ys = torch.cat([torch.full((n_per,), c, dtype=torch.long,
                                   device=device) for c in classes])
        return Xs, ys


class FeatureStatsFullCovK:
    """
    H1b: sen o pelnej kowariancji -- k klastrow per klasa (k-means), kazdy
    z PELNA macierza kowariancji (Cholesky). Pamiec: k*(D + D^2)/klase
    (k=1: ~66 KB, k=4: ~262 KB) -- wciaz zero przechowywanych probek.
    Motywacja: H1 wyeliminowal dryf; resztkowa luka = wiernosc snu.
    """
    def __init__(self, k=1, iters=10, eps=1e-4):
        self.k = k
        self.iters = iters
        self.eps = eps
        self.mean = {}   # klasa -> [k, D]
        self.chol = {}   # klasa -> [k, D, D]
        self.w = {}      # klasa -> [k]

    def _kmeans(self, fc):
        k = min(self.k, len(fc))
        idx = torch.randperm(len(fc), device=fc.device)[:k]
        cent = fc[idx].clone()
        for _ in range(self.iters):
            assign = torch.cdist(fc, cent).argmin(dim=1)
            for j in range(k):
                m = assign == j
                if m.any():
                    cent[j] = fc[m].mean(dim=0)
        return cent, assign, k

    def update(self, feats, y, classes):
        D = feats.shape[1]
        eye = torch.eye(D, device=feats.device)
        for c in classes:
            fc = feats[y == c].detach()
            cent, assign, k = self._kmeans(fc)
            means, chols, ws = [], [], []
            for j in range(k):
                m = assign == j
                sub = fc[m] if m.sum() > D else fc   # za maly klaster -> cala klasa
                mu = sub.mean(dim=0)
                cov = torch.cov(sub.T) + self.eps * eye
                chols.append(torch.linalg.cholesky(cov))
                means.append(mu)
                ws.append(float(m.sum()))
            self.mean[c] = torch.stack(means)
            self.chol[c] = torch.stack(chols)
            self.w[c] = torch.tensor(ws, device=feats.device) / sum(ws)

    def sample(self, c, n, device):
        m, L, w = (self.mean[c].to(device), self.chol[c].to(device),
                   self.w[c].to(device))
        comp = torch.multinomial(w, n, replacement=True)
        z = torch.randn(n, m.shape[1], device=device)
        x = m[comp] + torch.einsum("bij,bj->bi", L[comp], z)
        return x.clamp_min(0.0)

    def replay_batch(self, classes, n_per, device):
        if not classes:
            return None, None
        Xs = torch.cat([self.sample(c, n_per, device) for c in classes])
        ys = torch.cat([torch.full((n_per,), c, dtype=torch.long,
                                   device=device) for c in classes])
        return Xs, ys


def _train_pods_with_negatives(model, classes, feats, y, routed, neg_feats,
                               neg_y, epochs, lr, device):
    """
    Pody nowych klas: realne probki routowane do poda + negatywy:
    (a) realne probki INNYCH klas biezacego zadania, (b) pseudo-cechy
    starych klas z Gaussianow. CE 10-way => kalibracja miedzy zadaniami.
    """
    crit = nn.CrossEntropyLoss()
    for c in classes:
        m = routed == c
        if m.sum() < 10:
            m = y == c
        Xi = feats[m]
        yi = y[m]
        others = ~m
        if others.any():                      # realne negatywy z zadania
            k = min(int(others.sum()), 512)
            idx = others.nonzero(as_tuple=True)[0][:k]
            Xi = torch.cat([Xi, feats[idx]])
            yi = torch.cat([yi, y[idx]])
        if neg_feats is not None:             # pseudo-negatywy starych klas
            Xi = torch.cat([Xi, neg_feats])
            yi = torch.cat([yi, neg_y])
        W1 = torch.randn(BB_H, model.pod_hidden, device=device) * 0.01
        b1 = torch.zeros(model.pod_hidden, device=device)
        W2 = torch.randn(model.pod_hidden, 10, device=device) * 0.01
        b2 = torch.zeros(10, device=device)
        for t_ in (W1, b1, W2, b2):
            t_.requires_grad = True
        opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(len(Xi), device=device)
            for s in range(0, len(Xi), 512):
                idx = perm[s:s + 512]
                h = torch.relu(Xi[idx] @ W1 + b1)
                loss = crit(h @ W2 + b2, yi[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        model.pods[c] = {"W1": W1.detach(), "b1": b1.detach(),
                         "W2": W2.detach(), "b2": b2.detach()}


class MarsCLF3(MarsCLSystem):
    """F1d (losowy backbone, NCM) + pseudo-negatywy dla podow."""
    def __init__(self, replay_per_class=256, **kw):
        super().__init__(backbone_source="random", proto_mode="mean", **kw)
        self.replay_per_class = replay_per_class
        self.stats = FeatureStats()

    def learn_task(self, td, epochs, lr, device):
        classes = td["classes"]
        X, y = td["Xtr"], td["ytr"]
        with torch.no_grad():
            feats = self.feats_batched(X)
            emb = self.embed_from_feats(feats)
        old = list(self.seen_classes)
        # prototypy NCM (jak F1d)
        for c in classes:
            self.protos[c] = emb[y == c].mean(dim=0).detach()
        self.seen_classes = self.seen_classes + list(classes)
        self.stats.update(feats, y, classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        neg_f, neg_y = self.stats.replay_batch(old, self.replay_per_class,
                                               device)
        _train_pods_with_negatives(self, classes, feats, y, routed,
                                   neg_f, neg_y, epochs, lr, device)


class MarsCLSemanticF3(MarsCLSemantic):
    """
    g1_seq + replay Gaussianow dla PROJEKCJI i podow (glowny kandydat).
    Pokretla F3b (kontrola dryfu projekcji):
      epochs_proj : liczba epok douczania projekcji (None = jak pody);
                    mniej epok = mniej dryfu (obserwacja z F3 full vs smoke)
      l2sp        : kara l2sp * ||W - W_przed_zadaniem||^2 na projekcji
      stats_k     : k centroidow snu per klasa (1 = pojedynczy Gaussian)
    """
    def __init__(self, word_vecs, replay_per_class=256, epochs_proj=None,
                 l2sp=0.0, stats_k=1, dream_model="diag", **kw):
        super().__init__(word_vecs, proj_train="seq", **kw)
        self.replay_per_class = replay_per_class
        self.epochs_proj = epochs_proj
        self.l2sp = l2sp
        if dream_model == "full":                       # H1b
            self.stats = FeatureStatsFullCovK(k=stats_k)
        elif stats_k > 1:
            self.stats = FeatureStatsK(k=stats_k)
        else:
            self.stats = FeatureStats()

    def _fit_proj_feats(self, feats, y, old_classes, all_classes,
                        epochs, lr, device, batch=512):
        """
        Trening projekcji na cechach zadania + SWIEZO SNIONYCH cechach
        starych klas W KAZDYM KROKU (lekcja z F0-replay i smoke'a F3:
        statyczna domieszka tonie w danych nowego zadania; losowanie
        z Gaussianow jest darmowe, wiec balansujemy per krok --
        kazda stara klasa dostaje w mini-batchu tyle probek, ile srednio
        ma klasa nowa).
        """
        W = torch.stack([self.word_vecs[c].to(device) for c in all_classes])
        c2i = {c: i for i, c in enumerate(all_classes)}
        yi = torch.tensor([c2i[int(v)] for v in y.tolist()], device=device)
        n_new = len(all_classes) - len(old_classes)
        k_per_old = max(batch // max(n_new, 1) // 2, 32) if old_classes else 0
        crit = nn.CrossEntropyLoss()
        # F3b: kotwica L2-SP -- snapshot wag projekcji sprzed zadania
        prev = ({n: p.detach().clone() for n, p
                 in self.proj.named_parameters()}
                if (self.l2sp > 0 and old_classes) else None)
        n_epochs = self.epochs_proj if self.epochs_proj else epochs
        for p in self.proj.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(self.proj.parameters(), lr=lr)
        for _ in range(n_epochs):
            perm = torch.randperm(len(feats), device=device)
            for s in range(0, len(feats), batch):
                idx = perm[s:s + batch]
                xb, yb = feats[idx], yi[idx]
                if old_classes:
                    df, dy = self.stats.replay_batch(old_classes, k_per_old,
                                                     device)
                    dyi = torch.tensor([c2i[int(v)] for v in dy.tolist()],
                                       device=device)
                    xb = torch.cat([xb, df])
                    yb = torch.cat([yb, dyi])
                emb = F.normalize(self.proj(xb), dim=1)
                loss = crit(emb @ W.T / TEMP, yb)
                if prev is not None:
                    for n, p in self.proj.named_parameters():
                        loss = loss + self.l2sp * ((p - prev[n]) ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        for p in self.proj.parameters():
            p.requires_grad = False

    def learn_task(self, td, epochs, lr, device):
        classes = td["classes"]
        X, y = td["Xtr"], td["ytr"]
        with torch.no_grad():
            feats = self.feats_batched(X)
        old = list(self.seen_classes)
        self.stats.update(feats, y, classes)
        # projekcja: realne cechy zadania + swiezy "sen" starych klas per krok
        self._fit_proj_feats(feats, y, old, old + list(classes),
                             epochs, lr, device)
        neg_f, neg_y = self.stats.replay_batch(old, self.replay_per_class,
                                               device)
        # prototypy = slowa (istnieja a priori)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        _train_pods_with_negatives(self, classes, feats, y, routed,
                                   neg_f, neg_y, epochs, lr, device)


if __name__ == "__main__":
    # Smoke: 2 zadania, ksztalty + replay path.
    torch.manual_seed(0)
    X = torch.randn(400, 784).abs()
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    td0 = {"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200]}
    td1 = {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:]}
    m = MarsCLF3(replay_per_class=32)
    m.init_representation([td0, td1], epochs=0, lr=1e-3, device="cpu")
    m.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    m.learn_task(td1, epochs=1, lr=1e-3, device="cpu")
    assert m.forward(X[:16]).shape == (16, 10)
    assert set(m.stats.mean) == {0, 1, 2, 3}

    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    ms = MarsCLSemanticF3(wv, replay_per_class=32)
    ms.init_representation([td0, td1], epochs=1, lr=1e-3, device="cpu")
    ms.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    ms.learn_task(td1, epochs=1, lr=1e-3, device="cpu")
    assert ms.forward(X[:16]).shape == (16, 10)
    print("Smoke test OK.")
