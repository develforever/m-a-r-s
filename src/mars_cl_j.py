"""
mars_cl_j.py -- Droga J: naprawy i darmowe dzwignie (DROGA_J_PLAN.md).

NOWY plik -- kod v0.3 (mars_cl*.py, cifar_cl.py) pozostaje NIETKNIETY;
wszystko tu jest podklasa/utility na branchu droga-j.

Trzy mechanizmy (audyt 2026-07-10):

1. calibrate_bn(): losowy zamrozony backbone nigdy nie robil forwardu
   w trybie train, wiec BatchNorm mial running stats (0,1) = identycznosc
   we WSZYSTKICH runach F1d/G1/F3/F3b/H1/H1b/F4. Kalibracja: jeden
   przebieg po NIEETYKIETOWANYCH obrazach zadania 0 (momentum=None ->
   dokladne statystyki zbioru), PRZED jakakolwiek nauka, potem freeze
   na zawsze. Uczciwosc CL: ten sam zasob co ae0/F2 (obrazy task0 bez
   etykiet); nietykalnosc reprezentacji zachowana.

2. Sigma-norm cech: podzial cech backbone'u przez per-wymiarowe std
   policzone na task0. Celowo BEZ centrowania sredniej -- zachowuje
   nieujemnosc i DOKLADNE zera cech po ReLU (spojnosc z clamp_min(0)
   snu i ze spike-and-slab). Naprawia dominacje wymiarow o duzej skali
   w odleglosciach euklidesowych (NCM, k-means snu).
   Koszt inferencji: +D dzielen (~128 MAC, pomijalne; mac_per_sample
   odziedziczone bez zmian -- odnotowane).

3. FeatureStatsKSparse: sen spike-and-slab. Diagnoza H1: resztkowa luka
   77.57 -> 80.45 to WIERNOSC SNU. Realne cechy po ReLU maja dokladne
   zera z duzym prawdopodobienstwem; clampowany Gaussian generuje tam
   male dodatnie wartosci (poza rozmaitoscia danych). Tu per wymiar
   centroidu przechowujemy P(cecha>0) oraz srednia/wariancje WARUNKOWE
   (z samych wartosci dodatnich); sen = maska Bernoulliego * obciety
   Gaussian. Pamiec: 3 tablice k x D zamiast 2 (1.5x diag przy tym k).

Plus load_cifar10_norm(): normalizacja per kanal dla CIFAR-10 --
Fashion/MNIST zawsze ja mialy (run_D1), CIFAR nie (tylko /255).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_cl import N_OUT
from mars_cl_f3 import MarsCLSemanticF3, FeatureStatsK

# Standardowe statystyki CIFAR-10 (train set, po /255)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------- BN calib
def calibrate_bn(module, X, batch=1024):
    """
    Kalibruje running stats WSZYSTKICH warstw BatchNorm w module jednym
    przebiegiem po X (bez etykiet, bez gradientow). momentum=None =>
    kumulatywna srednia = dokladne statystyki X. Zwraca liczbe warstw BN.
    Wolac PRZED zamrozeniem/eval; po kalibracji stats juz sie nie zmieniaja
    (backbone wraca do eval w init_representation).
    """
    bns = [m for m in module.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    was_training = module.training
    module.train()
    with torch.no_grad():
        for s in range(0, len(X), batch):
            module(X[s:s + batch])
    module.train(was_training)
    return len(bns)


# ------------------------------------------------------------ sen sparse
class FeatureStatsKSparse(FeatureStatsK):
    """
    J3: spike-and-slab -- k centroidow (k-means po cechach, jak F3b),
    per wymiar: p = P(cecha > 0) oraz srednia/wariancja WARUNKOWE
    (liczone wylacznie z wartosci dodatnich). Sampling:
        x = Bernoulli(p) * clamp_min(mu + sigma*z, 0)
    Zera sa PRAWDZIWYMI zerami (jak po ReLU), nie ogonem Gaussiana.
    Wymiary bez zadnej wartosci dodatniej w klastrze: p=0 -> zawsze 0.
    Pamiec: k*(3*D+1) liczb na klase (1.5x FeatureStatsK).
    """
    def __init__(self, k=4, iters=10):
        super().__init__(k=k, iters=iters)
        self.p = {}   # klasa -> [k, D]: P(cecha>0)

    def update(self, feats, y, classes):
        for c in classes:
            fc = feats[y == c].detach()
            cent, assign, k = self._kmeans(fc)
            ps, means, vars_, ws = [], [], [], []
            for j in range(k):
                m = assign == j
                sub = fc[m] if m.any() else fc
                pos = sub > 0                            # [n_j, D]
                cnt = pos.sum(dim=0)
                p_j = cnt.float() / len(sub)
                safe = cnt.clamp_min(1).float()
                mu_j = (sub * pos).sum(dim=0) / safe     # warunkowa srednia
                ex2 = (sub ** 2 * pos).sum(dim=0) / safe
                var_j = (ex2 - mu_j ** 2).clamp_min(1e-6)
                ps.append(p_j)
                means.append(mu_j)
                vars_.append(var_j)
                ws.append(float(m.sum()))
            self.p[c] = torch.stack(ps)
            self.mean[c] = torch.stack(means)
            self.var[c] = torch.stack(vars_)
            self.w[c] = torch.tensor(ws, device=feats.device) / sum(ws)

    def sample(self, c, n, device):
        p, m, v, w = (self.p[c].to(device), self.mean[c].to(device),
                      self.var[c].to(device), self.w[c].to(device))
        comp = torch.multinomial(w, n, replacement=True)
        slab = (m[comp] + v[comp].sqrt()
                * torch.randn(n, m.shape[1], device=device)).clamp_min(0.0)
        mask = (torch.rand(n, m.shape[1], device=device) < p[comp]).float()
        return slab * mask


# ------------------------------------------------------- system z Drogi J
class MarsCLSemanticF3J(MarsCLSemanticF3):
    """
    MarsCLSemanticF3 + pokretla Drogi J:
      bn_calib     : kalibracja BN na obrazach task0 (bez etykiet), raz
      feat_signorm : cechy / per-wymiarowe std z task0 (bez centrowania)
      dream_model  : "diag" | "full" (jak F3b/H1b) | "sparse" (J3)
    Kondycjonowanie liczone W init_representation, potem zamrozone --
    caly pipeline (projekcja, sen, pody, forward) widzi juz cechy
    kondycjonowane, wiec zadna inna sciezka nie wymaga zmian.
    """
    def __init__(self, word_vecs, bn_calib=False, feat_signorm=False,
                 dream_model="diag", stats_k=1, **kw):
        base_dream = "diag" if dream_model == "sparse" else dream_model
        super().__init__(word_vecs, dream_model=base_dream,
                         stats_k=stats_k, **kw)
        if dream_model == "sparse":
            self.stats = FeatureStatsKSparse(k=stats_k)
        self.dream_model_j = dream_model
        self.bn_calib = bn_calib
        self.feat_signorm = feat_signorm
        self._feat_scale = None   # [BB_H] albo None

    # --- kondycjonowanie cech (jedyny punkt dotyku) ---
    def _condition(self, feats):
        if self._feat_scale is not None:
            feats = feats / self._feat_scale
        return feats

    def feats_batched(self, X, batch=2048):
        return self._condition(super().feats_batched(X, batch))

    def init_representation(self, task_data, epochs, lr, device):
        if self.bn_calib:
            n_bn = calibrate_bn(self.backbone, task_data[0]["Xtr"])
            print(f"    (bn_calib: skalibrowano {n_bn} warstw BN "
                  f"na {len(task_data[0]['Xtr'])} obrazach task0)")
        super().init_representation(task_data, epochs, lr, device)
        if self.feat_signorm:
            with torch.no_grad():
                f0 = self.feats_batched(task_data[0]["Xtr"])
                # _feat_scale jeszcze None -> f0 to surowe cechy
                self._feat_scale = (f0.std(dim=0, unbiased=True)
                                    .clamp_min(1e-6).to(device))

    def forward(self, x):
        """Jak MarsCLSystem.forward, ale przez _condition."""
        with torch.no_grad():
            feats = self._condition(self.backbone(x))
            emb = self.embed_from_feats(feats)
            routed = self.route(emb)
            out = torch.zeros(len(x), N_OUT, device=x.device)
            for c in self.seen_classes:
                m = routed == c
                if m.any():
                    out[m] = self._pod_forward_class(feats[m], c)
        return out


# ---------------------------------------------------------------- CIFAR
def load_cifar10_norm(device, root=None):
    """
    load_cifar10 + normalizacja per kanal (in-place, bez kopii ~600 MB).
    Naprawa J2: Fashion/MNIST zawsze mialy Normalize, CIFAR tylko /255.
    """
    from cifar_cl import load_cifar10, DATA_ROOT
    Xtr, ytr, Xte, yte = load_cifar10(device, root=root or DATA_ROOT)
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    for X in (Xtr, Xte):
        Xv = X.view(-1, 3, 32, 32)
        Xv.sub_(mean).div_(std)
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    # Smoke (CPU, dane syntetyczne, bez pobierania):
    torch.manual_seed(0)
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    td0 = {"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200]}
    td1 = {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:]}
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}

    # 1) pelny pipeline J: bn_calib + signorm + sen sparse
    m = MarsCLSemanticF3J(wv, bn_calib=True, feat_signorm=True,
                          dream_model="sparse", stats_k=4,
                          replay_per_class=16)
    m.init_representation([td0, td1], epochs=1, lr=1e-3, device="cpu")
    bns = [mm for mm in m.backbone.modules()
           if isinstance(mm, nn.modules.batchnorm._BatchNorm)]
    assert bns and not torch.allclose(
        bns[0].running_mean, torch.zeros_like(bns[0].running_mean)), \
        "BN nieskalibrowany"
    assert m._feat_scale is not None and m._feat_scale.shape == (128,)
    m.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    m.learn_task(td1, epochs=1, lr=1e-3, device="cpu")
    assert m.forward(X[:16]).shape == (16, 10)

    # 2) sen sparse: prawdziwe zera, nieujemnosc, ksztalt
    s = m.stats.sample(0, 256, "cpu")
    assert s.shape == (256, 128) and (s >= 0).all()
    frac_zero = (s == 0).float().mean().item()
    assert frac_zero > 0.01, f"sen bez zer? frac={frac_zero}"

    # 3) wariant bez kondycjonowania == zachowanie F3b (kontrola sciezki)
    m2 = MarsCLSemanticF3J(wv, dream_model="diag", stats_k=2,
                           replay_per_class=16)
    m2.init_representation([td0, td1], epochs=1, lr=1e-3, device="cpu")
    assert m2._feat_scale is None
    m2.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    m2.learn_task(td1, epochs=1, lr=1e-3, device="cpu")
    assert m2.forward(X[:16]).shape == (16, 10)

    print(f"Smoke test OK. (frakcja zer w snie sparse: {frac_zero:.2f})")
