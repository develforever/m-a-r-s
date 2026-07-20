"""
mars_cl_m.py -- Droga M: dlugi horyzont, uogolnienie stosu do N=100 klas
(DROGA_M_PLAN.md). NOWY plik -- kod v0.7 NIETKNIETY; branch droga-m.

Twarde "10" w stosie siedzi w czterech miejscach: MarsCLSystem.forward
i _train_pods (global N_OUT), mars_cl_f3._train_pods_with_negatives
(literal 10), MarsCLSemanticF3J.forward (import N_OUT). Tu: WIERNE
kopie tych czterech z jedyna zmiana wymiaru wyjscia na N_CLASSES=100.
Zero monkey-patchingu -- moduly bazowe nieruszone.

Dodatkowo: rejestracja slow CIFAR-100 w CLASS_WORDS (nazwy zlozone
usredniane po czlonach, konwencja load_word_vectors) oraz jednorazowa
ekstrakcja cech resnet18@224 dla CIFAR-100 z cache (wzor mars_cl_l).
"""
import os

import torch
import torch.nn as nn

from cl_common import BB_H
from mars_cl_semantic import CLASS_WORDS, MarsCLSemantic
from mars_collective import MarsCollective

N_CLASSES = 100

# ----------------------------------------------- CIFAR-100: slowa kotwic
CIFAR100_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
    "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
    "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
    "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
    "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm",
]
# fallbacki dla slow potencjalnie spoza slownika GloVe 6B
_EXTRA_WORDS = {"flatfish": ["flatfish", "flounder"]}

CLASS_WORDS["CIFAR-100"] = {
    i: _EXTRA_WORDS.get(name, name.split("_"))
    for i, name in enumerate(CIFAR100_NAMES)
}

TASKS20 = [tuple(range(i, i + 5)) for i in range(0, 100, 5)]

FEATS_CACHE_100 = os.path.join(os.path.dirname(__file__), "..", "data",
                               "cifar100_resnet18_224_feats.pt")


def extract_or_load_cifar100_feats(device, resize=224, chunk=128,
                                   cache=FEATS_CACHE_100):
    """Jednorazowa ekstrakcja cech resnet18 (512-d) dla CIFAR-100.
    Wejscie: piksele [0,1] -> normalizacja ImageNet -> resize."""
    import torch.nn.functional as F
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        return (d["Ftr"].to(device), d["ytr"].to(device),
                d["Fte"].to(device), d["yte"].to(device))
    import torchvision
    from cifar_cl import DATA_ROOT
    from mars_cl_l import PretrainedBackbone
    print(f"[M] Jednorazowa ekstrakcja cech resnet18@{resize} "
          f"dla CIFAR-100 (cache: {os.path.abspath(cache)})")
    torch.manual_seed(0)   # deterministycznie; reduce nieuzywane
    bb = PretrainedBackbone(resize=resize, chunk=chunk).to(device)
    outs_all, ys = [], []
    for train in (True, False):
        ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=train,
                                           download=True)
        X = (torch.tensor(ds.data, dtype=torch.float32, device=device)
             .permute(0, 3, 1, 2) / 255.0)          # [N,3,32,32] w [0,1]
        y = torch.tensor(ds.targets, device=device)
        outs = []
        for s in range(0, len(X), chunk):
            x = (X[s:s + chunk] - bb.imnet_mean) / bb.imnet_std
            x = F.interpolate(x, size=resize, mode="bilinear",
                              align_corners=False)
            with torch.no_grad():
                outs.append(bb.features(x).flatten(1))
            if (s // chunk) % 40 == 0:
                print(f"    ekstrakcja: {s}/{len(X)}", flush=True)
        outs_all.append(torch.cat(outs))
        ys.append(y)
    Ftr, Fte = outs_all
    ytr, yte = ys
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save({"Ftr": Ftr.cpu(), "ytr": ytr.cpu(),
                "Fte": Fte.cpu(), "yte": yte.cpu()}, cache)
    print(f"[M] Cache zapisany ({Ftr.shape[0]}+{Fte.shape[0]} x 512).")
    return Ftr, ytr, Fte, yte


# ------------------------------------------- pody z negatywami, n_out=100
def _train_pods_negatives_n(model, classes, feats, y, routed, neg_feats,
                            neg_y, epochs, lr, device, n_out=N_CLASSES):
    """WIERNA kopia mars_cl_f3._train_pods_with_negatives;
    jedyna zmiana: wymiar wyjscia n_out zamiast literalu 10."""
    crit = nn.CrossEntropyLoss()
    for c in classes:
        m = routed == c
        if m.sum() < 10:
            m = y == c
        Xi = feats[m]
        yi = y[m]
        others = ~m
        if others.any():
            k = min(int(others.sum()), 512)
            idx = others.nonzero(as_tuple=True)[0][:k]
            Xi = torch.cat([Xi, feats[idx]])
            yi = torch.cat([yi, y[idx]])
        if neg_feats is not None:
            Xi = torch.cat([Xi, neg_feats])
            yi = torch.cat([yi, neg_y])
        W1 = torch.randn(BB_H, model.pod_hidden, device=device) * 0.01
        b1 = torch.zeros(model.pod_hidden, device=device)
        W2 = torch.randn(model.pod_hidden, n_out, device=device) * 0.01
        b2 = torch.zeros(n_out, device=device)
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


# --------------------------------------------------- kolektyw/seq, N=100
class MarsCollectiveM(MarsCollective):
    """MarsCollective (F3J + protokol I) z wyjsciem N_CLASSES.
    forward / learn_task / adopt_classes = wierne kopie rodzicow;
    jedyna zmiana: N_CLASSES i _train_pods_negatives_n."""

    def forward(self, x):
        with torch.no_grad():
            feats = self._condition(self.backbone(x))
            emb = self.embed_from_feats(feats)
            routed = self.route(emb)
            out = torch.zeros(len(x), N_CLASSES, device=x.device)
            for c in self.seen_classes:
                m = routed == c
                if m.any():
                    out[m] = self._pod_forward_class(feats[m], c)
        return out

    def learn_task(self, td, epochs, lr, device):
        classes = td["classes"]
        X, y = td["Xtr"], td["ytr"]
        with torch.no_grad():
            feats = self.feats_batched(X)
        old = list(self.seen_classes)
        self.stats.update(feats, y, classes)
        self._fit_proj_feats(feats, y, old, old + list(classes),
                             epochs, lr, device)
        neg_f, neg_y = self.stats.replay_batch(old, self.replay_per_class,
                                               device)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        _train_pods_negatives_n(self, classes, feats, y, routed,
                                neg_f, neg_y, epochs, lr, device)

    def adopt_classes(self, classes, payloads, epochs, lr, device,
                      n_dream=500):
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
        self._fit_proj_feats(feats, y, old, old + list(classes),
                             epochs, lr, device)
        neg_f, neg_y = self.stats.replay_batch(old, self.replay_per_class,
                                               device)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        _train_pods_negatives_n(self, classes, feats, y, routed,
                                neg_f, neg_y, epochs, lr, device)


# ------------------------------------------------------ sufit "all", N=100
class MarsCLSemanticAllM(MarsCLSemantic):
    """MarsCLSemantic(proj_train='all') z wyjsciem N_CLASSES.
    forward / _train_pods = wierne kopie MarsCLSystem; zmiana: N_CLASSES."""

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
            emb = self.embed_from_feats(feats)
            routed = self.route(emb)
            out = torch.zeros(len(x), N_CLASSES, device=x.device)
            for c in self.seen_classes:
                m = routed == c
                if m.any():
                    out[m] = self._pod_forward_class(feats[m], c)
        return out

    def _train_pods(self, classes, X, y, epochs, lr, device):
        with torch.no_grad():
            feats = self.feats_batched(X)
            routed = self.route(self.embed_from_feats(feats))
        crit = nn.CrossEntropyLoss()
        for c in classes:
            m = routed == c
            if m.sum() < 10:
                m = y == c
            Xi, yi_c = feats[m], y[m]
            W1 = torch.randn(BB_H, self.pod_hidden, device=device) * 0.01
            b1 = torch.zeros(self.pod_hidden, device=device)
            W2 = torch.randn(self.pod_hidden, N_CLASSES,
                             device=device) * 0.01
            b2 = torch.zeros(N_CLASSES, device=device)
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


if __name__ == "__main__":
    # Smoke (CPU, syntetyczne cechy 512-d, 3 zadania x 5 klas):
    import torch.nn.functional as F
    from mars_cl_l import ReducedBackbone

    assert len(CIFAR100_NAMES) == 100 and len(TASKS20) == 20
    assert len(CLASS_WORDS["CIFAR-100"]) == 100

    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(100)}
    X = torch.randn(600, 512).abs()
    y = torch.cat([torch.full((40,), c) for c in range(15)])
    tds = [{"classes": list(range(i, i + 5)),
            "Xtr": X[i * 40:(i + 5) * 40], "ytr": y[i * 40:(i + 5) * 40]}
           for i in (0, 5, 10)]

    m = MarsCollectiveM(wv, backbone_module=ReducedBackbone(),
                        dream_model="sparse", stats_k=4,
                        replay_per_class=16)
    m.init_representation(tds, epochs=1, lr=1e-3, device="cpu")
    for td in tds[:2]:
        m.learn_task(td, epochs=1, lr=1e-3, device="cpu")
    assert m.forward(X[:16]).shape == (16, 100)
    payloads = {c: m.export_class_stats(c, 40) for c in (5, 6)}
    m2 = MarsCollectiveM(wv, backbone_module=ReducedBackbone(),
                         dream_model="sparse", stats_k=4,
                         replay_per_class=16)
    m2.init_representation([tds[0]], epochs=1, lr=1e-3, device="cpu")
    m2.learn_task(tds[0], epochs=1, lr=1e-3, device="cpu")
    m2.adopt_classes([5, 6], payloads, epochs=1, lr=1e-3, device="cpu",
                     n_dream=32)
    assert m2.forward(X[:16]).shape == (16, 100)
    assert m2.seen_classes == [0, 1, 2, 3, 4, 5, 6]

    ma = MarsCLSemanticAllM(wv, proj_train="all",
                            backbone_module=ReducedBackbone())
    ma.init_representation(tds, epochs=1, lr=1e-3, device="cpu")
    ma.learn_task(tds[0], epochs=1, lr=1e-3, device="cpu")
    assert ma.forward(X[:16]).shape == (16, 100)
    print("Smoke OK: N=100 forward/learn/adopt/sufit dzialaja.")


# ---------------------------------------- M1b: budzet snow staly lacznie
class MarsCollectiveMBalanced(MarsCollectiveM):
    """
    M1b: jedyna zmiana vs MarsCollectiveM -- budzet snow/negatywow
    STALY LACZNIE (dzielony po starych klasach), nie staly per klase.
    Naprawia zalew batcha snami przy dlugim horyzoncie (M1: ~90% snow
    przy 95 starych klasach). Przy T=5 zachowanie ~rownowazne.
    _fit_proj_feats = wierna kopia mars_cl_f3; zmiana: k_per_old.
    """

    def __init__(self, word_vecs, dream_per_old=None, neg_per_old=None,
                 **kw):
        super().__init__(word_vecs, **kw)
        # None = budzet lacznie (M1b); liczby = stale per klase (M1c)
        self.dream_per_old = dream_per_old
        self.neg_per_old = neg_per_old

    def _fit_proj_feats(self, feats, y, old_classes, all_classes,
                        epochs, lr, device, batch=512):
        import torch.nn.functional as F
        from mars_cl_semantic import TEMP
        W = torch.stack([self.word_vecs[c].to(device)
                         for c in all_classes])
        c2i = {c: i for i, c in enumerate(all_classes)}
        yi = torch.tensor([c2i[int(v)] for v in y.tolist()], device=device)
        # M1b: budzet lacznie = batch; M1c: jawny per klase
        if not old_classes:
            k_per_old = 0
        elif self.dream_per_old is not None:
            k_per_old = self.dream_per_old
        else:
            k_per_old = max(batch // len(old_classes), 4)
        crit = nn.CrossEntropyLoss()
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
                    df, dy = self.stats.replay_batch(old_classes,
                                                     k_per_old, device)
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

    def _neg_budget(self, old):
        if not old:
            return self.replay_per_class
        if self.neg_per_old is not None:
            return self.neg_per_old
        return max(512 // len(old), 4)

    def learn_task(self, td, epochs, lr, device):
        classes = td["classes"]
        X, y = td["Xtr"], td["ytr"]
        with torch.no_grad():
            feats = self.feats_batched(X)
        old = list(self.seen_classes)
        self.stats.update(feats, y, classes)
        self._fit_proj_feats(feats, y, old, old + list(classes),
                             epochs, lr, device)
        neg_f, neg_y = self.stats.replay_batch(old, self._neg_budget(old),
                                               device)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        _train_pods_negatives_n(self, classes, feats, y, routed,
                                neg_f, neg_y, epochs, lr, device)

    def adopt_classes(self, classes, payloads, epochs, lr, device,
                      n_dream=500):
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
        self._fit_proj_feats(feats, y, old, old + list(classes),
                             epochs, lr, device)
        neg_f, neg_y = self.stats.replay_batch(old, self._neg_budget(old),
                                               device)
        for c in classes:
            self.protos[c] = self.word_vecs[c].to(device)
        self.seen_classes = self.seen_classes + list(classes)
        with torch.no_grad():
            routed = self.route(self.embed_from_feats(feats))
        _train_pods_negatives_n(self, classes, feats, y, routed,
                                neg_f, neg_y, epochs, lr, device)
