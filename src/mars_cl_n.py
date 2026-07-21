"""
mars_cl_n.py -- Droga N: selektywne zapominanie z gwarancja
(DROGA_N_PLAN.md). NOWY plik -- kod v0.8 NIETKNIETY; branch droga-n.

Operacje na instancji MarsCollective (protokol I):
  unlearn_class(m, c, scrub=...)  : usuniecie klasy z pamieci;
      light = wpisy (statystyki, pod, prototyp, seen), projekcja
      nietknieta; scrub = + douczenie projekcji na snach WSZYSTKICH
      pozostalych klas (proba wymazania sladu z wag).
  relearn_small(m, c, X, y, ...)  : ponowna nauka klasy z malej probki
      przy ZAMROZONEJ projekcji (statystyki + pod; prototyp = kotwica
      slowna istnieje a priori). Tempo powrotu mierzy informacje
      resztkowa w projekcji -- por. plan, poziom 2.
  class_accs(m, task_data, allowed): per-class accuracy z maskowanym
      argmaxem po zbiorze `allowed` (wspolna maska dla porownan
      light/scrub/never -- mitygacja confoundu przestrzeni etykiet).
"""
import torch

from cl_common import masked_argmax
from mars_cl_f3 import _train_pods_with_negatives


def unlearn_class(m, c, scrub=False, epochs=15, lr=0.001, device="cpu",
                  n_dream_scrub=2000):
    """Usuwa klase c z pamieci agenta. Zwraca liste pozostalych klas."""
    for d in (m.stats.p, m.stats.mean, m.stats.var, m.stats.w):
        d.pop(c, None)
    m.pods.pop(c, None)
    m.protos.pop(c, None)
    m.seen_classes = [x for x in m.seen_classes if x != c]
    if scrub and m.seen_classes:
        rest = list(m.seen_classes)
        feats = torch.cat([m.stats.sample(r, n_dream_scrub, device)
                           for r in rest])
        y = torch.cat([torch.full((n_dream_scrub,), r, dtype=torch.long,
                                  device=device) for r in rest])
        # douczenie projekcji WYLACZNIE na snach pozostalych
        # (old=[] -> bez dodatkowego miksowania snow w batchu)
        m._fit_proj_feats(feats, y, [], rest, epochs, lr, device)
    return list(m.seen_classes)


def unlearn_reinit(m, c, epochs=15, lr=0.001, device="cpu",
                   n_dream_scrub=2000, seed=0):
    """N1c: light + REINICJALIZACJA projekcji (deterministycznie
    z seeda) + nauka od zera na snach pozostalych klas. Informacja
    o klasie c nie moze przetrwac (nowe wagi)."""
    import torch.nn as nn
    unlearn_class(m, c, scrub=False, device=device)
    torch.manual_seed(seed)
    mods = m.proj.modules() if hasattr(m.proj, "modules") else [m.proj]
    for mod in mods:
        if isinstance(mod, nn.Linear):
            mod.reset_parameters()
    rest = list(m.seen_classes)
    if rest:
        feats = torch.cat([m.stats.sample(r, n_dream_scrub, device)
                           for r in rest])
        y = torch.cat([torch.full((n_dream_scrub,), r, dtype=torch.long,
                                  device=device) for r in rest])
        m._fit_proj_feats(feats, y, [], rest, epochs, lr, device)
    return rest


def relearn_small(m, c, X, y, epochs=15, lr=0.001, device="cpu",
                  balanced_negatives=False):
    """Ponowna nauka klasy c z malej probki; projekcja ZAMROZONA.
    balanced_negatives=True (N1b): negatywy LACZNIE ~= liczbie
    pozytywow (naprawa podlogi z N1: 2304 negatywy na 100 pozytywow
    uczyly pod 'nigdy nie przewiduj c')."""
    with torch.no_grad():
        feats = m.feats_batched(X)
    old = list(m.seen_classes)
    m.stats.update(feats, y, [c])
    if balanced_negatives and old:
        neg_per = max(len(X) // len(old), 4)
    else:
        neg_per = m.replay_per_class
    neg_f, neg_y = m.stats.replay_batch(old, neg_per, device)
    m.protos[c] = m.word_vecs[c].to(device)
    m.seen_classes = m.seen_classes + [c]
    with torch.no_grad():
        routed = m.route(m.embed_from_feats(feats))
    _train_pods_with_negatives(m, [c], feats, y, routed, neg_f, neg_y,
                               epochs, lr, device)


def class_accs(m, task_data, allowed, batch=2048):
    """dict: klasa -> acc na jej przykladach testowych, argmax maskowany
    do zbioru `allowed` (predykcje ograniczone do allowed)."""
    Xte = torch.cat([td["Xte"] for td in task_data])
    yte = torch.cat([td["yte"] for td in task_data])
    preds = []
    for s in range(0, len(Xte), batch):
        logits = m.forward(Xte[s:s + batch])
        preds.append(masked_argmax(logits, allowed))
    preds = torch.cat(preds)
    out = {}
    for c in sorted(set(int(v) for v in yte.tolist())):
        msk = yte == c
        out[c] = float((preds[msk] == c).float().mean())
    return out


if __name__ == "__main__":
    # Smoke (CPU, syntetyczne): unlearn light/scrub + relearn + maski.
    import copy

    import torch.nn.functional as F
    from mars_collective import MarsCollective

    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    tds = [{"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200],
            "Xte": X[:200], "yte": y[:200]},
           {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:],
            "Xte": X[200:], "yte": y[200:]}]
    m = MarsCollective(wv, dream_model="sparse", stats_k=4,
                       replay_per_class=16)
    m.init_representation(tds, epochs=1, lr=1e-3, device="cpu")
    for td in tds:
        m.learn_task(td, epochs=1, lr=1e-3, device="cpu")

    # light: klasa znika ze wszystkich struktur, projekcja bez zmian
    m2 = copy.deepcopy(m)
    rest = unlearn_class(m2, 1, scrub=False, device="cpu")
    assert 1 not in m2.stats.p and 1 not in m2.pods and 1 not in m2.protos
    assert rest == [0, 2, 3]
    w_after = next(m2.proj.parameters()).detach()
    assert torch.allclose(next(m.proj.parameters()), w_after), \
        "light ruszyl projekcje"
    a = class_accs(m2, tds, allowed=m2.seen_classes)
    assert set(a) == {0, 1, 2, 3} and a[1] == 0.0, \
        "klasa usunieta ma niezerowe acc?"

    # scrub: projekcja sie zmienia
    m3 = copy.deepcopy(m)
    unlearn_class(m3, 1, scrub=True, epochs=1, device="cpu",
                  n_dream_scrub=64)
    assert not torch.allclose(next(m.proj.parameters()),
                              next(m3.proj.parameters())), \
        "scrub nie ruszyl projekcji"

    # relearn z malej probki, proj zamrozona
    w3 = next(m3.proj.parameters()).detach().clone()
    relearn_small(m3, 1, X[100:150], y[100:150], epochs=1, device="cpu")
    assert 1 in m3.pods and 1 in m3.stats.p and 1 in m3.seen_classes
    assert torch.allclose(w3, next(m3.proj.parameters())), \
        "relearn ruszyl projekcje"
    a3 = class_accs(m3, tds, allowed=[0, 1, 2, 3])
    print(f"Smoke OK. acc po relearn (syntetyk): {a3}")
