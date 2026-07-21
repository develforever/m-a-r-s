"""
mars_cl_i4.py -- Droga I4: kolektyw niezaufany (DROGA_I4_PLAN.md).
NOWY plik -- kod v0.10 NIETKNIETY; branch droga-i4.

Ataki na payload:
  forge_swap  : payload innej klasy pod cudza etykieta (podmiana)
  forge_noise : statystyki policzone na losowej mieszance cech
                (smiec o realistycznych momentach)

Detektory (bez danych zewnetrznych):
  payload_centroid : E[x] payloadu spike-and-slab (suma_k w_k p_k mu_k)
  rank_consistency : Spearman miedzy podobienstwami cechowymi
      (centroid payloadu vs centroidy wlasnych klas odbiorcy)
      a podobienstwami slownymi (kotwica deklarowana vs slowa znane)
  canary_probe     : probna adopcja na kopii; sygnal = spadek sr. acc
      wlasnych klas odbiorcy
"""
import copy

import torch
import torch.nn.functional as F

from mars_cl_j import FeatureStatsKSparse


# ------------------------------------------------------------- falszerstwa
def forge_swap(payload_other):
    """Podmiana etykiety: payload innej klasy uzyty pod cudza nazwa."""
    return {k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in payload_other.items()}


def forge_noise(feats_pool, k=16, n=6000, seed=0):
    """Statystyki na losowej mieszance cech (bez struktury klasowej)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(feats_pool), generator=g)[:n]
    sub = feats_pool[idx.to(feats_pool.device)]
    s = FeatureStatsKSparse(k=k)
    y = torch.zeros(len(sub), dtype=torch.long, device=sub.device)
    s.update(sub, y, [0])
    return {"p": s.p[0].cpu(), "mean": s.mean[0].cpu(),
            "var": s.var[0].cpu(), "w": s.w[0].cpu(), "n": int(n)}


# --------------------------------------------------------------- detektory
def payload_centroid(payload, device="cpu"):
    p = payload["p"].to(device)
    mu = payload["mean"].to(device)
    w = payload["w"].to(device)
    return (w.unsqueeze(1) * (p * mu)).sum(0)


def _rank(x):
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb) / denom) if denom > 0 else 0.0

def rank_consistency(m, payload, declared_c, device="cpu"):
    """Spearman: podobienstwa cechowe payloadu do wlasnych klas odbiorcy
    vs podobienstwa slowne kotwicy deklarowanej do slow tych klas."""
    known = [c for c in m.seen_classes if c != declared_c]
    if len(known) < 3:
        return 0.0
    pc = payload_centroid(payload, device)
    sim_f, sim_w = [], []
    for r in known:
        own = {"p": m.stats.p[r], "mean": m.stats.mean[r],
               "w": m.stats.w[r]}
        oc = (own["w"].unsqueeze(1).to(device)
              * (own["p"].to(device) * own["mean"].to(device))).sum(0)
        sim_f.append(float(F.cosine_similarity(pc, oc, dim=0)))
        sim_w.append(float(F.cosine_similarity(
            m.word_vecs[declared_c].to(device),
            m.word_vecs[r].to(device), dim=0)))
    return spearman(torch.tensor(sim_f), torch.tensor(sim_w))


def canary_probe(m, classes, payloads, task_data_own, epochs, lr,
                 device, n_dream=2000):
    """Probna adopcja na KOPII; zwraca spadek sr. acc wlasnych klas
    odbiorcy (w pp; dodatni = szkoda)."""
    from mars_cl_n import class_accs
    own = list(m.seen_classes)
    before = class_accs(m, task_data_own, allowed=own)
    probe = copy.deepcopy(m)
    probe.adopt_classes(classes, payloads, epochs=epochs, lr=lr,
                        device=device, n_dream=n_dream)
    after = class_accs(probe, task_data_own, allowed=own)
    drop = sum(before[c] - after[c] for c in own) / len(own)
    return drop * 100.0


if __name__ == "__main__":
    # Smoke (CPU, syntetyczne): falszerstwa maja poprawne ksztalty,
    # detektory zwracaja liczby, kanarek dziala na kopii.
    from mars_collective import MarsCollective

    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    X = torch.randn(600, 784)
    y = torch.cat([torch.full((100,), c) for c in range(6)])
    tds = [{"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200],
            "Xte": X[:200], "yte": y[:200]},
           {"classes": [2, 3], "Xtr": X[200:400], "ytr": y[200:400],
            "Xte": X[200:400], "yte": y[200:400]}]
    m = MarsCollective(wv, dream_model="sparse", stats_k=4,
                       replay_per_class=16)
    m.init_representation(tds, epochs=1, lr=1e-3, device="cpu")
    for td in tds:
        m.learn_task(td, epochs=1, lr=1e-3, device="cpu")

    A = MarsCollective(wv, dream_model="sparse", stats_k=4,
                       replay_per_class=16)
    A.init_representation(tds, epochs=1, lr=1e-3, device="cpu")
    tdA = {"classes": [4, 5], "Xtr": X[400:], "ytr": y[400:]}
    A.learn_task(tdA, epochs=1, lr=1e-3, device="cpu")
    clean = A.export_class_stats(4, 100)
    swapped = forge_swap(A.export_class_stats(5, 100))
    noise = forge_noise(m.feats_batched(X), k=4, n=200)

    for pl in (clean, swapped, noise):
        assert pl["p"].shape == (4, 128)
        rc = rank_consistency(m, pl, 4, device="cpu")
        assert -1.0 <= rc <= 1.0
    seen_before = list(m.seen_classes)
    drop = canary_probe(m, [4], {4: clean}, tds, epochs=1, lr=1e-3,
                        device="cpu", n_dream=32)
    assert isinstance(drop, float)
    assert m.seen_classes == seen_before, "kanarek zmodyfikowal model!"
    print(f"Smoke OK. rank_consistency(clean)={rc:.2f}, "
          f"kanarek drop={drop:+.2f}pp")
