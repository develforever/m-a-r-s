"""
mars_cl_o.py -- Droga O: konsolidacja snem (DROGA_O_PLAN.md).
NOWY plik -- kod v0.9 NIETKNIETY; branch droga-o.

consolidate(m, mode): po zakonczonej sekwencji jeden "gleboki sen" --
nauka projekcji na snach WSZYSTKICH widzianych klas naraz (joint na
snach, bez dryfu kolejnosci). mode="reinit": projekcja od zera
(deterministycznie z seeda); mode="finetune": douczenie istniejacej.
Pody NIETKNIETE (czysta atrybucja efektu do projekcji).
"""
import torch


def consolidate(m, mode="reinit", n_dream=2000, epochs=15, lr=0.001,
                device="cpu", seed=0):
    assert mode in ("reinit", "finetune")
    seen = list(m.seen_classes)
    if not seen:
        return
    if mode == "reinit":
        import torch.nn as nn
        torch.manual_seed(seed)
        mods = (m.proj.modules() if hasattr(m.proj, "modules")
                else [m.proj])
        for mod in mods:
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
    feats = torch.cat([m.stats.sample(c, n_dream, device) for c in seen])
    y = torch.cat([torch.full((n_dream,), c, dtype=torch.long,
                              device=device) for c in seen])
    # old=[] -> bez dodatkowego miksowania snow (wszystkie klasy sa
    # juz w batchu jako "zadanie")
    m._fit_proj_feats(feats, y, [], seen, epochs, lr, device)


if __name__ == "__main__":
    # Smoke (CPU, syntetyczne): oba tryby zmieniaja projekcje,
    # reinit deterministyczny, forward dziala.
    import copy

    import torch.nn.functional as F
    from mars_collective import MarsCollective

    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    X = torch.randn(400, 784)
    y = torch.cat([torch.full((100,), c) for c in range(4)])
    tds = [{"classes": [0, 1], "Xtr": X[:200], "ytr": y[:200]},
           {"classes": [2, 3], "Xtr": X[200:], "ytr": y[200:]}]
    m = MarsCollective(wv, dream_model="sparse", stats_k=4,
                       replay_per_class=16)
    m.init_representation(tds, epochs=1, lr=1e-3, device="cpu")
    for td in tds:
        m.learn_task(td, epochs=1, lr=1e-3, device="cpu")
    w0 = next(m.proj.parameters()).detach().clone()

    mf = copy.deepcopy(m)
    consolidate(mf, mode="finetune", n_dream=64, epochs=1, device="cpu")
    assert not torch.allclose(w0, next(mf.proj.parameters()))

    mr1 = copy.deepcopy(m)
    consolidate(mr1, mode="reinit", n_dream=64, epochs=1, device="cpu",
                seed=7)
    mr2 = copy.deepcopy(m)
    consolidate(mr2, mode="reinit", n_dream=64, epochs=1, device="cpu",
                seed=7)
    assert torch.allclose(next(mr1.proj.parameters()),
                          next(mr2.proj.parameters())), \
        "reinit niedeterministyczny"
    assert mr1.forward(X[:8]).shape == (8, 10)
    print("Smoke OK: konsolidacja reinit/finetune dziala.")
