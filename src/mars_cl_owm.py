"""
mars_cl_owm.py -- H1: Orthogonal Weight Modification na projekcji semantycznej.

IDEA (DROGA_H_PLAN.md; metoda: Zeng et al. 2019, OWM):
  Aktualizacje wag projekcji rzutowane w podprzestrzen ORTOGONALNA do cech
  starych klas: grad_W <- grad_W @ P. Dla warstwy LINIOWEJ gwarancja jest
  scisla: nowy trening nie zmienia mapowania zadnego wejscia, ktore wspiera
  podprzestrzen przeszlosci. To matematyczny hamulec dryfu -- komplementarny
  do snu parametrycznego (sen dostarcza NEGATYWY dla nowych slow, OWM
  chroni STARE kierunki).

DECYZJE IMPLEMENTACYJNE (jawne):
  1. P liczone rekurencja RLS (bez inwersji macierzy):
       P <- P - (P x)(P x)^T / (alpha + x^T P x),  P0 = I.
     Aktualizacja cechami zadania W TRAKCIE jego nauki (przed odrzuceniem
     danych) -- zero przechowywania, spojnie z reszta serii F.
  2. Bias projekcji NIE jest chroniony przez P (rzutnik dziala po stronie
     wejsc) -- bias ZAMROZONY po zadaniu 0.
  3. HIPOTEZA H1: z OWM dlugi trening projekcji przestaje szkodzic
     (F3b: 15 epok > dryf > 4 epoki) -- wiec epochs_proj=15 BEZ moderacji.
  4. Znane ryzyko OWM: null-space kurczy sie z zadaniami (spadek
     plastycznosci) -- mierzone krzywa R[t][t] (acc zadania tuz po nauce).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_common import BB_H
from mars_cl_f3 import MarsCLSemanticF3
from mars_cl_semantic import TEMP


class MarsCLSemanticOWM(MarsCLSemanticF3):
    """
    f3_sem + rzutnik ortogonalny na gradientach projekcji.
      owm_alpha : regularyzacja RLS (mniejsza = ostrzejszy rzutnik)
      use_dream : czy zachowac sen parametryczny (negatywy) przy fit_proj
    """
    def __init__(self, word_vecs, owm_alpha=1.0, use_dream=True,
                 owm_samples=2000, **kw):
        super().__init__(word_vecs, **kw)
        self.owm_alpha = owm_alpha
        self.use_dream = use_dream
        self.owm_samples = owm_samples
        self.register_buffer("P", torch.eye(BB_H))
        self._tasks_seen = 0

    # ------------------------------------------------------------ rzutnik
    @torch.no_grad()
    def _update_P(self, feats):
        """Rekurencja RLS na probce cech zadania (po jego nauce)."""
        n = min(self.owm_samples, len(feats))
        idx = torch.randperm(len(feats), device=feats.device)[:n]
        for x in feats[idx]:
            Px = self.P @ x
            self.P -= torch.outer(Px, Px) / (self.owm_alpha + x @ Px)

    # --------------------------------------------------- fit_proj z OWM
    def _fit_proj_feats(self, feats, y, old_classes, all_classes,
                        epochs, lr, device, batch=512):
        """Kopia rodzica + rzutowanie gradientow (grad_W @ P) przed step."""
        W = torch.stack([self.word_vecs[c].to(device) for c in all_classes])
        c2i = {c: i for i, c in enumerate(all_classes)}
        yi = torch.tensor([c2i[int(v)] for v in y.tolist()], device=device)
        n_new = len(all_classes) - len(old_classes)
        k_per_old = (max(batch // max(n_new, 1) // 2, 32)
                     if (old_classes and self.use_dream) else 0)
        crit = nn.CrossEntropyLoss()
        n_epochs = self.epochs_proj if self.epochs_proj else epochs
        for p in self.proj.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(self.proj.parameters(), lr=lr)
        protect = self._tasks_seen > 0
        for _ in range(n_epochs):
            perm = torch.randperm(len(feats), device=device)
            for s in range(0, len(feats), batch):
                idx = perm[s:s + batch]
                xb, yb = feats[idx], yi[idx]
                if k_per_old:
                    df, dy = self.stats.replay_batch(old_classes, k_per_old,
                                                     device)
                    dyi = torch.tensor([c2i[int(v)] for v in dy.tolist()],
                                       device=device)
                    xb = torch.cat([xb, df])
                    yb = torch.cat([yb, dyi])
                emb = F.normalize(self.proj(xb), dim=1)
                loss = crit(emb @ W.T / TEMP, yb)
                opt.zero_grad(); loss.backward()
                if protect:
                    with torch.no_grad():
                        self.proj.weight.grad @= self.P     # OWM
                        self.proj.bias.grad.zero_()          # bias frozen
                opt.step()
        for p in self.proj.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------ nauka
    def learn_task(self, td, epochs, lr, device):
        super().learn_task(td, epochs, lr, device)
        with torch.no_grad():
            feats = self.feats_batched(td["Xtr"])
        self._update_P(feats)
        self._tasks_seen += 1


if __name__ == "__main__":
    # Smoke: rzutnik dziala -- po update P, P@x ~ 0 dla widzianych x.
    torch.manual_seed(0)
    wv = {c: F.normalize(torch.randn(50), dim=0) for c in range(10)}
    m = MarsCLSemanticOWM(wv, owm_alpha=0.1, stats_k=1)
    X = torch.randn(300, 784).abs()
    y = torch.cat([torch.full((150,), 0), torch.full((150,), 1)])
    td0 = {"classes": [0, 1], "Xtr": X, "ytr": y}
    m.init_representation([td0], epochs=1, lr=1e-3, device="cpu")
    m.learn_task(td0, epochs=1, lr=1e-3, device="cpu")
    feats = m.feats_batched(X)
    resid = (feats @ m.P).norm(dim=1) / feats.norm(dim=1).clamp_min(1e-9)
    print(f"Smoke test OK. Sredni wzgledny residual P@x na widzianych "
          f"cechach: {resid.mean():.4f} (blisko 0 = rzutnik chroni)")
    assert m.forward(X[:8]).shape == (8, 10)
    assert m._tasks_seen == 1
