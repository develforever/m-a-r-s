"""
cl_common.py -- Droga F: wspolna infrastruktura continual learning.

Zawiera (uzywane przez F0 baseline'y i F1 MARS-CL):
  - splity zadan (Split-MNIST / Split-Fashion: 5 zadan x 2 klasy),
  - dwa protokoly ewaluacji:
      task-IL : znana etykieta zadania -> argmax w klasach zadania (latwy),
      class-IL: bez etykiety zadania -> argmax po klasach WIDZIANYCH (glowny),
  - metryki CL z macierzy R[t][j] (acc na zadaniu j po nauce zadania t):
      ACC = srednia po zadaniach na koncu sekwencji,
      Forgetting = sredni spadek: max_t R[t][j] - R[T-1][j],
      BWT = sredni wplyw nowych zadan na stare: R[T-1][j] - R[j][j],
  - zbalansowany bufor replay,
  - MonoS2: monolityczny model referencyjny (backbone S2 + glowica 10-way) --
    ten sam budzet cech co przyszly MARS-CL, zeby porownania byly czyste.
"""
import torch
import torch.nn as nn

from mars_v2_slim import SlimCNNBackbone

TASKS5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
BB_H = 128
S2 = dict(channels=(8, 16), downsample="maxpool", depthwise=False)


class MonoS2(nn.Module):
    """Monolit referencyjny: backbone S2 + jedna glowica 10-way."""
    def __init__(self):
        super().__init__()
        self.backbone = SlimCNNBackbone(backbone_hidden=BB_H, **S2)
        self.head = nn.Linear(BB_H, 10)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------- splity
def split_task(X, y, classes):
    """Podzbior probek nalezacych do podanych klas."""
    mask = torch.zeros_like(y, dtype=torch.bool)
    for c in classes:
        mask |= (y == c)
    return X[mask], y[mask]


def make_task_data(Xtr, ytr, Xte, yte, tasks=TASKS5):
    """Lista slownikow per zadanie: train/test + klasy."""
    out = []
    for classes in tasks:
        Xt, yt = split_task(Xtr, ytr, classes)
        Xv, yv = split_task(Xte, yte, classes)
        out.append({"classes": list(classes),
                    "Xtr": Xt, "ytr": yt, "Xte": Xv, "yte": yv})
    return out


# ------------------------------------------------------------- ewaluacja
def masked_argmax(logits, allowed):
    """Argmax ograniczony do dozwolonych klas (maska -inf na reszcie)."""
    m = torch.full_like(logits, float("-inf"))
    m[:, list(allowed)] = logits[:, list(allowed)]
    return m.argmax(dim=1)


@torch.no_grad()
def eval_protocols(forward_fn, task_data, upto, seen_classes, batch=2048):
    """
    Ewaluacja po nauce zadania `upto` na zadaniach 0..upto.
    Zwraca (row_class_il, row_task_il): listy acc per zadanie.
    forward_fn: x -> logity [B, 10].
    """
    row_c, row_t = [], []
    for j in range(upto + 1):
        td = task_data[j]
        preds_c, preds_t = [], []
        for s in range(0, len(td["Xte"]), batch):
            logits = forward_fn(td["Xte"][s:s + batch])
            preds_c.append(masked_argmax(logits, seen_classes))
            preds_t.append(masked_argmax(logits, td["classes"]))
        pc = torch.cat(preds_c)
        pt = torch.cat(preds_t)
        row_c.append(round((pc == td["yte"]).float().mean().item(), 4))
        row_t.append(round((pt == td["yte"]).float().mean().item(), 4))
    return row_c, row_t


def cl_metrics(R):
    """
    Metryki z dolnotrojkatnej macierzy R (lista wierszy rosnacej dlugosci).
    R[t][j] = acc na zadaniu j po nauce zadan 0..t.
    """
    T = len(R)
    final = R[-1]
    acc = sum(final) / T
    forg, bwt = [], []
    for j in range(T - 1):
        best = max(R[t][j] for t in range(j, T))
        forg.append(best - final[j])
        bwt.append(final[j] - R[j][j])
    return {
        "ACC": round(acc, 4),
        "forgetting": round(sum(forg) / len(forg), 4) if forg else 0.0,
        "BWT": round(sum(bwt) / len(bwt), 4) if bwt else 0.0,
        "final_per_task": final,
    }


# ----------------------------------------------------------------- replay
def balanced_buffer(task_data, upto, size, seed):
    """
    Zbalansowany bufor replay: size probek TRENINGOWYCH, po rowno na kazda
    widziana klase (zadania 0..upto). Deterministyczny dla seeda.
    """
    g = torch.Generator().manual_seed(seed * 1000 + upto)
    seen = [c for j in range(upto + 1) for c in task_data[j]["classes"]]
    per = max(size // len(seen), 1)
    Xs, ys = [], []
    for j in range(upto + 1):
        td = task_data[j]
        for c in td["classes"]:
            idx = (td["ytr"] == c).nonzero(as_tuple=True)[0]
            pick = idx[torch.randperm(len(idx), generator=g)[:per]]
            Xs.append(td["Xtr"][pick])
            ys.append(td["ytr"][pick])
    return torch.cat(Xs), torch.cat(ys)


# -------------------------------------------------------------------- EWC
class EWCState:
    """Online EWC: skumulowany Fisher diagonalny + snapshot parametrow."""
    def __init__(self, model):
        self.fisher = {n: torch.zeros_like(p) for n, p
                       in model.named_parameters()}
        self.star = {}

    @torch.no_grad()
    def _snapshot(self, model):
        self.star = {n: p.detach().clone() for n, p
                     in model.named_parameters()}

    def update_fisher(self, model, X, y, batch=512, max_batches=20):
        """Empiryczny Fisher diagonalny na danych zadania (prawdziwe etykiety)."""
        model.eval()
        crit = nn.CrossEntropyLoss()
        n_b = 0
        for s in range(0, len(X), batch):
            if n_b >= max_batches:
                break
            model.zero_grad()
            loss = crit(model(X[s:s + batch]), y[s:s + batch])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2
            n_b += 1
        for n in self.fisher:
            self.fisher[n] /= max(n_b, 1)
        self._snapshot(model)
        model.zero_grad()

    def penalty(self, model):
        if not self.star:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = 0.0
        for n, p in model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.star[n]) ** 2).sum()
        return loss


if __name__ == "__main__":
    # Smoke: splity, metryki, bufor.
    y = torch.randint(0, 10, (1000,))
    X = torch.randn(1000, 784)
    td = make_task_data(X, y, X[:200], y[:200])
    assert len(td) == 5 and td[2]["classes"] == [4, 5]
    R = [[0.99], [0.80, 0.98], [0.60, 0.70, 0.97]]
    m = cl_metrics(R)
    # cl_metrics zaokragla do 4 miejsc -> tolerancja 1e-3
    assert abs(m["ACC"] - (0.60 + 0.70 + 0.97) / 3) < 1e-3
    assert abs(m["forgetting"] - ((0.99 - 0.60) + (0.98 - 0.70)) / 2) < 1e-3
    bx, by = balanced_buffer(td, 1, 200, seed=0)
    assert set(by.tolist()) <= {0, 1, 2, 3}
    print("Smoke test OK.")
