"""
mars_v2_pc.py -- D7: Predictive coding routing (dialog router<->pody).

MOTYWACJA (z D7_PLAN.md):
  D4/D5 pokazaly, ze na shared backbone routing nie poprawi sie informacja,
  ktora juz jest w cechach. Predictive coding wnosi NOWY kanal: blad
  rekonstrukcji. Pod, ktory najlepiej odtwarza probke (wejscie lub cechy),
  "wyjasnia" ja -- to sygnal niezalezny od logitow klasyfikacji.

KLUCZOWE DECYZJE PROJEKTOWE:
  1. Dekodery to OSOBNY modul (PCDecoders), NIE czesc modelu. Trening fazy 1.5
     nie dotyka backbone'u ani podow (lekcja z D5: nie ruszac wspolnej
     reprezentacji). Model bazowy (CNN z D6 / S3 z D6b) trenowany standardowo
     przez train_phased i ZAMROZONY przed faza 1.5.
  2. Dekoder poda i trenowany WYLACZNIE na probkach realnie routowanych do
     poda i (spojnosc z train_phased). Dzieki temu dekoder i "zna" swoj
     klaster -- probki spoza niego rekonstruuje gorzej = sygnal dyskryminacyjny.
  3. Dekoder minimalny: jedna warstwa afiniczna pod_hidden -> target_dim
     (stacked, jak pody). Liczy sie SYGNAL WZGLEDNY miedzy podami, nie
     absolutna jakosc rekonstrukcji.

Warianty targetu (D7_PLAN.md sekcja 3):
  D7a: target = features (backbone_hidden)  -- tani, spojny z v2
  D7b: target = x (784)                     -- "obraz za mgla", bogatszy

Warianty uzycia sygnalu:
  hard: argmin bledu (zastepuje router)
  fuse: log_softmax(route_logits) + lambda * log_softmax(-e)
        (log-softmaxy zrownuja skale obu kanalow; lambda ze sweepu)
  iter: router daje top-k kandydatow, blad rekonstrukcji rozstrzyga
"""
import torch
import torch.nn as nn

from mars_v2 import N_IN, N_CLASSES


class PCDecoders(nn.Module):
    """
    Stacked dekodery per pod: [n_pods, pod_hidden, target_dim] (+ bias).
    Wejscie: ukryta reprezentacja poda (pod_hidden=24), wyjscie: rekonstrukcja
    targetu (features albo x). Analogia strukturalna do pod_W2/pod_b2.
    """
    def __init__(self, n_pods=N_CLASSES, pod_hidden=24, target_dim=N_IN):
        super().__init__()
        self.n_pods = n_pods
        self.pod_hidden = pod_hidden
        self.target_dim = target_dim
        self.W = nn.Parameter(torch.randn(n_pods, pod_hidden, target_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_pods, target_dim))

    def forward_all(self, pod_h_all):
        """Rekonstrukcje WSZYSTKICH podow. [B, n_pods, ph] -> [B, n_pods, T]."""
        return torch.einsum("bpk,pkt->bpt", pod_h_all, self.W) + self.b

    def forward_selected(self, pod_h, pod_ids):
        """Rekonstrukcja wybranych podow. [B, ph], [B] -> [B, T]."""
        W = self.W[pod_ids]                        # [B, ph, T]
        b = self.b[pod_ids]                        # [B, T]
        return torch.bmm(pod_h.unsqueeze(1), W).squeeze(1) + b


# ------------------------------------------------------------------ helpers
def pod_hidden_selected(model, feats, pod_ids):
    """Ukryta warstwa WYBRANYCH podow (jak pierwsza polowa pod_forward). [B, ph]."""
    W1 = model.pod_W1[pod_ids]
    b1 = model.pod_b1[pod_ids]
    return torch.relu(torch.bmm(feats.unsqueeze(1), W1).squeeze(1) + b1)


def pod_hidden_all(model, feats):
    """Ukryte warstwy WSZYSTKICH podow. [B, bb_h] -> [B, n_pods, ph]."""
    h = torch.einsum("bh,phk->bpk", feats, model.pod_W1) + model.pod_b1
    return torch.relu(h)


def recon_errors(model, decoders, feats, target):
    """
    Blad rekonstrukcji kazdego poda dla kazdej probki (MSE per wymiar).
    Zwraca e [B, n_pods] -- macierz "jak zle pod p wyjasnia probke b".
    """
    h_all = pod_hidden_all(model, feats)               # [B, n_pods, ph]
    x_hat = decoders.forward_all(h_all)                # [B, n_pods, T]
    return ((x_hat - target.unsqueeze(1)) ** 2).mean(dim=2)


# ------------------------------------------------------------ routing PC
def route_pc_hard(e):
    """Czysty predictive coding: pod z najmniejszym bledem. [B]."""
    return e.argmin(dim=1)

def route_pc_fuse(route_logits, e, lam):
    """
    Dialog: router proponuje, blad rekonstrukcji koryguje.
    Oba kanaly w log-przestrzeni prawdopodobienstw (wspolna skala).
    """
    combined = (torch.log_softmax(route_logits, dim=1)
                + lam * torch.log_softmax(-e, dim=1))
    return combined.argmax(dim=1)

def route_pc_iter(route_logits, e, k):
    """
    Pelny dialog, 2 rundy: router zaweza do top-k kandydatow,
    blad rekonstrukcji rozstrzyga wsrod nich (argmin po zamaskowaniu reszty).
    """
    topk_ids = route_logits.topk(k, dim=1).indices          # [B, k]
    e_masked = torch.full_like(e, float("inf"))
    e_masked.scatter_(1, topk_ids, e.gather(1, topk_ids))
    return e_masked.argmin(dim=1)


# ------------------------------------------------------------ faza 1.5
def train_decoders(model, decoders, Xtr, target_mode, epochs=10, lr=0.001,
                   batch=512, device="cpu"):
    """
    Faza 1.5: trening dekoderow przy CALKOWICIE zamrozonym modelu.
    Kazdy dekoder uczy sie rekonstruowac target TYLKO z probek, ktore router
    realnie do niego kieruje (spojnosc z train_phased; patrz naglowek pliku).

    target_mode: "features" (D7a) albo "input" (D7b).
    Gradient plynie WYLACZNIE do decoders.W / decoders.b -- wejscia dekodera
    (feats, pod_h) liczone pod no_grad, wiec backbone/pody nietykane.
    """
    assert target_mode in ("features", "input"), target_mode
    model.eval()
    decoders.train()
    opt = torch.optim.Adam(decoders.parameters(), lr=lr)
    mse = nn.MSELoss()

    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            x = Xtr[idx]
            with torch.no_grad():
                feats = model.features(x)
                pod_ids = model.route(feats)             # realny routing
                pod_h = pod_hidden_selected(model, feats, pod_ids)
                target = feats if target_mode == "features" else x
            x_hat = decoders.forward_selected(pod_h, pod_ids)
            loss = mse(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
    return decoders


# ------------------------------------------------------------------- eval
def evaluate_pc(model, decoders, Xte, yte, target_mode, variant, param=None):
    """
    Ewaluacja jednego wariantu routingu PC.
      variant: "hard" | "fuse" (param=lambda) | "iter" (param=k)
    Zwraca dict: routing_acc, system_acc, recon_mse (sredni blad WYBRANEGO poda).
    """
    model.eval(); decoders.eval()
    with torch.no_grad():
        feats = model.features(Xte)
        target = feats if target_mode == "features" else Xte
        e = recon_errors(model, decoders, feats, target)
        logits = model.route_logits(feats)

        if variant == "hard":
            ids = route_pc_hard(e)
        elif variant == "fuse":
            ids = route_pc_fuse(logits, e, lam=param)
        elif variant == "iter":
            ids = route_pc_iter(logits, e, k=param)
        else:
            raise ValueError(variant)

        routing_acc = (ids == yte).float().mean().item()
        out = model.pod_forward(feats, ids)
        system_acc = (out.argmax(1) == yte).float().mean().item()
        recon_mse = e.gather(1, ids.unsqueeze(1)).mean().item()
    return {"routing": routing_acc, "system": system_acc,
            "recon_mse": round(recon_mse, 6)}


def mac_pc(model, target_dim, n_evaluated):
    """
    MAC per probka dla routingu PC (uczciwy koszt DODATKOWY vs top-1):
      fixed    = backbone + routing (jak zawsze)
      hidden   = pierwsza warstwa poda dla n_evaluated podow (bb_h * ph kazdy)
      decoder  = n_evaluated dekoderow (ph * target_dim kazdy)
      finalny pod = druga warstwa wybranego poda (ph * n_out); jego hidden
                    juz policzony wyzej.
    hard/fuse: n_evaluated = n_pods (10). iter: n_evaluated = k.
    """
    base = model.mac_per_sample_top1()
    fixed = base["backbone"] + base["routing"]
    hidden = n_evaluated * model.backbone_hidden * model.pod_hidden
    dec = n_evaluated * model.pod_hidden * target_dim
    final = model.pod_hidden * model.n_out
    total = fixed + hidden + dec + final
    return {"fixed": fixed, "pod_hidden": hidden, "decoders": dec,
            "final_pod": final, "total": total,
            "overhead_vs_top1": total - base["total_top1"]}


if __name__ == "__main__":
    # Smoke: ksztalty + sanity routingu PC na losowych wagach.
    from mars_v2 import MarsV2System
    torch.manual_seed(0)
    m = MarsV2System(backbone_hidden=128, emb_dim=32, pod_hidden=24)
    dec_a = PCDecoders(pod_hidden=24, target_dim=128)   # D7a: features
    dec_b = PCDecoders(pod_hidden=24, target_dim=784)   # D7b: input
    x = torch.randn(16, N_IN)
    feats = m.features(x)
    for dec, tgt in ((dec_a, feats), (dec_b, x)):
        e = recon_errors(m, dec, feats, tgt)
        assert e.shape == (16, 10), e.shape
        assert route_pc_hard(e).shape == (16,)
        assert route_pc_fuse(m.route_logits(feats), e, 0.5).shape == (16,)
        assert route_pc_iter(m.route_logits(feats), e, 3).shape == (16,)
    # iter z k=10 == hard (pelny zbior kandydatow)
    e = recon_errors(m, dec_b, feats, x)
    assert torch.equal(route_pc_iter(m.route_logits(feats), e, 10),
                       route_pc_hard(e))
    print("Smoke test OK.")
    print("MAC PC (D7b, hard, 10 podow):", mac_pc(m, 784, 10))
    print("MAC PC (D7b, iter k=3):      ", mac_pc(m, 784, 3))
