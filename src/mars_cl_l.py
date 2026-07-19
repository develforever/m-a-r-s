"""
mars_cl_l.py -- Droga L: fork tozsamosci -- zamrozony pretrenowany
backbone pod niezmienionym mechanizmem (DROGA_L_PLAN.md).

NOWY plik -- kod v0.6 NIETKNIETY; na branchu droga-l.

PretrainedBackbone: ResNet18-ImageNet (torchvision, IMAGENET1K_V1,
zamrozony, zawsze eval) -> ZAMROZONA LOSOWA projekcja 512->BB_H=128
(z seeda agenta) -> ReLU. Interfejs identyczny z CifarBackbone
([B, 3072] w normalizacji CIFAR -> [B, 128], cechy nieujemne z zerami),
wiec caly stos (statystyki sparse, sen, pody, projekcja semantyczna,
payload protokolu I) dziala BEZ ZMIAN. Wspolny seed nadal synchronizuje
agentow: czesc pretrained identyczna z definicji, projekcja z seeda.

Wejscie: denorm CIFAR -> norm ImageNet -> resize 224 (bilinear).
Forward mikro-batchuje wewnetrznie (chunk=128), zeby feats_batched
(batch=2048) nie przekroczyl 4 GB VRAM przy 224x224.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_common import BB_H
from mars_cl_j import CIFAR10_MEAN, CIFAR10_STD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PretrainedBackbone(nn.Module):
    def __init__(self, out_dim=BB_H, resize=224, chunk=128):
        super().__init__()
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet18(weights=w)
        self.features = nn.Sequential(*list(m.children())[:-1])  # bez fc
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()
        # losowa zamrozona projekcja 512 -> out_dim (z aktualnego seeda)
        self.reduce = nn.Linear(512, out_dim, bias=False)
        for p in self.reduce.parameters():
            p.requires_grad = False
        self.resize = resize
        self.chunk = chunk
        cm = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        cs = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        im = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        is_ = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("cifar_mean", cm)
        self.register_buffer("cifar_std", cs)
        self.register_buffer("imnet_mean", im)
        self.register_buffer("imnet_std", is_)

    def train(self, mode=True):
        """Backbone zamrozony: BN zostaje w eval na zawsze."""
        return self

    def _forward_chunk(self, x):
        x = x.view(-1, 3, 32, 32)
        x = x * self.cifar_std + self.cifar_mean          # denorm CIFAR
        x = (x - self.imnet_mean) / self.imnet_std        # norm ImageNet
        x = F.interpolate(x, size=self.resize, mode="bilinear",
                          align_corners=False)
        f = self.features(x).flatten(1)                   # [B, 512]
        return torch.relu(self.reduce(f))                 # [B, 128], >=0

    def forward(self, x):
        with torch.no_grad():
            outs = [self._forward_chunk(x[s:s + self.chunk])
                    for s in range(0, len(x), self.chunk)]
        return torch.cat(outs)


# ------------------------------------------------------ szybka sciezka
# ResNet na 1050 Ti jest za wolny, by liczyc cechy w kazdym przebiegu
# (feats_batched + eval po kazdym tasku = wielokrotne przejscia 60k
# obrazow @224). Rozwiazanie BEZ zmiany semantyki: czesc pretrained
# jest deterministyczna i wspolna dla wszystkich seedow/agentow, wiec
# jej wyjscie [N, 512] liczymy RAZ i cache'ujemy na dysku; per seed
# zostaje tylko losowa projekcja 512->128 + ReLU (ReducedBackbone).
# Matematycznie rownowazne PretrainedBackbone (zlozenie tych samych
# funkcji), pre-rejestracja DROGA_L_PLAN bez zmian.

FEATS_CACHE = os.path.join(os.path.dirname(__file__), "..", "data",
                           "cifar_resnet18_224_feats.pt")


class ReducedBackbone(nn.Module):
    """Losowa zamrozona projekcja 512->BB_H + ReLU; wejscie = cechy
    resnet18 z cache (extract_or_load_cifar_feats)."""

    def __init__(self, out_dim=BB_H):
        super().__init__()
        self.reduce = nn.Linear(512, out_dim, bias=False)
        for p in self.reduce.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        return self

    def forward(self, x):
        with torch.no_grad():
            return torch.relu(self.reduce(x))


def extract_or_load_cifar_feats(device, resize=224, chunk=128,
                                cache=FEATS_CACHE):
    """Jednorazowa ekstrakcja cech resnet18 (512-d) dla calego
    CIFAR-10n; wynik cache'owany w data/. Zwraca (Ftr, ytr, Fte, yte)."""
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        return (d["Ftr"].to(device), d["ytr"].to(device),
                d["Fte"].to(device), d["yte"].to(device))
    from mars_cl_j import load_cifar10_norm
    print(f"[L] Jednorazowa ekstrakcja cech resnet18@{resize} "
          f"(pozniejsze runy czytaja cache: {os.path.abspath(cache)})")
    Xtr, ytr, Xte, yte = load_cifar10_norm(device)
    torch.manual_seed(0)   # deterministycznie; reduce nieuzywane
    bb = PretrainedBackbone(resize=resize, chunk=chunk).to(device)
    outs_all = []
    for X in (Xtr, Xte):
        outs = []
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk].view(-1, 3, 32, 32)
            x = x * bb.cifar_std + bb.cifar_mean
            x = (x - bb.imnet_mean) / bb.imnet_std
            x = F.interpolate(x, size=resize, mode="bilinear",
                              align_corners=False)
            with torch.no_grad():
                outs.append(bb.features(x).flatten(1))
            if (s // chunk) % 40 == 0:
                print(f"    ekstrakcja: {s}/{len(X)}", flush=True)
        outs_all.append(torch.cat(outs))
    Ftr, Fte = outs_all
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save({"Ftr": Ftr.cpu(), "ytr": ytr.cpu(),
                "Fte": Fte.cpu(), "yte": yte.cpu()}, cache)
    print(f"[L] Cache zapisany ({Ftr.shape[0]}+{Fte.shape[0]} x 512).")
    return Ftr, ytr, Fte, yte


if __name__ == "__main__":
    # Smoke (CPU/GPU, male wejscie): ksztalt, nieujemnosc, determinizm
    # projekcji z seeda, BN w eval mimo .train().
    torch.manual_seed(0)
    b1 = PretrainedBackbone(resize=64, chunk=32)
    torch.manual_seed(0)
    b2 = PretrainedBackbone(resize=64, chunk=32)
    assert torch.allclose(b1.reduce.weight, b2.reduce.weight), \
        "projekcja nie jest deterministyczna z seeda"
    b1.train()
    assert not b1.features.training, "BN wyszedl z eval"
    X = torch.randn(70, 3072)
    f = b1(X)
    assert f.shape == (70, 128) and (f >= 0).all()
    assert (f == 0).float().mean() > 0.0, "brak zer po ReLU?"
    assert not any(p.requires_grad for p in b1.parameters())
    # ReducedBackbone: rownowaznik na cechach 512-d
    torch.manual_seed(0)
    rb1 = ReducedBackbone()
    torch.manual_seed(0)
    rb2 = ReducedBackbone()
    assert torch.allclose(rb1.reduce.weight, rb2.reduce.weight)
    fr = rb1(torch.randn(70, 512))
    assert fr.shape == (70, 128) and (fr >= 0).all()
    print(f"Smoke OK. frakcja zer cech: {(f == 0).float().mean():.2f}; "
          f"ReducedBackbone OK.")
