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
    print(f"Smoke OK. frakcja zer cech: {(f == 0).float().mean():.2f}")
