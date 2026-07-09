"""
cifar_cl.py -- F4: infrastruktura Split-CIFAR-10.

CIFAR-10: 32x32x3 (3072 wejsc), 10 klas o semantycznie bogatych nazwach
(airplane/dog/cat... -- warunek stosowalnosci mechanizmu z G1/F3b).
Kontrakt jak reszta pipeline'u: dane plaskie [N, 3072], backbone
[B, 3072] -> [B, BB_H=128].
"""
import os

import torch
import torch.nn as nn

from cl_common import BB_H

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def load_cifar10(device, root=DATA_ROOT):
    import torchvision
    tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    def prep(ds):
        X = (torch.tensor(ds.data, dtype=torch.float32)
             .permute(0, 3, 1, 2).reshape(len(ds.data), -1) / 255.0)
        y = torch.tensor(ds.targets, dtype=torch.long)
        return X.to(device), y.to(device)
    Xtr, ytr = prep(tr)
    Xte, yte = prep(te)
    return Xtr, ytr, Xte, yte


class CifarBackbone(nn.Module):
    """
    2 bloki conv (3->c1->c2, maxpool 32->16->8) + projekcja do BB_H.
    Uzywany jako LOSOWY ZAMROZONY trzon MARS-CL (F1d/F3b) oraz jako
    trenowalny trzon monolitu (MonoCifar).
    """
    def __init__(self, channels=(16, 32), backbone_hidden=BB_H):
        super().__init__()
        c1, c2 = channels
        self.channels = channels
        self.backbone_hidden = backbone_hidden
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 16x16
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 8x8
        )
        self.proj = nn.Sequential(
            nn.Linear(8 * 8 * c2, backbone_hidden), nn.ReLU())

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        f = self.conv(x)
        return self.proj(f.view(f.shape[0], -1))

    def mac_backbone(self):
        c1, c2 = self.channels
        conv1 = 3 * c1 * 9 * 32 * 32
        conv2 = c1 * c2 * 9 * 16 * 16
        proj = 8 * 8 * c2 * self.backbone_hidden
        return conv1 + conv2 + proj


class MonoCifar(nn.Module):
    """Monolit referencyjny CIFAR: CifarBackbone + glowica 10-way."""
    def __init__(self, channels=(16, 32)):
        super().__init__()
        self.backbone = CifarBackbone(channels=channels)
        self.head = nn.Linear(BB_H, 10)

    def forward(self, x):
        return self.head(self.backbone(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    m = CifarBackbone()
    x = torch.randn(8, 3072)
    assert m(x).shape == (8, BB_H)
    mono = MonoCifar()
    assert mono(x).shape == (8, 10)
    print(f"Smoke test OK. MAC backbone: {m.mac_backbone():,}")
