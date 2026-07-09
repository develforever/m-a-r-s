"""
mars_v2_slim.py -- D6b: odchudzone warianty CNN backbone dla M.A.R.S. v2.

MOTYWACJA (z D6B_PLAN.md):
  D6 dal SYGNAL+ (+2.38pp Fashion), ale za ~19.7x MAC (215.6k -> 4.25M).
  Pytanie D6b: ile z zysku przetrwa przy MAC bliskim MLP? Jesli duzo --
  CNN to dzwignia efektywnosciowa (lokalnosc/inwariancja), nie surowy compute.

KLUCZOWA DECYZJA PROJEKTOWA (jak w D6):
  Zmieniamy WYLACZNIE backbone. Kontrakt identyczny: [B, 784] -> [B, bb_h].
  Cala reszta v2 (routing head, pody, train_phased, evaluate) REUZYTA
  z mars_v2.py -- porownanie z D6/D1c jest czyste.

Osie odchudzania (ortogonalne, patrz D6B_PLAN.md sekcja 2):
  - channels:   (32,64) pelny D6 -> (16,32) half -> (8,16) quarter
  - downsample: "maxpool" (conv s=1 + MaxPool2d, jak D6)
                "stride"  (conv s=2, bez poolingu -- conv liczy 4x mniej pozycji)
  - depthwise:  conv2 jako depthwise(3x3, groups=c1) + pointwise(1x1)
                zamiast pelnej konwolucji c1->c2
"""
import torch
import torch.nn as nn

from mars_v2 import MarsV2System, N_IN, N_CLASSES


class SlimCNNBackbone(nn.Module):
    """
    Parametryzowany backbone CNN: 2 stopnie (28x28 -> 14x14 -> 7x7) + projekcja
    FC do wspolnego wektora `backbone_hidden`. Geometria wyjscia STALA
    (7*7*c2), niezaleznie od trybu downsamplingu -- rozni sie tylko koszt.

    Kontrakt zgodny z backbone v2:
        wejscie  [B, 784]  (plaski obraz)
        wyjscie  [B, backbone_hidden]
    """
    def __init__(self, backbone_hidden=128, channels=(16, 32),
                 downsample="maxpool", depthwise=False):
        super().__init__()
        assert downsample in ("maxpool", "stride"), downsample
        c1, c2 = channels
        self.channels = channels
        self.downsample = downsample
        self.depthwise = depthwise

        s = 1 if downsample == "maxpool" else 2

        # --- stopien 1: 1 -> c1, 28x28 -> 14x14 ---
        stage1 = [nn.Conv2d(1, c1, 3, stride=s, padding=1),
                  nn.BatchNorm2d(c1), nn.ReLU()]
        if downsample == "maxpool":
            stage1.append(nn.MaxPool2d(2))

        # --- stopien 2: c1 -> c2, 14x14 -> 7x7 ---
        if depthwise:
            stage2 = [nn.Conv2d(c1, c1, 3, stride=s, padding=1, groups=c1),
                      nn.Conv2d(c1, c2, 1)]
        else:
            stage2 = [nn.Conv2d(c1, c2, 3, stride=s, padding=1)]
        stage2 += [nn.BatchNorm2d(c2), nn.ReLU()]
        if downsample == "maxpool":
            stage2.append(nn.MaxPool2d(2))

        self.conv = nn.Sequential(*stage1, *stage2)
        self.proj = nn.Sequential(
            nn.Linear(7 * 7 * c2, backbone_hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        f = self.conv(x)
        f = f.view(f.shape[0], -1)
        return self.proj(f)


class MarsV2SlimSystem(MarsV2System):
    """
    v2 ze SlimCNNBackbone. Dziedziczy CALA logike po MarsV2System,
    podmienia backbone i wzor MAC (zalezny od trybu downsample/depthwise).
    """
    def __init__(self, n_in=N_IN, backbone_hidden=128, n_pods=N_CLASSES,
                 emb_dim=32, pod_hidden=24, n_out=N_CLASSES,
                 channels=(16, 32), downsample="maxpool", depthwise=False):
        super().__init__(n_in=n_in, backbone_hidden=backbone_hidden,
                         n_pods=n_pods, emb_dim=emb_dim, pod_hidden=pod_hidden,
                         n_out=n_out)
        self.channels = channels
        self.downsample = downsample
        self.depthwise = depthwise
        self.backbone = SlimCNNBackbone(backbone_hidden=backbone_hidden,
                                        channels=channels,
                                        downsample=downsample,
                                        depthwise=depthwise)

    def mac_per_sample_top1(self):
        """
        MAC per probka. Wzor conv: out_ch * in_ch_per_group * k*k * H_out*W_out.
        MaxPool/BN/ReLU pomijane (konwencja jak w D6/B1b).
        """
        c1, c2 = self.channels
        if self.downsample == "maxpool":
            conv1_hw = 28 * 28   # conv1 s=1 liczy pelne 28x28, pool tnie potem
            conv2_hw = 14 * 14
        else:  # stride: conv liczy tylko pozycje wyjsciowe
            conv1_hw = 14 * 14
            conv2_hw = 7 * 7
        conv1 = 1 * c1 * 9 * conv1_hw
        if self.depthwise:
            conv2 = c1 * 9 * conv2_hw + c1 * c2 * conv2_hw  # dw 3x3 + pw 1x1
        else:
            conv2 = c1 * c2 * 9 * conv2_hw
        proj = 7 * 7 * c2 * self.backbone_hidden
        backbone_mac = conv1 + conv2 + proj
        routing_mac = self.backbone_hidden * self.emb_dim + self.emb_dim * self.n_pods
        pod_mac = self.backbone_hidden * self.pod_hidden + self.pod_hidden * self.n_out
        return {
            "backbone": backbone_mac,
            "routing": routing_mac,
            "pod": pod_mac,
            "total_top1": backbone_mac + routing_mac + pod_mac,
        }


if __name__ == "__main__":
    # Smoke: kontrakt wymiarow + zgodnosc MAC z szacunkami z D6B_PLAN.md.
    configs = {
        "S1_half":      dict(channels=(16, 32), downsample="maxpool", depthwise=False),
        "S2_quarter":   dict(channels=(8, 16),  downsample="maxpool", depthwise=False),
        "S3_stride":    dict(channels=(16, 32), downsample="stride",  depthwise=False),
        "S4_depthwise": dict(channels=(16, 32), downsample="maxpool", depthwise=True),
    }
    for name, cfg in configs.items():
        torch.manual_seed(0)
        m = MarsV2SlimSystem(backbone_hidden=128, emb_dim=32, pod_hidden=24, **cfg)
        x = torch.randn(8, N_IN)
        feats = m.features(x)
        assert feats.shape == (8, 128), (name, feats.shape)
        assert m.forward(x).shape == (8, 10), name
        mac = m.mac_per_sample_top1()
        print(f"{name:14s} MAC top-1 = {mac['total_top1']:>9,} "
              f"(backbone {mac['backbone']:,}) | x MLP(215.6k) = "
              f"{mac['total_top1']/215600:.2f}")
    print("Smoke test OK.")
