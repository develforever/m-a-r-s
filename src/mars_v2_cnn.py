"""
mars_v2_cnn.py -- D6: CNN backbone w architekturze M.A.R.S. v2.

MOTYWACJA (z DROGA_D_NOTATKI.md + B_SERIES_WYNIKI.md):
  Droga 2 (D4 consultation, D5 distillation) potwierdzila: na shared backbone
  MLP router osiagnal SUFIT reprezentacji. Kazda modyfikacja routingu = 0 lub
  szkodzi. Jedyna udowodniona dzwignia to LEPSZE CECHY -- seria B pokazala, ze
  CNN encoder bije MLP o +3.5pp na Fashion (routing 89% -> 93%).

HIPOTEZA D6:
  Jesli wymienimy MLP backbone (Linear 784->H) na CNN backbone, wspolna
  reprezentacja stanie sie bogatsza -> router (na tych cechach) routuje lepiej
  -> system_acc rosnie. Pody dzielą ten sam, lepszy backbone.

KLUCZOWA DECYZJA PROJEKTOWA:
  Zmieniamy WYLACZNIE backbone. Cala reszta architektury v2 (routing head
  prototypowy, stacked pody, train_phased / train_end_to_end, evaluate,
  forward_adaptive) pozostaje identyczna i jest REUZYWANA z mars_v2.py.
  Dzieki temu porownanie D6 (CNN) vs D1 (MLP) jest czyste: rozni je tylko
  ekstraktor cech. backbone przyjmuje [B, 784] i zwraca [B, backbone_hidden],
  wiec spelnia ten sam kontrakt co nn.Sequential(Linear, ReLU) z v2.
"""
import torch
import torch.nn as nn

from mars_v2 import MarsV2System, N_IN, N_CLASSES


class CNNBackbone(nn.Module):
    """
    Wspolny ekstraktor cech: 2 bloki konwolucyjne (jak B1b CNN(32,64)) +
    projekcja FC do wspolnego wektora `backbone_hidden`.

    Kontrakt zgodny z MLP backbone v2:
        wejscie  [B, 784]  (plaski obraz, jak reszta pipeline'u)
        wyjscie  [B, backbone_hidden]

    Reshape 784 -> 1x28x28 odbywa sie WEWNATRZ (pipeline trzyma dane plasko).
    """
    def __init__(self, backbone_hidden=128, channels=(32, 64)):
        super().__init__()
        c1, c2 = channels
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),   # 28x28
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 14x14
            nn.Conv2d(c1, c2, 3, padding=1),  # 14x14
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 7x7
        )
        self.proj = nn.Sequential(
            nn.Linear(7 * 7 * c2, backbone_hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        f = self.conv(x)
        f = f.view(f.shape[0], -1)
        return self.proj(f)


class MarsV2CNNSystem(MarsV2System):
    """
    v2 z CNN backbone. Dziedziczy CALA logike po MarsV2System
    (route_logits, route, pod_forward, forward, forward_adaptive, evaluate-able),
    podmienia jedynie `self.backbone` z MLP na CNNBackbone i koryguje wzor MAC.
    """
    def __init__(self, n_in=N_IN, backbone_hidden=128, n_pods=N_CLASSES,
                 emb_dim=32, pod_hidden=24, n_out=N_CLASSES, channels=(32, 64)):
        super().__init__(n_in=n_in, backbone_hidden=backbone_hidden,
                         n_pods=n_pods, emb_dim=emb_dim, pod_hidden=pod_hidden,
                         n_out=n_out)
        # Podmiana MLP backbone (Linear+ReLU) na CNN. Reszta parametrow
        # (routing_head, protos, pod_W/b) zdefiniowana w klasie bazowej i
        # pasuje wymiarowo, bo operuje na `backbone_hidden`.
        self.channels = channels
        self.backbone = CNNBackbone(backbone_hidden=backbone_hidden,
                                    channels=channels)

    def mac_per_sample_top1(self):
        """
        MAC per probka. backbone = conv1 + conv2 + projekcja FC (staly koszt).
        Wzory conv jak w B1b: out_ch * in_ch * k*k * H_out * W_out.
        """
        c1, c2 = self.channels
        conv1 = 1 * c1 * 3 * 3 * 28 * 28
        conv2 = c1 * c2 * 3 * 3 * 14 * 14
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
    torch.manual_seed(0)
    m = MarsV2CNNSystem(backbone_hidden=128, emb_dim=32, pod_hidden=24,
                        channels=(32, 64))
    x = torch.randn(64, N_IN)
    feats = m.features(x)
    assert feats.shape == (64, 128), feats.shape
    assert m.route(feats).shape == (64,)
    assert m.pod_forward(feats, m.route(feats)).shape == (64, 10)
    assert m.forward(x).shape == (64, 10)
    preds, stats = m.forward_adaptive(x, 0.95, 0.5)
    assert preds.shape == (64,)
    mac = m.mac_per_sample_top1()
    print("Smoke test OK.")
    print(f"  Params: {m.n_params():,}")
    print(f"  MAC top-1: {mac['total_top1']:,} "
          f"(backbone={mac['backbone']:,}, routing={mac['routing']:,}, pod={mac['pod']:,})")
