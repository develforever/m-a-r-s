"""
routers_v2.py — następcy tekstury 2D dla routingu M.A.R.S. (Droga A, krok A1).

Problem: dotychczasowy router (SOMProjectionRouter) ściska wejście do 2D-UV,
bo tekstura jest płaska. 2D ścina routing do ~44% na MNIST — za mało na
prawdziwą specjalizację (gdzie system accuracy = router accuracy).

Dwa następcy, oba pozwalają na bottleneck ≥4D:

  MLPRouter — klasyczne gating (input → bottleneck → softmax nad podami).
    To w istocie router z Mixture-of-Experts. Prosty, sprawdzony, tani.

  ProtoRouter — routing prototypowy (jeden wektor-centroid na pod,
    routing = najbliższy prototyp w przestrzeni embeddingu). Bliski
    duchowi SOM z dokumentów M.A.R.S., interpretowalny, łatwo rozszerzalny
    o nowe klasy (dokładasz prototyp). Zachowuje ideę "mapy topologicznej"
    bez płaskiej tekstury.

Tekstura zostaje porzucona jako mechanizm routingu — audyt pokazał, że
jej lookup był darmowy, ale wąskim gardłem i tak był encoder, a 2D
ograniczało jakość. Wybieramy router empirycznie wg accuracy na MNIST.
"""
import torch
import torch.nn as nn


class MLPRouter(nn.Module):
    """Gating MLP: input → enc_hidden → bottleneck → softmax nad podami."""
    def __init__(self, n_in, n_pods, enc_hidden=16, bottleneck=4):
        super().__init__()
        self.n_in = n_in
        self.n_pods = n_pods
        self.enc_hidden = enc_hidden
        self.bottleneck = bottleneck
        self.enc = nn.Sequential(
            nn.Linear(n_in, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, bottleneck),
        )
        self.head = nn.Linear(bottleneck, n_pods)

    def forward(self, x):
        """Zwraca logity nad podami [B, n_pods]."""
        return self.head(torch.relu(self.enc(x)))

    def route(self, x):
        """Zwraca capsule_ids [B] — wybrany pod per próbka."""
        return self.forward(x).argmax(dim=1)

    def mac_per_sample(self):
        return (self.n_in * self.enc_hidden
                + self.enc_hidden * self.bottleneck
                + self.bottleneck * self.n_pods)


class ProtoRouter(nn.Module):
    """
    Routing prototypowy: encoder → embedding, routing = najbliższy prototyp.
    Każdy pod ma uczony wektor-centroid. Logit = -dystans do prototypu.
    Bliski SOM/topologii z dokumentów, interpretowalny.
    """
    def __init__(self, n_in, n_pods, enc_hidden=16, emb=8):
        super().__init__()
        self.n_in = n_in
        self.n_pods = n_pods
        self.enc_hidden = enc_hidden
        self.emb = emb
        self.enc = nn.Sequential(
            nn.Linear(n_in, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, emb),
        )
        self.protos = nn.Parameter(torch.randn(n_pods, emb))

    def forward(self, x):
        """Zwraca logity = -dystans do prototypów [B, n_pods]."""
        e = self.enc(x)
        d = torch.cdist(e, self.protos)
        return -d

    def route(self, x):
        return self.forward(x).argmax(dim=1)

    def add_pod(self, init_from=None):
        """
        Rozszerzenie o nowy pod = dodanie prototypu. Naturalne dla
        continual learning (nie nadpisuje starych prototypów).
        """
        new = (init_from if init_from is not None
               else torch.randn(1, self.emb, device=self.protos.device))
        self.protos = nn.Parameter(torch.cat([self.protos.data, new], dim=0))
        self.n_pods += 1

    def mac_per_sample(self):
        # encoder + dystans do prototypów
        return (self.n_in * self.enc_hidden
                + self.enc_hidden * self.emb
                + self.emb * self.n_pods)
