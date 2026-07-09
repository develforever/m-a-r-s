"""
routers_v3.py — głębszy router dla M.A.R.S. (Seria B, krok B1).

Problem: ProtoRouter z routers_v2.py ma jedną warstwę ukrytą w encoderze.
Na MNIST to wystarczyło (luka 1.1pp do ORACLE), ale na Fashion-MNIST
encoder jest za płytki (luka 9.3pp). Trudniejsze rozkłady (T-shirt vs Shirt
vs Coat) wymagają więcej nieliniowości.

DeepProtoRouter — ulepszenia vs ProtoRouter:
  1. Głębszy encoder (2-3 warstwy z BatchNorm)
  2. Cosine similarity zamiast L2 distance (opcjonalnie)
  3. Learnable temperature (ostrość decyzji routingowej)
  4. Dropout w encoderze (regularyzacja)
  5. K-means inicjalizacja prototypów (opcja, patrz init_protos_kmeans)

Interfejs identyczny z ProtoRouter — drop-in replacement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepProtoRouter(nn.Module):
    """
    Routing prototypowy z głębszym encoderem.
    Encoder: 2–3 warstwy z BatchNorm + Dropout.
    Similarity: L2 (domyślnie) lub cosine.
    Temperature: learnable skalowanie logitów.
    """

    def __init__(self, n_in, n_pods, enc_hidden=256, enc_hidden2=128,
                 emb=32, dropout=0.1, use_cosine=False, init_temp=1.0):
        super().__init__()
        self.n_in = n_in
        self.n_pods = n_pods
        self.enc_hidden = enc_hidden
        self.enc_hidden2 = enc_hidden2
        self.emb = emb
        self.use_cosine = use_cosine

        # 2-layer encoder z BatchNorm
        layers = [
            nn.Linear(n_in, enc_hidden),
            nn.BatchNorm1d(enc_hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if enc_hidden2 > 0:
            layers.extend([
                nn.Linear(enc_hidden, enc_hidden2),
                nn.BatchNorm1d(enc_hidden2),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(enc_hidden2, emb))
        else:
            layers.append(nn.Linear(enc_hidden, emb))

        self.enc = nn.Sequential(*layers)
        self.protos = nn.Parameter(torch.randn(n_pods, emb))

        # Learnable temperature
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())

    @property
    def temperature(self):
        return self.log_temp.exp()

    def encode(self, x):
        """Zwraca embedding [B, emb]."""
        return self.enc(x)

    def forward(self, x):
        """Zwraca logity nad podami [B, n_pods]."""
        e = self.encode(x)
        if self.use_cosine:
            e_norm = F.normalize(e, dim=1)
            p_norm = F.normalize(self.protos, dim=1)
            sim = e_norm @ p_norm.T  # [B, n_pods], zakres [-1, 1]
            return sim / self.temperature
        else:
            d = torch.cdist(e, self.protos)
            return -d / self.temperature

    def route(self, x):
        """Zwraca capsule_ids [B]."""
        return self.forward(x).argmax(dim=1)

    def add_pod(self, init_from=None):
        """Rozszerzenie o nowy pod (continual learning)."""
        new = (init_from if init_from is not None
               else torch.randn(1, self.emb, device=self.protos.device))
        self.protos = nn.Parameter(torch.cat([self.protos.data, new], dim=0))
        self.n_pods += 1

    def init_protos_kmeans(self, X, max_samples=10000):
        """
        K-means inicjalizacja prototypów na embeddings z danych treningowych.
        Wywołaj PO kilku warmup epokach encodera (żeby embeddings miały sens).
        """
        self.eval()
        with torch.no_grad():
            idx = torch.randperm(len(X))[:max_samples]
            embs = self.encode(X[idx])

            # K-means++ init
            centroids = [embs[torch.randint(len(embs), (1,)).item()]]
            for _ in range(1, self.n_pods):
                dists = torch.cdist(embs, torch.stack(centroids))
                min_dists = dists.min(dim=1).values
                prob = min_dists / min_dists.sum()
                chosen = torch.multinomial(prob, 1).item()
                centroids.append(embs[chosen])
            centroids = torch.stack(centroids)

            # Kilka iteracji Lloyd
            for _ in range(20):
                assigns = torch.cdist(embs, centroids).argmin(dim=1)
                new_centroids = torch.zeros_like(centroids)
                for k in range(self.n_pods):
                    mask = assigns == k
                    if mask.any():
                        new_centroids[k] = embs[mask].mean(dim=0)
                    else:
                        new_centroids[k] = centroids[k]
                centroids = new_centroids

            self.protos.data.copy_(centroids)

    def mac_per_sample(self):
        """MAC per sample (encoder + similarity)."""
        # Layer 1
        mac = self.n_in * self.enc_hidden
        if self.enc_hidden2 > 0:
            # Layer 2
            mac += self.enc_hidden * self.enc_hidden2
            # Layer 3 (projection to emb)
            mac += self.enc_hidden2 * self.emb
        else:
            mac += self.enc_hidden * self.emb
        # Similarity computation
        if self.use_cosine:
            mac += self.emb * self.n_pods  # dot products
        else:
            mac += self.emb * self.n_pods  # L2 distance
        return mac
