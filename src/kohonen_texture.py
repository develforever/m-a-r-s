"""
kohonen_texture.py — Self-Organizing Map (Kohonen) + Texture Memory.

Rozwiązanie problemu z Etapu 4: bilinear filtering na RZADKIEJ teksturze
interpoluje z zerami → bzdura semantyczna.

FIX: Przed zapisem na teksturę przepuść embeddingi przez sieć Kohonena,
która WYMUSZA topologię — pojęcia podobne lądują obok siebie na siatce.
Wtedy bilinear filtering interpoluje między POKREWNYMI pojęciami,
nie między pojęciem a zerem.

Trzy podejścia implementowane i porównywane:
  A) Naiwna tekstura (Etap 4 oryginalny) — kontrola negatywna
  B) Kohonen SOM + tekstura — wymuszanie topologii
  C) Gęste embeddingi + dot product — alternatywa bez tekstur
"""

import numpy as np
import time


class KohonenSOM:
    """
    Self-Organizing Map — mapuje wektory wejściowe na siatkę 2D
    tak, że podobne wektory lądują w sąsiednich komórkach.

    Po treningu: każdy piksel (x, y) tekstury zawiera embedding,
    a sąsiednie piksele mają PODOBNE embeddingi → bilinear ma sens.
    """

    def __init__(self, grid_size, input_dim, seed=42):
        self.grid_size = grid_size
        self.input_dim = input_dim
        rng = np.random.default_rng(seed)
        # Wagi: każdy neuron w siatce ma wektor wag o wymiarze input_dim
        self.weights = rng.normal(0, 0.5, (grid_size, grid_size, input_dim)).astype(np.float32)
        # Precompute grid coordinates
        self._coords = np.array([(i, j) for i in range(grid_size)
                                 for j in range(grid_size)])

    def _find_bmu(self, x):
        """Find Best Matching Unit — neuron najbliższy do x."""
        diff = self.weights - x.reshape(1, 1, -1)
        dist = np.sum(diff ** 2, axis=2)
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx

    def _neighborhood(self, bmu, sigma):
        """Gaussian neighborhood function around BMU."""
        coords = self._coords.reshape(self.grid_size, self.grid_size, 2)
        bmu_coord = np.array([bmu[0], bmu[1]])
        dist_sq = np.sum((coords - bmu_coord) ** 2, axis=2)
        return np.exp(-dist_sq / (2 * sigma ** 2))

    def train(self, data, epochs=500, lr_start=0.5, lr_end=0.01,
              sigma_start=None, sigma_end=1.0):
        """
        Trening SOM: iteracyjne dopasowywanie wag do danych.
        sigma: promień sąsiedztwa (maleje z czasem).
        """
        if sigma_start is None:
            sigma_start = self.grid_size / 2

        n_samples = len(data)
        for epoch in range(epochs):
            t = epoch / max(epochs - 1, 1)
            lr = lr_start * (lr_end / lr_start) ** t
            sigma = sigma_start * (sigma_end / sigma_start) ** t

            # Random sample
            idx = epoch % n_samples
            x = data[idx]

            # Find BMU
            bmu = self._find_bmu(x)

            # Update weights
            h = self._neighborhood(bmu, sigma)[:, :, np.newaxis]
            self.weights += lr * h * (x - self.weights)

    def map_to_grid(self, x):
        """Zwraca pozycję (row, col) na siatce dla wektora x."""
        return self._find_bmu(x)

    def get_texture(self):
        """Zwraca macierz wag jako 'teksturę' — gęsto wypełnioną."""
        return self.weights.copy()

    def quantization_error(self, data):
        """Średni błąd kwantyzacji — miara jakości mapowania."""
        errors = []
        for x in data:
            bmu = self._find_bmu(x)
            errors.append(np.sum((x - self.weights[bmu]) ** 2))
        return float(np.mean(errors))


class KohonenTexture:
    """
    Tekstura 2D wypełniona przez SOM — każdy piksel ma wartość,
    sąsiedzi są semantycznie powiązani.
    """

    def __init__(self, som):
        self.som = som
        self.size = som.grid_size
        self.data = som.get_texture()  # [size, size, input_dim]

    def bilinear_lookup(self, x, y):
        """
        Bilinear interpolation w przestrzeni embeddingów.
        x, y: współrzędne [0, 1].
        Zwraca interpolowany wektor.
        """
        fx = x * (self.size - 1)
        fy = y * (self.size - 1)
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        x1 = min(x0 + 1, self.size - 1)
        y1 = min(y0 + 1, self.size - 1)
        wx = fx - x0
        wy = fy - y0
        c00 = self.data[y0, x0]
        c10 = self.data[y0, x1]
        c01 = self.data[y1, x0]
        c11 = self.data[y1, x1]
        result = (c00 * (1 - wx) * (1 - wy) +
                  c10 * wx * (1 - wy) +
                  c01 * (1 - wx) * wy +
                  c11 * wx * wy)
        return result

    def lookup_nearest(self, query_vec):
        """Znajdź najbliższy punkt na teksturze do query."""
        bmu = self.som.map_to_grid(query_vec)
        return self.data[bmu], bmu

    def interpolate_between(self, vec_a, vec_b, alpha):
        """
        Interpolacja na teksturze między dwoma koncepcjami:
        1. Znajdź pozycje A i B na siatce
        2. Bilinear interpolation wzdłuż ścieżki na siatce
        """
        bmu_a = self.som.map_to_grid(vec_a)
        bmu_b = self.som.map_to_grid(vec_b)
        # Interpolacja pozycji na siatce
        row = bmu_a[0] + alpha * (bmu_b[0] - bmu_a[0])
        col = bmu_a[1] + alpha * (bmu_b[1] - bmu_a[1])
        # Normalizuj do [0,1]
        x = col / (self.size - 1)
        y = row / (self.size - 1)
        return self.bilinear_lookup(x, y)


class DenseEmbeddingMemory:
    """
    Opcja C: Gęste embeddingi + dot product (bez tekstur).
    Każde pojęcie = wektor. Podobieństwo = cosine similarity.
    """

    def __init__(self, dim):
        self.dim = dim
        self.embeddings = []  # lista wektorów
        self.labels = []

    def store(self, label, vec):
        self.labels.append(label)
        self.embeddings.append(vec / (np.linalg.norm(vec) + 1e-8))

    def query(self, vec, top_k=3):
        """Znajdź top-k najbardziej podobnych pojęć (cosine similarity)."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        if not self.embeddings:
            return []
        emb_matrix = np.array(self.embeddings)
        similarities = emb_matrix @ vec_norm
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return [(self.labels[i], float(similarities[i])) for i in top_idx]

    def interpolate(self, vec_a, vec_b, alpha):
        """Lerp w przestrzeni embeddingów (ground truth)."""
        result = (1 - alpha) * vec_a + alpha * vec_b
        return result
