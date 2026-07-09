"""
cortex_texture.py — Cortex Core: pamięć jako tekstura 2D (RGBA).

Implementacja hipotezy "neuro-semantyczna mapa na teksturze":
  - Wiedza przechowywana w teksturze 2D (macierz float RGBA)
  - R = waga skojarzenia (siła połączenia)
  - G = świeżość (recency)
  - B = priorytet (ważność)
  - A = plastyczność (jak łatwo się zmienia)

Operacje "snu" realizowane jako filtry na teksturze:
  - Bilinear interpolation (lookup wiedzy z interpolacją)
  - Gaussian blur (rozlewanie kontekstu / konsolidacja)
  - Erosion (zapominanie — spadek świeżości)
  - Mipmapping (generalizacja — kompresja do niższych rozdzielczości)

KLUCZOWE PYTANIE BADAWCZE:
  Czy bilinear filtering na embeddingach daje SEMANTYCZNIE sensowną
  interpolację, czy tylko matematyczny szum?

Testujemy to empirycznie: umieszczamy znane embeddingi na teksturze,
interpolujemy między nimi i mierzymy, czy wynik jest bliski "prawidłowej"
interpolacji w przestrzeni embeddingów.
"""

import numpy as np
import time


class NeuroTexture:
    """
    Tekstura 2D przechowująca wiedzę w formacie RGBA.

    Mapowanie wiedzy na teksturę:
      - Każdy piksel (x, y) = jeden "slot" wiedzy
      - Kanały RGBA = właściwości tego slotu

    Parametry:
      size: rozdzielczość tekstury (size x size)
    """

    def __init__(self, size=64):
        self.size = size
        # Tekstura RGBA: [size, size, 4]
        # R=waga, G=świeżość, B=priorytet, A=plastyczność
        self.data = np.zeros((size, size, 4), dtype=np.float32)
        self.data[:, :, 1] = 1.0  # świeżość startowa = 1
        self.data[:, :, 3] = 1.0  # plastyczność startowa = 1

    def write(self, x, y, weight, priority=0.5):
        """Zapisz wiedzę w punkcie (x, y) — współrzędne [0, 1]."""
        ix = int(np.clip(x * (self.size - 1), 0, self.size - 1))
        iy = int(np.clip(y * (self.size - 1), 0, self.size - 1))
        self.data[iy, ix, 0] = weight      # R: siła
        self.data[iy, ix, 1] = 1.0         # G: świeżość (właśnie zapisane)
        self.data[iy, ix, 2] = priority     # B: priorytet
        # A (plastyczność) nie zmienia się przy zapisie

    def write_embedding(self, x, y, embedding, priority=0.5):
        """
        Zapisz embedding w otoczeniu punktu (x, y).
        Embedding kodowany jest wzdłuż kanałów i sąsiednich pikseli.
        """
        ix = int(np.clip(x * (self.size - 1), 0, self.size - 1))
        iy = int(np.clip(y * (self.size - 1), 0, self.size - 1))
        # Zapisujemy wektor w kanale R sąsiednich pikseli
        dim = len(embedding)
        for d in range(min(dim, self.size - ix)):
            self.data[iy, ix + d, 0] = embedding[d]
            self.data[iy, ix + d, 1] = 1.0
            self.data[iy, ix + d, 2] = priority

    def bilinear_lookup(self, x, y):
        """
        Bilinear interpolation — sprzętowa operacja na GPU (tex2D).
        Tu emulowana w NumPy. Zwraca interpolowane RGBA z 4 sąsiadów.
        """
        fx = x * (self.size - 1)
        fy = y * (self.size - 1)
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        x1 = min(x0 + 1, self.size - 1)
        y1 = min(y0 + 1, self.size - 1)
        # Wagi interpolacji
        wx = fx - x0
        wy = fy - y0
        # 4 narożniki
        c00 = self.data[y0, x0]
        c10 = self.data[y0, x1]
        c01 = self.data[y1, x0]
        c11 = self.data[y1, x1]
        # Bilinear
        result = (c00 * (1 - wx) * (1 - wy) +
                  c10 * wx * (1 - wy) +
                  c01 * (1 - wx) * wy +
                  c11 * wx * wy)
        return result

    def bilinear_lookup_row(self, x, y, length):
        """Odczyt ciągu pikseli z bilinear interpolation (emulacja tex2D line)."""
        fx = x * (self.size - 1)
        fy = y * (self.size - 1)
        result = np.zeros(length, dtype=np.float32)
        for d in range(length):
            px = (fx + d) / (self.size - 1)
            if px > 1.0:
                break
            val = self.bilinear_lookup(px, y)
            result[d] = val[0]  # kanał R (waga/wartość)
        return result

    # ─── Operacje "snu" ──────────────────────────────────────────────────

    def sleep_erosion(self, decay=0.05):
        """
        Erozja świeżości: kanał G spada o decay.
        Symuluje zapominanie — stare informacje blakną.
        """
        self.data[:, :, 1] = np.maximum(0, self.data[:, :, 1] - decay)

    def sleep_consolidation(self, sigma=1.0):
        """
        Gaussian blur na kanale R (waga skojarzenia).
        Symuluje konsolidację — rozlewanie wiedzy na sąsiadów.
        Odpowiednik "rozmycia skojarzeń" z dokumentów.
        """
        from scipy.ndimage import gaussian_filter
        self.data[:, :, 0] = gaussian_filter(self.data[:, :, 0], sigma=sigma)

    def sleep_consolidation_numpy(self, kernel_size=3):
        """
        Konsolidacja bez scipy — prosty box blur.
        """
        pad = kernel_size // 2
        padded = np.pad(self.data[:, :, 0], pad, mode='edge')
        result = np.zeros_like(self.data[:, :, 0])
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                result += padded[dy:dy+self.size, dx:dx+self.size]
        self.data[:, :, 0] = result / (kernel_size * kernel_size)

    def sleep_plasticity_decay(self, used_mask=None, decay=0.02, boost=0.1):
        """
        Metaplastyczność: często używane sloty "kostnieją" (A spada),
        nieużywane zachowują plastyczność.
        used_mask: macierz [size, size] bool — które sloty były używane.
        """
        if used_mask is not None:
            self.data[:, :, 3] = np.where(
                used_mask,
                np.maximum(0.1, self.data[:, :, 3] - decay),   # kostnienie
                np.minimum(1.0, self.data[:, :, 3] + boost)     # odzyskiwanie
            )

    def mipmap(self, level=1):
        """
        Mipmapping — generalizacja przez downsampling.
        Zwraca "rozmytą" wersję tekstury o niższej rozdzielczości.
        Odpowiednik: widok z lotu ptaka na wiedzę.
        """
        data = self.data
        for _ in range(level):
            h, w = data.shape[:2]
            if h < 2 or w < 2:
                break
            # Average pooling 2x2
            data = (data[0::2, 0::2] + data[1::2, 0::2] +
                    data[0::2, 1::2] + data[1::2, 1::2]) / 4.0
        return data

    def occupancy(self):
        """Ile slotów zawiera wiedzę (R > próg)."""
        active = np.sum(np.abs(self.data[:, :, 0]) > 0.01)
        return active / (self.size * self.size)

    def stats(self):
        """Statystyki tekstury."""
        return {
            "size": self.size,
            "occupancy_pct": round(self.occupancy() * 100, 2),
            "mean_weight": round(float(np.mean(np.abs(self.data[:, :, 0]))), 6),
            "mean_freshness": round(float(np.mean(self.data[:, :, 1])), 4),
            "mean_plasticity": round(float(np.mean(self.data[:, :, 3])), 4),
        }


# ─── Benchmark: texture lookup vs matrix multiplication ─────────────────────

def benchmark_lookup_vs_matmul(size=64, n_lookups=10000):
    """
    Porównanie: bilinear texture lookup vs standardowe mnożenie macierzy.
    Mierzymy czas (CPU) — na GPU tekstury byłyby sprzętowo przyspieszane.
    """
    tex = NeuroTexture(size)
    # Wypełnij losowymi danymi
    rng = np.random.default_rng(42)
    tex.data[:, :, 0] = rng.normal(0, 0.3, (size, size)).astype(np.float32)

    # Losowe współrzędne lookup
    coords = rng.random((n_lookups, 2)).astype(np.float32)

    # Benchmark: bilinear lookups
    t0 = time.perf_counter()
    results_tex = np.array([tex.bilinear_lookup(x, y) for x, y in coords])
    t_tex = time.perf_counter() - t0

    # Benchmark: equivalent matrix operation (lookup by index + interpolate)
    # Symulacja: mnożenie macierzy [n_lookups, size] x [size, 4]
    query_matrix = rng.random((n_lookups, size)).astype(np.float32)
    value_matrix = rng.random((size, 4)).astype(np.float32)

    t0 = time.perf_counter()
    results_mat = query_matrix @ value_matrix
    t_tex_mat = time.perf_counter() - t0

    return {
        "n_lookups": n_lookups,
        "texture_size": size,
        "time_bilinear_s": round(t_tex, 6),
        "time_matmul_s": round(t_tex_mat, 6),
        "speedup_matmul_over_bilinear": round(t_tex / max(t_tex_mat, 1e-9), 2),
        "note": "Na GPU bilinear jest sprzętowy (TMU) i ~darmowy. "
                "Tu emulacja CPU — porównanie orientacyjne."
    }
