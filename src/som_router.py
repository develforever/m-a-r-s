"""
som_router.py — SOM-Router: Kohonen Self-Organizing Map jako Engine Core.

INTEGRACJA Etap 3 (Router) + Etap 4B (Kohonen SOM):
  Zamiast sieci neuronowej (2× matmul + softmax + argmax), router to:
  1. SOM wytrenowany na przestrzeni wejściowej
  2. Każda komórka siatki ma przypisany capsule_id (majority voting)
  3. Inferencja = find_BMU (najbliższy neuron) → capsule_id

  Na GPU: BMU lookup = texture fetch (TMU) = O(1) sprzętowo.
  Na CPU: BMU = 1× distance matrix (N*M distances) vs neural = 2× matmul.

SLEEP v2 — zamiast blur:
  - Selective weight decay: świeżość (G) spada co cykl, stare neurony słabną
  - Hebbian wzmocnienie: co-aktywowane neurony zbliżają wagi
  - Pruning: neurony poniżej progu aktywacji → ponowne użycie (re-assign)

SOFT ROUTING via bilinear:
  - Zamiast hard BMU (1 kapsuła) → interpolacja wag sąsiadów
  - Sąsiednie komórki mogą mieć RÓŻNE capsule_id → confidence score
  - Jeśli >1 capsule_id w sąsiedztwie → top-2 routing z wagami
"""

import numpy as np
from metrics import MACCounter


class SOMRouter:
    """
    Router oparty na Self-Organizing Map.
    
    Trening:
      1. Trenuj SOM na X (przestrzeń wejściowa)
      2. Przypisz każdej komórce capsule_id (majority voting z etykiet regionów)
    
    Inferencja:
      - Hard: find_BMU → capsule_id (1 lookup)
      - Soft: bilinear interpolation → confidence per capsule
    
    MAC counting:
      - BMU search: porównanie z grid_size² wektorami (każdy to dot product/distance)
      - Na GPU: 0 MAC (texture fetch via TMU)
      - Na CPU: grid_size² × input_dim MAC
    """

    def __init__(self, n_in, n_pods, grid_size=8, seed=42):
        self.n_in = n_in
        self.n_pods = n_pods
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        
        # Wagi SOM: [grid_size, grid_size, n_in]
        self.weights = self.rng.normal(0, 0.5, 
            (grid_size, grid_size, n_in)).astype(np.float32)
        
        # Mapa: komórka → capsule_id, -1 = nieprzypisana
        self.cell_labels = np.full((grid_size, grid_size), -1, dtype=np.int32)
        
        # Metadata per cell (RGBA-style)
        self.freshness = np.ones((grid_size, grid_size), dtype=np.float32)  # G
        self.activation_count = np.zeros((grid_size, grid_size), dtype=np.int32)
        
    def _find_bmu(self, x):
        """Find Best Matching Unit — komórka najbliższa do x."""
        diff = self.weights - x.reshape(1, 1, -1)
        dist = np.sum(diff ** 2, axis=2)
        return np.unravel_index(np.argmin(dist), dist.shape)
    
    def _find_bmu_batch(self, X):
        """BMU dla batcha — zwraca listę (row, col) per sample."""
        bmus = []
        for x in X:
            bmus.append(self._find_bmu(x))
        return bmus
    
    def _neighborhood(self, bmu, sigma):
        """Gaussian neighborhood."""
        rows = np.arange(self.grid_size)[:, None]
        cols = np.arange(self.grid_size)[None, :]
        dist_sq = (rows - bmu[0])**2 + (cols - bmu[1])**2
        return np.exp(-dist_sq / (2 * sigma**2))
    
    def train_som(self, X, epochs=1000, lr_start=0.5, lr_end=0.01,
                  sigma_start=None, sigma_end=0.5):
        """Trening SOM na danych wejściowych."""
        if sigma_start is None:
            sigma_start = self.grid_size / 2
        
        n = len(X)
        for epoch in range(epochs):
            t = epoch / max(epochs - 1, 1)
            lr = lr_start * (lr_end / lr_start) ** t
            sigma = sigma_start * (sigma_end / sigma_start) ** t
            
            # Losowy sample
            idx = self.rng.integers(n)
            x = X[idx]
            
            bmu = self._find_bmu(x)
            h = self._neighborhood(bmu, sigma)[:, :, np.newaxis]
            self.weights += lr * h * (x - self.weights)
    
    def assign_labels(self, X, regions):
        """
        Przypisz capsule_id do każdej komórki SOM (majority voting).
        Po treningu SOM, przepuść dane i policz która kapsuła dominuje w każdej komórce.
        """
        votes = np.zeros((self.grid_size, self.grid_size, self.n_pods), dtype=np.int32)
        
        for x, region in zip(X, regions):
            bmu = self._find_bmu(x)
            votes[bmu[0], bmu[1], region] += 1
        
        # Majority voting
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if votes[i, j].sum() > 0:
                    self.cell_labels[i, j] = np.argmax(votes[i, j])
                else:
                    # Komórka bez danych → przypisz najbliższy sąsiedni label
                    self.cell_labels[i, j] = -1
        
        # Fill unassigned cells (propagate from nearest assigned)
        self._fill_unassigned()
    
    def _fill_unassigned(self):
        """Wypełnij komórki bez labela — propagacja z najbliższego przypisanego."""
        unassigned = np.argwhere(self.cell_labels == -1)
        assigned = np.argwhere(self.cell_labels >= 0)
        
        if len(assigned) == 0 or len(unassigned) == 0:
            return
        
        for ua in unassigned:
            dists = np.sum((assigned - ua) ** 2, axis=1)
            nearest = assigned[np.argmin(dists)]
            self.cell_labels[ua[0], ua[1]] = self.cell_labels[nearest[0], nearest[1]]
    
    def train(self, X, regions, som_epochs=2000, lr=0.5):
        """Pełny trening: SOM + label assignment."""
        self.train_som(X, epochs=som_epochs, lr_start=lr)
        self.assign_labels(X, regions)
    
    def predict_pod(self, X, mac=None):
        """
        Hard routing: BMU → capsule_id.
        
        MAC: na CPU = grid_size² × n_in per sample (distance computation)
             na GPU = 0 (TMU fetch)
        """
        predictions = np.zeros(len(X), dtype=np.int32)
        
        for i, x in enumerate(X):
            if mac is not None:
                # Koszt BMU search: porównanie z grid_size² wektorami
                # Każde porównanie = n_in subtrakcji + n_in mnożeń + n_in dodawań
                mac.count += self.grid_size * self.grid_size * self.n_in * 2
            
            bmu = self._find_bmu(x)
            predictions[i] = self.cell_labels[bmu[0], bmu[1]]
            self.activation_count[bmu[0], bmu[1]] += 1
        
        return predictions
    
    def predict_pod_soft(self, X, mac=None):
        """
        Soft routing: bilinear interpolation → confidence per capsule.
        Zwraca (capsule_id, confidence, second_capsule_id).
        
        Jeśli sąsiednie komórki mają różne capsule_id → niski confidence.
        """
        results = []
        
        for x in X:
            if mac is not None:
                mac.count += self.grid_size * self.grid_size * self.n_in * 2
            
            # Znajdź pozycję (nie tylko BMU, ale dokładną pozycję sub-pikselową)
            diff = self.weights - x.reshape(1, 1, -1)
            dist = np.sum(diff ** 2, axis=2)
            bmu = np.unravel_index(np.argmin(dist), dist.shape)
            
            # Sąsiedztwo 3x3 → policz capsule_id
            votes = {}
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = bmu[0] + dy, bmu[1] + dx
                    if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                        # Waga = odwrotność odległości wagi od x
                        w = 1.0 / (dist[ny, nx] + 1e-8)
                        label = int(self.cell_labels[ny, nx])
                        votes[label] = votes.get(label, 0) + w
            
            # Normalizuj
            total = sum(votes.values())
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            
            best_pod = sorted_votes[0][0]
            confidence = sorted_votes[0][1] / total
            second_pod = sorted_votes[1][0] if len(sorted_votes) > 1 else best_pod
            
            results.append((best_pod, confidence, second_pod))
        
        return results
    
    # ─── Sleep v2: selective decay + Hebbian ─────────────────────────────
    
    def sleep_decay(self, decay_rate=0.05):
        """
        Selective weight decay: świeżość spada.
        Komórki, które dawno nie były aktywowane → słabną.
        """
        self.freshness = np.maximum(0.1, self.freshness - decay_rate)
    
    def sleep_hebbian(self, X, regions, strength=0.01):
        """
        Hebbian wzmocnienie: jeśli komórka jest często aktywowana
        przez dane z tego samego regionu, zbliż ją do centroidu regionu.
        (Anti-blur: wzmacnianie KONKRETNYCH skojarzeń, nie rozmywanie)
        """
        # Policz centroid per region
        centroids = {}
        for x, r in zip(X, regions):
            if r not in centroids:
                centroids[r] = []
            centroids[r].append(x)
        centroids = {r: np.mean(vecs, axis=0) for r, vecs in centroids.items()}
        
        # Wzmocnij komórki w kierunku ich centroidu
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                label = self.cell_labels[i, j]
                if label >= 0 and label in centroids:
                    # Im wyższa freshness i activation_count → silniejsze wzmocnienie
                    factor = strength * self.freshness[i, j]
                    self.weights[i, j] += factor * (centroids[label] - self.weights[i, j])
    
    def sleep_prune(self, min_activations=2):
        """
        Pruning: komórki z minimalną aktywnością → reset (re-assign).
        Uwalnia "martwe" neurony dla nowych pojęć.
        """
        dead = self.activation_count < min_activations
        n_pruned = np.sum(dead)
        if n_pruned > 0:
            # Reset martwych neuronów do losowych wartości
            dead_coords = np.argwhere(dead)
            for coord in dead_coords:
                self.weights[coord[0], coord[1]] = self.rng.normal(0, 0.3, self.n_in)
                self.cell_labels[coord[0], coord[1]] = -1
            self._fill_unassigned()
        return int(n_pruned)
    
    def sleep_cycle(self, X, regions):
        """Pełny cykl snu: decay → hebbian → prune → reset counters."""
        self.sleep_decay()
        self.sleep_hebbian(X, regions)
        pruned = self.sleep_prune()
        self.activation_count[:] = 0  # reset na nowy cykl
        return pruned
    
    def topology_quality(self):
        """Miara jakości topologii: czy sąsiednie komórki mają ten sam label?"""
        same_neighbor = 0
        total = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for dy, dx in [(0, 1), (1, 0)]:
                    ni, nj = i + dy, j + dx
                    if ni < self.grid_size and nj < self.grid_size:
                        if self.cell_labels[i, j] == self.cell_labels[ni, nj]:
                            same_neighbor += 1
                        total += 1
        return same_neighbor / total if total > 0 else 0
