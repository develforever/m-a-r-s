"""
mars_torch.py — M.A.R.S. Faza 2: PyTorch implementation (GPU-ready).

SOM-Router z projekcją liniową + texture fetch (grid_sample) emulujący TMU.
Pipeline: Input[N] → Projection[N,2] → sigmoid → UV → grid_sample → capsule_id

Komponenty:
  - SOMProjectionRouter: lekka projekcja + grid_sample (emulacja TMU)
  - CapsulePod: mały MLP specjalista (1 per klasa)
  - MARSystem: pełny system modularny (router + pods + sleep)
  - BaselineMLP: monolityczny MLP do porównania

Uruchom:
    .venv\\Scripts\\python.exe src\\run_faza2_mnist.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class MACReport:
    """Raport kosztów MAC per inference."""
    router_mac: int = 0
    pod_mac: int = 0
    total_mac: int = 0
    method: str = ""


# ─── SOM Projection Router (TMU emulacja) ───────────────────────────────────

class SOMProjectionRouter(nn.Module):
    """
    Router oparty na projekcji + texture lookup (grid_sample).
    
    Architektura:
      1. Encoder: input[N_IN] → hidden[H] → UV[2] (mały, ale nieliniowy)
      2. Sigmoid → [0, 1] × [0, 1]  (normalizacja UV)
      3. grid_sample na label_map → capsule_id (emulacja TMU)
    
    MAC: N_IN × H_ENC + H_ENC × 2 (encoder) + 0 (TMU fetch)
    vs Neural Router: N_IN × HIDDEN + HIDDEN × N_PODS

    encoder_hidden=4 (domyślnie): zmierzone na MNIST — daje TĘ SAMĄ jakość
    routingu co 64, ale 94% mniej MAC w routerze. Oszczędność MAC całego
    systemu rośnie z 56.9% do 77.0% bez straty accuracy.
    """
    
    def __init__(self, n_in, n_pods, grid_size=32, encoder_hidden=4):
        super().__init__()
        self.n_in = n_in
        self.n_pods = n_pods
        self.grid_size = grid_size
        self.encoder_hidden = encoder_hidden
        
        # Encoder: N_IN → H → 2 (lekka projekcja z jedną nieliniowością)
        self.encoder = nn.Sequential(
            nn.Linear(n_in, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, 2),
        )
        
        # Label map: tekstura [1, 1, grid_size, grid_size] z capsule_id
        self.register_buffer(
            'label_map',
            torch.zeros(1, 1, grid_size, grid_size)
        )
        
        # Confidence map: [1, 1, grid_size, grid_size]
        self.register_buffer(
            'confidence_map',
            torch.ones(1, 1, grid_size, grid_size)
        )
    
    def encode_to_uv(self, x):
        """Encode input to UV coordinates [0, 1]²."""
        return torch.sigmoid(self.encoder(x))
    
    def forward(self, x):
        """
        Routing: encoder → UV → texture fetch → capsule_id.
        
        Returns: (capsule_ids, confidence)
        """
        # Krok 1: Encoder → UV
        uv = self.encode_to_uv(x)  # [B, 2]
        
        # Krok 2: Normalizacja do [-1, 1] dla grid_sample
        grid = uv * 2 - 1  # [B, 2] → [-1, 1]
        grid = grid.view(-1, 1, 1, 2)  # [B, 1, 1, 2] for grid_sample
        
        # Krok 3: TMU fetch — label_map z 'nearest' (ID podów, BEZ interpolacji),
        # confidence z 'bilinear' (ciągła pewność). To poprawne użycie tekstur:
        # interpolacja liniowa na ID podów dawałaby nieistniejące ID (szum).
        labels = F.grid_sample(
            self.label_map.expand(x.shape[0], -1, -1, -1),
            grid, mode='nearest', padding_mode='border',
            align_corners=True
        )  # [B, 1, 1, 1]
        
        confidence = F.grid_sample(
            self.confidence_map.expand(x.shape[0], -1, -1, -1),
            grid, mode='bilinear', padding_mode='border',
            align_corners=True
        )  # [B, 1, 1, 1]
        
        capsule_ids = labels.view(-1).long().clamp(0, self.n_pods - 1)
        conf = confidence.view(-1)
        
        return capsule_ids, conf
    
    def mac_per_sample(self):
        """Koszt MAC routera per sample."""
        # Encoder: n_in × encoder_hidden + encoder_hidden × 2
        return self.n_in * self.encoder_hidden + self.encoder_hidden * 2
    
    def train_projection(self, X, labels, epochs=100, lr=0.01):
        """
        Trenuj encoder end-to-end z cross-entropy przez UV grid.
        
        Metoda: encoder mapuje do UV → kwantyzacja na grid → classification loss.
        Używamy soft assignment zamiast hard grid (differentiable).
        """
        # Classifier head: UV[2] → logits[n_pods] (tylko do treningu)
        uv_classifier = nn.Linear(2, self.n_pods).to(X.device)
        
        all_params = list(self.encoder.parameters()) + list(uv_classifier.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        n_samples = len(X)
        batch_size = min(2048, n_samples)
        
        for epoch in range(epochs):
            # Mini-batch SGD
            perm = torch.randperm(n_samples, device=X.device)
            total_loss = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]
                x_batch = X[idx]
                y_batch = labels[idx]
                
                uv = self.encode_to_uv(x_batch)  # [B, 2]
                logits = uv_classifier(uv)  # [B, n_pods]
                loss = criterion(logits, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 20 == 0:
                # Quick accuracy check
                with torch.no_grad():
                    uv_all = self.encode_to_uv(X[:5000])
                    pred = uv_classifier(uv_all).argmax(dim=1)
                    acc = (pred == labels[:5000]).float().mean().item()
                print(f"    epoch {epoch+1}/{epochs}, loss={total_loss/n_batches:.4f}, uv_acc={acc*100:.1f}%")
    
    def build_label_map(self, X, labels):
        """
        Po treningu encodera, zbuduj label_map (teksturę):
        Każdy piksel [i, j] = dominujący capsule_id w tym regionie UV.
        """
        with torch.no_grad():
            # Process in batches to avoid OOM
            all_uv = []
            for start in range(0, len(X), 4096):
                end = min(start + 4096, len(X))
                uv_batch = self.encode_to_uv(X[start:end])
                all_uv.append(uv_batch)
            uv = torch.cat(all_uv, dim=0)  # [N, 2]
            
            # Zbuduj histogram per piksel
            votes = torch.zeros(self.grid_size, self.grid_size, self.n_pods,
                              device=X.device)
            
            # Kwantyzuj UV do grid
            grid_coords = (uv * (self.grid_size - 1)).long()
            grid_coords = grid_coords.clamp(0, self.grid_size - 1)
            
            for i in range(len(X)):
                gx, gy = grid_coords[i, 0].item(), grid_coords[i, 1].item()
                votes[gy, gx, labels[i].item()] += 1
            
            # Majority voting per piksel
            label_map = votes.argmax(dim=2).float()  # [G, G]
            
            # Confidence: max_votes / total_votes per piksel
            total = votes.sum(dim=2)
            max_votes = votes.max(dim=2).values
            conf_map = torch.where(total > 0, max_votes / total, 
                                  torch.ones_like(total) * 0.5)
            
            # Fill empty cells (propagate from nearest filled)
            empty = total == 0
            if empty.any():
                filled_coords = (~empty).nonzero(as_tuple=False).float()
                empty_coords = empty.nonzero(as_tuple=False).float()
                if len(filled_coords) > 0 and len(empty_coords) > 0:
                    # Batch nearest-neighbor to avoid OOM
                    for batch_start in range(0, len(empty_coords), 512):
                        batch_end = min(batch_start + 512, len(empty_coords))
                        batch_ec = empty_coords[batch_start:batch_end]
                        dists = torch.cdist(batch_ec, filled_coords)
                        nearest = dists.argmin(dim=1)
                        ec_indices = empty.nonzero(as_tuple=False)[batch_start:batch_end]
                        fc_indices = (~empty).nonzero(as_tuple=False)
                        for j in range(len(ec_indices)):
                            ec = ec_indices[j]
                            fc = fc_indices[nearest[j]]
                            label_map[ec[0], ec[1]] = label_map[fc[0], fc[1]]
                            conf_map[ec[0], ec[1]] = conf_map[fc[0], fc[1]] * 0.5
            
            self.label_map.copy_(label_map.view(1, 1, self.grid_size, self.grid_size))
            self.confidence_map.copy_(conf_map.view(1, 1, self.grid_size, self.grid_size))
            
            # Report coverage
            filled_pct = (total > 0).float().mean().item() * 100
            print(f"    Grid coverage: {filled_pct:.1f}% cells filled")


# ─── Capsule Pod (specjalista) ───────────────────────────────────────────────

class CapsulePod(nn.Module):
    """
    Mały MLP specjalista — jeden per klasa/region.
    Trenowany tylko na danych ze "swojego" regionu.
    """
    
    def __init__(self, n_in, n_hidden, n_out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out)
        )
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
    
    def forward(self, x):
        return self.net(x)
    
    def mac_per_sample(self):
        """MAC per sample for this pod."""
        return self.n_in * self.n_hidden + self.n_hidden * self.n_out


# ─── Pełny system M.A.R.S. ──────────────────────────────────────────────────

class MARSystem(nn.Module):
    """
    M.A.R.S. Modular Autonomous Refinement System.
    
    Architektura:
      Router (SOM Projection) → wybiera 1 z N kapsuł → inferencja
      Reszta kapsuł ŚPI (0 MAC).
    """
    
    def __init__(self, n_in, n_pods, pod_hidden=64, grid_size=32, encoder_hidden=4):
        super().__init__()
        self.n_in = n_in
        self.n_pods = n_pods
        self.encoder_hidden = encoder_hidden
        self.router = SOMProjectionRouter(n_in, n_pods, grid_size, encoder_hidden)
        self.pods = nn.ModuleList([
            CapsulePod(n_in, pod_hidden, n_out=n_pods)
            for _ in range(n_pods)
        ])
    
    def forward(self, x):
        """
        Routing: router wybiera kapsułę → aktywacja tylko 1 poda.
        Returns: (logits, capsule_ids, confidence)
        """
        capsule_ids, confidence = self.router(x)
        
        # Aktywuj TYLKO wybrany pod per sample
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, self.n_pods, device=x.device)
        
        for pod_id in range(self.n_pods):
            mask = capsule_ids == pod_id
            if mask.any():
                logits[mask] = self.pods[pod_id](x[mask])
        
        return logits, capsule_ids, confidence
    
    def forward_dense(self, x):
        """
        Baseline dense: aktywacja WSZYSTKICH podów, uśrednienie.
        (do porównania kosztu MAC)
        """
        all_logits = torch.stack([pod(x) for pod in self.pods], dim=0)
        return all_logits.mean(dim=0)
    
    def mac_report(self, batch_size=1):
        """Raport MAC per sample."""
        router_mac = self.router.mac_per_sample()
        pod_mac = self.pods[0].mac_per_sample()  # 1 pod active
        dense_mac = pod_mac * self.n_pods  # all pods active
        
        return {
            "routed": MACReport(
                router_mac=router_mac,
                pod_mac=pod_mac,
                total_mac=router_mac + pod_mac,
                method="M.A.R.S. (routed)"
            ),
            "dense": MACReport(
                router_mac=0,
                pod_mac=dense_mac,
                total_mac=dense_mac,
                method="Dense (all pods)"
            ),
            "savings_pct": round((1 - (router_mac + pod_mac) / dense_mac) * 100, 1)
        }
    
    def train_system(self, train_loader, device, epochs_proj=50, epochs_pods=10,
                     lr_proj=0.005, lr_pods=0.001):
        """
        Trening M.A.R.S.:
          1. Zbierz dane → trenuj projekcję (supervised kontrastywna)
          2. Zbuduj label_map
          3. Trenuj każdy pod na "swoich" danych
        """
        # Krok 1: Zbierz wszystkie dane (flatten images if needed)
        all_X, all_y = [], []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            if X_batch.dim() > 2:
                X_batch = X_batch.view(X_batch.shape[0], -1)
            all_X.append(X_batch)
            all_y.append(y_batch.to(device))
        X = torch.cat(all_X, dim=0)
        y = torch.cat(all_y, dim=0)
        
        print(f"  [1/3] Trening projekcji ({epochs_proj} epochs)...")
        self.router.train_projection(X, y, epochs=epochs_proj, lr=lr_proj)
        
        print(f"  [2/3] Budowa label_map ({self.router.grid_size}×{self.router.grid_size})...")
        self.router.build_label_map(X, y)
        
        # Krok 3: Trenuj każdy pod na WSZYSTKICH danych
        # Oszczędność MAC bierze się z routingu (1 pod aktywny),
        # nie z ograniczenia danych treningowych.
        print(f"  [3/3] Trening podów ({epochs_pods} epochs each)...")
        criterion = nn.CrossEntropyLoss()
        
        for pod_id in range(self.n_pods):
            optimizer = torch.optim.Adam(self.pods[pod_id].parameters(), lr=lr_pods)
            
            for epoch in range(epochs_pods):
                perm = torch.randperm(len(X))
                for start in range(0, len(X), 512):
                    end = min(start + 512, len(X))
                    idx = perm[start:end]
                    logits = self.pods[pod_id](X[idx])
                    loss = criterion(logits, y[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        print("  Trening zakończony.")
    
    def train_incremental(self, train_loader, device, new_classes,
                          epochs_proj=30, epochs_pods=5,
                          lr_proj=0.001, lr_pods=0.001):
        """
        Trening inkrementalny — dodawanie nowych klas BEZ zapominania starych.
        
        Strategia M.A.R.S.:
          1. Zamroź pody starych klas (nie tracą wiedzy)
          2. Dotrenuj encoder z niskim LR (adaptacja, nie destrukcja)
          3. Rozszerz label_map o nowe klasy (nie nadpisuj starych)
          4. Trenuj TYLKO pody nowych klas
        """
        # Zbierz dane
        all_X, all_y = [], []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            if X_batch.dim() > 2:
                X_batch = X_batch.view(X_batch.shape[0], -1)
            all_X.append(X_batch)
            all_y.append(y_batch.to(device))
        X = torch.cat(all_X, dim=0)
        y = torch.cat(all_y, dim=0)
        
        old_classes = set(range(self.n_pods)) - set(new_classes)
        
        # 1. Zamroź pody starych klas
        for pod_id in old_classes:
            for param in self.pods[pod_id].parameters():
                param.requires_grad = False
        
        # 2. Fine-tune encoder z niskim LR (zachowaj stare mapowanie)
        print(f"  [1/3] Fine-tune encoder ({epochs_proj} epochs, lr={lr_proj})...")
        self.router.train_projection(X, y, epochs=epochs_proj, lr=lr_proj)
        
        # 3. Rozszerz label_map (merge z istniejącą)
        print(f"  [2/3] Rozszerzenie label_map...")
        self._merge_label_map(X, y, new_classes)
        
        # 4. Trenuj TYLKO pody nowych klas
        print(f"  [3/3] Trening nowych podów ({epochs_pods} epochs)...")
        criterion = nn.CrossEntropyLoss()
        
        for pod_id in new_classes:
            # Odmroź
            for param in self.pods[pod_id].parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.Adam(self.pods[pod_id].parameters(), lr=lr_pods)
            
            for epoch in range(epochs_pods):
                perm = torch.randperm(len(X))
                for start in range(0, len(X), 512):
                    end = min(start + 512, len(X))
                    idx = perm[start:end]
                    logits = self.pods[pod_id](X[idx])
                    loss = criterion(logits, y[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Odmroź stare pody (na wypadek dalszego treningu)
        for pod_id in old_classes:
            for param in self.pods[pod_id].parameters():
                param.requires_grad = True
        
        print("  Trening inkrementalny zakończony.")
    
    def _merge_label_map(self, X, labels, new_classes):
        """Rozszerz label_map o nowe klasy bez nadpisywania starych."""
        with torch.no_grad():
            all_uv = []
            for start in range(0, len(X), 4096):
                end = min(start + 4096, len(X))
                uv_batch = self.router.encode_to_uv(X[start:end])
                all_uv.append(uv_batch)
            uv = torch.cat(all_uv, dim=0)
            
            gs = self.router.grid_size
            # Only count votes for NEW classes
            new_votes = torch.zeros(gs, gs, self.n_pods, device=X.device)
            
            grid_coords = (uv * (gs - 1)).long().clamp(0, gs - 1)
            
            for i in range(len(X)):
                label = labels[i].item()
                if label in new_classes:
                    gx, gy = grid_coords[i, 0].item(), grid_coords[i, 1].item()
                    new_votes[gy, gx, label] += 1
            
            # Current label map
            current_map = self.router.label_map.view(gs, gs)
            current_conf = self.router.confidence_map.view(gs, gs)
            
            # Merge: jeśli nowa klasa dominuje komórkę → aktualizuj
            new_total = new_votes.sum(dim=2)
            new_max_votes = new_votes.max(dim=2).values
            new_labels = new_votes.argmax(dim=2).float()
            
            # Aktualizuj tylko komórki gdzie nowe klasy mają silną obecność
            # i stary confidence jest niski (lub komórka pusta)
            update_mask = (new_total > 5) & (new_max_votes / (new_total + 1e-8) > 0.5)
            
            updated_map = current_map.clone()
            updated_conf = current_conf.clone()
            
            updated_map[update_mask] = new_labels[update_mask]
            updated_conf[update_mask] = (new_max_votes / (new_total + 1e-8))[update_mask]
            
            self.router.label_map.copy_(updated_map.view(1, 1, gs, gs))
            self.router.confidence_map.copy_(updated_conf.view(1, 1, gs, gs))
            
            n_updated = update_mask.sum().item()
            print(f"    Zaktualizowano {n_updated}/{gs*gs} komórek ({n_updated/gs/gs*100:.1f}%)")


# ─── Baseline MLP (do porównania) ───────────────────────────────────────────

class BaselineMLP(nn.Module):
    """Standardowy monolityczny MLP — baseline."""
    
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_out)
        )
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
    
    def forward(self, x):
        return self.net(x)
    
    def mac_per_sample(self):
        h2 = self.n_hidden // 2
        return (self.n_in * self.n_hidden + 
                self.n_hidden * h2 + 
                h2 * self.n_out)


# ─── Neural Router (Faza 1 baseline) ────────────────────────────────────────

class NeuralRouter(nn.Module):
    """
    Klasyczny router (2-layer MLP) — baseline do porównania z SOM.
    MAC: N_IN × HIDDEN + HIDDEN × N_PODS
    """
    
    def __init__(self, n_in, n_hidden, n_pods):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_pods)
        )
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_pods = n_pods
    
    def forward(self, x):
        return self.net(x).argmax(dim=1)
    
    def mac_per_sample(self):
        return self.n_in * self.n_hidden + self.n_hidden * self.n_pods


# ─── Utility: benchmark timing ──────────────────────────────────────────────

def benchmark_inference(model, x, n_warmup=10, n_runs=100, device='cuda'):
    """Benchmark inferencji z CUDA events."""
    model.eval()
    x = x.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            for _ in range(n_runs):
                model(x)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    
    time_per_batch_us = (elapsed_ms / n_runs) * 1000
    samples_per_sec = x.shape[0] / (elapsed_ms / n_runs / 1000)
    
    return {
        "time_per_batch_us": round(time_per_batch_us, 1),
        "samples_per_sec": round(samples_per_sec),
        "batch_size": x.shape[0],
        "n_runs": n_runs,
    }
