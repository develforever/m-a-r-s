"""
M.A.R.S. Faza 2 — Priorytet 4: Sleep v2 + Ternary Weights na GPU

Sleep v2: selective decay + Hebbian wzmocnienie + pruning martwych neuronów
Ternary Weights: kwantyzacja wag SOM encodera do [-1, 0, 1]

Cel: Potwierdzić stabilność systemu po 100+ cyklach snu
     + pokazać że ternary weights zachowują accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional
from mars_torch import MARSystem, BaselineMLP


# ═══════════════════════════════════════════════════════════════════════════════
# SLEEP v2 — Cykl konsolidacji pamięci
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SleepConfig:
    """Konfiguracja cyklu snu."""
    decay_rate: float = 0.01        # Selektywny decay (osłabienie nieużywanych ścieżek)
    hebbian_lr: float = 0.005       # Wzmocnienie Hebbian (co razem się aktywuje)
    prune_threshold: float = 0.01   # Próg przycinania martwych wag
    prune_reinit_scale: float = 0.1 # Skala reinicjalizacji przyciętych neuronów
    n_replay_samples: int = 1000    # Ile próbek replay podczas snu


class SleepCycle:
    """
    Sleep v2: Cykl konsolidacji pamięci dla M.A.R.S.
    
    Biologiczna inspiracja:
      1. Selective Decay — osłabianie rzadko używanych połączeń
      2. Hebbian Reinforcement — wzmacnianie korelowanych ścieżek
      3. Pruning + Re-init — usuwanie martwych neuronów, re-alokacja zasobów
    """
    
    def __init__(self, config: SleepConfig = SleepConfig()):
        self.config = config
        self.history: list[dict] = []
    
    def run_cycle(self, system: MARSystem, replay_data: Optional[torch.Tensor] = None,
                  replay_labels: Optional[torch.Tensor] = None) -> dict:
        """
        Wykonaj jeden cykl snu na systemie M.A.R.S.
        
        Returns: metryki cyklu (decay_magnitude, hebbian_magnitude, pruned_count)
        """
        metrics = {
            'decay_magnitude': 0.0,
            'hebbian_magnitude': 0.0,
            'pruned_count': 0,
            'reactivated_count': 0,
        }
        
        device = next(system.parameters()).device
        
        # ─── Phase 1: Selective Decay ─────────────────────────────────────────
        # Osłabianie wag proporcjonalnie do ich bezwzględnej wartości
        # (małe wagi → prawdopodobnie nieistotne → decay szybciej)
        total_decay = 0.0
        with torch.no_grad():
            for pod in system.pods:
                for param in pod.parameters():
                    if param.dim() >= 2:  # Tylko wagi (nie biasy)
                        magnitude = param.abs()
                        # Decay: wagi bliskie 0 tracą więcej
                        decay_mask = torch.exp(-magnitude * 10)  # małe wagi → duży decay
                        decay = param * decay_mask * self.config.decay_rate
                        param.sub_(decay)
                        total_decay += decay.abs().sum().item()
        
        metrics['decay_magnitude'] = total_decay
        
        # ─── Phase 2: Hebbian Reinforcement ───────────────────────────────────
        # Wzmacnianie wag które konsekwentnie współpracują (replay)
        if replay_data is not None and replay_labels is not None:
            hebbian_total = 0.0
            n_samples = min(self.config.n_replay_samples, len(replay_data))
            idx = torch.randperm(len(replay_data))[:n_samples]
            X_replay = replay_data[idx].to(device)
            y_replay = replay_labels[idx].to(device)
            
            # Forward pass — zbierz aktywacje
            system.eval()
            with torch.no_grad():
                capsule_ids, _ = system.router(X_replay)
                
                for pod_id in range(system.n_pods):
                    mask = capsule_ids == pod_id
                    if mask.sum() < 2:
                        continue
                    
                    X_pod = X_replay[mask]
                    y_pod = y_replay[mask]
                    
                    # Forward przez pod — zbierz gradienty korelacji
                    logits = system.pods[pod_id](X_pod)
                    correct = logits.argmax(dim=1) == y_pod
                    
                    if correct.sum() == 0:
                        continue
                    
                    # Hebbian: wzmocnij wagi które dały poprawne predykcje
                    X_correct = X_pod[correct]
                    for name, param in system.pods[pod_id].named_parameters():
                        if 'weight' in name and param.dim() >= 2:
                            # Wzmocnienie proporcjonalne do aktywacji
                            activation_strength = X_correct.abs().mean(dim=0)
                            if param.shape[1] == activation_strength.shape[0]:
                                reinforcement = self.config.hebbian_lr * \
                                    (activation_strength.unsqueeze(0) * param.sign())
                                param.add_(reinforcement)
                                hebbian_total += reinforcement.abs().sum().item()
            
            metrics['hebbian_magnitude'] = hebbian_total
        
        # ─── Phase 3: Pruning + Re-initialization ────────────────────────────
        # Usuwanie martwych wag i reinicjalizacja
        pruned = 0
        reactivated = 0
        with torch.no_grad():
            for pod in system.pods:
                for param in pod.parameters():
                    if param.dim() >= 2:
                        dead_mask = param.abs() < self.config.prune_threshold
                        n_dead = dead_mask.sum().item()
                        
                        if n_dead > 0:
                            # Reinicjalizuj martwe wagi z małym szumem
                            noise = torch.randn_like(param) * self.config.prune_reinit_scale
                            param[dead_mask] = noise[dead_mask]
                            pruned += n_dead
                            reactivated += n_dead
        
        metrics['pruned_count'] = pruned
        metrics['reactivated_count'] = reactivated
        
        self.history.append(metrics)
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TERNARY WEIGHTS — Kwantyzacja do [-1, 0, 1]
# ═══════════════════════════════════════════════════════════════════════════════

class TernaryQuantizer:
    """
    Kwantyzacja wag do [-1, 0, 1] (Ternary Weight Networks).
    
    Metoda: threshold-based quantization
      w_q = +1 jeśli w > threshold
      w_q =  0 jeśli |w| ≤ threshold  
      w_q = -1 jeśli w < -threshold
    
    Threshold = 0.7 × mean(|w|) (heurystyka z TWN paper)
    """
    
    @staticmethod
    def quantize_model(model: nn.Module, threshold_factor: float = 0.7) -> dict:
        """
        Kwantyzuj wszystkie wagi modelu do [-1, 0, 1].
        
        Returns: dict z metrykami kwantyzacji
        """
        metrics = {
            'total_params': 0,
            'nonzero_params': 0,
            'ternary_params': 0,
            'sparsity': 0.0,
            'layers': [],
        }
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() < 2:
                    continue  # Skip biases
                
                n_params = param.numel()
                metrics['total_params'] += n_params
                
                # Compute threshold
                threshold = threshold_factor * param.abs().mean()
                
                # Quantize
                ternary = torch.zeros_like(param)
                ternary[param > threshold] = 1.0
                ternary[param < -threshold] = -1.0
                
                # Store scaling factor for better approximation
                pos_mask = param > threshold
                neg_mask = param < -threshold
                alpha = 0.0
                n_nonzero = pos_mask.sum() + neg_mask.sum()
                if n_nonzero > 0:
                    alpha = (param[pos_mask].sum() - param[neg_mask].sum()) / n_nonzero
                
                # Apply: scaled ternary
                param.copy_(ternary * alpha.abs())
                
                layer_sparsity = (ternary == 0).float().mean().item()
                metrics['layers'].append({
                    'name': name,
                    'shape': list(param.shape),
                    'threshold': threshold.item(),
                    'alpha': alpha.item() if isinstance(alpha, torch.Tensor) else alpha,
                    'sparsity': layer_sparsity,
                    'n_params': n_params,
                })
                
                metrics['nonzero_params'] += n_nonzero.item()
                metrics['ternary_params'] += n_params
        
        if metrics['ternary_params'] > 0:
            metrics['sparsity'] = 1.0 - metrics['nonzero_params'] / metrics['ternary_params']
        
        return metrics
    
    @staticmethod
    def measure_compression(metrics: dict) -> dict:
        """Oblicz teoretyczny współczynnik kompresji."""
        # Full precision: 32 bits per param
        # Ternary: 2 bits per param (00=0, 01=+1, 10=-1) + 1 alpha per layer
        full_bits = metrics['total_params'] * 32
        ternary_bits = metrics['ternary_params'] * 2 + len(metrics['layers']) * 32
        
        return {
            'full_precision_bits': full_bits,
            'ternary_bits': ternary_bits,
            'compression_ratio': full_bits / max(ternary_bits, 1),
            'memory_savings_pct': (1 - ternary_bits / max(full_bits, 1)) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def run_sleep_stability_test(device):
    """Test stabilności po 100 cyklach snu."""
    print("\n" + "=" * 64)
    print("TEST 1: Sleep v2 — Stabilność po 100 cyklach")
    print("=" * 64)
    
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024)
    
    # Trenuj system
    print("\n  Trening M.A.R.S. (encoder_hidden=64, grid=64)...")
    system = MARSystem(784, 10, 64, 64, encoder_hidden=64).to(device)
    system.train()
    system.train_system(train_loader, device, epochs_proj=40, epochs_pods=3,
                        lr_proj=0.003, lr_pods=0.001)
    
    # Zbierz dane do replay
    all_X, all_y = [], []
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.view(-1, 784).to(device)
        all_X.append(X_batch)
        all_y.append(y_batch.to(device))
        if len(all_X) * 256 >= 5000:
            break
    replay_X = torch.cat(all_X, dim=0)[:5000]
    replay_y = torch.cat(all_y, dim=0)[:5000]
    
    # Evaluate bazowa accuracy
    def eval_acc():
        system.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.view(-1, 784).to(device)
                y_batch = y_batch.to(device)
                logits, _, _ = system(X_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        return correct / total
    
    initial_acc = eval_acc()
    print(f"  Accuracy przed snem: {initial_acc*100:.2f}%")
    
    # Run 100 cykli snu
    sleep = SleepCycle(SleepConfig(
        decay_rate=0.005,
        hebbian_lr=0.002,
        prune_threshold=0.005,
        n_replay_samples=500,
    ))
    
    print("\n  Uruchamiam 100 cykli snu...")
    accuracies = [initial_acc]
    checkpoints = [0, 10, 25, 50, 75, 100]
    
    for cycle in range(1, 101):
        metrics = sleep.run_cycle(system, replay_X, replay_y)
        
        if cycle in checkpoints:
            acc = eval_acc()
            accuracies.append(acc)
            print(f"    Cykl {cycle:3d}: acc={acc*100:.2f}% | "
                  f"decay={metrics['decay_magnitude']:.1f} | "
                  f"hebbian={metrics['hebbian_magnitude']:.1f} | "
                  f"pruned={metrics['pruned_count']}")
    
    final_acc = accuracies[-1]
    acc_drop = initial_acc - final_acc
    
    print(f"\n  ┌─ Wyniki Sleep v2 ──────────────────────────────────────────┐")
    print(f"  │ Accuracy początkowa:  {initial_acc*100:.2f}%")
    print(f"  │ Accuracy po 100 cykli: {final_acc*100:.2f}%")
    print(f"  │ Spadek:                {acc_drop*100:.2f}pp")
    print(f"  │ Stabilność:            {'✓ STABILNY' if acc_drop < 5.0 else '✗ NIESTABILNY'}")
    print(f"  └──────────────────────────────────────────────────────────────┘")
    
    return {
        'initial_acc': initial_acc,
        'final_acc': final_acc,
        'acc_drop_pp': acc_drop * 100,
        'stable': acc_drop < 5.0,
        'accuracies_at_checkpoints': {str(c): a for c, a in zip([0] + checkpoints[1:], accuracies)},
        'n_cycles': 100,
    }


def run_ternary_test(device):
    """Test kwantyzacji ternary weights."""
    print("\n" + "=" * 64)
    print("TEST 2: Ternary Weights — Kwantyzacja [-1, 0, 1]")
    print("=" * 64)
    
    from torchvision import datasets, transforms
    import copy
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_data = datasets.MNIST('./data', train=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    
    # Trenuj system
    print("\n  Trening M.A.R.S....")
    system = MARSystem(784, 10, 64, 64, encoder_hidden=64).to(device)
    system.train()
    system.train_system(train_loader, device, epochs_proj=40, epochs_pods=3,
                        lr_proj=0.003, lr_pods=0.001)
    
    # Eval przed kwantyzacją
    def eval_acc(model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.view(-1, 784).to(device)
                y_batch = y_batch.to(device)
                logits, _, _ = model(X_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        return correct / total
    
    acc_before = eval_acc(system)
    print(f"  Accuracy (full precision): {acc_before*100:.2f}%")
    
    # Kwantyzuj
    system_q = copy.deepcopy(system)
    quant_metrics = TernaryQuantizer.quantize_model(system_q, threshold_factor=0.7)
    compression = TernaryQuantizer.measure_compression(quant_metrics)
    
    acc_after = eval_acc(system_q)
    print(f"  Accuracy (ternary):        {acc_after*100:.2f}%")
    print(f"  Spadek:                    {(acc_before - acc_after)*100:.2f}pp")
    
    print(f"\n  ┌─ Metryki kompresji ─────────────────────────────────────────┐")
    print(f"  │ Parametry (total):     {quant_metrics['total_params']:>10,}")
    print(f"  │ Parametry (nonzero):   {quant_metrics['nonzero_params']:>10,}")
    print(f"  │ Sparsity:              {quant_metrics['sparsity']*100:>9.1f}%")
    print(f"  │ Compression ratio:     {compression['compression_ratio']:>9.1f}×")
    print(f"  │ Memory savings:        {compression['memory_savings_pct']:>9.1f}%")
    print(f"  └──────────────────────────────────────────────────────────────┘")
    
    # Benchmark: czy ternary jest szybszy?
    print("\n  Throughput comparison (batch=4096)...")
    x = torch.randn(4096, 784).to(device)
    
    # Full precision
    system.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            system(x)
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / 50
    
    # Ternary
    system_q.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            system_q(x)
    torch.cuda.synchronize()
    t_ternary = (time.perf_counter() - t0) / 50
    
    print(f"  Full precision: {4096/t_full:,.0f} samples/s ({t_full*1e6:.1f}μs/batch)")
    print(f"  Ternary:        {4096/t_ternary:,.0f} samples/s ({t_ternary*1e6:.1f}μs/batch)")
    print(f"  Speedup:        {t_full/t_ternary:.2f}×")
    
    return {
        'acc_full_precision': acc_before,
        'acc_ternary': acc_after,
        'acc_drop_pp': (acc_before - acc_after) * 100,
        'sparsity': quant_metrics['sparsity'],
        'compression_ratio': compression['compression_ratio'],
        'memory_savings_pct': compression['memory_savings_pct'],
        'throughput_full': 4096 / t_full,
        'throughput_ternary': 4096 / t_ternary,
        'speedup': t_full / t_ternary,
    }


def main():
    print("=" * 64)
    print("M.A.R.S. FAZA 2 — Priorytet 4: Sleep v2 + Ternary Weights")
    print("=" * 64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    
    # Test 1: Sleep stability
    sleep_results = run_sleep_stability_test(device)
    
    # Test 2: Ternary weights
    ternary_results = run_ternary_test(device)
    
    # Summary
    print("\n" + "=" * 64)
    print("PODSUMOWANIE — Sleep v2 + Ternary Weights")
    print("=" * 64)
    
    print(f"\n  ┌─ WERDYKT ─────────────────────────────────────────────────┐")
    print(f"  │ Sleep stabilny po 100 cykli: {'✓' if sleep_results['stable'] else '✗'} "
          f"(spadek {sleep_results['acc_drop_pp']:.1f}pp)")
    print(f"  │ Ternary accuracy drop < 5pp: {'✓' if ternary_results['acc_drop_pp'] < 5 else '✗'} "
          f"(spadek {ternary_results['acc_drop_pp']:.1f}pp)")
    print(f"  │ Kompresja pamięci:           {ternary_results['memory_savings_pct']:.0f}%")
    print(f"  │ Sparsity:                    {ternary_results['sparsity']*100:.0f}%")
    
    all_pass = sleep_results['stable'] and ternary_results['acc_drop_pp'] < 5
    print(f"  │")
    print(f"  │ WERDYKT: {'✓ POZYTYWNY' if all_pass else '⚠ WYMAGA DALSZEJ PRACY'}")
    print(f"  └────────────────────────────────────────────────────────────┘")
    
    # Save results
    results = {
        'experiment': 'M.A.R.S. Faza 2 P4 — Sleep v2 + Ternary',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'device': str(device),
        'sleep_v2': sleep_results,
        'ternary_weights': ternary_results,
        'verdict': 'POZYTYWNY' if all_pass else 'WYMAGA DALSZEJ PRACY',
    }
    
    import os
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'faza2_sleep_ternary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Wynik zapisany: {out_path}")


if __name__ == '__main__':
    main()
