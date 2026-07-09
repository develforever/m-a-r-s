"""
run_faza2_mnist.py — Faza 2 Priorytet 1: Walidacja MNIST.

Pipeline: Input[784] → Projection[784,2] → sigmoid → UV → grid_sample → capsule_id
Cel: udowodnić, że 98.5% oszczędności MAC NIE kosztem accuracy.

Testy:
  1. Accuracy: M.A.R.S. (10 pods + SOM router) vs Baseline MLP
  2. MAC: porównanie kosztów per inference
  3. Throughput: samples/s na GPU (CUDA)
  4. Catastrophic forgetting: nauka sekwencyjna (0-4 → 5-9)
  5. Sleep cycle: stabilność po N cyklach

Uruchom:
    .venv\\Scripts\\python.exe src\\run_faza2_mnist.py
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from mars_torch import (
    MARSystem, BaselineMLP, NeuralRouter, SOMProjectionRouter,
    benchmark_inference, MACReport
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_mnist(batch_size=256):
    """Załaduj MNIST train/test."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset


# ─── Test 1: Accuracy comparison ────────────────────────────────────────────

def test_accuracy(device):
    """Porównanie accuracy: M.A.R.S. vs Baseline MLP."""
    print("\n" + "=" * 64)
    print("TEST 1: Accuracy — M.A.R.S. vs Baseline MLP (MNIST)")
    print("=" * 64)
    
    train_loader, test_loader, _, _ = load_mnist(batch_size=512)
    
    # ─── Baseline MLP ───
    print("\n  [Baseline MLP] 784→256→128→10")
    baseline = BaselineMLP(784, 256, 10).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    baseline.train()
    for epoch in range(5):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.view(-1, 784).to(device)
            y_batch = y_batch.to(device)
            logits = baseline(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 2 == 0:
            print(f"    epoch {epoch+1}/5, loss={total_loss/len(train_loader):.4f}")
    
    # Eval baseline
    baseline.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.view(-1, 784).to(device)
            y_batch = y_batch.to(device)
            pred = baseline(X_batch).argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)
    baseline_acc = correct / total
    print(f"    Baseline accuracy: {baseline_acc*100:.2f}%")
    
    # ─── M.A.R.S. (10 pods + SOM Projection Router) ───
    print("\n  [M.A.R.S.] 10 pods × (784→64→10) + SOM Encoder Router (h=64)")
    mars = MARSystem(n_in=784, n_pods=10, pod_hidden=64, grid_size=64, 
                     encoder_hidden=64).to(device)
    
    # Trening systemowy
    mars.train()
    mars.train_system(train_loader, device, 
                      epochs_proj=60, epochs_pods=5,
                      lr_proj=0.003, lr_pods=0.001)
    
    # Eval M.A.R.S.
    mars.eval()
    correct, total = 0, 0
    routing_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.view(-1, 784).to(device)
            y_batch = y_batch.to(device)
            logits, capsule_ids, conf = mars(X_batch)
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            routing_correct += (capsule_ids == y_batch).sum().item()
            total += len(y_batch)
    mars_acc = correct / total
    routing_acc = routing_correct / total
    print(f"    M.A.R.S. accuracy:   {mars_acc*100:.2f}%")
    print(f"    Routing accuracy:    {routing_acc*100:.2f}%")
    print(f"    Δ vs baseline:       {(mars_acc - baseline_acc)*100:+.2f}pp")
    
    return {
        "baseline_acc": round(baseline_acc, 4),
        "mars_acc": round(mars_acc, 4),
        "routing_acc": round(routing_acc, 4),
        "delta_pp": round((mars_acc - baseline_acc) * 100, 2),
    }


# ─── Test 2: MAC comparison ─────────────────────────────────────────────────

def test_mac(device):
    """Porównanie kosztów MAC per sample."""
    print("\n" + "=" * 64)
    print("TEST 2: MAC Analysis — koszt per inference")
    print("=" * 64)
    
    n_in = 784
    n_pods = 10
    pod_hidden = 64
    grid_size = 64
    encoder_hidden = 64
    
    mars = MARSystem(n_in, n_pods, pod_hidden, grid_size, encoder_hidden)
    baseline = BaselineMLP(n_in, 256, n_pods)
    neural_router = NeuralRouter(n_in, 128, n_pods)
    
    # MAC calculations
    som_router_mac = mars.router.mac_per_sample()  # encoder: n_in*H + H*2
    neural_router_mac = neural_router.mac_per_sample()
    pod_mac = mars.pods[0].mac_per_sample()
    baseline_mac = baseline.mac_per_sample()
    
    mars_total = som_router_mac + pod_mac  # router + 1 pod
    mars_dense = som_router_mac + pod_mac * n_pods  # router + all pods
    
    print(f"\n  ┌─ MAC per sample ─────────────────────────────────────────┐")
    print(f"  │ Component              │ MAC        │ Notes              │")
    print(f"  ├────────────────────────┼────────────┼────────────────────┤")
    print(f"  │ SOM Encoder Router     │ {som_router_mac:>10,} │ N×H + H×2 (enc)    │")
    print(f"  │ Neural Router (128h)   │ {neural_router_mac:>10,} │ N_IN×H + H×K       │")
    print(f"  │ 1 Pod (784→64→10)     │ {pod_mac:>10,} │ 1 specjalista      │")
    print(f"  │ Baseline MLP           │ {baseline_mac:>10,} │ 784→256→128→10     │")
    print(f"  ├────────────────────────┼────────────┼────────────────────┤")
    print(f"  │ M.A.R.S. (routed)     │ {mars_total:>10,} │ SOM + 1 pod        │")
    print(f"  │ M.A.R.S. (dense)      │ {mars_dense:>10,} │ SOM + all pods     │")
    print(f"  │ Baseline MLP           │ {baseline_mac:>10,} │ monolityczny       │")
    print(f"  └────────────────────────┴────────────┴────────────────────┘")
    print(f"\n  Oszczędność M.A.R.S. vs Baseline:  {(1-mars_total/baseline_mac)*100:.1f}%")
    print(f"  Oszczędność SOM vs Neural Router:  {(1-som_router_mac/neural_router_mac)*100:.1f}%")
    
    return {
        "som_router_mac": som_router_mac,
        "neural_router_mac": neural_router_mac,
        "pod_mac": pod_mac,
        "baseline_mac": baseline_mac,
        "mars_routed_mac": mars_total,
        "mars_dense_mac": mars_dense,
        "savings_vs_baseline_pct": round((1 - mars_total / baseline_mac) * 100, 1),
        "savings_som_vs_neural_pct": round((1 - som_router_mac / neural_router_mac) * 100, 1),
    }


# ─── Test 3: Throughput benchmark ───────────────────────────────────────────

def test_throughput(device):
    """Benchmark throughput: samples/s."""
    print("\n" + "=" * 64)
    print("TEST 3: Throughput — samples/s (GPU benchmark)")
    print("=" * 64)
    
    batch_size = 4096
    x = torch.randn(batch_size, 784).to(device)
    
    # M.A.R.S.
    mars = MARSystem(784, 10, 64, 64, encoder_hidden=64).to(device)
    mars.eval()
    mars_bench = benchmark_inference(mars, x, device=str(device))
    print(f"\n  M.A.R.S. (routed):     {mars_bench['samples_per_sec']:>12,} samples/s")
    print(f"                         {mars_bench['time_per_batch_us']:>8.1f} μs/batch")
    
    # Baseline
    baseline = BaselineMLP(784, 256, 10).to(device)
    baseline.eval()
    baseline_bench = benchmark_inference(baseline, x, device=str(device))
    print(f"  Baseline MLP:          {baseline_bench['samples_per_sec']:>12,} samples/s")
    print(f"                         {baseline_bench['time_per_batch_us']:>8.1f} μs/batch")
    
    # Pure encoder + grid_sample (router only)
    router = SOMProjectionRouter(784, 10, 64, encoder_hidden=64).to(device)
    router.eval()
    router_bench = benchmark_inference(router, x, device=str(device))
    print(f"  SOM Router only:       {router_bench['samples_per_sec']:>12,} samples/s")
    print(f"                         {router_bench['time_per_batch_us']:>8.1f} μs/batch")
    
    speedup = baseline_bench['time_per_batch_us'] / mars_bench['time_per_batch_us']
    print(f"\n  Speedup M.A.R.S. vs Baseline: {speedup:.2f}×")
    
    return {
        "mars": mars_bench,
        "baseline": baseline_bench,
        "router_only": router_bench,
        "speedup_vs_baseline": round(speedup, 2),
        "device": str(device),
    }


# ─── Test 4: Catastrophic forgetting ────────────────────────────────────────

def test_forgetting(device):
    """Test catastrophic forgetting: nauka 0-4 → 5-9 → retencja."""
    print("\n" + "=" * 64)
    print("TEST 4: Catastrophic Forgetting (0-4 → 5-9)")
    print("=" * 64)
    
    _, _, train_dataset, test_dataset = load_mnist(batch_size=256)
    
    # Podziel na task A (cyfry 0-4) i task B (cyfry 5-9)
    def get_subset(dataset, classes):
        indices = [i for i, (_, y) in enumerate(dataset) if y in classes]
        return Subset(dataset, indices)
    
    task_a_classes = list(range(5))
    task_b_classes = list(range(5, 10))
    
    train_a = DataLoader(get_subset(train_dataset, task_a_classes), 
                        batch_size=256, shuffle=True)
    train_b = DataLoader(get_subset(train_dataset, task_b_classes),
                        batch_size=256, shuffle=True)
    test_a = DataLoader(get_subset(test_dataset, task_a_classes),
                       batch_size=256, shuffle=False)
    test_b = DataLoader(get_subset(test_dataset, task_b_classes),
                       batch_size=256, shuffle=False)
    
    def eval_accuracy(model, loader, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.view(-1, 784).to(device)
                y_batch = y_batch.to(device)
                if hasattr(model, 'router'):
                    logits, _, _ = model(X_batch)
                else:
                    logits = model(X_batch)
                pred = logits.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += len(y_batch)
        return correct / total if total > 0 else 0
    
    def train_on_loader(model, loader, device, epochs=5, lr=0.001):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.view(-1, 784).to(device)
                y_batch = y_batch.to(device)
                if hasattr(model, 'router'):
                    logits, _, _ = model(X_batch)
                else:
                    logits = model(X_batch)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    # ─── Baseline MLP ───
    print("\n  [Baseline MLP]")
    baseline = BaselineMLP(784, 256, 10).to(device)
    
    # Ucz na task A
    train_on_loader(baseline, train_a, device, epochs=5)
    base_acc_a_after_a = eval_accuracy(baseline, test_a, device)
    print(f"    Po nauce A (0-4): acc_A = {base_acc_a_after_a*100:.1f}%")
    
    # Ucz na task B
    train_on_loader(baseline, train_b, device, epochs=5)
    base_acc_a_after_b = eval_accuracy(baseline, test_a, device)
    base_acc_b_after_b = eval_accuracy(baseline, test_b, device)
    print(f"    Po nauce B (5-9): acc_A = {base_acc_a_after_b*100:.1f}% (retencja)")
    print(f"                      acc_B = {base_acc_b_after_b*100:.1f}%")
    base_retention = base_acc_a_after_b / max(base_acc_a_after_a, 1e-8)
    print(f"    Retencja A: {base_retention*100:.1f}%")
    
    # ─── M.A.R.S. (modularny — pody izolowane per task) ───
    print("\n  [M.A.R.S. modularny — trening inkrementalny]")
    mars = MARSystem(784, 10, 64, 64, encoder_hidden=64).to(device)
    
    # Ucz na task A (pełny trening systemowy)
    mars.train()
    mars.train_system(train_a, device, epochs_proj=60, epochs_pods=5,
                      lr_proj=0.003, lr_pods=0.001)
    mars_acc_a_after_a = eval_accuracy(mars, test_a, device)
    print(f"    Po nauce A (0-4): acc_A = {mars_acc_a_after_a*100:.1f}%")
    
    # Ucz na task B — INKREMENTALNIE:
    # Zamroź pody 0-4, fine-tune encoder, trenuj tylko pody 5-9
    mars.train()
    mars.train_incremental(train_b, device, new_classes=list(range(5, 10)),
                           epochs_proj=30, epochs_pods=5,
                           lr_proj=0.001, lr_pods=0.001)
    mars_acc_a_after_b = eval_accuracy(mars, test_a, device)
    mars_acc_b_after_b = eval_accuracy(mars, test_b, device)
    print(f"    Po nauce B (5-9): acc_A = {mars_acc_a_after_b*100:.1f}% (retencja)")
    print(f"                      acc_B = {mars_acc_b_after_b*100:.1f}%")
    mars_retention = mars_acc_a_after_b / max(mars_acc_a_after_a, 1e-8)
    print(f"    Retencja A: {mars_retention*100:.1f}%")
    
    print(f"\n  ┌─ Porównanie retencji ────────────────────────────────────┐")
    print(f"  │ System       │ Retencja A │ Acc B │ Δ retencji         │")
    print(f"  │ Baseline MLP │ {base_retention*100:>9.1f}% │ {base_acc_b_after_b*100:.1f}% │                    │")
    print(f"  │ M.A.R.S.     │ {mars_retention*100:>9.1f}% │ {mars_acc_b_after_b*100:.1f}% │ {(mars_retention-base_retention)*100:+.1f}pp           │")
    print(f"  └──────────────┴────────────┴───────┴────────────────────┘")
    
    return {
        "baseline": {
            "acc_a_after_a": round(base_acc_a_after_a, 4),
            "acc_a_after_b": round(base_acc_a_after_b, 4),
            "acc_b_after_b": round(base_acc_b_after_b, 4),
            "retention": round(base_retention, 4),
        },
        "mars": {
            "acc_a_after_a": round(mars_acc_a_after_a, 4),
            "acc_a_after_b": round(mars_acc_a_after_b, 4),
            "acc_b_after_b": round(mars_acc_b_after_b, 4),
            "retention": round(mars_retention, 4),
        },
        "delta_retention_pp": round((mars_retention - base_retention) * 100, 1),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("M.A.R.S. FAZA 2 — Walidacja MNIST")
    print(f"  Pipeline: Input[784] → Projection[784,2] → UV → TMU → capsule_id")
    print("=" * 64)
    
    device = get_device()
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    results = {
        "experiment": "M.A.R.S. Faza 2 — MNIST Validation",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "pytorch_version": torch.__version__,
    }
    
    # Test 1: Accuracy
    results["test1_accuracy"] = test_accuracy(device)
    
    # Test 2: MAC analysis
    results["test2_mac"] = test_mac(device)
    
    # Test 3: Throughput
    results["test3_throughput"] = test_throughput(device)
    
    # Test 4: Catastrophic forgetting
    results["test4_forgetting"] = test_forgetting(device)
    
    # ─── Summary ───
    print("\n" + "=" * 64)
    print("PODSUMOWANIE FAZY 2 — MNIST")
    print("=" * 64)
    
    t1 = results["test1_accuracy"]
    t2 = results["test2_mac"]
    t4 = results["test4_forgetting"]
    
    print(f"\n  Accuracy:   M.A.R.S. {t1['mars_acc']*100:.1f}% vs Baseline {t1['baseline_acc']*100:.1f}%")
    print(f"  MAC saving: {t2['savings_vs_baseline_pct']}% (SOM router: {t2['som_router_mac']} MAC)")
    print(f"  Retencja:   M.A.R.S. {t4['mars']['retention']*100:.1f}% vs Baseline {t4['baseline']['retention']*100:.1f}%")
    print(f"              Δ = {t4['delta_retention_pp']:+.1f}pp")
    
    # Werdykt
    mars_ok = t1['mars_acc'] >= 0.90
    mac_ok = t2['savings_vs_baseline_pct'] > 50
    ret_ok = t4['mars']['retention'] > t4['baseline']['retention']
    
    print(f"\n  ┌─ WERDYKT ─────────────────────────────────────────────────┐")
    print(f"  │ Accuracy ≥ 90%:              {'✓' if mars_ok else '✗'} ({t1['mars_acc']*100:.1f}%)")
    print(f"  │ MAC savings > 50%:           {'✓' if mac_ok else '✗'} ({t2['savings_vs_baseline_pct']}%)")
    print(f"  │ Retencja lepsza od baseline: {'✓' if ret_ok else '✗'} ({t4['delta_retention_pp']:+.1f}pp)")
    print(f"  │")
    if mars_ok and mac_ok and ret_ok:
        print(f"  │ WERDYKT: ✓ POZYTYWNY — hipoteza potwierdzona")
    elif mars_ok and mac_ok:
        print(f"  │ WERDYKT: ⚠ WARUNKOWO POZYTYWNY — accuracy+MAC OK, retencja wymaga pracy")
    else:
        print(f"  │ WERDYKT: ⚠ WYMAGA DALSZEJ PRACY")
    print(f"  └────────────────────────────────────────────────────────────┘")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "faza2_mnist.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Wynik zapisany: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
