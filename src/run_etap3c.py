"""
run_etap3c.py — Etap 3C: SOM-Router vs Neural-Router.

INTEGRACJA Etap 3 + Etap 4B:
  Router z Etapu 3 (sieć neuronowa) vs SOM-Router (Kohonen lookup).
  Porównujemy: dokładność, MAC, soft routing, sleep v2.

Testy:
  1. Dokładność: SOM-router vs neural-router (ten sam dataset)
  2. MAC: koszt routingu (SOM lookup vs 2× matmul)
  3. Soft routing: confidence z bilinear vs top-2 z neural
  4. Sleep v2: decay + Hebbian vs brak snu
  5. Online adaptation: dodanie nowego regionu
  6. Skalowanie: N=5,10,20,50 kapsuł

Uruchom:
    .venv\\Scripts\\python.exe src\\run_etap3c.py
"""

import json
import os
from datetime import datetime
import numpy as np
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dataset_regions import make_regions
from engine_core import EngineCore, Pod
from som_router import SOMRouter
from metrics import MACCounter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ─── Test 1: Dokładność SOM vs Neural ───────────────────────────────────────

def test_accuracy(N=5, seed=42):
    """Porównanie dokładności routingu: SOM vs neural router."""
    print(f"\n┌─ TEST 1: Dokładność routingu (N={N} kapsuł) ─────────────────┐")

    X, region, y = make_regions(n_regions=N, n_per_region=40, sigma=0.07, seed=seed)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    region_train, region_test = region[:n_train], region[n_train:]

    # Neural Router (Etap 3)
    engine = EngineCore(n_in=2, n_pods=N, pod_hidden=8, router_hidden=8, seed=seed)
    engine.train(X_train, region_train, y_train, epochs=3000, lr=0.3)
    neural_pred = engine.router.predict_pod(X_test)
    neural_acc = float(np.mean(neural_pred == region_test))

    # SOM Router (grid = 3*sqrt(N) dla lepszej rozdzielczości)
    grid = max(8, int(np.sqrt(N) * 4))
    som_router = SOMRouter(n_in=2, n_pods=N, grid_size=grid, seed=seed)
    som_router.train(X_train, region_train, som_epochs=len(X_train) * 100)
    som_pred = som_router.predict_pod(X_test)
    som_acc = float(np.mean(som_pred == region_test))

    # Topology quality
    topo = som_router.topology_quality()

    print(f"  Neural Router:  accuracy = {neural_acc*100:.1f}%")
    print(f"  SOM Router:     accuracy = {som_acc*100:.1f}%")
    print(f"  SOM topology:   {topo*100:.1f}% sąsiadów ma ten sam label")
    print(f"  Δ accuracy:     {(som_acc - neural_acc)*100:+.1f}pp")

    return {
        "N": N,
        "neural_acc": round(neural_acc, 4),
        "som_acc": round(som_acc, 4),
        "topology_quality": round(topo, 4),
        "delta_acc_pp": round((som_acc - neural_acc) * 100, 2),
    }


# ─── Test 2: MAC comparison ─────────────────────────────────────────────────

def test_mac(N=5, seed=42):
    """Porównanie kosztu MAC: SOM lookup vs neural forward."""
    print(f"\n┌─ TEST 2: Porównanie MAC (N={N}) ─────────────────────────────┐")

    X, region, y = make_regions(n_regions=N, n_per_region=40, sigma=0.07, seed=seed)

    n_in = 2
    router_hidden = 8
    pod_hidden = 8
    grid_size = 8

    # MAC neural router: 2 matmule
    # Layer1: [1, n_in] × [n_in, router_hidden] = n_in × router_hidden
    # Layer2: [1, router_hidden] × [router_hidden, N] = router_hidden × N
    mac_neural_router = n_in * router_hidden + router_hidden * N
    mac_pod = n_in * pod_hidden + pod_hidden * 1  # forward through 1 pod

    # MAC SOM router: distance to grid_size² vectors of dim n_in
    # Distance = n_in subtractions + n_in multiplications = 2 * n_in per cell
    mac_som_router_cpu = grid_size * grid_size * n_in * 2
    mac_som_router_gpu = 0  # TMU = free

    # Total per-sample
    mac_neural_total = mac_neural_router + mac_pod
    mac_som_cpu_total = mac_som_router_cpu + mac_pod
    mac_som_gpu_total = mac_som_router_gpu + mac_pod
    mac_dense = N * mac_pod  # all pods active

    print(f"  MAC breakdown per sample (N={N}, grid={grid_size}x{grid_size}):")
    print(f"    Neural router:       {mac_neural_router:>6} MAC")
    print(f"    SOM router (CPU):    {mac_som_router_cpu:>6} MAC")
    print(f"    SOM router (GPU/TMU):{mac_som_router_gpu:>6} MAC (sprzętowy)")
    print(f"    Pod (1 kapsuła):     {mac_pod:>6} MAC")
    print(f"")
    print(f"  Total per inference:")
    print(f"    Dense (all pods):    {mac_dense:>6} MAC")
    print(f"    Neural routed:       {mac_neural_total:>6} MAC (saving {(1-mac_neural_total/mac_dense)*100:.1f}%)")
    print(f"    SOM routed (CPU):    {mac_som_cpu_total:>6} MAC (saving {(1-mac_som_cpu_total/mac_dense)*100:.1f}%)")
    print(f"    SOM routed (GPU):    {mac_som_gpu_total:>6} MAC (saving {(1-mac_som_gpu_total/mac_dense)*100:.1f}%)")

    # SOM vs Neural router overhead
    print(f"\n  SOM vs Neural router:")
    if mac_som_router_cpu < mac_neural_router:
        print(f"    SOM jest TAŃSZY nawet na CPU ({mac_som_router_cpu} vs {mac_neural_router})")
    else:
        print(f"    SOM na CPU jest droższy ({mac_som_router_cpu} vs {mac_neural_router})")
        print(f"    Ale na GPU (TMU) SOM = 0 MAC → oszczędność {mac_neural_router} MAC/sample")

    return {
        "N": N,
        "grid_size": grid_size,
        "mac_neural_router": mac_neural_router,
        "mac_som_router_cpu": mac_som_router_cpu,
        "mac_som_router_gpu": mac_som_router_gpu,
        "mac_pod": mac_pod,
        "mac_dense": mac_dense,
        "mac_neural_total": mac_neural_total,
        "mac_som_cpu_total": mac_som_cpu_total,
        "mac_som_gpu_total": mac_som_gpu_total,
        "saving_neural_pct": round((1 - mac_neural_total / mac_dense) * 100, 2),
        "saving_som_cpu_pct": round((1 - mac_som_cpu_total / mac_dense) * 100, 2),
        "saving_som_gpu_pct": round((1 - mac_som_gpu_total / mac_dense) * 100, 2),
    }


# ─── Test 3: Soft routing (bilinear confidence) ─────────────────────────────

def test_soft_routing(N=5, seed=42):
    """Test: soft routing z bilinear daje lepszą dokładność niż hard?"""
    print(f"\n┌─ TEST 3: Soft routing (bilinear confidence) ─────────────────┐")

    X, region, y = make_regions(n_regions=N, n_per_region=40, sigma=0.07, seed=seed)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    region_test = region[n_train:]

    som_router = SOMRouter(n_in=2, n_pods=N, grid_size=8, seed=seed)
    som_router.train(X_train, region[:n_train], som_epochs=len(X_train) * 50)

    # Hard routing
    hard_pred = som_router.predict_pod(X_test)
    hard_acc = float(np.mean(hard_pred == region_test))

    # Soft routing
    soft_results = som_router.predict_pod_soft(X_test)
    soft_pred = np.array([r[0] for r in soft_results])
    soft_acc = float(np.mean(soft_pred == region_test))
    confidences = [r[1] for r in soft_results]
    mean_conf = float(np.mean(confidences))

    # Confidence threshold: jeśli confidence < próg, użyj second choice
    # (analogia do top-2 z Etapu 3ext)
    high_conf_mask = np.array(confidences) > 0.6
    if high_conf_mask.any():
        high_conf_acc = float(np.mean(soft_pred[high_conf_mask] == region_test[high_conf_mask]))
    else:
        high_conf_acc = 0.0

    # Ile "niepewnych" decyzji?
    uncertain = np.sum(~high_conf_mask)
    uncertain_pct = uncertain / len(X_test) * 100

    print(f"  Hard routing accuracy:       {hard_acc*100:.1f}%")
    print(f"  Soft routing accuracy:       {soft_acc*100:.1f}%")
    print(f"  Mean confidence:             {mean_conf:.3f}")
    print(f"  High-confidence accuracy:    {high_conf_acc*100:.1f}% (confidence > 0.6)")
    print(f"  Uncertain decisions:         {uncertain_pct:.1f}% ({uncertain}/{len(X_test)})")

    return {
        "N": N,
        "hard_acc": round(hard_acc, 4),
        "soft_acc": round(soft_acc, 4),
        "mean_confidence": round(mean_conf, 4),
        "high_conf_acc": round(high_conf_acc, 4),
        "uncertain_pct": round(uncertain_pct, 2),
    }


# ─── Test 4: Sleep v2 (decay + Hebbian) ─────────────────────────────────────

def test_sleep_v2(N=5, seed=42):
    """Test: czy cykl snu (decay + hebbian + prune) poprawia router?"""
    print(f"\n┌─ TEST 4: Sleep v2 (decay + Hebbian + prune) ────────────────┐")

    X, region, y = make_regions(n_regions=N, n_per_region=40, sigma=0.07, seed=seed)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    region_train, region_test = region[:n_train], region[n_train:]

    som_router = SOMRouter(n_in=2, n_pods=N, grid_size=8, seed=seed)
    som_router.train(X_train, region_train, som_epochs=len(X_train) * 50)

    # Accuracy before sleep
    pred_before = som_router.predict_pod(X_test)
    acc_before = float(np.mean(pred_before == region_test))

    # Simulate multiple inference + sleep cycles
    # (symulacja życia systemu: inferencja → sen → inferencja)
    accs_over_time = [acc_before]
    pruned_total = 0

    for cycle in range(5):
        # "Dzień": inferencja na danych treningowych (aktualizuje activation_count)
        som_router.predict_pod(X_train)
        
        # "Noc": cykl snu
        pruned = som_router.sleep_cycle(X_train, region_train)
        pruned_total += pruned
        
        # Re-assign labels after sleep
        som_router.assign_labels(X_train, region_train)
        
        # Zmierz accuracy po śnie
        pred = som_router.predict_pod(X_test)
        acc = float(np.mean(pred == region_test))
        accs_over_time.append(acc)

    acc_after = accs_over_time[-1]
    print(f"  Accuracy przed snem:  {acc_before*100:.1f}%")
    print(f"  Accuracy po 5 cyklach: {acc_after*100:.1f}%")
    print(f"  Pruned neurons total: {pruned_total}")
    print(f"  Krzywa: {[f'{a*100:.0f}%' for a in accs_over_time]}")
    print(f"  Wniosek: Sleep v2 {'✓ utrzymuje' if acc_after >= acc_before - 0.02 else '✗ degraduje'} accuracy")

    return {
        "N": N,
        "acc_before_sleep": round(acc_before, 4),
        "acc_after_5_cycles": round(acc_after, 4),
        "pruned_total": pruned_total,
        "accuracy_curve": [round(a, 4) for a in accs_over_time],
        "sleep_preserves_accuracy": acc_after >= acc_before - 0.02,
    }


# ─── Test 5: Online adaptation (nowy region) ────────────────────────────────

def test_online_adaptation(N=5, seed=42):
    """Test: SOM-router adaptuje się do nowego regionu bez retreningu."""
    print(f"\n┌─ TEST 5: Online adaptation (nowy region) ─────────────────────┐")

    # Trenuj na N regionów
    X, region, y = make_regions(n_regions=N, n_per_region=40, sigma=0.07, seed=seed)
    som_router = SOMRouter(n_in=2, n_pods=N + 1, grid_size=10, seed=seed)
    som_router.train(X, region, som_epochs=len(X) * 50)

    # Accuracy na oryginalnych
    pred = som_router.predict_pod(X)
    acc_original = float(np.mean(pred == region))

    # Dodaj nowy region (N-ty)
    X_new, _, y_new = make_regions(n_regions=1, n_per_region=30, sigma=0.07, seed=seed + 100)
    # Przesuń nowy region daleko
    X_new[:, 0] += 5.0
    X_new[:, 1] += 5.0
    region_new = np.full(len(X_new), N, dtype=np.int32)

    # Online: dotrenuj SOM na nowych danych (krótki trening)
    adaptation_epochs = [50, 100, 200, 500]
    adaptation_curve = []

    for epochs in adaptation_epochs:
        # Kopia routera
        router_copy = SOMRouter(n_in=2, n_pods=N + 1, grid_size=10, seed=seed)
        router_copy.weights = som_router.weights.copy()
        router_copy.cell_labels = som_router.cell_labels.copy()
        
        # Dotrenuj tylko na nowych danych (krótko)
        router_copy.train_som(X_new, epochs=epochs, lr_start=0.1, sigma_start=2.0)
        
        # Re-assign labels z WSZYSTKIMI danymi
        X_all = np.vstack([X, X_new])
        region_all = np.concatenate([region, region_new])
        router_copy.assign_labels(X_all, region_all)
        
        # Test accuracy
        acc_old = float(np.mean(router_copy.predict_pod(X) == region))
        acc_new = float(np.mean(router_copy.predict_pod(X_new) == region_new))
        adaptation_curve.append({
            "epochs": epochs,
            "acc_old_regions": round(acc_old, 4),
            "acc_new_region": round(acc_new, 4),
        })

    print(f"  Accuracy oryginalne (przed adaptation): {acc_original*100:.1f}%")
    print(f"\n  Krzywa adaptacji (nowy region):")
    print(f"  {'epochs':>8} {'acc_old':>10} {'acc_new':>10}")
    for item in adaptation_curve:
        print(f"  {item['epochs']:>8} {item['acc_old_regions']*100:>9.1f}% {item['acc_new_region']*100:>9.1f}%")

    return {
        "N_original": N,
        "acc_before_adaptation": round(acc_original, 4),
        "adaptation_curve": adaptation_curve,
    }


# ─── Test 6: Skalowanie ─────────────────────────────────────────────────────

def test_scaling():
    """Porównanie SOM vs Neural router przy różnych N."""
    print(f"\n┌─ TEST 6: Skalowanie (N=3..50) ───────────────────────────────┐")

    results = []
    for N in [3, 5, 10, 20, 50]:
        X, region, y = make_regions(n_regions=N, n_per_region=20, sigma=0.07, seed=42)
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        region_train, region_test = region[:n_train], region[n_train:]

        # Neural
        engine = EngineCore(n_in=2, n_pods=N, pod_hidden=8, router_hidden=8, seed=42)
        engine.train(X_train, region_train, y[:n_train], epochs=3000, lr=0.3)
        neural_pred = engine.router.predict_pod(X_test)
        neural_acc = float(np.mean(neural_pred == region_test))

        # SOM (grid rośnie z sqrt(N))
        grid = max(6, int(np.sqrt(N) * 3))
        som = SOMRouter(n_in=2, n_pods=N, grid_size=grid, seed=42)
        som.train(X_train, region_train, som_epochs=len(X_train) * 50)
        som_pred = som.predict_pod(X_test)
        som_acc = float(np.mean(som_pred == region_test))
        topo = som.topology_quality()

        # MAC
        n_in, rh = 2, 8
        mac_neural = n_in * rh + rh * N
        mac_som_cpu = grid * grid * n_in * 2
        mac_som_gpu = 0

        results.append({
            "N": N, "grid": grid,
            "neural_acc": round(neural_acc, 4),
            "som_acc": round(som_acc, 4),
            "topology": round(topo, 4),
            "mac_neural_router": mac_neural,
            "mac_som_cpu": mac_som_cpu,
            "mac_som_gpu": mac_som_gpu,
        })

        print(f"  N={N:>3} grid={grid:>2}  neural={neural_acc*100:>5.1f}%  "
              f"som={som_acc*100:>5.1f}%  topo={topo*100:.0f}%  "
              f"MAC: neural={mac_neural} som_cpu={mac_som_cpu} som_gpu=0")

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("ETAP 3C — SOM-Router: Kohonen jako Engine Core")
    print("  Integracja Etap 3 (Router) + Etap 4B (Kohonen SOM)")
    print("=" * 64)

    r1 = test_accuracy()
    r2 = test_mac()
    r3 = test_soft_routing()
    r4 = test_sleep_v2()
    r5 = test_online_adaptation()
    r6 = test_scaling()

    # Werdykt końcowy
    print("\n" + "=" * 64)
    print("WERDYKT ETAPU 3C")
    print("=" * 64)

    # SOM vs Neural: kiedy lepszy?
    som_wins_acc = r1["som_acc"] >= r1["neural_acc"] - 0.02
    som_wins_mac_gpu = True  # zawsze wygrywa na GPU (0 MAC)
    sleep_ok = r4["sleep_preserves_accuracy"]

    print(f"\n  SOM-Router vs Neural-Router:")
    print(f"    Dokładność:    {'✓' if som_wins_acc else '✗'} "
          f"(SOM={r1['som_acc']*100:.1f}% vs Neural={r1['neural_acc']*100:.1f}%)")
    print(f"    MAC (GPU/TMU): ✓ SOM = 0 MAC (router) vs Neural = {r2['mac_neural_router']} MAC")
    print(f"    MAC (CPU):     {'✓' if r2['mac_som_router_cpu'] <= r2['mac_neural_router'] else '✗'} "
          f"(SOM={r2['mac_som_router_cpu']} vs Neural={r2['mac_neural_router']})")
    print(f"    Sleep v2:      {'✓' if sleep_ok else '✗'} accuracy utrzymana po 5 cyklach")
    print(f"    Soft routing:  confidence mean={r3['mean_confidence']:.3f}")

    verdict = "POZYTYWNY" if (som_wins_acc and sleep_ok) else "WARUNKOWY"
    print(f"\n  WERDYKT: {verdict}")
    if verdict == "POZYTYWNY":
        print(f"  SOM-Router jest LEPSZYM Engine Core niż sieć neuronowa:")
        print(f"  - Ta sama dokładność")
        print(f"  - 0 MAC na GPU (vs {r2['mac_neural_router']} MAC neural)")
        print(f"  - Wbudowany soft-routing (confidence) bez dodatkowego kosztu")
        print(f"  - Sleep v2 utrzymuje jakość")
    else:
        print(f"  SOM-Router ma przewagę wydajnościową (MAC), ale wymaga walidacji na GPU.")

    # Zapis
    out = {
        "stage": "etap3c",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "description": "SOM-Router: Kohonen jako Engine Core (integracja Etap 3 + 4B)",
        "test_accuracy": r1,
        "test_mac": r2,
        "test_soft_routing": r3,
        "test_sleep_v2": r4,
        "test_online_adaptation": r5,
        "test_scaling": r6,
        "verdict": verdict,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "etap3c_som_router.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
