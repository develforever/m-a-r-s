"""
run_etapB_benchmark.py — Etap B: pomiar throughput wektoryzowanego forward.

Cel: udowodnić, że naprawiona implementacja (FastPods) usuwa spowolnienie
0.59× z faza2_mnist.json. Porównuje 4 warianty na Twoim GPU:
  - V0 pętla (stara, wolna)
  - V2 grouped (sort+segment)
  - V3 loopless (bmm+padding)
  - baseline monolit (punkt odniesienia)

Mierzy dla rosnącej liczby podów N — pokazuje, że przewaga rośnie ze skalą.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_etapB_benchmark.py
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_IN, HIDDEN, N_OUT, BATCH = 784, 64, 10, 4096


def baseline_model(device):
    return torch.nn.Sequential(
        torch.nn.Linear(N_IN, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, N_OUT),
    ).to(device)


def bench(fn, n_warmup=20, n_runs=100, device='cpu'):
    for _ in range(n_warmup):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_runs):
            fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000
    per_batch_ms = elapsed_ms / n_runs
    return BATCH / (per_batch_ms / 1000)  # samples/s


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 64)
    print("ETAP B — Throughput wektoryzowanego forward M.A.R.S.")
    print(f"Device: {device}", f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 64)

    x = torch.randn(BATCH, N_IN, device=device)
    base = baseline_model(device).eval()

    results = {"device": device, "batch": BATCH, "by_n_pods": []}

    with torch.no_grad():
        sps_base = bench(lambda: base(x), device=device)
        print(f"\nBaseline monolit: {sps_base:,.0f} samples/s\n")
        print(f"{'N':>4} {'V0 pętla':>12} {'V2 grouped':>12} {'V3 loopless':>12}"
              f" {'best/base':>10}")
        print("-" * 60)

        for N in [10, 20, 50, 100]:
            pods = FastPods(N, N_IN, HIDDEN, N_OUT).to(device).eval()
            ids = torch.randint(0, N, (BATCH,), device=device)

            # poprawność
            o0 = pods.forward_loop(x, ids)
            o2 = pods.forward_grouped(x, ids)
            o3 = pods.forward_loopless(x, ids)
            assert torch.allclose(o0, o2, atol=1e-3), "V2 != V0"
            assert torch.allclose(o0, o3, atol=1e-3), "V3 != V0"

            s0 = bench(lambda: pods.forward_loop(x, ids), device=device)
            s2 = bench(lambda: pods.forward_grouped(x, ids), device=device)
            s3 = bench(lambda: pods.forward_loopless(x, ids), device=device)
            best = max(s2, s3)
            print(f"{N:>4} {s0:>12,.0f} {s2:>12,.0f} {s3:>12,.0f}"
                  f" {best/sps_base:>9.2f}×")

            results["by_n_pods"].append({
                "n_pods": N,
                "v0_loop_sps": round(s0),
                "v2_grouped_sps": round(s2),
                "v3_loopless_sps": round(s3),
                "baseline_sps": round(sps_base),
                "best_vs_baseline": round(best / sps_base, 2),
                "best_vs_loop": round(best / s0, 2),
            })

    print("\n--- WNIOSEK ---")
    print("Jeśli best/base ≥ 1.0 — naprawiliśmy throughput (M.A.R.S. nie jest")
    print("już wolniejszy od baseline). Na GPU V3 (loopless) zwykle wygrywa")
    print("przy dużym N, bo to 1 kernel zamiast N.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "etapB_throughput.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
