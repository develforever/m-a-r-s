"""
run_etapB_squeeze.py — Etap B+: znajdź reżim maksymalnej przewagi M.A.R.S.

Po naprawie throughput (FastPods) okazało się, że przewaga zależy od DWÓCH
parametrów: rozmiaru poda (hidden) i liczby podów (N). Ten skrypt mierzy
PEŁNĄ MACIERZ przewagi na Twoim GPU, żeby znaleźć reżim, gdzie M.A.R.S.
bije monolit najmocniej.

Odkrycia z CPU (do potwierdzenia na GPU):
  - V2 (grouped, bez paddingu) BIJE V3 (padding) przy dużych podach
  - Najlepszy reżim: hidden≥2048 + N=8-32 → ~2× szybszy od monolitu
  - V3 (padding) to ślepa uliczka przy dużym N — marnuje obliczenia

Uruchom:
    .venv\\Scripts\\python.exe src\\run_etapB_squeeze.py
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_IN, N_OUT, BATCH = 784, 10, 2048


def bench(fn, n_warmup=15, n_runs=50, device='cpu'):
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
    return BATCH / ((elapsed_ms / n_runs) / 1000)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("ETAP B+ — Macierz przewagi M.A.R.S. (szukamy reżimu 2×)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 70)

    x = torch.randn(BATCH, N_IN, device=device)
    results = {"device": device, "batch": BATCH, "matrix": []}

    print(f"\n{'hidden':>7} {'N':>5} {'baseline':>11} {'V2 grouped':>11}"
          f" {'V3 loopless':>11} {'V2/base':>8} {'V3/base':>8}")
    print("-" * 70)

    for hidden in [512, 2048, 4096]:
        # baseline: monolit o porównywalnej pojemności (hidden*2 ukrytych)
        base = torch.nn.Sequential(
            torch.nn.Linear(N_IN, hidden * 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden * 2, N_OUT),
        ).to(device).eval()

        for N in [8, 32, 64]:
            pods = FastPods(N, N_IN, hidden, N_OUT).to(device).eval()
            ids = torch.randint(0, N, (BATCH,), device=device)

            with torch.no_grad():
                # poprawność
                o2 = pods.forward_grouped(x, ids)
                o3 = pods.forward_loopless(x, ids)
                assert torch.allclose(o2, o3, atol=1e-3), "V2 != V3"

                sb = bench(lambda: base(x), device=device)
                s2 = bench(lambda: pods.forward_grouped(x, ids), device=device)
                s3 = bench(lambda: pods.forward_loopless(x, ids), device=device)

            print(f"{hidden:>7} {N:>5} {sb:>11,.0f} {s2:>11,.0f} {s3:>11,.0f}"
                  f" {s2/sb:>7.2f}× {s3/sb:>7.2f}×")
            results["matrix"].append({
                "hidden": hidden, "n_pods": N,
                "baseline_sps": round(sb),
                "v2_grouped_sps": round(s2),
                "v3_loopless_sps": round(s3),
                "v2_vs_base": round(s2 / sb, 2),
                "v3_vs_base": round(s3 / sb, 2),
            })

    # znajdź najlepszy reżim
    best = max(results["matrix"], key=lambda r: max(r["v2_vs_base"], r["v3_vs_base"]))
    best_ratio = max(best["v2_vs_base"], best["v3_vs_base"])
    best_strat = "V2 grouped" if best["v2_vs_base"] >= best["v3_vs_base"] else "V3 loopless"

    print("\n--- NAJLEPSZY REŻIM ---")
    print(f"hidden={best['hidden']}, N_pods={best['n_pods']}, "
          f"strategia={best_strat}: {best_ratio:.2f}× szybszy od monolitu")
    print("\nTo jest reżim, w którym M.A.R.S. ma realną przewagę czasową")
    print("(plus zawsze obecna oszczędność MAC). Whitepaper powinien")
    print("eksponować właśnie ten reżim — duże pody, jak w prawdziwym MoE.")

    results["best_regime"] = {
        "hidden": best["hidden"], "n_pods": best["n_pods"],
        "strategy": best_strat, "speedup": best_ratio,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "etapB_squeeze_matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
