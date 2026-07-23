"""Audyt A2 (PLAN_V1.md): krzyzowa weryfikacja liczb headline vs results/*.json.

Skrypt jednorazowy, nie dotyka kodu eksperymentow. Przechodzi pliki wynikowe
serii F-O, agreguje per-seed ACC/forgetting (mean +/- std, n=5) i porownuje
z liczbami zadeklarowanymi w README/WHITEPAPER (tabela CLAIMS ponizej).

Uruchomienie: python scripts/audit_headline_numbers.py  (z katalogu repo)
Wyjscie: tabela zgodnosci; kod wyjscia 1 przy jakiejkolwiek rozbieznosci > 0.005pp.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable, Optional

RESULTS = Path(__file__).resolve().parent.parent / "results"

# (etykieta, plik, sciezka do per_seed, oczekiwane_ACC_pct, oczekiwane_std_pct)
Claim = tuple[str, str, str, float, Optional[float]]

CLAIMS: list[Claim] = [
    ("Fashion finetune", "F0_cl_baselines.json", "datasets/Fashion-MNIST/methods/finetune/per_seed", 17.96, 4.47),
    ("Fashion replay-200", "F0_cl_baselines.json", "datasets/Fashion-MNIST/methods/replay/per_seed", 76.97, 1.09),
    ("Fashion MARS k16 (H1b)", "H1b_dream_fidelity.json", "datasets/Fashion-MNIST/variants/k16/per_seed", 77.57, 1.02),
    ("Fashion sparse k16 (J3)", "J3_sparse_dreams.json", "datasets/Fashion-MNIST/variants/sparse_k16/per_seed", 78.49, 0.91),
    ("Fashion sparse x 300d (K1)", "K1_sparse300.json", "systems/fashion_sp16_300/per_seed", 79.23, 0.73),
    ("Fashion sufit g1_all 50d", "J4_glove300.json", "datasets/Fashion-MNIST/variants/all_50/per_seed", 80.45, 0.86),
    ("Fashion sufit g1_all 300d", "J4_glove300.json", "datasets/Fashion-MNIST/variants/all_300/per_seed", 81.16, 0.87),
    ("CIFAR replay-200 (norm)", "J2_cifar_normalized.json", "systems/replay/per_seed", 14.03, 4.93),
    ("CIFAR replay-200 (raw, F4)", "F4_split_cifar.json", "systems/replay/per_seed", 18.90, 8.80),
    ("CIFAR MARS diag k16 (J2)", "J2_cifar_normalized.json", "systems/mars_k16_raw/per_seed", 33.03, 1.16),
    ("CIFAR sparse k16 (J2b)", "J2b_cifar_sparse.json", "systems/sparse_k16/per_seed", 37.51, 1.35),
    ("CIFAR joint (norm)", "J2_cifar_normalized.json", "systems/joint/per_seed", 70.24, 0.69),
    ("CIFAR sufit K0 300d", "K0_cifar_ceiling.json", "systems/all_300/per_seed", 39.65, 1.21),
    ("L1 seq pretrained", "L1_pretrained.json", "systems/l1_seq/per_seed", 74.69, 0.69),
    ("L1 sufit all-data", "L1_pretrained.json", "systems/l1_all/per_seed", 77.23, 0.57),
    ("M1 seq 300d (100 klas)", "M1_long_horizon.json", "systems/m1_seq_300/per_seed", 40.70, 0.84),
    ("M1 sufit all 300d", "M1_long_horizon.json", "systems/m1_all_300/per_seed", 47.41, 0.49),
    ("M1 seq 50d (kotwice)", "M1_long_horizon.json", "systems/m1_seq_50/per_seed", 32.99, 0.81),
]

# Twierdzenia o deltach/werdyktach czytane wprost z pol `verdicts` (bez przeliczen).
VERDICT_CHECKS: list[tuple[str, str, str, str]] = [
    ("I3 kolektyw vs seq = SZUM", "I3_collective.json", "verdicts/collective_vs_seq/verdict", "SZUM"),
    ("L2 koszt protokolu = parowy-", "L2_collective_cifar.json", "verdicts/collective_vs_seq/verdict", "SYGNAL-parowy-"),
    ("N1c pelna gwarancja", "N1c_reinit.json", "verdicts/reinit_vs_never/verdict", "PELNA GWARANCJA WYMAZANIA"),
    ("I4b pelna naprawa zasiegowa", "I4b_full_repair.json", "verdicts/PELNA_NAPRAWA_ZASIEGOWA", "True"),
]


def dig(obj: Any, path: str) -> Any:
    for part in path.split("/"):
        obj = obj[part]
    return obj


def acc_stats(per_seed: list[dict[str, Any]]) -> tuple[float, float, int]:
    accs: list[float] = []
    for entry in per_seed:
        if "class_il" in entry:
            accs.append(float(entry["class_il"]["ACC"]))
        elif "ACC" in entry:
            accs.append(float(entry["ACC"]))
        else:
            raise KeyError(f"brak ACC w per_seed: {list(entry)}")
    return st.mean(accs) * 100, (st.stdev(accs) * 100 if len(accs) > 1 else 0.0), len(accs)


def main() -> int:
    failures = 0
    print(f"{'twierdzenie':38s} {'plik':28s} {'JSON':>14s} {'deklaracja':>12s}  status")
    for label, fname, path, exp_mean, exp_std in CLAIMS:
        try:
            data = json.loads((RESULTS / fname).read_text())
            mean, std, n = acc_stats(dig(data, path))
        except (OSError, KeyError, json.JSONDecodeError) as err:
            print(f"{label:38s} {fname:28s} BLAD ODCZYTU: {err}")
            failures += 1
            continue
        ok = abs(mean - exp_mean) <= 0.005 and (exp_std is None or abs(std - exp_std) <= 0.005)
        failures += 0 if ok else 1
        print(f"{label:38s} {fname:28s} {mean:6.2f}+-{std:4.2f} {exp_mean:6.2f}+-{exp_std:4.2f}  {'OK' if ok else 'ROZBIEZNOSC'} (n={n})")

    print()
    for label, fname, path, expected in VERDICT_CHECKS:
        try:
            data = json.loads((RESULTS / fname).read_text())
            actual = str(dig(data, path))
        except (OSError, KeyError, json.JSONDecodeError) as err:
            print(f"{label:44s} BLAD ODCZYTU: {err}")
            failures += 1
            continue
        ok = actual == expected
        failures += 0 if ok else 1
        print(f"{label:44s} JSON='{actual}'  {'OK' if ok else 'ROZBIEZNOSC (oczekiwano ' + expected + ')'}")

    print(f"\nWynik audytu: {'ZERO ROZBIEZNOSCI' if failures == 0 else f'{failures} ROZBIEZNOSCI'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
