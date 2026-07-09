"""
metrics.py — pomiar kosztu obliczeniowego i energetycznego.

To jest serce Etapu 0. Bez wspólnej, obiektywnej metryki zdanie
"M.A.R.S. zużywa 95% mniej energii" jest nieweryfikowalne.

Mierzymy DWIE rzeczy, niezależnie:

1. MAC (Multiply-Accumulate operations) — licznik operacji mnożenia-
   akumulacji. To metryka SPRZĘTOWO-NIEZALEŻNA: liczy ile mnożeń
   wykonał algorytm, niezależnie od tego, jak szybki masz procesor.
   To najuczciwszy sposób porównania backpropagation vs. uczenie lokalne,
   bo właśnie mnożenia zmiennoprzecinkowe są głównym kosztem energii w AI.

2. Czas i (przybliżona) energia CPU — mierzona przez czas wykonania.
   UWAGA: dokładny pomiar dżuli wymaga RAPL (Linux/Intel) lub
   sprzętowego watomierza. Na Windowsie bez specjalnego sprzętu mamy
   tylko czas i obciążenie CPU jako PROXY energii — i tak to oznaczamy,
   żeby nie wprowadzać w błąd.

Zasada uczciwości: nie udajemy, że mierzymy dżule, jeśli mierzymy czas.
"""

import time
import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class MACCounter:
    """
    Globalny licznik operacji mnożenia-akumulacji (MAC).

    Wywołuj .add_matmul(a_shape, b_shape) przy każdym mnożeniu macierzy,
    żeby zliczyć realny koszt obliczeniowy algorytmu — niezależnie
    od prędkości sprzętu.
    """

    def __init__(self):
        self.mac = 0

    def reset(self):
        self.mac = 0

    def add_matmul(self, a_shape, b_shape):
        """
        Dolicza MAC-i dla mnożenia macierzy A(m,k) x B(k,n).
        Liczba operacji mnożenia-akumulacji = m * k * n.
        """
        m, k = a_shape
        k2, n = b_shape
        assert k == k2, f"Niezgodne wymiary do mnożenia: {a_shape} x {b_shape}"
        self.mac += m * k * n

    def add(self, n_ops):
        """Ręcznie dolicz n operacji (np. dla aktywacji)."""
        self.mac += int(n_ops)


class EnergyTimer:
    """
    Mierzy czas wykonania i — jeśli dostępne — obciążenie CPU jako
    PROXY zużycia energii. Nie udaje pomiaru dżuli.

    Użycie:
        with EnergyTimer() as t:
            ... obliczenia ...
        print(t.report())
    """

    def __init__(self):
        self.wall_time = None
        self.cpu_time = None
        self._t0 = None
        self._c0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, *args):
        self.wall_time = time.perf_counter() - self._t0
        self.cpu_time = time.process_time() - self._c0

    def report(self):
        return {
            "wall_time_s": self.wall_time,
            "cpu_time_s": self.cpu_time,
            "psutil_available": _HAS_PSUTIL,
            "note": "cpu_time to PROXY energii, nie dzule. "
                    "Realny pomiar wymaga RAPL/watomierza.",
        }


def summarize(name, mac_counter, energy_timer, accuracy, extra=None):
    """
    Składa jeden spójny rekord wyniku do zapisu/porównania.
    """
    record = {
        "name": name,
        "mac_operations": mac_counter.mac,
        "accuracy": accuracy,
    }
    record.update(energy_timer.report())
    if extra:
        record.update(extra)
    return record
