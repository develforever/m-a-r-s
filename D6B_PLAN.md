# D6b — Odchudzenie CNN backbone (plan eksperymentu)

Data: 2026-07-06
Status: DO URUCHOMIENIA (decyzja użytkownika: D6b przed D7)
Kontekst: D6 dał SYGNAL+ (+2.38pp Fashion), ale za ~19.7× MAC (215.6k → 4.25M).
Pytanie D6b: **ile z tego zysku przetrwa przy MAC bliskim MLP?**

---

## 1. Hipoteza

> Zysk D6 pochodzi z lokalności/inwariancji konwolucji (lepsze CECHY dla routera),
> nie z surowego compute. Zatem znaczna część +2.38pp (Fashion) przetrwa
> po redukcji MAC do ≤2× MLP.

Falsyfikowalna: jeśli przy MAC ≤ 2× MLP delta spada do szumu, to D6 = "więcej
compute = więcej accuracy", nie dźwignia efektywnościowa. To też jest wynik.

## 2. Warianty (sweep, wspólny trzon mars_v2_cnn.py)

MAC top-1 poniżej to SZACUNKI ręczne — skrypt liczy dokładnie (jak w D6).

| Wariant | Architektura | MAC bb (szac.) | ×MLP (215.6k) |
|---|---|---|---|
| bazowy MLP | D1c, bb_h=256 | 200.7k | 1.0× |
| bazowy CNN | D6, ch=(32,64) | 4 240k | 19.7× |
| S1 half | ch=(16,32), bb_h=128 | ~1 217k | ~5.6× |
| S2 quarter | ch=(8,16), bb_h=128 | ~383k | ~1.8× |
| S3 stride | ch=(16,32), stride=2 zamiast MaxPool | ~455k | ~2.1× |
| S4 depthwise | ch=(16,32), depthwise-separable conv2 | ~340k | ~1.6× |

Kandydaci "efektywnościowi": S2/S3/S4 (MAC ~1.6–2.1× MLP).

## 3. Setup (identyczny z D6 — porównanie czyste)

- 5 seedów × 30 epok, GPU, train_phased reużyty, MNIST + Fashion-MNIST.
- Baseline'y: MLP (D1c/D6) i pełny CNN (D6) — te same seedy.
- Metryki per seed: routing_acc, system_acc, oracle_acc, MAC (backbone/routing/pod).
- Wyjście: krzywa Pareto Accuracy vs MAC (jak D1b) + `results/D6b_slim_cnn.json`.

## 4. Kryterium werdyktu (z góry)

Na Fashion, dla najlepszego wariantu z MAC ≤ 2.2× MLP, Δ vs MLP:

| Wynik | Werdykt |
|---|---|
| Δ ≥ +1.0pp i > próg szumu (std_MLP + std_wariantu) | **SYGNAL+ efektywnościowy** — CNN to realna dźwignia; ten backbone idzie do D7. |
| 0 < Δ ≤ próg szumu | **SZUM** — zysk D6 kupiony głównie compute; do D7 bierzemy pełny CNN, wątek "rewolucji" wraca do deski. |
| Δ < 0 | **SYGNAL−** — odchudzony CNN gorszy od MLP; lokalna konwolucja bez szerokości nie wystarcza. |

Dodatkowo raportować "retention": Δ_wariant / Δ_D6 (ile z +2.38pp przetrwało).

## 5. Ryzyka

1. **Wąskie conv1 (8 kanałów) może ubić routing** — routing był głównym
   beneficjentem D6; sprawdzić routing_acc osobno, nie tylko system.
2. **Stride vs MaxPool zmienia dwie rzeczy naraz** (downsampling + brak max);
   S3 trzymać przy ch=(16,32), żeby porównywać z S1, nie z S2.
3. **bb_h=128 wspólne dla wszystkich wariantów** — nie mieszać redukcji
   kanałów z redukcją głowicy.

## 6. Kolejność wykonania

1. `src/run_D6b_slim_cnn.py` — warianty S1–S4 jako parametryzacja
   mars_v2_cnn.py (kanały, stride, depthwise), dokładny licznik MAC.
2. Smoke 1 seed (sanity: MAC zgodny z szacunkami, acc > MLP na S1).
3. Pełny run 5 seedów × 4 warianty × 2 zbiory.
4. Wynik → `DROGA_D_NOTATKI.md` + decyzja backbone'u dla D7.
