# Droga H — notatki robocze

Plan: `DROGA_H_PLAN.md` (weryfikacja propozycji zewn. agenta: OWM / VSA / Morton).

## H1 — OWM na projekcji semantycznej (ZAKOŃCZONE, 08.07.2026):
## SZUM na Fashion, SYGNAL+ na MNIST — i kluczowa ELIMINACJA

Pliki: `src/mars_cl_owm.py`, `src/run_H1_owm.py`; wyniki: `results/H1_owm.json`.
5 seedów × 15 epok; rekurencja RLS (bez inwersji), bias frozen po task0,
P aktualizowane cechami zadania w trakcie (zero przechowywania).

| Wariant (class-IL) | Fashion ACC | MNIST ACC | Forgetting F/M |
|---|---|---|---|
| owm bez snu (a1) | 42.89 ± 9.65% | 38.19 ± 11.36% | 66.1 / 70.4pp |
| **owm+sen (a1)** | **76.20 ± 1.63%** | 66.09 ± 3.51% | 20.2 / 35.8pp |
| owm+sen (a01) | 73.56 ± 3.91% | 59.55 ± 3.16% | 19.2 / 40.0pp |
| owm+sen (a10) | 76.12 ± 1.41% | **68.56 ± 2.25%** | 21.6 / 33.8pp |
| [F3b] combo (ref) | 75.68 ± 1.17% | 63.56 ± 1.88% | 15.8 / 33.5pp |
| [F0] replay-200 | 76.97 ± 1.09% | 88.81 ± 1.06% | 27.0 / 13.5pp |

**WERDYKTY:** Fashion: SZUM vs F3b (+0.52pp, próg 2.80; vs replay −0.77pp
— dalej równoważność). MNIST: SYGNAL+ (+5.00pp > próg 4.12) — OWM pomaga
tam, gdzie kotwice słów są słabe (dryf robił większą różnicę).

**Ustalenia:**
1. **ELIMINACJA (najważniejsze):** OWM daje ścisłą gwarancję niezmienności
   starych mapowań — a wynik Fashion nie drgnął. Wniosek: resztkowa luka
   do sufitu g1_all (76.2 → 80.45) **NIE jest dryfem** (F3b go domknął).
   Pozostały deficyt = WIERNOŚĆ SNU: rozjazd wyśnionych Gaussianów
   z realnym rozkładem cech determinuje, gdzie lądują nowe słowa względem
   starych klas. Następna dźwignia (jeśli wracać): bogatszy model snu
   (pełna kowariancja / więcej centroidów / GMM) albo lepsze zamrożone
   cechy — nie kolejna ochrona wag.
2. **Geometria ≠ decyzja (5 seedów):** OWM bez snu = 42.9 ± 9.7% — rzutnik
   chroni mapowanie starych cech, ale nie chroni granic decyzyjnych, gdy
   nowe prototypy wchodzą w zajęte rejony. Sen strzeże decyzji, OWM
   geometrii; sam OWM jest niewystarczający i niestabilny.
3. **Plastyczność OK przy T=5:** R[t][t] ~94–96% na końcu sekwencji;
   kurczenie null-space nie ugryzło (przy 128 wymiarach i 5 zadaniach).
4. **Lekcja metodyczna:** pierwszy przypadek w projekcie, gdzie smoke
   (77.27, 1 seed) obiecał więcej niż pełny run (76.20 ± 1.63) — odczyty
   1-seedowe bywają optymistyczne w OBIE strony.

**Stan tezy po H1:** Fashion class-IL ustabilizowany na ~76% ≈ replay
(równoważność potwierdzona trzecim niezależnym mechanizmem), sufit 80.45.

## H1b — Wierność snu (ZAKOŃCZONE, 08.07.2026): NAJLEPSZY WYNIK SERII
## + negatyw o pełnej kowariancji

Pliki: `src/run_H1b_dream_fidelity.py` (+ `FeatureStatsFullCovK`
w mars_cl_f3.py); wyniki: `results/H1b_dream_fidelity.json`. 5 seedów.

| Wariant (Fashion, class-IL) | ACC | min | F | Pamięć/klasę |
|---|---|---|---|---|
| **k16 (16 centroidów diag)** | **77.57 ± 1.02%** | **76.59%** | **18.8pp** | ~16 KB |
| k8 | 76.95 ± 1.18% | 75.76% | 20.1pp | ~8 KB |
| full1 (pełna kowariancja) | 73.87 ± 1.18% | 72.16% | 28.0pp | ~66 KB |
| full4 | 74.70 ± 1.26% | 72.71% | 26.9pp | ~262 KB |
| [F0] replay-200 | 76.97 ± 1.09% | 75.23% | 27.0pp | 200 obrazów |

**WERDYKT (k16 vs replay): RÓWNOWAŻNOŚĆ** — pierwszy wariant ze średnią
NAD replay (+0.60pp), ale min per-seed −0.71 przy progu 2.11 → uczciwie
NIE ogłaszamy SYGNAL++. MNIST: k16 70.21 (+2.29 nad k4), dalej pod replay.

**Ustalenia:**
1. **Wierność snu potwierdzona jako dźwignia** (przewidziane przez
   eliminację z H1): k4→k16 = +1.79pp; drabina snu: 1 Gaussian 70.8 →
   k4 75.8 → k16 77.6, sufit 80.45. Malejące przyrosty — mechanizm
   wyżyłowany.
2. **NEGATYW: pełna kowariancja < lokalne centroidy** (73.9 vs 77.6 przy
   4–16× większej pamięci). Cechy po ReLU są rzadkie, nieujemne,
   wielomodalne — globalny Gaussian to zły model gęstości; sampling +
   clamp schodzi z rozmaitości danych. LOKALNOŚĆ > struktura kowariancji.
   Wskazówka projektowa dla feature-replay w ogóle.
3. **KOD ZAMROŻONY.** Seria mechanizmowa F/G/H zakończona. Wynik główny
   do publikacji: 77.6 ± 1.0% class-IL Split-Fashion bez jednego
   przechowanego obrazu ≈ replay-200 (statystyczna równoważność, nominalnie
   wyżej), forgetting −30%, MAC stały. Dalszy wzrost = lepsza zamrożona
   reprezentacja (decyzja o piętrze projektu, po publikacji).
