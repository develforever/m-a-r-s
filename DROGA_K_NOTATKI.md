# Droga K — notatki robocze

Plan: `DROGA_K_PLAN.md` (pre-rejestracja 2026-07-17, commit 51e1b58).
Runy: Robert, lokalnie (GTX 1050 Ti), 5 seedów, epochs=15, LR=0.001.
Werdykty zweryfikowane niezależnie od printów runnerów (przeliczenie
z JSON-ów): wszystkie zgodne.

## K0 — sufit zamrożonych cech CIFAR (ZAKOŃCZONE, 17.07.2026):
## gap_mech = 2.14pp → MECHANIZM NA CIFAR PRAKTYCZNIE DOMKNIĘTY

Plik: `src/run_K0_cifar_ceiling.py`; wyniki:
`results/K0_cifar_ceiling.json`. Czas: 123 s.

| Wariant (CIFAR-n, class-IL) | ACC | min | F |
|---|---|---|---|
| all_50 | 39.50 ± 1.21% | 38.12% | 19.8pp |
| **all_300** | **39.65 ± 1.21%** | 38.16% | 19.7pp |

**DIAGNOZA (pre-rejestrowana):** gap_mech = 39.65 − 37.51 = **+2.14pp**
(< 3pp) → mechanizm na CIFAR praktycznie domknięty; gap_repr = 70.24 −
39.65 = **30.59pp** → cała pozostała luka jest reprezentacyjna (Etap L).

**Ustalenia:**
1. **Mechanizm realizuje 94.6% sufitu swojej reprezentacji na CIFAR**
   (37.51 / 39.65). Uczciwy CL sekwencyjny traci do "projekcja widzi
   wszystko" tylko 2.14pp — sen sparse niemal domknął protokół.
2. **300d NIE podnosi sufitu CIFAR** (+0.15, głęboko w szumie) —
   inaczej niż Fashion (+0.72, 5/5 par w J4). Bogatsza geometria słów
   nie pomaga, gdy wąskim gardłem są cechy obrazu, nie kotwice.
3. Konsekwencja strategiczna: na CIFAR nie ma już dźwigni mechanizmowych
   do wzięcia — droga w górę wyłącznie przez mocniejszy zamrożony
   backbone (PLAN_GENERALNY, Etap L).

## K1 — sen sparse_k16 × GloVe 300d (ZAKOŃCZONE, 17.07.2026):
## SYGNAL-parowy+ na Fashion — NOWY BEST 79.23; SZUM na CIFAR

Plik: `src/run_K1_sparse300.py`; wyniki: `results/K1_sparse300.json`.
Czas: 98 s.

| Wariant (class-IL) | ACC | min | F | Werdykt vs sparse_k16×50d |
|---|---|---|---|---|
| [J3] fashion sp16×50 (baza) | 78.49 ± 0.91% | 77.72% | 16.0pp | — |
| **fashion_sp16_300** | **79.23 ± 0.73%** | **78.38%** | **15.5pp** | **SYGNAL-parowy+** |
| [J2b] cifar sp16×50 (baza) | 37.51 ± 1.35% | 36.20% | 32.7pp | — |
| cifar_sp16_300 | 38.00 ± 1.32% | 36.37% | 31.7pp | SZUM |

**WERDYKTY (pre-rejestrowane):**
- Fashion: klasyczny SZUM (+0.74 przy progu 1.64), ale pary 5/5 dodatnie
  (+0.38…+1.22) ORAZ śr. 0.74 > 2×std(delt) 0.69 → **SYGNAL-parowy+**
  (pierwsze użycie nowego kryterium).
- CIFAR: pary 5/5 dodatnie (+0.17…+0.87), ale śr. 0.49 < 2×std(delt)
  0.60 → uczciwie **SZUM** (kryterium parowe odmówiło — działa w obie
  strony).

**Ustalenia:**
1. **Złożenie dźwigni działa na Fashion:** J4 (diag×300d) był czystym
   nullem; sparse×300d daje systematyczny zysk — wierniejszy sen
   pozwala projekcji 128→300 skonsumować geometrię, której diag nie
   konsumował. Dokładnie hipoteza K1.
2. **Nowy nominalny best Fashion class-IL, 0 próbek: 79.23 ± 0.73
   (min 78.38)** — luka do sufitu all_300 (81.16) zmniejszona
   z 2.67 do **1.93pp (97.6% sufitu)**; najgorszy seed NAD średnią
   replay-200 (76.97) o +1.4pp.
3. CIFAR spójny z K0: sufit nie urósł z 300d, więc seq nie miał czego
   konsumować — kierunkowo dodatnio, formalnie nic.
4. Domyślna konfiguracja dla Drogi I (Fashion): **sparse_k16 × 300d**.

## K2 — OWM × sen sparse (ZAKOŃCZONE, 17.07.2026): ELIMINACJA OWM
## DOMKNIĘTA — SZUM na MNIST (przewaga z H1 znikła), parowy− na CIFAR

Pliki: `src/mars_cl_k.py`, `src/run_K2_owm_sparse.py`; wyniki:
`results/K2_owm_sparse.json`. Czas: 291 s.

| Wariant (class-IL) | ACC | F | R[t][t] | Werdykt vs sparse_k16 |
|---|---|---|---|---|
| mnist_owm_a10 (GŁÓWNY) | 72.02 ± 1.84% | 28.8pp | 95.1% | SZUM (d=−1.24, pary 5/5 ujemne) |
| mnist_owm_a1 | 69.47 ± 2.88% | 30.9pp | 94.2% | [obs] SZUM (d=−3.80) |
| cifar_owm_a10 (GŁÓWNY) | 36.73 ± 1.57% | 35.4pp | 65.0% | **SYGNAL-parowy−** (d=−0.78) |
| cifar_owm_a1 | 36.77 ± 1.41% | 35.8pp | 65.4% | [obs] SYGNAL-parowy− (d=−0.74) |
| fashion_owm_a1 (kontrola) | 78.04 ± 1.24% | 16.4pp | 91.1% | SZUM (pary mieszane) |

**Ustalenia:**
1. **SYGNAL+ OWM z H1 (MNIST, +5.0pp przy śnie diag) NIE przenosi się
   na sen sparse** — teraz 5/5 par ujemnych (−1.24, formalnie SZUM).
   Interpretacja mechanizmu: OWM chronił geometrię, którą słabszy sen
   psuł; wierny sen załatwia to sam, a rzutnik zostaje z samym kosztem
   (usztywnienie). To domyka eliminację z H1 z drugiej strony.
2. **Na CIFAR OWM systematycznie lekko szkodzi** (parowy− przy obu
   alfach). Kolejne potwierdzenie prawa projektowego M.A.R.S.: twarde
   ograniczenia nałożone na dobrze działający mechanizm częściej
   niszczą informatywność, niż pomagają (por. sigma-norm J1, pełna
   kowariancja H1b).
3. **Kontrola Fashion zgodna z pre-rejestrowanym oczekiwaniem** (SZUM,
   pary mieszane — czysty null): eliminacja H1 potwierdzona przy nowym
   śnie.
4. **Plastyczność bez zarzutu** (R[t][t] 94–95% MNIST, 91% Fashion,
   65% CIFAR ≈ poziomy bez OWM) — problemem OWM nie jest kurczenie
   null-space, tylko brak dryfu do chronienia.
5. **Kryterium parowe sprawdziło się w pierwszej serii w obu
   kierunkach** (K1 parowy+, K2 parowy−) i uczciwie odmówiło na
   K1-CIFAR oraz K2-MNIST (5/5 par jednego znaku, ale średnia
   < 2×std delt). Konwencja zostaje.

## STATUS KOŃCOWY SERII K: KOMPLET — kryterium wyjścia SPEŁNIONE

(a) Oba sufity zmierzone: Fashion 81.16 (J4 all_300), CIFAR 39.65 (K0).
(b) Wszystkie dźwignie rozstrzygnięte: sparse×300d = parowy+ Fashion /
SZUM CIFAR; OWM = wyeliminowany wszędzie.
(c) **Headline v0.5:** class-IL bez jednego przechowanego obrazu —
Fashion **79.23 ± 0.73 (97.6% sufitu zamrożonych cech)**, CIFAR
**37.51 ± 1.35 (94.6% sufitu)**. Mechanizm WYŻYŁOWANY: pozostała luka
na obu benchmarkach jest reprezentacyjna, nie mechanizmowa.

Dalej wg PLAN_GENERALNY: merge `droga-k` → main (decyzja Roberta),
potem DROGA_I_PLAN.md — Droga I startuje z sparse_k16 × 300d (Fashion),
protokół wymiany statystyk niezależny od backbone'u.
