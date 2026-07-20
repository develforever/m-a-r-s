# Droga M — notatki robocze

Plan: `DROGA_M_PLAN.md` (pre-rejestracja 2026-07-20). Runy: Robert,
lokalnie, 5 seedów, epochs=15, CIFAR-100 × 20 zadań, pretrained
(cache cech). Werdykty i dekompozycja przeliczone niezależnie z JSON-ów.

## M1 — długi horyzont (ZAKOŃCZONE, 20.07.2026): formalnie
## LOSS-OF-PLASTICITY, ale dekompozycja: ~79% spadku to twardnienie
## protokołu; realna degradacja = późne R[t][t] −7.8pp vs sufit.
## Pierwszy czysty SYGNAL+ dla geometrii kotwic (300d przy 100 klasach).

Plik: `src/run_M1_long_horizon.py`; wyniki:
`results/M1_long_horizon.json`. Czas: 768 s (+ ekstrakcja cache).

| Wariant (class-IL, T=20) | ACC | F | R[t][t] 1–5 → 16–20 |
|---|---|---|---|
| **m1_seq_300 (GŁÓWNY)** | **40.70 ± 0.84%** | 18.3pp | 78.6% → 42.9% |
| m1_all_300 (sufit) | 47.41 ± 0.49% | 13.0pp | 79.0% → 50.7% |
| m1_seq_50 | 32.99 ± 0.81% | 20.6pp | 78.7% → 33.6% |

**WERDYKT GŁÓWNY (pre-rejestrowany): LOSS-OF-PLASTICITY** — spadek
R[t][t] 35.68pp (pary 5/5, próg 5.50). Werdykt formalnie stoi.

**DEKOMPOZYCJA (uczciwa, kluczowa dla interpretacji):**
1. Sufit `all` — projekcja bez jakiegokolwiek dryfu sekwencyjnego —
   spada na własnych R[t][t] o **28.34pp** (z tych samych powodów
   protokolarnych: późne zadania to dyskryminacja w ~100 widzianych
   klasach, wczesne w 5–25). To NIE jest utrata plastyczności —
   to twardnienie zadania w class-IL.
2. Nadwyżka seq nad all: śr. 7.34pp, ale pary mieszane [12.9, 8.7,
   4.5, 12.3, −1.8] — na spadkach per-seed formalnie szum. Czysty
   sygnał jest w poziomach: **późne R[t][t] seq vs all: −7.77pp,
   pary 5/5 ujemne** [−9.4, −8.2, −7.6, −9.9, −3.7] przy równości
   wczesnej (−0.42, pary mieszane). Mechanizm uczy się późnych zadań
   gorzej niż sufit — o ~8pp, nie o 36.
3. mech% sufitu (ACC): **85.8% @T=20 vs 96.7% @T=5 (L1)** — realny,
   umiarkowany koszt horyzontu.
4. Stabilność trzyma: forgetting 18.3pp przy T=20 (Fashion @T=5: ~16) —
   problemem długiego horyzontu jest plastyczność, nie zapominanie.

**Lekcja metodyczna (odnotowana, bez zmiany werdyktu wstecz):**
pre-rejestrowana metryka mitygacyjna (R[t][t]/acc_all z KOŃCOWEGO
wiersza all) myli przestrzenie etykiet (157%→87% — nieinterpretowalne);
właściwe odniesienie to WŁASNE R[t][t] wariantu all (ten sam protokół).
Do przyszłych pre-rejestracji długiego horyzontu: porównanie
R[t][t]_seq vs R[t][t]_all parami per-seed jako kryterium główne.

**HIPOTEZA MECHANISTYCZNA (pre-rejestrowana w M1b PRZED runem):**
balansowanie snów per klasę (k_per_old=51) przy 95 starych klasach
daje ~4845 snów na 512 realnych próbek w każdym kroku projekcji
(~90% batcha) — nowe zadanie tonie w snach. Analogicznie pody:
256 negatywów × 95 klas = ~24k negatywów na ~500 realnych. Konwencja
zaprojektowana na T=5 (max 8 starych klas) przeskalowała się w drugą
stronę przy T=20. Poprawka M1b: budżet snów/negatywów STAŁY ŁĄCZNIE
(dzielony po starych klasach), nie stały per klasę.

**Obserwacja (kotwice): pierwszy czysty SYGNAL+ dla geometrii słów** —
300d vs 50d: **+7.71pp (pary 5/5: 6.9…8.2, próg ~1.6)**. Przy 10
klasach 50d wystarczało (J4: null); przy 100 klasach stłoczenie kotwic
w 50d kosztuje ~8pp. Wniosek skalowania: wymiar przestrzeni słów musi
rosnąć z liczbą klas.

Status: M1b (dopisek do planu PRZED runem) — w toku.

## M1b — budżet snów stały łącznie (ZAKOŃCZONE, 20.07.2026):
## HIPOTEZA SFALSYFIKOWANA — balans per klasę to mechanizm stabilności,
## nie bug; front stability–plasticity zmierzony z obu końców

Plik: `src/run_M1b_balanced_dreams.py`;
wyniki: `results/M1b_balanced_dreams.json`. Czas: 203 s.

| Wariant (T=20) | ACC | F | late R[t][t] | sny/klasę (95 starych) |
|---|---|---|---|---|
| [M1] seq_300 (per klasę) | 40.70 ± 0.84% | 18.3pp | 42.9% | 51 (~90% batcha) |
| m1b (budżet łączny) | 24.93 ± 2.48% | 44.9pp | 71.3% | ~5 (~50% batcha) |
| [M1] all_300 (sufit) | 47.41 ± 0.49% | 13.0pp | 50.7% | — |

**WERDYKTY (pre-rejestrowane):**
- ACC vs M1: **SYGNAL−** −15.77pp (pary 5/5, próg 3.31) — ryzyko
  pre-rejestrowane zmaterializowane w pełni.
- late R[t][t] vs M1: **SYGNAL+** +28.38pp (pary 5/5) — plastyczność
  przywrócona z nawiązką (71.3% — NAD sufitem all 50.7%, bo projekcja
  bez kotwicy rehearsalu nadpasowuje się do świeżych zadań).

**Ustalenia:**
1. **Koszt długiego horyzontu jest STRUKTURALNY, nie implementacyjny:**
   przy stałym batchu nie da się mieć naraz pełnego rehearsalu
   i pełnej plastyczności — budżet snów na starą klasę to pokrętło,
   które je wymienia. Dwa końce zmierzone: 51/klasę → F 18.3,
   late −7.8 vs sufit; ~5/klasę → late +20.6 vs sufit, F 44.9.
2. M1 (40.70) ≫ M1b (24.93) na ACC — konfiguracja per klasę pozostaje
   DOMYŚLNA; deficyt −7.8pp na późnych zadaniach to cena stabilności.
3. Kandydat M1c (dopisany PRZED runem): jeden punkt pośredni frontu.

## M1c — patrz dopisek w DROGA_M_PLAN.md (pre-rejestracja przed runem).

## M1c — punkt pośredni frontu (ZAKOŃCZONE, 20.07.2026): SYGNAL− na ACC
## — front ostry, M1 zostaje domyślne; M1b ZDOMINOWANE przez M1c

Plik: `src/run_M1c_mid_budget.py`; wyniki: `results/M1c_mid_budget.json`.
Czas: 231 s.

| Punkt frontu (T=20) | sny/neg na klasę | ACC | F | late R[t][t] |
|---|---|---|---|---|
| **M1 (domyślny)** | 51 / 256 | **40.70 ± 0.84%** | 18.3pp | 42.9% |
| M1c | 16 / 32 | 37.65 ± 1.35% | 43.0pp | 81.9% |
| M1b | ~5 / ~5 | 24.93 ± 2.48% | 44.9pp | 71.3% |
| [sufit all] | — | 47.41 ± 0.49% | 13.0pp | 50.7% |

**WERDYKT (pre-rejestrowany): SYGNAL−** — ACC −3.04pp (pary 5/5
ujemne [−1.41…−4.63], próg 2.19) → front jest ostry po stronie
stabilności; M1 (bogaty rehearsal) pozostaje konfiguracją domyślną.

**Ustalenia:**
1. **M1b nie leży na froncie Pareto:** M1c bije je na OBU osiach
   (ACC +12.7pp ORAZ late R[t][t] +10.6pp) — skrajne odcięcie snów
   szkodzi nawet plastyczności (projekcja bez kotwicy rehearsalu
   dryfuje chaotycznie). Front realny rozpina się między M1 a M1c.
2. Pokrętło budżetu snów przesuwa system po froncie w sposób
   przewidywalny: 51/klasę → ACC-optimum, stare klasy chronione;
   16/klasę → −3pp ACC, ale późne zadania uczone na 81.9% (NAD
   sufitem all) — konfiguracja dla aplikacji, gdzie świeże klasy
   ważniejsze niż archiwum. Wybór jest teraz świadomy i zmierzony.
3. Wysokie F przy M1c/M1b to artefakt wyższych szczytów (więcej jest
   do zapomnienia), nie niższych finałów — ACC końcowe różni się
   o 3pp, forgetting o 25pp.

## STATUS KOŃCOWY SERII M: KOMPLET

M1: formalnie LOSS-OF-PLASTICITY; dekompozycja — 79% spadku to
twardnienie protokołu, realny deficyt późny −7.8pp vs sufit; mech%
85.8% @T=20 · kotwice: SYGNAL+ 300d vs 50d (+7.71, 5/5) — wymiar słów
musi rosnąć z liczbą klas · M1b: falsyfikacja hipotezy implementacyjnej
— koszt strukturalny · M1c: front ostry, M1 domyślne; trójpunktowa
krzywa frontu stability–plasticity zmierzona (wykres do papieru).

Dalej: merge `droga-m` → main (v0.8, decyzja Roberta) → seria N
(unlearning, dwupoziomowe kryterium) → I4.
