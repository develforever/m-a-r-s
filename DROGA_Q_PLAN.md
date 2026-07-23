# Droga Q — kolektyw na długim horyzoncie (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: DO ZATWIERDZENIA przez Roberta;
runy WYŁĄCZNIE u Roberta. Branch: `droga-q` (main nietykalny; nowe pliki,
istniejący kod NIETKNIĘTY — MarsCollectiveM z serii M używany bez zmian).

## Pytanie i kontekst (zmierzony)

Kolektyw zmierzono dotąd tylko przy 10 klasach: I3 (Fashion, SZUM vs
sekwencyjny) i L2 (CIFAR-10 pretrained, koszt protokołu −0.56pp,
parowy−). Długi horyzont zmierzono tylko sekwencyjnie: M1 (CIFAR-100,
20 zadań, 40.70 ± 0.84 = 85.8% sufitu 47.41; deficyt późny −7.8pp,
strukturalny). Seria Q krzyżuje te osie: **czy protokół wymiany snów
przenosi się na 100 klas — i ile wtedy kosztuje?** To test skali
„rewolucji": 20 agentów, 95 klas z wiadomości po 24.1 KB, zero obrazów.

## Hipotezy

- H-Q1 (główna): koszt protokołu na 100 klasach pozostaje mały
  (rząd L2, <1pp) — adopcja używa tych samych statystyk, które
  sekwencyjny i tak śni; rosnąca przestrzeń etykiet uderza w obu
  tak samo (dekompozycja M).
- H-Q2 (ryzyko, pre-rejestrowane): adopcja może płacić WIĘCEJ niż
  uczenie na późnych zadaniach — projekcja u kolektora uczy się klas
  adoptowanych wyłącznie ze snów (5000→500 realnych cech nigdy nie
  widzi), a budżet snów na stare klasy dzieli się na coraz więcej klas.

## Setup

- Dane i podział: identyczne z M1 — CIFAR-100, TASKS20 (20 zadań × 5
  klas, kolejność rosnąca), cache cech resnet18 (extract_or_load),
  ReducedBackbone per seed, kotwice **300d** (SYGNAL+ z M1), sparse
  k=16, epochs 15, LR 0.001, seedy 0–4 (parowanie z M1 wymaga tych
  samych seedów — to NIE jest test obserwacji z tych seedów, więc
  świeże seedy nie są wymagane).
- Kolektyw: kolektor = agent 0 uczy task 0 lokalnie; nadawcy A_1..A_19:
  agent i uczy WYŁĄCZNIE task i (init na własnym tasku); kolektor
  adoptuje taski 1..19 w kolejności (paczki po 5 klas,
  `MarsCollectiveM.adopt_classes`, n_dream=500 — parytet liczności
  klasy CIFAR-100, jak w kodzie M). Payload: 5 × 24.1 KB na paczkę;
  łącznie 95 wiadomości, zero obrazów.
- Ewaluacja po każdej adopcji: eval_protocols na zadaniach 0..t
  (class-IL, pełna macierz R jak M1).

## Kryteria werdyktu (Z GÓRY)

1. **GŁÓWNE — koszt protokołu na długim horyzoncie:** ACC końcowe
   kolektywu vs `m1_seq_300` (results/M1_long_horizon.json, TE SAME
   seedy, pary per-seed, próg szumu std+std, SYGNAL-parowy obok):
   - SZUM → protokół przenosi się na 100 klas bez mierzalnego kosztu
     (headline: „95 klas z wiadomości 24 KB ≈ agent sekwencyjny");
   - SYGNAL−/parowy− → koszt protokołu zmierzony; obserwacja
     porównawcza (bez rangi werdyktu — inne dane): |d| vs 0.56pp z L2,
     czyli czy koszt rośnie z liczbą klas;
   - SYGNAL+ (symetrycznie, nieoczekiwany) → adopcja ze statystyk
     bije uczenie z cech realnych — wymagałby replikacji przed
     jakimkolwiek twierdzeniem.
2. **Q1b — deficyt późny adopcji (obserwacja rangi pomocniczej):**
   R[t][t] kolektywu (acc paczki t tuż po adopcji) early(1–5) vs
   late(16–20), na tle sufitu `m1_all_300` per zadanie (mitygacja
   confoundu trudności jak w M1). Pytanie: czy późne ADOPCJE płacą
   więcej niż późne UCZENIE (M1: deficyt −7.8pp vs sufit)? Raport par
   kolektyw-vs-seq na spadku early→late; bez progu twierdzenia
   (pierwszy pomiar tej osi).
3. Obserwacje: krzywa ACC po adopcjach; luka do sufitu 47.41;
   forgetting; pamięć kolektora (100 × 24.1 KB = 2.41 MB).

## Interpretacja z góry

SZUM w (1) = najmocniejszy wynik serii: skala kolektywu potwierdzona
na 100 klasach — kandydat na headline v1.2. SYGNAL− w (1) też jest
wynikiem: pierwsza zmierzona bariera skali protokołu (front do ataku
w ewentualnym Q2: budżet snów przy adopcji, kolejność paczek).
NIE planować Q2 przed werdyktem Q1.

## Plik, koszt, wynik

- Runner: `src/run_Q1_collective_horizon.py` (nowy; zero zmian
  w istniejących plikach).
- Wynik: `results/Q1_collective_horizon.json` (smoke: `_smoke`).
- Koszt FULL (szacunek z M1: 768 s / 3 warianty / 5 seedów ≈ 51 s na
  wariant-seed): kolektor ≈ 1 wariant-seed + 19 nadawców po 1 tasku
  (init + task, tanie) ≈ 2–3 min/seed → **~15 min FULL**. Smoke:
  1 seed, 4 epoki (~2 min).
- Wymaga: data/glove.6B.300d.txt, cache cech CIFAR-100 (z M1),
  results/M1_long_horizon.json (baza par).

## Zasady

5 seedów, pary per-seed, progi jw. zamrożone, min per-seed raportowany,
negatyw = wynik, merge `droga-q` po komplecie werdyktów → tag v1.2.
