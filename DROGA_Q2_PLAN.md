# Droga Q2 — naprawa wczesnego deficytu adopcji (pre-rejestracja)

Data pre-rejestracji: 2026-07-23 (po werdykcie Q1, PRZED jakimkolwiek
runem Q2). Status: DO ZATWIERDZENIA; runy WYŁĄCZNIE u Roberta.
Branch: `droga-q` (kontynuacja; nowe pliki, istniejące NIETKNIĘTE).

## Punkt wyjścia (zmierzony, Q1)

Bariera skali: kolektyw 34.02 ± 0.65 vs seq 40.70 ± 0.84 (−6.67 ± 0.88pp,
SYGNAL−). Koszt zlokalizowany: WCZESNE adopcje (~37% R[t][t] vs ~78%
uczenia), późne adopcje = późne uczenie (ratio sufitu 0.913 vs 0.870).
Hipoteza mechanizmu: adopcja ze snów w niedojrzałej projekcji.

## Hipotezy i przewidywania (Z GÓRY)

- **H-Q2a (re-adopcja):** wczesne paczki ponownie adoptowane w DOJRZAŁEJ
  projekcji (po komplecie 19 adopcji) doszlusują do poziomu późnych.
  Przewidywanie zapisane przed runem: **+2–3pp ACC** (5 wczesnych
  paczek × ~+8pp R / 20 zadań). Ryzyko pre-rejestrowane: przy 100
  klasach w tle budżet snów starych na klasę jest minimalny — re-adopcja
  może nie odtworzyć warunków późnych adopcji; negatyw = wynik
  („wczesny deficyt nie jest naprawialny re-adopcją").
- **H-Q2b (budżet snu):** n_dream adopcji 500→2500 częściowo kompensuje
  niedojrzałą projekcję. Ryzyko pre-rejestrowane: payload jest nasycony
  (I2b — saturacja między 500 a 3000 obrazów); więcej snów z TEJ SAMEJ
  wiadomości może nie nieść nowej informacji; negatyw potwierdziłby
  saturację jako mechanizm także po stronie odbiorcy.

## Setup

Baza = DOKŁADNIE ścieżka Q1 (TASKS20, pretrained cache, 300d, sparse
k16, epochs 15, LR 0.001, seedy 0–4 — parowanie z Q1 wymaga tych samych
seedów). Dwa warianty, każdy pełną ścieżką:

- **q2a_readopt**: ścieżka Q1 bez zmian (n_dream=500) + po 19. adopcji
  naprawa paczek 1–5 w kolejności: dla każdej paczki
  `unlearn_class(light)` na jej 5 klasach + `adopt_classes` z PAYLOADÓW
  PRZECHOWANYCH z pierwszej adopcji (pierwsza generacja statystyk —
  zero rekursji snu, zgodne z falsyfikacją O; maszyneria I4b).
- **q2b_dream2500**: ścieżka Q1 z n_dream=2500 przy KAŻDEJ adopcji
  (payloady bez zmian — 24.1 KB; rośnie tylko lokalna praca odbiorcy).

Metryka główna: ACC = średnia finalnego wiersza class-IL po wszystkim
(po naprawie w q2a; po 19. adopcji w q2b) — identycznie liczona jak
ACC Q1 (średnia finalnego wiersza).

## Kryteria werdyktu (Z GÓRY)

1. Per wariant: pary vs kolektyw Q1 (results/Q1_collective_horizon.json,
   te same seedy; próg szumu std+std, SYGNAL-parowy obok):
   - SYGNAL+/parowy+ → dźwignia działa; raport „odzysk bariery"
     = d / 6.67 (ułamek zamkniętej luki do seq);
   - SZUM → dźwignia nie działa (dla q2b: wsparcie saturacji payloadu);
   - SYGNAL− → dźwignia szkodzi (dla q2a: re-adopcja w zatłoczonej
     projekcji gorsza niż pierwotna — też wynik).
2. Mechanizm q2a (obserwacja): finalne acc zadań 1–5 przed naprawą
   (finalny wiersz Q1) vs po naprawie, pary per-seed.
3. Obserwacja domykająca: pary najlepszego wariantu vs m1_seq_300 —
   czy bariera zamknięta w całości (SZUM = kolektyw doszlusował).
4. Obserwacja: forgetting; czas.

## Interpretacja z góry

Dowolny SYGNAL+ = pierwsza zmierzona dźwignia przeciw barierze skali
(kandydat headline v1.2 razem z Q1). Podwójny SZUM/− = bariera jest
głębsza niż dojrzałość projekcji/budżet snu — kierunek na Q3
(harmonogram protokołu) do osobnej decyzji. NIE łączyć a+b w tym runie
(czysta atrybucja); kombinacja dopiero po werdyktach, jeśli oba +.

## Plik, koszt, wynik

- Runner: `src/run_Q2_early_repair.py` (nowy; zero zmian w istniejących).
- Wynik: `results/Q2_early_repair.json` (smoke: `_smoke`).
- Koszt FULL: q2a ≈ Q1 + 5 napraw ≈ 9 min; q2b ≈ 2–3× Q1 (adopcje na
  5× snach) ≈ 15–20 min; razem **~25–30 min**. Smoke: 1 seed, 4 epoki.
- Wymaga: cache cech CIFAR-100, glove 300d,
  results/Q1_collective_horizon.json (baza par), M1 (obs. 3).

## Zasady

5 seedów, pary per-seed, progi jw. zamrożone, min per-seed raportowany,
negatyw = wynik. Werdykty → DROGA_Q_NOTATKI.md, merge `droga-q` → v1.2.
