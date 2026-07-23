# Droga Q2c — kontrola uczciwości: sekwencyjny z symetrycznym budżetem snu (pre-rejestracja)

Data pre-rejestracji: 2026-07-23 (po werdyktach Q2, PRZED runem Q2c).
Status: DO ZATWIERDZENIA; runy WYŁĄCZNIE u Roberta. Branch: `droga-q`
(kontynuacja; nowy plik, istniejące NIETKNIĘTE).

## Po co ta kontrola (confound nazwany w DROGA_Q_NOTATKI.md)

q2b trenuje projekcję kolektora na 2500 śnionych próbkach/klasę nowej
paczki; m1_seq trenuje na 500 realnych cechach/klasę. Obserwacja
„kolektyw +3.59pp nad seq" miesza dwie zmienne: źródło materiału
(sen vs real) i jego LICZNOŚĆ. Q2c wyrównuje liczność po stronie seq.

## Wariant

**seq_selfdream2500**: agent sekwencyjny jak m1_seq_300, z jedną
zmianą w learn_task: po policzeniu statystyk nowej klasy z realnych
cech (pierwsza generacja — zgodne z falsyfikacją O) materiał treningowy
projekcji i podów jest dośniewany do 2500 próbek/klasę (500 realnych +
2000 snów z własnych statystyk). Stary rehearsal bez zmian. Pełna
symetria z q2b: te same statystyki (k=16 sparse z 500 realnych
próbek), ta sama liczność 2500, ta sama maszyneria snu.

Setup poza tym identyczny z M1/Q1/Q2: TASKS20, pretrained cache, 300d,
sparse k16, epochs 15, LR 0.001, seedy 0–4 (parowanie z M1 i Q2).

## Kryteria werdyktu (Z GÓRY)

1. **seq_selfdream2500 vs m1_seq_300** (pary): czy dźwignia budżetu
   działa też dla uczenia? SYGNAL+ = tak; SZUM = nie (budżet pomaga
   tylko adopcji); SYGNAL− = augmentacja snem szkodzi uczeniu
   z realnych cech.
2. **ROZSTRZYGNIĘCIE GŁÓWNE — q2b vs seq_selfdream2500** (pary,
   te same seedy; q2b z results/Q2_early_repair.json):
   - SZUM → **RÓWNOWAŻNOŚĆ**: headline „bariera skali była artefaktem
     budżetu snu; przy symetrycznych budżetach kolektyw (95 klas
     z wiadomości 24 KB) ≈ agent sekwencyjny na realnych danych";
   - SYGNAL+/parowy+ (kolektyw wyżej) → **przewaga kolektywu REALNA**:
     materiał śniony ze statystyk ≥ mały materiał realny dla tej
     projekcji — najmocniejszy możliwy headline v1.2;
   - SYGNAL−/parowy− → seq z pełnym budżetem odzyskuje przewagę;
     twierdzenie „kolektyw nad seq" upada (zostaje: bariera zamknięta
     do −ε; zmierzyć ε).
3. Obserwacje: luka najlepszego systemu do sufitu all-data 47.41;
   forgetting; porównanie do q2a.

## Interpretacja z góry

Każdy z trzech wyników rozstrzygnięcia domyka serię Q z czystym
twierdzeniem. Po Q2c: aktualizacja WHITEPAPER (sekcja Q), CLAIMS,
merge `droga-q` → tag v1.2. Ewentualne dalsze dźwignie (kombinacja
q2a+q2b, harmonogram) — osobna decyzja PO domknięciu.

## Plik, koszt, wynik

- Runner: `src/run_Q2c_seq_selfdream.py` (nowy; podklasa z learn_task
  jako wierna kopia MarsCollectiveM.learn_task + augmentacja — zero
  zmian w istniejących plikach).
- Wynik: `results/Q2c_seq_selfdream.json` (smoke: `_smoke`).
- Koszt FULL: 20 zadań × trening na ~12.5k cech ≈ tempo q2b ≈
  **~15 min** (5 seedów). Smoke: 1 seed, 4 epoki.
- Wymaga: cache cech CIFAR-100, glove 300d, results/M1_long_horizon.json
  i results/Q2_early_repair.json (bazy par).

## Zasady

5 seedów, pary per-seed, próg szumu std+std, SYGNAL-parowy obok,
min per-seed raportowany, negatyw = wynik.
