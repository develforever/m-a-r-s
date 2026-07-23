# Droga Q2d — kombinacja dźwigni: budżet 2500 + re-adopcja (pre-rejestracja)

Data pre-rejestracji: 2026-07-23 (po komplecie Q1–Q2c, PRZED runem Q2d).
Status: DO ZATWIERDZENIA; runy WYŁĄCZNIE u Roberta. Branch: `droga-q`
(kontynuacja; nowy plik, istniejące NIETKNIĘTE). NIE blokuje merge'a
v1.2 — headline serii Q stoi niezależnie od wyniku Q2d.

## Pytanie

Czy dźwignie q2a (re-adopcja wczesnych paczek) i q2b (budżet snu 2500)
są ADDYTYWNE? q2b zamknęło barierę (44.29), ale luka do seq_selfdream
(45.35, pary −1.07) i do sufitu all-data (47.41) zostaje. Jeśli przy
budżecie 2500 wczesne adopcje nadal odstają (analog Q1b — NIEZMIERZONE
przy 2500), re-adopcja powinna dodać kolejny kawałek.

## Hipotezy (Z GÓRY)

- H-Q2d: kombinacja > q2b (pary dodatnie) — wczesny deficyt nie znika
  w całości od samego budżetu.
- Anty-hipoteza (też wynik): budżet subsumuje naprawę (SZUM) — dojrzałość
  projekcji przestaje być wąskim gardłem przy dostatecznym materiale;
  wtedy luka do selfdream to koszt rezydualny snu vs realnych cech.

## Setup

Wariant **q2d_combo**: ścieżka q2b (n_dream=2500 przy każdej adopcji)
+ po 19. adopcji naprawa paczek 1–5 (unlearn light + re-adopcja
z payloadów przechowanych, n_dream=2500). Wszystko inne identyczne
z Q2 (TASKS20, pretrained cache, 300d, sparse k16, epochs 15, seedy
0–4). Metryka: ACC = średnia finalnego wiersza po naprawie.

## Kryteria werdyktu (Z GÓRY)

1. **GŁÓWNE — q2d_combo vs q2b** (pary, results/Q2_early_repair.json):
   SYGNAL+/parowy+ = dźwignie addytywne (raport: ile z luki 1.07pp do
   selfdream zamknięte); SZUM = budżet subsumuje naprawę; SYGNAL− =
   re-adopcja przy 2500 szkodzi (nadpisanie dojrzałych mapowań).
2. Obserwacje: pary vs seq_selfdream2500 (czy luka domknięta — SZUM
   = kolektyw doszlusował do najlepszego pojedynczego agenta);
   acc zadań 1–5 przed/po naprawie (mechanizm); luka do sufitu 47.41.

## Plik i koszt

- Runner: `src/run_Q2d_combo.py` (nowy). Wynik:
  `results/Q2d_combo.json` (smoke: `_smoke`).
- Koszt FULL ≈ q2b + 5 napraw na budżecie 2500 ≈ **~20 min**.
  Smoke: 1 seed, 4 epoki.
- Wymaga: cache CIFAR-100, glove 300d, results/Q2_early_repair.json,
  results/Q2c_seq_selfdream.json.

## Zasady

5 seedów, pary per-seed, próg std+std, SYGNAL-parowy obok, negatyw =
wynik. Werdykt → DROGA_Q_NOTATKI.md (dopisek po v1.2 lub w v1.2,
zależnie od momentu merge'a — decyzja Roberta).
