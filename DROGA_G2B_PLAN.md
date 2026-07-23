# Droga G2b — słownik atrybutów z dystansem kodowym (ECOC) (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: DO ZATWIERDZENIA; runy
WYŁĄCZNIE u Roberta. Branch: `droga-g2b` (nowe pliki, istniejące
NIETKNIĘTE). Otwarte od G2 (v0.3); ścieżka wskazana w werdykcie G2:
„attribute vocabularies need error-correcting code distance".

## Punkt wyjścia (zmierzony, G2)

Zero-shot z opisu atrybutowego: średnio 3.2% vs próg 30% (NEGATYW),
przy dwóch zdiagnozowanych wadach słownika 11 atrybutów:
(1) min dystans Hamminga między klasami = 1 (mylące pary różnią się
jednym bitem); (2) trzy atrybuty unikalne dla jednej klasy → stałe
w treningu bez niej → klasa nieosiągalna (reguła strukturalna,
potwierdzona 3/3: Sandal, Bag, AnkleBoot po 0.0%).

## Hipoteza

H-G2b: słownik naprawiający OBIE wady — min dystans kodowy ≥ 3 oraz
każdy atrybut warujący w każdym zbiorze treningowym leave-one-out —
podnosi zero-shot. Mechanizm: (a) dystans koryguje pojedyncze błędy
detektorów pojęć; (b) brak atrybutów stałych czyni wszystkie 10 klas
strukturalnie osiągalnymi.

## Słownik ATTRS21 (ZAMROŻONY tą pre-rejestracją)

21 binarnych atrybutów słownych, własności ZWERYFIKOWANE skryptem
przed runem: wiersze unikalne; **min parowy dystans Hamminga = 4**
(G2: 1); każda kolumna ma 2–8 jedynek → żaden atrybut nie jest stały
w żadnym leave-one-out (reguła osiągalności przewiduje **10/10
klas osiągalnych**); zero duplikatów kolumn. Nazwy (klasy z „1"):

zakrywa_tulow [0,2,3,4,6] · zakrywa_nogi [1,3] · ma_rekawy [0,2,4,6] ·
dlugi_rekaw [2,4,6] · jest_obuwiem [5,7,9] · siega_kostki [1,9] ·
odkryta_gora [5,8] · ma_klamre_lub_zapiecie [5,8,9] ·
pelna_dlugosc [3,4] · rozpinane_z_przodu [4,6] ·
nakladane_przez_glowe [0,2,3] · sznurowane [7,9] · z_dzianiny [0,2] ·
warstwa_wierzchnia [2,4] · na_zime [2,4,9] · na_lato [0,5] ·
dolna_czesc_ciala [1,5,7,9] · miekki_material [0,1,2,3,6] ·
sztywne_elementy [5,7,8,9] · przechowuje [4,8] · ze_skory [8,9]

(Semantyka: siega_kostki obejmuje spodnie; odkryta_gora — sandał
i torbę; przechowuje — kieszenie płaszcza i torbę; ze_skory — torbę
i botek. Każda nazwa to obronialne pojęcie wizualne, nie kod sztuczny.)

## Setup

Identyczny z G2 (reprodukowalność): Fashion-MNIST, losowy zamrożony
backbone per seed (MarsCLSystem random), projekcja liniowa 128→K,
uczenie POJĘĆ (BCE per atrybut, styl DAP), routing = najbliższy wektor
atrybutów (L2 po sigmoid), leave-one-out po 10 klasach, 5 seedów,
15 epok. Dwa warianty w jednym runnerze:
- **attrs11** — oryginalna macierz G2 (reprodukcja sanity, wzór
  precedensu J1/J3/J4: oczekiwane odtworzenie 3.2%);
- **attrs21** — słownik ECOC jw.

## Kryteria werdyktu (Z GÓRY)

1. **SUKCES PEŁNY**: średni ZS attrs21 (10 klas × 5 seedów) > 30%
   (oryginalny próg G2) → kompozycyjny zero-shot osiągnięty; negatyw
   G2 był wadą słownika, nie mechanizmu.
2. **SUKCES MECHANIZMU (słaby)**: pary per-seed średniego ZS
   attrs21 vs attrs11: SYGNAL+ (śr > próg szumu std+std, min par > 0)
   ORAZ śr ZS attrs21 ≥ 2× attrs11 → dystans kodowy działa, ale sufit
   niżej niż 30% (raportować, gdzie).
3. **NEGATYW**: brak obu → dystans kodowy nie jest dźwignią zero-shot
   na losowych cechach (kierunek: cechy, nie kod — też domyka G2b).
4. **Test reguły osiągalności** (przewidywanie z góry): wszystkie 10
   klas ma średni ZS > 0%, W TYM dawne porażki strukturalne
   {Sandal, Bag, AnkleBoot}. Jeśli któraś = 0.0% na wszystkich seedach
   → reguła osiągalności jest niepełna (osobny wynik, raportować).
5. Obserwacje: seen-acc (trade-off bias-to-seen z G2), ZS per klasa
   vs min dystans kodowy klasy, porównanie attrs11 z wynikiem G2
   (reprodukcja).

## Plik, koszt, wynik

- Runner: `src/run_G2b_ecoc.py` (nowy; run_holdout = wierna kopia G2).
- Wynik: `results/G2b_ecoc.json` (smoke: `_smoke`).
- Koszt FULL: 2 warianty × 5 seedów × 10 holdoutów × mała projekcja
  liniowa — rząd **kilku–kilkunastu minut**. Smoke: 1 seed, 4 epoki.
- Wymaga: Fashion-MNIST (auto), brak zależności od GloVe (atrybuty
  binarne jak w G2).

## Zasady

5 seedów, progi jw. zamrożone, min per-seed raportowany, negatyw =
wynik. Werdykty → DROGA_G2B_NOTATKI.md (nowy), potem CLAIMS/WHITEPAPER.
