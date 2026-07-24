# Droga G3 — kompozycyjność na cechach pretrained (isolacja dźwigni c) (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: **DO ZATWIERDZENIA**; runy
WYŁĄCZNIE u Roberta. Branch: `droga-g3` (nowe pliki, istniejący kod
NIETKNIĘTY; `run_holdout` z G2 reużywane WERBATIM). Oś: Part II
(routing / zero-shot ceiling — paper B), NIE główna narracja CL.

## Punkt wyjścia (zmierzony)

Diagnoza G2 wskazała trzy dźwignie kompozycyjnego zero-shot: (a) dystans
kodowy słownika, (b) dekorelacja detektorów, (c) cechy lepsze niż losowe.
- **G2** (v0.3): losowy backbone, słownik 11 atrybutów → ZS 3.2% vs próg
  30% (NEGATYW), reguła osiągalności 3/3.
- **G2b**: dźwignia (a) w izolacji (ECOC, min Hamming 4) → NEGATYW MOCNY,
  **backfire** (attrs21 0.18% < attrs11 3.17%), reguła 10/10 sfalsyfiko-
  wana. Mechanizm z notatek G2b: kod korekcyjny działa tylko poniżej
  progu błędu per-bit; detektory na losowych cechach są nad progiem.
  **Wąskie gardło zizolowane do CECH (dźwignia c), nie słownika.**

G3 testuje dźwignię (c) WPROST: ta sama maszyneria kompozycyjna, ten sam
słownik, jedyna zmiana = cechy losowe → cechy pretrained ResNet18.

## Hipoteza

H-G3: kompozycyjny zero-shot był zablokowany na jakości cech. Cechy
semantyczne (ResNet18-ImageNet) dają detektory pojęć per-atrybut poniżej
progu błędu per-bit → (i) ZS rośnie istotnie względem losowych cech, a
(ii) na mocnych cechach dystans kodowy (attrs21) PRZESTAJE szkodzić i
może wreszcie pomóc (bezpośredni test mechanizmu z G2b: interakcja
dźwigni a×c).

## Setup (identyczny z G2/G2b poza backbonem)

Fashion-MNIST, leave-one-out po 10 klasach, uczenie POJĘĆ (BCE per
atrybut, styl DAP), routing = najbliższy wektor atrybutów (L2 po
sigmoid), 5 seedów, 15 epok. `run_holdout` z `run_G2_compositional`
reużywane bez zmian (operuje na policzonych cechach `feats_tr`/`feats_te`
— jest agnostyczne wobec źródła i wymiaru cech).

Dwa źródła cech × dwa słowniki na TYCH SAMYCH seedach:
- **backbone:** `random` (MarsCLSystem losowy, jak G2b — reprodukcja
  sanity) vs `pretrained` (ResNet18-IMAGENET1K_V1 zamrożony, cechy
  penultimate 512-d).
- **słownik:** `attrs11` (oryginał G2) vs `attrs21` (ECOC z G2b).

**Infrastruktura z serii L (reużyta):** `mars_cl_l.PretrainedBackbone`
(ResNet18-IMAGENET1K_V1 zamrożony, zawsze eval) + wzór
`extract_or_load_cifar_feats` (jednorazowa ekstrakcja + cache na dysku,
bo ResNet na 1050 Ti jest za wolny na przelicznie w każdym przebiegu).
G3 dokłada wariant Fashion tej samej ścieżki.

**Ekstrakcja cech pretrained (Fashion → ResNet18, zamrożona,
cache'owana raz):** Fashion 1×28×28 → powielenie do 3 kanałów →
resize 28→224 (bilinear) → normalizacja ImageNet → zamrożony ResNet18
(features bez fc) → 512-d. Cache: `data/fashion_resnet18_224_feats.pt`
(wzór `extract_or_load_cifar_feats` z L). Backbone pretrained BEZ
losowej redukcji 512→128 — cechy pełne 512-d (wymiar współzmienia się
z jakością; to jest częścią „mocnych cech", raportowane jawnie). Losowy
backbone zostaje przy swoim natywnym wymiarze (reprodukcja G2b). Oba
słowniki (attrs11, attrs21 min Hamming 4) na TYCH SAMYCH 5 seedach dla
obu backbone'ów — pełny plan 2×2, pary per-seed.

## Kryteria werdyktu (Z GÓRY) — TWARDA DECYZJA BINARNA G3+ / G3−

Metryka: średni ZS (10 klas leave-one-out × 5 seedów). Statystyka
rozstrzygająca: `zs_pretrained_best = max(śr ZS pretrained-attrs11,
śr ZS pretrained-attrs21)`.

**LINIA (zamrożona): próg = 30%** (dokładnie ten sam próg falsyfikacji,
o który rozbiły się G2 i G2b — bezpośrednia porównywalność).

- **G3+ ⟺ `zs_pretrained_best > 30%`.** Kompozycyjny zero-shot osiągnięty
  na mocnych cechach → negatyw G2/G2b był w całości deficytem cech;
  dźwignia (c) potwierdzona jako THE wąskie gardło. (Raport: który
  słownik przekroczył i o ile.)
- **G3− ⟺ `zs_pretrained_best ≤ 30%` dla OBU słowników.** Kompozycyjność
  z ręcznego słownika + liniowych detektorów NIE przechodzi nawet na
  mocnych cechach → wąskim gardłem jest PODEJŚCIE, nie reprezentacja.
  Domyka serię G ostatecznie (kierunek: uczone atrybuty / inny paradygmat).

Werdykt główny projektu = ta jedna linia. Poniższe są OBOWIĄZKOWE testy
pomocnicze (raportowane niezależnie od G3+/G3−, nie zmieniają decyzji
binarnej):

- **T1 — dźwignia cech (nawet przy G3−):** pary per-seed pretrained vs
  random dla attrs11: SYGNAL+ (śr > próg szumu std+std, min par > 0)
  ORAZ śr ZS pretrained ≥ 2× random. Ranga: „G3− ze śladem dźwigni"
  vs „G3− twardy (cechy nie ruszają wyniku)".
- **T2 — reguła osiągalności na pretrained:** czy mocne cechy zmieniają,
  które klasy są osiągalne — w tym dawne porażki strukturalne
  {Sandal, Bag, AnkleBoot} = klasy 5/8/9. Per klasa.
- **T3 — interakcja a×c (attrs21 vs attrs11 na pretrained):** czy ECOC,
  który na losowych cechach szkodził (G2b: −2.98pp, backfire), POMAGA na
  mocnych. Pary per-seed: SYGNAL+ → dystans kodowy jest dźwignią, gdy
  błąd per-bit jest dość niski (potwierdza mechanizm G2b WPROST); SZUM/−
  → dystans nieistotny nawet przy dobrych cechach.
- **T4 — sanity reprodukcji:** warianty `random` odtwarzają G2b
  (attrs11 ~3.2%, attrs21 ~0.18%) — wzór precedensu G2b/J. Rozbieżność
  = błąd harnessu, blokuje interpretację.

## Przewidywanie (zapisane przed runem)

Pretrained daje SYGNAL+ vs random (spójne z powracającym w projekcie
motywem „cechy są wąskim gardłem": L +37pp, K „reszta luki reprezentacyj-
na") — czyli T1 spodziewane dodatnie. Otwarte, czy `zs_pretrained_best`
przekracza linię 30% (G3+) — obie ścieżki domykają: G3+ = dźwignia (c)
potwierdzona jako THE bottleneck; G3− mimo mocnych cech = granica
PODEJŚCIA (ręczny słownik + liniowe detektory), nie reprezentacji.
attrs21 (T3): przewidywana zmiana znaku względem G2b (z backfire na
neutralny/pomocny) w miarę spadku błędu per-bit.

## Plik, koszt, wynik

- Runner: `src/run_G3_pretrained_compositional.py` (nowy; wzór
  `run_G2b_ecoc.py`, `run_holdout` importowane, NIETKNIĘTE). Ekstraktor
  cech Fashion→ResNet18 (nowy helper lub w L-stylu). Backbone przełączany
  flagą; te same seedy dla obu.
- Wynik: `results/G3_pretrained.json` (smoke: `_smoke`).
- Koszt: pierwszy run — jednorazowa ekstrakcja cech Fashion@224 przez
  ResNet18 (rząd minut na 1050 Ti, potem cache); dalej liniowe projekcje
  (jak G2b: FULL 132 s). Smoke: 1 seed, 4 epoki, 1 holdout na wariant.
- Wymaga: Fashion-MNIST (auto), torchvision ResNet18 (jak L),
  brak GloVe (atrybuty binarne jak w G2).

## Instancjacja — RUNNER GOTOWY (2026-07-23, zielone światło Roberta)

`src/run_G3_pretrained_compositional.py` (nowy; kod G2/G2b/L NIETKNIĘTY):
- Ekstraktor `extract_or_load_fashion_feats` — Fashion 784 → denorm
  (0.2860/0.3530) → 3 kanały → resize 224 → norm ImageNet → zamrożony
  ResNet18 (bez fc) → 512-d, cache `data/fashion_resnet18_224_feats.pt`.
  Cechy deterministyczne (seed zmienia tylko uczenie pojęć).
- `run_holdout` importowane werbatim z G2; macierze `ATTRS11`/`ATTRS21`
  importowane z G2/G2b (jedno źródło prawdy); asercje własności macierzy
  przy starcie.
- Plan 2×2 na tych samych seedach; werdykt binarny G3+/G3- na
  `zs_pretrained_best` vs 30%; T1–T4 liczone i zapisane osobno.
- Wynik: `results/G3_pretrained.json` (smoke: `_smoke`).

Komendy (u Roberta): `python src/run_G3_pretrained_compositional.py --smoke`
→ `python src/run_G3_pretrained_compositional.py` (FULL; pierwszy run
robi jednorazową ekstrakcję cech Fashion@224 — potem cache).

## Zasady

5 seedów, progi jw. zamrożone, min-par raportowany, negatyw = wynik,
reprodukcja random jako sanity. Werdykty → DROGA_G_NOTATKI.md (dopisek
G3), potem CLAIMS/WHITEPAPER (sekcja 13 — G-seria).
