# Droga R2b — wyrównanie liniowe jako primary + kontrole (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: **DO ZATWIERDZENIA** (plan
zatwierdzony przez Roberta 2026-07-23; runner po tej pre-rejestracji);
runy WYŁĄCZNIE u Roberta. Branch: `droga-r` (nowe pliki, rdzeń I/L i
`adopt_classes` NIETKNIĘTE). Oś: **Part III (kolektyw, memory without
data)** — potwierdzenie pierwszego pozytywnego heterogenicznego transferu.

## Punkt wyjścia (zmierzony, R2)

R2 (Procrustes) zdał bramkę SANITY (80.15% ≈ CEILING 81.62% → translacja
w przestrzeni cech ZDROWA, naprawia R1b). Ortogonalny Procrustes okazał
się złym narzędziem (R2_HET 31.29%, −50pp vs sufit), a **ablacja liniowa
RIDGE_HET dała 65.05% = 80% sufitu, bijąc Procrustes o +33.76pp**.
Mechanizm: relacja H_A→H_B (R-mild) to ogólna mapa liniowa R_B R_A⁺,
NIE izometria — ortogonalność nie oddaje skalowania.

**Dyscyplina Q2c:** headline „liniowa 80%" to była ABLACJA, nie
pre-rejestrowany primary. R2b promuje mapę liniową do primary i
POTWIERDZA ją na ŚWIEŻYCH seedach z twardą bramką, zanim wejdzie do
CLAIMS.

## Hipoteza

H-R2b: uregularyzowana mapa liniowa H_A→H_B, fitowana na K klasach
dzielonych z automatycznym doborem λ, umożliwia heterogenicznemu
kolektywowi odzyskać ≥70% sufitu homogenicznego — i wynik replikuje się
na świeżych seedach (nie jest artefaktem seedów 0–4 z R2).

## Mechanizm (ZAMROŻONY) — primary = Ridge

- **Mapa:** `RidgeTranslator` (liniowa, zamknięta forma, z biasem,
  wyjście clamp_min 0), H_A(128) → H_B(128), fitowana na parach
  (H_A^cal, H_B^cal) — te same obrazy kalibracyjne przez oba backbone'y.
- **Auto-λ (nowe wg zlecenia Roberta):** λ dobierana z zamrożonego gridu
  **{1e-3, 1e-2, 1e-1, 1, 10}** przez rekonstrukcję na HELD-OUT próbkach
  kalibracyjnych (split 80/20 wewnątrz klas dzielonych; min MSE
  ‖map(H_A^val) − H_B^val‖). Zero wglądu w klasy adoptowane.
- **Adopcja:** payload cech klasy w H_A (jak w I) → śnij H_A → `map` →
  H_B → `adopt_classes_maptransform` → istniejące `adopt_classes`.
  (Ścieżka `RIDGE_HET` z R2, teraz jako primary.)

## Split (ZAMROŻONY) — umożliwia sweep K∈{2,4,6}

CIFAR-10, 10-way class-IL (parytet skali z R2, sufit ~80%):
- **CAL_POOL = [0,1,2,3,4,5]** — 6 klas dzielonych, uczonych realnie przez
  WSZYSTKICH agentów (część `seen`); mapa fitowana na PIERWSZYCH **K** z
  nich (K∈{2,4,6}). Wszystkie 6 uczone niezależnie od K → collector ma
  stałą projekcję; zmienia się TYLKO budżet kalibracyjny mapy.
- **OWN = [6,7]** — własne odbiorcy (realne).
- **ADOPT = [8,9]** — adoptowane od nadawcy przez mapę; metryka
  (recipient-relative, row_class_il dla [8,9]).
- Kalibracja per-próbka: te same obrazy CAL przez backbone nadawcy i
  odbiorcy → pary do fitu mapy (jak R2).

**Świeże seedy: 5–9** (precedens P1c — nie te, na których zaobserwowano
ablację w R2). 5 seedów.

## Warianty (te same świeże seedy 5–9)

- **CEILING** — homogeniczny, payload cech (real, ten sam backbone).
  Sufit adopcji [8,9] w tym splicie.
- **R0** — hetero, podłoga anchor-only (odniesienie).
- **R2b_SANITY** — homogeniczny, mapa liniowa (kontrola maszynerii;
  ≈ CEILING oczekiwane).
- **R2b_HET** — heterogeniczny R-mild, mapa liniowa. **PRIMARY.**

Każdy wariant liniowy przez sweep **K∈{2,4,6}**; auto-λ per fit.

## Kryteria werdyktu (Z GÓRY) — TWARDA BRAMKA

1. **BRAMKA PRIMARY (zamrożona):** **R2b_HET przy K=4 ≥ 70% × CEILING**
   (≈ 57% adopted ACC, przy sufi­cie ~81%) na świeżych seedach 5–9 →
   **wyrównanie liniowe POTWIERDZONE**; heterogeniczny kolektyw działa
   → wejście do CLAIMS/WHITEPAPER jako pierwszy pozytywny transfer
   representation-agnostic (R-mild). **< 70% → niepotwierdzone**: obserwacja
   R2 (65%) była artefaktem seedów/CAL → poza CLAIMS, R-mild zamknięte.
2. **Sanity maszynerii:** R2b_SANITY ≥ 65% (Ω-liniowa przy wspólnym bb
   ≈ CEILING); rozbieżność blokuje interpretację.
3. **Po zdanej bramce:**
   - R2b_HET vs CEILING: zmierzona CENA heterogeniczności (pary per-seed).
   - R2b_HET vs R0: SYGNAL+ (mapa niesie informację ponad podłogę).
   - **Krzywa K (2/4/6):** ile klas kalibracyjnych naprawdę trzeba;
     czy K=2 wystarcza, czy K=6 domyka lukę do sufitu. Obserwacja
     falsyfikowalna: monotoniczność ACC(K).

Progi (bramka 70% sufitu, sanity 65%) i grid λ zamrożone PRZED runem.
Min-par raportowany. Negatyw (w tym trafienie bramki) = wynik.

## Przewidywanie (zapisane przed runem)

R2b_HET(K=4) ≈ 65% (replikacja R2) → bramka zdana. K-krzywa rosnąca
(więcej kalibracji → lepsze wyrównanie), z nasyceniem; K=2 może nie
wystarczyć. Luka do sufitu (~−16pp) prawdopodobnie częściowo
nieredukowalna (utrata info 512→128 + ReLU) — R2b to zmierzy, nie domknie.

## Plik, koszt, ryzyko

- Runner: `src/run_R2b_linear.py` (nowy; wzór `run_R2_procrustes.py`,
  sweep K + auto-λ). `RidgeTranslator`, `adopt_classes_maptransform`,
  cache cech z L — istnieją, NIETKNIĘTE.
- Wynik: `results/R2b_linear.json` (smoke: `_smoke`).
- Koszt: niski (ridge zamknięty; 5 seedów × 3 K × 4 warianty; wzór R2
  FULL 182 s → rząd kilku–kilkunastu min). Smoke: 1 seed, K=4.
- Ryzyko: bramka może nie trafić na świeżych seedach (65% było blisko
  progu 57% z zapasem, więc raczej zda) — falsyfikowalne, uczciwe.

## Po R2b (jeśli bramka zdana)

R-hard: losowy ↔ pretrained, RÓŻNE wymiary (128 vs 512) → prostokątna
mapa ridge H_A(D_A)→H_B(D_B); osobna pre-rejestracja — właściwy test
rewolucji representation-agnostic (from-scratch ↔ foundation, metryka
względem sufitu odbiorcy, reżimy rozdzielone).

## Instancjacja — RUNNER GOTOWY (2026-07-23, zielone światło Roberta)

`src/run_R2b_linear.py` (nowy; `RidgeTranslator`,
`adopt_classes_maptransform`, cache L — reużyte, rdzeń NIETKNIĘTY):
- Split zapięty: CAL_POOL=[0–5] (uczone), mapa na pierwszych K∈{2,4,6};
  OWN=[6,7]; ADOPT=[8,9] (metryka = row[1] w eval [OWN,ADOPT]).
- Świeże seedy 5–9. Auto-λ z gridu {1e-3,1e-2,1e-1,1,10} po held-out
  (80/20) rekonstrukcji na klasach kalibracyjnych.
- Warianty: CEILING, R0, R2b_SANITY_K{2,4,6}, R2b_HET_K{2,4,6}.
- Bramka liczona automatycznie: R2b_HET_K4 ≥ 0.70·CEILING; raportuje
  ZDANA/NIEZDANA + ułamek sufitu; krzywa ACC(K); werdykty tylko po
  zdanej bramce.

Komendy (u Roberta): `python src/run_R2b_linear.py --smoke` (1 seed, K=4)
→ `python src/run_R2b_linear.py` (FULL, seedy 5–9, K∈{2,4,6}). Wynik:
`results/R2b_linear.json`.

## Zasady

5 świeżych seedów (5–9), bramka 70% sufitu i grid λ zamrożone PRZED
runem, negatyw = wynik. Werdykty → DROGA_R_NOTATKI.md (dopisek R2b),
CLAIMS/WHITEPAPER TYLKO po zdanej bramce. Runnera NIE piszemy przed
zatwierdzeniem tego planu przez Roberta.
