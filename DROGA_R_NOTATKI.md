# Droga R — kolektyw heterogeniczny — notatki

Plan (pre-rejestracja): `DROGA_R_PLAN.md`. Run: WYŁĄCZNIE u Roberta,
2026-07-23. Runner: `src/run_R1_heterogeneous.py` (poziom R-mild);
mechanizm: `src/mars_collective_hetero.py`. Wyniki:
`results/R1_heterogeneous.json`. 5 seedów × 15 epok, 4 warianty.

## R-mild v1 — WYNIK NIEROZSTRZYGAJĄCY (kontrola SANITY zadziałała)

| Wariant | adopted ACC | overall ACC |
|---|---|---|
| CEILING (homogeniczny = L2) | **73.44 ± 0.89%** | 74.13% |
| R0 (podłoga: sama kotwica) | 8.03 ± 2.10% | 21.62% |
| R1 (hetero + dekoder) | **0.42 ± 0.33%** | 19.37% |
| SANITY (R1, wspólny backbone) | **0.57 ± 0.62%** | 19.46% |

dec_mse (rekonstrukcja anchor→feature) ~0.03 — dekoder uczy się
przyzwoitej inwersji NA WŁASNYCH klasach.

Werdykty maszynowe (raportowane, ale patrz niżej — NIE są twierdzeniem):
R1 vs R0 SYGNAL− (−7.62pp); R1 vs CEILING SYGNAL− (−73.02pp);
**SANITY vs CEILING SYGNAL− (−72.87pp)**.

## Dlaczego to NIE jest twierdzenie o heterogeniczności

Pre-rejestracja (kryterium 3): „SANITY: R1 przy wspólnym backbone =
CEILING w szumie; inaczej błąd implementacji/mechanizmu **blokuje
interpretację 1–2**". SANITY zapadło do ~0.6% zamiast ~73% → kryterium
3 NIESPEŁNIONE. Zgodnie z własną zasadą: werdykty R1 vs R0 i R1 vs
CEILING są ZABLOKOWANE. Heterogeniczność NIE jest tu zmienną sprawczą —
SANITY nie ma żadnej heterogeniczności (ten sam backbone) i też pada.
To jest lekcja Q2c w czystej postaci: nie zgłaszać artefaktu
instancjacji jako wyniku naukowego.

## Diagnoza (mechanizm porażki, zmierzony)

Rdzeń: **dekoder anchor→feature uczony na 2 własnych klasach odbiorcy
NIE ekstrapoluje na nieznane rejony kotwicy.** Ścieżka SANITY (wspólny
backbone) obnaża to bez heterogeniczności:
1. Nadawca liczy Gaussa anchorowego klasy c → skupiony blisko kierunku
   `word_vec_c` (projekcja mapuje cechy c w okolicę słowa c).
2. Dekoder odbiorcy widział TYLKO rejony kotwic swoich 2 klas
   (inne słowa). Dostaje wejście blisko `word_vec_c` — rejon nigdy
   nie trenowany → **ekstrapolacja → pseudo-cechy śmieciowe** (mimo
   niskiego MSE na własnych klasach: MSE mierzy inwersję TAM, gdzie
   trenowano, nie w rejonie c).
3. `adopt_classes` trenuje projekcję I pody na tych śmieciach →
   **aktywnie psuje** routing (dlatego R1 0.42% < R0 8.03%: podłoga
   biernie ufa kotwicy, R1 czynnie uczy się śmieci; overall też spada
   19.4 vs 21.6).

Dodatkowo payload anchorowy (pojedynczy Gauss diag w 50-d po projekcji
128→50) niesie mało ponad samą tożsamość słowną — wąskie gardło
informacyjne nakłada się na problem ekstrapolacji dekodera.

To był ryzyko #1 zapisane w pre-rejestracji („dekoder wiele-do-jednego,
inwersja przybliżona, może nie oddać struktury; trenowany na własnych
klasach może nie generalizować") — zmaterializowało się już na R-mild.

## Wniosek i kierunek (R1b — do pre-rejestracji)

R-mild to miał być ŁATWY przypadek: feature_A i feature_B to dwa liniowe
obrazy tej samej 512, więc bliska-liniowa translacja ISTNIEJE z
konstrukcji. Zapadnięcie do ~0 znaczy, że **instancjacja (wąska kotwica
50-d + dekoder z 2 klas) nie wykorzystała dostępnej struktury** — nie że
koncept jest martwy. Zanim jakikolwiek werdykt R: **najpierw SANITY musi
przejść (≈ CEILING).**

Kandydaci na R1b (osobna pre-rejestracja, potem run u Roberta):
1. **Klasy kalibracyjne dzielone.** Założyć, że agenci dzielą kilka
   klas widzianych realnie (oba). Dekoder/translacja widzi wtedy WIELE
   rejonów kotwicy (nie 2) → ekstrapolacja maleje. Kontrola oracle:
   dekoder uczony na 10 klasach realnych — jeśli SANITY→CEILING, problem
   był czysto generalizacją dekodera (izoluje przyczynę).
2. **Translacja w przestrzeni CECH, nie 50-d kotwicy (dla R-mild).**
   Na klasach dzielonych ucz `feature_A → feature_B` wprost (Procrustes/
   regresja) — omija wąskie gardło 128→50→128. Wariant „sama kotwica bez
   klas dzielonych" zostaje jako TWARDSZA ablacja (ideał
   representation-agnostic).
3. **Bogatszy payload anchorowy:** k-komponentów zamiast 1 Gaussa;
   raczej wtórne wobec (1)/(2).

Status: R-mild v1 ZAMKNIĘTE jako nierozstrzygające (kontrola zadziałała).
CLAIMS/WHITEPAPER NIE dotknięte (brak ważnego twierdzenia). Następne:
pre-rejestracja R1b, gate = SANITY ≈ CEILING.

---

## R1b — BRAMKA NIEZDANA → SERIA R SFALSYFIKOWANA (poziom kotwica-interlingua)

Plan: `DROGA_R1B_PLAN.md`. Run u Roberta, 2026-07-23,
`results/R1b_translate.json`. Split: CAL=[0,1,2,3] dzielone, OWN=[4,5],
ADOPT=[6,7]+[8,9] (metryka). 5 seedów.

| Wariant | adopted ACC | overall |
|---|---|---|
| CEILING (homogeniczny, payload cech) | **81.62 ± 1.09%** | 77.08% |
| R0 (podłoga anchor-only) | 9.80 ± 3.62% | 34.43% |
| ORACLE_SANITY (dekoder MLP na 10 realnych, wspólny bb) | **4.00 ± 2.04%** | 27.83% |
| ORACLE_HET | 2.25 ± 0.62% | 26.79% |
| RBF_SANITY (kernel ridge, wspólny bb) — **BRAMKA** | **4.82 ± 2.31%** | 28.46% |
| RBF_HET | 3.00 ± 1.29% | 27.31% |
| RIDGE_HET (ablacja liniowa) | 1.35 ± 0.45% | 26.08% |

**BRAMKA (zamrożona ≥65%): RBF_SANITY = 4.82% → NIEZDANA (o rząd
wielkości).** Zgodnie z pre-rejestracją: **seria R na poziomie
kotwica-interlingua SFALSYFIKOWANA i NIE wchodzi do CLAIMS.md.**

### Dlaczego — przyczyna zizolowana ostatecznie (Krok 1 zadziałał)

**ORACLE_SANITY = 4.00%** rozstrzyga: to NIE jest ani generalizacja
dekodera, ani heterogeniczność. Oracle ma PEŁNĄ wiedzę (dekoder uczony
na 10 realnych klasach, MSE ~0.023 — inwersja działa), SANITY ma ZERO
heterogeniczności (wspólny backbone). Mimo to re-materializacja zapada
do 4% vs sufit 81.6%. **Wąskim gardłem jest sam payload anchorowy —
50-d kotwica jest z konstrukcji „class-collapsed".**

Mechanizm: projekcja jest trenowana tak, by cechy klasy c mapowały się
CIASNO w kierunek `word_vec_c` (CE po cosinusie). Więc znormalizowany
embedding anchorowy klasy c to niemal punkt (mała wariancja) ≈ słowo c.
Gauss anchorowy zwija klasę do jednego punktu — traci CAŁĄ wewnątrz-
klasową geometrię cech. Nawet doskonały dekoder odtworzy z tego jeden
punkt w przestrzeni cech, nie rozkład. Adopcja uczy projekcję i pody na
zdegenerowanej „klasie-punkcie" → nie rozróżnia jej od sąsiadów → 4%.
Kotwica niesie TOŻSAMOŚĆ klasy (które słowo), nie jej GEOMETRIĘ.

Potwierdzenia spójności:
- RBF_SANITY (4.82%) ≈ ORACLE_SANITY (4.00%) — jakość translatora
  prawie nie zmienia wyniku → wąskie gardło jest UPSTREAM (payload),
  nie w translatorze.
- **R0 (9.80%) > wszystkie warianty re-materializacji** — samo zaufanie
  kotwicy bije re-materializację; „wyrafinowany" mechanizm AKTYWNIE
  szkodzi (uczy proj/pody na zdegenerowanych cechach; overall też spada
  poniżej R0 34%).
- Monotonicznie: RIDGE < RBF < ORACLE ≈ próg-floorowy; spójne 5/5.

### Wniosek i (opcjonalny) następny poziom abstrakcji

Wspólna przestrzeń słów NIE jest wystarczającą interlingua do wymiany
snów między heterogenicznymi agentami — nie z powodu heterogeniczności
czy dekodera, lecz dlatego, że jest to przestrzeń ROUTINGU (nisko-
wymiarowa, zwinięta per klasa), nie przestrzeń REKONSTRUKCJI. Twierdzenie
graniczne domknięte. Zgodnie z bramką: brak wpisu do CLAIMS.

Jeśli Robert zechce ciągnąć R — kierunek zdiagnozowany, ale to INNY
poziom abstrakcji (osobna pre-rejestracja, ustępstwo od „representation-
agnostic przez same słowa"):
- **R2 / translacja w przestrzeni CECH:** na klasach dzielonych ucz
  `feature_A → feature_B` wprost (Procrustes/regresja), wysyłaj payload
  CECH (spike-and-slab jak w I), transluj do przestrzeni odbiorcy.
  Zachowuje wewnątrzklasową geometrię (omija wąskie gardło 50-d). Cena:
  wymaga korespondencji cech (klasy dzielone), więc nie jest już „tylko
  słowa". Przewidywanie: to jedyna droga zgodna z ORACLE_SANITY=4%.
- Tani wariant-łata przed R2: payload z NIEznormalizowanego wyjścia
  projekcji (więcej wariancji) — mało prawdopodobne, że ratuje (50-d
  bottleneck zostaje), ale koszt ~0. Do decyzji.

Status: seria R zamknięta na poziomie kotwica-interlingua (SFALSYFIKOWANA,
poza CLAIMS). Kod: `mars_translate.py`, `run_R1b_translate.py`,
`mars_collective_hetero.py`. Dalej wyłącznie za decyzją Roberta.

---

## R2 — WYRÓWNANIE CECH (PROCRUSTES) — BRAMKA ZDANA, R WRACA DO GRY

Plan: `DROGA_R2_PLAN.md`. Run u Roberta, 2026-07-23,
`results/R2_procrustes.json`. Runner: `src/run_R2_procrustes.py`.
Split jak R1b. 5 seedów.

| Wariant | adopted ACC | overall | disparity |
|---|---|---|---|
| CEILING (homogeniczny) | **81.62 ± 1.09%** | 77.08% | — |
| R0 (podłoga anchor-only) | 9.80 ± 3.62% | 34.43% | — |
| **R2_SANITY** (Procrustes, wspólny bb) | **80.15 ± 1.31%** | 76.60% | 0.0 |
| R2_HET (Procrustes, hetero) | 31.29 ± 8.31% | 45.65% | 0.056 |
| **RIDGE_HET** (mapa liniowa, hetero) | **65.05 ± 3.29%** | 67.59% | 0.032 |

**BRAMKA R2_SANITY = 80.15% ≥ 65% → ZDANA.** Translacja w przestrzeni
cech jest zdrowa — maszyneria NIE niszczy informacji (SANITY ≈ CEILING,
Ω = I przy wspólnym backbone, disparity=0). **To potwierdza diagnozę
R1b:** problemem była 50-d kotwica (class-collapsed), nie sama idea
translacji. Feature-space działa.

### Wynik pre-rejestrowany (Procrustes) i kluczowa ablacja

- **R2_HET vs R0: SYGNAL+ (+21.49pp)** — wyrównanie cech niesie REALNĄ
  informację ponad podłogę anchorową. Heterogeniczny transfer działa.
- **R2_HET vs CEILING: SYGNAL− (−50.33pp)** — ale ortogonalny Procrustes
  płaci dużą cenę, przy wysokiej wariancji (±8.31, min 17.13%).
- **RIDGE_HET vs R2_HET: SYGNAL− (−33.76pp na korzyść RIDGE)** — **mapa
  LINIOWA (ridge) bije ortogonalny Procrustes o +33.76pp**, osiągając
  **65.05% = 80% sufitu** (81.62), przy niskiej wariancji (±3.29).

### Mechanizm (dlaczego ortogonalność szkodzi)

R-mild: H_A = ReLU(R_A · f512), H_B = ReLU(R_B · f512), gdzie R_A, R_B to
RÓŻNE losowe rzuty 512→128. Relacja H_A→H_B to ogólna mapa liniowa
(≈ R_B R_A⁺), NIE izometria — ma anizotropowe skalowanie i ścinanie.
Ortogonalny Procrustes wymusza obrót+odbicie (zachowanie norm/kątów),
więc NIE MOŻE oddać skalowania → duży rezyduum na strukturze klas mimo
niskiego disparity ogólnego. Ridge (pełna mapa liniowa) łapie to i
odzyskuje 80% sufitu. **Głównym kosztem jest ograniczenie ortogonalności,
nie heterogeniczność.**

### Status i następny krok (R2b — do pre-rejestracji)

To pierwszy POZYTYWNY wynik heterogenicznego kolektywu: różne backbone'y
dzielą klasy przez wyrównanie cech na 4 klasach kalibracyjnych,
odzyskując 80% sufitu homogenicznego. ALE headline „mapa liniowa 80%" to
ABLACJA, nie pre-rejestrowany primary — **zgodnie z dyscypliną Q2c NIE
wpisuję do CLAIMS bez potwierdzenia.** CLAIMS/WHITEPAPER NIETKNIĘTE.

Pre-rejestracja **R2b** (primary = wyrównanie liniowe, z kontrolami):
1. Ridge/liniowa mapa jako PRIMARY, świeże seedy; próg z góry (np.
   R2b_HET ≥ 70% sufitu, czyli ≥ ~57pp) — potwierdzenie, że 65% to nie
   artefakt seedów/CAL.
2. Kontrola przecieku: mapa fitowana WYŁĄCZNIE na CAL (rozłączne z
   adoptowanymi — już tak jest); sweep K (2/4/6 klas dzielonych) — ile
   kalibracji naprawdę trzeba.
3. Czy luka do sufitu (−16.6pp) jest nieredukowalna (utrata info w
   512→128 + ReLU) czy domykalna większą kalibracją / regularyzacją.
4. Po zdanym R2b (R-mild): **R-hard** (losowy ↔ pretrained, RÓŻNE
   wymiary → prostokątna mapa liniowa ridge) — właściwy test rewolucji.

Kod: `mars_translate.py` (ProcrustesAlign, RidgeTranslator),
`mars_collective_hetero.py` (adopt_classes_maptransform),
`run_R2_procrustes.py`. Dalej za decyzją Roberta.

---

## R2b — WYRÓWNANIE LINIOWE POTWIERDZONE — PIERWSZY POZYTYWNY HETEROGENICZNY KOLEKTYW

Plan: `DROGA_R2B_PLAN.md`. Run u Roberta, 2026-07-23,
`results/R2b_linear.json`. Runner: `src/run_R2b_linear.py`. **Świeże
seedy 5–9** (nie 0–4 z R2 — potwierdzenie, nie samopotwierdzanie).
Split: CAL_POOL=[0–5] (6 dzielonych, uczone), mapa na K∈{2,4,6};
OWN=[6,7]; ADOPT=[8,9] (metryka). Auto-λ z gridu {1e-3..10}.

| Wariant | adopted ACC |
|---|---|
| CEILING (homogeniczny) | **86.87 ± 1.09%** |
| R0 (podłoga anchor-only) | 4.69 ± 1.96% |
| R2b_SANITY_K4 (liniowa, wspólny bb) | 84.77 ± 1.01% |
| **R2b_HET_K4 (liniowa, hetero)** | **79.98 ± 2.45%** |
| R2b_HET_K2 / K6 | 81.12 / 78.09% |

**BRAMKA R2b_HET_K4 = 79.98% ≥ 60.81% (70% sufitu) → ZDANA
[0.921 sufitu].** Przekroczona z ogromnym zapasem na świeżych seedach.
**Wyrównanie liniowe POTWIERDZONE.**

### Wynik: heterogeniczny kolektyw działa

- **R2b_HET vs R0: SYGNAL+ (+75.29pp)** — różne backbone'y dzielą klasy;
  transfer masywny ponad podłogę.
- **R2b_HET vs CEILING: SYGNAL− (−6.89 ± 2.33pp)** — MAŁA, zmierzona cena
  heterogeniczności (~8% względnie). Kolektyw odzyskuje **92% sufitu
  homogenicznego** mimo RÓŻNYCH backbone'ów.
- **R2b_SANITY ≈ CEILING** (84.8 vs 86.9) — maszyneria zdrowa (Ω-liniowa
  przy wspólnym bb, disparity=0).

### Krzywa K (falsyfikowalna obserwacja): K=2 wystarcza

K2=81.12 > K4=79.98 > K6=78.09 — **łagodny SPADEK** (przewidywanie
wzrostu OBALONE, ale na korzyść metody): **już 2 klasy kalibracyjne
wystarczają** do wyrównania (93% sufitu), więcej lekko szkodzi
(prawdopodobnie mapa rozprasza się na szerszy region kosztem precyzji na
klasach adoptowanych). Minimalny budżet kalibracji — mocny wynik
praktyczny. Auto-λ stabilnie wybiera λ=1.0 (K2) → 0.1 (K6).

### Domknięcie łuku serii R

Kotwica-interlingua sfalsyfikowana (R1b, ORACLE 4%) → diagnoza „transluj
w cechach" → feature-space zdrowe (R2 gate) → ortogonalność zły constraint
(R2, Procrustes 31%) → **mapa liniowa potwierdzona (R2b, 92% sufitu,
świeże seedy)**. Relacja H_A→H_B to ogólna mapa liniowa, nie izometria;
wyrównanie na 2–4 klasach dzielonych wystarcza. To PIERWSZY pozytywny
transfer między RÓŻNYMI reprezentacjami w protokole — kolektyw
representation-agnostic zrealizowany na poziomie R-mild.

**CLAIMS 42; WHITEPAPER (rozszerzenie sekcji kolektywu).** Następne:
**R-hard** (losowy ↔ pretrained, RÓŻNE wymiary 128 vs 512 → prostokątna
mapa ridge; reżimy zasobów rozdzielone) — właściwy test rewolucji;
osobna pre-rejestracja `DROGA_R_HARD_PLAN.md`.

---

## R-hard — KOLEKTYW MIĘDZY RÓŻNYMI REPREZENTACJAMI — BRAMKA ZDANA (POZYTYW)

Plan: `DROGA_R_HARD_PLAN.md` (X=50%, zakres (b) zatwierdzone 2026-07-24).
Run u Roberta, 2026-07-24, `results/R_hard.json`. Runner: `src/run_R_hard.py`;
helpery OBOK rdzenia w `mars_collective_hetero.py` (`feature_payload_from_feats`,
`paired_calib_feats`, `feats_through_front`, `_identity`). Rdzeń I/L i
`adopt_classes` NIETKNIĘTE. Świeże seedy 5–9, K∈{2,4,6}, auto-λ {1e-3..10}.

Krzyżujemy backbone **LOSOWY-OD-ZERA** (`CifarBackbone`, surowe piksele
3072→128) z **PRETRENOWANYM resnet18** (cache 512-d) — RÓŻNA treść ORAZ różne
wymiary. Prostokątna mapa ridge H_A(D_A)→H_B(D_B). Odbiorca zawsze 128-d
(rdzeń hardkoduje BB_H=128 w proj/podach); nadawca 512 eksportuje payload
stats-only (bez proj/podów). Kierunki ROZDZIELONE; metryka = adopted ACC
względem LOKALNEGO sufitu odbiorcy w jego reżimie.

| Wariant | adopted ACC | % sufitu odbiorcy |
|---|---|---|
| CEILING_S (scratch, homo) | 55.83 ± 1.64% | — |
| R0_S (scratch, floor) | 5.37 ± 3.30% | — |
| CEILING_PT (reduced, homo) | 86.87 ± 1.09% | — |
| R0_PT (reduced, floor) | 4.69 ± 1.96% | — |
| **SANITY_RECT_K4** (512→128, mapa z konstrukcji) | **81.18 ± 1.23%** | **93.4%** |
| **HET_FS_K4** (foundation→scratch, PRIMARY) | **42.28 ± 2.73%** | **75.7%** |
| HET_SF_K4 (scratch→reduced, asymetria) | 38.72 ± 5.50% | 44.6% |

**BRAMKA PRIMARY: HET_FS_K4 42.28% ≥ 27.91% (50%·CEILING_S) → ZDANA
[0.757 sufitu].** **SANITY: 81.18% ≥ 56.47% (65%·CEILING_PT) → ZDANA
[0.934].** Warunek konieczny **HET_FS vs R0_S = SYGNAL+** (+36.91pp, min
+32.9 > szum 6.04, 5/5). → Kolektyw representation-agnostic działa TAKŻE
między RÓŻNYMI reprezentacjami.

### Wynik: transfer między różną TREŚCIĄ i różnym WYMIAREM
- **HET_FS vs R0_S: SYGNAL+ (+36.91pp, 5/5)** — prostokątna mapa niesie
  realną informację ponad podłogę anchorową; foundation→scratch działa.
- **HET_FS vs CEILING_S: SYGNAL− (−13.55 ± 2.34pp, 5/5)** — zmierzona CENA
  różnicy reprezentacji: odzysk 76% sufitu odbiorcy. WIĘKSZA niż w R-mild
  (R2b −6.9pp / 92%) — R-hard trudniejszy (brak wspólnej bazy cech),
  zgodnie z przewidywaniem z pre-rejestracji.
- **SANITY_RECT ≈ 93% sufitu** (disp ~0.004) — maszyneria prostokątna (ridge
  512→128 + parowanie dwóch frontów na tych samych obrazach) zdrowa. Smoke
  wykazał collapse (2.55%): zdiagnozowany jako GŁÓD PRÓBEK payloadu 512-d
  (n_dream 256 → 16 klastrów spike-and-slab w 512-d), NIE bug — `--diag`
  (n_dream 5000) odzyskał 82% i potwierdził. (Hipoteza „różne instancje
  ReducedBackbone" obalona kodem: `col.backbone` to jeden zamrożony obiekt
  w kalibracji, adopcji i ewaluacji.)

### Krzywa K (przewidywanie ZAPISANE i POTWIERDZONE): F→S ROSNĄCA
HET_FS: K2 40.42 < K4 42.28 < K6 42.47 — **monotonicznie ROSNĄCA**,
nasycenie K4–K6. Odwrotnie niż R-mild (tam K=2 wystarczał, krzywa malejąca):
trudniejszą, między-treściową mapę lepiej constrainuje więcej klas
kalibracyjnych. Falsyfikowalne przewidywanie z pre-rejestracji potwierdzone.

### Asymetria kierunku (OSOBNO, z zastrzeżeniem kształtu mapy)
DIR-S→F (scratch→reduced, KWADRATOWY 128→128): HET_SF_K4 38.72% = 44.6%
sufitu-reduced; vs R0_PT SYGNAL+ (+34.03pp, 5/5), vs CEILING_PT SYGNAL−
(−48.15pp). disp ~0.06 (mapa słaba — losowe cechy CNN nie rekonstruują
bogatej bazy pretrained). **Krzywa K MALEJĄCA i to stromo** (K2 53.63 >
K4 38.72 > K6 27.42) — PRZECIWNIE niż F→S. UWAGA (zastrzeżenie C planu):
F→S jest prostokątny (512→128) + rich→poor, S→F kwadratowy (128→128) +
poor→rich — asymetria MIESZA kierunek treści z kształtem mapy; to
OBSERWACJA, nie twierdzenie. Metryka względem sufitu odbiorcy: do słabego
scratcha odzyskujemy 76%, do bogatego reduced 45%.

### Domknięcie łuku serii R
kotwica-interlingua sfalsyfikowana (R1b, ORACLE 4%) → feature-space zdrowe
(R2 gate) → ortogonalność zły constraint (R2, 31%) → mapa liniowa (R2b, 92%,
R-mild) → **prostokątna mapa między RÓŻNYMI reprezentacjami (R-hard, 76%
sufitu, foundation↔scratch, różna treść i wymiar)**. Kolektyw
representation-agnostic zrealizowany także dla genuinely różnych baz cech —
z większą, zmierzoną ceną niż R-mild. **CLAIMS 43; WHITEPAPER §15
(rozszerzenie).** Seria R domknięta na poziomie R-hard; dalej za decyzją
Roberta.
