# Droga Q — notatki wyników (Q1: kolektyw na długim horyzoncie)

Run FULL: 2026-07-23, 391 s, seedy 0–4 (parowane z M1), CUDA.
Plik: `results/Q1_collective_horizon.json`. Plan: `DROGA_Q_PLAN.md`
(kryteria zamrożone przed runem). Smoke bez anomalii.

## WERDYKT GŁÓWNY: SYGNAL− — pierwsza zmierzona bariera skali protokołu

| System (CIFAR-100, 20 zadań, class-IL) | ACC | Forgetting |
|---|---|---|
| agent sekwencyjny (M1 m1_seq_300) | 40.70 ± 0.84% | 18.3pp |
| **kolektyw N=20 (95 klas z wiadomości 24 KB)** | **34.02 ± 0.65%** | **13.0pp** |
| sufit all-data (m1_all_300) | 47.41 ± 0.49% | — |

Pary: [−7.04, −7.26, −6.10, −5.42, −7.54], d = −6.67 ± 0.88pp,
próg 1.48pp → **SYGNAL−** (5/5 ujemnych). Obserwacja porównawcza
(inne dane, bez rangi werdyktu): koszt protokołu przy 10 klasach
wynosił −0.56pp (L2) — przy 100 klasach jest ~12× większy.
Luka do sufitu: 13.39pp. H-Q1 (koszt pozostaje mały) SFALSYFIKOWANA;
H-Q2 (adopcja płaci więcej) potwierdzona kierunkowo — ale w
NIEOCZEKIWANYM miejscu (patrz niżej).

## Q1b — struktura kosztu: ODWRÓCONY profil względem uczenia

Deficyt NIE jest późny (jak zakładała H-Q2) — jest WCZESNY:

| R[t][t] | early (zadania 1–5) | late (16–20) | ratio do sufitu per zadanie |
|---|---|---|---|
| seq (M1, uczenie) | 75.4–81.9% | 40.4–46.8% | early 1.572 / late 0.870 |
| kolektyw (adopcja) | 34.2–48.6% | 43.7–48.1% | early 0.797 / late **0.913** |

Pary spadku early→late (kolektyw − seq): [−43.8, −41.0, −42.2, −36.9,
−42.5] — 5/5, profil odwrócony. **Późne adopcje są równoważne późnemu
uczeniu (a względem sufitu nominalnie LEPSZE: 0.913 vs 0.870); cały
koszt protokołu koncentruje się we wczesnych adopcjach** (paczki 1–5:
~37% vs ~78% u seq — połowa poziomu). Forgetting kolektywu jest
NIŻSZY (13.0 vs 18.3pp): klasy adoptowane, chronione własnymi
statystykami-wiadomością, trzymają się lepiej — ale startują nisko.

Interpretacja (hipoteza, nie twierdzenie): wczesna projekcja kolektora
jest ukształtowana na zaledwie 5 realnych klasach — adopcja ze snów
musi osadzać nowe klasy w niedojrzałej projekcji; w miarę wzrostu
liczby klas adopcja dojrzewa do poziomu uczenia. To NIE jest
przestrzeń etykiet (ta uderza w obu identycznie — dekompozycja M1).

## Status i kandydaci Q2 (decyzja Roberta; osobna pre-rejestracja)

Q1 KOMPLET: bariera skali zmierzona (−6.67pp), zlokalizowana
(wczesna faza) i oprofilowana względem sufitu. Kandydaci:

- **Q2a — re-adopcja wczesnych paczek** (najmocniejszy): po komplecie
  19 adopcji zapomnij-i-adoptuj-ponownie paczki 1–5 (maszyneria I4b,
  payloady pierwszej generacji — bez rekursji snu, zgodne z O).
  Hipoteza: wczesne paczki w dojrzałej projekcji doszlusują do
  poziomu late (0.91 sufitu); przewidywany zysk rzędu +2–3pp ACC.
  Koszt: minuty. Kryterium: pary vs Q1, próg std+std.
- **Q2b — budżet snu adopcji** (dźwignia): n_dream 500→2500 przy
  adopcji (praca pamięci bez zmian — sen jest lokalny). Hipoteza:
  słabość wczesnych adopcji częściowo kompensowalna liczbą snów.
- Q2c (drożej): harmonogram — opóźnić adopcje do czasu dojrzałości
  projekcji (zmienia protokół; dopiero po a/b).

Rekomendacja: Q2a+Q2b w jednym runnerze (2 warianty × 5 seedów,
~15 min), pary vs wynik Q1.

---

# Q2 — WYNIKI (run FULL 2026-07-23, 1484 s, seedy 0–4)

Plan: `DROGA_Q2_PLAN.md` (przewidywania i progi zamrożone przed runem).
Plik: `results/Q2_early_repair.json`. Smoke bez anomalii.

## WERDYKTY: podwójny SYGNAL+ — bariera skali ZAMKNIĘTA Z NADDATKIEM

| Wariant | ACC | pary vs Q1 (34.02) | Werdykt | Odzysk bariery |
|---|---|---|---|---|
| q2a_readopt (naprawa paczek 1–5) | 38.11 ± 0.67% | +4.09 ± 0.65pp (5/5) | **SYGNAL+** | 61% |
| **q2b_dream2500 (budżet snu adopcji)** | **44.29 ± 0.66%** | **+10.26 ± 0.49pp (5/5)** | **SYGNAL+** | **154%** |

- **q2a**: przewidywanie +2–3pp PRZEKROCZONE (+4.09). Mechanizm
  potwierdzony wprost: finalne acc zadań 1–5 rośnie z ~26.4% do ~41.8%
  (pary +10.9…+17.7, 5/5) — wczesne paczki w dojrzałej projekcji
  doszlusowują do poziomu późnych. Hipoteza niedojrzałej projekcji
  POTWIERDZONA.
- **q2b**: samo podniesienie n_dream adopcji 500→2500 (payload BEZ
  zmian, 24.1 KB) daje +10.26pp — więcej niż cała bariera. Ryzyko
  saturacji payloadu (I2b) NIE zmaterializowało się po stronie
  odbiorcy: te same 24 KB statystyk niosą dość informacji, by wyśnić
  2500 próbek/klasę z zyskiem. Q1 był niedoborem BUDŻETU snu, nie
  informacji w wiadomości.

## Obserwacja domykająca — kolektyw NAD agentem sekwencyjnym

q2b vs m1_seq_300: **44.29 vs 40.70 — pary +3.59 ± 0.52pp (5/5),
formalnie SYGNAL+** (ranga: obserwacja — kierunek nie był
pre-rejestrowany jako twierdzenie). Luka do sufitu all-data 47.41
spada do ~3.1pp (93.4% sufitu — poziom L-owy mimo 100 klas
i 95 klas z wiadomości).

## UCZCIWOŚĆ PRZED HEADLINE — konieczna kontrola (Q2c)

Twierdzenie „kolektyw > sekwencyjny" ma confound: q2b trenuje projekcję
na 2500 próbkach/klasę nowej paczki (sen), a seq na 500 realnych
cechach/klasę. Zanim ogłosimy przewagę kolektywu, seq musi dostać
symetryczną dźwignię: **Q2c — seq z augmentacją snem własnych nowych
klas do 2500/klasę** (sample z własnych, świeżo policzonych statystyk;
pierwsza generacja, zgodne z O — statystyki liczone z realnych cech).
Możliwe wyniki: (a) seq NIE zyskuje → przewaga kolektywu realna
(materiał śniony > mały materiał realny przy tej projekcji);
(b) seq zyskuje porównywalnie → headline brzmi „RÓWNOWAŻNOŚĆ przy
właściwym budżecie snu; bariera skali była artefaktem budżetu" —
też mocny wynik. Obie wersje domykają serię Q.

Status: Q1+Q2 KOMPLET; Q2c do pre-rejestracji (decyzja Roberta).

---

# Q2c — WYNIKI (run FULL 2026-07-23, 910 s, seedy 0–4) — SERIA Q DOMKNIĘTA

Plan: `DROGA_Q2C_PLAN.md`. Plik: `results/Q2c_seq_selfdream.json`.

## Werdykt 1: budżet snu działa TEŻ dla uczenia — SYGNAL+

seq_selfdream2500 (500 realnych + 2000 snów własnych/klasę):
**45.35 ± 0.49%** vs m1_seq_300 40.70 ± 0.84 — pary +4.66 ± 0.52pp
(5/5, min +3.96). **Nowy najlepszy pojedynczy agent na 100 klasach:
95.7% sufitu all-data (luka 2.06pp).** Self-dream augmentation =
nowa zmierzona dźwignia sekwencyjnego uczenia (statystyki 1. generacji
z realnych cech; zgodne z O — sen wzbogaca materiał ŚWIEŻEJ klasy,
niczego nie odbudowuje po fakcie).

## Werdykt 2 — ROZSTRZYGNIĘCIE: RÓWNOWAŻNOŚĆ przy symetrycznych budżetach

q2b (kolektyw, 44.29) vs seq_selfdream2500 (45.35): pary
[−0.36, −1.13, −1.69, −0.46, −1.69], d = −1.07 ± 0.64pp, próg 1.12
→ **SZUM** (nominalnie −1.07, wszystkie pary ujemne, ale poniżej progu
i poniżej kryterium parowego). Obserwacja z Q2 „kolektyw +3.59 nad seq"
była artefaktem asymetrii budżetu — kontrola uczciwości rozstrzygnęła.

## HEADLINE SERII Q (final, wszystko pre-rejestrowane)

**Bariera skali protokołu była artefaktem budżetu snu.** Przy
symetrycznych budżetach (2500 próbek/klasę materiału projekcji):
kolektyw 20 agentów — 95 ze 100 klas wyłącznie z wiadomości 24.1 KB,
zero obrazów — jest RÓWNOWAŻNY agentowi sekwencyjnemu uczonemu na
wszystkich realnych danych (44.29 ± 0.66 vs 45.35 ± 0.49, SZUM;
93.4% vs 95.7% sufitu all-data). Twierdzenie I3/L2 (równoważność
kolektywu) rozszerzone z 10 na 100 klas i 20 agentów. Po drodze
zmierzone: bariera przy budżecie domyślnym (−6.67, Q1), jej
lokalizacja (wczesne adopcje, Q1b), naprawialność re-adopcją (61%,
Q2a) i pełna kompensacja budżetem (154%, Q2b) — oraz bonus: self-dream
augmentation jako dźwignia KAŻDEGO agenta (+4.66pp, Q2c).

Uwaga o forgettingu (obserwacja): seq_selfdream ma wyższy forgetting
(27.6 vs 18.3pp) przy wyższym ACC — augmentacja podnosi R[t][t]
(szczyty), z których spadek liczy się głębiej; ACC końcowe rozstrzyga.

Status serii Q: **KOMPLET** (Q1 SYGNAL− · Q1b obs. · Q2a SYGNAL+ ·
Q2b SYGNAL+ · Q2c SYGNAL+/SZUM-rozstrzygnięcie). Merge `droga-q` →
tag v1.2. Kandydat dalszy (osobna decyzja): kombinacja q2a+q2b
(re-adopcja przy budżecie 2500 — czy domyka lukę 1.07 do selfdream
i 3.1 do sufitu); nie blokuje merge'a.

---

# Q2d — WYNIKI (run FULL 2026-07-23, 1345 s, seedy 0–4)

Plan: `DROGA_Q2D_PLAN.md`. Plik: `results/Q2d_combo.json`.

## WERDYKT: SZUM — budżet subsumuje naprawę (anty-hipoteza potwierdzona)

combo 44.60 ± 0.60 vs q2b 44.29 ± 0.66: pary [+0.09, +0.41, +0.19,
−0.04, +0.07], d = +0.14 ± 0.17pp, próg 1.23 → **SZUM**. Dźwignie NIE
są addytywne. Mechanizm widoczny wprost w danych: przy budżecie 2500
wczesne zadania 1–5 osiągają ~49.5% JUŻ PRZED naprawą (przy budżecie
500 było 26.4% — wczesny deficyt Q1b był w całości deficytem budżetu);
re-adopcja dodaje na nich +1.56…+1.88pp (5/5, lokalnie spójne), co
w ACC całości rozpływa się w szumie.

Obserwacja domykająca: combo vs seq_selfdream2500 = pary −0.92 ± 0.61,
próg 1.07 → **SZUM** — luka do najlepszego pojedynczego agenta
formalnie domknięta (potwierdzenie równoważności z Q2c na trzecim
systemie). Luka do sufitu all-data: 2.98pp.

## OSTATECZNY STAN SERII Q (v1.2)

Trzy systemy w równoważności przy symetrycznych budżetach:
kolektyw-2500 44.29 ≈ kolektyw-combo 44.60 ≈ seq-selfdream 45.35
(wszystkie pary SZUM), na tle sufitu 47.41. Rekomendowana konfiguracja
protokołu: **sam budżet 2500** (najprostsza; re-adopcja zbędna poza
scenariuszem naprawy po ataku — I4b). Seria Q zamknięta bez wątków
wiszących.
