# Droga F — notatki robocze

Plan: `DROGA_F_PLAN.md`. Kontekst: pivot na CL 06.07.2026 (decyzja po E4).

## F0 — Baseline'y CL (ZAKOŃCZONE, 06.07.2026)

Pliki: `src/cl_common.py`, `src/run_F0_cl_baselines.py`;
wyniki: `results/F0_cl_baselines.json`. Split 5 zadań × 2 klasy,
5 seedów × 15 epok/zadanie, MonoS2 (backbone S2 + głowica 10-way).

**class-IL (główny protokół):**

| Metoda | Fashion ACC | Fashion F | MNIST ACC | MNIST F |
|---|---|---|---|---|
| finetune | 17.96 ± 4.47% | 96.7pp | 19.77 ± 0.09% | 99.9pp |
| EWC λ=100 | 18.53 ± 3.18% | 96.0pp | 19.70 ± 0.05% | 99.9pp |
| EWC λ=1000 | 18.13 ± 4.08% | 94.2pp | 19.79 ± 0.06% | 99.9pp |
| **replay-200 (cel F1)** | **76.97 ± 1.09%** | **27.0pp** | **88.81 ± 1.06%** | 13.5pp |
| joint (sufit) | 90.37 ± 0.84% | — | 98.99 ± 0.13% | — |

**Obserwacje:**
1. Katastrofa potwierdzona z chirurgiczną czystością: finetune i EWC ≈ 20%
   = accuracy ostatniego zadania / 5. Sieć pamięta TYLKO ostatnie zadanie.
2. **EWC w class-IL nie działa W OGÓLE** (≈ finetune) — zgodne z literaturą
   i z lekcją Etapu 2; regularyzacja wag nie chroni przy współdzielonej
   głowicy i przesuwającym się rozkładzie klas.
3. Replay-200 to poważny przeciwnik: 77% Fashion / 89% MNIST przy buforze
   ledwie 200 próbek. Task-IL replay ~98.7% — przy znanym zadaniu problem
   praktycznie znika; CAŁA trudność siedzi w class-IL (wybór zadania).
4. Luka replay→joint: 13.4pp (Fashion) — przestrzeń, o którą gra się w CL.
5. Implementacyjna lekcja: replay wymaga próbkowania bufora per krok
   (konkatenacja = bufor tonie; naprawione po smoke, commit w run_F0).

**Cel dla F1 (pre-rejestrowany w run_F1):** class-IL ACC ≥ replay-200
(Fashion ~77%) bez żadnego bufora = SYGNAL+ "architektura zamiast pamięci".

## F1 — MARS-CL (ZAKOŃCZONE, 06.07.2026): PONIŻEJ REPLAY, ale z odkryciem

Pliki: `src/mars_cl.py`, `src/run_F1_mars_cl.py`; wyniki: `results/F1_mars_cl.json`.
5 seedów × 15 epok/zadanie. Prawo D5 jako architektura (backbone+proj frozen).

**class-IL:**

| Wariant | Fashion ACC | MNIST ACC | Forgetting (F/M) |
|---|---|---|---|
| F1a (backbone z task0) | 36.11 ± 2.46% | 34.13 ± 0.96% | 15.0 / 12.8pp |
| F1a-l (uczone protos) | 36.26 ± 2.18% | 33.67 ± 1.04% | 24.9 / 16.5pp |
| F1c (task0+1; diagnostyka) | 49.24 ± 1.69% | 61.68 ± 0.83% | 12.2 / 12.8pp |
| **F1d (LOSOWY backbone)** | **60.19 ± 5.19%** | **66.23 ± 2.73%** | 16.0 / 10.9pp |
| [F0] replay-200 (cel) | 76.97% | 88.81% | 27.0 / 13.5pp |

**WERDYKT (pre-rejestrowany): PONIŻEJ REPLAY** (Fashion −16.8pp, MNIST −22.6pp)
— granica podejścia zmierzona. Sanity vs finetune: +42/+46pp ✓. MAC ×1.0007 ✓.

**Kluczowe obserwacje:**
1. **GŁÓWNE ODKRYCIE — losowy backbone bije trenowany o 24–32pp** (F1d vs
   F1a), a nawet backbone z 4 klas (F1c) NIE dogania losowego. Trzecie
   potwierdzenie centralnego prawa projektu (po D5 i E2-v1): **wąsko uczona
   wspólna reprezentacja jest gorsza niż żadna** — trening na 2 klasach
   specjalizuje cechy i zabija ich ogólność. Publikowalna obserwacja
   sama w sobie.
2. **Architektura tłumi zapominanie zgodnie z projektem:** F ~10–16pp
   (replay: 27pp na Fashion) przy STAŁYM koszcie inferencji (×1.0007
   po 5 zadaniach). Deficyt NIE jest w pamięci — jest w bezwzględnym
   poziomie cech (task-IL F1d ~80–88% vs replay ~98.7%: nawet wewnątrz
   zadania losowe cechy ograniczają i pody, i routing).
3. Uczone prototypy ≈ class-mean, ale z gorszym forgettingiem (24.9 vs
   15.0pp) — NCM zostaje domyślne.
4. **Diagnoza dla F2:** cała gra o ~17–23pp do replay toczy się w jakości
   ZAMROŻONEJ reprezentacji ogólnej. Kierunki z danych: (a) szersze losowe
   cechy — losowość jest darmowa i nieobciążona, płacimy tylko MAC;
   (b) inicjalizacja bez etykiet (autoenkoder na obrazach task0 — ogólność
   bez wąskiej supervizji; uczciwy CL, etykiet nie dotykamy).

## F2 — Zamrożone cechy: szerokość × źródło (ZAKOŃCZONE, 06.07.2026): PLATEAU

Pliki: `src/run_F2_frozen_features.py`; wyniki: `results/F2_frozen_features.json`.
5 seedów × 15 epok/zadanie, proto=mean, class-IL.

| Wariant | Fashion ACC | MNIST ACC | MAC |
|---|---|---|---|
| rand (8,16) [=F1d] | 60.19 ± 5.19% | 66.23 ± 2.73% | 390k |
| rand (16,32) | 61.70 ± 2.00% | 66.58 ± 1.80% | 1 224k |
| rand (32,64) | 60.50 ± 2.99% | 65.78 ± 1.58% | 4 248k |
| AE-task0 (8,16) | 61.83 ± 0.96% | 64.79 ± 1.97% | 390k |
| AE-task0 (16,32) | 59.09 ± 3.10% | 58.86 ± 4.36% | 1 224k |

**WERDYKT: PONIŻEJ REPLAY na obu zbiorach** (−15.1 / −22.2pp).
1. **Szerokość losowych cech NIC nie daje:** 11× MAC → 0pp (w szumie).
   Plateau informacyjne losowej projekcji potwierdzone.
2. **Autoenkoder ≈ losowe (wąskie) lub gorzej (szersze):** rekonstrukcja
   pikselowa nie uczy cech dyskryminacyjnych. Czwarta instancja prawa
   "objective mismatch psuje wspólne cechy" (po D5, E2-v1, F1a).
3. Krzywa Pareto zamrożonych cech: sufit ~60–67% class-IL niezależnie od
   szerokości i źródła (bez semantyki). Do papieru jako charakterystyka
   granicy "architektura zamiast pamięci" na cechach generycznych.
4. **Jedyny nieprzetestowany kanał: informacja SPOZA obrazów → G1
   (prototypy semantyczne, słowa jako kotwice).**

## F3 — Parametryczny feature replay (ZAKOŃCZONE, 07.07.2026): PONIŻEJ REPLAY, połowa luki zamknięta

Pliki: `src/mars_cl_f3.py`, `src/run_F3_feature_replay.py`;
wyniki: `results/F3_feature_replay.json`. 5 seedów × 15 epok/zadanie,
sen per krok (zbalansowany; lekcja z F0-replay powtórzona i naprawiona).

| Wariant (class-IL) | Fashion ACC | MNIST ACC | Forgetting F/M |
|---|---|---|---|
| f3_ncm (pody+negatywy) | 60.19 ± 5.19% | 66.24 ± 2.73% | 16.0 / 10.9pp |
| **f3_sem (proj+sen)** | **70.80 ± 2.24%** (min 68.83) | 63.59 ± 3.66% | 28.8 / 40.8pp |
| [F0] replay-200 | 76.97% | 88.81% | 27.0 / 13.5pp |
| [G1] g1_all (sufit) | 80.45% | 78.32% | 8.7 / 10.4pp |

**WERDYKT: PONIŻEJ REPLAY** (Fashion −6.2pp). Ale:
1. **f3_sem zamknął ~52% luki F1d→sufit** (60.2 → 70.8 przy suficie 80.5)
   przy ~1 KB/klasę i zerze przechowywanych danych. +10.6pp nad NCM.
2. **f3_ncm = F1d co do class-IL na OBU zbiorach (Δ +0.0pp, 5 seedów)** —
   definitywne domknięcie: kalibracja podów NIE jest dźwignią; deficyt
   w 100% siedzi w routingu międzyzadaniowym.
3. **Diagnoza resztkowej luki (6.2pp do replay, 9.7pp do sufitu):
   dryf resztkowy projekcji** — F=28.8pp vs 8.7pp przy g1_all. Ciekawe:
   smoke (4 epoki) miał WYŻSZE ACC (73.5) i niższy F (21.4) niż full
   (15 epok) — dłuższy trening na zadaniu = silniejsze przeciąganie
   projekcji do realnych cech nowego zadania kosztem gaussowskich
   przybliżeń starych. Sen jest za słabym modelem przeszłości, żeby
   wytrzymać 15 epok teraźniejszości.
4. MNIST: semantyka słów cyfr strukturalnie słaba (potwierdzenie G1);
   f3_sem < f3_ncm — kierunek semantyczny jest dla danych, gdzie język
   koduje wygląd.

**Kandydaci F3b (pre-rejestrować przed runem):** (a) sweep epok projekcji
{4, 8, 15} — kompromis plastyczność/dryf to jawny parametr mechanizmu;
(b) kotwica L2-SP na projekcji (kara ||W−W_prev||²) — tani hamulec dryfu
komplementarny do snu; (c) bogatszy sen: k centroidów per klasa (mini
k-means w cechach, nadal ~kilka KB/klasę) zamiast 1 Gaussianu.

## F3b — Kontrola dryfu (ZAKOŃCZONE, 07.07.2026): **SYGNAL+ na Fashion**

Pliki: `src/run_F3b_drift_control.py` (+ rozszerzenia `mars_cl_f3.py`);
wyniki: `results/F3b_drift_control.json`. 6 wariantów × 5 seedów.

| Wariant (Fashion, class-IL) | ACC | min | Forgetting |
|---|---|---|---|
| **k4 (sen 4-centroidowy)** | **75.78 ± 1.39%** | 74.27% | 21.9pp |
| **combo (ep8+l2sp0.1+k4)** | **75.68 ± 1.17%** | 74.79% | **15.8pp** |
| l2sp_0.1 | 72.67 ± 1.36% | 71.22% | 20.0pp |
| ep4 / ep8 | 71.9 / 71.7% | — | 22.9 / 25.9pp |
| l2sp_1.0 (za tępy) | 67.74% | — | 16.7pp |
| [F0] replay-200 | 76.97 ± 1.09% | 75.23% | 27.0pp |

**WERDYKT (pre-rejestrowany, k4 vs replay): SYGNAL+** — Δ = −1.19pp przy
progu szumu 2.48pp = **równoważność statystyczna z experience replay przy
ZEROWYM buforze danych**. Uczciwie: to remis w granicach szumu, nie
przewaga; ale kryterium ("architektura+statystyki+semantyka ≥ replay−szum")
było zadeklarowane z góry i jest spełnione. Wiarygodność wyboru wariantu:
k4 i combo wskazał już smoke, oba lądują ~75.7% niezależnie (to nie
szczęśliwy traf jednego z sześciu).

**Najmocniejsza forma wyniku — combo:** ACC w szumie z replay przy
**forgettingu 15.8pp vs 27.0pp replay** (o 40% mniej zapominania), MAC
stały (×1.0007), pamięć ~4 KB/klasę statystyk, zero przechowanych próbek
(prywatność). "Architektura + sen parametryczny + semantyka ≈ pamięć
epizodyczna" — główna teza serii F potwierdzona na Fashion.

**Granica (uczciwie):** MNIST PONIŻEJ REPLAY (−20.9pp) — potwierdzenie
z G1: mechanizm wymaga, by nazwy klas kodowały podobieństwo wizualne
(ubrania ✓, słowa cyfr ✗). To definiuje zakres stosowalności, nie
obala tezy. Drabina Fashion: finetune 18 → F1d 60.2 → f3_sem 70.8 →
**F3b 75.8 ≈ replay 77.0** → sufit g1_all 80.5 → joint 90.4.

**Następne kroki:** (1) Split-CIFAR-10 — nazwy klas CIFAR są semantycznie
bogate (airplane/dog/cat...), mechanizm powinien się przenieść; to jest
próg wiarygodności publikacji. (2) G2 kompozycyjność (klasa z opisu).
(3) Whitepaper v0.3 z serią F/G.

## F4 — Split-CIFAR-10 (ZAKOŃCZONE, 07.07.2026): **SYGNAL+ — teza przenosi
## się, z odwróceniem ról**

Pliki: `src/cifar_cl.py`, `src/run_F4_split_cifar.py`;
wyniki: `results/F4_split_cifar.json`. 5 seedów × 15 epok/zadanie,
CifarBackbone (16,32) losowy zamrożony dla MARS, trenowalny dla monolitów.

| System (class-IL) | ACC | min | Forgetting |
|---|---|---|---|
| finetune | 10.14 ± 0.32% | 10.00% | 73.0pp |
| replay-200 | 18.90 ± **8.80**% (!) | 12.62% | 64.8pp |
| **mars_combo** | **32.04 ± 1.01%** | **30.57%** | **31.0pp** |
| mars_k4 | 30.08 ± 1.10% | 28.47% | 46.8pp |
| joint (sufit) | 68.73 ± 2.32% | — | — |

**WERDYKT (combo vs replay): SYGNAL+** — Δ = +13.14pp przy progu 9.81pp
(próg rozdęty wariancją replay; min per-seed MARS 30.57 > ŚREDNIA replay).

**Kluczowa obserwacja — na trudnych danych role się ODWRACAJĄ:**
replay-200 ZAŁAMAŁ SIĘ na CIFAR (18.9 ± 8.8 — niestabilny między seedami),
bo jego trenowalny backbone dryfuje na naturalnych obrazach, a 200 próbek
nie starcza za kotwicę. MARS z NIETYKALNĄ reprezentacją nie ma czego
dryfować — stabilność 1.0pp std, forgetting 31 vs 65pp. **Im trudniejsze
dane, tym cenniejsza nienaruszalna reprezentacja** — przewaga stacjonarności
rośnie z trudnością zbioru. To silniejsza forma tezy serii F niż wynik
z Fashion.

**Uczciwe zastrzeżenia:** (1) poziom bezwzględny skromny (32% vs sufit
68.7%) — losowe cechy ograniczają, zgodnie z pre-rejestrowanym ryzykiem;
teza to WZGLĘDNA odporność, nie "rozwiązany CIFAR-CL". (2) replay z większym
buforem (1000+) pewnie wygra — bufor to pokrętło, którego nie sweepowaliśmy
(200 pre-rejestrowane w F0); odnotować jako oś porównania w v0.3.
(3) Droga wzwyż jest znana: mocniejszy ZAMROŻONY backbone (nienadzorowany/
pretrenowany) — mechanizm CL zostaje bez zmian.

**ZABEZPIECZENIE NARRACYJNE do v0.3 (z recenzji zewnętrznego agenta,
07.07.2026, przyjęte):** wynik F3b NIE jest pokonaniem replay *ceteris
paribus* — system wstrzykuje zewnętrzną geometrię pojęć (GloVe; embeddingi
korpusowe, nie "model językowy"). Właściwa rama: **oś zasobów** — każda
metoda CL konsumuje zasób: replay konsumuje PRZECHOWANE PIKSELE użytkownika
(prywatność, pamięć rosnąca z danymi); MARS konsumuje PUBLICZNĄ, statyczną
geometrię słów (darmowa, niezależna od danych zadania, wymaga tylko nazw
klas). Kategoria rozwiązania: "pamięć epizodyczna → priory semantyczne +
sen parametryczny". MNIST = uczciwy dowód graniczny (bez geometrii
semantycznej nazw metoda przegrywa). Tak oprawić w v0.3.
