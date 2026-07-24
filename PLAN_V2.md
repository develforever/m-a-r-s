# M.A.R.S. — Plan po v1.2/v1.3: kandydaci na dalsze serie (mapa forward)

Data: 2026-07-23. Status: PROPOZYCJA (do zatwierdzenia przez Roberta).
Zastępuje część B `PLAN_V1.md` (WYCZERPANA: P·Q·G2b zrobione). Część A
`PLAN_V1.md` (przegląd v1.0) i część B (P/Q/R/G2b) zostają jako dowód
pre-rejestracji. Ta mapa dodaje jednego nowego kandydata (G3), przenosi
R bez zmian i porządkuje priorytety po zamknięciu G-serii.

Obowiązują bez zmian: runy WYŁĄCZNIE u Roberta (GTX 1050 Ti); 5 seedów,
pary per-seed, próg szumu std+std, SYGNAL-parowy obok; negatyw = wynik;
branch per seria, main nietykalny; submisje WSTRZYMANE — żaden punkt
tego planu nie proponuje submisji; pre-rejestrowany DROGA_*_PLAN.md
z progami PRZED pierwszym runem każdej serii.

---

## Zrealizowane od PLAN_V1 (stan wejściowy)

- **Seria P** (v1.1) — detekcja zatrucia: kandydat I4 sfalsyfikowany,
  ale BRAMA STRUKTURALNA D1>0.45 (60/60) i PRAWO DYSTANSU przeżyły jako
  pre-rejestrowane sukcesy. Polityka protokołu kompletna i zmierzona.
- **Seria Q** (v1.2) — kolektyw × długi horyzont: bariera skali =
  artefakt budżetu snu; równoważność kolektywu rozszerzona z 10 klas /
  5 agentów na 100 klas / 20 agentów; self-dream = dźwignia każdego
  agenta (nowy best pojedynczy 45.35). Q1→Q2d, CLAIMS 35–41.
- **G2b** — dystans kodowy ECOC: NEGATYW MOCNY, wąskie gardło
  zizolowane do CECH, nie słownika. CLAIMS 15b, DROGA_G2B_NOTATKI.md.

Wolne litery serii: **G3** (rozszerzenie G), **R** (heterogeniczny),
dalej S… Wnioski z G2b i Q przestawiają priorytety poniżej.

---

## Kandydaci (kolejność = proponowany priorytet)

### R — kolektyw heterogeniczny („rewolucja 2.0") — GŁÓWNY BET NARRACJI CL

Największy zysk koncepcyjny i jedyny kandydat na nowy headline w głównej
osi projektu (continual learning / protokół wymiany snów). Dziś protokół
zakłada WSPÓLNY seed/backbone — synchronizacja jest darmowa, bo wszyscy
agenci dzielą tę samą przestrzeń cech. R pyta: czy agenci o RÓŻNYCH
backbone'ach (losowy ↔ pretrained; dwa różne pretrained) mogą wymieniać
sny?

- **Problem rdzeniowy:** statystyki snu (spike-and-slab per wymiar) są
  wyrażone w przestrzeni cech NADAWCY. Odbiorca ma inną bazę — surowy
  payload jest bez znaczenia. Jedyny wspólny układ odniesienia w
  systemie to **przestrzeń kotwic słownych** (GloVe): niezmienna,
  a priori, dzielona przez konstrukcję.
- **Szkic mechanizmu translacji (do rozpisania w
  `ARSENAL_PRZEOCZONYCH_NARZEDZI.md` PRZED pre-rejestracją):**
  (a) każdy agent uczy projekcji cechy→kotwica (już ma — to jest
  mechanizm routingu); (b) nadawca nie wysyła statystyk cech, lecz
  statystyki w przestrzeni kotwic (po projekcji) LUB parametry generatora
  zakotwiczonego semantycznie; (c) odbiorca „śni" w swojej przestrzeni
  cech warunkowo na kotwicy i dopasowuje własną projekcję. Kandydat
  minimalny: wymiana wyłącznie (nazwa klasy + centroid w przestrzeni
  kotwic 300d) — test, ile klasy da się przenieść przez sam wspólny
  układ semantyczny.
- **Werdykt (do pre-rejestracji):** ACC klas adoptowanych między
  heterogenicznymi agentami vs (i) sufit lokalny odbiorcy, (ii) kolektyw
  homogeniczny z L2/Q jako górna kotwica. Trzy wyniki, każdy domyka:
  przenosi się w szumie → protokół w pełni „representation-agnostic"
  (headline); częściowo → zmierzona cena heterogeniczności; nie →
  granica: wspólna baza cech jest konieczna, sen nie jest uniwersalnym
  językiem.
- **Koszt:** średni–wysoki (dwa backbone'y, nowa warstwa translacji;
  wzór L+I na 1050 Ti — godziny). **Ryzyko: wysokie** (mechanizm może
  nie istnieć w prostej formie). **Warunek wejścia:** osobna zgoda
  Roberta + szkic mechanizmu w ARSENAL (nie ruszać runnera przed nim).
- **STATUS 2026-07-23:** szkic mechanizmu ZROBIONY (ARSENAL, Dopisek
  2026-07-23) + pre-rejestracja ZROBIONA (`DROGA_R_PLAN.md`). Mechanizm:
  kotwica-interlingua + dekoder per-agent (`anchor → feature`), drabina
  R-mild / R-hard, warianty R0/R1/ceiling/sanity, progi z góry.
- **ZAMKNIĘTE 2026-07-23 — SFALSYFIKOWANE (poziom kotwica-interlingua).**
  R-mild v1 nierozstrzygające (SANITY zadziałało); R1b (translacja
  uregularyzowana, klasy dzielone) — **BRAMKA NIEZDANA**: RBF_SANITY
  4.82% ≪ 65%. Kluczowa diagnoza: ORACLE_SANITY 4.00% (pełna wiedza
  dekodera + zero heterogeniczności i tak zapada) → wąskim gardłem jest
  50-d przestrzeń słów (class-collapsed), nie dekoder/heterogeniczność.
  Zgodnie z zamrożoną bramką: **poza CLAIMS.** Notatki: DROGA_R_NOTATKI
  (dopisek R1b). Ewentualny następny poziom (osobna pre-rejestracja,
  decyzja Roberta): **R2 = translacja w przestrzeni CECH** (Procrustes na
  klasach dzielonych) — jedyna droga zgodna z ORACLE=4%; ustępstwo od
  „tylko słowa".
- **ODBLOKOWANE 2026-07-23 (decyzja Roberta): R2 AKTYWNE na osi Part III.**
  Po domknięciu Part II (G/routing wyczerpane) powrót na główną oś CL.
  R2 = ortogonalny Procrustes Ω: H_A→H_B na K=4 klasach dzielonych
  (publiczny zbiór kalibracyjny), payload cech jak w I, `adopt_classes`
  nietknięte. Pre-rejestracja: `DROGA_R2_PLAN.md` (DO ZATWIERDZENIA),
  bramka R2_SANITY≥65%; warianty CEILING/R0/R2_SANITY/R2_HET(+RIDGE_HET
  ablacja), split jak R1b. R-mild najpierw (izometria istnieje z
  konstrukcji); R-hard = osobna pre-rejestracja po zdanym R-mild.
- **WYNIK R2 2026-07-23 — BRAMKA ZDANA, R WRACA.** R2_SANITY 80.15% ≈
  CEILING 81.62% → feature-space validated (naprawia R1b). Pierwszy
  pozytywny heterogeniczny transfer (R2_HET vs R0 SYGNAL+ +21pp). Ortog.
  Procrustes płaci cenę (R2_HET 31%, −50pp vs sufit), ale **ablacja
  liniowa RIDGE_HET 65% = 80% sufitu, +34pp nad Procrustesem** →
  ograniczenie ortogonalności jest kosztem, nie heterogeniczność. CLAIMS
  NIETKNIĘTE (headline liniowy = ablacja, wymaga potwierdzenia à la Q2c).
  Następne: **R2b** (mapa liniowa jako primary + kontrole: próg ≥70%
  sufitu, sweep K, przeciek) → potem R-hard (losowy↔pretrained,
  prostokątna mapa). To jest żywy tor główny CL (Part III).

### G3 — kompozycyjny zero-shot na cechach pretrained — TANI, DECYZYJNY, PERYFERYJNY

Bezpośrednia konsekwencja negatywu G2b: dźwignie (a) dystans kodowy i
(b*) osiągalność strukturalna wyczerpane; G2b zizolował wąskie gardło do
**cech** (dźwignia c). G3 testuje ją wprost — ten sam eksperyment
kompozycyjny na zamrożonym resnet18-ImageNet (infrastruktura L już
istnieje: `mars_cl_l.py`), oba słowniki (attrs11, attrs21) na tych
samych seedach.

- **Hipoteza:** semantyczne cechy dają detektory pojęć poniżej progu
  błędu per-bit → (i) ZS przekracza próg lub daje SYGNAL+ vs losowe
  cechy, ORAZ (ii) na mocnych cechach ECOC PRZESTAJE szkodzić (attrs21 ≥
  attrs11) — bezpośredni test mechanizmu z notatek G2b.
- **Werdykt:** G3+ → negatyw G2/G2b był w całości deficytem cech
  (kompozycyjność wymaga semantyki reprezentacji, nie ręcznego kodu);
  G3− → kompozycyjność z ręcznego słownika nie przechodzi nawet na
  mocnych cechach (granica podejścia, domyka serię G ostatecznie).
- **Koszt:** niski (G2b FULL = 132 s; pretrained dokłada tylko ekstrakcję
  cech — rząd minut). **Ryzyko:** niski (oba wyniki informatywne).
- **Uwaga narracyjna:** G należy do osi Part II (routing/zero-shot
  ceiling — drugi paper), NIE do głównej narracji CL. Domyka boundary
  condition czysto, ale nie tworzy headline'u CL.
- **ZAMKNIĘTE 2026-07-23 — G3− TWARDY (dźwignia c sfalsyfikowana).**
  Cechy pretrained ResNet18 NIE pomagają: pretrained/attrs11 2.77% ≈
  random 3.17% (T1 SZUM, ratio 0.87), best 2.77% ≪ 30%. T4 sanity:
  random odtwarza G2b. T3: ECOC na pretrained nadal backfire (−2.69pp).
  Wniosek: wąskim gardłem jest PODEJŚCIE (ręczny słownik + liniowe
  detektory), nie reprezentacja ani dystans kodowy — wszystkie 3
  dźwignie G2 wyczerpane. **Seria G DOMKNIĘTA** jako granica paradygmatu.
  CLAIMS 15c, WHITEPAPER §13, DROGA_G_NOTATKI (dopisek G3). Dalej tylko
  INNY paradygmat (atrybuty uczone end-to-end) — nie ten.

### G3b (opcjonalnie, warunkowo) — dekorelacja detektorów (dźwignia b)

Ostatnia nietknięta dźwignia z diagnozy G2: kara na korelację wyjść BCE
między atrybutami (detektory mają się nie powielać). Tania (zmiana lossu
w istniejącym runnerze). Sensowna TYLKO jako dodatek do G3+ (jeśli cechy
same nie wystarczą) — w izolacji na losowych cechach prawie na pewno
powtórzy G2b. Priorytet: dopiero po G3, warunkowo.

### VSA — wiązanie wektorowo-symboliczne jako zamiennik poda (long shot, badawczy)

Zaznaczone w roadmapie WHITEPAPER jako nieeksplorowane: pod (gęsty
matmul) zastąpiony wiązaniem VSA/HRR (bind/bundle w przestrzeni
hiperwymiarowej). Potencjał: kompozycyjność „za darmo" z algebry
wiązania zamiast z ręcznego słownika — mógłby ominąć całą przyczynę
negatywu G2/G2b. Ryzyko bardzo wysokie, mechanizm daleki od obecnego
kodu, zysk niepewny. Trzymać jako pozycję inwentarzową w ARSENAL, nie
jako następną serię.

### Tor bez runów — konsolidacja pod dwa papery

Niezależny od powyższych: treść jest rozcięta na paper A (CL) i paper B
(routing/zero-shot ceiling — tu trafiają G/G2/G2b/G3). Po G3 sekcja
kompozycyjna papera B jest kompletna (sukces LUB domknięty negatyw).
Kandydat na osobną sesję dokumentową, gdy Robert wróci do tematu
submisji (dziś WSTRZYMANE).

---

## Proponowana kolejność wykonawcza

1. **G3** — natychmiast wykonalny, tani, domyka granicę kompozycyjności
   i wprost testuje mechanizm z G2b; nie wymaga szkicu. Odpala się jak
   G2b (nowy runner na wzór, resnet18-ImageNet z L).
2. **Szkic mechanizmu R** w ARSENAL (bez runów) → decyzja Roberta →
   pre-rejestracja `DROGA_R_PLAN.md` → **R** jako główny bet headline'u.
3. G3b tylko warunkowo (po G3+), VSA tylko badawczo, tor papierowy przy
   powrocie do submisji.

Zasada domykająca: R jest jedynym punktem, który może dać nowe
twierdzenie w GŁÓWNEJ osi CL; G3/G3b/VSA obsługują oś Part II i granice.
Priorytet zależy więc od celu sesji — domknięcie granic (G3 najpierw)
albo pościg za headline'em (szkic R najpierw). Oba tory są niezależne.
