# Droga R-hard — kolektyw między RÓŻNYMI reprezentacjami (pre-rejestracja)

Data pre-rejestracji: 2026-07-24. Status: **PLAN ZATWIERDZONY** (Robert,
2026-07-24: **próg bramki X = 50%** sufitu odbiorcy; **zakres (b)** —
DIR-F→S prostokątny primary + DIR-S→F kwadratowy jako obserwacja
asymetrii). Runner piszę PO osobnym zielonym świetle „pisz runner"; runy
WYŁĄCZNIE u Roberta. Branch: `droga-r` (nowe pliki / nowe metody obok;
rdzeń I/L i `adopt_classes` NIETKNIĘTE). Oś: **Part III (kolektyw,
memory without data).**

R-hard to właściwy test rewolucji *representation-agnostic*: krzyżujemy
backbone **losowy-od-zera** (surowe piksele) z **pretrenowanym resnet18**
(cache 512-d) — RÓŻNA treść informacyjna ORAZ RÓŻNE wymiary cech. R2b
dowiózł kolektyw dla RÓŻNYCH backbone'ów tej samej rodziny (R-mild:
liniowa mapa istnieje z konstrukcji, oba to liniowe obrazy tej samej
512 → 92% sufitu). R-hard pyta: czy protokół działa, gdy **wspólnej bazy
cech NIE MA**.

## Punkt wyjścia (zmierzony, R2b)

R-mild odzyskał **92% sufitu** (R2b_HET_K4 79.98% vs CEILING 86.87%),
bramka ≥70% zdana z zapasem, mała cena heterogeniczności (−6.9pp), K=2
wystarczyło. Kluczowa przyczyna sukcesu: relacja H_A→H_B była **ogólną
mapą liniową R_B R_A⁺** — istniała z konstrukcji, bo oba backbone'y to
losowe rzuty tej samej 512. R-hard usuwa to założenie: treść cech jest
naprawdę różna.

## Hipoteza

**H-Rhard:** uregularyzowana **prostokątna** mapa liniowa H_A(D_A) →
H_B(D_B), fitowana na K klasach dzielonych (te same obrazy przez oba
fronty sensoryczne), pozwala odbiorcy odzyskać co najmniej **X%** jego
LOKALNEGO sufitu, przy transferze między agentem od-zera (surowe piksele)
a pretrenowanym (cache 512). X — propozycja niżej, do zatwierdzenia.

Przewidywanie (poniżej, „Przewidywanie") jest jawnie ostrożne: R-hard
jest znacznie trudniejszy niż R-mild i możliwy jest duży spadek lub
negatyw. Obie ścieżki domykają serię.

## Ograniczenie rdzenia (KLUCZOWE — determinuje projekt)

`mars_cl.py` hardkoduje wymiar cech `BB_H = 128` w dwóch miejscach
niezbędnych dla PEŁNEGO agenta: projekcja semantyczna
`self.proj = nn.Linear(BB_H, emb_dim)` (l. 63) oraz inicjalizacja podów
`torch.randn(BB_H, pod_hidden)` (l. 223; `mars_cl_f3.py` l. 203). Rdzeń
zostaje NIETKNIĘTY (zasada). Konsekwencje projektowe:

1. **Pełny ODBIORCA musi być 128-d.** Agent o cechach 512-d nie
   przejdzie `learn_task` (proj/pody zakładają 128). Więc odbiorca zawsze
   ma front 128-wymiarowy.
2. **NADAWCA może mieć dowolny wymiar (512), ale tylko „stats-only".**
   Nadawca nie potrzebuje routingu — dostarcza wyłącznie statystyki cech
   swoich klas adoptowanych. Payload 512-d budujemy BEZPOŚREDNIO z cech
   klasy (k-sparse, jak `FeatureStatsKSparse`), z pominięciem proj/podów —
   NOWA funkcja obok rdzenia (rdzeń nietknięty).
3. **Prostokątna mapa jest wykonalna tam, gdzie 512 jest po stronie
   NADAWCY:** `RidgeTranslator` już jest prostokątny (W=[D_A+1, D_B],
   zamknięta forma dla dowolnych D_A≠D_B — brak zmiany kodu, tylko nowy
   asert w smoke), a `adopt_classes_maptransform` jest dim-agnostyczny
   (śni H_A w D_A → map → H_B w D_B → statystyki 128-d odbiorcy →
   istniejące `adopt_classes`). Zatem **prostokątny transfer
   pretrained(512)→scratch(128) działa bez dotykania rdzenia.** Kierunek
   odwrotny z PRAWDZIWYM odbiorcą 512 wymagałby zmiany rdzenia — patrz
   „Decyzje", opcja (c).

## Fronty sensoryczne (istniejący kod, NIETKNIĘTY)

- **Scratch (od zera, surowe piksele):** `CifarBackbone()` — losowy
  zamrożony CNN, `[B,3072] → [B,128]` (2 bloki conv + proj+ReLU).
  Własny front na surowych pikselach CIFAR. Istnieje (F1d/F3b/J2/K0).
- **Foundation (pretrenowany), 512-d RAW:** cache `Ftr` z
  `extract_or_load_cifar_feats` (resnet18-ImageNet @224, zamrożony) —
  cechy 512-d BEZ redukcji. Front nadawcy w wariancie prostokątnym.
- **Foundation-reduced 128-d:** `ReducedBackbone()` — losowy zamrożony
  rzut `512 → 128` na cache (jak w R2b). Pełny agent 128-d.

**Parowanie per-próbka (te same OBRAZY przez oba fronty):** cache `Ftr[i]`
i surowe `Xtr[i]` są indeksowane tą samą kolejnością datasetu (jeden
`load_cifar10_norm` → `extract_or_load_cifar_feats` liczy `Ftr` w
kolejności i zapisuje z tym samym `ytr`). Kalibracja wybiera indeksy klas
dzielonych i podaje `Xtr[idx]` do frontu pikselowego oraz `Ftr[idx]` do
frontu 512 — pary (H_A, H_B) z tych samych obrazów. Runner ładuje OBA
tensory z jednego wczytania (wymóg zamrożony: spójna kolejność indeksów).

## Mechanizm (ZAMROŻONY) — primary = prostokątny Ridge

- **Mapa:** `RidgeTranslator` (liniowa, zamknięta forma, z biasem,
  clamp_min 0), **H_A(D_A) → H_B(D_B)** dla D_A≠D_B. Auto-λ z gridu
  **{1e-3, 1e-2, 1e-1, 1, 10}** po rekonstrukcji na HELD-OUT (split 80/20
  wewnątrz klas kalibracyjnych; min MSE ‖map(H_A^val)−H_B^val‖). Zero
  wglądu w klasy adoptowane. (Kod `fit_ridge_autolam` z R2b — reużyty.)
- **Payload nadawcy:** statystyki k-sparse cech klas adoptowanych w
  przestrzeni H_A nadawcy. Dla nadawcy 512 — NOWA funkcja
  `export_feature_payload_from_cache` (k-sparse wprost z cech cache;
  bez proj/podów). Dla nadawcy 128 — istniejące `export_class_stats`.
- **Adopcja:** `adopt_classes_maptransform(ADOPT, payload_A, map.predict,
  …)` — śnij H_A → `map` → H_B(128 odbiorcy) → statystyki → istniejące
  `adopt_classes`. NIETKNIĘTE.

## Split (ZAMROŻONY) — sweep K∈{2,4,6}

CIFAR-10, 10-way class-IL (parytet z R2b):
- **CAL_POOL = [0,1,2,3,4,5]** — 6 klas dzielonych, uczonych realnie
  przez odbiorcę; mapa fitowana na PIERWSZYCH **K∈{2,4,6}**. Wszystkie 6
  uczone niezależnie od K → projekcja odbiorcy stała, zmienia się TYLKO
  budżet kalibracyjny mapy.
- **OWN = [6,7]** — własne odbiorcy (realne).
- **ADOPT = [8,9]** — adoptowane od nadawcy przez mapę; metryka
  (recipient-relative, `row[1]` w eval `[OWN, ADOPT]`).

**Świeże seedy: 5–9** (precedens P1c/R2b). 5 seedów, pary per-seed.

## Warianty (kierunki ROZDZIELONE; metryka względem sufitu ODBIORCY)

Zasada bez zmian: metryka = ACC klas adoptowanych względem **lokalnego
sufitu ODBIORCY w jego reżimie**; NIGDY liczba scratch vs foundation
wprost. Kierunek transferu raportowany OSOBNO (asymetria = osobna
obserwacja).

**A. DIR-F→S (foundation→scratch) — PROSTOKĄTNY 512→128 — PRIMARY.**
Nadawca = pretrained RAW 512 (stats-only); odbiorca = `CifarBackbone`
(scratch, surowe piksele → 128). RÓŻNA treść ORAZ różne wymiary.
- `CEILING_S` — homogeniczny u scratcha (nadawca scratch-128, payload
  realny). Sufit odbiorcy w jego reżimie.
- `R0_S` — podłoga anchor-only u scratcha.
- `HET_FS_K{2,4,6}` — pretrained(512) → scratch(128), prostokątny ridge.
  **PRIMARY**, bramka przy K=4.

**B. SANITY (kontrola maszynerii) — prostokątny, mapa ISTNIEJE z
konstrukcji.** Nadawca = pretrained RAW 512; odbiorca = `ReducedBackbone`
(128 = losowy liniowy rzut TEJ SAMEJ 512). Wtedy H_B = ReLU(R·H_A) →
mapa 512→128 istnieje z konstrukcji → ridge powinien odzyskać ≈ sufit.
Izoluje zdrowie prostokątnej mapy + parowania dwóch frontów NIEZALEŻNIE
od różnicy treści.
- `CEILING_PT` — homogeniczny u foundation-reduced (odniesienie sufitu).
- `SANITY_RECT_K{2,4,6}` — pretrained(512) → reduced(128).

**C. DIR-S→F (scratch→foundation) — KWADRATOWY 128→128 — OBSERWACJA
ASYMETRII (opcjonalny, NIE bramkowany).** Nadawca = scratch-128; odbiorca
= `ReducedBackbone` (foundation-reduced 128). RÓŻNA treść, mapa
kwadratowa (bo prawdziwy odbiorca 512 wymagałby zmiany rdzenia).
- `CEILING_PT`, `R0_PT`, `HET_SF_K{2,4,6}`.
- Raportowany OSOBNO. Uwaga zamrożona: DIR-S→F jest dim-matched
  (kwadratowy), DIR-F→S dokłada wyzwanie prostokątności → asymetria
  F→S vs S→F MIESZA kierunek treści z kształtem mapy; to OBSERWACJA, nie
  twierdzenie.

Każdy wariant liniowy przez sweep **K∈{2,4,6}**; auto-λ per fit. Bramka
liczona przy K=4 (pre-fixed, bez selekcji po fakcie).

## Kryteria werdyktu (Z GÓRY) — TWARDA BRAMKA

1. **BRAMKA PRIMARY (zamrożona):** **HET_FS_K4 ≥ 50% × CEILING_S** (sufit
   scratcha) na seedach 5–9. **X = 50% ZATWIERDZONE (Robert, 2026-07-24).**
   Zdana → kolektyw *representation-agnostic* między RÓŻNYMI
   reprezentacjami POTWIERDZONY (częściowy, ze zmierzoną ceną) → wejście
   do CLAIMS/WHITEPAPER. Niezdana → patrz interpretacje miss.
2. **WARUNEK KONIECZNY:** **HET_FS_K4 vs R0_S = SYGNAL+** (pary per-seed,
   próg std+std). Mapa musi nieść informację ponad podłogę anchorową;
   bez tego „zdana" bramka byłaby artefaktem sufitu.
3. **SANITY maszynerii:** **SANITY_RECT_K4 ≥ 65% × CEILING_PT**.
   Rozbieżność BLOKUJE interpretację (à la R2b / R-mild v1): dowodzi, że
   prostokątny ridge + parowanie dwóch frontów są zdrowe, więc każdy
   spadek HET jest różnicą TREŚCI, nie zepsutą maszynerią.
4. **Po zdanej bramce (tylko wtedy):**
   - HET_FS vs CEILING_S: zmierzona CENA różnicy reprezentacji.
   - Krzywa K (2/4/6): ile kalibracji trzeba przy braku wspólnej bazy.
   - DIR-S→F: raport osobny (asymetria kierunku, z zastrzeżeniem C).

Progi (X%, 65% sanity, SYGNAL+ nad R0) i grid λ zamrożone PRZED runem.
Min-par raportowany. Negatyw (w tym niezdana bramka) = wynik.

### Uzasadnienie progu X (do zatwierdzenia)

R-mild miał sufit „za darmo" (mapa z konstrukcji) i zdał 70%. R-hard nie
ma wspólnej bazy — transferowalne jest tylko to, co cech odbiorcy (128)
da się LINIOWO przewidzieć z cech nadawcy (512) na klasach dzielonych.
Dlatego proponuję **niższy, ale wciąż merytoryczny próg X = 50%**:
„odzyskać co najmniej połowę homogenicznego sufitu ODBIORCY mimo innej
reprezentacji nadawcy" to mocne, jednoznaczne twierdzenie, jawnie ponad
podłogą (R0 w R2b ~5%), a zarazem realistyczne bez izometrii.
Alternatywy do rozważenia: **40%** (łagodniejszy, wyższa szansa zdania,
słabsze twierdzenie) / **60%** (mocniejsze twierdzenie, wyższe ryzyko
nieinformatywnego faila). Rekomendacja: 50%.

### Interpretacje miss (pre-commit)

- **HET_FS ≥ X% i SYGNAL+** → headline: protokół *representation-agnostic*
  działa też między RÓŻNYMI reprezentacjami (nie tylko wspólną bazą).
- **HET_FS < X% ale SYGNAL+** → zmierzona GRANICA: transfer częściowy;
  wspólna baza pomaga, lecz nie jest ściśle konieczna; różnica
  reprezentacji ma zmierzoną cenę. Domyka łuk.
- **HET_FS ≈ R0 (brak SYGNAL+)** → twarda GRANICA: naprawdę różne
  reprezentacje nie dzielą snów przez wyrównanie liniowe → wspólna baza
  (R-mild) konieczna. Domyka łuk.

## Przewidywanie (zapisane przed runem)

R-hard ≪ R-mild (brak izometrii/wspólnej bazy — różna TREŚĆ cech).
SANITY_RECT zda (mapa istnieje z konstrukcji, jak R-mild → oczekiwane
~sufit). HET_FS: szerokie pasmo; jeśli cechy foundation liniowo informują
losową bazę scratcha — plausibly ~40–70% (niskiego) sufitu scratcha;
jeśli nie — blisko podłogi. **Krzywa K rosnąca** (odwrotnie niż R-mild,
gdzie K=2 wystarczyło): trudniejszą, między-treściową mapę lepiej
constrainuje więcej klas kalibracyjnych — obserwacja falsyfikowalna.
Asymetria DIR-F→S vs DIR-S→F oczekiwana (rich→poor 512→128 vs
poor→rich treści) — osobna obserwacja, z zastrzeżeniem kształtu mapy.

## Plik, koszt, ryzyko

- **Runner (nowy):** `src/run_R_hard.py` (wzór `run_R2b_linear.py`: dwa
  fronty, kierunki rozdzielone, prostokątny ridge, auto-λ, sweep K,
  bramka liczona automatycznie).
- **Nowy kod obok rdzenia (rdzeń NIETKNIĘTY):** `export_feature_payload_
  from_cache` (payload 512-d k-sparse wprost z cech, bez proj/podów) +
  parowana ekstrakcja kalibracyjna przez dwa fronty — w
  `mars_collective_hetero.py` jako NOWE metody lub mały nowy moduł.
  `RidgeTranslator` i `adopt_classes_maptransform` — REUŻYTE bez zmian
  (dodać tylko asert prostokątnego kształtu w smoke translatora).
- **Reużyte:** `CifarBackbone`, `ReducedBackbone`,
  `extract_or_load_cifar_feats`, `load_cifar10_norm`, cache. Rdzeń I/L
  i `adopt_classes` NIETKNIĘTE.
- **Wynik:** `results/R_hard.json` (smoke: `_smoke`).
- **Koszt:** ridge zamknięty (tani), ale odbiorca-scratch liczy realny
  CNN na surowych pikselach, a nadawca-foundation korzysta z cache →
  rząd R2b × liczba kierunków; smoke 1 seed / K=4 najpierw.
- **Ryzyko:** (i) bramka może nie trafić (uczciwie — przewidziane);
  (ii) BN w `CifarBackbone` (zamrożony, eval) — konfiguracja znana z
  J2/K0, ale zweryfikować w smoke; (iii) feat-dim 512 nadawcy — obejście
  stats-only weryfikowane w smoke PRZED FULL.

## Decyzje ZATWIERDZONE (Robert, 2026-07-24)

- **(1) Próg bramki X = 50%** sufitu odbiorcy (zamrożone).
- **(2) Zakres = (b):** DIR-F→S prostokątny (512→128) jako PRIMARY +
  DIR-S→F kwadratowy (128→128) jako osobna obserwacja asymetrii.
  Opcja (c) — nowa podklasa z parametrem feat_dim / prawdziwy odbiorca
  512 — ODRZUCONA na tym etapie (rdzeń pozostaje nietknięty).

Progi zamrożone. Pozostaje osobne zielone światło „pisz runner"
(`src/run_R_hard.py`) — przed nim runnera nie piszę.

## Zasady

5 świeżych seedów (5–9); pary per-seed; próg szumu std+std; bramka X%,
sanity 65% i grid λ zamrożone PRZED runem; runy WYŁĄCZNIE u Roberta;
branch `droga-r`, main nietykalny; rdzeń I/L i `adopt_classes` NIETKNIĘTE;
negatyw = wynik; dyscyplina Q2c (headline z ablacji poza CLAIMS bez
pre-rejestrowanego potwierdzenia). Werdykty → `DROGA_R_NOTATKI.md`
(dopisek R-hard), CLAIMS/WHITEPAPER TYLKO po zdanej bramce. Runnera NIE
piszemy przed zatwierdzeniem tego planu.
