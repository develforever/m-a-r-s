# Droga R1b — naprawa translacji heterogenicznej (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: **DO ZATWIERDZENIA**; runy
WYŁĄCZNIE u Roberta. Branch: `droga-r` (nowe pliki, rdzeń I/L
NIETKNIĘTY). Poprzednik: R-mild v1 NIEROZSTRZYGAJĄCY — kontrola SANITY
zadziałała (`DROGA_R_NOTATKI.md`): dekoder MLP uczony na 2 własnych
klasach nie ekstrapoluje na nieznane rejony kotwicy; `adopt_classes`
uczy się śmieciowych pseudo-cech. Bramka przed jakimkolwiek werdyktem R:
**SANITY ≈ CEILING**.

## Hipoteza

H-R1b: porażka R-mild v1 jest artefaktem instancjacji (wąska kotwica
50-d + niestabilny dekoder MLP z 2 klas), nie konceptu. Przy (1) pełnej
diagnozie wąskiego gardła kotwicy i (2) uregularyzowanej, stabilnej
translacji na klasach dzielonych, ścieżka dekodera przestaje niszczyć
adopcję i SANITY wraca do CEILING.

Metryka główna (bez zmian): ACC klas ADOPTOWANYCH u odbiorcy
(row_class_il[1:]), względem lokalnego sufitu. 5 seedów, pary per-seed,
próg szumu std+std. CEILING = kolektyw homogeniczny (L2, ~73.4%).

---

## KROK 1 — R1b-Oracle (diagnostyka wąskiego gardła kotwicy)

**Cel:** rozdzielić dwie przyczyny porażki v1 — (a) wąskie gardło
informacyjne payloadu anchorowego (128→50) vs (b) generalizacja
dekodera. Pod PEŁNĄ wiedzą dekodera pozostaje tylko (a).

**Konstrukcja:** dekoder `anchor → feature_B` odbiorcy uczony na cechach
WSZYSTKICH 10 klas realnych (oracle — górna granica jakości dekodera;
NIE jest systemem do zgłoszenia, bo łamie „odbiorca zna tylko swoje
klasy" — wyłącznie kontrola diagnostyczna, jawnie oznaczona). Payload
anchorowy i cała reszta jak v1.

**Rozstrzygnięcie (z góry):**
- SANITY-Oracle **≈ CEILING** (≥ blisko 73%): 50-d kotwica PRZENOSI
  dość; porażka v1 była czystą generalizacją dekodera → Krok 2 ma sens
  (lepszy, uregularyzowany translator na klasach dzielonych).
- SANITY-Oracle **nadal zapada** (≪ CEILING): to sama kotwica 50-d jest
  za stratną interlingua, nie dekoder → anchor-only jako interlingua
  jest wątpliwe; Krok 2 (RBF na klasach dzielonych) i tak testuje, czy
  translacja w bogatszej reprezentacji ratuje, ale prognoza pesymistyczna.

Koszt: niski (zmiana zbioru treningowego dekodera; reszta runnera v1).

---

## KROK 2 — R1b-RBF (uregularyzowana translacja na klasach dzielonych)

**Cel:** zastąpić niestabilny dekoder MLP gładkim, uregularyzowanym
translatorem, uczonym na K=4 klasach DZIELONYCH (oba agenty widzą je
realnie) — realistyczne, mało wymagające założenie („mały publiczny
zbiór kalibracyjny"), dające translatorowi wiele rejonów kotwicy zamiast
2.

**Translator (zamrożony wybór):**
- Podstawowy: **kernel ridge / interpolacja RBF** `anchor(50) →
  feature_B(128)` — jądro Gaussa nad embeddingami anchorowymi klas
  dzielonych, regularyzacja λ (grid zamrożony z góry, np.
  {1e-2,1e-1,1} — wybór po SANITY klas dzielonych, nie adoptowanych).
  Gładka interpolacja ekstrapoluje łagodniej niż MLP.
- Ablacja liniowa: **Ridge Regression** (zamknięta forma, najstabilniej-
  sza) — jeśli liniowy translator wystarcza, tym lepiej (R-mild:
  feature_A, feature_B to liniowe obrazy tej samej 512, więc bliska-
  liniowa mapa istnieje z konstrukcji).

**Klasy dzielone (K=4, zamrożone):** 4 klasy kalibracyjne, które KAŻDY
agent dodatkowo trenuje na danych realnych, ROZŁĄCZNE ze zbiorem klas
adoptowanych mierzonych w metryce. Translator fitowany wyłącznie na ich
parach (embed_from_feats(f), f) u odbiorcy. Split CIFAR-10 pinowany:
4 kalibracyjne (dzielone, realne u wszystkich) + 6 protokołowych
(odbiorca uczy 2 własne, adoptuje 4 od nadawców); adopted-ACC liczone
na 4 adoptowanych. (Reszta protokołu jak L2/R.)

**Warianty na tych samych seedach 0–4:** CEILING (homogeniczny),
R1b-RBF (hetero + translator RBF), SANITY (R1b-RBF, wspólny backbone),
oraz R1b-Ridge (ablacja liniowa). R0 (podłoga anchor-only) przeniesione
z v1 jako odniesienie.

---

## Kryteria werdyktu (Z GÓRY) i BRAMKA FALSYFIKACJI

1. **BRAMKA (zasada wyjścia, zamrożona):** jeśli **R1b-RBF na SANITY nie
   osiągnie min 65%** adopted ACC (blisko CEILING ~73.4%), **seria R na
   tym poziomie abstrakcji (kotwica-interlingua) zostaje SFALSYFIKOWANA
   i NIE wchodzi do CLAIMS.md.** Zamiast headline'u: uczciwy negatyw
   („wspólna przestrzeń słów nie jest wystarczającą interlingua do
   re-materializacji cech; heterogeniczny kolektyw wymaga wspólnej bazy
   cech lub innego mostu"). Domyka oś rewolucji na tym poziomie.
2. **Jeśli SANITY ≥ 65% (bramka zdana):** dopiero WTEDy interpretujemy
   heterogeniczność:
   - R1b-RBF vs CEILING: SZUM → heterogeniczność darmowa (R-mild)
     = headline rewolucji dla tego poziomu; SYGNAL− → zmierzona cena.
   - R1b-RBF vs R0: SYGNAL+ oczekiwany (translator wnosi ponad kotwicę).
   - Dopiero po zdanym R-mild → osobna pre-rejestracja **R-hard**
     (losowy ↔ pretrained).
3. **Obserwacja:** R1b-Ridge vs R1b-RBF (czy liniowy translator starcza);
   błąd rekonstrukcji translatora na klasach dzielonych vs adopted-ACC.

Progi zamrożone: bramka 65% (SANITY), próg szumu std+std dla par.
Min-par raportowany. Negatyw (w tym trafienie bramki) = wynik.

## Pliki, koszt, ryzyko

- Runner: `src/run_R1b_translate.py` (nowy). Translator: nowa klasa
  (kernel ridge / Ridge) w `src/mars_collective_hetero.py`-sąsiedztwie
  (nowy plik; `adopt_classes`, eksport anchorowy, rdzeń NIETKNIĘTE —
  translator podmienia tylko dekoder MLP w ścieżce re-materializacji).
- Wynik: `results/R1b_translate.json` (smoke: `_smoke`).
- Koszt: niski–średni (kernel ridge zamknięty/tani; cache resnet z L;
  wzór R v1 — v1 FULL 193 s). Smoke: 1 seed.
- Ryzyko: bramka może trafić (translacja przez 50-d kotwicę może być
  fundamentalnie za stratna) — to jest właśnie falsyfikowalny test,
  nie porażka procesu.

## Instancjacja — RUNNER GOTOWY (2026-07-23, zielone światło Roberta)

Nowe pliki (rdzeń I/L NIETKNIĘTY; `adopt_classes` i eksport anchorowy
nietknięte — translator podmienia tylko materializację):
- `src/mars_translate.py` — `RidgeTranslator` (liniowy, zamknięta forma)
  i `KernelRidgeTranslator` (RBF, γ = heurystyka mediany, wsparcie ≤400),
  wyjście clamp_min(0). Smoke CPU w `__main__`.
- `src/mars_collective_hetero.py` — dopięte: `train_decoder_on` (oracle
  na 10 realnych), `adopt_classes_translate` (materializacja
  translatorem), wspólny `_adopt_via` → istniejące `adopt_classes`.
- `src/run_R1b_translate.py` — split zapięty: CAL=[0,1,2,3] dzielone,
  OWN=[4,5], ADOPT=[6,7]+[8,9] (metryka). 7 wariantów: CEILING, R0,
  ORACLE_SANITY, ORACLE_HET, RBF_SANITY, RBF_HET, RIDGE_HET. λ z gridu
  {1e-2,1e-1,1} wybierana po rekonstrukcji na held-out klas dzielonych
  (bez wycieku). Bramka liczona na RBF_SANITY; werdykty RBF_HET vs
  CEILING/R0 tylko po zdanej bramce.

Komendy (u Roberta): `python src/mars_translate.py` (smoke translatora,
CPU) → `python src/run_R1b_translate.py --smoke` →
`python src/run_R1b_translate.py` (FULL). Wynik:
`results/R1b_translate.json`.

## Zasady

5 seedów, progi i bramka 65% zamrożone PRZED runem, negatyw = wynik.
Werdykty → DROGA_R_NOTATKI.md (dopisek R1b), CLAIMS/WHITEPAPER TYLKO po
zdanej bramce. Runnera NIE piszemy przed zatwierdzeniem tego planu przez
Roberta.
