# Droga R — kolektyw heterogeniczny (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: **DO ZATWIERDZENIA**; runy
WYŁĄCZNIE u Roberta. Branch: `droga-r` (nowe pliki, istniejący kod
NIETKNIĘTY). Oś: rewolucja — protokół wymiany snów w pełni
**representation-agnostic**. Szkic mechanizmu: `ARSENAL_...md`
(Dopisek 2026-07-23). Wejście po osobnej zgodzie Roberta.

## Punkt wyjścia (zmierzony)

Kolektyw HOMOGENICZNY dowieziony: I3 (10 klas / 5 agentów, równoważność
z seq, SZUM), L2 (mocne cechy, koszt protokołu −0.56pp), Q (100 klas /
20 agentów, równoważność przy symetrycznym budżecie snu). Wspólne
założenie WSZYSTKICH: ten sam seed/backbone → wspólna przestrzeń cech.
R zdejmuje to założenie.

## Hipoteza

H-R: agenci o RÓŻNYCH backbone'ach mogą dzielić klasy, jeśli payload jest
wyrażony w przestrzeni KOTWIC (wspólnej z definicji), a każdy odbiorca
re-materializuje adoptowaną klasę własnym dekoderem `anchor → feature`
uczonym wyłącznie na swoich widzianych klasach. Mechanizm: kotwica to
interlingua; dekoder to prywatna inwersja projekcji. `adopt_classes`
nietknięte — sen/projekcja/pody działają na re-materializowanych cechach.

## Warianty (jeden runner, te same seedy 0–4)

- **R0 — podłoga (kontrola „czy sama kotwica wystarczy"):** odbiorca
  adoptuje klasę c ustawiając wyłącznie proto_c = word_vec_c, BEZ
  re-materializacji i douczenia projekcji na c. Przewidywanie z góry:
  porażka (G1: projekcja bez douczenia przyciąga niewidziane do znanych,
  ZS 5.9% < 10%).
- **R1 — mechanizm (kotwica-interlingua + dekoder per-agent):** payload
  w przestrzeni kotwic; `dec_B: anchor → feature_B` uczony na własnych
  klasach odbiorcy; sen zdekodowanych pseudo-cech → `adopt_classes`.
- **CEILING — kolektyw homogeniczny:** ten sam mechanizm przy wspólnym
  backbone (liczba z I3/Q wg benchmarku) — górna kotwica, do której R1
  dąży.
- **SANITY — heterogeniczność zerowa:** R1 przy backbone_A = backbone_B
  (ten sam seed) musi zredukować się do CEILING (reprodukcja, wzór
  sanity attrs11 z G2b / J1).

## Drabina heterogeniczności (dwa poziomy, oba pre-rejestrowane)

- **R-mild:** ten sam resnet18-ImageNet, RÓŻNY seed projekcji 512→128
  (feature_A, feature_B = dwa losowe rzuty tej samej 512). Translacja
  liniowa istnieje z konstrukcji. Cel: dekoder działa (mechanizm żyje).
- **R-hard:** backbone losowy-od-zera ↔ pretrained resnet18. Różna treść
  informacyjna, brak wspólnej bazy cech. Cel: czy kotwica jest
  wystarczającą interlingua bez wspólnych cech.

## Kryteria werdyktu (Z GÓRY)

Metryka główna: **ACC klas ADOPTOWANYCH u ODBIORCY** (nie własnych),
zawsze względem LOKALNEGO sufitu odbiorcy w JEGO reżimie zasobów.
Pary per-seed, próg szumu std+std, min-par raportowany.

1. **R1 vs R0** (czy dekoder cokolwiek wnosi ponad samą kotwicę):
   SYGNAL+ → translacja przez przestrzeń cech niesie informację ponad
   tożsamość słowną; SZUM/− → kotwica sama nie starcza, a dekoder nie
   dokłada (kierunek: mechanizm martwy, raportować na którym poziomie).
2. **R1 vs CEILING homogeniczny** (cena heterogeniczności):
   - SZUM → **heterogeniczność jest darmowa: protokół representation-
     agnostic** (headline rewolucji, jeśli padnie na R-hard);
   - SYGNAL− → zmierzona CENA heterogeniczności (nadal wynik: „ile
     kosztuje brak wspólnego backbone'u");
   - raportować OSOBNO dla R-mild i R-hard (dwa różne twierdzenia).
3. **SANITY**: R1 przy wspólnym backbone = CEILING w szumie; inaczej
   błąd implementacji (blokuje interpretację 1–2).
4. **Obserwacja strukturalna:** jakość dekodera (błąd rekonstrukcji
   `dec_B(proj_B(f)) vs f` na widzianych klasach) vs ACC adoptowanych —
   czy translacja gubi geometrię wewnątrzklasową (przewidywane ryzyko).

## Reżimy zasobów (zasada bez zmian)

R-hard krzyżuje backbone losowy (from-scratch) z pretrained (foundation).
NIGDY nie porównujemy liczby from-scratch z liczbą foundation wprost:
każde pairing oceniane względem sufitu ODBIORCY w jego reżimie. Kierunek
transferu (kto nadaje, kto adoptuje) raportowany jawnie — asymetria
losowy→pretrained vs pretrained→losowy to osobna obserwacja.

## Plik, koszt, ryzyko

- Runner: `src/run_R1_heterogeneous.py` (nowy); nowa klasa dekodera
  w `src/mars_collective.py`-sąsiedztwie (nowy plik, `adopt_classes`
  i eksport nietknięte — payload-kotwicowy jako osobna ścieżka).
- Wynik: `results/R1_heterogeneous.json` (smoke: `_smoke`).
- Koszt: średni (dwa backbone'y, cache cech resnet18 z L istnieje;
  dekoder = mała sieć; wzór L+I na 1050 Ti — rząd
  kilkunastu–kilkudziesięciu minut FULL). Smoke: 1 seed, R-mild.
- Ryzyko: WYSOKIE — dekoder anchor→feature jest wiele-do-jednego;
  inwersja przybliżona może nie oddać struktury wewnątrzklasowej.
  Negatyw R-hard = uczciwe twierdzenie graniczne, nie porażka projektu.

## Instancjacja R-mild — ZAPIĘTA (runner gotowy, 2026-07-23)

Poziom R-mild zapięty w kodzie (nowe pliki, rdzeń I/L NIETKNIĘTY):
- `src/mars_collective_hetero.py` — `MarsCollectiveHetero(MarsCollective)`:
  `export_anchor_payload` (Gauss diag mean/var po embeddingu anchorowym
  klasy, wymiar słów E=50), `AnchorDecoder` (MLP E→256→256→D, głowa ReLU
  → cechy nieujemne), `train_decoder` (MSE anchor→feature na WŁASNYCH
  klasach odbiorcy), `adopt_classes_hetero` (dekoduj → `FeatureStatsKSparse`
  → **wywołanie istniejącego `adopt_classes` bez zmian**),
  `adopt_classes_anchor_only` (podłoga R0: proto + stały pod ufający
  kotwicy, nie crashuje `forward`). Smoke syntetyczny CPU w `__main__`.
- `src/run_R1_heterogeneous.py` — 4 warianty (CEILING/R0/R1/SANITY) na
  seedach 0–4, `ReducedBackbone` z RÓŻNYM seedem nadawca vs odbiorca
  (R0/R1), metryka = ACC klas adoptowanych (row[1:]) + overall.
- Heterogeniczność R-mild = ten sam cache resnet18 (512), różny seed
  projekcji 512→128. **R-hard** (losowy-od-zera ↔ pretrained: różne
  wejścia, 3072 px vs 512) = OSOBNY runner, po walidacji R-mild.
- Decyzje projektowe zapięte przed runem (część pre-rejestracji): payload
  anchorowy = Gauss diag (przestrzeń słów jest gęsta/ciągła po projekcji,
  nie rzadka — spike-and-slab tu nie pasuje); dekoder z głową ReLU
  (parytet z cechami po backbone); budżet dekodowania = n_dream.

Komendy (u Roberta): `python src/mars_collective_hetero.py` (smoke
wiring, CPU) → `python src/run_R1_heterogeneous.py --smoke` →
`python src/run_R1_heterogeneous.py` (FULL). Wynik:
`results/R1_heterogeneous.json`.

## Zasady

5 seedów, progi jw. zamrożone, min-par raportowany, negatyw = wynik,
kierunki i poziomy (mild/hard) raportowane osobno. Werdykty →
DROGA_R_NOTATKI.md (nowy), potem CLAIMS/WHITEPAPER.
