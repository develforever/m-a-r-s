# Droga P — notatki wyników (P1: detekcja zatrucia, pretrained vs random)

Run FULL: 2026-07-23, 146 s, 5 seedów, CUDA. Plik: `results/P1_detect_pretrained.json`.
Plan: `DROGA_P_PLAN.md` (kryteria zamrożone przed runem). Smoke bez anomalii.

## WERDYKT (wg pre-rejestracji): PODWÓJNY NEGATYW

| Detektor | pretrained: pełna separacja | pretrained: clean-vs-swap | random (kontrola) | Werdykt |
|---|---|---|---|---|
| D1 rank_consistency | NIE | NIE | NIE | **NEGATYW** |
| D2 canary_probe | NIE | NIE | NIE | **NEGATYW** |

**Semantyka cech nie wystarcza do detekcji wewnątrzpayloadowej w
pre-rejestrowanym kształcie.** Twierdzenie I4 („detekcja prawdopodobnie
wymaga pretrained") jest sfalsyfikowane w swojej prostej formie: samo
podmienienie backbone'u na resnet18-ImageNet nie daje separacji clean
od obu ataków żadnym z dwóch detektorów. Negatyw = wynik; wątek
detekcji w kształcie I4/P1 domknięty.

## Surowe wartości detektorów (5 seedów)

pretrained:
- D1: clean [0.809, 0.762, 0.762, 0.857, 0.762] | swap [0.881, 0.833,
  0.809, 0.786, 0.809] | noise [−0.119, 0.095, 0.214, 0.191, −0.024]
- D2 [pp]: clean [0.93, 2.09, 1.60, 2.90, 1.40] | swap [0.29, 0.48,
  0.54, 1.13, 0.78] | noise [1.80, 2.78, 2.44, 2.50, 2.18]

random:
- D1: clean [0.952, 0.857, 0.762, 0.667, 0.881] | swap [0.881, 0.548,
  0.452, 0.619, 0.809] | noise [0.286, 0.238, −0.095, −0.262, 0.167]
- D2 [pp]: clean [4.40, 4.18, 4.34, 4.29, 5.19] | swap [2.46, 2.23,
  1.78, 1.33, 1.76] | noise [4.48, 1.70, 2.33, 2.30, 1.59]

## Obserwacje POST-HOC (nie werdykty; kandydaci do osobnej pre-rejestracji)

1. **D1 w pełni rozdziela noise od clean na OBU podłożach** (5/5,
   min clean 0.762 > max noise 0.214 na pretrained; 0.667 > 0.286 na
   random). Nie było to pre-rejestrowane kryterium (I4 wymagał separacji
   od OBU ataków), więc raportujemy jako obserwację. Interpretacja:
   D1 wykrywa BRAK STRUKTURY KLASOWEJ payloadu, nie podmianę etykiety.
   Kontrast z I4/Fashion (random): tam clean miał D1 ≈ 0 (brak korelacji
   cechowo-słownej) — na CIFAR wysoka zgodność rankowa clean występuje
   już na losowym backbone (własność danych, nie semantyki cech —
   spójne z anomalią kontrolną, której formalnie nie było, bo swap
   pozostaje nieodseparowany).
2. **Swap jest niewykrywalny dla D1 prawdopodobnie DLATEGO, że klasy
   współadoptowane są semantycznie bliskie** (ship↔truck: bliskie
   kotwice → payload truck zachowuje ranking ship; D1 swap ≈ clean,
   wręcz nominalnie wyżej). Hipoteza P1c-b: swap klas ODLEGŁYCH
   (np. ship↔bird) powinien łamać ranking.
3. **D2 ma kierunek ODWROTNY niż pre-rejestrowany**: swap daje NIŻSZY
   spadek kanarkowy niż clean (na random wręcz pełne rozdzielenie
   w złą stronę: max swap 2.46 < min clean 4.17). Mechanizm
   (hipoteza): uczciwa adopcja dwóch nowych klas odbiera własnym
   klasom więcej masy routingu niż adopcja sprzeczna, która
   „kanibalizuje" głównie współadoptowaną parę. Post-hoc — wymaga
   pre-rejestracji na świeżych seedach, jeśli ma być twierdzeniem.
4. **Mapa szkody spójna z I4 na obu podłożach** (obserwacja):
   swap niszczy obie klasy paczki (pretrained: acc8 −68.8pp SYGNAL−,
   acc9 −87.6pp SYGNAL−; random analogicznie), klasy własne odporne
   (acc_own wręcz +2.0/+3.4pp). Nowość: na pretrained payload noise
   „częściowo działa" (acc8 56.9 ± 6.6 vs 10.0 ± 1.9 na random) —
   semantyczne cechy + kotwica dociągają nawet śmieciowe statystyki
   do połowicznej klasy. Do przemyślenia przy polityce protokołu:
   na mocnych cechach śmieć jest MNIEJ widoczny w acc.

## Status serii P i kandydaci

P1 domknięte (negatyw ×2, zgodnie z planem — bez P2/kwarantanny).
Kandydaci do decyzji Roberta (każdy z osobną pre-rejestracją,
NOWE seedy, progi z góry):
- **P1c-a**: „detektor śmieci" — D1 z progiem jako test struktury
  klasowej payloadu (obserwacja 1 → twierdzenie).
- **P1c-b**: wykrywalność swap vs dystans semantyczny pary
  (obserwacja 2; oś: pary bliskie/średnie/odległe).
- **P1c-c**: D2 z odwróconym znakiem (obserwacja 3).
Alternatywa: zostawić detekcję (dwa uczciwe negatywy wystarczą do
narracji „adopcje paczkami + odwołanie"), wrócić do mapy: seria Q.

---

# P1c — WYNIKI (run FULL 2026-07-23, 275 s, seedy 5–9)

Plan: `DROGA_P1C_PLAN.md` (progi zamrożone przed runem, świeże seedy).
Plik: `results/P1c_gate_distance.json`. Smoke bez anomalii.

## WERDYKT a — brama strukturalna D1 (θ=0.45): **SUKCES**

60/60 poprawnych klasyfikacji na obu podłożach:

| Podłoże | min clean | max noise | Werdykt |
|---|---|---|---|
| pretrained | 0.667 | 0.000 | spełnione (15/15 clean > θ, 15/15 noise < θ) |
| random | 0.691 | 0.357 | spełnione (15/15, 15/15) |

**Twierdzenie: D1 z progiem 0.45 jest bramą odrzucającą payloady bez
struktury klasowej, niezależnie od reprezentacji.** Kontekst z metryki
pre-rejestrowanej: przez bramę przechodzi 5/15 swapów na każdym podłożu
— i jest to DOKŁADNIE 5 seedów swap_close (wszystkie swap_mid/swap_far
mają D1 ujemne, więc spadają pod próg). Brama łapie więc nie tylko
śmieć, ale i podmiany średnio- i daleko-dystansowe; przepuszcza
wyłącznie podmiany bliskie — te, które prawo b uznaje za niewykrywalne
z konstrukcji.

## WERDYKT b — prawo dystansu semantycznego: **SUKCES MOCNY**

Na OBU podłożach: pełna separacja clean-vs-swap_mid ORAZ clean-vs-
swap_far, brak separacji clean-vs-swap_close (5/5, min-max):

| Wariant (cos kotwic) | mediana D1 swap, pretrained | random | separacja |
|---|---|---|---|
| swap_close (airplane, 0.775) | +0.667 | +0.786 | NIE / NIE |
| swap_mid (horse, 0.487) | −0.643 | −0.714 | TAK / TAK |
| swap_far (deer, 0.139) | −0.595 | −0.833 | TAK / TAK |

**Prawo graniczne potwierdzone na końcach osi: wykrywalność podmiany
rośnie z dystansem semantycznym pary; podmiana klas bliskich jest
niewykrywalna z konstrukcji** (payload donora spełnia ranking
deklarowanej kotwicy). Z referencją P1 (truck 0.615, niewykrywalny):
próg wykrywalności leży między cos 0.615 a 0.487. Monotonia median:
random TAK; pretrained NIE (mid −0.643 vs far −0.595 — oba głęboko
ujemne, odwrócenie o 0.05 przy spełnionym kryterium głównym; sam
mechanizm rankingowy nie musi być monotoniczny poniżej progu łamania).
Mechanizm negatywów I4/P1 wyjaśniony: tam podmieniano pary bliskie
(ship↔truck), jedyne strukturalnie niewidzialne.

## Obserwacja c — znak D2 (bez rangi twierdzenia)

Replikacja kierunku z P1 na świeżych seedach: swap < clean w 4/5
(close) i 5/5 (mid, far) seedów na obu podłożach; median D2 clean
2.01pp (pretrained) / 5.96pp (random). Kandydat na pełną
pre-rejestrację odłożony — brama a + prawo b pokrywają potrzebę.

## POLITYKA PROTOKOŁU — komplet (wszystko zmierzone)

1. Adopcje paczkami, możliwość odwołania paczki źródła (I4).
2. Brama strukturalna D1 > 0.45 na wejściu — odrzuca śmieć (60/60)
   i podmiany o cos < ~0.5 (P1c-a/b).
3. Podmiany klas bliskich (cos ≳ 0.6) — niewykrywalne z konstrukcji
   (P1, P1c-b); pokrywa je wyłącznie naprawa zapomnij-i-adoptuj (I4b,
   pełna, bez historii).
Status serii P: KOMPLET (P1 negatyw ×2 + P1c sukces ×2 + obs. c).
