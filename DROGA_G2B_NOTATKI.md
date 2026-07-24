# Droga G2b — słownik atrybutów z dystansem kodowym (ECOC) — notatki

Plan (pre-rejestracja): `DROGA_G2B_PLAN.md` (2026-07-23, zamrożony przed
runem). Run: WYŁĄCZNIE u Roberta, 2026-07-23 (noc). Runner:
`src/run_G2b_ecoc.py`; wyniki: `results/G2b_ecoc.json`
(smoke: `_smoke`). 5 seedów × 15 epok, leave-one-out 10 klas,
2 warianty słownika na tych samych seedach.

## Wynik (zmierzony)

| Wariant | min Hamming | średni ZS | seen-acc (zakres) |
|---|---|---|---|
| attrs11 (reprodukcja G2) | 1 | **3.17 ± 0.90%** | 0.65–0.76 |
| attrs21 (ECOC, zamrożony) | 4 | **0.18 ± 0.08%** | 0.72–0.81 |

Pary per-seed (attrs21 − attrs11, pp): −3.46 / −4.12 / −3.02 / −2.28 /
−2.03; średnia **−2.98pp**, próg szumu (std+std) 0.98pp, 5/5 ujemne.

## Werdykty (wg kryteriów z góry)

**Werdykt główny: NEGATYW — i to mocny.** attrs21 nie tylko nie
przekroczył progu 30% (kryterium 1) ani nie dał sygnału mechanizmu
(kryterium 2, wymagało ≥2× attrs11 — jest ~0.06×), ale **obniżył**
zero-shot poniżej attrs11 — para ujemna, |średnia| > próg szumu, 5/5.
Dystans kodowy nie jest dźwignią zero-shot na losowych cechach; jest
przeciw-dźwignią.

**Test reguły osiągalności: PRZEWIDYWANIE 10/10 SFALSYFIKOWANE.**
Słownik ATTRS21 był zaprojektowany tak, że żaden atrybut nie jest stały
w żadnym leave-one-out (kolumny 2–8 jedynek) — reguła osiągalności
z G2 przewidywała wtedy 10/10 klas osiągalnych, w tym dawne porażki
strukturalne {Sandal, Bag, AnkleBoot}. Zmierzono: **6 klas nadal
dokładnie 0.0%** na wszystkich seedach (Trouser, Pullover, Coat,
Sandal, Bag, AnkleBoot), w tym całe {Sandal, Bag, AnkleBoot}. Jedyna
klasa z niezerowym ZS to Shirt (1.74%) — ta sama, która niosła resztkę
sygnału w attrs11. **Wniosek: osiągalność strukturalna jest warunkiem
KONIECZNYM, nie wystarczającym.** Usunięcie atrybutów stałych otwiera
klasę geometrycznie, ale nie sprawia, że detektory pojęć potrafią ją
trafić.

**Obserwacja (trade-off seen/unseen, zgodna z G2, wzmocniona):**
21 atrybutów podniosło seen-acc (0.72–0.81 vs 0.65–0.76 dla attrs11)
przy jednoczesnym zapadnięciu ZS. Więcej atrybutów = więcej pojemności
na zapamiętanie kombinacji WIDZIANYCH = silniejszy bias-to-seen. Ta
sama oś, którą G2 zmierzyło epokami (45→70% seen, 57→18% ZS), tu
pojawia się w wymiarze słownika.

## Mechanizm (dlaczego więcej dystansu ZASZKODZIŁO)

Teoria ECOC gwarantuje korekcję ⌊(d−1)/2⌋ błędów bitowych **pod
warunkiem**, że błąd per-bit jest poniżej progu. Na losowym zamrożonym
backbone detektory pojęć per-atrybut są blisko losowe — błąd per-bit
jest wysoki. W tym reżimie wydłużenie kodu (11→21 bitów, d: 1→4) nie
koryguje: każdy dodatkowy zaszumiony bit MNOŻY prawdopodobieństwo, że
zdekodowane słowo wypadnie poza właściwą klasą. Dłuższy kod = trudniej
trafić dokładny wektor docelowy przy tej samej jakości detektorów.
To domyka trójdzielną diagnozę z G2 („(a) dystans kodowy, (b)
dekorelacja detektorów, (c) cechy lepsze niż losowe"): dźwignia (a)
przetestowana W IZOLACJI i **backfire** → wąskie gardło to (c) — cechy,
nie słownik. Testowanie dystansu bez naprawy detektorów było testowaniem
kodu korekcyjnego nad kanałem powyżej pojemności Shannona.

## Status słownika i wariantu attrs11

- attrs11 odtworzył G2 (3.17% ≈ 3.2%) — sanity reprodukcji zaliczony,
  wzór precedensu J1/J3/J4.
- ATTRS21 pozostaje zamrożony w `DROGA_G2B_PLAN.md` jako dowód
  pre-rejestracji; własności macierzy (min Hamming 4, brak stałych
  kolumn, brak duplikatów) potwierdzone asercjami przy starcie runnera.

## Domknięcie serii G i kierunek

Kompozycyjny zero-shot z opisu atrybutowego jest **zablokowany na
cechach, nie na słowniku** — teraz zmierzone, nie założone. Ścieżka
naprzód (patrz `PLAN_V2.md`): **G3 — ten sam eksperyment na cechach
pretrained (resnet18-ImageNet, wzór L)**, izolujący dźwignię (c). Dwa
możliwe wyniki, oba domykają: G3+ → negatyw G2/G2b był w całości
deficytem cech (kompozycyjność wymaga semantyki reprezentacji);
G3− → kompozycyjność z ręcznego słownika nie przechodzi nawet na
mocnych cechach (granica podejścia, nie realizacji). Opcjonalny
wariant (b): dekorelacja detektorów (kara na korelację wyjść BCE) —
tania, ale drugorzędna wobec (c).

CLAIMS: 15b. WHITEPAPER: sekcja 13 (dopisek G2b) + linia roadmapy.
