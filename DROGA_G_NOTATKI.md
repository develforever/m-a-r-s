# Droga G — notatki robocze

Plan: `DROGA_G_PLAN.md` (grounding językowy; pomysł użytkownika 06.07.2026).

## G1 — Prototypy semantyczne (ZAKOŃCZONE, 07.07.2026)

Pliki: `src/mars_cl_semantic.py`, `src/run_G1_semantic.py`;
wyniki: `results/G1_semantic.json`. 5 seedów × 15 epok/zadanie,
backbone losowy zamrożony (8,16), prototypy = wektory GloVe 50d (a priori).

| Wariant (class-IL) | Fashion ACC | MNIST ACC | Forgetting F/M |
|---|---|---|---|
| g1_t0 (proj z task0) | 17.89 ± 2.26% | 18.28 ± 1.91% | 14.8 / 12.5pp |
| g1_seq (proj douczana) | 19.89 ± 0.01% | 19.38 ± 0.16% | **98.5 / 98.6pp** |
| **g1_all (diagnostyka)** | **80.45 ± 0.86%** | 78.32 ± 2.02% | 8.7 / 10.4pp |
| [F0] replay-200 | 76.97 ± 1.09% | 88.81 ± 1.06% | 27.0 / 13.5pp |
| [F1] F1d (NCM) | 60.19 ± 5.19% | 66.23 ± 2.73% | 16.0 / 10.9pp |

**WERDYKT formalny (uczciwe warianty): SYGNAL−** — kotwice semantyczne BEZ
mechanizmu ochrony projekcji są gorsze od NCM (t0: wąska projekcja się nie
przenosi; seq: dryf = katastrofa 98.5pp — prawo D5 w czystej postaci).

**Ale diagnostyka przesądza o wartości kierunku:**
1. **g1_all na Fashion POBIŁ replay-200: 80.45 vs 76.97 (+3.5pp, min
   per-seed 79.20 > replay mean)** — na losowym backbone, bez jednego
   przechowanego obrazu, z prototypami istniejącymi przed danymi. Plateau
   F2 (60–67%) pęka, gdy przestrzeń routingu jest semantyczna. Górna
   granica podejścia = 80%+; cała gra o sekwencyjne dojście do niej = F3.
2. **Asymetria Fashion/MNIST mówi COŚ o groundingu:** g1_all wygrywa na
   Fashion (nazwy ubrań mają strukturę semantyczną skorelowaną z wyglądem:
   koszula↔płaszcz), a przegrywa z replay na MNIST (78 vs 89 — słowa
   "four"/"nine" nie niosą semantyki WIZUALNEJ cyfr). Grounding działa
   tam, gdzie język koduje podobieństwo wizualne. Obserwacja do papieru.
3. **Zero-shot: NEGATYWNY i pouczający** — ZS routing unseen 5.9% < losowe
   10%: projekcja z 2 klas nie jest neutralna wobec klas niewidzianych,
   tylko AKTYWNIE je przyciąga do znanych słów. Geometria GloVe sama się
   nie "zaczepia" — potrzebne kotwiczenie na wielu klasach (→ F3).

**Następny krok: F3** (`run_F3_feature_replay.py`) — douczanie projekcji
z parametrycznym "snem" cech starych klas (Gaussiany, ~1 KB/klasę).
Hipoteza: f3_sem zbiega do poziomu g1_all sekwencyjnie. Cel na Fashion:
> 76.97% (replay) przy zerowym buforze danych.

## G2 — Kompozycyjny zero-shot przez atrybuty (ZAKOŃCZONE, 08.07.2026):
## SZUM/NEGATYWNY z częściowym sygnałem i potwierdzoną strukturą

Plik: `src/run_G2_compositional.py`; wyniki: `results/G2_compositional.json`.
11 ręcznych atrybutów słownych, leave-one-out 10 klas × 5 seedów,
uczenie pojęć (BCE per atrybut, styl DAP) na losowym zamrożonym backbone.

**Wynik:** średni ZS 3.2% (osiągalne strukturalnie: 4.5%) przy progu 30%
— WERDYKT NEGATYWNY. Częściowy sygnał: Sneaker 18.2 ± 6.2%, T-shirt
10.2 ± 6.9% (>0, duża wariancja); reszta ~0%.

**Trzy ustalenia:**
1. **Reguła osiągalności strukturalnej potwierdzona w 3/3:** klasy
   z atrybutem unikalnym (Sandal/Bag/AnkleBoot) mają dokładnie 0.0%
   na wszystkich seedach — przewidziane z góry.
2. **Lekcja metodyczna "CE uczy klas, BCE uczy pojęć":** v1 (CE po
   prototypach) dawała ZS=0 przy 80% na widzianych — softmax odpycha
   od nieobecnych prototypów. Zapisana w kodzie.
3. **Trade-off seen/unseen zmierzony wprost:** 15 epok vs 4 epoki —
   widziane 45→70%, ale ZS SPADŁ (Sneaker 57→18%). Dłuższy trening
   koreluje detektory pojęć z kombinacjami widzianymi (znany w ZSL
   bias-to-seen). Kompozycyjność wymaga: (a) kodów klas z zapasem
   dystansu Hamminga (upper-body różni się 1 bitem — ECOC jako zasada
   projektowa słownika), (b) dekorelacji detektorów, (c) cech lepszych
   niż losowe. G2b (rozszerzony słownik ~16 atrybutów, dystans ≥3)
   — future work, pre-rejestrowane pokrętła znane.

**Interpretacja dla serii:** kompozycyjność z opisu na losowych cechach
i ręcznym słowniku NIE przechodzi progu — ale mechanizm daje mierzalny
sygnał tam, gdzie kod jest odległy, a porażki są przewidywalne z czystej
struktury słownika. Do v0.3 jako negatyw z wyznaczoną drogą.

## G3 — kompozycyjność na cechach pretrained (ZAKOŃCZONE, 2026-07-23): G3− TWARDY — DŹWIGNIA (c) SFALSYFIKOWANA, SERIA G DOMKNIĘTA

Plan: `DROGA_G3_PLAN.md`. Run u Roberta, `results/G3_pretrained.json`.
Runner: `src/run_G3_pretrained_compositional.py` (`run_holdout` z G2
werbatim; cechy Fashion→ResNet18-ImageNet 512-d). 2×2 na tych samych
5 seedach.

| Backbone / słownik | średni ZS |
|---|---|
| random / attrs11 | 3.17 ± 0.90% |
| random / attrs21 | 0.18 ± 0.08% |
| **pretrained / attrs11** | **2.77 ± 0.15%** |
| pretrained / attrs21 | 0.08 ± 0.02% |

**WERDYKT BINARNY: G3−** (`zs_pretrained_best` = 2.77% ≪ linia 30%).

### Rozstrzygnięcie: cechy NIE są dźwignią (przewidywanie obalone)

Wchodziliśmy z mocnym priorem „cechy są wąskim gardłem" (L +37pp, K).
**Obalone.** T1 (pretrained vs random, attrs11): **SZUM**, pary
−0.40 ± 0.88pp, **ratio 0.87** — cechy ImageNet są statystycznie
równoważne losowym, nominalnie odrobinę GORSZE. Silna reprezentacja nie
przesuwa kompozycyjnego zero-shot ani o krok. To domyka trójdzielną
diagnozę G2 „(a) dystans, (b) dekorelacja, (c) cechy": (a) backfire
(G2b), **(c) bez efektu (G3)** — obie zewnętrzne dźwignie wyczerpane.

### Wąskim gardłem jest PODEJŚCIE, nie reprezentacja

Wniosek serii: kompozycyjny zero-shot z RĘCZNEGO słownika atrybutów +
LINIOWYCH detektorów pojęć + routingu po kodzie atrybutów jest
ograniczony przez sam paradygmat, niezależnie od jakości cech i dystansu
kodowego. Detektory per-atrybut pozostają zawodne nawet na cechach
ImageNet — prawdopodobnie bo (i) ręczne atrybuty („ma_rekawy",
„jest_obuwiem") nie są liniowo kodowane w przestrzeni ResNet dla
Fashion (domena szara 28→224 daleka od ImageNet), (ii) trade-off
bias-to-seen jest własnością mechanizmu routingu po kodzie, nie cech.

Potwierdzenia spójności:
- **T4 (sanity):** random odtwarza G2b co do liczby (attrs11 3.17%,
  attrs21 0.18%) — harness ważny, porównanie uczciwe.
- **T3 (interakcja a×c):** ECOC na pretrained nadal szkodzi
  (attrs21 vs attrs11: −2.69 ± 0.14pp, **SYGNAL−**). Mechanizm z G2b
  potwierdzony od drugiej strony: skoro cechy nie obniżyły błędu per-bit
  detektorów, kod korekcyjny dalej nie ma czego korygować i backfire
  zostaje. „ECOC pomaga, gdy błąd per-bit niski" — na tym zadaniu błąd
  NIGDY nie jest niski, więc ECOC nigdy nie pomaga.
- **T2 (osiągalność):** per-klasa na pretrained bez zmian — sygnał tylko
  na dawnych łatwych (T-shirt ~8, Sneaker ~12), dawne porażki
  strukturalne {Sandal, Bag, AnkleBoot}=5/8/9 nadal 0.0%. Mocne cechy
  nie otwierają klas zamkniętych strukturą słownika.

### Domknięcie serii G

Trzy wyniki układają się w spójną granicę: **G2** (negatyw, reguła
strukturalna), **G2b** (dystans kodowy backfire — nie słownik), **G3**
(cechy bez efektu — nie reprezentacja). Kompozycyjny zero-shot z ręcznego
słownika jest sfalsyfikowany na wszystkich trzech dźwigniach zewnętrznych;
dalsza droga (gdyby ktoś chciał) to INNY paradygmat — atrybuty UCZONE
end-to-end / nieliniowe detektory — nie ten. Seria G ZAMKNIĘTA jako
granica podejścia. CLAIMS 15c; WHITEPAPER sekcja 13.
