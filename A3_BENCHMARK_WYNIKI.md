# M.A.R.S. — Droga A, krok A3: pomiar zbiorczy + przegląd strategiczny

Data: 2026-06-15
Status: Droga A zmierzona w pełni. Czas na uczciwą ocenę.

---

## Tabela zbiorcza (MNIST, GTX 1050 Ti)

| System | accuracy | MAC | throughput | oszczędność |
|---|---|---|---|---|
| Baseline monolit | 97.7% | 234,752 | 3,184,888 | — |
| Redundantny (h=64) | 97.4% | 63,568 | 1,989,472 | 72.9% |
| Specjaliści (h=24) | 96.1% | 44,816 | 1,971,213 | 80.9% |

---

## Uczciwy obraz: gdzie M.A.R.S. wygrywa, a gdzie nie

Na MNIST (małe pody, łatwe zadanie):
- accuracy: monolit 97.7% > specjaliści 96.1% — **monolit lepszy**
- throughput: monolit 3.18M > specjaliści 1.97M (0.62×) — **monolit lepszy**
- MAC/energia: specjaliści 80.9% mniej — **M.A.R.S. lepszy**

**Na małym zadaniu M.A.R.S. wygrywa TYLKO energię.** To trzeba nazwać wprost.

---

## Ale to jest kwestia BENCHMARKU, nie architektury

Z Etapu B wiemy: przewaga throughput rośnie z rozmiarem poda.
- hidden=24 (MNIST): 0.62× — przegrywa
- hidden=2048: 2.57× — wygrywa

MNIST to zabawka. Monolit (235k MAC) jest banalnie mały i szybki, więc
M.A.R.S. nie ma jak pokazać przewagi. Architektura M.A.R.S. jest
projektowana dla SKALI, gdzie:
- pody są duże (eksperci po tysiące neuronów),
- jedno zadanie nie mieści się w monolicie,
- energia realnie kosztuje.

Na MNIST tego nie widać. To nie znaczy, że projekt zawiódł — znaczy,
że MNIST jest złym benchmarkiem do pokazania jego przewagi.

---

## DWA UCZCIWE WNIOSKI (oba prawdziwe)

### Co M.A.R.S. UDOWODNIŁ (zmierzone, solidne):
1. Prawdziwa modularność: router = serce systemu (routing acc = system acc)
2. Specjalizacja działa: 62% mniejsze pody, accuracy w granicach 1.6pp
3. Oszczędność MAC: 80.9% mniej operacji (realne dla energii)
4. Sufit specjalistów wysoki: ORACLE 99% (router jest jedynym ograniczeniem)
5. Throughput: wektoryzacja działa, przewaga czasowa od hidden≈2048

### Czego M.A.R.S. NIE udowodnił (uczciwie):
1. Na MNIST nie bije monolitu w accuracy ani czasie — tylko w energii
2. Przewaga czasowa wymaga dużej skali, niezweryfikowanej end-to-end
3. To wciąż "tylko" sprawny MoE z prototypowym routingiem — nie przełom

---

## DECYZJA: co dalej (przed A4)

Mamy trzy uczciwe drogi:

**Droga 1 — pokazać M.A.R.S. w jego reżimie (duże pody).**
Zbudować zadanie/model, gdzie pody mają hidden≥1024, i pokazać, że TAM
M.A.R.S. bije monolit na accuracy I throughput I energii naraz. To
najmocniejszy możliwy dowód tezy. Większa praca.

**Droga 2 — domknąć MNIST (top-2 routing, sufit 99%).**
Tani trik: sprawdzać 2 najbliższe pody zamiast 1. Może podnieść z 96.1%
do bliżej 99% (sufit ORACLE). Pokazałby M.A.R.S. dorównujący monolitowi
w accuracy przy oszczędności energii. Szybkie.

**Droga 3 — przejść do A4 (catastrophic forgetting).**
To jest pole, gdzie M.A.R.S. ma NATURALNĄ przewagę (modularność chroni
przed zapominaniem), a monolit jest z natury słaby. Tu M.A.R.S. może
wygrać czysto, bo to jego mocna strona.

REKOMENDACJA: Droga 3 (A4). Bo catastrophic forgetting to miejsce, gdzie
modularność daje przewagę NIEMOŻLIWĄ dla monolitu — a nie tylko "mniej
energii za cenę accuracy". To najmocniejsza karta M.A.R.S.
