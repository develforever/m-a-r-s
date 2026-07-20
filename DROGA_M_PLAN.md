# Droga M — plan pre-rejestrowany: długi horyzont (loss-of-plasticity)

Data pre-rejestracji: 2026-07-20 (branch `droga-m`; main/v0.7 nietknięty).
Kontekst: ostatnie niezmierzone ryzyko z listy CL (analiza sesji
„Decentralized AI training architecture", 19.07): przy T=5 plastyczność
jest zmierzona i zdrowa (K2: R[t][t] 94–95%), ale loss-of-plasticity
(Dohare et al. 2024) ujawnia się dopiero przy dziesiątkach zadań.
Pytanie M: czy mechanizm (zamrożona reprezentacja + sen sparse +
kotwice) utrzymuje zdolność uczenia NOWYCH klas po 20 zadaniach?

## Setup

Split-CIFAR-100: 20 zadań × 5 klas (klasy 0–99 w porządku indeksów
torchvision, z góry ustalonym). Backbone: zamrożony resnet18-ImageNet
przez cache cech + losowa projekcja 512→128 (jak Droga L; linia
foundation — na losowych cechach sufit byłby za nisko, by pomiar
cokolwiek rozstrzygał). Sen sparse_k16, epochs=15, epochs_proj=15,
l2sp=0, LR=0.001, 5 seedów. CIFAR-100 ma 500 obrazów/klasę — dokładnie
na progu nasycenia payloadu z I2b (obserwacja do raportu).

Uogólnienie stosu do N=100 (nowe pliki, wierne kopie czterech metod
z jedyną zmianą wymiaru wyjścia; kotwice = GloVe, nazwy złożone
uśredniane po członach jak w konwencji CLASS_WORDS).

Warianty:
  m1_seq_300 : sekwencyjny, kotwice 300d (GŁÓWNY — 100 klas w 50d
               grozi stłoczeniem; J4/K1: 300d nie szkodzi seq)
  m1_all_300 : sufit zamrożonych cech (proj_train="all")
  m1_seq_50  : obserwacja — czy stłoczenie kotwic w 50d realnie boli

## Kryteria (Z GÓRY)

- **GŁÓWNE (plastyczność, m1_seq_300):** pary per-seed
  spadek = śr. R[t][t] zadań 1–5 − śr. R[t][t] zadań 16–20.
  Werdykt LOSS-OF-PLASTICITY jeśli śr. spadku > próg szumu
  (std+std) ORAZ wszystkie pary dodatnie; parowy analogicznie
  (wszystkie pary jednego znaku ORAZ |śr.| > 2×std delt);
  inaczej: PLASTYCZNOŚĆ UTRZYMANA. Raport pełnej krzywej R[t][t].
- Obserwacja 1: mech% sufitu = seq/all przy T=20 (kontekst: L1 dało
  96.7% przy T=5; inna liczba klas — bez werdyktu).
- Obserwacja 2: m1_seq_300 vs m1_seq_50 (pary) — koszt stłoczenia
  100 kotwic w 50d.
- Raport: ACC, forgetting, pamięć klas (100 × 24.1 KB = 2.41 MB),
  krzywa ACC po zadaniach.

## Ryzyka pre-rejestrowane

1. **Confound trudności:** klasy późnych zadań mogą być obiektywnie
   trudniejsze/łatwiejsze niż wczesnych (porządek indeksów, nie
   losowy) — spadek R[t][t] może odzwierciedlać trudność, nie utratę
   plastyczności. Mitygacja: krzywa pełna + porównanie z all_300
   per zadanie (sufit per task jako odniesienie trudności:
   plastyczność mierzona też jako R[t][t]/acc_all(t) — raportowane).
2. Routing w 100 klas: więcej kolizji najbliższej kotwicy —
   ACC bezwzględne będzie niższe niż na CIFAR-10; to nie jest
   przedmiotem werdyktu.
3. k-means k=16 na 500 próbkach/klasę — szumniejsze centroidy
   (fallbacki FeatureStatsKSparse).

## Kolejność uruchomień u Roberta

1. `python src/mars_cl_m.py` (smoke jednostkowy, CPU, sekundy)
2. `python src/run_M1_long_horizon.py --smoke`
   (pierwszy run: pobranie CIFAR-100 + jednorazowa ekstrakcja cech
   do cache — kilka minut)
3. `python src/run_M1_long_horizon.py` (FULL)

Kandydat M2 (NIE pre-rejestrowany tutaj): kolektyw 20 agentów × 5 klas
na CIFAR-100 — decyzja po werdykcie M1.

Wyniki do DROGA_M_NOTATKI.md; merge po komplecie i decyzji Roberta.
