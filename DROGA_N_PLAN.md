# Droga N — plan pre-rejestrowany: selektywne zapominanie z gwarancją

Data pre-rejestracji: 2026-07-20 (branch `droga-n`; main/v0.8 nietknięty).
Motywacja (analiza wywiadu Leahy'ego, 20.07): w monolitach „oducz się
klasy X" to otwarty problem frontierowy (machine unlearning; wymóg
prawny right-to-forget) — wiedza jest rozsmarowana w wagach. W M.A.R.S.
pamięć klasy jest adresowalna (payload + pod + kotwica), więc usunięcie
POWINNO być operacją. Ale uczciwość wymaga dwóch poziomów: usunięcie
z routingu jest trywialne z konstrukcji; prawdziwe pytanie brzmi, czy
PROJEKCJA — douczana przez całą sekwencję na snach klasy — nadal niesie
o niej informację. Trójca protokołu do domknięcia: nauczyć się ze snów
(I1) · podzielić się (I3) · zapomnieć na żądanie (N1).

## Setup

Fashion, konfiguracja K1 (sparse_k16 × GloVe 300d, epochs=15,
epochs_proj=15, l2sp=0, LR=0.001), pełna sekwencja 5 zadań, 5 seedów.
Operacje (`src/mars_cl_n.py`):
  unlearn_light : usunięcie wpisów klasy (statystyki, pod, prototyp,
                  seen) — projekcja NIETKNIĘTA;
  unlearn_scrub : light + douczenie projekcji na snach WSZYSTKICH
                  pozostałych klas (n_dream_scrub=2000/klasę,
                  epochs_proj) — próba wymazania śladu z wag;
  relearn_small : ponowna nauka klasy z n=100 obrazów przy ZAMROŻONEJ
                  projekcji (statystyki + pod; prototyp=kotwica słowna
                  istnieje a priori) — tempo powrotu mierzy informację
                  resztkową w projekcji.

## N1 — poziom 1 (funkcjonalny): macierz usunięć 10×

Po pełnym treningu, dla KAŻDEJ z 10 klas osobno (na kopii modelu):
unlearn light i scrub; pomiar per-class acc pozostałych 9 klas
(maska = aktualne seen).

**Kryteria (Z GÓRY):**
- Odnotowanie (nie werdykt): acc klasy usuniętej = 0 z konstrukcji
  (poza seen nie ma predykcji) — kontrast z LLM, ale trywialny.
- GŁÓWNE 1: śr. Δ acc pozostałych klas po unlearn (agregacja per seed
  po 10 usunięciach; pary vs przed): SZUM = NIETKNIĘTE (sukces
  poziomu 1); SYGNAL− = zmierzony koszt uboczny (dla scrub realny —
  douczanie ze snów może kosztować; ryzyko pre-rejestrowane).
  Dla light możliwy SYGNAL+ (mniej konkurencji w routingu) — też
  raportowany.

## N1 — poziom 2 (informacyjny): tempo powrotu vs klasa nigdy nie widziana

Klasa testowa wybrana Z GÓRY: c*=4 („coat", środek sekwencji, task 2).
Trzy systemy per seed:
  relearn_light : pełny trening → unlearn_light(4) → relearn(4, n=100)
  relearn_scrub : pełny trening → unlearn_scrub(4) → relearn(4, n=100)
  relearn_never : trening sekwencji BEZ taska 2 (klasy 4,5 nigdy nie
                  widziane) → relearn(4, n=100)  [kontrola zerowa]
Referencja górna: acc(4) w pełnym systemie przed usunięciem.

Mitygacja confoundu przestrzeni etykiet (pre-rejestrowana): kontrola
never nie zna klasy 5, więc acc(4) we WSZYSTKICH trzech systemach
liczone z maską ograniczoną do WSPÓLNEGO zbioru 9 klas {0,1,2,3,4,
6,7,8,9} (bez 5).

**Kryteria (Z GÓRY, pary per-seed na acc(4) po relearn):**
- GŁÓWNE 2: relearn_light vs relearn_never — SYGNAL+ = resztkowa
  informacja o klasie w projekcji ZMIERZONA w pp (oczekiwane:
  projekcja przez 3 zadania trenowała na snach klasy 4); SZUM = już
  light wymazuje do poziomu zerowego.
- GŁÓWNE 3: relearn_scrub vs relearn_never — SZUM = scrub wymazuje
  do poziomu klasy nigdy nie widzianej („zapominanie z gwarancją
  empiryczną"); SYGNAL+ = nawet scrub zostawia ślad (kwantyfikowany).
- Obserwacja: relearn_scrub vs relearn_light (ile ścierania daje scrub).

## Ryzyka pre-rejestrowane

1. Scrub douczaniem (nie od zera) może nie wymazać — informacja
   w wagach może przetrwać douczenie; SYGNAL+ w GŁÓWNYM 3 jest
   realny i jest wynikiem (granica gwarancji).
2. n=100 może być w reżimie, gdzie pod nasyca się niezależnie od
   projekcji (sufit z I2b: payload nasyca się 500–3000) — różnice
   mogą być małe; próg szumu rozstrzygnie.
3. Kontrola never widziała 4 zadania (inna długość sekwencji) —
   odnotowane jako ograniczenie konstrukcyjne.

## Kolejność uruchomień u Roberta

1. `python src/mars_cl_n.py` (smoke jednostkowy, CPU, sekundy)
2. `python src/run_N1_unlearning.py --smoke`, potem FULL (~15–20 min)

Wyniki do DROGA_N_NOTATKI.md; merge po komplecie i decyzji Roberta.
Po N: I4 (weryfikacja payloadu — „wykryj i zapomnij" dostaje tu
mechanizm naprawczy).

## N1b (dopisane 2026-07-20, PRZED runem) — naprawiony instrument poziomu 2

N1 poziom 2 unieważniony: relearn z negatywami 256/klasę (2304 vs 100
pozytywów) daje podłogę ~0% we wszystkich ścieżkach — instrument bez
zakresu dynamicznego. N1b zmienia JEDNO: budżet negatywów w relearn
ŁĄCZNIE ≈ liczbie pozytywów (neg_per_class = max(n_pozytywów // liczba
starych klas, 4) ≈ 11/klasę przy n=100). Ścieżki, klasa c*=4, n=100,
maska 9 klas, kryteria — bez zmian względem planu poziomu 2:
- GŁÓWNE 2: relearn_light vs relearn_never (SYGNAL+ = resztkowa
  informacja w projekcji, w pp),
- GŁÓWNE 3: relearn_scrub vs relearn_never (SZUM = empiryczna
  gwarancja wymazania),
- obserwacja: scrub vs light; referencja górna: pełny system.
Plik: `src/run_N1b_relearn_balanced.py` → `results/N1b_relearn_balanced.json`

## N1c (dopisane 2026-07-20, PRZED runem) — pełne wymazanie: reinit projekcji

N1b: scrub douczaniem zostawia 11.2pp odzyskiwalności. Kandydat pełnej
gwarancji: `unlearn_reinit` = light + REINICJALIZACJA projekcji
(deterministycznie z seeda) + nauka od zera na snach pozostałych 9 klas
(epochs_proj). Informacja o klasie nie może przetrwać, bo wagi są nowe;
pytanie brzmi, ILE TO KOSZTUJE pozostałe klasy (projekcja uczona na
snach zamiast na realnych danych sekwencji).

Plik: `src/run_N1c_reinit.py` → `results/N1c_reinit.json`

**Kryteria (Z GÓRY, pary per-seed):**
- GŁÓWNE 4: relearn po reinit vs never (never z N1b, TE SAME seedy
  i próbki): SZUM = PEŁNA empiryczna gwarancja wymazania.
- GŁÓWNE 5 (koszt): śr. acc pozostałych 9 klas po reinit vs pełny
  system (pary): SZUM = wymazanie darmowe (mocny wynik — sny
  wystarczają do odbudowy projekcji); SYGNAL− = zmierzona cena pełnej
  gwarancji.
- Obserwacja: relearn(4) po reinit vs scrub — domknięcie taksonomii
  light/scrub/reinit.
