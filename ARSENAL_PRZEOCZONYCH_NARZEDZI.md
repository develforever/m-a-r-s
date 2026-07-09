# Arsenał przeoczonych narzędzi — inwentarz pod "rewolucję"

Data: 2026-07-06
Pytanie: jakie mechanizmy sprzętowe i matematyczne obecne systemy pomijają,
ignorują lub źle używają — i które z nich pasują do naszej tezy (Droga F: CL)?
Ocena uczciwa: przy każdym narzędziu werdykt REALNE / NISZOWE / NIE DLA NAS.

---

## 1. GPU fixed-function (poza ALU) — nasz stary teren i co jeszcze tam leży

**TMU / tekstury (UŻYWAMY, Faza 1).** Fetch + bilinear za darmo, 41.9M próbek/s.
Niewykorzystane rozszerzenie: **mipmapy = darmowa piramida wielorozdzielcza**.
Łańcuch mip label-mapy to sprzętowa hierarchia coarse→fine — czyli routing
hierarchiczny z E2 w fixed-function: poziom N = grupa, poziom 0 = klasa.
Werdykt: REALNE, elegancko skleja E2 z Fazą 1 (wątek deployment, nie CL-core).

**Z-buffer = sprzętowy argmin.** Depth test odrzuca fragmenty dalsze niż
najbliższy — to jest argmin w fixed-function. Renderując prototypy jako
fragmenty z depth = dystans, z-buffer ZOSTAWIA najbliższy prototyp bez żadnego
ALU. Routing prototypowy (nasz rdzeń!) ma sprzętową implementację, której
nikt tak nie używa. Werdykt: REALNE i oryginalne; wątek deployment/WebGPU.

**Rasterizer = darmowa interpolacja barycentryczna.** Dowód, że ta klasa
pomysłów wygrywa: 3D Gaussian Splatting zrewolucjonizował neural rendering
właśnie rasterizerem. U nas: funkcje kawałkowo-liniowe nad 2D embeddingiem
za darmo. Werdykt: NISZOWE (nasz embedding ma 32D, nie 2D — wymagałoby
powrotu do UV z v1).

**Blending (ROP) = ważona akumulacja za darmo** (ensemble/top-2 bez ALU).
Werdykt: NISZOWE — D4 pokazał, że ensemble nic nie wnosi informacyjnie.

**NVENC/NVDEC (kodeki wideo) = darmowa estymacja ruchu.** Motion vectors
z enkodera H.264 to gotowy optical flow w fixed-function — dla danych wideo
cechy czasowe za darmo. Werdykt: REALNE dla przyszłego wątku wideo-CL;
poza obecnym zakresem.

**RT cores (BVH traversal = sprzętowe zapytania najbliższego sąsiada)** —
używane w literaturze do kNN. Werdykt: NIE DLA NAS (1050 Ti ich nie ma).

## 2. CPU — sprzęt, który E4 właśnie wskazał palcem

E4 smoke udowodnił: GPU jest ZŁYM sprzętem dla małych, rzadkich, modularnych
sieci (3% peak przy slim CNN). Wniosek odwrotny: **właściwym sprzętem dla
MAŁYCH PODÓW może być CPU** — i to jest dokładnie teren, który pole ignoruje,
bo wszyscy benchmarkują na GPU.

**XNOR + popcount = iloczyn skalarny binarny w 1-2 cyklach.** AVX2 robi 256
binarnych MAC na cykl na rdzeń. Nasze pody ternary (B8: za darmo co do acc,
30% sparsity) + binaryzacja wejść → pod liczony popcountem. Pod (128×24+24×10)
mieści się W CAŁOŚCI w L1/L2 cache. Werdykt: **REALNE i spójne z F** —
"continual learning na CPU edge, bez GPU w ogóle" to mocna teza deployment.

**LUT / product quantization (styl FAISS): mnożenie zastąpione odczytem
tabeli.** Routing prototypowy przez PQ = zero mnożeń w inferencji routingu.
Werdykt: REALNE, tanie do zmierzenia.

**SLIDE (Rice): trening przez LSH na CPU bijący GPU** — dowód literaturowy,
że hash zamiast matmul działa. Patrz sekcja 3. Werdykt: inspiracja wprost.

**Sparsity niestrukturalna** — GPU jej nie umie, CPU tak (skip zer). Ternary
pody mają ~30% zer za darmo. Werdykt: REALNE, składnik wątku CPU.

## 3. Matematyka pomijana — najciekawsze dla Drogi F

**Hashing / LSH jako router.** Locality-Sensitive Hashing daje przybliżonego
najbliższego sąsiada w O(1) BEZ UCZENIA. Router, który nie ma wag, nie może
zapomnieć — w CL to nie optymalizacja, to WŁASNOŚĆ. Werdykt: **REALNE,
kandydat na F2** (router LSH vs prototypowy: retencja i koszt).

**Bloom filter / szkice (sketches) = detekcja nowości za grosze.** E1: luka
jest niskopewna — router "wie, że nie wie". W CL to samo pytanie brzmi:
"czy ta próbka jest z nowego zadania?". Bloom/count-min nad skwantyzowanymi
embeddingami daje task-boundary detection niemal za darmo → **task-free CL**
(system SAM wie, kiedy powołać nowy pod — bez etykiet zadań, których
większość metod CL po cichu wymaga). Werdykt: **REALNE, najmocniejszy
pojedynczy pomysł tego inwentarza; kandydat na F3 i wyróżnik publikacji.**

**Reservoir computing / random features (ELM).** Zamrożony LOSOWY backbone +
uczone tylko głowice. Jeśli losowe cechy konwolucyjne wystarczą (na skali
MNIST/Fashion często prawie tak), znika ryzyko nr 1 Drogi F (transfer cech
z zadania 1) — bo backbone nigdy niczego nie uczył, więc niczego nie
faworyzuje. Werdykt: **REALNE, tani wariant F1d — dopisać do planu F.**

**Hyperdimensional computing (HDC).** Klasy jako binarne hiperwektory
(10k bitów); bind = XOR, bundle = majority; podobieństwo = Hamming (popcount).
One-shot dodawanie klasy = jedno bundle. Naturalnie continual, naturalnie
CPU. Werdykt: REALNE jako alternatywny mechanizm prototypów (F-later);
ryzyko: sufit accuracy niżej niż nasz router.

**Kompresja jako podobieństwo** (gzip-kNN itp.). Werdykt: NIE DLA NAS
(obrazy, nie tekst; wolne).

**Stochastic computing, analog in-memory** — Werdykt: NIE DLA NAS (sprzęt).

## 4. Czego NIE przeoczyliśmy (kontrola uczciwości)

Sprawdziliśmy i zamknęliśmy z twardymi liczbami: uczenie lokalne (FF/CHL —
działa, droższe), tekstury naiwne (obalone), SOM+TMU (warunkowo działa),
adaptive compute na shared backbone (strukturalnie słabe), distillation,
ensemble, predictive coding (sufit reprezentacji), ternary routera
(katastrofa −18pp; routera nie wolno kwantyzować).

## 5. Synteza — co wchodzi do planu

Priorytet dla Drogi F (rdzeń naukowy):
1. **F1d: random frozen backbone (reservoir)** — tani, rozbraja ryzyko nr 1.
2. **F2: router bez wag (LSH) vs prototypowy** — "nie ma wag = nie ma
   zapominania" jako teza mierzalna.
3. **F3: task-free CL przez detekcję nowości (Bloom/sketch na embeddingach)**
   — potencjalny wyróżnik całej serii.

Wątek deployment (drugi filar, po wynikach F):
4. **CPU-SIMD ternary pody** (popcount, L1-resident) — właściwy sprzęt dla
   modularności; zastępuje pogrzebaną tezę GPU-energy.
5. **Z-buffer argmin + mipmap hierarchia** — sprzętowy routing (WebGPU),
   unikat inżynierski do demo.

Zasada bez zmian: każdy pomysł wchodzi przez pre-rejestrowany eksperyment
z kryterium werdyktu; wynik negatywny = wynik.
