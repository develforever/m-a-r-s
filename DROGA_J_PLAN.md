# Droga J — plan pre-rejestrowany: naprawy i darmowe dźwignie wyniku

Data pre-rejestracji: 2026-07-10 (branch `droga-j`; main/v0.3 nietknięty).
Źródło: audyt kodu serii F/G/H z 2026-07-10 (sesja Cowork). Litera J,
bo I jest zarezerwowane dla kolektywnej wymiany snów (ARSENAL, dopisek).

Motywacja — trzy znaleziska audytu, każde tanie do zmierzenia:

1. **Martwy BatchNorm w losowym zamrożonym backbone.** Przy
   `backbone_source="random"` `init_representation` nigdy nie robi
   forwardu w trybie train, więc running stats BN zostają na (0,1) —
   BN jest identycznością we WSZYSTKICH runach F1d/G1/F3/F3b/H1/H1b/F4.
   Cechy losowe mają nieskalibrowane skale per wymiar, a NCM (Euklides)
   i k-means snu są na to czułe.
2. **CIFAR bez normalizacji wejścia.** `load_cifar10` robi tylko /255;
   Fashion/MNIST mają pełną Normalize (run_D1:42–50). Monolity nadrabiają
   trenowalnym BN — zamrożony losowy backbone MARS-a nie.
3. **Sen ignoruje rzadkość cech po ReLU.** Sampler robi tylko
   `clamp_min(0)`; realne cechy mają dokładne zera z dużym
   prawdopodobieństwem, sen generuje tam małe dodatnie wartości
   (gęstość poza rozmaitością danych). H1 wskazał wierność snu jako
   JEDYNE pozostałe wąskie gardło (77.57 → sufit 80.45).

Plus niedokończony punkt: F4 (CIFAR) nigdy nie dostał zwycięskiego snu
k16 z H1b (kod zamrożono po H1b; F4 biegał na k4).

## Zasady (bez zmian)

- Kod v0.3 (main) NIETYKALNY: wszystko w NOWYCH plikach
  (`src/mars_cl_j.py`, `src/run_J*.py`), na branchu `droga-j`.
- 5 seedów (0–4), pary per-seed, próg szumu = std(baza) + std(wariant).
- Werdykty: SYGNAL+ wymaga śr. delta > próg ORAZ min per-seed > 0
  (lekcja B8/E4); SYGNAL− symetrycznie; inaczej SZUM.
- Wynik negatywny = wynik. Runy wyłącznie u Roberta, lokalnie.
- Uczciwość CL: kalibracja BN i sigma-norm używają WYŁĄCZNIE
  nieetykietowanych obrazów zadania 0, JEDEN raz, PRZED jakąkolwiek
  nauką — potem reprezentacja zamrożona na zawsze (precedens: ae0/F2
  używało obrazów task0 bez etykiet). Nietykalność zachowana.

## J1 — kondycjonowanie cech losowego backbone (Fashion + MNIST)

Plik: `src/run_J1_feature_conditioning.py` → `results/J1_feature_conditioning.json`

Baza wspólna: k16 diag, epochs_proj=15, l2sp=0 (dokładnie H1b/k16).

| Wariant | bn_calib | sigma-norm |
|---|---|---|
| k16_raw (baza, reprodukcja H1b) | – | – |
| k16_bncal | task0, momentum=None (dokładne statystyki) | – |
| k16_signorm | – | podział cech przez per-wymiarowe std z task0 |
| k16_cond | tak | tak |

sigma-norm celowo BEZ centrowania średniej: zachowuje nieujemność
i dokładne zera cech (spójność z clamp_min(0) snu i z J3).

**Kryteria (Z GÓRY, class-IL Fashion = główne; MNIST = obserwacja):**
- SYGNAL+ : najlepszy wariant kondycjonowany vs k16_raw (pary per-seed):
  śr. d > próg szumu ORAZ min per-seed > 0.
- SZUM / SYGNAL− symetrycznie.
- Sanity: k16_raw musi odtworzyć H1b k16 (77.57 ± 1.02) w granicach szumu;
  jeśli nie — STOP, szukać niedeterminizmu.
- Kontekst raportowany: replay-200 (76.97), sufit g1_all (80.45).
- Ryzyko pre-rejestrowane: sigma-norm zmienia geometrię NCM/k-means —
  może pogorszyć; werdykt symetryczny.

## J2 — Split-CIFAR-10 z poprawnym wejściem + sen k16

Plik: `src/run_J2_cifar_normalized.py` → `results/J2_cifar_normalized.json`

Zmiany vs F4: (a) normalizacja per kanał (mean/std CIFAR-10) dla
WSZYSTKICH systemów — monolity też (uczciwie); (b) siatka MARS 2×2:
stats_k ∈ {4,16} × kondycjonowanie ∈ {raw, cond=bn_calib+sigma-norm};
epochs_proj=15, l2sp=0 dla wszystkich (konwencja H1b). Baseline'y:
finetune / replay-200 / joint, jak F4, te same seedy.

**Kryteria (Z GÓRY, class-IL):**
- Główne: mars_k16_cond vs replay-200 (pary per-seed, w tym samym runie):
  SYGNAL+ jeśli śr. d > próg ORAZ min > 0 (oczekiwane — F4 dało +13pp).
- Pytanie naprawcze (sedno J2): najlepszy nowy MARS vs stary F4
  mars_combo 32.04 ± 1.01 (referencja z results/F4_split_cifar.json,
  niesparowane — inne przygotowanie danych): czy śr. > 32.04 + 1.01?
  Jeśli tak → niski wynik CIFAR był częściowo artefaktem przygotowania
  danych, nie tylko granicą losowych cech.
- Dekompozycja raportowana: k4_raw→k4_cond (efekt kondycjonowania),
  k4_cond→k16_cond (efekt wierności snu), przesunięcie sufitu joint.

## J3 — sen spike-and-slab (wierność snu, atak na 80.45)

Plik: `src/run_J3_sparse_dreams.py` → `results/J3_sparse_dreams.json`

Mechanizm (`FeatureStatsKSparse`): per klasa k centroidów (k-means jak
F3b), ale per wymiar przechowujemy P(cecha>0) oraz średnią i wariancję
WARUNKOWE (tylko z wartości dodatnich). Sen: maska Bernoulliego ⊙
(mu + sigma·z, clamp≥0) — zera są PRAWDZIWYMI zerami, nie ogonem
Gaussiana. Pamięć: 3 tablice k×D zamiast 2 (1.5× diag przy tym samym k).

| Wariant | Pamięć/klasę (D=128) |
|---|---|
| diag_k16 (baza, = H1b k16) | ~16 KB |
| sparse_k4 | ~6 KB |
| sparse_k8 | ~12 KB (mniej niż baza!) |
| sparse_k16 | ~24 KB |

Flaga `--conditioning {none,cond}` (domyślnie none) stosuje się do
WSZYSTKICH wariantów naraz (porównania zostają sparowane w obrębie
reżimu). Sekwencjonowanie: najpierw run none; jeśli J1 da SYGNAL+,
dodatkowy run cond.

**Kryteria (Z GÓRY, class-IL Fashion = główne):**
- SYGNAL+ : sparse_k16 vs diag_k16 (pary per-seed): śr. d > próg
  ORAZ min > 0. Cel kierunkowy (nie kryterium): zbliżenie do 80.45.
- Obserwacja równopamięciowa: sparse_k8 (~12 KB) vs diag_k16 (~16 KB) —
  czy struktura rzadkości bije surowe k przy mniejszej pamięci?
- Ryzyko: przy małych klastrach estymata P/momentów jest szumna
  (fallback: wymiary bez dodatnich → P=0, wariancja 1e-3).

## J4 (OPCJONALNY) — GloVe 300d jako bogatsza geometria słów

Plik: `src/run_J4_glove300.py` → `results/J4_glove300.json`
Wymaga: `python scripts/download_glove_300d.py` (ponowne ~822 MB —
zip 6B jest kasowany po ekstrakcji 50d; stąd osobny skrypt).

Warianty (Fashion): all_50 / all_300 (diagnostyczny sufit g1_all,
proj_train="all") oraz seq: k16_50 / k16_300 (uczciwy CL, F3b+H1b).

**Kryteria (Z GÓRY):**
- Główne (uczciwy CL): k16_300 vs k16_50 — SYGNAL+/SZUM/SYGNAL− po
  progu szumu i min per-seed.
- Diagnostyka: all_300 vs all_50 — czy 300d podnosi sam sufit.
- Ryzyko pre-rejestrowane: projekcja 128→300 ma 6× więcej parametrów
  (38.4k vs 6.4k) = większa powierzchnia dryfu; sen może jej nie
  utrzymać → SYGNAL− jest realny i też jest wynikiem.

## Kolejność uruchomień u Roberta

1. `python src/run_J1_feature_conditioning.py --smoke`, potem FULL.
2. `python src/run_J2_cifar_normalized.py --smoke`, potem FULL
   (najdłuższy — rząd 1.5× F4).
3. `python src/run_J3_sparse_dreams.py --smoke`, potem FULL
   (`--conditioning cond` tylko jeśli J1 = SYGNAL+).
4. (opcja) `python scripts/download_glove_300d.py`, potem
   `python src/run_J4_glove300.py`.

Wyniki dopisujemy do DROGA_J_NOTATKI.md (powstanie po pierwszym runie);
merge do main dopiero po komplecie werdyktów i decyzji Roberta.
