# Droga J — notatki robocze

Plan: `DROGA_J_PLAN.md` (pre-rejestracja 2026-07-10, kod: commit d23daaf).
Runy: Robert, lokalnie (GTX 1050 Ti), 5 seedów, epochs=15, LR=0.001.

## J3 — sen spike-and-slab (ZAKOŃCZONE, 10.07.2026): formalnie SZUM,
## ale najspójniejszy mały efekt w historii projektu + nowy nominalny best

Pliki: `src/mars_cl_j.py` (FeatureStatsKSparse), `src/run_J3_sparse_dreams.py`;
wyniki: `results/J3_sparse_dreams.json` (conditioning=none). Czas: 340 s.

| Wariant (Fashion, class-IL) | Pamięć/kl. | ACC | min | F |
|---|---|---|---|---|
| diag_k16 (baza = H1b) | ~16 KB | 77.57 ± 1.02% | 76.59% | 18.8pp |
| sparse_k4 | ~6 KB | 76.78 ± 1.19% | 75.62% | 18.1pp |
| sparse_k8 | ~12 KB | 77.78 ± 1.16% | 76.80% | 17.2pp |
| **sparse_k16** | ~24 KB | **78.49 ± 0.91%** | **77.72%** | **16.0pp** |

| Wariant (MNIST, class-IL) | ACC | min | F |
|---|---|---|---|
| diag_k16 (baza) | 70.21 ± 2.80% | 66.64% | 32.0pp |
| sparse_k4 | 72.16 ± 2.10% | 69.12% | 29.0pp |
| sparse_k8 | 72.52 ± 2.01% | 70.10% | 28.4pp |
| sparse_k16 | 73.26 ± 2.16% | 70.45% | 27.6pp |

**WERDYKTY (pre-rejestrowane, sparse_k16 vs diag_k16):**
Fashion: +0.91pp (min +0.65) przy progu 1.93 → **SZUM**.
MNIST: +3.05pp (min +2.29) przy progu 4.96 → **SZUM**.
Kryterium uszanowane — NIE ogłaszamy SYGNAL+.

**Ustalenia:**
1. **Reprodukcja idealna:** diag_k16 odtworzył H1b co do 0.01pp na obu
   zbiorach (77.57 ± 1.02 / 70.21 ± 2.80) — determinizm ścieżki J
   potwierdzony; porównania są czyste.
2. **Efekt mały, ale skrajnie spójny (obserwacja, nie werdykt):**
   pary per-seed 10/10 dodatnie; delty Fashion +0.65…+1.19
   (std delt 0.24), MNIST +2.29…+3.81 (std delt 0.70); drabina
   monotoniczna sparse k4 < k8 < k16 na obu zbiorach; forgetting
   w dół (18.8→16.0 / 32.0→27.6pp); min per-seed w górę na obu.
   Konwencja progu (std bazy + std wariantu) jest z konstrukcji ślepa
   na sparowane efekty tej skali — to znana cecha (por. H1b vs replay).
   NIE zmieniamy kryterium post-hoc; do PRZYSZŁYCH pre-rejestracji
   odnotowujemy propozycję dodatkowego kryterium parowego (np. wszystkie
   seedy dodatnie ORAZ śr. delt > 2×std delt) — decyzja przed J5/kolejną
   serią, nie wstecz.
3. **Nowy nominalny best projektu (Fashion class-IL, 0 próbek):**
   78.49 ± 0.91 (min 77.72), ~24 KB/klasę. Vs replay-200 (76.97 ± 1.09):
   +1.52pp przy progu 2.00 → nadal RÓWNOWAŻNOŚĆ, nominalnie wyżej;
   najgorszy seed (77.72) NAD średnią replay. Luka do sufitu g1_all
   (80.45) zmniejszona z 2.88 do 1.96pp.
4. **Równopamięciowo struktura ≥ rozdzielczość:** sparse_k8 (12 KB)
   ≥ diag_k16 (16 KB) na obu zbiorach (+0.21 / +2.31pp) — rzadkość
   (prawdziwe zera) wnosi to, czego nie kupują dodatkowe centroidy.
   Spójne z negatywem H1b o pełnej kowariancji: model gęstości ma
   szanować geometrię cech po ReLU (lokalność + rzadkość), nie dokładać
   struktury globalnej.
5. **Warunek brzegowy bez zmian:** MNIST dalej wyraźnie pod replay
   (73.3 vs 88.8) — słowa cyfr nie niosą semantyki wizualnej.

Status: czekamy na pełny J1 (od niego zależy ewentualny run J3
`--conditioning cond`; smoke J1 sugeruje raczej brak).

## J1 — kondycjonowanie cech (ZAKOŃCZONE, 10.07.2026): SZUM ×2 —
## hipoteza audytu SFALSYFIKOWANA; seria F/G/H nie wymaga korekty

Pliki: `src/mars_cl_j.py` (calibrate_bn, sigma-norm),
`src/run_J1_feature_conditioning.py`; wyniki:
`results/J1_feature_conditioning.json`. Czas: 326 s.

| Wariant (Fashion, class-IL) | ACC | min | F |
|---|---|---|---|
| **k16_raw (baza)** | **77.57 ± 1.02%** | 76.59% | 18.8pp |
| k16_bncal | 76.87 ± 0.42% | 76.36% | 19.4pp |
| k16_signorm | 76.31 ± 1.57% | 74.52% | 20.7pp |
| k16_cond | 76.18 ± 0.66% | 75.39% | 20.4pp |

| Wariant (MNIST, class-IL) | ACC | min | F |
|---|---|---|---|
| k16_raw (baza) | 70.21 ± 2.80% | 66.64% | 32.0pp |
| k16_bncal | 70.34 ± 2.58% | 68.12% | 31.1pp |
| k16_signorm | 70.87 ± 4.56% | 65.37% | 31.0pp |
| k16_cond | 70.97 ± 3.67% | 67.09% | 30.4pp |

**WERDYKTY (pre-rejestrowane):** Fashion: SZUM (najlepszy kondycjonowany
= bncal, −0.71pp przy progu 1.44). MNIST: SZUM (+0.76pp przy progu 6.47).
Sanity: k16_raw = H1b k16 co do 0.00pp na obu zbiorach.

**Ustalenia:**
1. **Falsyfikacja hipotezy audytu (wynik negatywny = wynik):** martwy
   BatchNorm w losowym zamrożonym backbone był REALNYM przeoczeniem
   inżynierskim, ale NIE tłumaczy poziomu wyników — kalibracja nic nie
   daje (MNIST) lub lekko szkodzi (Fashion). Seria F/G/H nie wymaga
   korekty; opublikowane liczby stoją na najlepszej znanej konfiguracji.
   Do papieru: jedno zdanie w appendixie "checked and cleared".
2. **sigma-norm konsekwentnie szkodzi na Fashion:** pary per-seed 5/5
   ujemne (−0.65…−2.48, śr. −1.26); k16_cond również 5/5 ujemne.
   Hipoteza mechanizmu (do ewentualnego sprawdzenia, nie twierdzenie):
   w losowych cechach ReLU wariancja wymiaru koreluje z jego
   informatywnością — wyrównanie skal AWANSUJE wymiary szumowe
   w k-means snu i podach; projekcja i tak uczy się skal dla routingu.
3. **Obserwacja stabilizacyjna:** bncal zmniejsza rozrzut seedów na
   Fashion (std 1.02 → 0.42) bez podniesienia średniej — kalibracja
   ujednolica, nie ulepsza.
4. **Decyzja sekwencyjna (wg planu):** J1 ≠ SYGNAL+ → run J3
   `--conditioning cond` ODWOŁANY. J2 biegnie wg planu (pełna siatka —
   CIFAR to inny reżim: tam wejście było nieznormalizowane, więc
   kondycjonowanie może działać inaczej; rozstrzygną pary per-seed).

Status: pozostał pełny J2 (CIFAR) i ewentualnie J4 (GloVe 300d).
