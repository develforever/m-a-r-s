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
