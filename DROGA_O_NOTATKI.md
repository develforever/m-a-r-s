# Droga O — notatki robocze

Plan: `DROGA_O_PLAN.md` (pre-rejestracja 2026-07-20). Runy: Robert,
lokalnie, 5 seedów. Sanity determinizmu: bazy odtworzone co do
0.000pp na OBU benchmarkach (piąta czysta reprodukcja ścieżki).

## O1 — konsolidacja snem (ZAKOŃCZONE, 20.07.2026): HIPOTEZA
## SFALSYFIKOWANA — czwarta funkcja snu NIE istnieje; sen chroni,
## przenosi i odbudowuje, ale NIE ulepsza ponad realne dane

Plik: `src/run_O1_consolidation.py`;
wyniki: `results/O1_consolidation.json`. Czas: 106 s.

| Benchmark | baza | o1_reinit | o1_finetune |
|---|---|---|---|
| Fashion (K1, sufit 81.16) | 79.23 ± 0.73 | 78.61 (**parowy−** −0.62, 5/5) | 79.01 (SZUM −0.21) |
| CIFAR-n (L1, sufit 77.23) | 74.69 ± 0.69 | 73.89 (**parowy−** −0.80, 5/5) | 74.03 (SZUM −0.66, 5/5 ujemnych pod progiem) |

**WERDYKTY (pre-rejestrowane):** reinit = SYGNAL-parowy− na obu;
finetune = SZUM na obu. Konsolidacja snem po sekwencji nie pomaga —
odbudowa od zera systematycznie lekko szkodzi, douczanie nic nie daje.

**Ustalenia:**
1. **Ryzyko #1 zmaterializowane, mechanizm jasny:** sen to model
   gęstości, nie dane. Projekcja sekwencyjna widziała REALNE cechy
   każdej klasy w jej zadaniu; „joint na snach" zastępuje je
   przybliżeniem — i przegrywa mimo braku dryfu kolejności. Realne
   dane w momencie nauki > brak dryfu.
2. **Kierunkowa obserwacja z N1c (+0.7pp) sfalsyfikowana na czystym
   pomiarze** — wzorcowy cykl dyscypliny: obserwacja uboczna →
   pre-rejestrowany test → falsyfikacja. Odbudowa ze snów (N1c)
   pozostaje ważna jako NARZĘDZIE ZAPOMINANIA (koszt ~0), nie jako
   dźwignia wyniku.
3. **finetune=SZUM potwierdza nasycenie:** projekcja sekwencyjna po
   K-serii jest wyżyłowana (97.6% sufitu) — dodatkowy sen niczego
   nie wnosi ani nie psuje. Spójne z całą serią K.
4. Funkcje snu po serii O — trzy zmierzone, czwarta odrzucona:
   OCHRONA (rehearsal, F3b→J3) · TRANSFER (I1/I3) · ODBUDOWA
   (N1c, koszt ~0) · ~~konsolidacja~~ (O1: falsyfikacja).

## STATUS KOŃCOWY SERII O: KOMPLET (jeden run, wynik negatywny = wynik)

Dalej: merge `droga-o` → main (v0.10, decyzja Roberta). Ostatni
kandydat z mapy: I4 (kolektyw niezaufany — weryfikacja payloadu przez
kotwicę + reinit jako naprawa). Po I4: przegląd całości pod v1.0.
