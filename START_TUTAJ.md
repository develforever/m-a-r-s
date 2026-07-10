# START TUTAJ — stan projektu i wejście do nowej sesji

Aktualizacja: 2026-07-10 (wieczór). Seria J ZAKOŃCZONA na branchu droga-j: audyt zamknięty (usterki niewinne, J1/J2), sen spike-and-slab = SYGNAL+ na CIFAR (J2b, 37.51 ± 1.35). REPO PUBLICZNE: https://github.com/develforever/m-a-r-s (tag v0.3-freeze; po merge droga-j: v0.4). J4 (GloVe 300d) opcjonalny, nieuruchomiony.
DECYZJE 2026-07-10: (1) treść rozcięta na DWA papery (A: CL; B: routing
ceiling); (2) submisje (arXiv/TMLR/konferencje) WSTRZYMANE do osobnej
decyzji Roberta — etap bieżący to WYŁĄCZNIE repo publiczne.
Related work zweryfikowany → `RELATED_WORK.md`. Plan → `private/PLAN_PUBLIKACJI.md`.

## Prompt otwierający nową sesję (skopiuj i wklej)

> Kontynuujemy projekt M.A.R.S. Stan: v0.4 po serii J (audyt + sen
> sparse — DROGA_J_NOTATKI.md); treść
> rozcięta na dwa papery (A: CL, B: routing ceiling); submisje
> WSTRZYMANE — etap bieżący to wyłącznie repo publiczne (checklist
> w private/PLAN_PUBLIKACJI.md pkt 3). Przeczytaj START_TUTAJ.md,
> private/PLAN_PUBLIKACJI.md, RELATED_WORK.md i WHITEPAPER.md (w razie potrzeby
> DROGA_F/G/H_NOTATKI.md). Nie proponuj submisji, dopóki sam nie wrócę
> do tematu. Zasady: eksperymenty uruchamiam tylko ja lokalnie; 5 seedów;
> kryteria werdyktów z góry; wynik negatywny = wynik; cytowania tylko
> zweryfikowane web searchem.

## Stan w trzech zdaniach

Główny wynik: MARS-CL (losowy zamrożony backbone + prototypy-słowa GloVe +
parametryczny sen; od serii J spike-and-slab szanujący rzadkość po ReLU)
osiąga **77.6 ± 1.0% class-IL na Split-Fashion (sparse: 78.5 ± 0.9
nominalnie, w szumie) — równoważność z replay-200, zero przechowanych
próbek, stały MAC**; na Split-CIFAR przewaga rośnie: **37.5 ± 1.4 vs
replay 14.0 ± 4.9** (wejście znormalizowane; SYGNAL+ +4.5pp za sam sen
sparse — J2b; skala efektu rośnie z trudnością danych).
Granice zmierzone: wymaga semantycznych nazw klas (MNIST poniżej), sufit
losowych cech (g1_all 80.45 Fashion / joint 70.2 CIFAR), G2 kompozycyjność
negatywna z regułą strukturalną 3/3. Wisienka metodologiczna z Części II:
oracle inflation (realna przestrzeń routingu ~0.5pp, nie 6pp).

## Mapa plików

- `WHITEPAPER.md` — v0.3, pełny draft (Part I PoC, Part II routing ceiling,
  Part III memory without data). Przed submisją: related work + cytowania.
- `DROGA_D/E/F/G/H/J_NOTATKI.md` — pełne tabele wyników i werdykty.
- `DROGA_F/G/H/J_PLAN.md`, `D6B/D7_PLAN.md` — pre-rejestrowane plany.
- `ARSENAL_PRZEOCZONYCH_NARZEDZI.md` — inwentarz pomysłów z ocenami.
- `SLOWNIK_POJEC.md` — słownik dla czytelnika.
- `src/` — 20+ runnerów, każdy pisze JSON do `results/`; smoke → full.
- `results/*.json` — wszystkie wyniki (per-seed).

## Następne kroki (kolejność uzgodniona)

1. Related work: zweryfikować i wpleść cytowania — OWM (Zeng et al. 2019),
   DAP (Lampert), iCaRL, GDumb (krytyka replay), Expert Gate, ECOC,
   generative/feature replay, NCM/prototypy w CL. KAŻDE źródło sprawdzone
   web searchem, zero cytowań z pamięci.
2. Decyzja: jeden paper czy dwa (metodologiczny: oracle inflation +
   routing ceiling → TMLR/workshop; główny CL → CoLLAs).
3. arXiv: endorsement (maile do autorów cytowanych prac), konto,
   kategoria cs.LG.
4. Repo publiczne: README naukowe, licencja, seedy reprodukcji.
5. PO publikacji (nie wcześniej): decyzja o piętrze reprezentacji
   (mocniejszy zamrożony backbone vs from-scratch H2/H3), G2b (ECOC).

## Zasady współpracy (obowiązują każdego agenta)

- Eksperymenty uruchamia WYŁĄCZNIE Robert, lokalnie (GTX 1050 Ti).
- Wyniki do `results/` pisze jeden tor pracy; inni agenci czytają.
- 5 seedów, progi szumu, kryteria werdyktu PRZED uruchomieniem,
  min per-seed raportowany, negatyw = wynik.
- Kod zamrożony do publikacji — zmiany tylko za zgodą Roberta.
