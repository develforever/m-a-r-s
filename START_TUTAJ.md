# START TUTAJ — stan projektu i wejście do nowej sesji

Aktualizacja: 2026-07-10. KOD ZAMROŻONY po H1b. Etap: REPO PUBLICZNE.
DECYZJE 2026-07-10: (1) treść rozcięta na DWA papery (A: CL; B: routing
ceiling); (2) submisje (arXiv/TMLR/konferencje) WSTRZYMANE do osobnej
decyzji Roberta — etap bieżący to WYŁĄCZNIE repo publiczne.
Related work zweryfikowany → `RELATED_WORK.md`. Plan → `private/PLAN_PUBLIKACJI.md`.

## Prompt otwierający nową sesję (skopiuj i wklej)

> Kontynuujemy projekt M.A.R.S. Stan: kod zamrożony po H1b; treść
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
parametryczny sen k=16 centroidów) osiąga **77.6 ± 1.0% class-IL na
Split-Fashion — nominalnie NAD replay-200 (77.0), formalnie równoważność —
przy zerze przechowanych próbek, stałym MAC i forgettingu 18.8 vs 27pp**;
na Split-CIFAR przewaga odporności rośnie (32.0 ± 1.0 vs 18.9 ± 8.8).
Granice zmierzone: wymaga semantycznych nazw klas (MNIST poniżej), sufit
losowych cech (g1_all 80.45 Fashion / joint 68.7 CIFAR), G2 kompozycyjność
negatywna z regułą strukturalną 3/3. Wisienka metodologiczna z Części II:
oracle inflation (realna przestrzeń routingu ~0.5pp, nie 6pp).

## Mapa plików

- `WHITEPAPER.md` — v0.3, pełny draft (Part I PoC, Part II routing ceiling,
  Part III memory without data). Przed submisją: related work + cytowania.
- `DROGA_D/E/F/G/H_NOTATKI.md` — pełne tabele wyników i werdykty.
- `DROGA_F/G/H_PLAN.md`, `D6B/D7_PLAN.md` — pre-rejestrowane plany.
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
