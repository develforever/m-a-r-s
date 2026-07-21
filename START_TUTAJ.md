# START TUTAJ — stan projektu i wejście do nowej sesji

Aktualizacja: 2026-07-20 (późn.). Seria N (selektywne zapominanie, v0.9) ZAKOŃCZONA: taksonomia light/scrub/reinit zmierzona (0%/~84%/100% wymazania; reinit = pełna gwarancja przy koszcie ≤0; klasa nieznana projekcji routingowo nieosiągalna). Trójca protokołu domknięta: sen uczy · dzieli · zapomina. Kandydaci: O1 (konsolidacja snem — możliwy nowy best), I4. Wcześniej — seria M (długi horyzont, v0.8): CIFAR-100 × 20 zadań na pretrained — 40.70 ± 0.84 (85.8% sufitu; deficyt późny −7.8pp vs sufit, strukturalny — front stability–plasticity zmierzony w 3 punktach, rehearsal per klasę zostaje domyślny); kotwice: SYGNAL+ 300d vs 50d (+7.7) przy 100 klasach. Następne: seria N (unlearning), I4. Wcześniej — mapa K→I→L (v0.7): seria L — fork tożsamości: zamrożony resnet18-ImageNet pod niezmienionym mechanizmem daje CIFAR 74.69 ± 0.69 (+37.2pp, SYGNAL+; 96.7% sufitu 77.23; NAD trenowalnym joint 70.24), kolektyw na mocnych cechach 74.13 (koszt protokołu −0.56pp, parowy−); I2b — fuzja działa tylko poniżej nasycenia payloadu (parowy+ przy n=100). Wcześniej: K (v0.5) i I (v0.6): kolektywna wymiana snów, v0.6) ZAKOŃCZONE: mechanizm realizuje 94.6–97.6% sufitów zamrożonych cech (K0/K1, best Fashion 79.23 ± 0.73), a kolektyw 5 agentów wymieniających wyłącznie statystyki snu (24 KB/klasę, zero obrazów) jest RÓWNOWAŻNY agentowi sekwencyjnemu (78.87 vs 79.23 — I3). W przygotowaniu: Droga L (jawny fork tożsamości — pretrained backbone, DROGA_L_PLAN.md) i I2b (fuzja low-data). Mapa etapów: PLAN_GENERALNY.md. REPO PUBLICZNE: https://github.com/develforever/m-a-r-s (tagi: v0.3-freeze … v0.6).
DECYZJE 2026-07-10: (1) treść rozcięta na DWA papery (A: CL; B: routing
ceiling); (2) submisje (arXiv/TMLR/konferencje) WSTRZYMANE do osobnej
decyzji Roberta — etap bieżący to WYŁĄCZNIE repo publiczne.
Related work zweryfikowany → `RELATED_WORK.md`. Plan → `private/PLAN_PUBLIKACJI.md`.

## Prompt otwierający nową sesję (skopiuj i wklej)

> Kontynuujemy projekt M.A.R.S. Stan: v0.6 po seriach K i I
> (DROGA_K_NOTATKI.md, DROGA_I_NOTATKI.md; mapa etapów
> PLAN_GENERALNY.md; L i I2b ZAKOŃCZONE — DROGA_L_NOTATKI.md; kandydaci:
> seria M długi horyzont, I4 weryfikacja payloadu); treść
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
sparse — J2b; skala efektu rośnie z trudnością danych). Seria K domknęła mechanizm liczbowo (79.23 = 97.6% sufitu Fashion; 37.5 = 94.6% sufitu losowych cech CIFAR — reszta luki jest reprezentacyjna), a seria I dowiozła kolektyw: 5 agentów × 2 klasy, 8 klas ze snów po 24 KB, wynik równoważny agentowi sekwencyjnemu i nominalnie nad replay-200.
Granice zmierzone: wymaga semantycznych nazw klas (MNIST poniżej), sufit
losowych cech (g1_all 80.45 Fashion / joint 70.2 CIFAR), G2 kompozycyjność
negatywna z regułą strukturalną 3/3. Wisienka metodologiczna z Części II:
oracle inflation (realna przestrzeń routingu ~0.5pp, nie 6pp).

## Mapa plików

- `WHITEPAPER.md` — v0.3, pełny draft (Part I PoC, Part II routing ceiling,
  Part III memory without data). Przed submisją: related work + cytowania.
- `DROGA_D/E/F/G/H/J/K/I_NOTATKI.md` — pełne tabele wyników i werdykty.
- `DROGA_F/G/H/J/K/I/L_PLAN.md`, `D6B/D7_PLAN.md` — pre-rejestrowane plany;
  `PLAN_GENERALNY.md` — mapa etapów K→I→L (zasada warstw).
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
5. ZREALIZOWANE 2026-07-17: piętro reprezentacji = Droga L (jawny fork
   tożsamości, pre-rejestrowana, runy w toku); G2b (ECOC) dalej otwarte.

## Zasady współpracy (obowiązują każdego agenta)

- Eksperymenty uruchamia WYŁĄCZNIE Robert, lokalnie (GTX 1050 Ti).
- Wyniki do `results/` pisze jeden tor pracy; inni agenci czytają.
- 5 seedów, progi szumu, kryteria werdyktu PRZED uruchomieniem,
  min per-seed raportowany, negatyw = wynik.
- Kod zamrożony do publikacji — zmiany tylko za zgodą Roberta.
