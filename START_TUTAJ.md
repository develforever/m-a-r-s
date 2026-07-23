# START TUTAJ — stan projektu i wejście do nowej sesji

Aktualizacja: 2026-07-23 (noc). SERIA P KOMPLET (branch droga-p, kandydat merge → v1.1): P1 = podwójny negatyw (semantyka cech nie ratuje detektorów — kandydat z I4 sfalsyfikowany), a P1c (świeże seedy 5–9, progi z góry) konwertuje obserwacje w twierdzenia: (a) BRAMA STRUKTURALNA D1>0.45 — SUKCES 60/60 na obu podłożach (odrzuca śmieć ORAZ swapy średnio/daleko-dystansowe; przepuszcza tylko bliskie); (b) PRAWO DYSTANSU — SUKCES MOCNY: wykrywalność swap rośnie z dystansem kotwic (separacja przy cos 0.487/0.139, brak przy 0.775/0.615 — negatywy I4/P1 wyjaśnione mechanizmem; próg między 0.615 a 0.487); (c) znak D2 odwrócony — obserwacja 4–5/5. POLITYKA PROTOKOŁU KOMPLETNA I ZMIERZONA: paczki+odwołanie · brama na wejściu · naprawa pokrywa niewykrywalne bliskie podmiany. Docs: DROGA_P_NOTATKI.md, CLAIMS 32c/32d, WHITEPAPER sekcja 21. Wcześniej tego dnia: PRZEGLĄD v1.0 ZAKOŃCZONY (tag v1.0, audyt liczb zero rozbieżności, CLAIMS.md, related work N/I4 — RELATED_WORK.md F/G). SERIA Q (branch droga-q) — Q1 ZAKOŃCZONE: SYGNAL− — pierwsza zmierzona BARIERA SKALI protokołu: kolektyw 34.02 ± 0.65 vs seq 40.70 ± 0.84 (pary −6.67 ± 0.88pp, 5/5; przy 10 klasach było −0.56 — wzrost ~12×). Struktura kosztu ODWRÓCONA względem uczenia: cały deficyt we WCZESNYCH adopcjach (~37% vs ~78% R[t][t]), późne adopcje = późne uczenie (ratio do sufitu 0.913 vs 0.870), forgetting kolektywu NIŻSZY (13.0 vs 18.3pp) — DROGA_Q_NOTATKI.md, CLAIMS 35/36. Q2 ZAKOŃCZONE — PODWÓJNY SYGNAL+, bariera ZAMKNIĘTA Z NADDATKIEM: q2a (re-adopcja paczek 1–5) +4.09pp = 61% bariery, hipoteza niedojrzałej projekcji potwierdzona (zadania 1–5: 26%→42%); q2b (budżet snu adopcji 500→2500, payload bez zmian) +10.26pp = 154% bariery → kolektyw 44.29 ± 0.66 (93.4% sufitu), NOMINALNIE NAD agentem sekwencyjnym (+3.59pp, 5/5 — SYGNAL+ w randze obserwacji). Q1 był niedoborem budżetu snu, nie informacji w 24 KB payloadzie. Q2c (kontrola uczciwości) ROZSTRZYGNĘŁO — SERIA Q KOMPLET: (1) self-dream augmentation = dźwignia KAŻDEGO agenta: seq 500 realnych + 2000 snów własnych/klasę → 45.35 ± 0.49 (+4.66pp, SYGNAL+ — nowy best pojedynczego agenta, 95.7% sufitu); (2) ROZSTRZYGNIĘCIE: przy symetrycznych budżetach kolektyw ≈ seq (44.29 vs 45.35, pary −1.07 ± 0.64, SZUM) — „+3.59 nad seq" z Q2 był artefaktem asymetrii budżetu. HEADLINE v1.2: bariera skali była artefaktem budżetu snu; równoważność kolektywu (I3/L2) rozszerzona z 10 klas/5 agentów na 100 klas/20 agentów, 95/100 klas wyłącznie z wiadomości 24 KB. Docs: DROGA_Q_NOTATKI.md (komplet), CLAIMS 35–40, WHITEPAPER sekcja 22. NASTĘPNE: merge droga-q → tag v1.2; potem decyzje: kombinacja q2a+q2b (opcjonalna, nie blokuje), G2b, R (PLAN_V1.md część B).

Poprzednia aktualizacja: 2026-07-20 (noc, finał). I4 (kolektyw niezaufany, v0.11) ZAKOŃCZONE: atak razi tylko paczkę adopcyjną (własne klasy odporne), detekcja na losowym backbone negatywna (uczciwie), naprawa zapomnij-i-adoptuj-ponownie PEŁNA (SZUM ×3). MAPA PROJEKTU WYCZERPANA (K·I·L·M·N·O·I4) — następny krok: przegląd całości pod v1.0. Wcześniej — seria O (v0.10) FALSYFIKACJĄ: odbudowa projekcji ze snów po sekwencji = parowy− na obu benchmarkach — sen chroni/przenosi/odbudowuje, ale nie ulepsza ponad realne dane; funkcje snu domknięte. Został I4, potem przegląd pod v1.0. Wcześniej — seria N (v0.9) ZAKOŃCZONA: taksonomia light/scrub/reinit zmierzona (0%/~84%/100% wymazania; reinit = pełna gwarancja przy koszcie ≤0; klasa nieznana projekcji routingowo nieosiągalna). Trójca protokołu domknięta: sen uczy · dzieli · zapomina. Kandydaci: O1 (konsolidacja snem — możliwy nowy best), I4. Wcześniej — seria M (długi horyzont, v0.8): CIFAR-100 × 20 zadań na pretrained — 40.70 ± 0.84 (85.8% sufitu; deficyt późny −7.8pp vs sufit, strukturalny — front stability–plasticity zmierzony w 3 punktach, rehearsal per klasę zostaje domyślny); kotwice: SYGNAL+ 300d vs 50d (+7.7) przy 100 klasach. Następne: seria N (unlearning), I4. Wcześniej — mapa K→I→L (v0.7): seria L — fork tożsamości: zamrożony resnet18-ImageNet pod niezmienionym mechanizmem daje CIFAR 74.69 ± 0.69 (+37.2pp, SYGNAL+; 96.7% sufitu 77.23; NAD trenowalnym joint 70.24), kolektyw na mocnych cechach 74.13 (koszt protokołu −0.56pp, parowy−); I2b — fuzja działa tylko poniżej nasycenia payloadu (parowy+ przy n=100). Wcześniej: K (v0.5) i I (v0.6): kolektywna wymiana snów, v0.6) ZAKOŃCZONE: mechanizm realizuje 94.6–97.6% sufitów zamrożonych cech (K0/K1, best Fashion 79.23 ± 0.73), a kolektyw 5 agentów wymieniających wyłącznie statystyki snu (24 KB/klasę, zero obrazów) jest RÓWNOWAŻNY agentowi sekwencyjnemu (78.87 vs 79.23 — I3). W przygotowaniu: Droga L (jawny fork tożsamości — pretrained backbone, DROGA_L_PLAN.md) i I2b (fuzja low-data). Mapa etapów: PLAN_GENERALNY.md. REPO PUBLICZNE: https://github.com/develforever/m-a-r-s (tagi: v0.3-freeze … v0.6).
DECYZJE 2026-07-10: (1) treść rozcięta na DWA papery (A: CL; B: routing
ceiling); (2) submisje (arXiv/TMLR/konferencje) WSTRZYMANE do osobnej
decyzji Roberta — etap bieżący to WYŁĄCZNIE repo publiczne.
Related work zweryfikowany → `RELATED_WORK.md`. Plan → `private/PLAN_PUBLIKACJI.md`.

## Prompt otwierający nową sesję (skopiuj i wklej)

> Kontynuujemy projekt M.A.R.S. Stan: mapa K·I·L·M·N·O·I4 WYCZERPANA
> (tag v0.11); trwa przegląd pod v1.0 wg PLAN_V1.md (część A —
> konsolidacja; stan odhaczony w nagłówku START_TUTAJ.md). Po v1.0
> kandydaci na serie: P (detekcja zatrucia na pretrained), Q (kolektyw
> × długi horyzont), G2b (ECOC), R (heterogeniczny) — PLAN_V1.md część
> B; każda wymaga pre-rejestrowanego DROGA_*_PLAN.md PRZED runami.
> Treść rozcięta na dwa papery (A: CL, B: routing ceiling); submisje
> WSTRZYMANE — etap bieżący to wyłącznie repo publiczne (checklist
> w private/PLAN_PUBLIKACJI.md pkt 3). Przeczytaj START_TUTAJ.md,
> PLAN_V1.md, CLAIMS.md i WHITEPAPER.md (w razie potrzeby
> DROGA_*_NOTATKI.md). Nie proponuj submisji, dopóki sam nie wrócę
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

## Następne kroki (mapa: PLAN_V1.md)

1. Dokończyć część A przeglądu v1.0: related work N/I4 (machine
   unlearning; data poisoning/byzantine FL — każde źródło zweryfikowane
   webem), decyzje Roberta: renormalizacja CRLF, tag v0.6, tag v1.0.
2. Po v1.0 — wybór serii (część B): P (detekcja na pretrained, tanio,
   domyka negatyw I4) → Q (kolektyw × długi horyzont, kandydat na
   headline) → G2b (ECOC); R (heterogeniczny) tylko za osobną zgodą.
3. Submisje pozostają WSTRZYMANE (private/PLAN_PUBLIKACJI.md) —
   historyczna lista kroków publikacyjnych tamże, pkt 4.

## Zasady współpracy (obowiązują każdego agenta)

- Eksperymenty uruchamia WYŁĄCZNIE Robert, lokalnie (GTX 1050 Ti).
- Wyniki do `results/` pisze jeden tor pracy; inni agenci czytają.
- 5 seedów, progi szumu, kryteria werdyktu PRZED uruchomieniem,
  min per-seed raportowany, negatyw = wynik.
- Kod zamrożony do publikacji — zmiany tylko za zgodą Roberta.
