# Droga L — notatki robocze

Plan: `DROGA_L_PLAN.md` (pre-rejestracja 2026-07-17; dopisek
implementacyjny cache cech 2026-07-19 — bez zmiany semantyki).
Runy: Robert, lokalnie, 5 seedów, epochs=15. Encoder: resnet18
IMAGENET1K_V1 (zamrożony) → losowa zamrożona projekcja 512→128 → ReLU;
interfejs D=128, pamięć klasy i payload protokołu I bez zmian.
Werdykty przeliczone niezależnie z JSON-ów: zgodne.

## L1 — pretrained backbone na CIFAR (ZAKOŃCZONE, 19.07.2026):
## SYGNAL+ +37.18pp; mechanizm realizuje 96.7% nowego sufitu;
## SEKWENCYJNY AGENT BIJE TRENOWALNY JOINT

Plik: `src/run_L1_pretrained.py`; wyniki: `results/L1_pretrained.json`.
Czas: 50 s (+ jednorazowa ekstrakcja cech do cache).

| System (CIFAR-n, class-IL) | ACC | min | F |
|---|---|---|---|
| [J2b] sparse_k16, losowy backbone | 37.51 ± 1.35% | 36.20% | 32.7pp |
| [K0] sufit losowych cech | 39.65 ± 1.21% | — | — |
| [J2] joint (trenowalny monolit) | 70.24 ± 0.69% | 69.19% | — |
| **l1_seq (sparse_k16, pretrained)** | **74.69 ± 0.69%** | **73.99%** | **18.1pp** |
| l1_all (sufit pretrained) | 77.23 ± 0.57% | 76.50% | 10.3pp |

**WERDYKT (pre-rejestrowany): SYGNAL+** — +37.18pp (min per-seed
+35.67) przy progu 2.04. Największa delta w historii projektu.

**Ustalenia:**
1. **Teza „representation-agnostic" zmierzona, nie deklarowana:**
   gap_mech_L = 2.54pp → mechanizm realizuje **96.7% sufitu** także na
   mocnych cechach (na losowych: 94.6%). Mechanizm żyłuje każdą
   reprezentację, jaką dostanie — dokładnie po to był Etap L.
2. **Zamrożone + sen > trenowalne joint:** l1_seq (74.69) bije joint
   (70.24) o +4.45pp — uczciwy CL sekwencyjny na zamrożonym pretrained
   wygrywa z monolitem trenowanym na wszystkich danych naraz.
   Rozróżnienie pre-rejestrowane rozstrzygnięte: fork działa przez
   encoder, nie przez losowy rzut (sufit 39.65 → 77.23, +37.58pp).
3. **Prognoza K0 potwierdzona:** luka 30.6pp zdiagnozowana jako
   reprezentacyjna została zamknięta niemal w całości podmianą
   reprezentacji, bez dotykania mechanizmu.
4. Narracja: linia „foundation-embedding" raportowana osobno od
   głównej „from-scratch"; oś zasobów (pretraining ImageNet 1.28M
   obrazów) jawna.

## L2 — kolektyw na mocnych cechach (ZAKOŃCZONE, 19.07.2026):
## SYGNAL-parowy− — protokół przenosi się z małym, zmierzonym
## kosztem ~0.6pp; KOLEKTYW NAD TRENOWALNYM JOINT

Plik: `src/run_L2_collective_cifar.py`;
wyniki: `results/L2_collective_cifar.json`. Czas: 50 s.

| System (CIFAR-n, class-IL) | ACC | min |
|---|---|---|
| [L1] agent sekwencyjny (pretrained) | 74.69 ± 0.69% | 73.99% |
| **kolektyw N=5 (task0 z danych + 8 klas ze snów)** | **74.13 ± 0.57%** | 73.47% |

**WERDYKT (pre-rejestrowany): SYGNAL-parowy−** — −0.56pp, pary 5/5
ujemne (−0.33…−0.95), |śr.| 0.56 > 2×std(delt) 0.47, próg klasyczny
1.26 nieprzekroczony. Luka do sufitu l1_all: 3.10pp.

**Ustalenia:**
1. **Pierwszy zmierzony koszt protokołu wymiany snów:** na Fashion
   (I3) kolektyw był równoważny agentowi sekwencyjnemu (SZUM, −0.36);
   na trudniejszych danych koszt staje się systematyczny, ale mały:
   **−0.56pp za nauczenie 8/10 klas wyłącznie z wiadomości 24 KB**.
   Limit protokołu istnieje i wynosi ~pół punktu — to wynik, nie
   porażka.
2. **Kolektyw bije trenowalny joint o +3.89pp** (74.13 vs 70.24):
   pięciu agentów, każdy widział 2 klasy, zero wymienionych obrazów —
   lepsi niż scentralizowany monolit trenowany na wszystkim naraz.
   To jest headline linii foundation.
3. Krzywa po adopcjach (89 → 78.5 → 74.8 → 74.1) — spadek odzwierciedla
   rosnącą liczbę klas, bez kolapsu; kumulacja dryfu przez 4 adopcje
   ponownie nieobecna.

## STATUS: mapa PLAN_GENERALNY K→I→L DOMKNIĘTA PIERWSZYM PRZEJŚCIEM

K: mechanizm wyżyłowany (94.6–97.6% sufitów) · I: kolektyw równoważny
na Fashion, przeszczep −1.29pp, fuzja działa w reżimie low-data (I2b)
· L: fork podnosi sufit +37.6pp, mechanizm trzyma 96.7%, kolektyw
przenosi się z kosztem −0.56pp i bije joint. Zasada warstw zadziałała:
trzy etapy, każdy zmienił jedną warstwę, żaden nie unieważnił
poprzednich wyników.

Kandydaci dalej (decyzje Roberta, NIE pre-rejestrowane): seria M —
długi horyzont (Split-CIFAR-100, 20 zadań, na pretrained; test loss-of-
plasticity), I4 — weryfikacja payloadu przez kotwicę (kolektyw
niezaufany), akapit model collapse → RELATED_WORK (zrobione przy
merge v0.7).
