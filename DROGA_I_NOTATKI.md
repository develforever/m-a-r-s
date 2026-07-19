# Droga I — notatki robocze

Plan: `DROGA_I_PLAN.md` (pre-rejestracja 2026-07-17). Runy: Robert,
lokalnie, 5 seedów, epochs=15, konfiguracja K1 (sparse_k16 × 300d),
n_dream=6000. Werdykty przeliczone niezależnie z JSON-ów: zgodne.

## I1 — przeszczep klasy (ZAKOŃCZONE, 17.07.2026): SUKCES SŁABY
## (strata 1.29pp, o 0.2pp od progu równoważności); pełny ACC = SZUM
## (równoważność z nauką lokalną). BRAMKA OTWARTA.

Plik: `src/run_I1_transplant.py`; wyniki: `results/I1_transplant.json`.
Czas: 88 s.

| Wariant (Fashion, class-IL) | ACC | przeszczep (task4) | stare (0–3) |
|---|---|---|---|
| [K1] local seq (baza) | 79.23 ± 0.73% | 95.55% | — |
| **transplant_end** | **79.02 ± 0.90%** | **94.26%** | 75.22% |
| transplant_mid (I1b) | 79.35 ± 0.99% | 91.42% | 76.33% |

**WERDYKTY (pre-rejestrowane, progi zatwierdzone z góry):**
- Strata przeszczepu: +1.29pp (pary [0.90…1.65], 5/5 dodatnie) przy
  progu szumu 1.09 → **SUKCES SŁABY** (< 3pp; do MOCNEGO zabrakło
  0.2pp). Bramka na I2/I3 OTWARTA.
- Pełny ACC transplant_end vs local: −0.20pp, pary mieszane → **SZUM**
  = agent z klasami przeszczepionymi jest RÓWNOWAŻNY agentowi uczonemu
  w całości z danych.
- I1b (obserwacja): przeszczep w środku sekwencji traci na klasie
  przeszczepionej 2.84pp więcej niż na końcu (5/5 par, [2.25…3.40]) —
  późniejsza nauka zjada przeszczep szybciej niż własne klasy — ALE
  odzyskuje na starych (+1.11pp) i na pełnym ACC (79.35, nominalnie
  NAD local): timing przesuwa koszt między klasami, nie zmienia sumy.

**Ustalenie główne: klasa nauczona WYŁĄCZNIE z wiadomości 24 KB
(statystyki spike-and-slab), bez jednego obrazu, osiąga 94.26% tam,
gdzie nauka na 12000 realnych obrazów daje 95.55%.** Protokół wymiany
snów działa niemal bezstratnie.

## I2 — fuzja statystyk (ZAKOŃCZONE, 17.07.2026): SZUM ×3 —
## fuzja zbędna, bo payload NASYCA SIĘ już na połowie danych

Plik: `src/run_I2_fusion.py`; wyniki: `results/I2_fusion.json`.
Czas: 158 s.

| Wariant (przeszczep task4) | ACC task4 | ACC |
|---|---|---|
| half_A (3000 obrazów) | 94.38 ± 0.82% | 79.00% |
| fusion_cat [2k] | 94.33 ± 0.78% | 79.04% |
| fusion_red [k16] | 93.66 ± 1.26% | 78.97% |
| full_stats (6000 obrazów) | 94.17 ± 0.77% | 79.04% |

**WERDYKTY:** fusion_cat vs half_A: SZUM (−0.05); fusion_cat vs full:
SZUM (+0.16); fusion_red vs cat: SZUM (−0.67; 5/5 par ujemnych pod
progiem parowym — kierunkowy koszt kompresji k-means na snach).

**Ustalenia:**
1. **Statystyki klasy nasycają się szybko:** payload z 3000 obrazów =
   payload z 6000 (−0.21pp, szum). Wiadomość jest tania w danych —
   dobra wiadomość dla protokołu, zła dla testu fuzji: przy nasyconym
   payloadzie fuzja nie ma czego dodać.
2. Kandydat I2b (NIE pre-rejestrowany, decyzja Roberta): fuzja
   w reżimie małych próbek (np. 100–500 obrazów/agenta) — tam suma
   dwóch częściowych widoków może realnie przewyższać każdy z osobna.
3. Kompresja redream kosztuje ~0.7pp kierunkowo — cat (2× pamięć,
   bezstratna) preferowana, gdy 48 KB/klasę nie boli.

## I3 — kolektyw N=5 (ZAKOŃCZONE, 17.07.2026): RÓWNOWAŻNOŚĆ
## Z AGENTEM SEKWENCYJNYM — headline rewolucji potwierdzony

Plik: `src/run_I3_collective.py`; wyniki: `results/I3_collective.json`.
Czas: 56 s.

| System (Fashion, class-IL) | ACC | min |
|---|---|---|
| [K1] agent sekwencyjny (5 tasków z danych) | 79.23 ± 0.73% | 78.38% |
| [F0] replay-200 (bufor obrazów) | 76.97 ± 1.09% | — |
| **kolektyw N=5 (task0 z danych + 8 klas ze snów)** | **78.87 ± 1.01%** | 78.13% |

**WERDYKT (pre-rejestrowany): SZUM = RÓWNOWAŻNOŚĆ** — −0.36pp
(pary mieszane [+0.04, −0.89, −0.16, +0.08, −0.87]) przy progu 1.74.
Luka do sufitu g1_all_300 (81.16): 2.29pp. Kumulacja dryfu przez
4 kolejne adopcje NIE zmaterializowała się (ryzyko pre-rejestrowane
odrzucone); klasa własna kolektora trzyma 84–87% na końcu.

**Ustalenie główne (headline): pięciu agentów, z których każdy widział
wyłącznie 2 klasy, wymieniając 8 wiadomości po ~24 KB (zero obrazów,
zero gradientów, zero wag), osiąga wynik statystycznie równoważny
agentowi uczonemu sekwencyjnie na wszystkich danych (78.87 vs 79.23)
i nominalnie lepszy od replay-200 z buforem obrazów (+1.90pp).**
8 z 10 klas kolektora nauczone wyłącznie ze snów.

## STATUS KOŃCOWY SERII I: KOMPLET

I1 SUKCES SŁABY (przeszczep −1.29pp od lokalnej; pełny ACC równoważny) ·
I1b obserwacja timingu · I2 SZUM (nasycenie payloadu — fuzja zbędna
przy pełnych danych; kandydat I2b low-data) · I3 RÓWNOWAŻNOŚĆ
(rewolucja w formie równoważności, nie przewagi).

Teza Drogi I potwierdzona: **kolektywne uczenie bez wymiany danych
działa** — na wspólnym losowym backbone statystyki spike-and-slab są
wystarczającym nośnikiem wiedzy o klasie. Dalej wg PLAN_GENERALNY:
merge `droga-i` → main (decyzja Roberta, tag v0.6), aktualizacja
WHITEPAPER (część Serii I), potem decyzja o Etapie L (fork tożsamości:
mocniejszy zamrożony backbone pod ten sam mechanizm i protokół —
jedyna droga w górę na CIFAR wg K0).

## I2b — fuzja low-data (ZAKOŃCZONE, 19.07.2026): SYGNAL-parowy+
## przy n=100 — fuzja działa dokładnie tam, gdzie payload nienasycony

Plik: `src/run_I2b_fusion_lowdata.py`;
wyniki: `results/I2b_fusion_lowdata.json`. Czas: 155 s.

| Wariant (acc task4) | ACC task4 | Werdykt fuzja vs half |
|---|---|---|
| half_100 | 91.82 ± 1.08% | — |
| fusion_100 | 93.04 ± 1.10% | **SYGNAL-parowy+** (+1.22, pary 5/5 [0.90…1.60], 2×std 0.52) |
| half_500 | 93.91 ± 0.76% | — |
| fusion_500 | 94.10 ± 0.87% | SZUM (+0.19, pary mieszane) |

**Ustalenia:**
1. Hipoteza I2b potwierdzona w obu kierunkach: fuzja dwóch częściowych
   widoków daje systematyczny zysk przy payloadzie nienasyconym (n=100)
   i znika przy nasyconym (n=500) — interpretacja nulla z I2
   („nasycenie, nie bezużyteczność fuzji") jest teraz przyczynowa.
2. Krzywa nasycenia wiadomości 24 KB: 91.8 (100) → 93.9 (500) →
   94.4 (3000) → 94.2 (6000) obrazów/klasę — payload osiąga sufit
   między 500 a 3000 próbek. Wiadomość jest tania w danych.
3. Wniosek protokolarny: fuzję warto włączać tylko dla klas z małą
   liczbą próbek u nadawców; przy nasyceniu wystarczy pojedynczy widok.
