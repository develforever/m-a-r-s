# Droga E — notatki robocze

Plan: `DROGA_E_PLAN.md`. Kontekst: seria D domknięta (sufit routingu D4+D5+D7
+ stress-test S3; dźwignia cech D6; efektywność D6b). Whitepaper v0.2.

## E1 — Anatomia błędu (ZAKOŃCZONE, 06.07.2026)

Pliki: `src/run_E1_error_anatomy.py`, wyniki: `results/E1_error_anatomy.json`.
Setup: 5 seedów × 30 epok, pełny CNN D6 (luka router→oracle 6.11pp Fashion).

**Fashion-MNIST — luka MA strukturę (kryterium E2 spełnione):**

1. **Koncentracja: top-4 pary = 60% błędów routera** (per-seed 57–61%; kryterium ≥60% na granicy, ale patrz p. 2 — realna koncentracja jest silniejsza).
2. **Wszystkie top-6 par leżą WEWNĄTRZ jednego klastra upper-body** {T-shirt, Pullover, Dress, Coat, Shirt}: T-shirt↔Shirt 901, Pullover↔Shirt 532, Coat↔Shirt 491, Pullover↔Coat 473, Dress↔Shirt 251, Dress↔Coat 234 (sumy z 5 seedów). Epicentrum: **Shirt** (1103 z 3240 próbek odzyskiwalnych = 34%).
3. **86% próbek odzyskiwalnych ma klasę upper-body** (2785/3240). Grupa proponowana w E2 pokrywa problem niemal w całości.
4. **86% odzyskiwalnych w Q1 pewności routera** — luka jest niskopewna; router "wie, że nie wie".
5. Dekompozycja: pod miss (B) = 4–61/seed, odzyskanie (C) = 6–93/seed — oba marginalne. Luka ≈ czysta strata routingowa (D ~766/seed).

**MNIST — kontrola, zgodnie z oczekiwaniem słaba struktura:**
koncentracja 42% (<60%), B=0 i C=0 na WSZYSTKICH seedach (pody idealne, oracle
100.00%), odzyskiwalne ~82/seed, 99% w Q1. Hub konfuzji: cyfra 9 (4↔9=63,
7↔9=29, 8↔9=18). E2 na MNIST ma charakter kontrolny.

**Decyzje po E1:**
- Grupy Fashion do E2 POTWIERDZONE bez zmian: {0,2,3,4,6} / {5,7,9} / {1} / {8}.
- Grupy MNIST SKORYGOWANE z danych (greedy pokrycie par): {2,4,7,9} / {3,5} /
  {0,6} / {1} / {8} — pokrywa 194 błędów vs 165 w wersji wstępnej.
- E5 (sygnały spoza reprezentacji) dostaje zielone światło warunkowe: luka
  jest niskopewna, więc mechanizmy selektywne mają target — ale lekcja D4
  obowiązuje: sygnał musi być NOWY, nie ten sam kanał ponownie.

## E2 — Routing hierarchiczny (KOD v2, przed pełnym runem)

Pliki: `src/mars_v2_hier.py`, `src/run_E2_hierarchical.py`.
Hipoteza, kryterium werdyktu i sweep pod_hidden {24,64} — pre-rejestrowane
w docstringu runnera. Baseline: płaski CNN D6 per-seed.

### E2 — WYNIKI (ZAKOŃCZONE, 06.07.2026): SYGNAL+ na Fashion (marginalny, ale realny)

Wyniki: `results/E2_hierarchical.json`. 5 seedów × 30 epok × ph {24,64}.

| Fashion | system | grupRout | oracle(hier) | MAC | Δ vs flat |
|---|---|---|---|---|---|
| flat CNN (D6) | 91.99 ± 0.12% | rout 91.98% | 98.46% (flat) | 4 247 600 | — |
| **hier ph=24** | **92.20 ± 0.12%** | **99.53%** | **92.61%** | **4 247 408** | **+0.21 ± 0.06pp** |
| hier ph=64 | 92.06 ± 0.06% | 99.54% | 92.45% | 4 252 928 | +0.07 ± 0.14pp |

**WERDYKT (Fashion, ph24, próg 0.18pp): SYGNAL+** — wszystkie seedy dodatnie,
przy MAC minimalnie NIŻSZYM niż flat. **Nowy najlepszy wynik projektu: 92.20%.**
MNIST (kontrola): SZUM (+0.04/+0.07pp, próg 0.11) — zgodnie z oczekiwaniem.

**Kluczowe obserwacje:**
1. **Struktura daje dywidendę tam, gdzie algorytmy dały zero** (D4/D5/D7:
   0.00 / −2.5 / +0.05pp; E2: +0.21pp powyżej progu). Skromna, ale pierwsza.
2. **NAJWAŻNIEJSZE — rekalibracja sufitu.** Hier: routing grupowy 99.5%,
   system≈oracle (luka 0.41pp vs 6.11pp flat). Ale zysk absolutny to tylko
   +0.21pp, bo gardło przeniosło się do rozróżniania wewnątrz grupy. Wniosek:
   **flat oracle (98.10%) był ZAWYŻONY** — wybór poda po prawdziwej etykiecie
   przecieka informację o etykiecie (pod "wie", że próbka jest jego klasy).
   Oracle hierarchii (92.61%, przeciek tylko grupy) to uczciwszy sufit tej
   reprezentacji. Realna przestrzeń nad flat routerem wynosiła ~0.5pp, nie
   6pp — i E2 zebrał z niej połowę. Luka z E1 była w większości iluzją
   pomiarową, nie niewykorzystaną informacją.
3. **ph=64 gorszy od ph=24** — pojemność głowicy poda nie jest ograniczeniem;
   ograniczeniem jest informacja w cechach. Spójne z całą serią D.
4. Konsekwencja strategiczna: na tej reprezentacji i tym zbiorze routing jest
   DOMKNIĘTY (99.5% grup, system przy uczciwym suficie). Dalszy wzrost =
   wyłącznie cechy i skala → **E3 (CIFAR-10) i E4 (dżule) są teraz jedyną
   grą.** Na CIFAR (więcej klas, głębsza hierarchia) dywidenda strukturalna
   może rosnąć — hipoteza do E3.

## E4 — Stos efektywności + czas GPU (ZAKOŃCZONE, 06.07.2026)

Wyniki: `results/E4_energy.json`. Pomiar: s/10k (proxy energii przy nasyceniu,
util 99–100% wszędzie; GTX 1050 Ti nie raportuje power.draw — bound 75 W).
3 seedy acc, 5× pomiar czasu, batche {512, 4096}.

| System (Fashion) | acc | MAC | s/10k (b4096) |
|---|---|---|---|
| mono_mlp | 89.28 ± 0.20% | 234 752 | **0.0035** |
| mono_s2 (1 głowica) | 90.80 ± 0.21% | 383 872 | 0.0503 |
| mars_s2 | **91.11 ± 0.07%** | 390 320 | 0.0564 |
| mars_s2 + ternary(0.5) | 88.39 ± 2.78% (!) | 390 320 | 0.0556 |
| mars_full (seed0) | 92.00% (= D6 ✓) | 4 247 600 | 0.1834 |

**Wnioski (z dwiema uczciwymi korektami werdyktów):**
1. **V2 NEGATYWNY (zgodnie z pre-rejestrowanym ryzykiem):** MARS-S2 ma
   +1.83pp nad mono-MLP, ale 16× więcej czasu GPU. MAC NIE przewiduje czasu
   na CUDA (MLP: ~50% peak; slim conv: ~3%). Teza energetyczna na desktopowym
   GPU zamknięta ostatecznie; właściwy sprzęt dla małych podów = CPU SIMD /
   NPU (patrz `ARSENAL_PRZEOCZONYCH_NARZEDZI.md`).
2. **V1 ternary — KOREKTA: formalnie "POTWIERDZONE", faktycznie NIE.**
   Kryterium |Δ|≤szum przeszło TYLKO przez rozdęty próg (std delty 2.77pp →
   próg 2.84pp). Realnie: Δ = −2.72 ± 2.77pp, seed 0 traci −5.88pp.
   **Wynik B8 ("ternary za darmo") nie replikuje się w stosie v2** — pody
   128→24→10 trenowane na routowanych próbkach nie mają redundancji podów
   B8 (większych, trenowanych na pełnych danych). Lekcja: kwantyzowalność
   zależy od rozmiaru i reżimu treningu głowic; do whitepapera jako caveat.
   Wniosek metodologiczny: kryterium "≤ próg szumu" jest podatne na
   przejście przez WYSOKĄ wariancję — w przyszłych planach dodawać warunek
   na max stratę per-seed.
3. **Narzut modularności mały i się opłaca:** mars_s2 vs mono_s2 (ten sam
   backbone) = 1.12× czasu, +0.31pp acc (na granicy szumu). Routing + pody
   nie są ciężarem.
4. Reprodukowalność: mars_full seed0 92.00% = D6 91.99% ✓.

**Lekcja ze smoke'a v1 (06.07.2026) — gruboziarnisty trening niszczy
drobnoziarniste cechy.** Wersja 1 uczyła backbone w fazie 1 na etykietach
GRUP (4-way CE). Wynik: routing grupowy 99.4%, ale oracle ZAPADŁ SIĘ do 68%
(Fashion, ph=24) — reprezentacja nauczyła się rozdzielać grupy i zgubiła
informację wewnątrzgrupową; zamrożone cechy przestały nieść to, co pod ma
rozróżniać. Obserwacja wartościowa sama w sobie (analogia do collapse przy
coarse supervision). Poprawka v2: trening trójfazowy — faza 1 identyczna
z płaskim v2 (10-way CE → TE SAME cechy co baseline; test czysto struktury
decyzji), faza 2 głowica grupowa na zamrożonych cechach, faza 3 pody na
realnym routingu grupowym. Głowica klasowa poza inferencją (MAC bez zmian).
