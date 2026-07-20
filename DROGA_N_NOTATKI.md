# Droga N — notatki robocze

Plan: `DROGA_N_PLAN.md` (pre-rejestracja 2026-07-20).
Runy: Robert, lokalnie, 5 seedów, konfiguracja K1.

## N1 (ZAKOŃCZONE, 20.07.2026): poziom 1 ZALICZONY;
## poziom 2 — INSTRUMENT NIEWAŻNY (floor effect), naprawa w N1b

Plik: `src/run_N1_unlearning.py`; wyniki: `results/N1_unlearning.json`.
Czas: 118 s (200 usunięć, 15 relearnów, 5 kontroli never).

**Poziom 1 (funkcjonalny) — macierz 10 usunięć × light/scrub:**

| Pomiar | light | scrub |
|---|---|---|
| acc klasy usuniętej | 0.00 (10/10 klas, 5/5 seedów) | 0.00 |
| Δ acc pozostałych 9 klas | +1.41pp (pary 5/5) | +1.25pp (pary 5/5) |

Usunięcie działa z konstrukcji i NIE uszkadza pozostałych klas —
przeciwnie, lekko pomaga. Zastrzeżenie uczciwości (ta sama lekcja co
w M1): część +1.4pp to mechanika maski (argmax po 9 zamiast 10 klas),
nie poprawa systemu; dla wniosku „brak szkód ubocznych" wystarcza
kierunek nieujemny. Scrub nie kosztuje pozostałych (≈light).
**Werdykt poziomu 1: SUKCES** (usunięcie selektywne, bez szkód).

**Poziom 2 (informacyjny) — UNIEWAŻNIONY, nie rozstrzygnięty:**
relearn(n=100) dał ~0% we WSZYSTKICH ścieżkach (light 0.28 ± 0.15,
scrub 0.36 ± 0.26, never 0.00 ± 0.00; pełny system: 69.5). Formalne
werdykty (light vs never „SYGNAL+ +0.28pp") są puste — porównują
podłogi szumu. Diagnoza instrumentu: `relearn_small` trenował pod na
100 pozytywach vs 2304 negatywach ze snów (replay_per_class=256 × 9
klas) — CE uczy klasyfikatora „nigdy nie przewiduj c". Błąd
konstrukcji instrumentu, nie własność systemu; ta sama klasa błędu
skalowania budżetów co M1/M1b (budżet per klasę vs łączny).
Ciekawostka do odnotowania (bez rangi): never = dokładnie 0.00 w 5/5
seedów, light/scrub > 0 w 9/10 przypadków — sugestia resztkowej
informacji, ale nierozstrzygalna przy podłodze.

**Naprawa: N1b (dopisek do planu PRZED runem)** — relearn ze
zbalansowanym budżetem negatywów (łącznie ≈ liczbie pozytywów).

## N1b — relearn zbalansowany (ZAKOŃCZONE, 20.07.2026): poziom 2
## ROZSTRZYGNIĘTY — light nie wymazuje NIC, scrub ściera ~84%,
## klasa nigdy nie widziana jest routingowo nieosiągalna

Plik: `src/run_N1b_relearn_balanced.py`;
wyniki: `results/N1b_relearn_balanced.json`. Czas: 76 s.

| Ścieżka (acc c*=4 po relearn n=100, maska 9 klas) | ACC |
|---|---|
| referencja: pełny system | 69.50 ± 3.30% |
| **relearn po unlearn_light** | **69.50 ± 3.30% (= full CO DO 4 MIEJSC, 5/5)** |
| relearn po unlearn_scrub | 11.22 ± 3.44% |
| relearn po never (kontrola zerowa) | **0.00 ± 0.00% (twarde zero, 5/5)** |

**WERDYKTY (pre-rejestrowane):**
- GŁÓWNE 2 (light vs never): **SYGNAL+ +69.50pp** — unlearn_light
  usuwa DOSTĘP, nie INFORMACJĘ: 100 obrazów przywraca klasę do
  identycznej sprawności (predykcje bitowo równe pełnemu systemowi).
- GŁÓWNE 3 (scrub vs never): **SYGNAL+ +11.22pp** (pary 5/5
  [7.7…15.4]) — douczanie na snach pozostałych ściera ~84%
  odzyskiwalności (69.5 → 11.2), ale ZOSTAWIA zmierzony ślad;
  ryzyko pre-rejestrowane („scrub douczaniem może nie wymazać")
  zmaterializowane i skwantyfikowane.

**Ustalenia:**
1. **Nośnikiem informacji o klasie jest wyłącznie projekcja; pody są
   konfirmacyjne.** Light (projekcja nietknięta) → pełny powrót ze
   100 próbek; acc(c) jest limitowane routingiem, świeży pod na 100
   przykładach wystarcza w 100%.
2. **Taksonomia zapominania zmierzona:** light = zawieszenie dostępu
   (odwracalne, natychmiastowe, zero kosztu); scrub = częściowe
   wymazanie (resztka 11.2pp); pełna gwarancja wymaga N1c.
3. **Własność bezpieczeństwa przy okazji:** klasa, której projekcja
   nigdy nie widziała, jest routingowo nieosiągalna (twarde 0.00 mimo
   dodanej kotwicy i poda) — system nie umie przewidywać klas, których
   go nie nauczono, nawet mając ich nazwę.

Status: N1c (dopisek PRZED runem) — pełne wymazanie przez reinicjalizację.
