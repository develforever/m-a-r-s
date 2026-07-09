# M.A.R.S. — DROGA A: Prawdziwa specjalizacja — PLAN (poparty pomiarami)

Data: 2026-06-15
Decyzja: budujemy prawdziwą specjalizację podów (cel projektu od początku).
Metoda: każdy krok poparty pomiarem, nie wiarą.

---

## Diagnoza wyjściowa (zmierzona)

Obecny M.A.R.S. ma router-atrapę: pody są redundantnymi pełnymi
klasyfikatorami (każdy umie 10 klas), więc router prawie nie wpływa na wynik.
To NIE jest prawdziwa specjalizacja z dokumentów ("Specialist Pods").

---

## Kluczowy trade-off (zmierzony)

Gdy pody są WĄSKIE (prawdziwi specjaliści), system accuracy = router accuracy.
Router przestaje być atrapą — staje się wszystkim.

| router acc | pody szerokie (obecne) | pody wąskie (specjaliści) |
|---|---|---|
| 40% | 96% | 40% ← katastrofa |
| 80% | 96% | 79% |
| 95% | 96% | 94% |
| 100% | 96% | 99% |

**Wniosek:** specjalizacja opłaca się TYLKO gdy router > ~85%.
Nasz router na MNIST: ~44%. Wąskie pody dałyby teraz ~44% (katastrofa).

---

## Dlaczego router daje tylko 44%? (zmierzone)

Wąskie gardło: router ściska 784 wymiary do **2D** (wymóg tekstury UV).
2D to za mało na 10 klas.

| bottleneck routera | accuracy |
|---|---|
| 2 (UV tekstury, obecne) | 79.9% |
| 4 | 100% |
| 8+ | 100% |

Już 4 wymiary wystarczą. **2D-UV tekstury jest fundamentalnym ograniczeniem.**

---

## PLAN DROGI A (kolejność wymuszona przez zależności)

### Krok A1 — Lepszy router (FUNDAMENT, bez tego reszta nie ma sensu)
Zmienić router z 2D-UV (tekstura) na większy bottleneck (≥4D).
- KOSZT: tekstura 2D przestaje wystarczać jako mechanizm routingu.
  Trzeba zastąpić texture-lookup małym MLP (4→N_pods) albo
  użyć wielowymiarowej tekstury / wielu tekstur.
- CEL: router accuracy >85% na MNIST.
- To OZNACZA porzucenie czystego "TMU lookup" jako routera — ale tekstura
  i tak była już tylko darmowym dodatkiem, nie źródłem przewagi (audyt).

### Krok A2 — Wąskie pody (prawdziwa specjalizacja)
Gdy router >85%: zwęzić pody.
- Każdy pod uczony TYLKO na swoich danych (nie na wszystkich).
- Pod ma wąskie wyjście (1-2 klasy zamiast 10).
- To realizuje "Specialist Pods" z dokumentów.

### Krok A3 — Pomiar prawdy
Zmierzyć: system accuracy, MAC, throughput dla specjalistów vs redundantnych.
- Pytanie: czy specjalizacja daje DODATKOWĄ oszczędność (mniejsze pody)
  przy zachowaniu accuracy?
- Tu specjalizacja albo pokaże przewagę (mniejsze pody = mniej MAC),
  albo jej granicę (router nie dociąga).

### Krok A4 — Catastrophic forgetting na specjalistach
Sprawdzić, czy wąskie pody + lepszy router rozwiązują problem retencji
(naprawa dryfu encodera z audytu A4).

---

## Uczciwa ocena szans

Droga A ma sens i jest zmierzalna, ale to NIE jest pewna rewolucja:
- Plus: jeśli zadziała, mamy prawdziwą modularność + lepszą oszczędność
  (mniejsze pody) + router który ma znaczenie. To mocna teza.
- Ryzyko: "router >85% + wąskie pody" może po prostu odtworzyć zwykły
  Mixture-of-Experts, który już istnieje w literaturze. Wtedy wkład jest
  inżynierski (efektywna implementacja), nie przełomowy naukowo.
- Realna wartość: nawet jeśli to "tylko" dobry MoE — działający, zmierzony,
  energooszczędny MoE na zwykłym GPU to wciąż solidne portfolio.

Pierwszy krok (A1: lepszy router) jest niezależny od oceny szans —
trzeba go zrobić tak czy inaczej. Zaczynamy od niego.
