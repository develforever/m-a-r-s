# M.A.R.S. — top-2 routing: WYNIK NEGATYWNY (ale pouczający)

Data: 2026-06-15
Status: top-2 ODRZUCONE. Ale odkrycie cenne dla zrozumienia architektury.

---

## Wynik (MNIST, GPU)

| Strategia | accuracy | pod MAC | total | oszczędność |
|---|---|---|---|---|
| top-1 (1 pod) | 96.7% | 19,056 | 44,816 | 80.9% |
| top-2 confidence | 94.8% | 38,112 | 63,872 | 72.8% |
| top-2 agregacja | 94.7% | 38,112 | 63,872 | 72.8% |
| ORACLE (sufit) | 99.0% | — | — | — |

**Top-2 ZASZKODZIŁO: -1.9pp, przy 2× koszcie podów.** Kontrintuicyjne —
w klasycznym MoE top-2 zwykle pomaga.

---

## Dlaczego top-2 szkodzi — głębokie odkrycie

Dekompozycja:
- Gdy top-1 poprawny (96.7% próbek): drugi pod może zepsuć dobrą odpowiedź.
  Dużo przypadków do ZEPSUCIA.
- Gdy top-1 błędny (3.3% próbek): drugi pod może uratować. Mało do ZYSKANIA.
- Netto: straty > zyski, bo top-1 jest już dobry.

**Mechanizm głębszy:** top-2 (ensemble) pomaga, gdy eksperci się UZUPEŁNIAJĄ.
Nasi specjaliści NIE uzupełniają się — każdy zna tylko swoją klasę. Drugi
pod to nie "druga opinia eksperta", lecz "opinia laika" o cudzej klasie.

---

## To jest CECHA, nie bug — i ważne odkrycie o architekturze

M.A.R.S. z wąskimi specjalistami jest FUNDAMENTALNIE INNY niż klasyczny MoE:
- Klasyczny MoE: eksperci częściowo się pokrywają, ensemble (top-k) pomaga.
- M.A.R.S.: specjalizacja jest tak ostra, że łączenie ekspertów SZKODZI.

To realna własność do opisania w whitepaperze. Pokazuje, że nasza
specjalizacja jest "twarda" — każdy pod to wąski ekspert, nie częściowo
nakładający się member ensemble. Top-1 jest tu OPTYMALNY z natury.

Konsekwencja praktyczna: cała poprawa jakości musi iść przez ROUTER
(top-1 wybór), nie przez łączenie podów. Router jest jedyną dźwignią.
To upraszcza dalszy plan: inwestujemy w router, nie w strategie łączenia.

---

## Wartość tego (negatywnego) pomiaru

Gdybyśmy pominęli top-2 "bo pewnie pomoże jak w MoE", nie odkrylibyśmy,
że nasza architektura jest jakościowo inna od MoE. To dokładnie ten
przypadek, o którym mówił użytkownik: pominięty pomiar = pominięty wgląd.

Wynik negatywny, ale wiedza pozytywna:
1. Top-1 jest optymalny dla ostrej specjalizacji (nie marnujemy MAC na top-2)
2. Router to jedyna dźwignia jakości (jasny kierunek dalszych prac)
3. M.A.R.S. ≠ MoE — to odróżnia projekt (cenne dla narracji)

---

## Następny krok

Top-2 odrzucone. Pozostają dwa wyciśnięcia z planu:
- duże pody (pokazać throughput w reżimie gdzie M.A.R.S. wygrywa)
- potem A4 (catastrophic forgetting)

Dźwignia jakości na MNIST = mocniejszy router (ku sufitowi 99%),
ale to osobny wątek — najpierw duże pody i throughput.
