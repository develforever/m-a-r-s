# M.A.R.S. — Wyciśnięcie: odchudzenie encodera (MNIST) — WYNIKI

Data: 2026-06-15
Status: ZMIERZONE na prawdziwym MNIST (GTX 1050 Ti).

---

## Wynik główny: encoder można zmiażdżyć 16×

| encoder hidden | router MAC | routing acc | system acc |
|---|---|---|---|
| 64 (oryginał) | 50,304 | 38.1% | 95.9% |
| 32 | 25,152 | 40.5% | 96.1% |
| 16 | 12,576 | 41.2% | 96.0% |
| 8 | 6,288 | 44.4% | 95.9% |
| **4** | **3,144** | **44.4%** | **95.9%** |

Encoder hidden=4 → **94% mniej MAC w routerze**, ZERO straty accuracy
(nawet lepsze routing acc). Zweryfikowane na realnym MNIST.

---

## Wpływ na całkowitą oszczędność MAC

| | router MAC | total MAC | oszczędność vs baseline |
|---|---|---|---|
| encoder h=64 | 50,304 | 101,120 | 56.9% |
| encoder h=4 | 3,144 | 53,960 | **77.0%** |

**Oszczędność MAC rośnie z 56.9% do 77.0% (+20 pkt proc.)** za jedną
zmianę hiperparametru, bez straty jakości. Realne wyciśnięcie.

---

## Niewygodna, ale ważna diagnoza

To, że encoder z 4 neuronami routuje tak samo jak z 64, jest DOWODEM,
że router w obecnym M.A.R.S. jest niemal bez znaczenia. Łącznie z audytem:

- routing accuracy ~40% (router myli się w 60% przypadków)
- system accuracy 96% mimo to
- encoder 4 neurony ≈ encoder 64 neurony

**Wniosek: system działa NIE dzięki routerowi, lecz pomimo niego.**
Pody są tak redundantne (każdy to pełny klasyfikator 10 klas — audyt A3),
że prawie nieważne, do którego router trafi. Router mógłby być losowy.

To nie porażka — to diagnoza wskazująca prawdziwy problem architektury:
obecny M.A.R.S. to "conditional computation z redundantnymi pełnymi
klasyfikatorami", a nie "router + wyspecjalizowani eksperci".

---

## Co z tego wynika dla wyciskania i Etapu C

Wyciśnięcie throughput (Etap B+) i MAC (encoder) DOMKNIĘTE. Mamy:
- throughput: 2.57× w reżimie dużych podów
- MAC: 77% oszczędności (po odchudzeniu encodera)
- accuracy: 96% (bez zmian)

Ale diagnoza pokazuje, że dalsze wyciskanie OBECNEJ architektury ma limit:
router jest atrapą, pody redundantne. Żeby pójść dalej, trzeba w Etapie C
rozstrzygnąć fundamentalne pytanie:

**Czy budujemy prawdziwą specjalizację (węższe pody, dane per region,
router który MA znaczenie), czy uczciwie opisujemy obecny system jako
conditional computation i na tym budujemy whitepaper?**

Obie drogi są legalne. Pierwsza to większa praca, ale mocniejsza teza
naukowa. Druga jest szybsza i wciąż obroni oszczędność MAC + throughput,
ale to skromniejszy wkład (bliski istniejącym pracom o sparsity).

REKOMENDACJA: zaktualizować mars_torch.py na encoder_hidden=4 (darmowy
zysk 77% MAC), potem podjąć decyzję o kierunku Etapu C.
