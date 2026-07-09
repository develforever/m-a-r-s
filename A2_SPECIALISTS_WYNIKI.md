# M.A.R.S. — Droga A, krok A2: WYNIKI (MNIST, GPU)

Data: 2026-06-15
Status: SPECJALIZACJA DZIAŁA. Cel Drogi A osiągnięty.

---

## Wyniki na prawdziwym MNIST

| System | acc | pod MAC | total MAC | oszczędność |
|---|---|---|---|---|
| Redundantny (stary) | 96.2% | 50,816 | 63,568 | 72.9% |
| **Specjaliści (nowy)** | **94.3%** | **19,056** | **31,808** | **86.5%** |

Pod specjalisty 62% mniejszy. Accuracy −1.8 pp. Oszczędność +13.6 pp.

---

## NAJWAŻNIEJSZE: router przestał być atrapą

Stary system (audyt): routing 40% → system 96%. Router NIEWAŻNY.
Nowy system: routing 94.2% → system 94.3%. **Router = serce systemu.**

To zrównanie (94.2% ≈ 94.3%) jest dowodem prawdziwej modularności.
Specjaliści naprawdę polegają na routerze — to nie ensemble redundantnych
kopii, lecz wyspecjalizowane jednostki. CEL DROGI A OSIĄGNIĘTY.

---

## Niuans: router jest teraz sufitem

To samo zrównanie znaczy: system nie może być lepszy niż router.
Strata 1.8 pp = cena tego, że router ma 94%, nie 96%+.
Gdy router się myli, specjalista dostaje obcą klasę, której słabo zna.

To nie problem — to jasna mapa poprawy. Router = dźwignia jakości.

---

## Opcja A2b: mocniejszy router (do zmierzenia)

Z A1: ProtoRouter 16D = 96.5% (vs 8D = 93.9%).

| Konfiguracja | sufit (router) | total MAC | oszczędność |
|---|---|---|---|
| Router 8D + specjaliści (obecny) | ~94% | 31,808 | 86.5% |
| Router 16D + specjaliści | ~96% | 44,816 | 80.9% |

Hipoteza: router 16D podniesie system do ~96% — DORÓWNA redundantnemu,
przy wciąż 62% mniejszych podach i 80.9% oszczędności MAC.
Koszt: +13k MAC w routerze, oszczędność spada 86.5%→80.9%.

To realny wybór punktu operacyjnego:
- 8D: maks. oszczędność (86.5%), accuracy 94.3%
- 16D: maks. accuracy (~96%), oszczędność 80.9%

---

## Status Drogi A

- A1 ✅ lepszy router (ProtoRouter, 94-96% vs tekstura 44%)
- A2 ✅ specjalizacja działa (62% mniejsze pody, −1.8pp accuracy)
- A2b ⬜ (opcja) mocniejszy router → odzyskać 1.8pp
- A3 ⬜ pełny pomiar throughput specjalistów + porównanie zbiorcze
- A4 ⬜ catastrophic forgetting na specjalistach + naprawa dryfu encodera

M.A.R.S. jest teraz PRAWDZIWIE MODULARNY. Router decyduje, specjaliści
są wąscy i wyspecjalizowani. To realizacja wizji "Specialist Pods"
z dokumentów — zmierzona, nie deklarowana.
