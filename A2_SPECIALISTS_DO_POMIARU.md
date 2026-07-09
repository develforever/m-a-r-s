# M.A.R.S. — Droga A, krok A2: specjalizacja — DO POMIARU

Data: 2026-06-15

---

## Co zbudowano

`src/mars_specialists.py` — SpecialistSystem: ProtoRouter (z A1) + wąscy
specjaliści (NarrowPod, hidden=24 zamiast 64).

Kluczowa różnica vs stary system:
- Pody WĘŻSZE: hidden=24 (62% mniej MAC: 19,056 vs 50,816)
- Uczone z PRZEWAGĄ swoich danych (70/30) — specjalizacja z robustnością
- Router (ProtoRouter, ~94%) faktycznie decyduje o jakości

---

## Zweryfikowane lokalnie (smoke test)

- forward, routing, MAC działają
- pod specjalisty = 62% mniej MAC niż redundantny
- total routed MAC (router + 1 pod): 31,808

---

## DO POMIARU (Twój GPU)

```
.venv\Scripts\python.exe src\run_A2_specialists.py
```

Porównuje na PRAWDZIWYM MNIST:
- REDUNDANTNY: ProtoRouter + pełne pody (hidden=64, wszystkie dane)
- SPECJALIŚCI: ProtoRouter + wąskie pody (hidden=24, dane 70/30)

Mierzy: system accuracy, MAC, oszczędność.

---

## Co rozstrzyga ten pomiar

Pytanie: czy specjaliści dają porównywalną accuracy przy MNIEJSZYCH podach?

Scenariusze:
- **Specjalizacja działa** (acc spadek <2pp, pod 62% mniejszy):
  M.A.R.S. jest wreszcie prawdziwie modularny. Dodatkowa oszczędność MAC.
  To realny krok ku tezie projektu.
- **Specjalizacja kosztuje accuracy** (spadek >2pp): router 94% ścina
  jakość. Opcje ratunku: ProtoRouter 16D (96.5%) lub top-2 routing.
- Niejednoznaczne: strojenie own_ratio / pod_hidden.

Niezależnie od wyniku — to uczciwy test. Albo pokaże, że specjalizacja
daje przewagę, albo dokładnie zmierzy jej koszt. Oba wyniki są wartościowe.

---

## Po pomiarze A2

Jeśli specjalizacja działa → A3 (pełny pomiar: throughput specjalistów,
porównanie zbiorcze) i A4 (catastrophic forgetting na specjalistach +
naprawa dryfu encodera).

Jeśli kosztuje → wracamy do routera (mocniejszy / top-2) przed A3.
