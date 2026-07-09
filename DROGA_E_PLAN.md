# Droga E — plan badawczy (cel: przełom, nie tylko paper)

Data: 2026-07-06
Status: PLAN (E1 gotowe do uruchomienia)
Kontekst: Seria D domknięta — sufit routingu udowodniony (D4+D5+D7), dźwignia
cech potwierdzona (D6) i efektywna (D6b). Whitepaper v0.2 zaktualizowany.

---

## 0. Szczera rama: czego wymaga "rewolucja"

Obecne wyniki są solidną nauką, ale przełom w ML wymaga trzech rzeczy, których
jeszcze NIE mamy — i Droga E jest zbudowana dokładnie pod nie:

1. **Trudniejszy benchmark.** MNIST/Fashion to poligon kalibracyjny. Żaden
   wynik na 28×28 nie zmieni pola. Pierwszy realny próg: CIFAR-10; dalej
   TinyImageNet. Teza modularności musi przeżyć zderzenie z danymi, gdzie
   backbone nie jest tani.
2. **Dżule, nie MAC.** "Rewolucja energetyczna" liczona w MAC to obietnica.
   Zmierzona w watach na GPU (nvidia-smi power sampling) — to dowód.
3. **Jedno zdanie, którego nikt inny nie ma.** Kandydat po serii D:
   *"Charakteryzujemy empirycznie sufit routingu i pokazujemy, że modularność
   skaluje się przez cechy + strukturę decyzji, nie przez algorytmy routingu —
   przy koszcie bliskim monolitu."* Droga E ma to zdanie udowodnić lub obalić.

Zasady bez zmian: 5 seedów, kryteria werdyktu z góry, wynik negatywny = wynik.

---

## E1 — Anatomia błędu (GOTOWE DO URUCHOMIENIA)

**Pytanie:** z czego DOKŁADNIE składa się luka router→oracle (6.11pp na CNN)?
Wiemy, że algorytmy jej nie zamkną (D4/D5/D7). Nie wiemy, KTÓRE próbki ją
tworzą i czy mają strukturę.

Plik: `src/run_E1_error_anatomy.py`. Mierzy per seed (5×, pełny CNN):
- **Dekompozycja błędu systemu** na 4 rozłączne klasy:
  A router✓+pod✓ (OK), B router✓+pod✗ ("pod miss"),
  C router✗+system✓ ("szczęśliwe odzyskanie" — zły pod i tak trafia),
  D router✗+system✗ (strata właściwa).
- **Zbiór odzyskiwalny**: oracle✓ & system✗ — to jest dokładnie luka; jego
  rozkład po klasach.
- **Macierz konfuzji routera** i top pary myłek (czy błąd jest skoncentrowany
  w kilku parach klas — Shirt/T-shirt/Pullover/Coat?).
- **Błąd vs confidence routera** (kwartyle): czy luka siedzi w niskiej
  pewności (→ mechanizmy selektywne mają target) czy jest rozsmarowana
  (→ nie mają).

**Kryterium ciekawości (nie werdyktu — E1 to diagnostyka):** jeśli ≥60% luki
koncentruje się w ≤4 parach klas → E2 (hierarchia) ma silny cel. Jeśli luka
rozsmarowana i wysokopewna → hierarchia też nie pomoże; wtedy dźwignią są
wyłącznie cechy i skala.

## E2 — Routing hierarchiczny (structure, nie algorithm)

**Hipoteza:** sufit z D4/D5/D7 dotyczy algorytmów NA tej samej decyzji
(10-way). Zmiana STRUKTURY decyzji — najpierw grupa (np. {upper-body},
{footwear}, {reszta}), potem klasa w grupie — to inna gra: grubszą decyzję
łatwiej podjąć na tych samych cechach, a wewnątrz grupy specjalista widzi
tylko trudne rozróżnienia.

- Grupy zdefiniowane Z DANYCH E1 (konfuzje), nie ręcznie.
- Falsyfikowalna: jeśli 2-poziomowy routing ≈ płaski (w szumie), to sufit
  reprezentacji obejmuje też strukturę decyzji → naprawdę zostają tylko cechy.
- Baseline: pełny CNN D6 (92.0% Fashion). Metryki + MAC jak zawsze.

## E3 — CIFAR-10 (próg wiarygodności)

**Pytanie:** czy CAŁA narracja (sufit routingu + dźwignia cech + efektywność
slim) replikuje się na danych, gdzie obraz jest kolorowy, tło szumi, a klasy
nie są wycentrowane?

- Konieczne: pody konwolucyjne albo mocniejszy wspólny trzon + bogatsze głowy.
- Plan minimalny: (a) v2+CNN backbone na CIFAR-10, kalibracja baseline'u,
  (b) replika D6b (slim sweep), (c) replika E2 jeśli dała sygnał.
- Ryzyko uczciwe: przy 32×32×3 backbone dominuje koszt jeszcze mocniej;
  oszczędność podowa może zniknąć → wynik i tak publikowalny (granica podejścia).

## E4 — Stos efektywności + DŻULE

**Cel:** jedna liczba, której nikt nie może podważyć: "system X osiąga Y%
accuracy przy Z dżulach na 10k próbek, monolit potrzebuje W dżuli".

- Złożyć: slim CNN S2 (D6b) + ternary pody (B8, 16× kompresja za darmo)
  + opcjonalnie routing TMU (Faza 1) w jednym systemie.
- Pomiar energii: nvidia-smi --query-gpu=power.draw w pętli podczas inferencji
  (10k próbek, batche, ~30 s okna pomiarowego, powtórzone 5×).
- Porównanie: monolityczny MLP/CNN o tej samej accuracy.
- To jest wątek "rewolucji energetycznej" w wersji dowodliwej.

## E5 — Sygnały spoza reprezentacji (opcja, po E1)

Sufit dotyczy informacji W cechach. Kanały spoza nich są legalne:
- cechy wejściowe niskiego poziomu bezpośrednio do routera (pixel statistics,
  edge density) obok cech backbone'u — tanie, omija wspólną reprezentację;
- augmentacyjny konsensus w inferencji (ta sama próbka ×2-3 tanie transformaty,
  zgodność routingu jako confidence) — koszt ×k, mierzyć uczciwie.
Uruchamiać tylko, jeśli E1 pokaże, że luka jest niskopewna (jest co łowić).

---

## Kolejność wykonania

1. **E1** (diagnostyka, ~35 min GPU) — natychmiast; wyniki sterują E2/E5.
2. **E4** (stos efektywności + dżule) — równolegle, niezależny od E1.
3. **E2** (hierarchia) — po E1, grupy z danych.
4. **E3** (CIFAR-10) — po E2; największy i najważniejszy krok.
5. Wyniki → `DROGA_E_NOTATKI.md`, whitepaper v0.3.
