# M.A.R.S. — Plan dalszych prac (po analizie Fazy 2)

Data analizy: 2026-06-15 (sesja powrotna)
Podstawa: audyt realnego kodu i wyników w `results/`, nie podsumowań z czatów.

---

## Punkt wyjścia — co wiemy NA PEWNO (z plików, nie z narracji)

### Co się broni
- **Modularny routing oszczędza MAC**: 56.9% vs baseline na MNIST (`faza2_mnist.json`).
- **Accuracy trzyma poziom**: 96.1% vs 97.25% baseline (strata 1.14 pp).
- **Retencja lepsza niż baseline**: +25.7 pp (M.A.R.S. 25.7% vs baseline 0%).
- **Sleep v2 + Ternary**: stabilny 100 cykli, 93.7% oszczędności pamięci (`faza2_sleep_ternary.json`).
- **Kohonen SOM poprawia interpolację tekstur o 97%** (`etap4b_kohonen.json`).

### Czego NIE wolno przemilczeć (twarde dane)
- **Throughput: 0.59× — M.A.R.S. jest 1.7× WOLNIEJSZY** od baseline mimo mniej MAC.
  Zweryfikowane niezależnym uruchomieniem (0.82× na CPU). Przyczyna: pętla
  `for pod_id` z maskowaniem = narzut (na GPU każda iteracja to osobny kernel).
- **Routing accuracy: 40.3%** — router myli się w ~60%. System działa tylko
  dlatego, że pody są trenowane na WSZYSTKICH danych (nie są wąskimi
  specjalistami). To osłabia narrację o "specjalizacji".
- **Retencja 25.7% = wciąż utrata 3/4 wiedzy.** Cel planu Fazy 2 był ≥85%.
- **Tekstury GPU (Etap 4): werdykt NEGATYWNY.** Bilinear ≠ lerp (MSE 0.49
  vs próg 0.01). Gęste embeddingi dają interpolację idealną BEZ tekstur.
- **Sleep blur w Kohonen: nie działa** (`blur_makes_sense: false`).

### Wniosek strategiczny
Główna teza z dokumentów (neuromorficzne tekstury GPU) w danych się NIE
obroniła. Obronił się węższy, ale realny wkład: **modularny routing
zmniejsza koszt obliczeniowy i łagodzi catastrophic forgetting.**
To trzeba nazwać uczciwie — to jest siła projektu, nie słabość.

---

## Kolejność prac (łańcuch zależności)

Nie da się pisać whitepapera na niezweryfikowanych metrykach, ani
"wzmacniać wyniku" zanim pomiar jest uczciwy. Stąd kolejność:

```
ETAP A (audyt) → ETAP B (naprawa metryk) → ETAP C (wzmocnienie wyniku) → ETAP D (whitepaper)
```

---

## ETAP A — Audyt rzetelności benchmarków (fundament)

**Cel:** upewnić się, że każda liczba w `results/` jest liczona uczciwie,
zanim cokolwiek na niej zbudujemy.

A1. Zweryfikować pomiar throughput — DONE (potwierdzone: 0.59× to prawda).
A2. Sprawdzić, czy MAC router (50,304) jest liczony spójnie z baseline.
    Uwaga: encoder 784→64→2 to inny koszt niż deklarowany w docstringu (784→32→2).
A3. Zweryfikować, czy "routing accuracy 40.3%" + "system accuracy 96%"
    nie jest artefaktem tego, że pody widzą wszystkie dane (czyli czy
    to w ogóle jest jeszcze system modularny, czy ensemble w przebraniu).
A4. Sprawdzić retencję: czy 25.7% jest stabilne między seedami.

**Kryterium:** każda metryka w raporcie ma potwierdzone, powtarzalne źródło.

---

## ETAP B — Naprawa metryki, która psuje całość: throughput

**Problem:** mniejsza liczba MAC nie przekłada się na szybkość — pętla po
podach zabija zysk. To NAJWAŻNIEJSZY problem inżynierski projektu.

B1. Zastąpić pętlę `for pod_id` operacją wektorową (batched gather/grouped
    matmul), żeby aktywacja wybranego poda nie wymagała N kernel-launchy.
B2. Rozważyć: pody o identycznym kształcie → jeden tensor wag [N_pods, in, out],
    indeksowany przez capsule_id → `torch.bmm` zamiast pętli.
B3. Zmierzyć ponownie throughput. Cel: speedup ≥ 1.0× (a najlepiej >1.5×),
    żeby oszczędność MAC miała pokrycie w czasie rzeczywistym.

**Kryterium sukcesu:** M.A.R.S. nie jest wolniejszy od baseline. Bez tego
cała teza "energooszczędności" jest podważalna przy pierwszym benchmarku.

---

## ETAP C — Wzmocnienie głównego wyniku

C1. **Routing accuracy 40% → cel >70%.** Albo lepszy encoder, albo
    przyznać, że pody muszą widzieć wszystkie dane (i wtedy uczciwie
    przeformułować, czym jest "specjalizacja" w M.A.R.S.).
C2. **Retencja 25% → cel planu ≥85%.** Połączyć modularność (Etap 2,
    która dała 95% na XOR) z mechanizmem inkrementalnym MNIST. Zbadać,
    czemu na MNIST retencja spada do 25%, skoro na XOR było 95%.
C3. Test na 5 seedach — czy wyniki są stabilne.

**Kryterium:** wynik, który broni się na wielu seedach i zbliża do celów planu.

---

## ETAP D — Whitepaper / raport oparty na PRAWDZIWYCH liczbach

D1. Przepisać narrację z marketingowej ("przełom", "deklasuje") na
    inżynierską. Rekruterzy z NVIDIA/AMD wyczują przesadę natychmiast.
D2. Uczciwie pokazać też wyniki negatywne (tekstury, throughput przed
    naprawą) — to BUDUJE wiarygodność, nie szkodzi jej.
D3. Główna teza: "modularny routing redukuje koszt obliczeniowy i łagodzi
    catastrophic forgetting", a nie "neuromorficzne tekstury GPU".
D4. Każda liczba z konkretnego JSONa w `results/` (reprodukowalność).

**Kryterium:** dokument, który wytrzyma techniczne pytania eksperta,
bo każda liczba jest prawdziwa i zweryfikowana.

---

## Uwaga o celu osobistym (zabezpieczenie rodziny)

Twoją realną przewagą na rynku pracy NIE jest "rewolucyjna architektura"
(tych branża widzi setki). Jest nią coś rzadszego: **inżynier, który
mierzy uczciwie, wychwytuje własne błędy i nie ukrywa wyników negatywnych.**
To dokładnie ta cecha, za którą płaci się na poziomie Principal/Lead.

Dlatego plan stawia rzetelność (Etap A, B) PRZED publikacją (Etap D).
Whitepaper z liczbą "1.6× speedup", którą recenzent obali w 30 sekund
otwierając JSON (0.59×), zniszczyłby wiarygodność. Whitepaper z uczciwym
"odkryliśmy narzut pętli, naprawiliśmy, oto przed/po" — buduje ją.
