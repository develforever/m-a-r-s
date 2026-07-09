# M.A.R.S. — Stan projektu (Faza 1 ZAKOŃCZONA)

Data ostatniej sesji: 2026-06-15
Status: Faza 1 (PoC) KOMPLETNA — Etapy 0–4B + 3ext + 3C ukończone. Pełny raport: `RAPORT_FINAL.md`

---

## Gdzie jesteśmy — w jednym akapicie

Zbudowaliśmy działający, mierzalny szkielet dowodowy architektury M.A.R.S.
Każdy etap kończy się twardym pomiarem (operacje MAC + dokładność), a wyniki
są w pełni powtarzalne (deterministyczne, te same liczby na różnych maszynach).
To nie jest jeszcze "produkt", ale jest to coś znacznie cenniejszego niż pomysł:
**dowody liczbowe, które albo potwierdzają, albo obalają tezy z dokumentów.**

---

## Co udowodniliśmy (uczciwie)

### Etap 0 — Baseline
Klasyczna sieć z backpropagation uczy się XOR w 100%, koszt ~560 000 MAC.
To punkt odniesienia dla wszystkiego.

### Etap 1 — Uczenie lokalne (bez backpropagation)
Forward-Forward i Contrastive Hebbian uczą się XOR w 100% BEZ propagacji wstecz.
Ale: zużywają WIĘCEJ operacji niż backprop (5–10 mln MAC), nie mniej.
Wniosek: uczenie lokalne działa i jest fundamentem pod modularność, ale samo
w sobie nie daje oszczędności energii. To była ważna, uczciwa korekta oczekiwań.

### Etap 2 — Modularność i ochrona wiedzy (PIERWSZA REALNA PRZEWAGA)
Test catastrophic forgetting: ucz XOR, potem AND, zmierz ile XOR zostało.
- Baseline (wspólna sieć): retencja A spada do 50% (zapomina).
- M.A.R.S. (osobne pule neuronów + router): retencja A = 95%, przy nauce B = 100%.
Zysk +45 punktów procentowych. Bez poświęcania nowej wiedzy.
Uczciwa uwaga: to konsekwencja przydzielenia osobnych neuronów (kupujemy brak
zapominania za cenę pojemności) — znana własność systemów modularnych.

### Etap 3 — Engine Core (router) i usypianie
Router uczy się SAM rozpoznawać, którą kapsułę obudzić (100% trafności na
rozróżnialnych regionach). Usypianie reszty daje oszczędność energii — ale
KRZYWA, nie stała:
- N=2 kapsuły: -17% (routing DROŻSZY, narzut routera)
- N=10: +50%, N=50: +63%
Wniosek: obietnica "aktywuje się tylko 0.01% sieci" jest prawdziwa, ale
warunkowa — opłaca się dopiero przy WIELU kapsułach. Bonus: routing poprawia
też jakość (98% vs 66%), bo nie uśrednia nieprzystających specjalistów.

---

## Dwie inżynierskie lekcje warte zapamiętania

1. Sama ochrona wag (EWC) nie wystarcza przy współdzielonej pojemności małej
   sieci — chronione wagi zmieniały się 8× mniej, a wiedza i tak ginęła.
   Rozwiązaniem była modularność (osobne neurony), nie sprytniejsza ochrona.
2. JEDEN współdzielony parametr (bias wyjścia) potrafił zniszczyć całą
   modularność — przesuwał próg decyzyjny dla wszystkich zadań naraz.
   Router musi izolować pule kompletnie.

---

## Struktura projektu

```
m-a-r-s/
├── src/
│   ├── dataset.py          # XOR, AND (Etapy 0-2)
│   ├── dataset_regions.py  # rozróżnialne regiony (Etap 3)
│   ├── metrics.py          # licznik MAC + pomiar czasu
│   ├── baseline_mlp.py     # backprop baseline (Etap 0)
│   ├── capsule_ff.py       # Forward-Forward (Etap 1)
│   ├── capsule_chl.py      # Contrastive Hebbian (Etap 1)
│   ├── capsule_sleep.py    # wspólna sieć + EWC (Etap 2 baseline)
│   ├── capsule_modular.py  # modularna kapsuła + router (Etap 2)
│   ├── engine_core.py      # Router + Pods + usypianie (Etap 3)
│   ├── run_etap0.py ... run_etap3.py   # runnery, każdy zapisuje JSON
├── results/                # etap0_*.json ... etap3_*.json
└── requirements.txt        # numpy, psutil
```

Uruchomienie dowolnego etapu: `cd src && python run_etapN.py`

---

### Etap 3ext — Zimny start routera (NOWE)
Zmierzona odporność routera na błędy:
- Przy N≥5: routing opłacalny NAWET przy 50% błędach routera
- Przy N=3: breakeven przy ~33% błędów
- Online adaptation: 100% trafność po <100 epokach douczania nowego regionu
- Strategia retry lepsza niż top-2 (mniejszy średni koszt MAC)

### Etap 4 — Tekstury GPU jako pamięć (OBALONA w naiwnej formie)
Hipoteza "bilinear filtering = interpolacja semantyczna" jest FAŁSZYWA na rzadkiej siatce.

### Etap 4B — Kohonen SOM + Tekstura (WARUNKOWO POZYTYWNY)
- Kohonen SOM wymusza topologię → bilinear MA sens (MSE 0.85 → 0.027, poprawa 97%)
- Topologia zachowana: ratio inter/intra = 3.70
- Blur jako konsolidacja NIE działa (SOM już optymalny)
- Werdykt: tekstury z SOM = realny mechanizm, ale wymaga GPU (TMU)

### Etap 3C — SOM-Router: Kohonen jako Engine Core (WARUNKOWO POZYTYWNY)
- SOM-Router na GPU: 0 MAC (TMU fetch), 80% oszczędności total
- SOM-Router na CPU: droższy niż neural (256 vs 56 MAC)
- Dokładność: SOM 92.5% vs Neural 95% (delta -2.5pp, akceptowalne)
- Soft routing z bilinear: +2.5pp accuracy za darmo
- Sleep v2 (decay + Hebbian + prune): utrzymuje accuracy
- Online adaptation: 100% na nowym regionie po 50 epokach
- Werdykt: SOM zastępuje neural router NA GPU; hybrid dla edge

### Etap 3C GPU — Walidacja sprzętowa (CUDA, GTX 1050 Ti)
- Pure TMU fetch (grid_sample): **1.6x szybszy** niż neural router (matmul)
- Throughput: 41.9M samples/s na czystym texture fetch
- SOM-full (dist + TMU): wolniejszy — distance computation dominuje
- Wniosek: cache BMU pozycji lub natywne WebGPU samplowanie = klucz do pełnej przewagi
- Werdykt GPU: **POZYTYWNY** — hipoteza potwierdzona sprzętowo

---

## Co dalej (Faza 2 — skalowanie)

Pełny plan: `.windsurf/plans/mars-faza1-faza2-d6be7d.md`

1. **Etap 6 — Port na PyTorch + pomiar energii w dżulach** (nvidia-smi)
2. **Etap 7 — MNIST** (10 kapsuł, routing, retencja, cykl snu)
3. **Etap 8 — Benchmark + decyzja o produkcie** (forma: edge/cloud/framework)
4. **Etap 9 — CIFAR-10** (konwolucyjne kapsuły)
5. **Etap 10 — Ternary weights** (wagi [-1, 0, 1])

---

## Szczera myśl na koniec (do przemyślenia)

To, co zbudowaliśmy, jest dobrym ćwiczeniem inżynierskim i realnym proof of
concept — ale warto zachować trzeźwość co do skali. Modularność, routing
(Mixture of Experts), uczenie lokalne i ochrona przed zapominaniem to aktywne,
zaawansowane pola badawcze, nad którymi pracują duże zespoły. Twoja przewaga nie
musi polegać na "wymyśleniu wszystkiego od zera", lecz na konkretnym, dobrze
zmierzonym wkładzie w jeden wąski problem. Najmocniejsze w tym projekcie jest to,
że KAŻDY wynik jest zmierzony i uczciwie zinterpretowany — łącznie z tym, co nie
zadziałało. To jest dokładnie ta rzetelność, która buduje wiarygodność, czy to
przy grancie, czy przy rozmowie o pracę, czy przy publikacji open-source.

Sukces programistyczny "na lata" bierze się raczej z konsekwencji i rzetelności
niż z jednego genialnego pomysłu. Ten projekt pokazuje, że masz jedno i drugie.
Wróć, gdy będziesz gotowy — kod czeka, jest cały u Ciebie na dysku, działa
i jest powtarzalny.
