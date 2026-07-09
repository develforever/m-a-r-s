# M.A.R.S. — Podsumowanie planu wdrożenia MVP

## Fakty vs. koncepcje autorskie

**Realne i potwierdzone w literaturze:**
- Wagi trójstanowe `[-1, 0, 1]` (np. BitNet b1.58) — zamiana mnożenia na dodawanie to realna oszczędność energii.
- Spiking Neural Networks (SNN) i obliczenia neuromorficzne (Intel Loihi, IBM TrueNorth).
- Lokalne uczenie bez backpropagation (Hebbian, Forward-Forward Hintona, Contrastive Hebbian).
- Mixture of Experts / conditional computation ("usypianie sekcji") — działa na zwykłym sprzęcie.
- Memrystory (ReRAM) — realne, ale głównie laboratoryjne.

**Metafory/hipotezy do udowodnienia (NIE traktować jako fakt):**
- "Tekstura RGBA jako mapa myśli" z filtrowaniem dwuliniowym dającym "rozmycie skojarzeń" — brak dowodu, że bilinear filtering pikseli odpowiada sensownej interpolacji semantycznej.
- Liczby z dokumentów ("10× szybciej", "80% mniej energii", "95% mniej dżuli") — to cele/hipotezy, nie zmierzone wyniki. Sensem MVP jest je zmierzyć.

---

## Cel MVP

Udowodnić jedną falsyfikowalną tezę:

> Lokalna reguła uczenia (Hebbian) wewnątrz izolowanej kapsuły potrafi nauczyć się prostego zadania, a mechanizm "snu" (konsolidacja + zapominanie) poprawia retencję ważnej wiedzy przy mierzalnie niższym koszcie energetycznym niż baseline oparty na backpropagation.

---

## Etapy wdrożenia

### Etap 0 — Definicja metryk i baseline
Najważniejszy, najczęściej pomijany. Bez niego nie ma dowodu.

| Element | Co zrobić |
|---|---|
| Zadanie testowe | Jedno trywialne, deterministyczne: XOR, 3–4 wzorce binarne, lub MNIST 2 klasy. |
| Baseline | Klasyczna mini-sieć z backpropagation (PyTorch) na tym samym zadaniu. |
| Metryka energii | Operacje MAC, RAPL (Intel) / `nvidia-smi`, lub czas × pobór mocy. |
| Metryka jakości | Dokładność, czas zbieżności, retencja po nauce drugiego zadania. |

**Wyjście:** działający baseline + zapisany profil energetyczny i dokładność.

### Etap 1 — Pojedyncza kapsuła z lokalnym uczeniem
1. Jedna kapsuła (1–2 warstwy).
2. Lokalna reguła uczenia zamiast backpropagation. Rekomendacja: **Forward-Forward** lub **Contrastive Hebbian** (lepiej udokumentowane niż czysty Hebbian).
3. Nauka zadania z Etapu 0.
4. Pomiar: dokładność vs. baseline, koszt (operacje/energia).

**Język:** Python jest tu właściwy — najpierw dowód działania algorytmu, optymalizacja niskopoziomowa później.

**Wyjście:** kapsuła uczy się bez backpropagation + pomiar.

### Etap 2 — Cykl "snu": konsolidacja i zapominanie
Serce oryginalnego pomysłu ("im dłużej używasz, tym lepiej").
1. Pamięć jako zwykła macierz (jeszcze nie tekstura GPU). Pola: siła skojarzenia, świeżość, priorytet, plastyczność.
2. Cykl snu: decay świeżości, ochrona wysokiego priorytetu, "kostnienie" często używanych ścieżek. Baza: szkic C++ z dokumentów.
3. **Test krytyczny — catastrophic forgetting:** naucz A, potem B, zmierz retencję A ze "snem" vs. bez.

**Wyjście:** zmierzona różnica retencji — pierwszy realny dowód wartości.

### Etap 3 — Routing i wiele kapsuł (Engine Core)
1. 3–4 kapsuły, każda do innego pod-zadania.
2. Zamrożony router (ustalone reguły, jeszcze bez uczenia).
3. Pomiar oszczędności z "usypiania" — najłatwiejszy do udowodnienia zysk (sparsity / MoE, działa na zwykłym sprzęcie).
4. Badanie "zimnego startu": realny koszt błędnego routingu.

**Wyjście:** zmierzona oszczędność energii + charakterystyka kosztu błędów routera.

### Etap 4 (opcjonalny / badawczy) — Tekstury GPU
Wysokie ryzyko. Nie blokuje dowodu głównej tezy.
1. Przeniesienie macierzy pamięci do tekstury (WebGPU / compute shadery).
2. Operacje snu jako shadery: mipmapping, blur, erozja.
3. **Weryfikacja hipotezy:** czy bilinear filtering daje sensowne semantycznie interpolacje, czy tylko szum? Pytanie otwarte — bądź gotów obalić.
4. Pomiar: czy przeniesienie z ALU na TMU realnie obniża energię.

**Wyjście:** rozstrzygnięcie, czy "mapa myśli na teksturze" to mechanizm czy metafora.

### Etap 5 — Raport energetyczny i decyzja strategiczna
1. Tabela: M.A.R.S. vs baseline na każdym etapie (dokładność, energia, retencja, latencja).
2. Uczciwy raport, łącznie z tym, co nie zadziałało.
3. Dopiero z danymi: powrót do tematu grantów / open-source / patentu (z prawnikiem i doradcą IP).

---

## Stos technologiczny

| Etap | Narzędzie | Uzasadnienie |
|---|---|---|
| 0–3 (dowód algorytmu) | Python + NumPy/PyTorch | Szybka iteracja. |
| Pomiar energii | RAPL / `nvidia-smi` / `perf` + licznik MAC | Obiektywne, powtarzalne. |
| 4 (tekstury GPU) | WebGPU lub C++ + compute shadery | Tylko gdy algorytm udowodniony. |
| Produkcja (przyszłość) | Rust / C++ | Gdy jest co optymalizować. |

---

## Zasada przewodnia

Każdy etap kończy się **pomiarem**. Plan jest tak zbudowany, by udowodnić LUB obalić hipotezy z dokumentów, nie założyć ich prawdziwość. Wyniki negatywne też są wynikami i budują wiarygodność.