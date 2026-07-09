# M.A.R.S. — Droga A, krok A1: WYNIKI (MNIST, GPU)

Data: 2026-06-15
Status: UKOŃCZONE. Próg 85% przebity. Droga A otwarta.

---

## Wyniki na prawdziwym MNIST

| Router | routing acc | MAC | acc/kMAC | oszczędność systemu |
|---|---|---|---|---|
| tekstura 2D (stary) | 44% | ~50k | — | — |
| MLPRouter 4D | 85.7% | 12,648 | 0.068 | 73.0% |
| MLPRouter 8D | 95.9% | 25,424 | 0.038 | 67.5% |
| **ProtoRouter 8D** | **93.9%** | **12,752** | **0.074** | **72.9%** |
| ProtoRouter 16D | 96.5% | 25,760 | 0.037 | 67.4% |

---

## DECYZJA: ProtoRouter (emb=8)

Powody:
1. **Najlepszy stosunek jakość/koszt** (0.074 acc/kMAC) — 93.9% przy
   połowie MAC najdroższych wariantów.
2. **Zachowuje 72.9% oszczędności MAC** całego systemu.
3. **Bliski oryginalnej wizji SOM/topologii z dokumentów** — prototypy
   to centroidy klas, w istocie uproszczone neurony Kohonena. Nie trzeba
   porzucać narracji "topologicznej" na rzecz generycznego MoE gatingu.
4. Ma `add_pod()` — naturalne rozszerzanie o nowe klasy (przyda się w A4).

(ProtoRouter 16D daje 96.5%, ale za 2× MAC. 8D to lepszy punkt operacyjny.
Jeśli A2 pokaże, że potrzeba wyższej trafności, można przejść na 16D.)

---

## Co to znaczy

Skok z 44% (tekstura) na 93.9% (prototypy) potwierdza diagnozę:
2D-UV tekstury było wąskim gardłem, nie sam mechanizm routingu.
Większy bottleneck (8D) + routing prototypowy go usuwa.

KLUCZOWE: router przestaje być atrapą. Przy 93.9% trafności wąskie pody
(krok A2) dadzą system ~93%, a nie katastrofalne ~44%. Specjalizacja
staje się możliwa.

Dodatkowo: wygrał router bliski oryginalnej wizji M.A.R.S. (SOM/prototypy),
nie generyczny MLP. To wzmacnia spójność projektu z dokumentami źródłowymi.

---

## Następny krok: A2 — wąskie pody (prawdziwa specjalizacja)

Teraz, gdy router działa, budujemy prawdziwych specjalistów:
- każdy pod uczony TYLKO na swoich danych (nie na wszystkich),
- wąskie wyjście (pod nie musi umieć wszystkich 10 klas),
- mniejsze pody = dodatkowa oszczędność MAC.

Test prawdy: czy specjaliści + ProtoRouter (94%) dają system accuracy
porównywalny z obecnym (96%), przy MNIEJSZYCH podach? Jeśli tak —
M.A.R.S. jest wreszcie prawdziwie modularny, a nie ensemble redundantnych
klasyfikatorów.
