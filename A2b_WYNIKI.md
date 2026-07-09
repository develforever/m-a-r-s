# M.A.R.S. — Droga A, krok A2b: WYNIKI (MNIST, GPU)

Data: 2026-06-15
Status: ACCURACY ODZYSKANE Z NADDATKIEM. Cel A2b osiągnięty.

---

## Wyniki

| Scenariusz | routing | system | total MAC | oszczędność |
|---|---|---|---|---|
| Router 8D (obecny) | 95.4% | 95.4% | 31,808 | 86.5% |
| **Router 16D** | **96.7%** | **96.7%** | 44,816 | 80.9% |
| ORACLE (idealny) | 100% | **99.0%** | 19,056 | — |
| stary redundantny | — | 96.2% | 63,568 | 72.9% |

---

## Dwa kluczowe wnioski

### 1. Accuracy odzyskane Z NADDATKIEM
Router 16D + specjaliści = 96.7%, czyli PRZEBIŁ stary redundantny (96.2%)
przy podach 62% mniejszych. Nie "dorównaliśmy" — przeskoczyliśmy.

### 2. ORACLE 99.0% — odkrycie, nie tylko potwierdzenie
Gdyby router był idealny, wąscy specjaliści osiągnęliby 99%. To znaczy:
- Specjaliści (hidden=24, 62% mniejsze) są LEPSI niż całe redundantne pody.
- Prawdziwy sufit systemu to 99%, nie 96%.
- Cała dalsza poprawa = poprawa ROUTERA (luka 2.3pp do sufitu).

### 3. Idealne przełożenie routing → system
Na MNIST: routing 95.4% → system 95.4%, routing 96.7% → system 96.7%.
Każdy 1pp routingu = 1pp systemu. To CZYSTY dowód prawdziwej modularności:
system jest dokładnie tak dobry, jak jego router.

---

## Wybór punktu operacyjnego

| | system acc | oszczędność MAC |
|---|---|---|
| 8D (oszczędny) | 95.4% | 86.5% |
| 16D (dokładny) | 96.7% | 80.9% |

Oba są dobre. 16D bije stary system na obu frontach (lepsza accuracy +
wciąż większa oszczędność niż 72.9% starego). Rekomendacja: 16D jako
domyślny (najlepszy z obu światów), 8D gdy priorytetem jest maks. oszczędność.

---

## Co zostało do wyciśnięcia (decyzja w A3)

Luka 2.3pp między routerem 16D (96.7%) a sufitem specjalistów (99%).
Opcje domknięcia luki:
- top-2 routing (sprawdź 2 najbliższe pody) — tani trik, może dużo dać
- większy emb (32D) — rośnie MAC
- głębszy encoder routera

ALE: 96.7% przy 80.9% oszczędności to JUŻ mocny wynik. Czy gonić ostatnie
2.3pp — to decyzja kosztowa, którą podejmiemy w A3 po pełnym pomiarze
throughput.

---

## Status Drogi A
- A1 ✅ lepszy router (ProtoRouter)
- A2 ✅ specjalizacja działa (62% mniejsze pody)
- A2b ✅ accuracy odzyskane (96.7% > 96.2% redundantnego)
- A3 ⬜ pełny pomiar throughput + decyzja o gonieniu sufitu 99%
- A4 ⬜ catastrophic forgetting na specjalistach
