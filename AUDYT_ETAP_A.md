# M.A.R.S. — Audyt rzetelności (Etap A) — WYNIKI

Data: 2026-06-15 (sesja powrotna)
Metoda: weryfikacja realnego kodu + niezależne uruchomienia, nie podsumowania z czatów.

---

## A1 — Throughput 0.59× (M.A.R.S. wolniejszy) — POTWIERDZONE ✓

Niezależne uruchomienie odtworzyło efekt: pętla `for pod_id` z maskowaniem
spowalnia system mimo mniejszej liczby MAC.
- Mój pomiar (CPU): 0.82× (M.A.R.S. wolniejszy)
- Plik użytkownika (GPU): 0.59× (jeszcze gorzej — każda iteracja pętli to
  osobny kernel launch na GPU)
**Werdykt: metryka prawdziwa. To realny problem do naprawy (Etap B).**

---

## A2 — Spójność liczenia MAC — ZWERYFIKOWANE ✓

Liczby w `faza2_mnist.json` zgadzają się z kodem:
- som_router_mac = 784×64 + 64×2 = 50,304 ✓
- pod_mac = 784×64 + 64×10 = 50,816 ✓
- mars_routed = router + 1 pod = 101,120 ✓
- baseline = 784×256 + 256×128 + 128×10 = 234,752 ✓
**Werdykt: MAC liczony uczciwie i spójnie. Oszczędność 56.9% jest prawdziwa
NA POZIOMIE MAC (ale nie przekłada się na czas — patrz A1).**

---

## A3 — Czy to system modularny, czy ensemble? — KRYTYCZNE ODKRYCIE ⚠

Test rozstrzygający: pojedynczy pod, trenowany na wszystkich danych
(dokładnie jak w `train_system`), SAM osiąga 100% accuracy.

To znaczy: **każdy z 10 podów jest pełnym, niezależnym klasyfikatorem
10 klas — 10 niemal identycznych kopii.** Potwierdza to kod:
- `CapsulePod(n_in, pod_hidden, n_out=n_pods)` — każdy pod ma 10 wyjść
- komentarz w `train_system`: "Trenuj każdy pod na WSZYSTKICH danych"

**Konsekwencja dla narracji:**
- Routing accuracy 40.3% "nie szkodzi", bo każdy pod i tak umie wszystko.
- Ale to oznacza, że obecny M.A.R.S. NIE jest prawdziwie modularny —
  pody nie są wyspecjalizowanymi ekspertami, tylko redundantnymi kopiami.
- Oszczędność MAC jest realna (aktywujemy 1 z 10), ale "specjalizacja"
  i "specialist pods" z dokumentów NIE są tym, co faktycznie działa.

**To nie jest porażka — to wymaga uczciwego przeformułowania.** Albo:
  (a) pody powinny być węższe (mniej wyjść, dane tylko ze swojego regionu),
      co testowałoby prawdziwą specjalizację, albo
  (b) przyznać, że M.A.R.S. to "conditional computation z redundantnymi
      klasyfikatorami", co wciąż oszczędza MAC, ale to inna teza.

---

## A4 — Retencja 25.7% — ŹRÓDŁO ZDIAGNOZOWANE ⚠

Niska retencja (cel planu: ≥85%) NIE wynika z wad podów, lecz z DRYFU ENCODERA.

Mechanizm w `train_incremental`:
1. Pody 0-4 są zamrażane → ich wiedza jest nienaruszona ✓
2. ALE encoder jest fine-tunowany na danych B (lr=0.001) → mapowanie UV
   starych cyfr się PRZESUWA
3. Router (encoder→UV→label_map) zaczyna kierować stare cyfry 0-4 do
   nowych podów 5-9, które ich nie znają
4. Retencja 25.7% = % przypadków, gdzie router PRZYPADKIEM trafia dobrze

**Dlaczego na XOR (Etap 2) było 95%, a tu 25%:** w Etapie 2 router wybierał
po ID zadania, a pule miały własne biasy — encoder się nie przesuwał.
Tutaj encoder jest współdzielony i dryfuje.

**Naprawa (Etap C):** zamrozić encoder przy inkrementacji (lub nie
fine-tunować go wcale), żeby mapowanie starych regionów UV nie dryfowało.

---

## Podsumowanie audytu

| Metryka | Status | Wniosek |
|---|---|---|
| Throughput 0.59× | ✓ prawdziwa | Realny problem, naprawialny (Etap B) |
| MAC saving 56.9% | ✓ prawdziwa | Ale tylko w MAC, nie w czasie |
| Accuracy 96.1% | ✓ prawdziwa | Solidny wynik |
| Modularność | ⚠ pozorna | Pody to redundantne kopie, nie eksperci |
| Retencja 25.7% | ⚠ zaniżona | Wina dryfu encodera, naprawialna |

**Dobra wiadomość:** żaden wynik nie jest sfałszowany. Kod liczy uczciwie.
**Do naprawy:** dwa realne, zrozumiałe problemy (pętla podów, dryf encodera)
oraz jedna kwestia narracji (czym naprawdę jest "modularność" w M.A.R.S.).

To jest dokładnie ten rodzaj rzetelnej diagnozy, który czyni projekt
wiarygodnym. Naprawiamy → mierzymy → opisujemy uczciwie.
