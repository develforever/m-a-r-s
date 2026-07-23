# Droga P1c — brama strukturalna D1 i prawo dystansu semantycznego (pre-rejestracja)

Data pre-rejestracji: 2026-07-23 (po werdykcie P1, PRZED jakimkolwiek runem
P1c). Status: DO ZATWIERDZENIA; runy WYŁĄCZNIE u Roberta. Branch: `droga-p`
(kontynuacja; nowe pliki, istniejące nietknięte). Kontekst: DROGA_P_NOTATKI.md
— P1 = podwójny NEGATYW; P1c konwertuje trzy obserwacje POST-HOC w twierdzenia
na ŚWIEŻYCH seedach, z progami ustalonymi tutaj.

## Świeże seedy i higiena

Seedy 5–9 (P1 używało 0–4 — obserwacje nie mogą testować samych siebie).
Konfiguracja i kod detektorów: identyczne z P1 (sparse k=16, epochs 15,
GloVe-50d, LR 0.001; D1/D2 z mars_cl_i4 bez zmian). Oba podłoża jak w P1:
pretrained (ReducedBackbone) i random (CifarBackbone). Bez pełnej adopcji
i mapy szkody (zmierzona w P1) — tylko detekcja.

## Setup wariantowy (oś dystansu)

Deklarowana klasa zawsze 8 (ship). Donor d ∈ {0, 7, 4} wg CosINUSÓW KOTWIC
50d zmierzonych metodą load_word_vectors (zapisane PRZED runem):

| Wariant | Donor d | cos(kotwica ship, kotwica d) |
|---|---|---|
| swap_close | 0 airplane | +0.775 |
| swap_mid | 7 horse | +0.487 |
| swap_far | 4 deer | +0.139 |

(Referencja P1: truck +0.615, niewykrywalny.) Dla donora d: B uczy 4 taski
z klas {0..9}\{8,d} (pary rosnąco); A uczy jeden task {d,8}; paczka
adopcyjna = {d,8} z payloadami {d: clean_d, 8: wariant}. Warianty payloadu
8: clean (stats 8 od A) / swap_d (stats d od A pod etykietą 8 — struktura
ataku identyczna z P1/I4, zmienia się TYLKO dystans pary) / noise (jak P1).
Mierzone per wariant: D1 rank_consistency (przed adopcją), D2 canary_probe
(adopcja próbna na kopii, n=2000).

## Kryteria werdyktu (Z GÓRY)

### P1c-a — brama strukturalna D1 (próg θ = 0.45)

θ = 0.45: środek zmierzonej w P1 luki (max noise 0.286/0.214 vs min clean
0.667/0.762), zamrożony przed runem. Test na WSZYSTKICH payloadach clean
(3 donory × 5 seedów = 15/podłoże) i noise (15/podłoże):

- **SUKCES**: 100% clean > θ ORAZ 100% noise < θ na OBU podłożach
  (60/60 poprawnych) → twierdzenie: „D1 z progiem jest bramą odrzucającą
  payloady bez struktury klasowej, niezależnie od reprezentacji".
- **CZĘŚCIOWY**: warunek spełniony na dokładnie jednym podłożu.
- **NEGATYW**: na żadnym. Jawnie: brama NIE twierdzi, że wykrywa podmiany
  (swap przechodzi przez bramę — raportować liczbowo jako kontekst).

### P1c-b — wykrywalność swap vs dystans semantyczny (D1)

Separacja = pełna, per podłoże: min(clean) > max(swap_d) w 5/5 seedów
(kierunek z P1: swap obniża D1 dopiero, gdy donor łamie ranking).

- **SUKCES MOCNY**: na OBU podłożach — separacja clean-vs-swap_far ORAZ
  brak separacji clean-vs-swap_close → prawo graniczne: „wykrywalność
  podmiany rośnie z dystansem semantycznym pary; podmiana klas bliskich
  jest niewykrywalna z konstrukcji (payload donora spełnia ranking
  deklarowanej kotwicy)".
- **SUKCES SŁABY**: powyższe na dokładnie jednym podłożu, LUB monotonia
  median D1 (close > mid > far) w 5/5 seedów bez pełnej separacji far.
- **NEGATYW**: swap_far nieodseparowany na obu podłożach i brak monotonii
  → dystans semantyczny nie jest osią wykrywalności (też domyka).
- Obserwacja wspierająca (nie werdykt): mediany D1 per wariant + Spearman
  (D1, cos kotwic) per seed.

### P1c-c — znak D2 (METRYKA WTÓRNA, ranga obserwacji)

Replikacja kierunku z P1 na świeżych seedach: median D2(swap_d) <
median D2(clean) — raportować liczbę seedów spełniających per donor
i podłoże. BEZ rangi twierdzenia głównego niezależnie od wyniku
(kandydat na pre-rejestrację pełną dopiero, jeśli 5/5 wszędzie).

## Interpretacja końcowa (zapisana z góry)

Niezależnie od werdyktów a/b polityka protokołu po P1c jest kompletna
i w całości zmierzona: adopcje paczkami (I4) + naprawa zapomnij-i-adoptuj
(I4b) + [jeśli a+] brama strukturalna na wejściu + [jeśli b+] jawna
granica: podmiany bliskich klas niewykrywalne wewnątrzpayloadowo —
pokrywa je wyłącznie naprawa. Sukces b = negatywy I4/P1 wyjaśnione
mechanizmem, nie epizodem.

## Plik, koszt, wynik

- Runner: `src/run_P1c_gate_distance.py` (nowy; zero zmian w istniejących).
- Wynik: `results/P1c_gate_distance.json` (smoke: `_smoke`).
- Pętla: 2 podłoża × 3 donory × 5 seedów (B: 4 taski, A: 1 task,
  3×(D1+D2), bez pełnej adopcji). Szacunek FULL: ~10–20 min.
- Smoke: 1 seed (5), 4 epoki, n_probe 128 — sanity kształtów, bez wniosków.
