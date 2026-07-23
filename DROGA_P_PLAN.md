# Droga P — detekcja zatrucia payloadu na cechach semantycznych (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: DO ZATWIERDZENIA przez Roberta;
runy WYŁĄCZNIE u Roberta (GTX 1050 Ti), smoke → FULL. Branch: `droga-p`
(main nietykalny; nowe pliki, kod v1.0 NIETKNIĘTY).

## Punkt wyjścia (zmierzony)

I4 (v0.11): na LOSOWYM backbone (Fashion) żaden z dwóch pre-rejestrowanych
detektorów nie separuje ataków od uczciwych payloadów:
- D1 `rank_consistency` (Spearman: podobieństwa cechowe payloadu do klas
  własnych odbiorcy vs podobieństwa słowne kotwic) — brak separacji,
- D2 `canary_probe` (próbna adopcja na kopii; spadek acc klas własnych)
  — brak separacji.
Negatyw I4 jawnie wskazał kandydata: „detekcja semantyczna prawdopodobnie
wymaga cech pretrenowanych". Seria P testuje dokładnie to.

## Hipoteza

H-P: Na cechach semantycznych (zamrożony resnet18-ImageNet, konfiguracja
serii L) geometria międzyklasowa cech koreluje z geometrią słów, więc
D1 (a możliwie i D2) zyskuje separację clean vs atak, której nie ma na
cechach losowych. Mechanizm oczekiwany: centroid payloadu „ship" jest
cechowo bliżej „truck" niż „bird" — a payload podmieniony (swap) łamie
ten ranking względem kotwicy deklarowanej.

## Setup (parytet z L2/I4 — jedyna zmienna to backbone)

- Dane: Split-CIFAR-10n (5 zadań × 2 klasy, wejście znormalizowane).
- Odbiorca B: taski 0–3 (klasy 0–7). Nadawca A: task 4 (8=ship, 9=truck).
- Payloady klasy 8: `clean` / `swap` (payload 9 pod etykietą 8) /
  `noise` (statystyki na losowej mieszance cech — śmieć o realnych
  momentach). Payload 9 zawsze clean. Wzór: run_I4_untrusted.py.
- Dwa podłoża (P1a/P1b W JEDNYM runnerze, te same seedy):
  - **P1a pretrained**: `ReducedBackbone` (cache cech resnet18 512-d,
    losowa zamrożona projekcja 512→128 z seeda) — jak L1/L2,
  - **P1b random (kontrola)**: `CifarBackbone` na pikselach — jak J2b;
    kontrola przypisania efektu (replikacja negatywu I4 na CIFAR).
- Konfiguracja: sparse k=16, epochs 15, n_dream 5000, kanarek n=2000,
  GloVe-50d (parytet L), LR 0.001, 5 seedów (0–4).

## Kryteria werdyktu (Z GÓRY, przed pierwszym runem)

Definicja separacji (jak I4): PEŁNA SEPARACJA detektora = każdy seed
clean po właściwej stronie każdego seeda OBU ataków (min–max bez
przecięcia, 5/5).

1. **SUKCES MOCNY P**: ≥1 detektor ma pełną separację na pretrained
   (P1a) ORAZ ten sam detektor NIE ma jej na random (P1b)
   → detekcja możliwa, efekt przypisany reprezentacji. Headline v1.1.
2. **SUKCES SŁABY**: pełna separacja na pretrained tylko clean-vs-swap
   (noise nieodseparowany lub odwrotnie) → detekcja częściowa;
   raportować, który atak jest wykrywalny.
3. **NEGATYW**: brak separacji na pretrained → twierdzenie „semantyka
   cech NIE wystarcza; detekcja wymaga informacji spoza payloadu
   (np. drugiego świadka klasy)". Negatyw = wynik, domyka wątek
   detekcji wewnątrzpayloadowej.
4. **Anomalia kontrolna**: jeśli COŚ separuje na random (P1b) —
   efekt jest własnością danych/skali CIFAR, nie semantyki cech;
   raportować uczciwie, wniosek o reprezentacji unieważniony.

Obserwacje (nie werdykty): mapa szkody pary-vs-clean na obu podłożach
(acc8/acc9/acc_own — spójność z I4 na nowym podłożu); surowe wartości
D1/D2 per seed do wykresu.

## Warunkowe P2 (osobna pre-rejestracja PO werdykcie P1)

Tylko jeśli sukces (mocny/słaby): polityka kwarantanny — próg detektora
ustawiony na P1, mierzone: recall na atakach + koszt fałszywych alarmów
na uczciwych payloadach (nowe seedy nadawców). Nie kodować przed
werdyktem P1.

## Plik i koszt

- Runner: `src/run_P1_detect_pretrained.py` (nowy; importy z mars_cl_i4 /
  mars_cl_l / cifar_cl — zero zmian w istniejących plikach).
- Wynik: `results/P1_detect_pretrained.json` (smoke: `_smoke`).
- Szacunek FULL: ~25–40 min (P1a tanie po cache'u cech L; P1b liczy
  backbone na pikselach jak J2b; brak fazy naprawy — I4b już domknął).

## Zasady

5 seedów, pary per-seed, kryteria jw. zamrożone tą pre-rejestracją,
min per-seed raportowany, negatyw = wynik. Smoke (1 seed, 4 epoki,
n_dream 256) tylko jako sanity kształtów — bez wniosków.
