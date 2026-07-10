# M.A.R.S. — Plan generalny po v0.4: K → I → L

Data: 2026-07-11. Status: PROPOZYCJA (do zatwierdzenia przez Roberta;
pre-rejestracje szczegółowe powstają per droga, PRZED runami).
Zastępuje PLAN_DALSZYCH_PRAC.md (2026-06-15, historyczny — sprzed serii D–J).

## Cel końcowy (definicja „rewolucji")

Kolektyw N agentów uczy się razem BEZ wymiany danych, gradientów i wag —
wymieniając wyłącznie sen: statystyki cech klasy (k×(3D+1) liczb,
~12–24 KB) + nazwę klasy (kotwica słowna). Twierdzenie docelowe:
**wynik kolektywu ≈ scentralizowany sufit (g1_all), zero przesłanych
próbek**. Poziom absolutny podnosi potem wymiana backbone'u (fork L)
bez zmiany mechanizmu i protokołu.

## Zasada warstw — dlaczego przejścia są bezpieczne

Stos systemu: (1) zamrożony backbone → (2) pamięć klasy: statystyki
sparse + sen → (3) projekcja + kotwice słów. Kontrakt między warstwami
= format statystyk per klasa (`FeatureStatsKSparse`, J3) + nazwa klasy.

- **K** żyłuje (2)×(3) w pojedynczym agencie (złożenia zmierzonych dźwigni).
- **I** dokłada protokół wymiany NAD (2) — nie dotyka żadnej warstwy.
- **L** podmienia (1) POD spodem — (2), (3) i protokół I bez zmian
  („representation-agnostic" z WHITEPAPER przechodzi z deklaracji w test).

Każdy etap zmienia JEDNĄ warstwę, więc wyniki poprzednich etapów
pozostają ważne i porównywalne; kolejną drogę dokłada się bez ruszania
poprzedniej.

## Metodologia (obowiązuje od K, nie wstecz)

Bez zmian: 5 seedów (0–4), pary per-seed, próg szumu std(baza)+std(wariant),
SYGNAL wymaga min per-seed > 0, wynik negatywny = wynik, branch per droga,
main nietykalny, merge po komplecie werdyktów, runy wyłącznie u Roberta.

NOWE [DECYZJA ROBERTA przed pierwszym runem K]: dodatkowa klasa werdyktu
**SYGNAL-parowy**: wszystkie pary jednego znaku ORAZ |śr. delt| > 2×std(delt).
Raportowany OBOK progu std+std (nie zamiast) — stare werdykty zostają
porównywalne. Motywacja: J3 (10/10 par, formalnie SZUM) vs J4 (pary
mieszane, czysty null) niosą różną informację, którą dotychczasowa
konwencja zlewa.

Sanity na wejściu każdego etapu: reprodukcja znanego wariantu do 0.00pp
(precedens: J1/J3/J4 — cztery czyste reprodukcje).

## Etap K — wyżyłowanie obecnej drogi (branch `droga-k`)

Cel: zamknąć pytanie „ile mechanizm może wziąć na losowych cechach"
liczbami, nie deklaracją. Trzy runnery, wszystkie tanie (rząd 100–1500 s).

- **K0 — brakujący sufit CIFAR (diagnostyka):** proj_train="all" na
  cechach zamrożonego losowego backbone'u CIFAR (analog g1_all z G1/J4;
  NIGDY nie zmierzony — F4/J2 raportują tylko joint 70.24, który jest
  trenowalny, więc NIE jest sufitem mechanizmu). Rozstrzyga, ile z luki
  37.51→70.24 jest mechanizmowe (do wzięcia w K/I), a ile reprezentacyjne
  (do wzięcia dopiero w L). Interpretacja z góry: mała luka 37.51→K0
  = mechanizm domknięty na CIFAR; duża = są jeszcze dźwignie.
- **K1 — złożenie dźwigni (dawny J4b):** sparse_k16 × GloVe 300d,
  Fashion i CIFAR (300d pobrane po J4). Pary vs sparse_k16 × 50d.
  Cel kierunkowy: Fashion w stronę nowego sufitu 81.16.
- **K2 — OWM × sen sparse, tam gdzie boli:** H1 testował OWM ze snem
  diagonalnym sprzed J — SYGNAL+ na MNIST (+5.0pp), nigdy na CIFAR
  (forgetting 32.7pp), a eliminacja dryfu dotyczyła tylko Fashion.
  Warianty: MNIST i CIFAR (OWM+sparse_k16 vs sparse_k16); Fashion tylko
  jako kontrola eliminacji (oczekiwany SZUM — potwierdzenie H1 przy
  nowym śnie).

Kryterium wyjścia z K: (a) oba sufity zmierzone (Fashion 81.16, CIFAR K0),
(b) każda dźwignia rozstrzygnięta (SYGNAL/SZUM, w tym parowy),
(c) headline v0.5 zapisany. Jeśli wszystko SZUM — mechanizm ogłaszamy
domkniętym; to też jest „maksimum z obecnej drogi".

## Etap I — kolektywna wymiana snów (branch `droga-i`) — kandydat na rewolucję

Wejście: domyślna konfiguracja zwycięska z K (sen sparse × kotwice 50d/300d
wg K1). Fundament zmierzony: stacjonarność (F1d, J2 SYGNAL+) +
wystarczalność statystyk do wyśnienia klasy (J2b SYGNAL+). Protokół nie
zależy od backbone'u → przeżyje fork L bez zmian.

- **I1 — przeszczep:** agenci A i B, wspólny seed backbone'u
  (synchronizacja darmowa). A uczy się klas niewidzianych przez B →
  wysyła statystyki + nazwy → B śni, doucza projekcję. Metryka: ACC klasy
  przeszczepionej u B vs ta sama klasa uczona lokalnie (te same dane).
  Próg sukcesu [DECYZJA ROBERTA]: propozycja dwustopniowa — sukces mocny:
  strata przeszczepu < próg szumu (równoważność); sukces słaby: < 3pp.
- **I1b — moment wymiany:** na końcu sekwencji vs w trakcie (jak
  forgetting traktuje klasy przeszczepione względem uczonych).
- **I2 — fuzja:** ta sama klasa, rozłączne połówki danych u A i B; fuzja
  statystyk vs każda osobno vs klasa na całości. Specyfikacja fuzji do
  DROGA_I_PLAN (propozycja: suma ważona liczebnością per centroid,
  ewent. re-k-means na unii centroidów; ryzyko pre-rejestrowane: fuzja
  może być gorsza niż lepsza z połówek).
- **I3 — skala (headline rewolucji):** N=5 agentów × 2 klasy Fashion →
  system zbiorczy vs g1_all (80.45/81.16) i vs pojedynczy agent seq
  (78.49). Twierdzenie docelowe: „N agentów, zero wymienionych obrazów,
  ≈ scentralizowany sufit".

Pozycjonowanie: federated learning wymienia gradienty/wagi; my wymieniamy
sen. Prywatność: żadna próbka nie opuszcza agenta; wiadomość jest
generatywna tylko w przestrzeni cech zamrożonego backbone'u.

## Etap L — fork tożsamości (branch `droga-l`; decyzja PO komplecie I)

Podmiana warstwy (1): zamrożony pretrenowany/SSL encoder (kandydaci wg
RELATED_WORK: SSL ResNet / CLIP image encoder). Mechanizm i protokół BEZ
ZMIAN. Jawnie osobna oś zasobów (from-scratch vs foundation) — stare
liczby nietykane, narracja rozdzielona jak w RELATED_WORK.

- **L1:** single-agent Split-CIFAR-10 na mocnych cechach (cel: znad
  37.51 w stronę sufitu nowej reprezentacji — mierzonego analogiem K0).
- **L2:** kolektyw I3 na mocnych cechach.

[DECYZJA ROBERTA]: wejście w L dopiero po I; wybór encodera przy
pre-rejestracji DROGA_L_PLAN.

## Kolejność

1. Decyzja: kryterium parowe (kształt jw. albo korekta Roberta).
2. DROGA_K_PLAN.md — pre-rejestracja szczegółowa K0/K1/K2 + kod
   (nowe pliki, wzór serii J).
3. Runy K u Roberta (smoke → FULL), werdykty, merge `droga-k`.
4. DROGA_I_PLAN.md (progi sukcesu wg decyzji), kod, runy I, merge.
5. Decyzja o forku → DROGA_L_PLAN.md → runy L.

Po każdym merge'u: aktualizacja WHITEPAPER (część serii), tag.
