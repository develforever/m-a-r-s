# Droga F — Continual Learning (plan badawczy; TO jest wątek "rewolucji")

Data: 2026-07-06
Status: PLAN (decyzja kierunkowa użytkownika: CL zamiast efektywności GPU)
Zasady serii: 5 seedów, kryteria werdyktu z góry, wynik negatywny = wynik.

---

## 0. Dlaczego tu, i dlaczego uczciwie tu

E4 (smoke) pokazał, że na CUDA efektywność MAC-owa nie realizuje się w czasie
(MLP 13× szybszy od slim CNN mimo większego MAC) — rewolucji energetycznej na
desktopowych GPU nie będzie. Ale projekt od Fazy 1 nosi w sobie wynik, którego
nie eksploatowaliśmy: **modularność strukturalnie chroni przed catastrophic
forgetting** (Etap 2: +45pp retencji; MNIST split: +25.7pp). Monolity nie
zapominają "trochę" — zapominają wszystko (baseline: 0% retencji), i to nie
jest kwestia strojenia, tylko architektury.

Continual learning to otwarty problem pola z realną stawką (agenci uczący się
w trakcie życia, edge bez retrainingu, prywatność — dane nie wracają do
serwera). Tu modularność nie walczy z GPU o throughput — robi coś, czego
monolit nie potrafi wcale. To jest właściwy adres dla "rewolucji".

**Uczciwe pozycjonowanie (żeby nie odkrywać koła):** pomysł "ekspert per
zadanie + bramka" istnieje (Expert Gate, Progressive Nets, PackNet...).
Nasza przewaga to nie sam pomysł, lecz: (a) prototypowy router w trybie
class-incremental BEZ etykiety zadania w teście — najtrudniejszy wariant CL,
na którym większość metod się łamie; (b) prawa projektowe z serii D
(D5: nie ruszać wspólnej reprezentacji — w CL to jest DOKŁADNIE mechanizm
zapominania!); (c) stały koszt inferencji vs liczba zadań (top-1 pod);
(d) rygor metodyczny, w tym uczciwe suficity (lekcja oracle inflation z E2).

## 1. Nić łącząca z serią D (to nie jest nowy projekt)

Seria D odkryła: dotknięcie wspólnego backbone'u psuje skalibrowane pody
(D5: −2.5pp od samego fine-tuningu celu pomocniczego; E2-v1: coarse labels
zawaliły oracle z 98% do 68%). **Catastrophic forgetting to to samo zjawisko
w wersji sekwencyjnej.** Droga F podnosi prawo z D5 do rangi zasady
konstrukcyjnej: reprezentacja wspólna jest nietykalna po nauczeniu; cała
plastyczność żyje w prototypach routera i podach.

## 2. Benchmark i protokoły (F0)

Zbiory: Split-MNIST i Split-Fashion (5 zadań × 2 klasy, standard pola),
później Split-CIFAR-10 (E3 wchłonięte przez Drogę F).

Dwa protokoły (oba mierzymy zawsze):
- **Task-IL** (łatwy): w teście znana etykieta zadania → wybór głowicy.
- **Class-IL** (trudny, GŁÓWNY): test bez etykiety zadania — system sam musi
  rozpoznać, "z którego świata" jest próbka. Tu głównie łamią się metody.

Metryki (standard CL + nasze):
- **ACC** — średnia accuracy po wszystkich zadaniach na koniec sekwencji.
- **Forgetting (F)** — średni spadek per zadanie: max_acc_t − final_acc_t.
- **BWT** (backward transfer) — wpływ nowych zadań na stare.
- **MAC(T)** — koszt inferencji jako funkcja liczby zadań (nasza specjalność:
  top-1 pod → powinien być ~stały; monolit rośnie lub wymaga retrainingu).
- Krzywe acc(t) po każdym zadaniu, 5 seedów, per-seed parowanie.

Baseline'y (F0, bez nich wyniki nie znaczą nic):
1. **Monolit fine-tune** — dolna granica (katastrofa oczekiwana).
2. **Monolit joint** — górna granica (wszystkie dane naraz; nieosiągalne
   w CL, ale uczciwy sufit — bez oracle inflation, patrz E2).
3. **Replay buffer** (mały, np. 200 próbek) — najprostsza silna metoda;
   jeśli MARS nie bije naiwnego replay, nie ma tezy.
4. **EWC** — klasyczna regularyzacja (mieliśmy w Etapie 2: nie wystarcza).

## 3. MARS-CL — architektura F1

Backbone: slim CNN S2 (D6b; tani i wystarczający) trenowany na PIERWSZYM
zadaniu (albo wariant F1c: na danych pomocniczych), potem ZAMROŻONY na zawsze
(prawo D5). Plastyczność wyłącznie w:
- **prototypach routera**: nowe zadanie = nowe prototypy (2 na zadanie);
  stare prototypy ZAMROŻONE (nie ma interferencji wag — kandydat na mechanizm
  odporności),
- **podach**: nowy pod per zadanie (stacked, jak v2), stare pody zamrożone.

Inferencja class-IL: router porównuje embedding z WSZYSTKIMI prototypami
(starych i nowych zadań) → top-1 pod. Koszt: backbone + routing(T·2 protos)
+ 1 pod = ~stały (routing rośnie liniowo, ale to ~promile kosztu).

Warianty F1 (pre-rejestrowane):
- **F1a frozen-backbone** (opisany wyżej) — czysta teza.
- **F1b growing-protos-tuned**: prototypy starych zadań LEKKO dostrajane
  (lr×0.01) — test, czy sztywność pomaga czy szkodzi.
- **F1c aux-backbone**: backbone trenowany na zadaniu 1 vs na wszystkich
  klasach zadania 1+2 — pomiar, jak bardzo jakość cech z pierwszego zadania
  ogranicza późniejsze (transfer cech; uczciwe ryzyko nr 1).
- **F1d random frozen backbone (reservoir)**: backbone NIGDY nie trenowany
  (losowe wagi, zamrożone od init). Jeśli losowe cechy wystarczą, ryzyko nr 1
  znika z definicji — backbone niczego nie faworyzuje. Tani test o dużej
  wadze (patrz `ARSENAL_PRZEOCZONYCH_NARZEDZI.md`).

Dalsze etapy z arsenału (po werdykcie F1): router bez wag (LSH —
"nie ma wag = nie ma zapominania"), task-free CL przez detekcję
nowości (Bloom/sketch na embeddingach — system sam wykrywa nowe zadanie).

## F3 — Parametryczny feature replay ("sen" bez danych; pomysł z analizy
## zewnętrznego agenta 07.07.2026, zmodyfikowany)

**Diagnoza-cel:** pody 10-way widzą w treningu niemal wyłącznie klasy
własnego zadania → brak negatywów międzyzadaniowych → logity
nieskalibrowane między zadaniami (podejrzany główny składnik luki
~15pp do replay przy niskim forgettingu).

**Mechanizm:** backbone zamrożony ⇒ cechy stacjonarne ⇒ zamiast bufora
obrazów przechowujemy per klasa TYLKO średnią i wariancję cech
(2×128 liczb, diagonalny Gaussian). Przy nauce zadania t pody trenują na
cechach zadania + PRÓBKACH z Gaussianów starych klas (pseudo-negatywy).
Zero przechowywanych danych (prywatność), koszt pamięci ~1 KB/klasę.

**Werdykt (z góry, Split-Fashion class-IL):** F3 vs replay-200 per-seed:
Δ ≥ −próg szumu → SYGNAL+ ("statystyki zamiast danych" ≈ replay);
dodatkowo F3 vs F1d: ile luki zamknęły pseudo-negatywy. Min per-seed
raportowany (lekcja E4). Wariant: kalibracja też prototypów (NCM bez zmian
vs re-fit na mieszance) — pre-rejestrowane oba.

## Horyzont (nie teraz): Active Inference (agentowe "ruchy oka",
Friston) — odnotowane po analizie zewnętrznej; odroczone świadomie:
28×28 daje mało do aktywnego próbkowania, D7 pokazał redundancję kanału
predykcyjnego na naszych cechach, RL = eksplozja zakresu. Wrócić po CIFAR.

## 4. Kryteria werdyktu (Z GÓRY)

Na Split-Fashion, class-IL, 5 zadań, vs baseline'y z F0:

| Porównanie | SYGNAL+ jeśli |
|---|---|
| MARS vs monolit fine-tune | ACC wyżej o >10pp (oczekiwane; kontrola sanity) |
| **MARS vs replay-200** | **ACC ≥ replay przy ZEROWYM buforze** (główna teza: prywatność/pamięć za darmo) |
| MARS vs EWC | ACC wyżej ponad próg szumu |
| Koszt | MAC(T=5)/MAC(T=1) ≤ 1.05 przy top-1 (teza stałego kosztu) |

Każdy wynik publikowalny: jeśli replay-200 bije MARS, to mierzalna granica
podejścia "architektura zamiast pamięci" — też wynik.

## 5. Ryzyka (uczciwie)

1. **Jakość cech z zadania 1.** Backbone widzi 2 klasy → cechy mogą nie nieść
   informacji o późniejszych klasach (Fashion: nauka na T-shirt/Trouser nie
   przygotuje na Sneaker/Bag). F1c mierzy to wprost. Mitygacja badawcza:
   backbone z augmentacją/kontrastem — ale to osobny eksperyment (F2).
2. **Router drift w class-IL**: prototypy nowych klas mogą "wygrywać"
   z niepewnymi starymi (recency bias). Pomiar: routing per zadanie po każdej
   fazie. Ewentualna mitygacja: kalibracja per-proto (F2).
3. **Nowość ograniczona** — patrz pozycjonowanie w sekcji 0; bijemy się
   rygorem i trybem class-IL, nie samym pomysłem.
4. Split-MNIST bywa "za łatwy" — dlatego Split-Fashion jako główny,
   Split-CIFAR jako cel serii.

## 6. Kolejność wykonania

1. **F0** — `src/run_F0_cl_baselines.py`: splity + 4 baseline'y + metryki
   ACC/F/BWT (wspólny moduł `cl_common.py`).
2. **F1** — `src/mars_cl.py` + `src/run_F1_mars_cl.py`: warianty a/b/c.
3. Wyniki → `DROGA_F_NOTATKI.md`; werdykt → decyzja o F2 (mechanizmy)
   i Split-CIFAR.
4. E4 (pełny) domknąć w tle — liczby MAC/czas wchodzą do tezy stałego kosztu
   i limitations; wątek energetyczny w whitepaperze przechodzi do limitations.
