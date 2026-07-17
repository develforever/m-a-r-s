# Droga I — plan pre-rejestrowany: kolektywne uczenie przez wymianę snów

Data pre-rejestracji: 2026-07-17 (branch `droga-i`; main/v0.5 nietknięty).
Kontekst: `PLAN_GENERALNY.md` (Etap I — kandydat na rewolucję),
`ARSENAL_PRZEOCZONYCH_NARZEDZI.md` (dopisek 2026-07-10, pytania a/b/c).
Fundament zmierzony: stacjonarność (F1d, J2 SYGNAL+), wystarczalność
statystyk do wyśnienia klasy (J2b SYGNAL+), mechanizm wyżyłowany
(seria K: 97.6% sufitu Fashion, 94.6% CIFAR).

## Idea i wiadomość

N agentów MARS-CL z TYM SAMYM zamrożonym losowym backbonem (wspólny
seed — synchronizacja darmowa) i tą samą przestrzenią słów. Agent A
uczy się klasy na swoich danych i wysyła WYŁĄCZNIE:

  payload(c) = { p:[k,D], mean:[k,D], var:[k,D], w:[k], n:int }
             + nazwa klasy (kotwica GloVe)

tj. statystyki spike-and-slab (FeatureStatsKSparse, k=16, D=128;
~24.1 KB fp32) + liczność próby. Zero obrazów, zero gradientów, zero
wag. Odbiorca B śni cechy klasy z payloadu i uczy się jej DOKŁADNIE
ścieżką learn_task (projekcja + pody), z cechami wyśnionymi zamiast
realnych — to jedyna różnica (`adopt_classes` w `src/mars_collective.py`).

Konfiguracja domyślna (zwycięzca K1): Fashion, sparse_k16 × GloVe 300d,
epochs=15, epochs_proj=15, l2sp=0, LR=0.001. n_dream=6000/klasę
(parytet z liczebnością klasy Fashion). CIFAR poza zakresem I
(K0: mechanizm domknięty, reprezentacja słaba — kolektyw na CIFAR
dopiero w L2).

## Zasady

Jak seria K: 5 seedów (0–4), pary per-seed, próg szumu std+std,
SYGNAL± (śr. > próg ORAZ min > 0 / śr. < −próg), SYGNAL-parowy±
(wszystkie pary jednego znaku ORAZ |śr.| > 2×std delt, tylko przy
klasycznym SZUM), wynik negatywny = wynik, nowe pliki, runy u Roberta.

Higiena RNG (pary z K1 legalne): odbiorca konstruowany i uczony
tasków 0–3 PRZED konstrukcją nadawcy — strumień RNG odbiorcy identyczny
z run_K1 do końca taska 3; nadawca z torch.manual_seed(seed) → te same
wagi backbone'u (wymóg protokołu, nie sztuczka).

Progi sukcesu I1 (ZATWIERDZONE przez Roberta 2026-07-17, z góry):
strata przeszczepu = acc(task4)_local − acc(task4)_transplant,
pary per-seed, baza local = K1 fashion_sp16_300 (R[-1][4]):
- sukces MOCNY: |śr. straty| < próg szumu (równoważność przeszczepu
  z nauką lokalną);
- sukces SŁABY: śr. straty < 3pp;
- PORAŻKA: ≥ 3pp (też wynik — mierzy granicę protokołu).

BRAMKA SEKWENCYJNA (z góry): jeśli I1 = PORAŻKA, runy I2/I3 wstrzymane
do analizy (nie miękkie „zobaczymy" — twarda kolejność jak J1→J3-cond).

## I1 — przeszczep klasy (+ I1b: moment wymiany)

Plik: `src/run_I1_transplant.py` → `results/I1_transplant.json`

Warianty:
  transplant_end : B uczy taski 0–3 z danych; A (świeży agent, ten sam
                   seed) uczy task 4 u siebie; B adoptuje klasy {8,9}
                   z payloadu NA KOŃCU.
  transplant_mid : jak wyżej, ale adopcja PO tasku 1, potem B uczy
                   taski 2–3 (I1b: czy późniejsza nauka zjada przeszczep).
Baza (bez re-runu): local = K1 fashion_sp16_300 (te same seedy).

**Kryteria (Z GÓRY):**
- Główne: strata przeszczepu w transplant_end wg progów jw.
- Werdykt kierunkowy na pełnym ACC (10 klas): transplant_end vs local —
  SYGNAL±/parowy±/SZUM (transplant może być LEPSZY: adopcja śni świeże
  negatywy; symetria raportowana).
- I1b (obserwacja): strata przeszczepu mid vs end (pary) + acc klas 0–7
  (czy adopcja w środku zaburza stare klasy inaczej niż na końcu).
- Ryzyko pre-rejestrowane: pody klasy przeszczepionej uczone WYŁĄCZNIE
  na snach — strata >3pp znaczy, że sen wystarcza do ochrony, ale nie
  do nauki od zera; to wyznacza granicę protokołu.

## I2 — fuzja statystyk (ta sama klasa, rozłączne dane)

Plik: `src/run_I2_fusion.py` → `results/I2_fusion.json`

Setup: odbiorca C uczy taski 0–3; dane taska 4 dzielone losowo na pół
(generator=seed): połówka A i połówka B. Nadawca-obserwator (ten sam
seed) liczy payloady klas {8,9} na każdej połówce osobno.

Warianty adopcji u C (świeży C na wariant, identyczny do taska 3):
  half_A     : payload z połówki A (górna granica straty)
  fusion_cat : unia komponentów [2k], wagi ważone n_A/n_B (bez straty
               informacji, 2× pamięć)
  fusion_red : re-dream: sen z fusion_cat → ponowny k-means do k=16
               (kompresja z powrotem do 24 KB)
  full_stats : payload liczony na CAŁYM tasku 4 (referencja górna)

**Kryteria (Z GÓRY):**
- Główne: fusion_cat vs half_A (pary per-seed, acc task4):
  SYGNAL+/parowy+/SZUM/SYGNAL− — czy fuzja dwóch częściowych widoków
  bije pojedynczy widok?
- Obserwacje: fusion_cat vs full_stats (ile kosztuje podział danych),
  fusion_red vs fusion_cat (ile kosztuje kompresja k-means na snach).
- Ryzyko: k-means na snach = kompresja stratna; fusion_red może zgubić
  modalności.

## I3 — skala: N=5 agentów × 2 klasy (headline rewolucji)

Plik: `src/run_I3_collective.py` → `results/I3_collective.json`

Setup: agent_i (i=0..4, wspólny seed) uczy się WYŁĄCZNIE taska i jako
swojego jedynego zadania. Agent 0 = kolektor: po nauce taska 0 adoptuje
kolejno payloady tasków 1→4 (4 adopcje × 2 klasy; symulacja
przychodzących wiadomości). Finał: 10-klasowy class-IL agenta 0.

**Kryteria (Z GÓRY):**
- Główne: kolektyw (agent 0) vs pojedynczy agent sekwencyjny
  (K1 fashion_sp16_300, 79.23 ± 0.73, te same seedy, pary):
  SYGNAL+/parowy+/SZUM/SYGNAL−. SYGNAL+ = twierdzenie rewolucji:
  „5 agentów, zero wymienionych obrazów, wynik ≥ scentralizowanego
  uczenia sekwencyjnego".
- Obserwacje: luka do sufitu g1_all_300 (81.16, J4); kontekst
  replay-200 (76.97); krzywa acc po każdej adopcji (degradacja?).
- Ryzyko pre-rejestrowane: 4 adopcje z rzędu = projekcja douczana
  4× na samych snach; kumulacja dryfu → SYGNAL− realny i też jest
  wynikiem (wyznacza limit skali protokołu).

## Kolejność uruchomień u Roberta

1. `python src/mars_collective.py` (smoke jednostkowy, CPU, sekundy)
2. `python src/run_I1_transplant.py --smoke`, potem FULL
   → BRAMKA: jeśli PORAŻKA, stop i analiza.
3. `python src/run_I2_fusion.py --smoke`, potem FULL
4. `python src/run_I3_collective.py --smoke`, potem FULL

Wymagane: data/glove.6B.300d.txt, results/K1_sparse300.json (baza par),
results/J4_glove300.json (sufit), results/F0_cl_baselines.json (kontekst).

Wyniki do DROGA_I_NOTATKI.md; merge po komplecie i decyzji Roberta.
