# Droga K — plan pre-rejestrowany: wyżyłowanie obecnej drogi

Data pre-rejestracji: 2026-07-17 (branch `droga-k`; main/v0.4 nietknięty).
Kontekst strategiczny: `PLAN_GENERALNY.md` (f777550) — K jest etapem
„maksimum z obecnej tożsamości" przed Drogą I (kolektyw) i L (fork).
Decyzja Roberta 2026-07-17: plan generalny zatwierdzony, w tym kryterium
parowe i progi sukcesu I1.

## Zasady (jak seria J) + NOWE kryterium parowe

Bez zmian: 5 seedów (0–4), pary per-seed, epochs=15, LR=0.001,
próg szumu = std(baza) + std(wariant), SYGNAL+ wymaga śr. > próg ORAZ
min per-seed > 0 (SYGNAL− symetrycznie po średniej), wynik negatywny =
wynik, nowe pliki, runy wyłącznie u Roberta, merge po komplecie werdyktów.

NOWE (obowiązuje OD K, nie wstecz — decyzja przed pierwszym runem,
zgodnie z zapowiedzią w J3/ustalenie 2): dodatkowa klasa werdyktu
**SYGNAL-parowy±**: wszystkie pary per-seed jednego znaku ORAZ
|śr. delt| > 2×std(delt). Sprawdzany TYLKO gdy klasyczny werdykt = SZUM.
Hierarchia: SYGNAL± → SYGNAL-parowy± → SZUM. Interpretacja: efekt mały,
ale systematyczny (typ J3), odróżniony od czystego nulla (typ J4).

Legalność par bez re-runu bazy: konstrukcja modeli w K odtwarza
kolejność konsumpcji RNG z runnerów bazowych (precedens J2b vs J2;
determinizm ścieżki potwierdzony czterema reprodukcjami 0.00pp).
Klasa OWM nie konsumuje RNG w konstrukcji (bufor P = eye), więc wagi
startowe są identyczne z bazą przy tym samym seedzie.

## K0 — brakujący sufit zamrożonych cech na CIFAR (diagnostyka)

Plik: `src/run_K0_cifar_ceiling.py` → `results/K0_cifar_ceiling.json`

Motywacja: na Fashion mamy g1_all (80.45/81.16) jako sufit mechanizmu,
na CIFAR NIE — F4/J2 raportują tylko joint 70.24, który jest TRENOWALNY
(nie jest sufitem zamrożonych cech). Bez K0 nie wiadomo, ile z luki
37.51→70.24 jest mechanizmowe (do wzięcia w K/I), a ile reprezentacyjne
(do wzięcia dopiero w Etapie L).

Warianty (wejście znormalizowane jak J2/J2b; `MarsCLSemantic`,
`proj_train="all"`, `backbone_module=CifarBackbone()`):
  all_50  : GloVe 50d
  all_300 : GloVe 300d

**Kryteria (Z GÓRY):** K0 to diagnostyka, nie test hipotezy — bez
werdyktu SYGNAL/SZUM. Raportujemy: sufit_50, sufit_300,
gap_mech = max(sufit) − 37.51 (J2b sparse_k16),
gap_repr = 70.24 (J2 joint) − max(sufit).
Pre-rejestrowana interpretacja: gap_mech < 3pp → mechanizm na CIFAR
praktycznie domknięty (dalszy wzrost tylko przez Etap L);
gap_mech > 5pp → jest przestrzeń dla dźwigni mechanizmowych (K2, I);
3–5pp → strefa szara, decyzja po K2.

## K1 — złożenie dźwigni: sen sparse_k16 × GloVe 300d

Plik: `src/run_K1_sparse300.py` → `results/K1_sparse300.json`

Motywacja: J3/J2b — rzadkość snu to jedyna dźwignia SYGNAL+;
J4 — 300d podnosi sufit (+0.72, 5/5 par), ale seq z diag_k16 to null.
Pytanie: czy wierniejszy sen (sparse) pozwala projekcji 128→300
skonsumować bogatszą geometrię, której diag nie skonsumował?

Warianty (epochs_proj=15, l2sp=0, bez kondycjonowania):
  fashion_sp16_300 : Fashion, sparse_k16 × 300d
  cifar_sp16_300   : CIFAR-n, sparse_k16 × 300d

Bazy par (TE SAME seedy, bez re-runu):
  Fashion: `results/J3_sparse_dreams.json` sparse_k16 (78.49 ± 0.91)
  CIFAR:   `results/J2b_cifar_sparse.json` sparse_k16 (37.51 ± 1.35)

**Kryteria (Z GÓRY, class-IL, per dataset):**
- SYGNAL+ : śr. d > próg (std+std) ORAZ min per-seed > 0;
  SYGNAL− symetrycznie; SYGNAL-parowy± wg nowej reguły; inaczej SZUM.
- Kontekst: sufit Fashion 81.16 (J4 all_300), sufit CIFAR = K0 all_300.
- Ryzyko pre-rejestrowane (z J4): projekcja 128→300 = 6× parametrów =
  większy dryf; sen sparse może go nie utrzymać → SYGNAL− jest realny.

## K2 — OWM × sen sparse, tam gdzie boli

Pliki: `src/mars_cl_k.py` (MarsCLSemanticOWMSparse),
`src/run_K2_owm_sparse.py` → `results/K2_owm_sparse.json`

Motywacja: H1 (OWM przy śnie diag sprzed J): Fashion SZUM + eliminacja
(resztkowa luka ≠ dryf), MNIST SYGNAL+ (+5.0pp przy a10). OWM nigdy nie
biegał na CIFAR (forgetting 32.7pp — największy w projekcie) ani ze
snem sparse. Sen strzeże decyzji, OWM geometrii — złożenie nietestowane.
Kotwice 50d (izolacja dźwigni OWM od dźwigni K1; złożenie potrójne
ewentualnie jako K3 po werdyktach — NIE pre-rejestrowane tutaj).

Warianty (stats_k=16 sparse; epochs_proj=15, l2sp=0; use_dream=True,
owm_samples=2000 — dokładnie H1):
  mnist_owm_a10   : MNIST, owm_alpha=10   (GŁÓWNY — H1 wskazał a10)
  mnist_owm_a1    : MNIST, owm_alpha=1    (obserwacja)
  cifar_owm_a10   : CIFAR-n, owm_alpha=10 (GŁÓWNY)
  cifar_owm_a1    : CIFAR-n, owm_alpha=1  (obserwacja)
  fashion_owm_a1  : Fashion, owm_alpha=1  (kontrola eliminacji H1 —
                    oczekiwany SZUM; SYGNAL− też informacja)

Bazy par (TE SAME seedy): sparse_k16 z J3 (MNIST 73.26, Fashion 78.49)
i J2b (CIFAR 37.51).

**Kryteria (Z GÓRY, class-IL):**
- Główne (osobno MNIST i CIFAR): wariant a10 vs baza sparse_k16 —
  SYGNAL+/SYGNAL−/SYGNAL-parowy±/SZUM wg reguł jw. Wariant a1 =
  obserwacja (bez rangi werdyktu — przeciw grzebaniu w alfach post-hoc).
- Fashion (kontrola): każdy wynik raportowany; oczekiwanie
  pre-rejestrowane: SZUM (potwierdzenie eliminacji H1 przy nowym śnie).
- Metryka plastyczności: średnie R[t][t] (jak H1) — czy kurczenie
  null-space gryzie przy 5 zadaniach.

## Kolejność uruchomień u Roberta

1. `python src/mars_cl_k.py` (smoke jednostkowy klasy, CPU, sekundy)
2. `python src/run_K0_cifar_ceiling.py --smoke`, potem FULL
3. `python src/run_K1_sparse300.py --smoke`,     potem FULL
4. `python src/run_K2_owm_sparse.py --smoke`,    potem FULL

Wymagane pliki: data/glove.6B.50d.txt, data/glove.6B.300d.txt (jest po
J4), results/J2_cifar_normalized.json, J2b_cifar_sparse.json,
J3_sparse_dreams.json, J4_glove300.json.

Wyniki dopisujemy do DROGA_K_NOTATKI.md (powstanie po pierwszym runie);
merge do main po komplecie werdyktów i decyzji Roberta. Po merge'u:
DROGA_I_PLAN.md wg PLAN_GENERALNY (progi sukcesu I1 zatwierdzone:
mocny < próg szumu, słaby < 3pp).
