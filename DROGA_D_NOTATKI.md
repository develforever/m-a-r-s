# Droga D — notatki robocze (stan na 17.06.2026)

## Gdzie jesteśmy

Ukończone i zapisane w `src/`:
- `mars_v2.py` — rdzeń v2: Shared Backbone + Adaptive Compute (forward_adaptive: EE/top-1/top-2), oba tryby treningu (phased / end-to-end), sweep + Pareto.
- `run_D1_mars_v2_baseline.py` — Etap 2: trening + punkt kontrolny (router v2 = 98.3% = C4 ✓) + baseline v1 zrównany parametrycznie (408.9k).
- `run_D1b_pareto_sweep.py` — Etap 3: krzywa Pareto (Accuracy vs MAC).
- `run_D1c_multiseed.py` — Droga 1: walidacja 5-seed.

Wyniki w `results/`: `D1_mars_v2_baseline.json`, `D1b_pareto_sweep.json`, `D1c_multiseed.json`.

## Kluczowe wnioski z D1 (wszystkie multi-seed, 5 seedów)

1. **v2 ≈ v1 przy zrównanych parametrach.** Różnice v2−v1 to SZUM:
   - MNIST: −0.03 / −0.07pp przy std 0.10pp.
   - Fashion: +0.15 / −0.09pp przy std 0.27pp.
   Shared backbone NIE bije architektury separate. Pozorna przewaga C4 brała się z treningu podów na ORACLE.

2. **Phased ≈ end-to-end.** Nieodróżnialne. Zamrażanie backbone'u NIE jest konieczne (wbrew założeniu pierwotnego planu).

3. **ORACLE skorygowany w dół.** v1 ORACLE 99.6%, v2 ~95.9% (phased) / 94.4% (e2e) — stabilnie. Uczciwy trening podów (na routingu, nie etykiecie) systematycznie obniża zawyżony ORACLE z C4.

4. **Adaptive Compute strukturalnie słaby na shared backbone.** Early Exit oszczędza tylko pod_mac (~9.5k z 323k); backbone (301k) płacony zawsze. Nawet 95% próbek na EE ścina ~3% MAC. Selective Top-2 neutralny do lekko szkodliwego.

## Wniosek strategiczny

C3 + D1 zgodnie: **router jest jedynym wąskim gardłem.** Przebudowa wszystkiego wokół routera nic nie ruszyła. Pole gry dla poprawy to przede wszystkim **Fashion-MNIST** (luka ~10pp do ORACLE); MNIST jest praktycznie pod sufitem.

REALISTYCZNE OCZEKIWANIE: jeśli D4/D5 dadzą +1–2pp na Fashion = sukces. Jeśli nic — router osiągnął sufit przy tej reprezentacji, a prawdziwa dźwignia to lepsze CECHY (B1b CNN router, +3.5pp), nie sprytniejszy routing. Wtedy następny krok: CNN router jako fundament v2, nie dalsze kombinowanie przy routingu.

## D5 — Distillation pod→router (ZAKOŃCZONE, wynik negatywny)

Plik: `src/run_D5_distillation.py`, wyniki: `results/D5_distillation.json`

**Wyniki (5 seedów):**

| Model | MNIST system | Fashion system | Δ Fashion |
|---|---|---|---|
| base phased (D1c) | 98.36 ± 0.05% | 89.50 ± 0.11% | — |
| D5a soft (dynamic teacher) | 98.31 ± 0.08% | 87.11 ± 1.00% | **−2.39pp** [SYGNAL] |
| D5b oracle-pod | 98.30 ± 0.08% | 86.96 ± 1.05% | **−2.54pp** [SYGNAL] |
| D5c self-distil (frozen teacher) | 98.30 ± 0.08% | 86.95 ± 1.03% | **−2.55pp** [SYGNAL] |

**Kluczowe obserwacje:**
1. Sygnal jest wyraźny (>1std), ale **negatywny** — distillation PSUJE system.
2. Routing_acc zostaje prawie bez zmian (~89.4→89.5%) — router routuje tak samo.
3. System_acc spada mimo zamrożonych podów. Mechanizm: distillation fine-tunuje backbone, żeby `route_logits` upodabniały się do `pod_out`. Ale `pod_out` też zależy od backbone — optymalizacja zaburza przestrzeń cech, na której pody (z zamrożonymi wagami) są skalibrowane.
4. Nie ma znaczenia wariant (soft/oracle/self) ani siła sygnału nauczyciela — efekt jest identyczny. To wina mechanizmu, nie wyboru nauczyciela.

**Wniosek:** Distillation pod→router na shared backbone to sprzeczny cel — nie da się upodobnić głowicy routingu do głowicy poda bez zaburzenia wspólnej reprezentacji. Wynik negatywny, wart odnotowania w paperze jako ograniczenie architektury shared backbone.

**Implikacja strategiczna:** D5 zamknięte. Dalsze próby distillation w tej architekturze bezsensowne.

## NASTĘPNY KROK: Droga 2 — atak na router

Kolejność ustalona: **najpierw D5 (distillation), potem D4 (consultation).**

### D5 — distillation pod→router (ZAKOŃCZONE — patrz wyniki powyżej)

### D4 — consultation (ZAKOŃCZONE, wynik: SZUM)

Plik: `src/run_D4_consultation.py`, wyniki: `results/D4_consultation.json`

**Wyniki (5 seedów, sweep θ ∈ {0.5..1.01} × k ∈ {2,3,5}):**

| Dataset | Base top-1 | Najlepsza consultation | Delta |
|---|---|---|---|
| MNIST | 98.36 ± 0.05% | 98.36% (każda config) | **+0.00pp** |
| Fashion-MNIST | 89.50 ± 0.11% | 89.50% (top1, 0% consult) | **+0.00pp** |

**Kluczowe obserwacje:**
1. MNIST: identyczne liczby per seed we WSZYSTKICH konfiguracjach, włącznie z θ=1.01 (konsultuj wszystkich). Ważona suma softmaxów top-k daje ten sam argmax — pody zbyt pewne, ensemble nic nie zmienia.
2. Fashion: nawet θ=1.01 (full ensemble) nie pomaga. Consultation ≥0.01pp kosztuje więcej MAC bez zysku acc. Pareto front = tylko punkty z 0% konsultacji.
3. Wynik spójny z D5: router i pody widzą te same cechy backbone — pytanie dodatkowych podów nie dodaje informacji, która nie byłaby już zakodowana w rozkładzie routingu.

**Wniosek:** D4 potwierdza tezę z D1+C3+D5: **router osiągnął sufit tej reprezentacji**. Sprytniejsze routing nie pomaga; potrzebne są lepsze cechy.

## Wniosek z Drogi 2 (D4 + D5) — potwierdzenie sufitu routera

D4 i D5 zgodnie: **brak zysku z jakiejkolwiek modyfikacji routera po stronie architektury v2 (shared backbone, te same cechy)**.

- D5 distillation: pody nie mogą nauczyć routera niczego poza CE → aktywnie psuje (zaburza reprezentację na shared backbone).
- D4 consultation: pytanie top-k podów przy niskim confidence = 0 zysku → pody i router mają te same cechy, redundancja zupełna.

**Strategia dalej: Droga 3 — CNN backbone jako fundament v2.**

Opcje (do dyskusji z użytkownikiem):
- **D6 — CNN backbone** (jak B1b, +3.5pp na Fashion): zamiana MLP backbone na małe CNN → nowa, bogatsza reprezentacja → re-run v2 z CNN.
- **D7 — v3 Dialogue** (Future Work z mapy wersji): iteracyjny router↔pody.

## D6 — CNN backbone (ZAKOŃCZONE, wynik: SYGNAL+ POTWIERDZONY, 06.07.2026)

Pliki: `src/mars_v2_cnn.py`, `src/run_D6_cnn_backbone.py`; wyniki: `results/D6_cnn_backbone.json`.
Setup: 5 seedów × 30 epok, GPU (GTX 1050 Ti), train_phased REUŻYTY (różni modele TYLKO backbone). MLP bb_h=256 vs CNN bb_h=128 ch=(32,64).

| Dataset | MLP system | CNN system | Δ system | per-seed min | Werdykt |
|---|---|---|---|---|---|
| Fashion-MNIST | 89.61 ± 0.21% | 91.99 ± 0.12% | **+2.38 ± 0.33pp** | +1.90 (wszystkie 5 dodatnie) | **SYGNAL+** |
| MNIST | 98.34 ± 0.05% | 99.19 ± 0.06% | **+0.86 ± 0.10pp** | +0.76 (wszystkie 5 dodatnie) | **SYGNAL+** |

**Kluczowe obserwacje:**
1. **Zysk płynie przez ROUTING.** Fashion routing 89.30 → 91.88% (+2.58pp); to on ciągnie system. To domyka tezę serii D: router był ograniczony REPREZENTACJĄ, nie algorytmem. D4/D5 nie ruszyły routingu kombinując przy routingu; D6 rusza go zmianą CECH. Teza potwierdzona empirycznie i pozytywnie.
2. **Pody też lepsze.** Fashion oracle 95.49 → 98.10%. Bogatszy backbone podnosi cały system, nie tylko router.
3. **Sygnał czysty.** Delta >> próg szumu (0.33pp) na obu zbiorach; brak przeplotu między seedami (min Δ dodatnie).
4. **KOSZT: ~19.7× MAC** (215,600 → 4,247,600; dominuje conv2 32→64 na 14×14). To uczciwa cena: +2.38pp Fashion za ~20× compute. Dla tezy "lepsze cechy podnoszą sufit" — OK. Dla tezy o EFEKTYWNOŚCI (rewolucja energetyczna) — NIE liczy się jeszcze; wymaga odchudzenia CNN.
5. **Luka router→oracle nadal otwarta.** CNN Fashion: system 91.99 vs oracle 98.10 = **6.11pp** wciąż do zdobycia lepszym routingiem. Router pozostaje wąskim gardłem NAWET na CNN → D7 (predictive coding routing) ma sens: pody potrafią 98%, jeśli je dobrze zaadresować.

**Wniosek strategiczny:** CNN backbone = fundament v2 od teraz. Dwa równoległe kierunki:
- **Efektywność:** odchudzić CNN (mniej kanałów / stride zamiast MaxPool / depthwise) i zmierzyć, ile z +2.38pp przetrwa przy MAC bliższym MLP. To jest wątek "rewolucji" (dokładność za rozsądny koszt).
- **Sufit routera:** D7 na CNN backbone (patrz `D7_PLAN.md`) — luka 6.11pp do oracle to przestrzeń dla dialogu router↔pody.

## D6b — odchudzenie CNN (ZAKOŃCZONE, wynik: SYGNAL+ EFEKTYWNOŚCIOWY, 06.07.2026)

Pliki: `src/mars_v2_slim.py`, `src/run_D6b_slim_cnn.py`; wyniki: `results/D6b_slim_cnn.json`.
Setup: 5 seedów × 30 epok, GPU; baseline'y MLP i pełny CNN reużyte per-seed z D6 (te same seedy 0–4, ten sam protokół — porównanie parami uczciwe). Budżet efektywnościowy: MAC ≤ 2.2× MLP (215.6k).

| Wariant (Fashion) | MAC | ×MLP | system | Δ vs MLP | min Δ | retention vs D6 |
|---|---|---|---|---|---|---|
| S1 half (16,32) | 1 224k | 5.68× | 91.55 ± 0.25% | +1.94 ± 0.36pp | +1.52 | **82%** |
| **S2 quarter (8,16)** | **390k** | **1.81×** | **91.27 ± 0.23%** | **+1.66 ± 0.39pp** | **+1.18** | **70%** |
| S3 stride (16,32) | 462k | 2.14× | 90.58 ± 0.20% | +0.97 ± 0.32pp | +0.42 | 41% |
| S4 depthwise (16,32) | 450k | 2.09× | 91.01 ± 0.40% | +1.40 ± 0.55pp | +0.73 | 59% |

**WERDYKT (Fashion, S2 w budżecie): SYGNAL+ efektywnościowy** — Δ +1.66pp > 1.0pp i > próg szumu (0.61pp); wszystkie 5 seedów dodatnie.

MNIST: S2 +0.73 ± 0.07pp (retention 85%), wszystkie seedy dodatnie, Δ >> próg szumu (0.12pp). Skrypt wypisał "SZUM", bo kryterium ≥1.0pp było kalibrowane pod Fashion — statystycznie to czysty sygnał, tylko poniżej progu wielkości efektu. Odnotować, nie przereklamować.

**Kluczowe obserwacje:**
1. **Teza efektywnościowa POTWIERDZONA.** 70% zysku D6 przetrwało przy 1.81× MAC (zamiast 19.7×). Zysk CNN pochodzi z lokalności/inwariancji konwolucji, nie z surowego compute. To domyka wątek "rewolucji": +1.66pp za +81% MAC to realna dźwignia, nie kupowanie accuracy compute'em.
2. **Diminishing returns wyraźne:** 1.81× → +1.66pp; 5.68× → +1.94pp; 19.7× → +2.38pp. Większość zysku jest w pierwszych ~2× kosztu.
3. **S3 (stride) — najciekawsza anomalia:** najniższy system (90.58%), ale oracle 99.53% — WYŻSZY niż pełny CNN z D6 (98.10%). Stride pomaga podom, szkodzi routerowi; luka router→oracle ≈ 8.9pp (największa w serii). Kandydat na stress-test D7: jeśli predictive coding działa, na S3 powinien dać najwięcej.
4. **S4 (depthwise) niestabilny** (std 0.55pp, rozrzut per-seed +0.73…+1.86) — bez przewagi nad S2 przy podobnym MAC.

**Wniosek strategiczny:** S2_quarter = backbone efektywnościowy (390k MAC, 91.27% Fashion). Decyzja dla D7: uruchomić na **pełnym CNN D6** (najbogatsze cechy = uczciwszy test rekonstrukcji, zgodnie z D7_PLAN.md ryzyko 4), z opcjonalnym drugim biegiem na S3 (największa luka do oracle). S2 wchodzi do whitepapera jako punkt Pareto "rewolucji".

## D7 — Predictive coding routing (ZAKOŃCZONE, wynik: SZUM, 06.07.2026)

Pliki: `src/mars_v2_pc.py`, `src/run_D7_predictive_coding.py`; wyniki: `results/D7_predictive_coding.json`. Plan i hipoteza: `D7_PLAN.md`.
Setup: 5 seedów × 30 epok + 10 epok dekoderów, backbone = pełny CNN D6, baseline = TEN SAM model routowany logitami (parowanie per-seed idealne). Warianty: D7a (rekonstrukcja cech) / D7b (rekonstrukcja wejścia) × hard / fuse (λ ∈ 0.25–4) / iter (k ∈ 2,3).

| Wariant (Fashion) | system | Δ vs baseline | Werdykt |
|---|---|---|---|
| baseline (logity) | 92.08 ± 0.27% | — | — |
| D7a_fuse (najlepszy: λ=4) | 92.12 ± 0.22% | +0.05 ± 0.07pp | SZUM (próg 0.34pp; per-seed −0.03…+0.17) |
| D7a_hard | 91.14 ± 0.46% | −0.94 ± 0.41pp | gorszy od routera |
| D7a_iter k2/k3 | 91.3–91.5% | −0.62…−0.75pp | gorszy |
| D7b_hard | 72.23 ± 3.81% | −19.84pp | katastrofa |
| D7b_iter k2/k3 | 79–84% | −8…−13pp | katastrofa |

MNIST: identyczny obraz (najlepszy fuse +0.00 ± 0.01pp, próg 0.05pp — SZUM).

**WERDYKT: SZUM** — zgodnie z kryterium z D7_PLAN.md sekcja 5: błąd rekonstrukcji jest funkcją tych samych cech backbone'u i nie wnosi informacji ponad logity routera. **Sufit reprezentacji potwierdzony także dla kanału predictive coding. Droga 3 od strony routingu DOMKNIĘTA.**

**Kluczowe obserwacje:**
1. **Redundancja potwierdzona wprost (ryzyko 1 z planu):** D7a_hard routuje 90.24% vs router 91.98% — rekonstrukcja cech odtwarza NIEMAL tę samą decyzję, tylko trochę gorzej. Kanał PC ≈ zaszumiona kopia kanału logitów.
2. **Fuse nigdy nie psuje i nigdy nie pomaga** (λ do 4.0, Δ w [−0.03, +0.05]pp) — sygnał PC jest zdominowany przez router, niesprzeczny, ale pusty informacyjnie.
3. **D7b (rekonstrukcja wejścia) za słaby strukturalnie:** dekoder 24→784 nie ma pojemności (mse 0.35–0.56 vs 0.15–0.02 dla cech); routing po nim −20pp. Iter psuje wybór wśród top-k z tego samego powodu.
4. Spójność serii: D4 (ensemble logitów) = 0.00pp, D5 (distillation) = −2.5pp, D7 (rekonstrukcja) = +0.05pp. Trzy niezależne kanały, jeden wniosek — **na shared backbone jedyną dźwignią routingu są CECHY (D6/D6b), nie algorytm routingu.**

**Konsekwencja dla whitepapera:** narracja serii D jest kompletna i symetryczna: (i) routing ma sufit reprezentacji (D4+D5+D7, trzy kanały), (ii) cechy ten sufit podnoszą (D6, +2.38pp), (iii) i robią to efektywnie (D6b, 70% zysku za 1.81× MAC).

**Stress-test na S3 (WYKONANY, 06.07.2026): SZUM potwierdzony w najtrudniejszych warunkach.** Wyniki: `results/D7_predictive_coding_s3.json`. S3 to backbone o NAJWIĘKSZEJ luce router→oracle (Fashion: system 90.62 ± 0.27%, oracle 99.36% — luka 8.74pp; pody niemal doskonałe). Jeśli kanał rekonstrukcji miałby gdziekolwiek zadziałać, to tu. Wynik: najlepszy fuse +0.05 ± 0.04pp (próg 0.31pp — SZUM), hard −2.07pp, iter −1.3…−1.7pp. MNIST identycznie (+0.00pp przy oracle 100.00%). Predictive coding nie adresuje podów lepiej nawet tam, gdzie pody potrafią 99.4%. Sufit reprezentacji jest twardy niezależnie od backbone'u — wynik do whitepapera jako najmocniejsza forma dowodu.

## Mapa wersji (do whitepapera)

- v1 Separate → v2 Shared+Adaptive (= remis z v1, wynik negatywny wart publikacji) → v2+CNN backbone (D6, SYGNAL+ na routingu, +2.38pp Fashion za ~20× MAC) → v2+slim-CNN (D6b, SYGNAL+ efektywnościowy: 70% zysku D6 za 1.81× MAC — S2_quarter) → **v3 Dialogue / Predictive Coding (D7, SZUM — rekonstrukcja redundantna wobec logitów; sufit reprezentacji potwierdzony trzecim kanałem, Droga 3 od strony routingu domknięta)**. Seria D kompletna: sufit routingu (D4+D5+D7) + dźwignia cech (D6) + efektywność (D6b).
