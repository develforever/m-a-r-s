# Droga R2 — kolektyw heterogeniczny przez wyrównanie cech (Procrustes) (pre-rejestracja)

Data pre-rejestracji: 2026-07-23. Status: **DO ZATWIERDZENIA**; runy
WYŁĄCZNIE u Roberta. Branch: `droga-r` (nowe pliki, rdzeń I/L NIETKNIĘTY).
Oś: **Part III (memory without data / protokół kolektywu)** — powrót na
główną narrację CL. Odblokowuje serię R na INNYM poziomie abstrakcji niż
sfalsyfikowany R1b.

## Punkt wyjścia (zmierzony — dlaczego feature-space, nie kotwica)

R1b sfalsyfikował kotwicę-interlingua. Diagnostyka rozstrzygająca:
**ORACLE_SANITY = 4.00%** (dekoder z pełną wiedzą 10 klas + zero
heterogeniczności) vs CEILING 81.6% → wąskim gardłem NIE był dekoder ani
heterogeniczność, lecz sam payload anchorowy: 50-d przestrzeń słów jest
„class-collapsed" (projekcja zwija klasę do ~word_vec), niesie tożsamość,
nie geometrię. Wniosek zapisany wtedy: jedyna droga zgodna z ORACLE=4% to
**translacja w PRZESTRZENI CECH** (zachowuje wewnątrzklasową geometrię).
R2 realizuje to wprost.

Ustępstwo świadome: R2 wymaga KORESPONDENCJI cech (mały publiczny zbiór
kalibracyjny wspólny obu agentom), więc nie jest już „representation-
agnostic przez same słowa". Zostaje jednak reprezentacyjnie-agnostyczny
w mocnym sensie: **agenci zachowują PRYWATNE backbone'y**, przez sieć idą
tylko statystyki cech (jak w serii I) + jednorazowo cechy na PUBLICZNYM
zbiorze kalibracyjnym.

## Hipoteza

H-R2: dwa agenty o różnych backbone'ach mogą dzielić klasy, jeśli ich
przestrzenie cech wyrówna się ortogonalną transformacją Procrustesa
Ω: H_A → H_B, wyestymowaną na K=4 wspólnych klasach kalibracyjnych.
Payload klasy (statystyki spike-and-slab w H_A, jak w I) jest śniony
i transformowany przez Ω do H_B, po czym adoptowany istniejącym
`adopt_classes`. Ponieważ Ω zachowuje geometrię (izometria), struktura
wewnątrzklasowa — której zabrakło w R1b — przechodzi.

## Mechanizm (ZAMROŻONY)

**Wyrównanie (jednorazowe, na zbiorze kalibracyjnym):**
- K=4 klasy kalibracyjne z PUBLICZNYM zbiorem obrazów wspólnym obu
  agentom. A liczy cechy `H_A^cal` = backbone_A(obrazy_cal), B liczy
  `H_B^cal` = backbone_B(tych samych obrazów) → pary per-próbka.
- **Ortogonalny Procrustes:** Ω = U Vᵀ, gdzie U Σ Vᵀ = SVD(H_Aᵀ H_B)
  (bez centrowania; cechy nieujemne po ReLU). Ω ortogonalna [D×D]
  (R-mild: D=128 u obu). Zachowuje normy i kąty.
- Dobór z góry raportowany: rezyduum Procrustesa `‖H_A^cal Ω − H_B^cal‖`
  na held-out kalibracji (obserwacja jakości wyrównania).

**Adopcja klasy c (payload jak w I):**
- A wysyła statystyki spike-and-slab klasy c w H_A (~24 KB, jak
  `export_class_stats`).
- B śni próbki z tych statystyk w H_A → transformuje `@ Ω` → H_B →
  buduje statystyki H_B (`FeatureStatsKSparse`) → **istniejące
  `adopt_classes` BEZ ZMIAN** (projekcja + pody).

Nowe pliki: klasa Procrustesa + ścieżka adopcji w sąsiedztwie
`mars_collective_hetero.py`. `adopt_classes`, eksport, rdzeń NIETKNIĘTE.

## Split (jak R1b, dla porównywalności)

CIFAR-10: CAL=[0,1,2,3] dzielone (publiczny zbiór kalibracyjny do Ω),
OWN=[4,5] własne odbiorcy, ADOPT=[6,7]+[8,9] adoptowane (metryka).
Metryka = ACC klas ADOPTOWANYCH u odbiorcy (row_class_il[1:]),
recipient-relative. 5 seedów.

## Warianty (te same seedy 0..4)

- **CEILING** — homogeniczny, payload cech (== L2). Górna kotwica (~81%).
- **R0** — hetero, podłoga anchor-only (odniesienie z R1b).
- **R2_SANITY** — homogeniczny, ścieżka Procrustesa (Ω ≈ I przy wspólnym
  backbone). *** BRAMKA: adopted ≥ 65% ***.
- **R2_HET** — heterogeniczny R-mild (ten sam resnet18, różny seed
  projekcji 512→128), ścieżka Procrustesa. Wynik główny.
- **RIDGE_HET** (ablacja) — hetero, mapa liniowa (ridge, nieortogonalna)
  zamiast Procrustesa — czy ograniczenie ortogonalności kosztuje.

## Kryteria werdyktu (Z GÓRY) i BRAMKA

1. **BRAMKA (zamrożona):** **R2_SANITY < 65% adopted → R2 SFALSYFIKOWANE**
   (maszyneria wyrównania niszczy informację nawet przy Ω≈I) — poza
   CLAIMS, jak reguła R1b. Oczekiwane: SANITY blisko CEILING (Ω≈I).
2. **Po zdanej bramce — interpretacja heterogeniczności:**
   - **R2_HET vs CEILING:** SZUM → heterogeniczność DARMOWA (R-mild) =
     kolektyw representation-agnostic zrealizowany (headline Part III);
     SYGNAL− → zmierzona CENA wyrównania.
   - **R2_HET vs R0:** SYGNAL+ oczekiwany (wyrównanie cech niesie realną
     informację ponad podłogę anchorową).
   - **RIDGE_HET vs R2_HET:** czy ortogonalność (Procrustes) wystarcza,
     czy potrzebna pełna mapa liniowa (obserwacja).
3. **Obserwacja:** rezyduum Procrustesa na held-out kalibracji vs
   adopted-ACC (czy jakość wyrównania przewiduje sukces adopcji).

## Przewidywanie (zapisane przed runem)

R-mild: H_A, H_B to dwa liniowe obrazy tej samej 512 (modulo ReLU), więc
bliska-izometria istnieje — Procrustes powinien odzyskać większość →
R2_HET oczekiwany blisko CEILING (SZUM lub mała cena). To jest właśnie
test „czy wracamy do gry": jeśli R-mild przejdzie, R-hard (losowy ↔
pretrained, RÓŻNE wymiary — Procrustes uogólniony/rektangularny) jako
osobna pre-rejestracja. Jeśli nawet R-mild nie przejdzie mimo istniejącej
izometrii → wyrównanie cech przez wspólną kalibrację jest granicą.

## Plik, koszt, ryzyko

- Runner: `src/run_R2_procrustes.py` (nowy). Klasa `ProcrustesAlign`
  (SVD, zamknięta forma) + ścieżka adopcji przez Ω. Wzór R1b.
- Wynik: `results/R2_procrustes.json` (smoke: `_smoke`).
- Koszt: niski–średni (SVD [128×128] trywialne; cache cech z L; wzór R1b
  ~kilka min FULL). Smoke: 1 seed.
- Ryzyko: R-mild powinien przejść (izometria istnieje); realne ryzyko na
  R-hard (różne wymiary + różna treść) — osobna pre-rejestracja.

## Reżimy zasobów (zasada bez zmian)

R-hard (osobno) krzyżuje from-scratch z foundation — metryka zawsze
względem sufitu ODBIORCY w jego reżimie; kierunek transferu raportowany.
R2 (R-mild) jest wewnątrz jednego reżimu (oba pretrained), więc czysty.

## Instancjacja — RUNNER GOTOWY (2026-07-23, zielone światło Roberta)

Nowe/rozszerzone pliki (rdzeń I/L i `adopt_classes` NIETKNIĘTE):
- `src/mars_translate.py` — dopięta `ProcrustesAlign` (Ω = U Vᵀ z
  SVD(H_Aᵀ H_B), `transform`, `disparity`); smoke odzyskuje ortogonalną
  mapę (disp < 1e-6).
- `src/mars_collective_hetero.py` — dopięte `adopt_classes_maptransform`:
  śnij H_A z payloadu (statystyki cech jak w I) → `map_fn` (Ω lub mapa
  liniowa, clamp_min 0) → statystyki H_B → istniejące `adopt_classes`.
- `src/run_R2_procrustes.py` — split jak R1b; Ω liczona PER NADAWCA na
  cechach kalibracyjnych CAL (te same obrazy 512 przez oba backbone'y);
  warianty CEILING/R0/R2_SANITY/R2_HET/RIDGE_HET; bramka na R2_SANITY;
  obserwacja `disparity` (rezyduum wyrównania). RIDGE_HET = ta sama
  ścieżka z mapą liniową (ridge, λ=0.1) zamiast ortogonalnej.

Komendy (u Roberta): `python src/mars_translate.py` (smoke Procrustesa,
CPU) → `python src/run_R2_procrustes.py --smoke` →
`python src/run_R2_procrustes.py` (FULL). Wynik: `results/R2_procrustes.json`.

## Zasady

5 seedów, bramka 65% i progi zamrożone PRZED runem, min-par raportowany,
negatyw (w tym trafienie bramki) = wynik. Werdykty → DROGA_R_NOTATKI.md
(dopisek R2), CLAIMS/WHITEPAPER TYLKO po zdanej bramce. Runnera NIE
piszemy przed zatwierdzeniem tego planu przez Roberta.
