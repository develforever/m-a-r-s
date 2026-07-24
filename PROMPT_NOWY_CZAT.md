Kontynuujemy projekt M.A.R.S. (repo `C:\Users\robert\code\m-a-r-s`).

## Stan (2026-07-23)
Właśnie domknęliśmy **R2b — pierwszy pozytywny heterogeniczny kolektyw**
(CLAIMS 42). Mapa liniowa (ridge) H_A→H_B na 2–4 klasach kalibracyjnych
pozwala agentom o RÓŻNYCH backbone'ach dzielić klasy: R-mild (ten sam
resnet18, różny seed 512→128) odzyskuje **92% sufitu homogenicznego**
(79.98% vs 86.87%, świeże seedy 5–9, bramka ≥70% zdana). Łuk serii R:
kotwica-interlingua sfalsyfikowana (R1b, ORACLE 4%) → cechy zdrowe
(R2 gate) → ortogonalny Procrustes zły constraint (R2, 31%) → mapa
liniowa potwierdzona (R2b, 92%).

Wcześniej w tej sesji: **oś Part II (routing & zero-shot ceiling)
ostatecznie zamknięta** — G3 pokazało, że cechy pretrained ≈ losowe dla
kompozycyjności (G3−, granica paradygmatu, nie reprezentacji); seria G
domknięta. Wróciliśmy na główną oś CL (Part III).

Najpierw przeczytaj: `START_TUTAJ.md`, `PLAN_V2.md`, `CLAIMS.md`,
`DROGA_R_NOTATKI.md`, `DROGA_R2B_PLAN.md`. Kod serii R:
`src/mars_translate.py` (RidgeTranslator, ProcrustesAlign),
`src/mars_collective_hetero.py` (adopt_classes_maptransform),
`src/run_R2b_linear.py`.

## Zadanie: pre-rejestracja + implementacja R-hard
Napisz `DROGA_R_HARD_PLAN.md` (pre-rejestracja z twardymi progami PRZED
runem), potem — po moim zielonym świetle — runner. R-hard to właściwy
test rewolucji representation-agnostic:
- **Krzyżujemy backbone losowy-od-zera (from-scratch) z pretrained
  resnet18** — RÓŻNA treść informacyjna i RÓŻNE wymiary cech (np. losowy
  128 vs pretrained 512). Mapa wyrównania musi być **prostokątna**
  (ridge H_A(D_A)→H_B(D_B), nie kwadratowa) — uogólnij `RidgeTranslator`
  lub dodaj wariant; rdzeń I/L i `adopt_classes` NIETKNIĘTE.
- Uwaga na wejścia: agent losowy konsumuje surowe piksele (784/3072),
  pretrained konsumuje cache 512-d — każdy agent ma własny front
  sensoryczny; kalibracja per-próbka wymaga tych samych OBRAZÓW przez
  oba fronty.
- **Reżimy zasobów ROZDZIELONE** (zasada bez zmian): metryka = ACC klas
  adoptowanych względem LOKALNEGO sufitu ODBIORCY w jego reżimie; NIGDY
  liczba from-scratch vs foundation wprost. Kierunek transferu
  (losowy→pretrained vs pretrained→losowy) raportowany OSOBNO — asymetria
  to osobna obserwacja.
- Warianty à la R2b: CEILING (homogeniczny u odbiorcy), R0 (podłoga),
  R_HARD_SANITY (wspólny bb, mapa — kontrola maszynerii), R_HARD_HET
  (primary). Zdefiniuj bramkę z góry (np. R_HARD_HET ≥ X% sufitu
  odbiorcy — zaproponuj X, ja zatwierdzę). Sweep K jeśli sensowny.
- Przewidywanie zapisane z góry: R-hard jest znacznie trudniejszy niż
  R-mild (brak izometrii/wspólnej bazy — różna TREŚĆ cech), więc możliwy
  duży spadek lub negatyw; obie ścieżki domykają („kolektyw
  representation-agnostic działa też między różnymi reprezentacjami"
  vs „wymaga wspólnej bazy cech — granica").

## Zasady (bez zmian)
- **Eksperymenty uruchamiam TYLKO ja lokalnie (GPU u Roberta).** Ty:
  kod do repo + `py_compile` (sama składnia), NIGDY nie odpalaj
  eksperymentów w piaskownicy. Ja wklejam wyniki.
- 5 seedów; pary per-seed; próg szumu std+std; SYGNAL-parowy obok;
  **progi/bramki zamrożone PRZED runem**; negatyw = wynik.
- **Dyscyplina Q2c:** headline z ablacji nie wchodzi do CLAIMS bez
  pre-rejestrowanego potwierdzenia; przy niezdanej bramce — poza CLAIMS.
- Branch per seria (`droga-r`), main nietykalny; rdzeń I/L i
  `adopt_classes` NIETKNIĘTE (nowe pliki / nowe metody obok).
- Submisje (arXiv/TMLR/konferencje) WSTRZYMANE — nie proponuj, dopóki
  sam nie wrócę do tematu; etap = wyłącznie repo publiczne.
- Cytowania tylko zweryfikowane web searchem. CRLF-szum nie do commita.
- Po zdanej bramce: notatki (`DROGA_R_NOTATKI.md` dopisek), CLAIMS,
  WHITEPAPER (sekcja 15). Aktualizuj `START_TUTAJ.md`.

Zacznij od przeczytania plików stanu i napisania `DROGA_R_HARD_PLAN.md`
z propozycją progu bramki — czekaj na moje zielone światło przed runnerem.
