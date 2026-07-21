# Droga I4 — plan pre-rejestrowany: kolektyw niezaufany (wykryj i zapomnij)

Data pre-rejestracji: 2026-07-20 (branch `droga-i4`; main/v0.10 nietknięty).
Kontekst: ostatni kandydat mapy. Protokół I zakłada zaufanych nadawców;
w sieci otwartej payload może być zatruty („zatruta studnia" —
RELATED_WORK część E). I4 mierzy trzy rzeczy w kolejności: ile atak
SZKODZI, czy da się go WYKRYĆ bez danych zewnętrznych, i czy seria N
pozwala go NAPRAWIĆ. Narzędzia już istnieją: weryfikacja semantyczna
(kotwice a priori) i unlearn (N).

## Setup i ataki

Fashion, protokół I1 (odbiorca B: taski 0–3 z danych; adopcja klas
{8,9} z payloadów nadawcy A). Konfiguracja K1, 5 seedów. Payload
klasy 8 w trzech wariantach:
  clean : uczciwy payload klasy 8 od A
  swap  : payload klasy 9 zadeklarowany jako 8 (podmiana etykiety —
          nadawca podmienia własne klasy; najtańszy atak semantyczny)
  noise : statystyki policzone na cechach losowej mieszanki obrazów
          ze wszystkich zadań, zadeklarowane jako 8 (śmieć
          o realistycznych momentach)
Payload klasy 9 zawsze uczciwy (izolacja ataku).

## Pytanie 1 — SZKODA (adopcja bez obrony)

B adoptuje [8,9] z wariantowym payloadem 8. Miary vs clean (pary
per-seed): acc(8) (ofiara), acc(9) (sąsiad), śr. acc klas 0–7
(collateral). Werdykty SYGNAL−/parowy/SZUM per miara — mapa szkód.

## Pytanie 2 — DETEKCJA (bez danych zewnętrznych, PRZED pełną adopcją)

Dwa pre-rejestrowane detektory:
- D1 „spójność rang" (bez adopcji): centroid cech payloadu
  (E[x] = Σ_k w_k·p_k·μ_k) porównany z centroidami WŁASNYCH statystyk
  znanych klas B (0–7); korelacja rang (Spearman) listy podobieństw
  cechowych z listą podobieństw słownych kotwicy DEKLAROWANEJ do słów
  klas znanych. Uczciwy payload: struktura cech ~ struktura słów;
  swap/noise łamią zgodność. Ryzyko pre-rejestrowane: na losowym
  backbone zgodność cechy↔słowa może być słaba nawet dla clean.
- D2 „kanarek" (próbna adopcja na kopii): adoptuj payload na kopii B;
  sygnał = spadek śr. acc WŁASNYCH klas 0–7 (zatruta adopcja wymusza
  większy dryf projekcji niż uczciwa).
Kryterium (Z GÓRY, per detektor): SEPARACJA PEŁNA = w 5/5 seedów
wartość dla clean po właściwej stronie skrajnej wartości OBU ataków
(min-max bez przecięcia); inaczej separacja częściowa/brak (raport
wartości — to diagnostyka wykonalności, nie strojenie progu).

## Pytanie 3 — NAPRAWA (wykryj i zapomnij)

Po ataku swap (najgroźniejszy semantycznie): unlearn_light(8) →
ponowna adopcja uczciwego payloadu 8 → pełny wynik vs ścieżka clean
(pary per-seed). SZUM = pełna naprawa lightem (projekcja zniosła
zatrute sny); SYGNAL− = resztkowa szkoda → raport i wskazanie scrub
przed re-adopcją jako kandydat (NIE pre-rejestrowany tu).

## Kolejność uruchomień u Roberta

1. `python src/mars_cl_i4.py` (smoke jednostkowy, CPU, sekundy)
2. `python src/run_I4_untrusted.py --smoke`, potem FULL (~10 min)

Wyniki do DROGA_I4_NOTATKI.md; merge po komplecie i decyzji Roberta.
Po I4: przegląd całości pod v1.0.
