# Droga I4 — notatki robocze

Plan: `DROGA_I4_PLAN.md` (pre-rejestracja 2026-07-20). Runy: Robert,
lokalnie, 5 seedów, konfiguracja K1.

## I4 (ZAKOŃCZONE, 20.07.2026): mapa szkód zmierzona; detekcja NIE
## separuje (wynik negatywny); naprawa pełna na naprawianej klasie,
## ale szkoda ROZLEWA SIĘ na klasę współadoptowaną → I4b

Plik: `src/run_I4_untrusted.py`; wyniki: `results/I4_untrusted.json`.
Czas: 84 s.

**P1 — SZKODA (pary vs clean):**

| Atak | acc(8) ofiara | acc(9) sąsiad | własne 0–7 |
|---|---|---|---|
| swap (9-as-8) | **−77.58 SYGNAL−** | **−94.38 SYGNAL−** | +0.58 (parowy+) |
| noise (śmieć) | −10.94 SYGNAL− | −0.66 SZUM | −0.93 (parowy−) |

1. **Swap niszczy OBIE adoptowane klasy:** cechy klasy 9 dostają
   sprzeczne targety (kotwica 8 z zatrutego payloadu + kotwica 9
   z uczciwego) w jednej adopcji — konflikt rozstrzyga się
   katastrofalnie dla obu (acc9 ~1–6%). Promień rażenia ataku =
   cała paczka adopcyjna, nie jedna klasa.
2. **Klasy własne odbiorcy są odporne** (±1pp) — sen (rehearsal)
   broni starych klas nawet przy zatrutej adopcji. Szkoda jest
   ograniczona do adoptowanych.
3. **Noise tworzy „klasę-pochłaniacz":** acc(8)=85% mimo payloadu bez
   informacji o klasie 8 — szeroki basen atrakcji łapie testowe próbki
   ofiary (wysoki recall, zła precyzja); szkoda rozproszona i mała.
   Uwaga interpretacyjna: per-class acc nie mierzy tu wiedzy o klasie.

**P2 — DETEKCJA: OBA detektory bez pełnej separacji (wynik negatywny).**
D1 (spójność rang): clean ~+0.05, swap ~−0.06, noise ~+0.31 — noise ma
WYŻSZĄ spójność niż clean (śmieciowy centroid koreluje ze wszystkim);
przecięcia w obu atakach. Pre-rejestrowane ryzyko (słaba zgodność
cechy↔słowa na losowym backbone) zmaterializowane. D2 (kanarek):
kierunkowo swap<clean<noise, przecięcia w 5 seedach. Detekcja
payloadu na losowym backbone pozostaje OTWARTA — uczciwie: protokół I
w obecnej formie wymaga zaufanych nadawców albo mocniejszej
reprezentacji (kandydat: D1 na pretrained — cechy L mają semantykę).

**P3 — NAPRAWA (wykryj-i-zapomnij, zakres = tylko klasa 8):**
acc(8): **PEŁNA NAPRAWA** (−0.04, szum) — unlearn_light + re-adopcja
przywraca ofiarę w 100% (projekcja zniosła zatrute sny). ALE acc(9)
pozostaje zniszczone (−94.38) — naprawa nie objęła klasy zatrutej
POŚREDNIO. **Zasada: zasięg naprawy musi pokrywać zasięg szkody** —
stąd I4b (dopisek PRZED runem): naprawa pełnej paczki adopcyjnej.
