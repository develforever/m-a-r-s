# D7 — Dialogue / Predictive Coding router↔pody (plan eksperymentu)

Data: 2026-07-06
Status: PROPOZYCJA (do uruchomienia po zamknięciu D6)
Styl: seria D — twarda hipoteza, metryka, baseline, kryterium werdyktu, ryzyka.

---

## 1. Motywacja (skąd to się bierze)

Dwa niezależne wątki się tu spotykają:

**(a) Wynik badawczy.** D4 (consultation) i D5 (distillation) pokazały zgodnie:
na shared backbone router osiągnął SUFIT reprezentacji. Pytanie kolejnych podów
o te same cechy (D4) = 0.00pp. Distillation pod→router (D5) = −2.5pp. Wniosek:
**nie da się poprawić routingu dokładając informację, której już nie ma w
cechach backbone.** Każdy pod widzi ten sam wektor, więc jego "opinia" jest
redundantna względem rozkładu routingu.

**(b) Intuicja biologiczna (pytanie użytkownika, 06.07).** Mózg nie odczytuje
obrazu jednokierunkowo. Percepcja = wnioskowanie (Helmholtz, predictive coding,
Friston): górne obszary generują HIPOTEZĘ "co tam prawdopodobnie jest",
a w dół płynie sygnał BŁĘDU — różnica między hipotezą a wejściem. To pętla,
nie feedforward. Klasyczny CNN (i obecny MARS v2) jest jednokierunkowy.

**Połączenie:** żeby przełamać sufit z D4, dialog router↔pody musi wnosić
NOWE źródło informacji. Predictive coding je daje: **błąd rekonstrukcji**.
Jeśli pod potrafi zrekonstruować wejście (albo cechy) lepiej niż inne pody,
to jest to dowód "ten ekspert wyjaśnia tę próbkę" — sygnał NIEZALEŻNY od
logitów klasyfikacji, więc nieredundantny względem tego, co router już wie.

---

## 2. Hipoteza D7

> Routing oparty na BŁĘDZIE REKONSTRUKCJI podów (predictive coding) niesie
> informację, której nie ma w logitach routera, więc podnosi routing_acc
> (a przez to system_acc) na Fashion-MNIST — tam, gdzie router jest wąskim
> gardłem — mimo że D4 (ensemble na tych samych cechach) nic nie dał.

Falsyfikowalna: jeśli routing po błędzie rekonstrukcji ≈ routing po logitach
(w granicach szumu między seedami), to znaczy, że rekonstrukcja też jest
funkcją tych samych cech i nie wnosi nic nowego → sufit potwierdzony także
dla tego kanału → zamykamy Drogę 3 od strony routingu.

---

## 3. Mechanizm (minimalna zmiana architektury v2)

Dokładamy KAŻDEMU podowi lekki dekoder (predictive head). Reszta v2 bez zmian
(reużycie `features`, `route_logits`, `pod_forward`, `train_phased`, `evaluate`).

```
features h = backbone(x)                    # jak teraz, liczone RAZ
Dla poda i:
    rekonstrukcja  x_hat_i = decoder_i(pod_hidden_i(h))   # "hipoteza" poda i
    błąd           e_i = || target - x_hat_i ||^2         # sygnał predictive coding
routing_pc = argmin_i e_i                    # ekspert, który najlepiej wyjaśnia
```

Warianty targetu rekonstrukcji (do porównania):
- **D7a — rekonstrukcja cech**: target = h (backbone_hidden). Tani, spójny z v2.
- **D7b — rekonstrukcja wejścia**: target = x (784). Bliżej "obrazu za mgłą",
  droższy, ale bogatszy sygnał.

Warianty użycia sygnału:
- **D7-hard**: routuj czystym argmin błędu (zastępuje router).
- **D7-fuse**: routuj sumą `route_logits + λ·(−e)` (dialog: router proponuje,
  błąd rekonstrukcji koryguje). λ dobierane na walidacji (sweep jak w D4).
- **D7-iter** (pełny dialog, 2 rundy): router wybiera top-k kandydatów →
  liczymy błąd rekonstrukcji tylko dla nich → finalny wybór po błędzie.
  Koszt kontrolowany (tylko k dekoderów, nie wszystkie).

Trening: faza 1 = jak teraz (backbone + router). Faza 1.5 (NOWA) = trening
dekoderów na rekonstrukcję przy zamrożonym backbone. Faza 2 = pody na realnym
routingu (jak w train_phased). Dekodery NIE zmieniają backbone (kluczowa lekcja
z D5 — nie wolno ruszać wspólnej reprezentacji).

---

## 4. Metryki (te same co seria D, dla ciągłości)

Per seed × {MNIST, Fashion-MNIST}, 5 seedów, mean ± std (Bessel n−1):
- `routing_acc` — baseline (logity) vs D7a/D7b × hard/fuse/iter
- `system_acc` — główna metryka
- `oracle_acc` — sufit diagnostyczny
- `recon_mse` — jakość rekonstrukcji (sanity: czy dekodery w ogóle działają)
- `MAC` — koszt: fixed (backbone+routing) + koszt dekoderów (k lub n razy).

Baseline: D1c / D6 (ten sam kod, routing po logitach). Porównanie CZYSTE —
różni je tylko kanał routingu.

---

## 5. Kryterium werdyktu (z góry, żeby nie oszukiwać po fakcie)

| Wynik na Fashion (Δ system vs baseline) | Werdykt |
|---|---|
| Δ > +(std_base + std_D7) | **SYGNAL+** — predictive coding łamie sufit D4. Duży wynik do paperu: dialog > ensemble. |
| \|Δ\| ≤ (std_base + std_D7) | **SZUM** — rekonstrukcja to ta sama informacja. Sufit reprezentacji potwierdzony na kolejnym kanale. Domyka Drogę 3. |
| Δ < −(...) | **SYGNAL−** — rekonstrukcja aktywnie myli routing (jak D5). Też wynik: pokazuje granicę predictive coding na shared backbone. |

KAŻDY z trzech wyników jest publikowalny — to jest siła tego projektu.

---

## 6. Ryzyka i pułapki

1. **Redundancja (najprawdopodobniejsza).** Dekoder też jest funkcją h, więc
   błąd rekonstrukcji może być deterministyczną funkcją logitów → 0 zysku,
   jak D4. Mitygacja: D7b (rekonstrukcja x, nie h) daje sygnał częściowo spoza
   przestrzeni routingu. Jeśli i to nic nie da — to mocny dowód sufitu.
2. **Koszt MAC.** n dekoderów = n× koszt rekonstrukcji. Dlatego D7-iter (tylko
   top-k). Raportować MAC uczciwie — zysk acc za wzrost compute to nie sukces
   sam w sobie (ta sama uwaga co CNN backbone w D6: +3pp za ~20× MAC).
3. **Zaburzenie backbone.** Nie trenować dekoderów wstecz do backbone (lekcja
   D5). Backbone zamrożony w fazie 1.5.
4. **Kolejność z D6.** Jeśli D6 (CNN backbone) da SYGNAL+, D7 uruchomić na
   CNN backbone, nie MLP — bogatsze cechy = uczciwszy test rekonstrukcji.

---

## 7. Kolejność wykonania

1. Zamknąć D6 (pełny 5-seed) → decyzja MLP vs CNN backbone jako fundament.
2. `mars_v2_pc.py` — podklasa z decoder heads (wzór jak mars_v2_cnn.py:
   dziedziczy wszystko, dokłada tylko dekodery + `route_pc()`).
3. `run_D7_predictive_coding.py` — smoke (1 seed) → pełny (5 seedów),
   warianty D7a/D7b × hard/fuse, sweep λ dla fuse.
4. Wynik do `DROGA_D_NOTATKI.md` + aktualizacja mapy wersji
   (v2 Shared → v3 Dialogue = D7).

---

## 8. Dlaczego to jest właściwy krok w stronę "rewolucji"

Nie dlatego, że przypomina mózg — samo podobieństwo nic nie gwarantuje
(backprop jest niebiologiczny i wygrywa; Etap 1 pokazał, że uczenie lokalne
było bardziej mózgowe, ale droższe). Dlatego, że to KONKRETNY, MIERZALNY
mechanizm, który celuje w dokładnie zdiagnozowane wąskie gardło (router,
D1+C3+D4+D5) nowym kanałem informacji (błąd rekonstrukcji). Jeśli zadziała —
mamy wynik "dialog bije ensemble". Jeśli nie — domykamy dowód sufitu
reprezentacji i wiemy, że dźwignia jest wyłącznie w cechach (D6), nie w
routingu. Tak czy siak projekt idzie do przodu na twardych liczbach.
