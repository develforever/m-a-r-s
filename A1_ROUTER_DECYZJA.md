# M.A.R.S. — Droga A, krok A1: zastąpienie tekstury — DECYZJA + DO POMIARU

Data: 2026-06-15

---

## Czym była tekstura i dlaczego ją zastępujemy

Tekstura pełniła trzy role naraz (router, pamięć, semantyka) — i to był błąd.
- Jako pamięć/semantyka: poległa (Etap 4, MSE 0.49).
- Jako router: lookup był darmowy, ale wąskim gardłem był encoder (audyt).
- Jedyne realne ograniczenie: WYMUSZA 2D (tekstura jest płaska) → router 44%.

Wniosek: tekstura nie jest źródłem przewagi, a jej 2D blokuje specjalizację.
Zastępujemy ją mechanizmem pozwalającym na bottleneck ≥4D.

---

## Trzy rozważone zamienniki

| Zamiennik | Plus | Minus | Werdykt |
|---|---|---|---|
| Mały MLP-router | prosty, sprawdzony, tani | to zwykły MoE gating | **kandydat główny** |
| Wielokanałowa tekstura | zachowuje TMU | kruchy, zysk marginalny | odrzucony |
| Prototypy / k-NN | bliski SOM z dok., interpretowalny, łatwo rozszerzalny | mniej standardowy | **kandydat badawczy** |

Odrzucamy wielokanałową teksturę: skoro lookup i tak jest darmowy (audyt),
komplikowanie tekstur nie daje nic poza ryzykiem. Zostają dwa czyste podejścia.

---

## Co zbudowano (src/routers_v2.py)

- `MLPRouter` — gating input→bottleneck→softmax. Bottleneck konfigurowalny.
- `ProtoRouter` — routing prototypowy (centroid per pod, najbliższy wygrywa).
  Ma `add_pod()` — naturalne rozszerzanie o nowe klasy (continual learning).
  Zachowuje ducha "mapy topologicznej" SOM z dokumentów, bez płaskiej tekstury.

Oba zweryfikowane: forward, route, MAC, add_pod działają.
Na danych syntetycznych oba dają 100% (ale to łatwe dane — MNIST rozstrzygnie).

---

## DO POMIARU (Twój GPU)

```
.venv\Scripts\python.exe src\run_A1_router.py
```

Zmierzy 4 warianty (MLP 4D/8D, Proto 8D/16D) na PRAWDZIWYM MNIST.
Punkt odniesienia: stara tekstura dawała ~44%.

**Próg sukcesu: >85% routing accuracy.** To warunek konieczny dla kroku A2
(wąskie pody), bo przy specjalizacji system accuracy = router accuracy.

Czego się spodziewać:
- Jeśli któryś router przebije 85% → Droga A otwarta, idziemy do A2.
- Jeśli żaden → trzeba mocniejszego encodera/większego bottleneck przed A2.
- Ciekawostka do obserwacji: czy ProtoRouter (bliski SOM) dorówna MLP.
  Jeśli tak — zachowujemy narrację "topologiczną" z dokumentów.

---

## Następny krok (po pomiarze)

A2: wąskie pody (każdy uczony tylko na swoich danych, wąskie wyjście).
Wtedy router PRZESTAJE być atrapą — staje się sercem systemu.
To jest moment, w którym M.A.R.S. staje się prawdziwie modularny.
