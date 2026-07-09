# Słownik pojęć projektu M.A.R.S.

Cel: wyjaśnić każde określenie, wzór i funkcję używane w projekcie tak, żeby dało się czytać notatki (`DROGA_D_NOTATKI.md`), plany (`D6B_PLAN.md`, `D7_PLAN.md`) i kod (`src/`) bez zgadywania. Definicje odnoszą się do faktycznego kodu, nie do podręcznikowych ogólników.

---

## 1. Architektura M.A.R.S. — z czego składa się system

**System (M.A.R.S. v2)** — jedna sieć złożona z trzech części: wspólnego backbone'u, routera i 10 podów. Obraz wchodzi, backbone zamienia go na wektor cech, router wybiera jeden pod, wybrany pod wydaje ostateczną klasyfikację. Kod: klasa `MarsV2System` w `src/mars_v2.py`.

**Backbone (kręgosłup, ekstraktor cech)** — pierwsza część sieci, wspólna dla routera i wszystkich podów. Zamienia surowy obraz (784 liczby = 28×28 pikseli) na krótszy wektor cech (`backbone_hidden`, np. 256 lub 128 liczb), który streszcza "co widać na obrazie". Liczony RAZ na próbkę i reużywany — to źródło oszczędności. Wersje: MLP (`Linear 784→256 + ReLU`), CNN (`mars_v2_cnn.py`, D6), odchudzone CNN (`mars_v2_slim.py`, D6b).

**Shared backbone (wspólny backbone)** — decyzja projektowa v2 (z eksperymentu C4): router i pody patrzą na TE SAME cechy, zamiast każdy liczyć swoje. Tańsze, ale ma konsekwencję odkrytą w D4/D5: skoro wszyscy widzą to samo, pody nie mają żadnej informacji, której router by już nie miał.

**Cechy / features** — wynik backbone'u: wektor `[B, backbone_hidden]` (B = liczba próbek w paczce). Metoda `features(x)`. "Reprezentacja" to inne słowo na to samo — sposób, w jaki sieć "widzi" dane. Główny wniosek serii D: jakość tej reprezentacji ogranicza routing bardziej niż sam algorytm routingu.

**Router (routing head)** — mała sieć decydująca, który pod obsłuży próbkę. Działa prototypowo: rzutuje cechy na krótki wektor (embedding, `emb_dim=32`), mierzy odległość euklidesową do 10 uczonych prototypów (po jednym na klasę) i wybiera najbliższy. Kod: `route_logits()` — zwraca `-dystans` jako logity (im bliżej prototypu, tym wyższy logit); `route()` — argmax, czyli twardy wybór.

**Prototyp** — uczony wektor `[emb_dim]` reprezentujący "wzorcowy" punkt klasy w przestrzeni embeddingu (`self.protos`, 10 sztuk). Router klasyfikuje przez "do którego wzorca najbliżej".

**Embedding** — niskowymiarowe rzutowanie cech (256→32) w którym mierzy się odległości. Sens: porównywanie w 32 wymiarach jest tańsze i wymusza kompresję informacji istotnej dla routingu.

**Pod (pod head, ekspert)** — mały klasyfikator (2 warstwy: `backbone_hidden → pod_hidden=24 → 10`) wyspecjalizowany w próbkach, które router do niego kieruje. Jest ich 10 (po jednym na klasę). Trzymane jako "stacked" tensory `[n_pods, in, out]` i liczone przez `torch.bmm` (mnożenie macierzy per-próbka) — wektoryzacja zamiast pętli po podach. Kod: `pod_forward(features, pod_ids)`.

**Oracle** — tryb diagnostyczny: "co by było, gdyby router był idealny". Zamiast decyzji routera podajemy każdej próbce jej PRAWDZIWĄ klasę jako wybór poda (`pod_forward(feats, yte)`). Oracle to sufit systemu — mówi, ile potrafią pody, gdy adresowanie jest bezbłędne. NIE jest metryką do publikacji, tylko narzędziem diagnostycznym.

**Luka router→oracle** — różnica `oracle_acc − system_acc`. Mierzy, ile accuracy tracimy przez błędy routera. Po D6 na Fashion: 98.10 − 91.99 = 6.11pp — to przestrzeń, w którą celuje D7.

---

## 2. Metryki — co mierzymy i jak czytać liczby

**routing_acc** — odsetek próbek, dla których router wskazał pod zgodny z prawdziwą klasą. Mierzy sam routing, bez podów.

**system_acc** — GŁÓWNA metryka: odsetek próbek poprawnie sklasyfikowanych przez cały łańcuch backbone → router → wybrany pod. To jest "wynik" każdego eksperymentu.

**oracle_acc** — accuracy przy idealnym routingu (patrz Oracle wyżej). Diagnostyka sufitu.

**pp (punkt procentowy)** — różnica dwóch procentów. 89.61% → 91.99% to +2.38pp (nie +2.38%). W projekcie wszystkie delty podajemy w pp, żeby nie mylić zmiany bezwzględnej ze względną.

**MAC (multiply-accumulate)** — jednostka kosztu obliczeń: jedna operacja "pomnóż i dodaj". Liczba MAC na próbkę mierzy, ile pracy wykonuje sieć przy jednej predykcji — to nasza miara "ceny energetycznej". Wzór dla warstwy Linear: `wejście × wyjście`. Dla konwolucji: `kanały_wy × kanały_we_na_grupę × k×k × wysokość_wy × szerokość_wy`. Kod: `mac_per_sample_top1()` — rozbija koszt na backbone (płacony ZAWSZE) + routing + pod. Konwencja: pomijamy ReLU/BatchNorm/MaxPool (koszt pomijalny wobec mnożeń).

**×MLP (mnożnik MAC)** — koszt wariantu podzielony przez koszt bazowego MLP (215 600 MAC). Pełny CNN z D6 = 19.7×; budżet "efektywnościowy" w D6b to ≤2.2×.

**n_params** — liczba parametrów (wag) modelu. Używana do uczciwego zrównywania budżetów przy porównaniach (np. v1 vs v2 w D1), ale to MAC — nie parametry — mierzy koszt inferencji.

**Retention** — miara z D6b: `delta_wariantu / delta_pełnego_D6`. Mówi, jaki procent zysku z pełnego CNN przetrwał odchudzenie. Retention 60% przy MAC 2× zamiast 20× = mocny argument efektywnościowy.

---

## 3. Metodologia — jak odróżniamy wynik od szumu

**Seed (ziarno losowości)** — liczba inicjalizująca generator losowy (`torch.manual_seed(seed)`). Ten sam seed = identyczne losowe wagi startowe i kolejność danych = powtarzalny wynik. Różne seedy = niezależne "światy równoległe" tego samego eksperymentu.

**Multi-seed (5 seedów)** — każdy eksperyment serii D biegnie 5 razy (seedy 0–4). Jeden run może być fuksem; pięć pozwala policzyć średnią i rozrzut. Porównania robimy PARAMI per seed (wariant seed 3 vs baseline seed 3), co eliminuje część szumu.

**mean ± std** — średnia i odchylenie standardowe z 5 seedów. Std liczone z poprawką Bessela (dzielenie przez n−1, nie n) — poprawna estymacja rozrzutu przy małej próbce. Wzór w kodzie: `var = Σ(v−mean)² / (n−1)`.

**Próg szumu** — umowne kryterium projektu: `std_baseline + std_wariantu`. Delta większa od tej sumy = raczej realny efekt; mniejsza = nieodróżnialna od losowości.

**SYGNAL+ / SYGNAL− / SZUM** — trzy możliwe werdykty każdego eksperymentu, ustalane Z GÓRY (przed zobaczeniem wyników, żeby nie naginać interpretacji): delta ponad próg szumu w górę / w dół / w granicach szumu. Kluczowa zasada projektu: każdy z trzech wyników jest wartościowy — SZUM i SYGNAL− też domykają hipotezy (D4 = SZUM, D5 = SYGNAL−, D6 = SYGNAL+).

**Baseline (punkt odniesienia)** — model, z którym porównujemy nowy wariant. Musi być trenowany tym samym kodem, na tych samych seedach i epokach — inaczej porównanie jest "jabłka z gruszkami". Dlatego D6b czyta baseline'y per-seed z `D6_cnn_backbone.json` zamiast liczyć je od nowa.

**Smoke test** — szybki przebieg (1 seed, mniej epok) sprawdzający, że kod działa i liczby są sensowne, ZANIM spalimy godzinę GPU na pełny run. Nie służy do wniosków — tylko do wykrycia błędów.

**Krzywa / front Pareto** — zbiór wariantów "niezdominowanych": takich, dla których nie istnieje inny wariant jednocześnie dokładniejszy I tańszy. Odpowiada na pytanie "ile accuracy da się kupić za dany budżet MAC". Kod: `pareto_front()` w `mars_v2.py`.

**Hipoteza falsyfikowalna** — sformułowanie eksperymentu tak, żeby dało się go PRZEGRAĆ: z góry określamy, jaki wynik obala hipotezę. Standard serii D (sekcje "Kryterium werdyktu" w planach).

---

## 4. Elementy sieci neuronowych używane w projekcie

**MLP (perceptron wielowarstwowy)** — sieć z warstw Linear: każde wyjście to ważona suma WSZYSTKICH wejść. Nie wie nic o strukturze 2D obrazu — piksel lewego górnego rogu i środkowy są dla niej "równie daleko".

**Warstwa Linear (w pełni połączona, FC)** — `y = Wx + b`: mnożenie wektora wejść przez macierz wag plus przesunięcie. Koszt: `in × out` MAC.

**CNN (sieć konwolucyjna)** — sieć używająca konwolucji. Kluczowa przewaga na obrazach: lokalność (filtr patrzy na małe okno, np. 3×3) i współdzielenie wag (ten sam filtr przesuwa się po całym obrazie), co daje inwariancję na przesunięcia — "kieszeń to kieszeń, niezależnie gdzie jest". To dlatego D6 podniósł routing: cechy z CNN lepiej opisują treść obrazu.

**Konwolucja (Conv2d)** — przesuwanie filtra k×k (u nas 3×3) po obrazie; w każdej pozycji: iloczyn skalarny filtra z oknem pikseli. Parametry: kanały wejściowe/wyjściowe, rozmiar filtra, stride, padding.

**Kanał** — "warstwa" mapy cech. Obraz wejściowy ma 1 kanał (skala szarości); po konwolucji z 32 filtrami mamy 32 kanały — każdy to mapa odpowiedzi jednego filtra ("gdzie są krawędzie pionowe", "gdzie rogi" itp.). Zapis `ch=(32,64)`: pierwszy blok daje 32 kanały, drugi 64.

**Padding** — dopełnienie brzegów obrazu zerami przed konwolucją (`padding=1` przy 3×3), żeby mapa wyjściowa miała ten sam rozmiar co wejściowa.

**Stride (krok)** — co ile pikseli przesuwa się filtr. `stride=1`: filtr liczy każdą pozycję. `stride=2`: co drugą w obu osiach → mapa wyjściowa 2× mniejsza w każdej osi, a konwolucja liczy 4× MNIEJ pozycji. Dlatego wariant S3 w D6b jest tani: downsampling robi sama konwolucja, "za darmo".

**MaxPool2d (pooling)** — zmniejszanie mapy przez branie maksimum z okien 2×2 (28×28 → 14×14). Zero parametrów, zero MAC (konwencja), ale konwolucja PRZED poolingiem liczy pełną rozdzielczość. Różnica stride vs maxpool w D6b: maxpool zachowuje operację "max" (odporność na drobne przesunięcia), stride ją traci — S3 testuje, czy to "max" jest warte 4× kosztu conv.

**Depthwise separable (konwolucja rozdzielona)** — rozbicie pełnej konwolucji c1→c2 na dwa tanie kroki: depthwise (każdy kanał filtrowany OSOBNO swoim filtrem 3×3, `groups=c1`) + pointwise (konwolucja 1×1 mieszająca kanały). Koszt spada z `c1·c2·9·HW` do `c1·9·HW + c1·c2·HW`. Trik znany z MobileNet; w D6b to wariant S4.

**BatchNorm2d** — normalizacja aktywacji w paczce (średnia 0, wariancja 1 per kanał, plus uczone skalowanie). Stabilizuje i przyspiesza trening CNN. W liczeniu MAC pomijana.

**ReLU** — funkcja aktywacji `max(0, x)`: przepuszcza dodatnie, zeruje ujemne. Wprowadza nieliniowość — bez niej cała sieć byłaby jednym wielkim mnożeniem macierzy.

**Logity** — surowe wyjścia sieci przed zamianą na prawdopodobieństwa: dowolne liczby, większa = "bardziej ta klasa". W routerze logit = `-dystans do prototypu`.

**Softmax** — zamiana logitów na prawdopodobieństwa: `p_i = exp(z_i) / Σ exp(z_j)` (wszystkie dodatnie, sumują się do 1). Używany do confidence routera i w agregacji top-2.

**Argmax / argmin** — indeks największej / najmniejszej wartości. `argmax(logity)` = wybrana klasa/pod. Uwaga techniczna: argmax jest nieróżniczkowalny — gradient przez niego nie płynie, dlatego w treningu end-to-end wybór poda jest w `torch.no_grad()`.

**Confidence (pewność)** — najwyższe prawdopodobieństwo z softmaxu routera (`probs.max()`). Steruje adaptive compute: wysoka pewność = można iść na skróty.

**CrossEntropyLoss (entropia krzyżowa, CE)** — funkcja straty do klasyfikacji: kara `-log(p_prawdziwej_klasy)`. Pewna i poprawna predykcja → strata ~0; pewna i błędna → strata ogromna.

**Adam** — optymalizator (ulepszony spadek gradientu z adaptacyjnym krokiem per parametr). Standard w projekcie, `lr=0.001`.

**lr (learning rate)** — długość kroku optymalizatora. Za duży = trening rozjeżdża się, za mały = ślimaczy.

**Epoka** — jedno pełne przejście przez zbiór treningowy. Seria D: 30 epok (smoke: 8).

**Batch (paczka)** — porcja próbek przetwarzana naraz (u nas 512). Gradient liczony na paczce = kompromis między szybkością a stabilnością.

**Backpropagation (propagacja wsteczna)** — algorytm liczenia gradientów straty względem wszystkich wag (od wyjścia wstecz do wejścia). `loss.backward()` w kodzie.

**Freeze (zamrożenie)** — wyłączenie uczenia części sieci: `requires_grad = False`. Zamrożone wagi nie zmieniają się. Kluczowe w train_phased (faza 2) i w planie D7 (lekcja z D5: nie wolno ruszać wspólnej reprezentacji).

---

## 5. Trening i wnioskowanie — mechanizmy specyficzne dla projektu

**train_phased (trening dwufazowy, D1a)** — standard serii D. Faza 1: backbone + router uczą się klasyfikować (CE na logitach routingu). Faza 2: backbone i router ZAMROŻONE; pody uczą się na próbkach, które router im REALNIE przydziela (argmax), nie na idealnym przydziale. Sens: pod widzi w treningu dokładnie ten rozkład danych, który dostanie w teście — bez train/test mismatch.

**Trening na oracle (błąd z C4)** — historyczna pułapka: pody uczone na prawdziwej etykiecie (idealny przydział) dawały oracle=100%, ale pod nigdy nie widział próbek błędnie skierowanych przez router. D1 to naprawił — stąd "uczciwy" oracle ~95–98%, nie 100%.

**train_end_to_end (D1b)** — alternatywa jednofazowa: wspólny loss `L = L_router + α·L_pods`, wszystko uczy się naraz. Wynik D1: nieodróżnialny od phased — używamy phased dla czystości porównań.

**Adaptive Compute / forward_adaptive (Etap 3)** — trójpoziomowe wnioskowanie sterowane pewnością routera: pewność ≥ θ_high → **Early Exit** (odpowiada sam router, 0 podów); pomiędzy → **top-1** (1 pod); pewność < θ_low → **top-2** (2 pody). Backbone + routing płacone zawsze; pody to koszt dodatkowy 0/1/2×. Wniosek z D1: na shared backbone strukturalnie słabe, bo pod to ~3% kosztu — nie ma czego oszczędzać.

**Early Exit (EE)** — skrót: dla łatwych próbek predykcją jest klasa wskazana przez router, pody pomijane.

**Top-1 / Top-2** — ile podów odpowiada. Top-2: dwa najlepsze wg routera, wyjścia uśrednione z wagami = ich confidence (`w0·out0 + w1·out1`).

**θ (theta_high / theta_low)** — progi pewności rozdzielające trzy poziomy. Sweep po siatce progów → punkty do krzywej Pareto.

**Sweep** — systematyczne przejście po siatce wartości hiperparametru i pomiar każdej konfiguracji.

**Distillation (destylacja, D5 — wynik negatywny)** — technika "uczeń naśladuje nauczyciela": router miał się uczyć od podów (upodabniać swoje logity do ich wyjść). Na shared backbone AKTYWNIE SZKODZI (−2.5pp): dostrajanie backbone'u pod routing zaburza przestrzeń cech, na której skalibrowane są zamrożone pody. Lekcja: nie ruszać wspólnej reprezentacji.

**Consultation / ensemble (D4 — wynik: szum)** — przy niskiej pewności zapytaj top-k podów i uśrednij (ensemble = komitet modeli). Zysk 0.00pp, bo pody widzą te same cechy co router — ich "opinie" nie wnoszą nowej informacji. Redundancja zupełna.

**Sufit reprezentacji** — centralny wniosek D1+C3+D4+D5: routing nie poprawi się od sprytniejszego algorytmu, bo cała informacja dostępna w cechach jest już wykorzystana. Poprawić można tylko same cechy (→ D6) albo dodać NOWY kanał informacji (→ D7).

---

## 6. Pojęcia planu D7 (predictive coding)

**Predictive coding (kodowanie predykcyjne)** — teoria z neuronauki (Helmholtz, Friston): mózg nie odczytuje bodźców biernie, tylko generuje HIPOTEZĘ "co tam jest" i porównuje ją z wejściem; w górę płynie tylko błąd predykcji. W D7: pod, który najlepiej "wyjaśnia" próbkę (najmniejszy błąd rekonstrukcji), jest prawdopodobnie właściwym ekspertem.

**Dekoder / rekonstrukcja** — mała sieć doklejona do poda, która próbuje ODTWORZYĆ wejście (lub cechy) z wewnętrznej reprezentacji poda. Jakość odtworzenia = miara "jak dobrze ten ekspert rozumie tę próbkę".

**Błąd rekonstrukcji (MSE)** — `e = ||target − rekonstrukcja||²` (średnia kwadratów różnic). Sygnał routingowy w D7: `routing_pc = argmin_i e_i`. Kluczowa nadzieja: to informacja NIEZALEŻNA od logitów klasyfikacji, więc może przełamać sufit z D4.

**D7-hard / D7-fuse / D7-iter** — warianty użycia sygnału: hard = routuj czystym argmin błędu; fuse = dodaj `λ·(−e)` do logitów routera (λ strojona sweepem); iter = router wybiera top-k kandydatów, błąd rekonstrukcji rozstrzyga między nimi (kontrola kosztu — tylko k dekoderów).

---

## 7. Zbiory danych i sprzęt

**MNIST** — 70 000 obrazów 28×28 ręcznie pisanych cyfr (0–9). "Łatwy" zbiór — nasz system ma na nim ~99%, praktycznie sufit; służy jako kontrola.

**Fashion-MNIST** — ten sam format, ale 10 kategorii ubrań (koszulka, torebka, but...). Trudniejszy — tu jest całe pole gry projektu (luka do oracle, delty wariantów).

**784** — liczba pikseli obrazu (28×28) po spłaszczeniu do wektora. `N_IN` w kodzie.

**CUDA / GPU** — obliczenia na karcie graficznej (u Ciebie GTX 1050 Ti); `device="cuda"`. Trening sieci to głównie mnożenia macierzy, które GPU robi równolegle.

**bb_h, emb_dim, pod_hidden (hiperparametry)** — rozmiary warstw ustalane przez nas (nie uczone): szerokość wektora cech (256 MLP / 128 CNN), wymiar embeddingu routingu (32), warstwa ukryta poda (24).

---

## 8. Mapa eksperymentów (gdzie czego szukać)

Seria A — pomiary bazowe routera i podów (pliki `A*_WYNIKI.md`). Seria B — porównania enkoderów; B1b pokazał przewagę CNN (+3.5pp routing na Fashion) — nasienie D6. Seria C — przełomy architektoniczne: C1 adaptive compute, C4 shared backbone. Seria D — systematyczna walidacja v2: D1 baseline multi-seed (v2 ≈ v1), D4 consultation (SZUM), D5 distillation (SYGNAL−), D6 CNN backbone (SYGNAL+, +2.38pp Fashion), D6b odchudzanie CNN (w toku), D7 predictive coding (plan). Notatki bieżące: `DROGA_D_NOTATKI.md`; plany: `D6B_PLAN.md`, `D7_PLAN.md`.
