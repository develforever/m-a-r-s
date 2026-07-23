# M.A.R.S. — Plan po v0.11: przegląd pod v1.0 + kandydaci na dalsze serie

Data: 2026-07-23. Status: PROPOZYCJA (do zatwierdzenia przez Roberta).
Kontekst: mapa projektu WYCZERPANA (K·I·L·M·N·O·I4, tag v0.11).
Zastępuje mapę PLAN_GENERALNY.md (2026-07-11, historyczna — zrealizowana
w całości; zostaje jako dowód pre-rejestracji etapów).

Obowiązują bez zmian: runy WYŁĄCZNIE u Roberta (GTX 1050 Ti); 5 seedów,
pary per-seed, próg szumu std+std, SYGNAL-parowy obok; negatyw = wynik;
branch per seria, main nietykalny; submisje WSTRZYMANE — żaden punkt
tego planu nie proponuje submisji.

---

## CZĘŚĆ A — Przegląd całości pod v1.0 (bez runów; tylko dokumenty i porządek)

Cel: stan repo, w którym każda liczba headline ma jedno źródło prawdy,
każdy dokument mówi to samo, a tag `v1.0` oznacza „komplet twierdzeń
z mapy K→I4, zweryfikowany krzyżowo".

### A1. Audyt spójności dokumentów (największa praca)

- `WHITEPAPER.md`: nagłówek mówi „Draft v0.4", treść sięga v0.11.
  Serie K–I4 siedzą dziś w jednym akapicie-molochu w sekcji Roadmap
  (linia ~224). Do zrobienia: bump wersji do v1.0, rozbicie molocha na
  właściwe sekcje Części III (L: fork; M: horyzont; N: zapominanie;
  O: falsyfikacja konsolidacji; I4: kolektyw niezaufany), aktualizacja
  abstraktu o trójcę snu (uczy · dzieli · zapomina) i granice.
- `README.md`: „Code at v0.6" — nieaktualne; bump + weryfikacja, że
  każda komenda reprodukcji wskazuje istniejący runner w `src/` i że
  drabina wyników zawiera L/M/N/O/I4.
- `START_TUTAJ.md`: prompt otwierający (sekcja 2) opisuje stan sprzed
  M — przepisać na stan v1.0; „Następne kroki" wskazać na ten plik.
- `PLAN_GENERALNY.md`: dopisać na górze notę „ZREALIZOWANY W CAŁOŚCI,
  historyczny; aktualna mapa: PLAN_V1.md".
- `RELATED_WORK.md`: dopisać pozycjonowanie dla N (machine unlearning)
  i I4 (data poisoning / byzantine FL) — dwie krótkie rundy searchy,
  każde cytowanie zweryfikowane webem (zasada bez zmian).

### A2. Krzyżowa weryfikacja liczb (v1.0 = zaufanie do tabel)

Skrypt jednorazowy (nie dotyka kodu eksperymentów): przejść wszystkie
liczby headline w README/WHITEPAPER i porównać z `results/*.json`
(per-seed → średnia ± std). Raport rozbieżności do decyzji; zero
ręcznego przepisywania liczb bez tego sprawdzenia.

### A3. Porządek w repo

- Tagi: brakuje `v0.6` (jest v0.5 → v0.7) — sprawdzić, czy to
  przeoczenie i ewentualnie dotagować commit merge'a droga-i.
- Working tree: ~20 plików zmodyfikowanych — wg wcześniejszej decyzji
  CRLF-szum NIE idzie do commita; rozstrzygnąć raz: `.gitattributes`
  z `* text=auto eol=lf` + jednorazowa normalizacja, albo checkout
  odrzucający szum. Osobno obejrzeć `.github/workflows/deploy.yml`
  (zmiana realna czy szum?).
- Checklist repo publicznego (private/PLAN_PUBLIKACJI.md pkt 3):
  odhaczyć zrealizowane (README EN, LICENSE), domknąć brakujące —
  angielski indeks plików, skrypt pobierający GloVe (weryfikacja,
  że wektory nie są w repo).

### A4. Domknięcie: `CLAIMS.md` + tag

Jedna tabela wszystkich twierdzeń projektu (twierdzenie → seria →
werdykt → plik wyników), w tym granice i falsyfikacje (O, detekcja I4,
G2, MNIST). To jest „przegląd całości" w formie artefaktu. Po komplecie
A1–A4: merge, tag `v1.0`.

Kryterium wyjścia z A: (a) zero rozbieżności liczb (A2), (b) wszystkie
dokumenty wskazują v1.0, (c) working tree czysty, (d) CLAIMS.md istnieje.

---

## CZĘŚĆ B — Kandydaci na dalsze serie (po v1.0; pre-rejestracja per seria PRZED runami)

Wolne litery: P, Q, R (+ otwarte G2b). Kolejność = proponowany priorytet;
każda seria dostanie własny DROGA_*_PLAN.md z progami PRZED pierwszym runem.

### B1. Seria P — detekcja zatrucia na pretrained (naturalna kontynuacja I4)

Wejście: negatyw I4 (oba detektory bez separacji na losowym backbone)
z jawnie zapisanym kandydatem: „detekcja na pretrained". Infrastruktura
gotowa: `mars_cl_l.py` (resnet18-ImageNet) + `mars_cl_i4.py` (atak swap).
- **P1:** te same dwa detektory z I4 na cechach L. Hipoteza: spójność
  semantyczna kotwica↔statystyki jest mierzalna dopiero na cechach
  z semantyką. Werdykt: separacja atak/uczciwy (próg pre-rejestrowany,
  np. pełna separacja rozkładów lub AUC z progiem ustalonym z góry).
- **P2 (warunkowo, jeśli P1+):** polityka protokołu — auto-kwarantanna
  paczki nad progiem detektora; koszt fałszywych alarmów na uczciwych
  payloadach.
- Koszt: niski (runy rzędu minut, wzór I4). Ryzyko: kolejny uczciwy
  negatyw — też domyka twierdzenie („detekcja wymaga semantyki cech").

### B2. Seria Q — kolektyw na długim horyzoncie (I × M)

Dotąd kolektyw mierzony tylko na 10 klasach (I3, L2). M pokazało deficyt
strukturalny przy 100 klasach u JEDNEGO agenta.
- **Q1:** N=10 agentów × 10 klas CIFAR-100 (pretrained, kotwice 300d wg M)
  vs agent sekwencyjny 20 zadań vs sufit all-data. Pytanie: czy protokół
  wymiany snów płaci na 100 klasach więcej niż zmierzone −0.56pp z L2,
  i czy deficyt późny (−7.8pp) dotyczy też klas adoptowanych.
- To jest test skali „rewolucji" — najmocniejszy kandydat na nowy
  headline v1.x. Koszt: średni (CIFAR-100 na 1050 Ti — wzór M, godziny).

### B3. G2b — atrybuty z kodami korekcyjnymi (ECOC) — otwarte od G2

Negatyw G2 (kompozycyjność) z regułą strukturalną 3/3 sugeruje, że
słownik atrybutów wymaga dystansu kodowego. Hipoteza: kody ECOC nad
atrybutami przesuwają osiągalność klas złożonych. Werdykt wg wzoru G2.
Koszt: niski. Priorytet za P/Q, bo nie dotyka głównej narracji CL.

### B4. Seria R — kolektyw heterogeniczny (long shot; „rewolucja 2.0")

Wymiana snów między agentami o RÓŻNYCH backbone'ach (losowy ↔ pretrained)
— dziś protokół zakłada wspólny seed/backbone. Wymaga translacji
statystyk między przestrzeniami cech; jedyny wspólny układ odniesienia
to przestrzeń kotwic słownych. Ryzyko wysokie, zysk koncepcyjny
największy (protokół w pełni „representation-agnostic"). Wejście dopiero
po decyzji Roberta i osobnym szkicu mechanizmu translacji w
ARSENAL_PRZEOCZONYCH_NARZEDZI.md.

---

## Proponowana kolejność wykonawcza

1. CZĘŚĆ A w całości (przegląd → tag v1.0) — nic nie blokuje, zero runów.
2. Decyzja Roberta: P (tanio, domyka I4) → Q (headline skali) → G2b;
   R tylko po osobnej zgodzie.
3. Po każdej serii: notatki, aktualizacja WHITEPAPER/CLAIMS, tag v1.x.
