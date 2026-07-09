# Droga G — Grounding językowy (plan; pomysł użytkownika, 06.07.2026)

Status: PLAN, uruchomienie PO werdykcie F1 (G buduje na infrastrukturze F).
Pomysł źródłowy (Robert): uczyć model liter/wyrazów/zdań i łączyć słowną
reprezentację z obrazami — "myślimy o rzeczach, opisując je".

## 0. Mapowanie na pole (żeby wiedzieć, na czyich barkach stoimy)

- Obraz↔słowo we wspólnej przestrzeni: CLIP (skala poza nami, idea nie).
- Obraz → pojęcia-słowa → klasa: Concept Bottleneck Models (CBM).
- Klasa z opisu cech bez przykładów: zero-shot learning przez atrybuty.
- Litery→wyrazy→zdania: curriculum/kompozycyjność (poza zasięgiem 1050 Ti
  w wersji językowej pełnej; uczciwie odnotowane).

Nasz wkład (jeśli wyjdzie): połączenie semantycznych kotwic z modularnym CL —
prototypy, które istnieją PRZED danymi, nie mogą dryfować i nie wymagają uczenia.

## G1 — Prototypy semantyczne (words as anchors)

**Hipoteza:** projekcja obraz→przestrzeń wektorów słów (GloVe/fastText, 50d,
zamrożone) pozwala routować do prototypów-słów z accuracy porównywalną
z uczonymi prototypami — a w CL nowa klasa dostaje prototyp ZA DARMO
(jej wektor słowa), zanim system zobaczy pierwszy obraz.

- Fashion: nazwy klas mają realną semantykę (shirt/coat/sneaker...).
- Trening: tylko projekcja 128→50 (backbone S2 zamrożony jak w F1);
  strata: dopasowanie embeddingu obrazu do wektora słowa jego klasy
  (cosine/MSE) — reszta architektury F1 bez zmian.
- Werdykt: G1 vs F1a (prototypy z obrazów): ACC class-IL w progu szumu =
  słowa są równie dobrymi kotwicami (a tańszymi); dodatkowy pomiar
  zero-shot: routing do klasy, której projekcja nigdy nie widziała.
- Bonus diagnostyczny: hierarchia grup z odległości słów vs grupy z E1
  (zbieżność = język zna nasze konfuzje).

## G2 — Concept bottleneck (myślenie cechami-słowami)

**Hipoteza:** warstwa pojęć (8–12 binarnych atrybutów słownych dla Fashion:
ma-rękawy, zakrywa-tułów, jest-obuwiem, ma-uchwyt, sięga-kostki...) jako
przestrzeń routingu daje (a) interpretowalne decyzje, (b) KOMPOZYCYJNY CL:
klasa zdefiniowana samym opisem atrybutów, bez ani jednego przykładu.

- Test kompozycyjny (główny): trenuj na 9 klasach, 10. klasę zdefiniuj
  wyłącznie wektorem atrybutów → zmierz acc na niewidzianej klasie.
  Baseline: losowy wybór / najbliższa widziana klasa.
- Ryzyko uczciwe: atrybuty ręczne = wąskie gardło jakości; przy 10 klasach
  atrybutów jest mało — wynik może być kruchy. Kryterium z góry:
  acc niewidzianej klasy > 3× losowe = sygnał kompozycyjności.

## G3 (horyzont) — curriculum litery→wyrazy→sceny

Pełna wersja językowa poza zasięgiem sprzętu; realna ścieżka na dziś:
EMNIST (litery) jako dodatkowa domena CL dla F/G (transfer między domenami
obraz-cyfry/litery/ubrania na wspólnym zamrożonym backbone). Decyzja po G1/G2.

## Kolejność

F1 werdykt → G1 (tani: jedna projekcja) → G2 (kompozycyjność) → decyzja o G3.
Zasady serii bez zmian (5 seedów, kryteria z góry, negatywny = wynik).
