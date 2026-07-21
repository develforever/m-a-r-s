# Droga O — plan pre-rejestrowany: konsolidacja snem

Data pre-rejestracji: 2026-07-20 (branch `droga-o`; main/v0.9 nietknięty).
Motywacja (odkrycie uboczne N1c): projekcja odbudowana od zera na snach
pozostałych klas okazała się LEPSZA niż sekwencyjna (~+0.7pp po
odjęciu efektu maski) — odbudowa ze snów to „joint na snach", bez
dryfu kolejności zadań. Pytanie O1: czy jeden „głęboki sen" po
zakończeniu sekwencji (odbudowa/douczenie projekcji na snach WSZYSTKICH
widzianych klas) podnosi wynik końcowy — bez confoundu maski,
na pełnych benchmarkach? Jeśli tak: czwarta funkcja snu (ochrona ·
transfer · odbudowa · KONSOLIDACJA) i możliwy nowy best.

## Setup

Dwa benchmarki, TE SAME seedy i konstrukcje co bazy (pary legalne;
sanity: przed konsolidacją finalny wiersz musi odtworzyć bazę):
- Fashion: konfiguracja K1 (sparse_k16 × 300d); baza par:
  `results/K1_sparse300.json` fashion_sp16_300 (79.23 ± 0.73; sufit 81.16).
- CIFAR-n: konfiguracja L1 (pretrained ReducedBackbone, 50d); baza par:
  `results/L1_pretrained.json` l1_seq (74.69 ± 0.69; sufit 77.23).

Konsolidacja (`src/mars_cl_o.py`, `consolidate()`): sny 2000/klasę ze
wszystkich widzianych klas, nauka projekcji epochs_proj=15; pody
NIETKNIĘTE (czysta atrybucja do projekcji). Dwa tryby:
  o1_reinit   : reinicjalizacja projekcji (deterministycznie z seeda)
                + nauka od zera na snach (GŁÓWNY — poparty pomiarem N1c)
  o1_finetune : douczenie istniejącej projekcji sekwencyjnej na snach
                (może zachować informację z realnych cech sekwencji)

## Kryteria (Z GÓRY, class-IL, per dataset, pary per-seed vs baza)

- GŁÓWNE: o1_reinit vs baza — SYGNAL+ / SYGNAL-parowy+ = konsolidacja
  działa (przy SYGNAL+ na obu benchmarkach: nowy domyślny krok
  końcowy protokołu i prawdopodobny nowy best); SZUM/SYGNAL− = efekt
  z N1c nie przenosi się poza confound maski (też wynik).
- RÓWNORZĘDNE: o1_finetune vs baza (ten sam schemat werdyktu).
- Obserwacje: reinit vs finetune (pary); dystans do sufitów
  (81.16 / 77.23); sanity determinizmu (baza odtworzona ≤0.01pp).

## Ryzyka pre-rejestrowane

1. Sen to model gęstości, nie dane: odbudowa ze snów traci informację
   z realnych cech — SYGNAL− realny, zwłaszcza na CIFAR (K0/L1:
   trudniejsze cechy). Werdykt symetryczny.
2. Pody były uczone na próbkach routowanych STARĄ projekcją — nowa
   projekcja zmienia routing; niedopasowanie pod↔routing może zjeść
   zysk. Odnotowane; ewentualna pełna konsolidacja (projekcja + pody)
   to kandydat O1b, NIE pre-rejestrowany tutaj.
3. N1c robił odbudowę na 9 klasach bez usuwanej; tu 10 klas — efekt
   może się różnić.

## Kolejność uruchomień u Roberta

1. `python src/mars_cl_o.py` (smoke jednostkowy, CPU, sekundy)
2. `python src/run_O1_consolidation.py --smoke`, potem FULL (~5 min)

Wyniki do DROGA_O_NOTATKI.md; merge po komplecie i decyzji Roberta.
