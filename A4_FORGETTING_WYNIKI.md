# M.A.R.S. -- Krok A4: Catastrophic forgetting (Split-MNIST) -- WYNIKI

Data: 2026-06-16
Device: CUDA (NVIDIA GeForce GTX 1050 Ti)
Plik wynikowy: `results/A4_forgetting.json`

---

## Pytanie

Czy modularna architektura M.A.R.S. chroni stara wiedze przy nauce nowej
na prawdziwym MNIST? To boisko domowe M.A.R.S. -- i tu monolit jest
z natury slaby.

Split-MNIST: Task A = digits 0-4, Task B = digits 5-9.
Sekwencja: train A -> train B -> mierz ile A zostalo.

---

## Wyniki (usrednione po 3 seedach)

| Scenariusz | A przed | A po | B | retencja |
|---|---|---|---|---|
| **Monolith (baseline)** | 99.4% | **0.0%** | 98.6% | **0.0%** |
| **M.A.R.S. replay** | 99.1% | **97.4%** | 95.3% | **98.4%** |
| **M.A.R.S. expand** | 99.1% | **94.1%** | 63.2% | **95.0%** |

**Delta retencji: +98.4 pkt proc. (M.A.R.S. replay vs monolit)**

---

## Interpretacja

### Monolit: kompletne zapomnienie (0.0%)
Po nauce digits 5-9 monolit CALKOWICIE zapomina 0-4. Nie 50% (losowo),
nie 20% (zgadywanie 1 z 5) -- dosLOWNIE 0.0%. Siec nauczyla sie
generowac TYLKO klasy 5-9 i nigdy nie odpowiada 0-4. To jest gorsze
niz na XOR/AND (tam bylo 50%), bo z 5 klasami jest wiecej do zapomnienia.

### M.A.R.S. replay: niemal idealna ochrona (98.4%)
Pody 0-4 zamrozone, pody 5-9 wytrenowane, router ponownie nauczony na
A+B. Retencja A: 97.4% (z 99.1% -- minimalny spadek). Nauka B: 95.3%.
Lacznie: 96.4% na wszystkich 10 cyfrach.

To jest najsilniejszy wynik projektu: M.A.R.S. uczy sie NOWEGO zadania
(95.3%) prawie BEZ poswiecania STAREGO (97.4%). Monolit musi wybrac
jedno: albo pamietaj stare (nie ucz nowego), albo ucz nowe (zapomnij stare).
M.A.R.S. robi oba naraz.

### M.A.R.S. expand: retencja swietna, nauka B slaba (63.2%)
Retencja A jest swietna (95.0% -- pody zamrozone, encoder zamrozony,
stare protos zamrozone). ALE nauka B jest slaba (63.2%) -- zamrozony
encoder nie generalizuje dobrze na nowe klasy. Nowe prototypy sa
trenowane w przestrzeni embeddingu zoptymalizowanej dla digits 0-4,
co ogranicza routing nowych klas.

**Wniosek o expand:** encoder jest waskim gardlem continual learning.
Jesli go zamrozimy, chronimy stare routing idealnie, ale nowe klasy
nie maja dobrej reprezentacji. Rozwiazanie: encoder pre-trainowany na
szerszej dystrybucji, albo lekki fine-tuning z regularyzacja (EWC na
encoderze). To jest dobrze zdefiniowany problem do rozwiazania w przyszlosci.

---

## Porownanie z wczesniejszymi wynikami

| Test | Monolit retencja | M.A.R.S. retencja | Delta |
|---|---|---|---|
| Etap 2 (XOR->AND, NumPy) | 50% | 95% | +45 pp |
| Faza 2 (MNIST, stary system) | ~0% | 25.7% | +25.7 pp |
| **A4 (MNIST, specjalisci)** | **0%** | **98.4%** | **+98.4 pp** |

Postep jest dramatyczny. Stary system (redundantne pody, router-atrapa)
dawal tylko 25.7% retencji. Nowy system (prawdziwi specjalisci +
ProtoRouter 96.7%) daje 98.4%. Roznica wynika z dwoch rzeczy:
1. Router faktycznie kieruje do wlasciwego poda (96.7% vs 40%)
2. Pody sa prawdziwymi specjalistami (zamrazanie ma sens)

---

## Co to znaczy dla projektu

A4 to najsilniejsza karta M.A.R.S. Monolit w tym tescie jest
bezbronny (0% retencji), a M.A.R.S. osiaga 98.4%. To jest przewaga
STRUKTURALNA -- wynika z samej architektury (osobne pody = osobna wiedza),
nie z sprytnego triku.

Uczciwa uwaga: ta przewaga istnieje w literaturze (modularnosc jako
ochrona przed zapominaniem to znany mechanizm w continual learning).
Ale to, ze mamy go ZMIERZONEGO na MNIST z pelnym pipelinem (router +
specjalisci + FastPods), z 3 seedami, z porownaniem do monolitu --
to jest solidny, obronialny wynik.

---

## Pliki

- `src/run_A4_forgetting.py` -- skrypt A4
- `results/A4_forgetting.json` -- wyniki
- Trzy scenariusze: monolit, replay (router A+B), expand (bez starych danych)
