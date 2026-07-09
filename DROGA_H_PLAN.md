# Droga H — "Reboot matematyczny" (plan; propozycje zewn. agenta 08.07.2026,
# zweryfikowane i uszeregowane)

Status: PLAN. Uruchomienie po whitepaper v0.3 (dyscyplina: najpierw domknąć
serię F/G w dokumencie, potem nowe hipotezy).

## Weryfikacja propozycji (uczciwa, względem naszych pomiarów)

| Propozycja | Ocena | Dlaczego |
|---|---|---|
| OWM (rzutnik ortogonalny) | **PRIORYTET — H1** | Publikowana metoda (Zeng 2019), gwarancja ścisła dla warstw LINIOWYCH — a nasza plastyczność to małe warstwy liniowe. Celuje w zmierzony dryf projekcji (F3b: F=15.8pp). P⊥ 128×128 = tani. |
| VSA/HDC binding | **H2, z korektą** | "Nigdy nie kolidują" = przesada: quasi-ortogonalność, interferencja rośnie z liczbą kontekstów; uczenie wspólnej macierzy wymaga ortogonalizacji aktualizacji (= OWM; komplementarne, nie niezależne). Motywacja GPU słuszna (E4). |
| LSH+Morton+TMU | **H3, demo** | "0 MAC" nieścisłe (16 bitów LSH = 16×D MAC; darmowe dopiero na FPGA). Routing to ~0.1% kosztu inferencji (zmierzone) — zerowanie go nic nie daje. Realna wartość: router bez wag nie zapomina (już w planie F) + efektowne demo WebGPU. |

## H1 — OWM na projekcji semantycznej (following F3b)

**Hipoteza:** rzutnik ortogonalny do podprzestrzeni cech starych klas
(liczony "we śnie" z kowariancji — spójne z F3) eliminuje resztkowy dryf
projekcji, którego sen parametryczny nie domknął.

- Mechanizm: aktualizacja projekcji ΔW ← ΔW·P⊥, gdzie P⊥ z kowariancji
  cech klas widzianych (aktualizowana po każdym zadaniu; wzór rekurencyjny
  RLS-podobny, bez przechowywania danych).
- Warianty: (a) OWM zamiast snu, (b) OWM + sen (podejrzenie: synergia —
  sen daje negatywy dla NOWYCH słów, OWM chroni STARE kierunki).
- Kryterium (z góry): class-IL Fashion vs f3b_combo (75.68%): Δ > próg
  szumu → SYGNAL+; cel ambitny: > replay (76.97) i zbliżenie do g1_all
  (80.45). Forgetting < 15.8pp. Pomiar plastyczności: acc ostatniego
  zadania vs numer zadania (null-space się kurczy — znane ryzyko OWM).

## H2 — Binding zamiast podów (superpozycja)

**Hipoteza:** pody "holograficzne" (wejście⊕kontekst przez JEDNĄ gęstą
macierz) osiągają izolację wiedzy porównywalną z fizycznymi podami przy
gęstości obliczeniowej monolitu (lekcja E4: GPU kocha gęste).

- Uczciwy test pojemności: ACC/forgetting vs liczba klas przy RÓWNYCH
  parametrach z podami stacked; zmierzyć interferencję superpozycji
  (to jest główna niewiadoma, nie zakładać ortogonalności).
- Wymiar: sweep D ∈ {1k, 4k, 10k} (quasi-ortogonalność rośnie z D).
- Prawdopodobnie wymaga H1 (aktualizacje ortogonalne) — kolejność H1→H2.

## H3 — Morton+TMU router (demo deployment, po H1/H2)

Bez werdyktu naukowego (routing nie jest wąskim gardłem — zmierzone).
Wartość: WebGPU demo "router w fixed-function" + wątek "router bez wag
nie zapomina" (spójny z F2 z planu Drogi F). Zestawić z z-buffer argmin
z arsenału.

## Relacja do "Memory OS" (odłożone przez użytkownika, odnotowane)

Nasze notatki po F4 wskazują "mocniejszy zamrożony backbone" jako drogę
wzwyż — embeddingi modeli fundamentalnych to to samo pod inną nazwą.
Dwa piętra projektu, nie sprzeczność: mechanizmy CL (F/G/H) są agnostyczne
wobec źródła cech; wizja "from scratch" (H1–H3) walczy o dolne piętro.
Decyzja o piętrze górnym — po wynikach H.
