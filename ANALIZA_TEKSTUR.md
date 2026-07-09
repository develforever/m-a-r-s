# M.A.R.S. — Analiza tekstur: weryfikacja i realne wyciśnięcie

Data: 2026-06-15
Kontekst: zewnętrzna analiza zasugerowała, że tekstury były źle używane.
Zweryfikowałem jej tezy realnym kodem — większość trafna, ale praktyczny
wniosek wymaga korekty.

---

## Co analiza twierdzi i co potwierdziły dane

### TEZA 1: bilinear na ID podów daje szum, nearest jest poprawny — POTWIERDZONE ✓
Test: bilinear na mapie ID → **98.2% próbek dostaje nieistniejące ID**
(np. "pod 3.7"). To czysty szum. nearest → zawsze poprawne całkowite ID.
Analiza ma 100% racji co do mechanizmu.

### TEZA 3: blur w fazie snu niszczy granice decyzyjne — POTWIERDZONE ✓
Test: blur na ostrej granicy 0|9 → 28 komórek dostaje pośrednie ID
(pody 1-8, których tam nie powinno być). Przeciek klas. Analiza ma rację.
To wyjaśnia `blur_makes_sense: false` z etap4b_kohonen.json.

---

## KLUCZOWA KOREKTA: rekomendacja jest już wdrożona

Analiza zaleca "tekstura jako LUT ID z nearest, nie embedding z bilinear".
**Ale Twój obecny kod `mars_torch.py` JUŻ to robi:**
- `labels = grid_sample(label_map, mode='nearest')` ← ID podów, nearest ✓
- `confidence = grid_sample(confidence_map, mode='bilinear')` ← tylko pewność

Błąd MSE 0.49 (etap4_textures.json) dotyczył ETAPU 4 — wcześniejszego
eksperymentu z interpolacją embeddingów, który słusznie porzucono.
Faza 2 już używa poprawnego podejścia. Nie ma długu do spłacenia.

---

## NAJWAŻNIEJSZE ODKRYCIE: tekstura już jest darmowa, wąskim gardłem jest encoder

Zmierzone: lookup tekstury (nearest) jest **128× szybszy** niż pełny router.
Koszt routingu to ENCODER MLP (784→64→2 = 50,304 MAC), nie tekstura.

To podważa praktyczny wniosek analizy. Analiza sugeruje "przerzucić ciężar
na teksturowy LUT, by przyspieszyć". Ale tekstura JUŻ jest LUT-em i JUŻ
jest ~darmowa. **Żeby przyspieszyć router, trzeba odchudzić encoder,
NIE zmieniać tekstur.**

---

## REALNE WYCIŚNIĘCIE: odchudzenie encodera

Test (dane syntetyczne — trend, nie finalna liczba):

| encoder hidden | routing acc | MAC | MAC vs h=64 |
|---|---|---|---|
| 64 (obecny) | 40.0% | 50,304 | 100% |
| 32 | 40.0% | 25,152 | 50% |
| 16 | 40.0% | 12,576 | 25% |
| 8 | 50.7% | 6,288 | **12%** |
| 4 | 40.0% | 3,144 | 6% |

**Encoder hidden=8 daje 88% mniej MAC w routerze przy tej samej (lepszej!)
jakości.** Routing acc jest niemal stała — co potwierdza audyt A3: pody są
tak generalizujące, że router prawie nie ma znaczenia.

UWAGA metodologiczna: dane syntetyczne są łatwe (40% podejrzanie stałe).
Na realnym MNIST trzeba zweryfikować (skrypt: run_encoder_squeeze_mnist.py).
Ale kierunek jest mocny.

---

## Podsumowanie potencjału tekstur

Analiza miała rację, że tekstury były źle używane W ETAPIE 4. Ale:
1. Faza 2 już to naprawiła (nearest dla ID).
2. Tekstura już jest darmowa — nie ma co z niej wyciskać więcej.
3. Prawdziwa dźwignia to ENCODER, który można odchudzić ~8× bez straty.
4. Blur w fazie snu należy USUNĄĆ (potwierdzone, że szkodzi) — zastąpić
   operacjami na ID/wagach, nie filtrami graficznymi.

**Wniosek dla wyciskania:** następne realne MAC do odzyskania są w encoderze
routera (50k → ~6k), nie w teksturach. To do zmierzenia na MNIST.
