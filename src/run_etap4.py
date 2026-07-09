"""
run_etap4.py — Etap 4: Tekstury GPU jako pamięć (weryfikacja hipotezy).

KLUCZOWE PYTANIE: Czy bilinear filtering na embeddingach daje
SEMANTYCZNIE sensowną interpolację, czy tylko matematyczny szum?

Testy:
  1. Semantyczna interpolacja: umieść 2 embeddingi na teksturze,
     interpoluj między nimi bilinearnie, porównaj z "poprawną" interpolacją
     (lerp w przestrzeni embeddingów). Czy to to samo?

  2. Operacje snu: czy blur/erozja/mipmap zachowują strukturę wiedzy?

  3. Benchmark: texture lookup vs matmul (orientacyjny, CPU-emulacja).

  4. Werdykt: TAK (tekstury mają sens) lub NIE (to tylko metafora).

Uruchom:
    cd src
    python run_etap4.py
"""

import json
import os
from datetime import datetime
import numpy as np

from cortex_texture import NeuroTexture, benchmark_lookup_vs_matmul

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─── Test 1: Bilinear = Lerp? ───────────────────────────────────────────────

def test_bilinear_equals_lerp():
    """
    HIPOTEZA: Bilinear interpolation na teksturze jest MATEMATYCZNIE
    równoważna liniowej interpolacji (lerp) embeddingów.

    Dowód/obalenie:
      Umieszczamy embedding A w punkcie (0.2, 0.5) i embedding B w (0.8, 0.5).
      Robimy bilinear lookup w punkcie (0.5, 0.5) — środek.
      Porównujemy z: 0.5*A + 0.5*B (lerp).

    Jeśli są identyczne → bilinear = lerp (w granicach dyskretyzacji).
    Jeśli różne → bilinear wprowadza artefakty z dyskretyzacji siatki.
    """
    print("\n┌─ TEST 1: Bilinear interpolation = Lerp embeddingów? ──────────┐")

    tex = NeuroTexture(size=128)  # wyższa rozdzielczość = mniej błędów siatki

    # Dwa "embeddingi" (wektory 8D zakodowane w kanale R wzdłuż osi X)
    rng = np.random.default_rng(42)
    emb_A = rng.normal(0, 1, 8).astype(np.float32)
    emb_B = rng.normal(0, 1, 8).astype(np.float32)

    # Umieść na teksturze
    y_line = 0.5  # ten sam wiersz
    x_A, x_B = 0.2, 0.8

    tex.write_embedding(x_A, y_line, emb_A)
    tex.write_embedding(x_B, y_line, emb_B)

    # Interpolacja w różnych punktach
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    errors = []

    print(f"  Embedding A (w x=0.2): {np.round(emb_A, 3)}")
    print(f"  Embedding B (w x=0.8): {np.round(emb_B, 3)}")
    print(f"\n  {'alpha':>6} {'lerp[0:3]':>20} {'bilinear[0:3]':>20} {'MSE':>10}")
    print(f"  {'─'*6} {'─'*20} {'─'*20} {'─'*10}")

    for alpha in alphas:
        x_interp = x_A + alpha * (x_B - x_A)
        # Prawdziwy lerp
        lerp_result = (1 - alpha) * emb_A + alpha * emb_B
        # Bilinear z tekstury
        bilinear_result = tex.bilinear_lookup_row(x_interp, y_line, 8)
        # Błąd
        mse = float(np.mean((lerp_result - bilinear_result) ** 2))
        errors.append(mse)
        print(f"  {alpha:>6.2f} {str(np.round(lerp_result[:3], 3)):>20} "
              f"{str(np.round(bilinear_result[:3], 3)):>20} {mse:>10.6f}")

    mean_mse = np.mean(errors)
    max_mse = np.max(errors)

    # Werdykt
    # Bilinear na siatce dyskretnej nie jest DOKŁADNIE lerp, bo:
    # 1. Embeddingi nie lądują dokładnie na pikselach (kwantyzacja)
    # 2. Sąsiednie piksele mogą "wyciekać" do interpolacji
    # Ale jeśli MSE < 0.01, to jest funkcjonalnie równoważne.
    threshold = 0.01
    is_equivalent = mean_mse < threshold

    print(f"\n  Średni MSE: {mean_mse:.6f}")
    print(f"  Max MSE:    {max_mse:.6f}")
    print(f"  Próg:       {threshold}")
    if is_equivalent:
        print(f"  WERDYKT:    ✓ Bilinear ≈ Lerp (MSE < próg)")
        print(f"              Interpolacja na teksturze JEST sensowna semantycznie,")
        print(f"              o ile embeddingi leżą na tej samej osi siatki.")
    else:
        print(f"  WERDYKT:    ✗ Bilinear ≠ Lerp (artefakty dyskretyzacji)")
        print(f"              Interpolacja teksturowa NIE jest wiernym odpowiednikiem")
        print(f"              lerp w ciągłej przestrzeni embeddingów.")

    return {
        "test": "bilinear_vs_lerp",
        "texture_size": 128,
        "embedding_dim": 8,
        "mean_mse": round(mean_mse, 8),
        "max_mse": round(max_mse, 8),
        "threshold": threshold,
        "is_equivalent": is_equivalent,
        "errors_per_alpha": {str(a): round(e, 8) for a, e in zip(alphas, errors)},
    }


# ─── Test 2: 2D interpolacja (prawdziwy bilinear, nie 1D) ───────────────────

def test_2d_bilinear_semantics():
    """
    Bardziej wymagający test: 4 embeddingi w rogach kwadratu.
    Bilinear w środku powinien dać średnią z 4 rogów.
    Testuje PRAWDZIWY 2D bilinear (4 narożniki), nie uproszczony 1D.
    """
    print("\n┌─ TEST 2: 2D bilinear — 4 narożniki → interpolacja ───────────┐")

    tex = NeuroTexture(size=64)

    # 4 "koncepcje" w rogach kwadratu na teksturze
    concepts = {
        "cat":     np.array([1.0, 0.0, 0.5, 0.0], dtype=np.float32),
        "dog":     np.array([0.8, 0.2, 0.5, 0.0], dtype=np.float32),
        "car":     np.array([0.0, 1.0, 0.0, 0.5], dtype=np.float32),
        "truck":   np.array([0.0, 0.8, 0.0, 0.7], dtype=np.float32),
    }
    # Pozycje na teksturze (ćwiartki)
    positions = {
        "cat":   (0.25, 0.25),
        "dog":   (0.75, 0.25),
        "car":   (0.25, 0.75),
        "truck": (0.75, 0.75),
    }

    # Zapisz na teksturze (używamy kanału R dla prostoty — 1 wartość per piksel)
    for name, (x, y) in positions.items():
        tex.write(x, y, concepts[name][0], priority=concepts[name][2])

    # Interpolacja w środku (0.5, 0.5) — powinno dać średnią
    center = tex.bilinear_lookup(0.5, 0.5)
    expected_r = np.mean([concepts[n][0] for n in concepts])

    # Interpolacja między cat i dog (0.5, 0.25)
    cat_dog_mid = tex.bilinear_lookup(0.5, 0.25)
    expected_cat_dog_r = (concepts["cat"][0] + concepts["dog"][0]) / 2

    print(f"  Koncepcje w rogach:")
    for name, vec in concepts.items():
        print(f"    {name:>6}: R={vec[0]:.1f} (w pozycji {positions[name]})")

    print(f"\n  Interpolacja w środku (0.5, 0.5):")
    print(f"    Oczekiwane R (średnia 4 rogów): {expected_r:.4f}")
    print(f"    Bilinear R:                     {center[0]:.4f}")
    print(f"    Błąd:                           {abs(center[0] - expected_r):.6f}")

    print(f"\n  Interpolacja cat↔dog (0.5, 0.25):")
    print(f"    Oczekiwane R (średnia cat+dog):  {expected_cat_dog_r:.4f}")
    print(f"    Bilinear R:                      {cat_dog_mid[0]:.4f}")
    print(f"    Błąd:                            {abs(cat_dog_mid[0] - expected_cat_dog_r):.6f}")

    # Problem: na dyskretnej siatce 64x64, pozycja (0.5, 0.5) może nie trafić
    # dokładnie między piksele z wartościami. To KLUCZOWY artefakt.
    err_center = abs(center[0] - expected_r)
    err_mid = abs(cat_dog_mid[0] - expected_cat_dog_r)

    # Sensowność semantyczna: "coś między kotem a psem" powinno mieć R≈0.9
    # "coś między zwierzęciem a pojazdem" powinno mieć R≈0.45
    semantic_ok = (cat_dog_mid[0] > center[0])  # cat-dog bliżej "zwierząt" niż centrum

    print(f"\n  Semantyczna spójność: cat↔dog R > centrum R? "
          f"{'✓ TAK' if semantic_ok else '✗ NIE'}")
    print(f"    (zwierzęta mają R≈0.9, pojazdy R≈0.0, interpolacja powinna to respektować)")

    return {
        "test": "2d_bilinear_semantics",
        "error_center": round(float(err_center), 6),
        "error_cat_dog_mid": round(float(err_mid), 6),
        "semantic_ordering_correct": bool(semantic_ok),
        "note": "Artefakty wynikają z dyskretyzacji siatki — na wyższej "
                "rozdzielczości błąd maleje."
    }


# ─── Test 3: Operacje snu — czy zachowują strukturę? ────────────────────────

def test_sleep_operations():
    """
    Testujemy: po operacjach snu (blur, erozja) czy struktura wiedzy
    jest zachowana (kolejność siły skojarzeń, proporcje).
    """
    print("\n┌─ TEST 3: Operacje snu — zachowanie struktury wiedzy ──────────┐")

    tex = NeuroTexture(size=32)

    # Zapisz 3 "koncepcje" o różnych wagach
    tex.write(0.2, 0.5, weight=1.0, priority=0.9)   # silna
    tex.write(0.5, 0.5, weight=0.5, priority=0.5)   # średnia
    tex.write(0.8, 0.5, weight=0.1, priority=0.2)   # słaba

    before = tex.data[:, :, 0].copy()
    occupancy_before = tex.occupancy()

    # Cykl snu: erozja + konsolidacja
    tex.sleep_erosion(decay=0.1)
    tex.sleep_consolidation_numpy(kernel_size=3)

    after = tex.data[:, :, 0].copy()
    occupancy_after = tex.occupancy()

    # Sprawdź: czy kolejność siły jest zachowana?
    ix_strong = int(0.2 * 31)
    ix_medium = int(0.5 * 31)
    ix_weak = int(0.8 * 31)
    iy = int(0.5 * 31)

    val_strong = after[iy, ix_strong]
    val_medium = after[iy, ix_medium]
    val_weak = after[iy, ix_weak]

    ordering_preserved = val_strong > val_medium > val_weak

    print(f"  Przed snem: occupancy={occupancy_before*100:.1f}%")
    print(f"  Po śnie:    occupancy={occupancy_after*100:.1f}%")
    print(f"\n  Wartości po śnie (kanał R):")
    print(f"    Silna (x=0.2):  {val_strong:.4f}")
    print(f"    Średnia (x=0.5): {val_medium:.4f}")
    print(f"    Słaba (x=0.8):  {val_weak:.4f}")
    print(f"  Kolejność zachowana? {'✓ TAK' if ordering_preserved else '✗ NIE'}")

    # Mipmap: generalizacja
    mip1 = tex.mipmap(level=1)
    mip2 = tex.mipmap(level=2)
    print(f"\n  Mipmapping (generalizacja):")
    print(f"    Level 0: {tex.size}x{tex.size}")
    print(f"    Level 1: {mip1.shape[0]}x{mip1.shape[1]}")
    print(f"    Level 2: {mip2.shape[0]}x{mip2.shape[1]}")
    print(f"    Informacja zachowana w mipmap L1: {np.sum(np.abs(mip1[:,:,0]) > 0.001)} aktywnych pikseli")

    # Plastyczność: symulacja kostnienia
    used_mask = np.zeros((32, 32), dtype=bool)
    used_mask[iy, ix_strong] = True  # silny slot był używany
    plasticity_before = tex.data[iy, ix_strong, 3]
    tex.sleep_plasticity_decay(used_mask)
    plasticity_after = tex.data[iy, ix_strong, 3]

    print(f"\n  Metaplastyczność:")
    print(f"    Slot silny — plastyczność przed: {plasticity_before:.2f}, po: {plasticity_after:.2f}")
    print(f"    (częste użycie → kostnienie → ochrona przed nadpisaniem)")

    return {
        "test": "sleep_operations",
        "occupancy_before_pct": round(occupancy_before * 100, 2),
        "occupancy_after_pct": round(occupancy_after * 100, 2),
        "ordering_preserved": bool(ordering_preserved),
        "mipmap_levels": [tex.size, mip1.shape[0], mip2.shape[0]],
        "plasticity_decay_works": plasticity_after < plasticity_before,
    }


# ─── Test 4: Benchmark lookup vs matmul ─────────────────────────────────────

def test_benchmark():
    """Porównanie czasu: bilinear lookup vs matrix multiplication."""
    print("\n┌─ TEST 4: Benchmark — bilinear lookup vs matmul (CPU) ─────────┐")

    results = benchmark_lookup_vs_matmul(size=64, n_lookups=5000)

    print(f"  Operacji: {results['n_lookups']}")
    print(f"  Tekstura: {results['texture_size']}x{results['texture_size']}")
    print(f"  Czas bilinear (CPU emulacja): {results['time_bilinear_s']*1000:.1f} ms")
    print(f"  Czas matmul (NumPy):          {results['time_matmul_s']*1000:.1f} ms")
    print(f"  Matmul szybszy {results['speedup_matmul_over_bilinear']}x")
    print(f"\n  UWAGA: Na GPU bilinear jest realizowany sprzętowo przez TMU")
    print(f"  (Texture Mapping Unit) i jest praktycznie darmowy. Ta emulacja")
    print(f"  CPU nie oddaje realnej przewagi. Test sensowny dopiero na CUDA.")
    print()

    return results


# ─── Test 5: Werdykt końcowy ────────────────────────────────────────────────

def synthesize_verdict(test1, test2, test3, test4):
    """
    Finalna ocena: czy hipoteza "tekstury jako pamięć" ma sens?
    """
    print("\n" + "=" * 64)
    print("WERDYKT ETAPU 4: Tekstury GPU jako pamięć")
    print("=" * 64)

    pros = []
    cons = []

    # Analiza test 1
    if test1["is_equivalent"]:
        pros.append("Bilinear ≈ Lerp: interpolacja na teksturze jest matematycznie "
                    "równoważna interpolacji embeddingów (MSE < 0.01)")
    else:
        cons.append(f"Bilinear ≠ Lerp: artefakty dyskretyzacji (MSE={test1['mean_mse']:.6f})")

    # Analiza test 2
    if test2["semantic_ordering_correct"]:
        pros.append("Semantyczna spójność zachowana w 2D (cat↔dog > centrum)")
    else:
        cons.append("Semantyczna kolejność naruszona w interpolacji 2D")

    # Analiza test 3
    if test3["ordering_preserved"]:
        pros.append("Operacje snu zachowują hierarchię ważności wiedzy")
    else:
        cons.append("Operacje snu niszczą kolejność ważności")

    if test3["plasticity_decay_works"]:
        pros.append("Metaplastyczność działa: kostnienie chroni ważną wiedzę")

    # Analiza test 4
    cons.append(f"Na CPU bilinear jest {test4['speedup_matmul_over_bilinear']}x WOLNIEJSZY "
                f"niż matmul (ale na GPU TMU jest sprzętowy)")

    print("\n  ARGUMENTY ZA:")
    for i, p in enumerate(pros, 1):
        print(f"    {i}. {p}")

    print("\n  ARGUMENTY PRZECIW / OGRANICZENIA:")
    for i, c in enumerate(cons, 1):
        print(f"    {i}. {c}")

    # Werdykt
    score = len(pros) - len(cons)
    if len(pros) >= 3 and test1["is_equivalent"]:
        verdict = "WARUNKOWO POZYTYWNY"
        explanation = (
            "Bilinear filtering na teksturze JEST matematycznie sensowną "
            "interpolacją embeddingów, pod warunkami:\n"
            "    - Rozdzielczość tekstury wystarczająco wysoka (≥64x64)\n"
            "    - Embeddingi ułożone na siatce (nie losowo)\n"
            "    - Realizacja na GPU (TMU) dla realnej przewagi wydajnościowej\n"
            "  To NIE jest magiczny mechanizm 'rozmycia skojarzeń' — to po prostu\n"
            "  sprzętowo przyspieszony lerp na dyskretnej siatce. Wartość polega\n"
            "  na TYM, że GPU robi to ZA DARMO (jednostki TMU), nie na tym, że\n"
            "  daje jakościowo inną interpolację niż zwykły lerp."
        )
    else:
        verdict = "NEGATYWNY"
        explanation = (
            "Bilinear filtering nie daje przewagi jakościowej nad zwykłym lerp.\n"
            "  Artefakty dyskretyzacji siatki mogą zniekształcać interpolację.\n"
            "  Jedyna potencjalna przewaga to wydajność (TMU na GPU), ale wymaga\n"
            "  specyficznego hardware'u i nie przenosi się na CPU/edge."
        )

    print(f"\n  ══════════════════════════════════════════════════════")
    print(f"  WERDYKT: {verdict}")
    print(f"  ══════════════════════════════════════════════════════")
    print(f"  {explanation}")
    print()

    return {
        "verdict": verdict,
        "pros": pros,
        "cons": cons,
        "explanation": explanation.strip(),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("ETAP 4 — Tekstury GPU jako pamięć (weryfikacja hipotezy)")
    print("=" * 64)

    test1 = test_bilinear_equals_lerp()
    test2 = test_2d_bilinear_semantics()
    test3 = test_sleep_operations()
    test4 = test_benchmark()
    verdict = synthesize_verdict(test1, test2, test3, test4)

    # ─── Zapis wyniku ────────────────────────────────────────────────────
    out = {
        "stage": "etap4",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "description": "Weryfikacja hipotezy: tekstury GPU jako pamięć "
                       "(bilinear filtering = lerp embeddingów?)",
        "test_bilinear_vs_lerp": test1,
        "test_2d_semantics": test2,
        "test_sleep_operations": test3,
        "test_benchmark": test4,
        "verdict": verdict,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "etap4_textures.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Wynik zapisany: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
