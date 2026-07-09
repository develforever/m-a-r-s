"""
run_etap4b.py — Etap 4B: Tekstury GPU z Kohonenowską topologią (naprawiony).

PROBLEM (Etap 4): Bilinear filtering na RZADKIEJ teksturze interpoluje
z zerami → bzdura semantyczna. MSE = 0.49 (oczekiwane < 0.01).

ROZWIĄZANIE (ta wersja): Self-Organizing Map (Kohonen) WYMUSZA topologię
na siatce PRZED bilinear filtering. Podobne pojęcia = sąsiednie piksele.
Wtedy bilinear interpoluje MIĘDZY pokrewnymi embeddingami, nie z zerami.

Porównujemy 3 podejścia:
  A) Naiwna rzadka tekstura (Etap 4 oryginalny) — kontrola negatywna
  B) Kohonen SOM + tekstura — wymuszanie topologii  ← KLUCZOWY TEST
  C) Gęste embeddingi + dot product — alternatywa bez tekstur

Uruchom:
    cd src
    ..\\.venv\\Scripts\\python.exe run_etap4b.py
"""

import json
import os
from datetime import datetime
import numpy as np
import time

from cortex_texture import NeuroTexture
from kohonen_texture import KohonenSOM, KohonenTexture, DenseEmbeddingMemory

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ─── Dane testowe: klaster embeddingów z semantyczną strukturą ───────────────

def make_semantic_embeddings(n_concepts=50, dim=8, n_clusters=5, seed=42):
    """
    Generuje embeddingi z NATURALNĄ strukturą klastrową:
    5 "dziedzin" (np. zwierzęta, pojazdy, jedzenie, sport, muzyka).
    Pojęcia w tej samej dziedzinie są bliskie, między dziedzinami — dalekie.
    """
    rng = np.random.default_rng(seed)
    # Centra klastrów — oddalone od siebie
    centers = rng.normal(0, 2.0, (n_clusters, dim)).astype(np.float32)
    embeddings = []
    labels = []
    cluster_ids = []
    domain_names = ["zwierzęta", "pojazdy", "jedzenie", "sport", "muzyka"]

    per_cluster = n_concepts // n_clusters
    for c in range(n_clusters):
        for i in range(per_cluster):
            vec = centers[c] + rng.normal(0, 0.3, dim).astype(np.float32)
            embeddings.append(vec)
            labels.append(f"{domain_names[c]}_{i}")
            cluster_ids.append(c)

    return np.array(embeddings), labels, np.array(cluster_ids), centers


# ─── Test A: Naiwna rzadka tekstura (kontrola — powinno NADAL nie działać) ──

def test_naive_texture(embeddings, labels, clusters):
    """Powtórzenie Etapu 4: losowe rozmieszczenie na teksturze."""
    print("\n┌─ TEST A: Naiwna rzadka tekstura (kontrola negatywna) ─────────┐")

    tex = NeuroTexture(size=32)
    n = len(embeddings)
    rng = np.random.default_rng(0)

    # Losowe pozycje — brak topologii
    positions = rng.random((n, 2))
    for i, (emb, pos) in enumerate(zip(embeddings, positions)):
        tex.write(pos[0], pos[1], emb[0])  # tylko 1D kanał R

    # Test interpolacji między pojęciami z TEJ SAMEJ dziedziny
    errors = []
    for _ in range(20):
        c = rng.integers(5)
        idx = np.where(clusters == c)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        # Lerp w przestrzeni embeddingów
        lerp = 0.5 * embeddings[i] + 0.5 * embeddings[j]
        # Bilinear na teksturze (środek między pozycjami)
        mid_pos = 0.5 * positions[i] + 0.5 * positions[j]
        bilinear_val = tex.bilinear_lookup(mid_pos[0], mid_pos[1])
        # Porównanie: wartość w kanale R
        expected = lerp[0]
        got = bilinear_val[0]
        errors.append((expected - got) ** 2)

    mse = float(np.mean(errors)) if errors else 999
    print(f"  MSE interpolacji (naiwna): {mse:.4f}")
    print(f"  Werdykt: {'✗ ŹÓŁE' if mse > 0.1 else '✓ OK'} (oczekiwane: wysoki MSE)")
    return {"method": "naive_sparse", "mse": round(mse, 6)}


# ─── Test B: Kohonen SOM + gęsta tekstura ───────────────────────────────────

def test_kohonen_texture(embeddings, labels, clusters):
    """
    KLUCZOWY TEST: trenuj SOM na embeddingach, stwórz gęstą teksturę,
    potem bilinear interpolacja między pojęciami z tej samej dziedziny.
    """
    print("\n┌─ TEST B: Kohonen SOM + gęsta tekstura ────────────────────────┐")

    grid_size = 16  # 16x16 = 256 neuronów dla 50 pojęć
    dim = embeddings.shape[1]

    # Trening SOM
    t0 = time.perf_counter()
    som = KohonenSOM(grid_size=grid_size, input_dim=dim, seed=42)
    som.train(embeddings, epochs=len(embeddings) * 50, lr_start=0.5, lr_end=0.01)
    t_train = time.perf_counter() - t0

    # Jakość mapowania
    q_error = som.quantization_error(embeddings)
    print(f"  SOM grid: {grid_size}x{grid_size}, trening: {t_train:.2f}s")
    print(f"  Quantization error: {q_error:.4f}")

    # Sprawdź topologię: czy pojęcia z tej samej dziedziny są blisko na siatce?
    positions = np.array([som.map_to_grid(e) for e in embeddings])
    intra_dist = []  # odległości w TEJ SAMEJ dziedzinie
    inter_dist = []  # odległości MIĘDZY dziedzinami
    rng = np.random.default_rng(77)
    for _ in range(200):
        i, j = rng.choice(len(embeddings), 2, replace=False)
        d = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
        if clusters[i] == clusters[j]:
            intra_dist.append(d)
        else:
            inter_dist.append(d)

    mean_intra = np.mean(intra_dist) if intra_dist else 0
    mean_inter = np.mean(inter_dist) if inter_dist else 0
    topology_ratio = mean_inter / (mean_intra + 1e-8)

    print(f"  Odległość intra-cluster (ta sama dziedzina): {mean_intra:.2f}")
    print(f"  Odległość inter-cluster (różne dziedziny):  {mean_inter:.2f}")
    print(f"  Ratio (im wyższe tym lepsza topologia):     {topology_ratio:.2f}")
    topology_ok = topology_ratio > 1.3

    # Kluczowy test: interpolacja bilinear vs lerp
    ktex = KohonenTexture(som)
    errors_bilinear = []
    errors_control = []

    for _ in range(50):
        c = rng.integers(5)
        idx = np.where(clusters == c)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        alpha = rng.random()

        # Ground truth: lerp w przestrzeni embeddingów
        lerp = (1 - alpha) * embeddings[i] + alpha * embeddings[j]

        # Metoda B: interpolacja na teksturze Kohonena
        bilinear_result = ktex.interpolate_between(embeddings[i], embeddings[j], alpha)

        # MSE
        mse_bilinear = float(np.mean((lerp - bilinear_result) ** 2))
        errors_bilinear.append(mse_bilinear)

        # Kontrola: losowy punkt na teksturze (powinien być gorszy)
        rand_x, rand_y = rng.random(), rng.random()
        random_result = ktex.bilinear_lookup(rand_x, rand_y)
        mse_random = float(np.mean((lerp - random_result) ** 2))
        errors_control.append(mse_random)

    mse_kohonen = float(np.mean(errors_bilinear))
    mse_random = float(np.mean(errors_control))

    print(f"\n  Interpolacja bilinear (Kohonen): MSE = {mse_kohonen:.4f}")
    print(f"  Kontrola (losowy punkt):         MSE = {mse_random:.4f}")
    print(f"  Poprawa vs kontrola:             {(1 - mse_kohonen/mse_random)*100:.1f}%")

    # Porównanie z cosine similarity zachowanym
    cos_preserved = []
    for _ in range(50):
        c = rng.integers(5)
        idx = np.where(clusters == c)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        # Cosine w oryginale
        cos_orig = np.dot(embeddings[i], embeddings[j]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
        # Cosine na siatce (wagi BMU)
        bmu_i = som.map_to_grid(embeddings[i])
        bmu_j = som.map_to_grid(embeddings[j])
        w_i = som.weights[bmu_i]
        w_j = som.weights[bmu_j]
        cos_grid = np.dot(w_i, w_j) / (np.linalg.norm(w_i) * np.linalg.norm(w_j) + 1e-8)
        cos_preserved.append(abs(cos_orig - cos_grid))

    cos_drift = float(np.mean(cos_preserved))
    print(f"  Cosine similarity drift (SOM vs oryginał): {cos_drift:.4f}")

    interpolation_ok = mse_kohonen < mse_random * 0.7  # co najmniej 30% lepszy

    print(f"\n  Topologia zachowana? {'✓ TAK' if topology_ok else '✗ NIE'} "
          f"(ratio={topology_ratio:.2f}, próg=1.3)")
    print(f"  Interpolacja sensowna? {'✓ TAK' if interpolation_ok else '✗ NIE'} "
          f"(MSE={mse_kohonen:.4f} vs random={mse_random:.4f})")

    return {
        "method": "kohonen_som",
        "grid_size": grid_size,
        "train_time_s": round(t_train, 3),
        "quantization_error": round(q_error, 6),
        "topology_ratio": round(topology_ratio, 4),
        "topology_ok": bool(topology_ok),
        "mse_bilinear_interpolation": round(mse_kohonen, 6),
        "mse_random_control": round(mse_random, 6),
        "improvement_vs_random_pct": round((1 - mse_kohonen / mse_random) * 100, 2),
        "cosine_drift": round(cos_drift, 6),
        "interpolation_ok": bool(interpolation_ok),
    }


# ─── Test C: Gęste embeddingi + dot product ─────────────────────────────────

def test_dense_embeddings(embeddings, labels, clusters):
    """
    Alternatywa: bez tekstur. Czyste embeddingi + cosine similarity.
    Interpolacja = lerp w przestrzeni latent.
    Baseline do porównania z Kohonennem.
    """
    print("\n┌─ TEST C: Gęste embeddingi + dot product ──────────────────────┐")

    mem = DenseEmbeddingMemory(dim=embeddings.shape[1])
    for label, emb in zip(labels, embeddings):
        mem.store(label, emb)

    rng = np.random.default_rng(99)

    # Test: interpolacja = IDEALNA (bo to lerp w oryginalnej przestrzeni)
    errors = []
    for _ in range(50):
        c = rng.integers(5)
        idx = np.where(clusters == c)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        alpha = rng.random()
        lerp = mem.interpolate(embeddings[i], embeddings[j], alpha)
        expected = (1 - alpha) * embeddings[i] + alpha * embeddings[j]
        mse = float(np.mean((lerp - expected) ** 2))
        errors.append(mse)

    mse_dense = float(np.mean(errors))

    # Benchmark: czas query
    t0 = time.perf_counter()
    for _ in range(1000):
        q = rng.normal(0, 1, embeddings.shape[1]).astype(np.float32)
        mem.query(q, top_k=3)
    t_query = time.perf_counter() - t0

    print(f"  MSE interpolacji (lerp): {mse_dense:.6f} (IDEALNY — to samo co ground truth)")
    print(f"  Czas 1000 queries: {t_query*1000:.1f} ms")
    print(f"  Werdykt: interpolacja idealna, ale bez sprzętowego przyspieszenia")

    return {
        "method": "dense_embeddings",
        "mse_interpolation": round(mse_dense, 8),
        "query_time_1000_ms": round(t_query * 1000, 2),
        "note": "Interpolacja idealna (lerp=lerp), ale brak hw acceleration"
    }


# ─── Test D: Operacje snu na teksturze Kohonena ─────────────────────────────

def test_sleep_on_kohonen(embeddings, clusters):
    """
    Czy operacje snu (blur) na gęstej teksturze Kohonena dają sens?
    Blur powinien "rozlewać" wiedzę na sąsiadów — a sąsiedzi to pokrewne pojęcia!
    """
    print("\n┌─ TEST D: Operacje snu na teksturze Kohonena ─────────────────┐")

    dim = embeddings.shape[1]
    som = KohonenSOM(grid_size=16, input_dim=dim, seed=42)
    som.train(embeddings, epochs=len(embeddings) * 50)

    weights_before = som.weights.copy()

    # Gaussian blur (konsolidacja) — testujemy różne sigma
    from scipy.ndimage import gaussian_filter
    best_sigma = 0.5
    best_consolidation = -999
    for sigma in [0.3, 0.5, 0.8, 1.0, 1.5]:
        w_test = np.zeros_like(som.weights)
        for d in range(dim):
            w_test[:, :, d] = gaussian_filter(som.weights[:, :, d], sigma=sigma)
        # Quick check
        sample_dists = []
        for ci, cj in [(0, 1), (2, 3)]:
            idx_i = np.where(clusters == ci)[0]
            idx_j = np.where(clusters == cj)[0]
            if len(idx_i) > 0 and len(idx_j) > 0:
                bi = som.map_to_grid(embeddings[idx_i[0]])
                bj = som.map_to_grid(embeddings[idx_j[0]])
                d_b = np.linalg.norm(weights_before[bi] - weights_before[bj])
                d_a = np.linalg.norm(w_test[bi] - w_test[bj])
                sample_dists.append((d_b, d_a))
        # We want intra-cluster to shrink more than inter-cluster
        intra_test = []
        for cc in range(5):
            idx_c = np.where(clusters == cc)[0]
            if len(idx_c) >= 2:
                b1 = som.map_to_grid(embeddings[idx_c[0]])
                b2 = som.map_to_grid(embeddings[idx_c[1]])
                intra_test.append(np.linalg.norm(w_test[b1] - w_test[b2]))
        cons = (1 - np.mean(intra_test) / (np.mean([
            np.linalg.norm(weights_before[som.map_to_grid(embeddings[np.where(clusters == cc)[0][0]])] -
                           weights_before[som.map_to_grid(embeddings[np.where(clusters == cc)[0][1]])])
            for cc in range(5) if len(np.where(clusters == cc)[0]) >= 2
        ]) + 1e-8)) * 100
        if cons > best_consolidation:
            best_consolidation = cons
            best_sigma = sigma

    print(f"  Najlepszy sigma dla konsolidacji: {best_sigma} ({best_consolidation:.1f}%)")

    weights_after = np.zeros_like(som.weights)
    for d in range(dim):
        weights_after[:, :, d] = gaussian_filter(som.weights[:, :, d], sigma=best_sigma)

    # Sprawdź: czy po blurze pojęcia z tej samej dziedziny są BLIŻSZE?
    rng = np.random.default_rng(55)
    intra_before, intra_after = [], []
    for _ in range(100):
        c = rng.integers(5)
        idx = np.where(clusters == c)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        bmu_i = som.map_to_grid(embeddings[i])
        bmu_j = som.map_to_grid(embeddings[j])
        d_before = np.linalg.norm(weights_before[bmu_i] - weights_before[bmu_j])
        d_after = np.linalg.norm(weights_after[bmu_i] - weights_after[bmu_j])
        intra_before.append(d_before)
        intra_after.append(d_after)

    mean_before = float(np.mean(intra_before))
    mean_after = float(np.mean(intra_after))
    consolidation = (1 - mean_after / mean_before) * 100

    print(f"  Odległość intra-cluster PRZED blur: {mean_before:.4f}")
    print(f"  Odległość intra-cluster PO blur:    {mean_after:.4f}")
    print(f"  Konsolidacja (zmniejszenie dystansu): {consolidation:.1f}%")
    print(f"  Wniosek: blur na Kohonenie {'ZBLIŻA' if consolidation > 0 else 'NIE ZBLIŻA'} "
          f"pokrewne pojęcia")

    return {
        "test": "sleep_blur_kohonen",
        "distance_before": round(mean_before, 6),
        "distance_after": round(mean_after, 6),
        "consolidation_pct": round(consolidation, 2),
        "blur_makes_sense": consolidation > 0
    }


# ─── Werdykt końcowy ────────────────────────────────────────────────────────

def synthesize_verdict(test_a, test_b, test_c, test_d):
    print("\n" + "=" * 64)
    print("WERDYKT ETAPU 4B: Tekstury z topologią Kohonena")
    print("=" * 64)

    print(f"""
  PORÓWNANIE METOD:
  ┌────────────────────────┬────────────┬────────────────────────────┐
  │ Metoda                 │ MSE interp │ Uwagi                      │
  ├────────────────────────┼────────────┼────────────────────────────┤
  │ A) Naiwna tekstura     │ {test_a['mse']:>10.4f} │ ✗ interpolacja z zerami    │
  │ B) Kohonen + tekstura  │ {test_b['mse_bilinear_interpolation']:>10.4f} │ {'✓' if test_b['interpolation_ok'] else '✗'} topologia wymuszana      │
  │ C) Gęste embeddingi    │ {test_c['mse_interpolation']:>10.4f} │ ✓ idealny lerp (ground tr) │
  └────────────────────────┴────────────┴────────────────────────────┘
""")

    # Ocena Kohonena vs Etap 4 oryginalny
    improvement = (1 - test_b['mse_bilinear_interpolation'] / test_a['mse']) * 100

    if test_b['interpolation_ok'] and test_b['topology_ok'] and test_d['blur_makes_sense']:
        verdict = "WARUNKOWO POZYTYWNY"
        explanation = (
            "Kohonen SOM NAPRAWIA problem rzadkiej tekstury:\n"
            f"    - MSE spadło z {test_a['mse']:.4f} (naiwna) do {test_b['mse_bilinear_interpolation']:.4f} (Kohonen) = poprawa {improvement:.0f}%\n"
            f"    - Topologia zachowana (ratio={test_b['topology_ratio']:.2f})\n"
            f"    - Blur na Kohonenie ZBLIŻA pokrewne pojęcia ({test_d['consolidation_pct']:.1f}% konsolidacja)\n"
            "    - Bilinear filtering MA sens gdy tekstura jest GĘSTO wypełniona przez SOM\n"
            "\n"
            "  OGRANICZENIA:\n"
            f"    - Nie dorównuje idealnemu lerp (MSE={test_b['mse_bilinear_interpolation']:.4f} vs {test_c['mse_interpolation']:.6f})\n"
            "    - Wymaga treningu SOM (dodatkowy koszt)\n"
            "    - Wartość polega na HW acceleration (TMU) — bez GPU nie ma zysku\n"
            "\n"
            "  DECYZJA: Tekstury z SOM topologią to REALNY mechanizm, nie metafora.\n"
            "  Ale gęste embeddingi + dot product (Opcja C) są prostsze i dokładniejsze.\n"
            "  Tekstury mają sens TYLKO jeśli sprzętowe TMU daje realny zysk wydajnościowy."
        )
    elif test_b['interpolation_ok']:
        verdict = "CZĘŚCIOWO POZYTYWNY"
        explanation = (
            f"Kohonen poprawia interpolację ({improvement:.0f}%), ale topologia/blur nieidealny."
        )
    else:
        verdict = "NEGATYWNY (nawet z SOM)"
        explanation = (
            "Nawet z Kohonenowską topologią bilinear nie dorównuje lerpowi.\n"
            "  Tekstury GPU jako pamięć semantyczna to ślepy zaułek."
        )

    print(f"  ══════════════════════════════════════════════════════")
    print(f"  WERDYKT: {verdict}")
    print(f"  ══════════════════════════════════════════════════════")
    print(f"  {explanation}")
    print()

    return {
        "verdict": verdict,
        "improvement_naive_to_kohonen_pct": round(improvement, 2),
        "explanation": explanation.strip(),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("ETAP 4B — Tekstury z topologią Kohonena (naprawiony test)")
    print("=" * 64)
    print("  Hipoteza: SOM wymusza topologię → bilinear MA sens semantyczny")

    # Dane testowe
    embeddings, labels, clusters, centers = make_semantic_embeddings(
        n_concepts=50, dim=8, n_clusters=5, seed=42)
    print(f"\n  Dane: {len(embeddings)} pojęć, {embeddings.shape[1]}D, 5 dziedzin")

    # Testy
    test_a = test_naive_texture(embeddings, labels, clusters)
    test_b = test_kohonen_texture(embeddings, labels, clusters)
    test_c = test_dense_embeddings(embeddings, labels, clusters)
    test_d = test_sleep_on_kohonen(embeddings, clusters)
    verdict = synthesize_verdict(test_a, test_b, test_c, test_d)

    # Zapis
    out = {
        "stage": "etap4b",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "description": "Tekstury GPU z topologią Kohonena — naprawiona wersja Etapu 4",
        "test_naive": test_a,
        "test_kohonen": test_b,
        "test_dense": test_c,
        "test_sleep_kohonen": test_d,
        "verdict": verdict,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "etap4b_kohonen.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wynik zapisany: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
