"""
run_G2b_ecoc.py -- G2b: slownik atrybutow z dystansem kodowym (ECOC)
(DROGA_G2B_PLAN.md).

NOWY plik (branch droga-g2b) -- istniejacy kod NIETKNIETY.
run_holdout = wierna kopia run_G2_compositional.run_holdout.

Warianty: attrs11 (oryginalna macierz G2 -- reprodukcja sanity)
vs attrs21 (ECOC: min Hamming 4, kazdy atrybut waruje w kazdym
leave-one-out -- wlasnosci weryfikowane asercjami przy starcie).

Kryteria (Z GORY, DROGA_G2B_PLAN.md):
  1) SUKCES PELNY: sredni ZS attrs21 > 30%.
  2) SUKCES MECHANIZMU: pary attrs21 vs attrs11 SYGNAL+ ORAZ
     sr ZS attrs21 >= 2x attrs11.
  3) NEGATYW: brak obu.
  4) Test reguly osiagalnosci: 10/10 klas ZS > 0% (przewidywanie),
     w tym dawne porazki {Sandal, Bag, AnkleBoot}.

Tryb szybki:  python src/run_G2b_ecoc.py --smoke
Pelny:        python src/run_G2b_ecoc.py
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from mars_cl import MarsCLSystem
from run_D1_mars_v2_baseline import load_dataset
from run_G2_compositional import (ATTRS as ATTRS11, CLASS_NAMES,
                                  run_holdout)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OLD_STRUCTURAL_FAILS = {5, 8, 9}

ATTR21_NAMES = [
    "zakrywa_tulow", "zakrywa_nogi", "ma_rekawy", "dlugi_rekaw",
    "jest_obuwiem", "siega_kostki", "odkryta_gora",
    "ma_klamre_lub_zapiecie", "pelna_dlugosc", "rozpinane_z_przodu",
    "nakladane_przez_glowe", "sznurowane", "z_dzianiny",
    "warstwa_wierzchnia", "na_zime", "na_lato", "dolna_czesc_ciala",
    "miekki_material", "sztywne_elementy", "przechowuje", "ze_skory"]
_ATTR21_CLASSES = [
    [0, 2, 3, 4, 6], [1, 3], [0, 2, 4, 6], [2, 4, 6], [5, 7, 9],
    [1, 9], [5, 8], [5, 8, 9], [3, 4], [4, 6], [0, 2, 3], [7, 9],
    [0, 2], [2, 4], [2, 4, 9], [0, 5], [1, 5, 7, 9], [0, 1, 2, 3, 6],
    [5, 7, 8, 9], [4, 8], [8, 9]]
ATTRS21 = torch.zeros(10, len(ATTR21_NAMES))
for _j, _cls in enumerate(_ATTR21_CLASSES):
    for _c in _cls:
        ATTRS21[_c, _j] = 1.0

VARIANTS = {"attrs11": ATTRS11, "attrs21": ATTRS21}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def check_matrix(M):
    rows = [tuple(int(x) for x in r) for r in M]
    assert len(set(rows)) == 10, "wiersze nieunikalne"
    dists = [sum(a != b for a, b in zip(r1, r2))
             for r1, r2 in itertools.combinations(rows, 2)]
    col_sums = [int(s) for s in M.sum(dim=0)]
    return min(dists), col_sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # asercje wlasnosci ECOC (pre-rejestrowane w planie)
    mn11, _ = check_matrix(ATTRS11)
    mn21, cs21 = check_matrix(ATTRS21)
    assert mn21 >= 3, f"min Hamming attrs21 = {mn21} < 3"
    assert all(2 <= s <= 8 for s in cs21), f"osiagalnosc: col sums {cs21}"

    print("=" * 72)
    print(f"G2b -- ECOC: attrs11 (minH={mn11}) vs attrs21 (minH={mn21}) "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epochs={epochs} | "
          f"prog sukcesu pelnego: sredni ZS > 30%")
    print("=" * 72)

    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)

    t0 = time.perf_counter()
    out = {"experiment": "G2b_ecoc", "device": device,
           "n_seeds": n_seeds, "epochs": epochs,
           "attr21_names": ATTR21_NAMES, "attrs21": ATTRS21.tolist(),
           "min_hamming": {"attrs11": mn11, "attrs21": mn21},
           "variants": {}, "verdicts": {}}

    zs_mean_per_seed = {v: [] for v in VARIANTS}
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        sys_ = MarsCLSystem(backbone_source="random").to(device)
        feats_tr = sys_.feats_batched(Xtr)
        feats_te = sys_.feats_batched(Xte)
        for vname, attrs_cpu in VARIANTS.items():
            attrs = attrs_cpu.to(device)
            zs_list = []
            for h in range(10):
                zs, seen = run_holdout(h, feats_tr, ytr, feats_te, yte,
                                       attrs, seed, epochs, device)
                out["variants"].setdefault(vname, {}).setdefault(
                    CLASS_NAMES[h], []).append(
                    {"seed": seed, "zs_acc": round(zs, 4),
                     "seen_acc": round(seen, 4)})
                zs_list.append(zs)
            zs_mean_per_seed[vname].append(sum(zs_list) / 10)
            print(f"seed {seed} [{vname}]: sredni ZS "
                  f"{sum(zs_list)/10*100:5.1f}% | per klasa: "
                  + " ".join(f"{z*100:.0f}" for z in zs_list))

    if not args.smoke:
        m11 = zs_mean_per_seed["attrs11"]
        m21 = zs_mean_per_seed["attrs21"]
        d = [(a - b) * 100 for a, b in zip(m21, m11)]
        noise = (stats([x * 100 for x in m11])["std"]
                 + stats([x * 100 for x in m21])["std"])
        ds = stats(d)
        mech = (ds["mean"] > noise and ds["min"] > 0
                and stats(m21)["mean"] >= 2 * stats(m11)["mean"])
        pelny = stats(m21)["mean"] > 0.30
        if pelny:
            w = "SUKCES PELNY (ZS > 30%)"
        elif mech:
            w = "SUKCES MECHANIZMU (SYGNAL+ i >=2x, ponizej 30%)"
        else:
            w = "NEGATYW (dystans kodowy nie jest dzwignia)"
        out["verdicts"]["glowny"] = {
            "zs_attrs11": stats(m11), "zs_attrs21": stats(m21),
            "pairs_pp": [round(x, 2) for x in d],
            "noise_pp": round(noise, 4), "werdykt": w}
        # test reguly osiagalnosci (przewidywanie: 10/10 > 0%)
        reach = {}
        for h in range(10):
            recs = out["variants"]["attrs21"][CLASS_NAMES[h]]
            zmean = sum(r["zs_acc"] for r in recs) / len(recs)
            reach[CLASS_NAMES[h]] = round(zmean, 4)
        zeros = [k for k, v in reach.items() if v == 0.0]
        out["verdicts"]["regula_osiagalnosci"] = {
            "zs_per_klasa": reach,
            "dawne_porazki": {CLASS_NAMES[h]: reach[CLASS_NAMES[h]]
                              for h in OLD_STRUCTURAL_FAILS},
            "klasy_zerowe": zeros,
            "przewidywanie_10_z_10": len(zeros) == 0}

    print(f"\n--- G2b (n={n_seeds}) ---")
    for key, vd in out.get("verdicts", {}).items():
        print(f"  {key}: {json.dumps(vd, ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "G2b_ecoc_smoke.json" if args.smoke else "G2b_ecoc.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
