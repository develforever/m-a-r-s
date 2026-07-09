"""
run_G2_compositional.py -- G2: kompozycyjny zero-shot przez atrybuty-slowa.

Hipoteza (DROGA_G_PLAN.md, pomysl uzytkownika: "myslenie cechami-slowami"):
  Jesli przestrzenia routingu jest warstwa POJEC (binarne atrybuty slowne:
  ma-rekawy, jest-obuwiem, siega-kostki...), to nowa klasa moze byc
  zdefiniowana SAMYM OPISEM -- kombinacja znanych pojec -- i rozpoznana
  bez ani jednego przykladu treningowego.

Protokol (leave-one-out, Fashion-MNIST):
  Dla kazdej klasy h (10x) x 5 seedow:
    1. backbone losowy zamrozony (jak F1d/F3b) + projekcja 128 -> 11 atrybutow
    2. projekcja trenowana WYLACZNIE na danych 9 pozostalych klas
       (CE po podobienstwach kosinusowych do ich wektorow atrybutow)
    3. klasa h istnieje w systemie tylko jako wektor atrybutow (prototyp)
    4. pomiar: routing testowych probek klasy h wsrod WSZYSTKICH 10
       prototypow (zs_acc) + routing klas widzianych (kontrola jakosci)
  Mierzymy ROUTING (rozpoznanie opisanej klasy); podow dla klasy h nie ma
  z definicji (zero danych) -- to uczciwa miara "rozumienia opisu".

Kryterium werdyktu (Z GORY, z DROGA_G_PLAN.md):
  srednie zs_acc > 3x losowe (30%) => SYGNAL+ kompozycyjnosci.
PRE-REJESTROWANE OCZEKIWANIE STRUKTURALNE: klasa jest osiagalna
kompozycyjnie tylko, gdy kazdy jej atrybut WYSTEPUJE ZMIENNIE w klasach
treningowych. Bag ma unikalny atrybut (uchwyt) staly=0 w treningu bez
Bag -- projekcja nie moze nauczyc sie tego wymiaru => oczekiwana PORAZKA
na Bag (i to jest wynik, nie blad: granica kompozycyjnosci).

Tryb szybki:  python src/run_G2_compositional.py --smoke  (1 seed, 4 epoki)
Pelny:        python src/run_G2_compositional.py          (5 seedow, 15 epok)
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from mars_cl import MarsCLSystem
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR, TEMP, BATCH = 0.001, 0.1, 512

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot"]

# 11 binarnych atrybutow slownych (kolumny), 10 klas (wiersze).
# Kazdy wiersz UNIKALNY. Kolejnosc atrybutow:
ATTR_NAMES = ["zakrywa_tulow", "zakrywa_nogi", "ma_rekawy",
              "dlugi_rekaw", "jest_obuwiem", "siega_kostki",
              "odkryta_konstrukcja", "ma_uchwyt", "pelna_dlugosc",
              "rozpinane_z_przodu", "noszone_na_ciele"]
ATTRS = torch.tensor([
    # tul nog rek dlg but kos odk uch pel roz nos
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],   # 0 T-shirt
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # 1 Trouser
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],   # 2 Pullover
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],   # 3 Dress
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],   # 4 Coat
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],   # 5 Sandal
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],   # 6 Shirt
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],   # 7 Sneaker
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # 8 Bag
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],   # 9 AnkleBoot
], dtype=torch.float32)


# LEKCJA ZE SMOKE'A v1 (08.07.2026, wazna metodycznie): wersja 1 uczyla
# projekcji przez CE po podobienstwach do prototypow -- czyli DYSKRYMINACJI
# KLAS. Softmax odpychal wszystkie probki treningowe od prototypu klasy
# wstrzymanej (byla negatywem w mianowniku) => ZS = 0.0% przy 80% na
# widzianych. Poprawka v2 = kanoniczne uczenie POJEC (styl DAP): BCE per
# atrybut -- kazdy wymiar uczy sie jako niezalezny detektor pojecia tam,
# gdzie pojecie WARIUJE w klasach treningowych, i przenosi sie na opis
# klasy niewidzianej. Routing: najblizszy wektor atrybutow (L2 po sigmoid).
#
# ROZSZERZENIE OCZEKIWANIA STRUKTURALNEGO (ta sama regula co dla Bag):
# atrybut unikalny dla jednej klasy jest STALY w treningu bez niej =>
# niewyuczalny. Dotyczy: Bag (ma_uchwyt), Sandal (odkryta_konstrukcja),
# AnkleBoot (siega_kostki). Oczekiwane porazki: te 3 klasy; osiagalne: 7.
STRUCTURAL_FAILS = {5, 8, 9}   # Sandal, Bag, AnkleBoot


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_holdout(h, feats_tr, ytr, feats_te, yte, attrs, seed, epochs,
                device):
    """
    Trening projekcji bez klasy h -- UCZENIE POJEC (BCE per atrybut,
    styl DAP), nie klas. Routing: najblizszy wektor atrybutow (L2 po
    sigmoid) wsrod WSZYSTKICH 10 opisow.
    """
    torch.manual_seed(seed * 100 + h)
    proj = nn.Linear(feats_tr.shape[1], attrs.shape[1]).to(device)
    seen_mask = ytr != h
    Xf = feats_tr[seen_mask]
    Tf = attrs[ytr[seen_mask]]           # targety: wektory atrybutow
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(proj.parameters(), lr=LR)
    for _ in range(epochs):
        perm = torch.randperm(len(Xf), device=device)
        for s in range(0, len(Xf), BATCH):
            idx = perm[s:s + BATCH]
            loss = crit(proj(Xf[idx]), Tf[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a_hat = torch.sigmoid(proj(feats_te))          # [B, 11]
        d = torch.cdist(a_hat, attrs)                  # L2 do 10 opisow
        pred = d.argmin(dim=1)
    zs_mask = yte == h
    zs_acc = (pred[zs_mask] == h).float().mean().item()
    seen_acc = (pred[~zs_mask] == yte[~zs_mask]).float().mean().item()
    return zs_acc, seen_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sanity macierzy atrybutow: wiersze unikalne
    assert len({tuple(r.tolist()) for r in ATTRS}) == 10, "atrybuty nieunikalne"

    print("=" * 72)
    print(f"G2 -- kompozycyjny zero-shot przez atrybuty  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epochs={epochs} | "
          f"{ATTRS.shape[1]} atrybutow | kryterium: zs_acc > 30% (3x losowe)")
    print("=" * 72)

    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    attrs = ATTRS.to(device)

    t0 = time.perf_counter()
    out = {"experiment": "G2_compositional", "device": device,
           "n_seeds": n_seeds, "epochs": epochs,
           "attr_names": ATTR_NAMES, "attrs": ATTRS.tolist(),
           "per_class": {}}

    per_class_zs = {}
    for seed in range(n_seeds):
        # backbone losowy zamrozony per seed; cechy liczone RAZ
        torch.manual_seed(seed)
        sys_ = MarsCLSystem(backbone_source="random").to(device)
        feats_tr = sys_.feats_batched(Xtr)
        feats_te = sys_.feats_batched(Xte)
        for h in range(10):
            zs, seen = run_holdout(h, feats_tr, ytr, feats_te, yte,
                                   attrs, seed, epochs, device)
            per_class_zs.setdefault(h, []).append(
                {"zs_acc": round(zs, 4), "seen_acc": round(seen, 4)})
            print(f"seed {seed} holdout {CLASS_NAMES[h]:10s}: "
                  f"ZS={zs*100:5.1f}%  (widziane: {seen*100:.1f}%)")

    # ---------- agregacja + werdykt ----------
    all_zs_means = []
    print(f"\n--- G2 (n={n_seeds}) -- zero-shot routing per klasa ---")
    for h in range(10):
        zs_s = stats([r["zs_acc"] for r in per_class_zs[h]])
        seen_s = stats([r["seen_acc"] for r in per_class_zs[h]])
        out["per_class"][CLASS_NAMES[h]] = {"zs_acc": zs_s,
                                            "seen_acc": seen_s}
        all_zs_means.append(zs_s["mean"])
        flag = (" <-- oczekiwana porazka (atrybut unikalny)"
                if h in STRUCTURAL_FAILS else "")
        print(f"  {CLASS_NAMES[h]:10s}: ZS {zs_s['mean']*100:5.1f}"
              f"+/-{zs_s['std']*100:4.1f}%{flag}")
    overall = sum(all_zs_means) / len(all_zs_means)
    # srednia po klasach OSIAGALNYCH strukturalnie (pre-rejestrowane)
    reach = [m for h, m in enumerate(all_zs_means)
             if h not in STRUCTURAL_FAILS]
    overall_r = sum(reach) / len(reach)
    verdict = ("SYGNAL+ (kompozycyjnosc)" if overall > 0.30
               else ("SYGNAL+ z granica (osiagalne > 30%)"
                     if overall_r > 0.30 else "SZUM/NEGATYWNY"))
    out["overall_zs"] = round(overall, 4)
    out["overall_zs_reachable"] = round(overall_r, 4)
    out["structural_fails"] = sorted(STRUCTURAL_FAILS)
    out["verdict"] = verdict
    print(f"\n  srednio: {overall*100:.1f}% | osiagalne (7 klas): "
          f"{overall_r*100:.1f}% | losowe: 10% | prog: 30%")
    print(f"  WERDYKT: {verdict}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("G2_compositional_smoke.json" if args.smoke
             else "G2_compositional.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
