"""
run_G3_pretrained_compositional.py -- G3: kompozycyjny zero-shot na
zamrozonych cechach pretrained ResNet18-ImageNet (DROGA_G3_PLAN.md).

NOWY plik (branch droga-g3) -- istniejacy kod NIETKNIETY. `run_holdout`
z run_G2_compositional reuzywane WERBATIM (buduje nn.Linear(feats_dim,
n_attrs) -- agnostyczne wobec zrodla i wymiaru cech). Macierze attrs11/
attrs21 z G2/G2b (jedno zrodlo prawdy).

Izolacja dzwigni (c) z diagnozy G2/G2b: ta sama maszyneria kompozycyjna,
te same slowniki, jedyna zmiana = cechy losowe -> cechy pretrained.

Plan 2x2 na tych samych 5 seedach:
  backbone: random (MarsCLSystem, per-seed) vs pretrained (ResNet18
            512-d, deterministyczne -- seed zmienia tylko uczenie pojec)
  slownik:  attrs11 (oryginal G2) vs attrs21 (ECOC, min Hamming 4)

WERDYKT BINARNY (zamrozony, prog 30% -- ta sama linia co G2/G2b):
  zs_pretrained_best = max(sr ZS pretrained-attrs11, pretrained-attrs21)
  G3+ <=> zs_pretrained_best > 30%   (dzwignia c = THE waskie gardlo)
  G3- <=> <= 30% dla OBU slownikow   (granica PODEJSCIA, nie cech)
Testy pomocnicze (nie zmieniaja binarnej decyzji): T1 pretrained vs
random (attrs11), T2 osiagalnosc, T3 attrs21 vs attrs11 na pretrained,
T4 sanity reprodukcji random ~ G2b.

Wymaga: Fashion-MNIST (auto), torchvision ResNet18 (jak L).
Cache cech Fashion@224: data/fashion_resnet18_224_feats.pt (raz).

Tryb szybki:  python src/run_G3_pretrained_compositional.py --smoke
Pelny:        python src/run_G3_pretrained_compositional.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from mars_cl import MarsCLSystem
from mars_cl_l import IMAGENET_MEAN, IMAGENET_STD
from run_D1_mars_v2_baseline import load_dataset
from run_G2_compositional import ATTRS as ATTRS11, CLASS_NAMES, run_holdout
from run_G2b_ecoc import (ATTRS21, ATTR21_NAMES, OLD_STRUCTURAL_FAILS,
                          check_matrix)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FEATS_CACHE = os.path.join(os.path.dirname(__file__), "..", "data",
                           "fashion_resnet18_224_feats.pt")
FASHION_MEAN, FASHION_STD = 0.2860, 0.3530     # jak load_dataset
GATE_PP = 30.0                                 # linia G3+/G3- (prog G2/G2b)
DICTS = {"attrs11": ATTRS11, "attrs21": ATTRS21}


def stats(vals: Sequence[float]) -> Dict[str, float]:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def verdict_paired(deltas_pp: Sequence[float], noise_pp: float):
    d = stats(deltas_pp)
    if d["mean"] > noise_pp and d["min"] > 0:
        v = "SYGNAL+"
    elif d["mean"] < -noise_pp:
        v = "SYGNAL-"
    elif all(x > 0 for x in deltas_pp) and d["mean"] > 2 * d["std"]:
        v = "SYGNAL-parowy+"
    elif all(x < 0 for x in deltas_pp) and -d["mean"] > 2 * d["std"]:
        v = "SYGNAL-parowy-"
    else:
        v = "SZUM"
    return v, d


def extract_or_load_fashion_feats(device: str, resize: int = 224,
                                  chunk: int = 128, cache: str = FEATS_CACHE):
    """Jednorazowa ekstrakcja cech ResNet18 (512-d) dla Fashion-MNIST;
    cache w data/. Zwraca (Ftr, ytr, Fte, yte) na device. Cechy
    DETERMINISTYCZNE (backbone zamrozony) -- niezalezne od seeda."""
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        return (d["Ftr"].to(device), d["ytr"].to(device),
                d["Fte"].to(device), d["yte"].to(device))
    import torchvision
    print(f"[G3] Jednorazowa ekstrakcja cech resnet18@{resize} dla Fashion "
          f"(pozniejsze runy czytaja cache: {os.path.abspath(cache)})",
          flush=True)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    m = torchvision.models.resnet18(weights=w)
    feat_net = nn.Sequential(*list(m.children())[:-1]).to(device).eval()
    for p in feat_net.parameters():
        p.requires_grad = False
    imnet_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    imnet_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    outs = []
    for X in (Xtr, Xte):
        parts = []
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk].view(-1, 1, 28, 28)
            x = x * FASHION_STD + FASHION_MEAN            # denorm -> [0,1]
            x = x.repeat(1, 3, 1, 1)                      # 1 -> 3 kanaly
            x = (x - imnet_mean) / imnet_std              # norm ImageNet
            x = F.interpolate(x, size=resize, mode="bilinear",
                              align_corners=False)
            with torch.no_grad():
                parts.append(feat_net(x).flatten(1))      # [b, 512]
            if (s // chunk) % 40 == 0:
                print(f"    ekstrakcja: {s}/{len(X)}", flush=True)
        outs.append(torch.cat(parts))
    Ftr, Fte = outs
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save({"Ftr": Ftr.cpu(), "ytr": ytr.cpu(),
                "Fte": Fte.cpu(), "yte": yte.cpu()}, cache)
    print(f"[G3] Cache zapisany ({Ftr.shape[0]}+{Fte.shape[0]} x "
          f"{Ftr.shape[1]}).", flush=True)
    return Ftr, ytr, Fte, yte


def holdout_means(feats_tr, ytr, feats_te, yte, attrs, seed, epochs,
                  device) -> List[float]:
    """Zwraca liste ZS per klasa (10) dla jednego seeda/slownika."""
    zs = []
    for h in range(10):
        z, _ = run_holdout(h, feats_tr, ytr, feats_te, yte, attrs, seed,
                           epochs, device)
        zs.append(z)
    return zs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mn11, _ = check_matrix(ATTRS11)
    mn21, cs21 = check_matrix(ATTRS21)
    assert mn21 >= 3, f"min Hamming attrs21 = {mn21} < 3"
    assert all(2 <= s <= 8 for s in cs21), f"osiagalnosc: col sums {cs21}"

    print("=" * 72)
    print(f"G3 -- kompozycyjnosc na pretrained ResNet18 "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epochs={epochs} | "
          f"minH attrs11={mn11} attrs21={mn21} | LINIA G3+/G3- = {GATE_PP}%")
    print("=" * 72)

    Xtr784, ytr, Xte784, yte = load_dataset("Fashion-MNIST", device)
    Ftr_pre, ytr_p, Fte_pre, yte_p = extract_or_load_fashion_feats(device)

    t0 = time.perf_counter()
    out: Dict[str, object] = {
        "experiment": "G3_pretrained", "device": device, "n_seeds": n_seeds,
        "epochs": epochs, "gate_pp": GATE_PP,
        "min_hamming": {"attrs11": mn11, "attrs21": mn21},
        "attr21_names": ATTR21_NAMES,
        "encoder_pretrained": "resnet18_IMAGENET1K_V1 penultimate 512-d "
                              "(Fashion 28->224, deterministyczne)",
        "variants": {}, "verdicts": {}}

    # zs_mean[backbone][dict] = lista per-seed srednich ZS (10 klas)
    zs_mean: Dict[str, Dict[str, List[float]]] = {
        b: {dn: [] for dn in DICTS} for b in ("random", "pretrained")}
    # per-klasa (do reguly osiagalnosci), akumulacja pretrained
    reach_acc: Dict[str, Dict[str, List[float]]] = {
        dn: {CLASS_NAMES[h]: [] for h in range(10)} for dn in DICTS}

    for seed in range(n_seeds):
        # --- random backbone: cechy per-seed (jak G2b) ---
        torch.manual_seed(seed)
        sysr = MarsCLSystem(backbone_source="random").to(device)
        rand_tr = sysr.feats_batched(Xtr784)
        rand_te = sysr.feats_batched(Xte784)
        for dn, attrs_cpu in DICTS.items():
            attrs = attrs_cpu.to(device)
            zs = holdout_means(rand_tr, ytr, rand_te, yte, attrs, seed,
                               epochs, device)
            zs_mean["random"][dn].append(sum(zs) / 10)
            print(f"[seed {seed}] random/{dn:7s}: sredni ZS "
                  f"{sum(zs)/10*100:5.1f}%")

        # --- pretrained backbone: cechy stale (seed zmienia tylko proj) ---
        for dn, attrs_cpu in DICTS.items():
            attrs = attrs_cpu.to(device)
            zs = holdout_means(Ftr_pre, ytr_p, Fte_pre, yte_p, attrs, seed,
                               epochs, device)
            zs_mean["pretrained"][dn].append(sum(zs) / 10)
            for h in range(10):
                reach_acc[dn][CLASS_NAMES[h]].append(zs[h])
            print(f"[seed {seed}] pretrained/{dn:7s}: sredni ZS "
                  f"{sum(zs)/10*100:5.1f}% | per klasa: "
                  + " ".join(f"{z*100:.0f}" for z in zs))

    for b in zs_mean:
        out["variants"][b] = {dn: stats(zs_mean[b][dn]) for dn in DICTS}

    # ---------- WERDYKT BINARNY G3+/G3- ----------
    best_pre = max(stats(zs_mean["pretrained"]["attrs11"])["mean"],
                   stats(zs_mean["pretrained"]["attrs21"])["mean"])
    g3_plus = best_pre * 100 > GATE_PP
    which = ("attrs11" if stats(zs_mean["pretrained"]["attrs11"])["mean"]
             >= stats(zs_mean["pretrained"]["attrs21"])["mean"] else "attrs21")
    out["verdicts"]["G3_binary"] = {
        "zs_pretrained_best_pp": round(best_pre * 100, 2),
        "slownik_najlepszy": which, "prog_pp": GATE_PP,
        "werdykt": "G3+" if g3_plus else "G3-",
        "opis": ("G3+ : kompozycyjnosc osiagnieta na mocnych cechach -- "
                 "dzwignia (c) potwierdzona jako THE waskie gardlo"
                 if g3_plus else
                 "G3- : granica PODEJSCIA (reczny slownik + liniowe "
                 "detektory), nie reprezentacji -- seria G domknieta")}

    if not args.smoke:
        # T1: pretrained vs random (attrs11), pary per-seed
        pre11 = zs_mean["pretrained"]["attrs11"]
        rnd11 = zs_mean["random"]["attrs11"]
        d = [(a - b) * 100 for a, b in zip(pre11, rnd11)]
        noise = (stats([x * 100 for x in pre11])["std"]
                 + stats([x * 100 for x in rnd11])["std"])
        v, ds = verdict_paired(d, noise)
        ratio = (stats(pre11)["mean"] / stats(rnd11)["mean"]
                 if stats(rnd11)["mean"] > 0 else float("inf"))
        out["verdicts"]["T1_pretrained_vs_random"] = {
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(noise, 4), "verdict": v,
            "ratio_pre_over_rand": round(ratio, 2),
            "dzwignia_cech": bool(v == "SYGNAL+" and ratio >= 2.0)}

        # T2: regula osiagalnosci na pretrained (per slownik)
        out["verdicts"]["T2_osiagalnosc_pretrained"] = {}
        for dn in DICTS:
            per = {k: round(sum(v) / len(v), 4) for k, v in reach_acc[dn].items()}
            zeros = [k for k, val in per.items() if val == 0.0]
            out["verdicts"]["T2_osiagalnosc_pretrained"][dn] = {
                "zs_per_klasa": per, "klasy_zerowe": zeros,
                "dawne_porazki": {CLASS_NAMES[h]: per[CLASS_NAMES[h]]
                                  for h in OLD_STRUCTURAL_FAILS},
                "wszystkie_osiagalne": len(zeros) == 0}

        # T3: attrs21 vs attrs11 na pretrained (interakcja a x c)
        pre21 = zs_mean["pretrained"]["attrs21"]
        d3 = [(a - b) * 100 for a, b in zip(pre21, pre11)]
        n3 = (stats([x * 100 for x in pre21])["std"]
              + stats([x * 100 for x in pre11])["std"])
        v3, ds3 = verdict_paired(d3, n3)
        out["verdicts"]["T3_ecoc_na_pretrained"] = {
            "pairs_pp": [round(x, 2) for x in d3], "delta": ds3,
            "noise_pp": round(n3, 4), "verdict": v3,
            "opis": "ECOC pomaga na mocnych cechach?" }

        # T4: sanity reprodukcji random ~ G2b
        out["verdicts"]["T4_sanity_random_repro"] = {
            "attrs11_pp": round(stats(rnd11)["mean"] * 100, 2),
            "attrs21_pp": round(stats(zs_mean["random"]["attrs21"])["mean"]
                                * 100, 2),
            "oczekiwane_G2b": "attrs11 ~3.2%, attrs21 ~0.18%"}

    # ---------- raport ----------
    print(f"\n--- G3 (n={n_seeds}) ---")
    for b in ("random", "pretrained"):
        for dn in DICTS:
            s = out["variants"][b][dn]
            print(f"  {b:10s}/{dn:7s}: ZS {s['mean']*100:5.2f}"
                  f"+/-{s['std']*100:.2f}% (min {s['min']*100:.1f})")
    g = out["verdicts"]["G3_binary"]
    print(f"\n  WERDYKT: {g['werdykt']} | zs_pretrained_best="
          f"{g['zs_pretrained_best_pp']}% ({g['slownik_najlepszy']}) "
          f"vs linia {GATE_PP}%")
    for key in ("T1_pretrained_vs_random", "T3_ecoc_na_pretrained",
                "T4_sanity_random_repro"):
        if key in out["verdicts"]:
            print(f"  {key}: {json.dumps(out['verdicts'][key], ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("G3_pretrained_smoke.json" if args.smoke else "G3_pretrained.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
