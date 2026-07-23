"""
run_P1_detect_pretrained.py -- P1: detekcja zatrucia payloadu na cechach
semantycznych vs losowych (DROGA_P_PLAN.md).

NOWY plik (branch droga-p) -- kod v1.0 NIETKNIETY.

Setup = I4 przeniesiony na Split-CIFAR-10n, DWA podloza w jednym runie
(te same seedy):
  P1a pretrained : ReducedBackbone (cache cech resnet18 512-d, losowa
                   projekcja 512->128 z seeda) -- jak L1/L2,
  P1b random     : CifarBackbone na pikselach -- jak J2b (kontrola
                   przypisania efektu; oczekiwana replikacja negatywu I4).

B uczy taski 0-3 (klasy 0-7); A uczy task 4; payload 8 w wariantach
clean / swap (payload 9 jako 8) / noise. Payload 9 zawsze clean.

Detektory (identyczny kod co I4 -- mars_cl_i4):
  D1 rank_consistency (bez adopcji), D2 canary_probe (adopcja na kopii).

Kryteria (Z GORY, DROGA_P_PLAN.md):
  PELNA SEPARACJA = kazdy clean po wlasciwej stronie OBU atakow, 5/5
  (min-max bez przeciecia).
  SUKCES MOCNY: separacja na pretrained AND brak na random (ten sam
  detektor). SUKCES SLABY: separacja tylko clean-vs-swap na pretrained.
  NEGATYW: brak separacji na pretrained. Anomalia: separacja na random.
  Obserwacje: mapa szkody pary-vs-clean (acc8/acc9/acc_own).

Wymaga: data/glove.6B.50d.txt, cache cech L (powstanie automatycznie).

Tryb szybki:  python src/run_P1_detect_pretrained.py --smoke
Pelny:        python src/run_P1_detect_pretrained.py  (~25-40 min)
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cifar_cl import CifarBackbone
from cl_common import make_task_data
from mars_cl_i4 import (canary_probe, forge_noise, forge_swap,
                        rank_consistency)
from mars_cl_j import load_cifar10_norm
from mars_cl_l import ReducedBackbone, extract_or_load_cifar_feats
from mars_cl_n import class_accs
from mars_cl_semantic import load_word_vectors
from mars_collective import MarsCollective

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
VARIANTS = ("clean", "swap", "noise")
BACKBONES = ("pretrained", "random")


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def verdict_paired(deltas_pp, noise_pp):
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


def separation(clean_vals, attack_vals_list, higher_is_clean):
    """Pelna separacja: kazdy clean po wlasciwej stronie kazdego ataku."""
    attacks = [v for vals in attack_vals_list for v in vals]
    if higher_is_clean:
        return min(clean_vals) > max(attacks)
    return max(clean_vals) < min(attacks)


def build(bb_name, wv, seed, device):
    torch.manual_seed(seed)
    bb = ReducedBackbone() if bb_name == "pretrained" else CifarBackbone()
    m = MarsCollective(wv, backbone_module=bb, **CFG)
    m.to(device)
    return m


def run_seed(bb_name, wv, task_data, seed, epochs, n_dream, n_probe,
             device):
    """Jeden seed na jednym podlozu: B(0-3), A(4), detekcja + adopcja."""
    td_own = task_data[:4]
    td4 = task_data[4]
    # odbiorca B PRZED nadawca A -- higiena RNG jak I1/I4
    B = build(bb_name, wv, seed, device)
    B.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    for t in range(4):
        B.learn_task(task_data[t], epochs=epochs, lr=LR, device=device)
    A = build(bb_name, wv, seed, device)
    A.init_representation([td4], epochs=epochs, lr=LR, device=device)
    A.learn_task(td4, epochs=epochs, lr=LR, device=device)

    n8 = int((td4["ytr"] == 8).sum())
    n9 = int((td4["ytr"] == 9).sum())
    clean8 = A.export_class_stats(8, n8)
    clean9 = A.export_class_stats(9, n9)
    with torch.no_grad():
        pool = B.feats_batched(torch.cat([td["Xtr"] for td in task_data]))
    payload8 = {"clean": clean8,
                "swap": forge_swap(A.export_class_stats(9, n9)),
                "noise": forge_noise(pool, k=CFG["stats_k"], n=n_dream,
                                     seed=seed)}

    rec = {}
    for var in VARIANTS:
        d1 = rank_consistency(B, payload8[var], 8, device=device)
        d2 = canary_probe(B, [8, 9], {8: payload8[var], 9: clean9},
                          td_own, epochs=epochs, lr=LR, device=device,
                          n_dream=n_probe)
        Bv = copy.deepcopy(B)
        Bv.adopt_classes([8, 9], {8: payload8[var], 9: clean9},
                         epochs=epochs, lr=LR, device=device,
                         n_dream=n_dream)
        a = class_accs(Bv, task_data, allowed=Bv.seen_classes)
        rec[var] = {"D1_rank": round(d1, 4),
                    "D2_canary_pp": round(d2, 4),
                    "acc8": round(a[8], 4), "acc9": round(a[9], 4),
                    "acc_own": round(sum(a[c] for c in range(8)) / 8, 4)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else 5000
    n_probe = 128 if args.smoke else 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"P1 -- detekcja zatrucia: pretrained vs random "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream} | warianty={list(VARIANTS)}")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    data = {}
    Ftr, ytr_f, Fte, yte_f = extract_or_load_cifar_feats(device)
    data["pretrained"] = make_task_data(Ftr, ytr_f, Fte, yte_f)
    Xtr, ytr_x, Xte, yte_x = load_cifar10_norm(device)
    data["random"] = make_task_data(Xtr, ytr_x, Xte, yte_x)

    t0 = time.perf_counter()
    out = {"experiment": "P1_detect_pretrained", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "n_dream": n_dream, "cfg": CFG,
           "encoders": {
               "pretrained": "resnet18_IMAGENET1K_V1 -> random frozen "
                             "512->128 (ReducedBackbone, jak L1/L2)",
               "random": "CifarBackbone na pikselach (jak J2b)"},
           "per_seed": {bb: [] for bb in BACKBONES}, "verdicts": {}}

    for bb in BACKBONES:
        for seed in range(n_seeds):
            rec = run_seed(bb, wv, data[bb], seed, epochs, n_dream,
                           n_probe, device)
            out["per_seed"][bb].append(rec)
            print(f"[{bb}] seed {seed}: "
                  + " | ".join(f"{v}: acc8={rec[v]['acc8']*100:.1f} "
                               f"D1={rec[v]['D1_rank']:+.2f} "
                               f"D2={rec[v]['D2_canary_pp']:+.2f}pp"
                               for v in VARIANTS))

    if not args.smoke:
        sep = {}
        for bb in BACKBONES:
            ps = out["per_seed"][bb]
            # szkoda (obserwacja -- spojnosc z I4)
            for metric in ("acc8", "acc9", "acc_own"):
                for var in ("swap", "noise"):
                    d = [(p[var][metric] - p["clean"][metric]) * 100
                         for p in ps]
                    base = [p["clean"][metric] * 100 for p in ps]
                    new = [p[var][metric] * 100 for p in ps]
                    noise_t = stats(base)["std"] + stats(new)["std"]
                    v, ds = verdict_paired(d, noise_t)
                    out["verdicts"][f"{bb}_szkoda_{var}_{metric}"] = {
                        "pairs_pp": [round(x, 2) for x in d],
                        "delta": ds, "noise_pp": round(noise_t, 4),
                        "verdict": v, "ranga": "obserwacja"}
            # detekcja (werdykt glowny)
            for det, hic in (("D1_rank", True), ("D2_canary_pp", False)):
                cl = [p["clean"][det] for p in ps]
                sw = [p["swap"][det] for p in ps]
                no = [p["noise"][det] for p in ps]
                full = separation(cl, [sw, no], hic)
                swap_only = separation(cl, [sw], hic)
                sep[(bb, det)] = (full, swap_only)
                out["verdicts"][f"{bb}_detekcja_{det}"] = {
                    "clean": [round(x, 3) for x in cl],
                    "swap": [round(x, 3) for x in sw],
                    "noise": [round(x, 3) for x in no],
                    "pelna_separacja": full,
                    "separacja_clean_vs_swap": swap_only}
        # werdykt koncowy wg DROGA_P_PLAN.md
        for det in ("D1_rank", "D2_canary_pp"):
            p_full, p_swap = sep[("pretrained", det)]
            r_full, r_swap = sep[("random", det)]
            if r_full or r_swap:
                w = "ANOMALIA KONTROLNA (separacja na random)"
            elif p_full:
                w = "SUKCES MOCNY (separacja tylko na pretrained)"
            elif p_swap:
                w = "SUKCES SLABY (tylko clean-vs-swap na pretrained)"
            else:
                w = "NEGATYW (semantyka cech nie wystarcza)"
            out["verdicts"][f"WERDYKT_{det}"] = w

    print(f"\n--- P1 (n={n_seeds}) ---")
    for key, vd in out.get("verdicts", {}).items():
        if key.startswith("WERDYKT"):
            print(f"  {key}: {vd}")
        elif "detekcja" in key:
            print(f"  {key}: pelna={vd['pelna_separacja']} "
                  f"swap-only={vd['separacja_clean_vs_swap']} "
                  f"| clean {vd['clean']} | swap {vd['swap']} | "
                  f"noise {vd['noise']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("P1_detect_pretrained_smoke.json" if args.smoke
             else "P1_detect_pretrained.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
