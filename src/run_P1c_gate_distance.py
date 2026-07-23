"""
run_P1c_gate_distance.py -- P1c: brama strukturalna D1 (prog 0.45)
i prawo dystansu semantycznego dla swap (DROGA_P1C_PLAN.md).

NOWY plik (branch droga-p) -- istniejacy kod NIETKNIETY.

Swieze seedy 5-9 (P1 uzylo 0-4). Deklarowana klasa zawsze 8 (ship);
donor d wg cosinusow kotwic 50d (zapisane w planie PRZED runem):
  swap_close: d=0 airplane (+0.775)
  swap_mid  : d=7 horse    (+0.487)
  swap_far  : d=4 deer     (+0.139)
Dla donora d: B uczy 4 taski z {0..9}\\{8,d} (pary rosnaco); A uczy
{d,8}; payload 8 = clean / swap_d (stats d jako 8) / noise. Mierzone:
D1 rank_consistency, D2 canary_probe (kopia, n=2000). BEZ pelnej
adopcji (mapa szkody zmierzona w P1).

Kryteria (Z GORY, DROGA_P1C_PLAN.md):
  a) brama: SUKCES = 100% clean > 0.45 AND 100% noise < 0.45 na OBU
     podlozach (60/60); CZESCIOWY = jedno podloze; NEGATYW = zadne.
  b) dystans: MOCNY = na obu podlozach separacja clean-vs-swap_far
     (min clean > max swap_far, 5/5) AND brak separacji
     clean-vs-swap_close; SLABY = jedno podloze lub sama monotonia
     median (close>mid>far); NEGATYW = brak obu.
  c) znak D2 (obserwacja): median D2(swap_d) < median D2(clean).

Wymaga: data/glove.6B.50d.txt, cache cech L.

Tryb szybki:  python src/run_P1c_gate_distance.py --smoke
Pelny:        python src/run_P1c_gate_distance.py  (~10-20 min)
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cifar_cl import CifarBackbone
from cl_common import make_task_data
from mars_cl_i4 import canary_probe, forge_noise, forge_swap, rank_consistency
from mars_cl_j import load_cifar10_norm
from mars_cl_l import ReducedBackbone, extract_or_load_cifar_feats
from mars_cl_semantic import load_word_vectors
from mars_collective import MarsCollective

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR = 0.001
CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
           bn_calib=False, feat_signorm=False)
DECLARED = 8
DONORS = {"swap_close": 0, "swap_mid": 7, "swap_far": 4}  # wg planu
ANCHOR_COS = {"swap_close": 0.775, "swap_mid": 0.487, "swap_far": 0.139}
GATE_THETA = 0.45
BACKBONES = ("pretrained", "random")
SEEDS = (5, 6, 7, 8, 9)


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def tasks_for_donor(d):
    """4 taski B (pary rosnaco z {0..9}\\{8,d}) + task A {d,8}."""
    rest = [c for c in range(10) if c not in (DECLARED, d)]
    tasks_b = [tuple(rest[i:i + 2]) for i in range(0, 8, 2)]
    task_a = tuple(sorted((d, DECLARED)))
    return tasks_b, task_a


def build(bb_name, wv, seed, device):
    torch.manual_seed(seed)
    bb = ReducedBackbone() if bb_name == "pretrained" else CifarBackbone()
    m = MarsCollective(wv, backbone_module=bb, **CFG)
    m.to(device)
    return m


def run_cell(bb_name, wv, X, seed, d, epochs, n_probe, device):
    """Jedna komorka (podloze, donor, seed): D1/D2 dla 3 wariantow."""
    Xtr, ytr, Xte, yte = X
    tasks_b, task_a = tasks_for_donor(d)
    td_b = make_task_data(Xtr, ytr, Xte, yte, tasks=tasks_b)
    td_a = make_task_data(Xtr, ytr, Xte, yte, tasks=[task_a])[0]

    B = build(bb_name, wv, seed, device)
    B.init_representation(td_b, epochs=epochs, lr=LR, device=device)
    for td in td_b:
        B.learn_task(td, epochs=epochs, lr=LR, device=device)
    A = build(bb_name, wv, seed, device)
    A.init_representation([td_a], epochs=epochs, lr=LR, device=device)
    A.learn_task(td_a, epochs=epochs, lr=LR, device=device)

    n8 = int((td_a["ytr"] == DECLARED).sum())
    nd = int((td_a["ytr"] == d).sum())
    clean8 = A.export_class_stats(DECLARED, n8)
    clean_d = A.export_class_stats(d, nd)
    with torch.no_grad():
        pool = B.feats_batched(torch.cat([td["Xtr"] for td in td_b]
                                         + [td_a["Xtr"]]))
    payload8 = {"clean": clean8,
                "swap": forge_swap(A.export_class_stats(d, nd)),
                "noise": forge_noise(pool, k=CFG["stats_k"], n=5000,
                                     seed=seed)}

    rec = {}
    for var in ("clean", "swap", "noise"):
        d1 = rank_consistency(B, payload8[var], DECLARED, device=device)
        d2 = canary_probe(B, [d, DECLARED],
                          {d: clean_d, DECLARED: payload8[var]},
                          td_b, epochs=epochs, lr=LR, device=device,
                          n_dream=n_probe)
        rec[var] = {"D1": round(d1, 4), "D2_pp": round(d2, 4)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = SEEDS[:1] if args.smoke else SEEDS
    epochs = 4 if args.smoke else 15
    n_probe = 128 if args.smoke else 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"P1c -- brama D1 + dystans semantyczny "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={list(seeds)} | epok={epochs} | "
          f"donory={DONORS} | theta={GATE_THETA}")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    data = {"pretrained": extract_or_load_cifar_feats(device),
            "random": load_cifar10_norm(device)}

    t0 = time.perf_counter()
    out = {"experiment": "P1c_gate_distance", "device": device,
           "seeds": list(seeds), "epochs_per_task": epochs,
           "cfg": CFG, "gate_theta": GATE_THETA, "donors": DONORS,
           "anchor_cos": ANCHOR_COS,
           "per_cell": {bb: {v: [] for v in DONORS} for bb in BACKBONES},
           "verdicts": {}}

    for bb in BACKBONES:
        for vname, d in DONORS.items():
            for seed in seeds:
                rec = run_cell(bb, wv, data[bb], seed, d, epochs,
                               n_probe, device)
                out["per_cell"][bb][vname].append(rec)
                print(f"[{bb}][{vname} d={d}] seed {seed}: "
                      + " | ".join(f"{v}: D1={rec[v]['D1']:+.3f} "
                                   f"D2={rec[v]['D2_pp']:+.2f}pp"
                                   for v in ("clean", "swap", "noise")))

    if not args.smoke:
        # ---------- a) brama strukturalna ----------
        gate = {}
        for bb in BACKBONES:
            cl = [r["clean"]["D1"] for v in DONORS
                  for r in out["per_cell"][bb][v]]
            no = [r["noise"]["D1"] for v in DONORS
                  for r in out["per_cell"][bb][v]]
            sw = [r["swap"]["D1"] for v in DONORS
                  for r in out["per_cell"][bb][v]]
            ok = (all(x > GATE_THETA for x in cl)
                  and all(x < GATE_THETA for x in no))
            gate[bb] = ok
            out["verdicts"][f"brama_{bb}"] = {
                "clean_min": round(min(cl), 3),
                "noise_max": round(max(no), 3),
                "swap_przechodzi_przez_brame": sum(
                    1 for x in sw if x > GATE_THETA),
                "swap_n": len(sw), "spelnione": ok}
        w = ("SUKCES" if all(gate.values())
             else "CZESCIOWY" if any(gate.values()) else "NEGATYW")
        out["verdicts"]["WERDYKT_a_brama"] = w

        # ---------- b) dystans semantyczny ----------
        dist = {}
        for bb in BACKBONES:
            cells = out["per_cell"][bb]
            cl_min = {v: min(r["clean"]["D1"] for r in cells[v])
                      for v in DONORS}
            sw_max = {v: max(r["swap"]["D1"] for r in cells[v])
                      for v in DONORS}
            sep = {v: cl_min[v] > sw_max[v] for v in DONORS}
            med = {v: median([r["swap"]["D1"] for r in cells[v]])
                   for v in DONORS}
            mono = med["swap_close"] > med["swap_mid"] > med["swap_far"]
            dist[bb] = {"sep": sep, "mono": mono}
            out["verdicts"][f"dystans_{bb}"] = {
                "separacja": sep,
                "mediany_D1_swap": {v: round(med[v], 3) for v in DONORS},
                "monotonia_close>mid>far": mono}
        strong = {bb: dist[bb]["sep"]["swap_far"]
                  and not dist[bb]["sep"]["swap_close"]
                  for bb in BACKBONES}
        if all(strong.values()):
            w = "SUKCES MOCNY (prawo dystansu na obu podlozach)"
        elif any(strong.values()) or any(dist[bb]["mono"]
                                         for bb in BACKBONES):
            w = "SUKCES SLABY (jedno podloze lub sama monotonia)"
        else:
            w = "NEGATYW (dystans nie jest osia wykrywalnosci)"
        out["verdicts"]["WERDYKT_b_dystans"] = w

        # ---------- c) znak D2 (obserwacja) ----------
        for bb in BACKBONES:
            cells = out["per_cell"][bb]
            med_cl = median([r["clean"]["D2_pp"] for v in DONORS
                             for r in cells[v]])
            rep = {v: sum(1 for r in cells[v]
                          if r["swap"]["D2_pp"] < r["clean"]["D2_pp"])
                   for v in DONORS}
            out["verdicts"][f"obs_c_znakD2_{bb}"] = {
                "median_D2_clean": round(med_cl, 3),
                "seedy_swap<clean_per_donor": rep,
                "ranga": "obserwacja"}

    print(f"\n--- P1c (seeds={list(seeds)}) ---")
    for key, vd in out.get("verdicts", {}).items():
        print(f"  {key}: {json.dumps(vd, ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("P1c_gate_distance_smoke.json" if args.smoke
             else "P1c_gate_distance.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
