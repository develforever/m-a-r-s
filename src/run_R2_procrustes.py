"""
run_R2_procrustes.py -- Droga R2: kolektyw heterogeniczny przez
wyrownanie cech ortogonalnym Procrustesem (DROGA_R2_PLAN.md).
Poziom R-mild (ten sam resnet18, rozny seed projekcji 512->128).

Sciezka zdiagnozowana przez ORACLE_SANITY=4% w R1b: translacja w
PRZESTRZENI CECH zachowuje wewnatrzklasowa geometrie (kotwica 50-d ja
zwijala). Omega: H_A -> H_B na K=4 klasach dzielonych; payload cech
jak w I; adopt_classes NIETKNIETE.

Split (jak R1b): CAL=[0,1,2,3] dzielone (publiczny zbior kalibracyjny
do Omega -- te same obrazy przez oba backbone'y), OWN=[4,5],
ADOPT=[6,7]+[8,9] (metryka). Metryka = ACC klas adoptowanych.

Warianty (te same seedy 0..4):
  CEILING    -- homogeniczny, payload cech (== L2). Gorna kotwica.
  R0         -- hetero, podloga anchor-only.
  R2_SANITY  -- homogeniczny, sciezka Procrustesa (Omega ~ I).
                *** BRAMKA: adopted >= 65% ***.
  R2_HET     -- heterogeniczny R-mild, Procrustes. Wynik glowny.
  RIDGE_HET  -- hetero, mapa liniowa (ridge, nieortogonalna) -- ablacja.

BRAMKA (zamrozona): R2_SANITY < 65% => R2 sfalsyfikowane (maszyneria
wyrownania niszczy informacje nawet przy Omega~I), poza CLAIMS.
Werdykty heterogenicznosci (R2_HET vs CEILING/R0) TYLKO po zdanej bramce.

Wymaga: data/cifar_resnet18_224_feats.pt (cache z L), data/glove.6B.50d.txt.

Tryb szybki:  python src/run_R2_procrustes.py --smoke
Pelny:        python src/run_R2_procrustes.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import eval_protocols, make_task_data
from mars_cl_l import ReducedBackbone, extract_or_load_cifar_feats
from mars_cl_semantic import load_word_vectors
from mars_collective_hetero import MarsCollectiveHetero
from mars_translate import ProcrustesAlign, RidgeTranslator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR = 0.001
TEACHER_OFFSET = 1000
GATE_PP = 65.0
RIDGE_LAM = 0.1
CAL_CLASSES = [0, 1, 2, 3]
OWN_TASK = [4, 5]
ADOPT_TASKS = [[6, 7], [8, 9]]
PROTOCOL_TASKS = [OWN_TASK] + ADOPT_TASKS
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
              bn_calib=False, feat_signorm=False)

VARIANTS: Dict[str, Tuple[bool, str]] = {
    "CEILING":   (True,  "real"),
    "R0":        (False, "floor"),
    "R2_SANITY": (True,  "procrustes"),
    "R2_HET":    (False, "procrustes"),
    "RIDGE_HET": (False, "ridge"),
}


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


def build_agent(wv, seed: int, device: str) -> MarsCollectiveHetero:
    torch.manual_seed(seed)   # rozny seed => rozna losowa projekcja 512->128
    m = MarsCollectiveHetero(wv, backbone_module=ReducedBackbone(), **COMMON)
    m.to(device)
    return m


def cal_features_512(Ftr, ytr, classes: Sequence[int], n_per: int
                     ) -> torch.Tensor:
    """Macierz cech 512-d dla klas kalibracyjnych (te same wiersze dla
    obu backbone'ow -> pary per-probka do Procrustesa)."""
    parts = [Ftr[(ytr == c).nonzero(as_tuple=True)[0][:n_per]]
             for c in classes]
    return torch.cat(parts)


def run_variant(name: str, wv, cal_task, protocol_td, Ftr, ytr, seed: int,
                epochs: int, n_dream: int, device: str) -> Dict[str, object]:
    homo, method = VARIANTS[name]
    col = build_agent(wv, seed, device)
    col.init_representation([cal_task], epochs=epochs, lr=LR, device=device)
    col.learn_task(cal_task, epochs=epochs, lr=LR, device=device)     # CAL real
    col.learn_task(protocol_td[0], epochs=epochs, lr=LR, device=device)  # OWN

    hb_cal = None
    if method in ("procrustes", "ridge"):
        f512_cal = cal_features_512(Ftr, ytr, CAL_CLASSES, n_dream)
        hb_cal = col.backbone(f512_cal)                # H_B na kalibracji

    row: List[float] = []
    disparities: List[float] = []
    for i in range(1, len(protocol_td)):
        teacher_seed = seed if homo else seed + TEACHER_OFFSET + i
        Ai = build_agent(wv, teacher_seed, device)
        Ai.init_representation([protocol_td[i]], epochs=epochs, lr=LR,
                               device=device)
        Ai.learn_task(protocol_td[i], epochs=epochs, lr=LR, device=device)
        td = protocol_td[i]

        if method == "real":
            payloads = {c: Ai.export_class_stats(c, int((td["ytr"] == c).sum()))
                        for c in td["classes"]}
            col.adopt_classes(td["classes"], payloads, epochs=epochs, lr=LR,
                              device=device, n_dream=n_dream)
        elif method == "floor":
            col.adopt_classes_anchor_only(td["classes"], device=device)
        else:  # procrustes / ridge
            ha_cal = Ai.backbone(f512_cal)             # H_A na tej samej kalibr.
            payloads_a = {c: Ai.export_class_stats(c, int((td["ytr"] == c).sum()))
                          for c in td["classes"]}
            if method == "procrustes":
                al = ProcrustesAlign().fit(ha_cal, hb_cal)
                disparities.append(al.disparity(ha_cal, hb_cal))
                omega = al.omega

                def map_fn(h, o=omega):
                    return (h @ o).clamp_min(0.0)
            else:  # ridge
                lin = RidgeTranslator(lam=RIDGE_LAM).fit(ha_cal, hb_cal)
                disparities.append(float(((lin.predict(ha_cal) - hb_cal) ** 2)
                                         .mean()))
                map_fn = lin.predict
            col.adopt_classes_maptransform(td["classes"], payloads_a, map_fn,
                                           epochs=epochs, lr=LR, device=device,
                                           n_dream=n_dream,
                                           stats_k=COMMON["stats_k"])

        row, _ = eval_protocols(col.forward, protocol_td, i,
                                list(col.seen_classes))

    overall = sum(row) / len(row)
    adopted = sum(row[1:]) / len(row[1:])
    return {"final_row_class_il": row, "overall_ACC": round(overall, 4),
            "adopted_ACC": round(adopted, 4),
            "disparity": (round(sum(disparities) / len(disparities), 6)
                          if disparities else None)}


def _pairs(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [round((x - y) * 100, 2) for x, y in zip(a, b)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else 5000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"R2 -- kolektyw heterogeniczny przez Procrustes (R-mild) "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream}")
    print(f"CAL={CAL_CLASSES} | OWN={OWN_TASK} | ADOPT={ADOPT_TASKS} | "
          f"BRAMKA R2_SANITY>={GATE_PP}%")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    Ftr, ytr, Fte, yte = extract_or_load_cifar_feats(device)
    cal_task = make_task_data(Ftr, ytr, Fte, yte, tasks=[CAL_CLASSES])[0]
    protocol_td = make_task_data(Ftr, ytr, Fte, yte, tasks=PROTOCOL_TASKS)

    t0 = time.perf_counter()
    out: Dict[str, object] = {
        "experiment": "R2_procrustes", "level": "R-mild", "device": device,
        "n_seeds": n_seeds, "epochs_per_task": epochs, "n_dream": n_dream,
        "common": COMMON, "gate_pp": GATE_PP, "cal_classes": CAL_CLASSES,
        "own_task": OWN_TASK, "adopt_tasks": ADOPT_TASKS,
        "encoder": "resnet18_IMAGENET1K_V1 -> random frozen 512->128 "
                   "(rozny seed nadawca vs odbiorca w R0/R2_HET/RIDGE_HET)",
        "systems": {}, "verdicts": {}}

    per_seed: Dict[str, List[Dict[str, object]]] = {v: [] for v in VARIANTS}
    for seed in range(n_seeds):
        for name in VARIANTS:
            r = run_variant(name, wv, cal_task, protocol_td, Ftr, ytr, seed,
                            epochs, n_dream, device)
            per_seed[name].append(r)
            print(f"[seed {seed}] {name:11s}: adopted={r['adopted_ACC']*100:5.2f}% "
                  f"| overall={r['overall_ACC']*100:5.2f}% "
                  f"| disparity={r['disparity']}")

    for name in VARIANTS:
        out["systems"][name] = {
            "per_seed": per_seed[name],
            "agg_adopted": stats([p["adopted_ACC"] for p in per_seed[name]]),
            "agg_overall": stats([p["overall_ACC"] for p in per_seed[name]])}

    gate_val = out["systems"]["R2_SANITY"]["agg_adopted"]["mean"] * 100
    gate_pass = gate_val >= GATE_PP
    out["verdicts"]["GATE_R2_SANITY"] = {
        "adopted_mean_pp": round(gate_val, 2), "prog_pp": GATE_PP,
        "zdana": gate_pass,
        "opis": ("bramka zdana -- interpretacja heterogenicznosci odblokowana"
                 if gate_pass else
                 "BRAMKA NIEZDANA -- R2 sfalsyfikowane; poza CLAIMS")}

    if not args.smoke and gate_pass:
        het = [p["adopted_ACC"] for p in per_seed["R2_HET"]]
        ceil = [p["adopted_ACC"] for p in per_seed["CEILING"]]
        r0 = [p["adopted_ACC"] for p in per_seed["R0"]]
        rid = [p["adopted_ACC"] for p in per_seed["RIDGE_HET"]]
        for key, base in (("R2_HET_vs_CEILING", ceil), ("R2_HET_vs_R0", r0),
                          ("R2_HET_vs_RIDGE", rid)):
            d = _pairs(het, base)
            noise = (stats([x * 100 for x in het])["std"]
                     + stats([x * 100 for x in base])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"][key] = {"pairs_pp": d, "delta": ds,
                                    "noise_pp": round(noise, 4), "verdict": v}

    # ---------- raport ----------
    print(f"\n--- R2 (n={n_seeds}) -- adopted ACC (class-IL) ---")
    for name in VARIANTS:
        a = out["systems"][name]["agg_adopted"]
        o = out["systems"][name]["agg_overall"]
        print(f"  {name:11s}: adopted {a['mean']*100:5.2f}+/-{a['std']*100:.2f}% "
              f"(min {a['min']*100:.2f}) | overall {o['mean']*100:5.2f}%")
    g = out["verdicts"]["GATE_R2_SANITY"]
    print(f"\n  BRAMKA R2_SANITY: {g['adopted_mean_pp']:.2f}% vs prog "
          f"{GATE_PP}% -> {'ZDANA' if g['zdana'] else 'NIEZDANA (R2 sfalsyfikowane)'}")
    for key in ("R2_HET_vs_CEILING", "R2_HET_vs_R0", "R2_HET_vs_RIDGE"):
        if key in out["verdicts"]:
            vd = out["verdicts"][key]
            print(f"  {key:18s} (prog {vd['noise_pp']:.2f}pp): "
                  f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
                  f"(min {vd['delta']['min']:+.2f}) | pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("R2_procrustes_smoke.json" if args.smoke else "R2_procrustes.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
