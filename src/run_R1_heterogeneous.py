"""
run_R1_heterogeneous.py -- Droga R (poziom R-mild): kolektyw
HETEROGENICZNY na mocnych cechach (DROGA_R_PLAN.md).

R-mild = ten sam pretrained resnet18 (cache 512-d z L), ale ROZNY seed
losowej projekcji 512->128 u nadawcy i odbiorcy => rozne przestrzenie
feature_B (dwa liniowe obrazy tej samej 512). Translacja liniowa istnieje
z konstrukcji -> to jest sanity mechanizmu, nie R-hard.

Cztery warianty na TYCH SAMYCH seedach 0..4 (class-IL, Split-CIFAR-10n):
  CEILING  -- kolektyw homogeniczny (nadawca = seed odbiorcy): payload
              cech + adopt_classes (== L2). Gorna kotwica.
  R0       -- podloga hetero: adopcja sama kotwica (proto + staly pod),
              bez dekodera. Przewidziana porazka (G1).
  R1       -- hetero + dekoder: payload anchorowy (Gauss) + dekoder
              anchor->feature_B (uczony na wlasnych klasach odbiorcy) ->
              re-materializacja -> adopt_classes.
  SANITY   -- R1 przy WSPOLNYM backbone (nadawca = seed odbiorcy): sciezka
              dekodera przy zerowej heterogenicznosci; ma ~= CEILING
              (dekoder nie niszczy informacji).

Metryka (recipient-relative): ACC klas ADOPTOWANYCH u odbiorcy =
srednia row_class_il[1:] (task 0 = wlasne klasy odbiorcy). Raportujemy
tez overall ACC (porownywalne z L2).

Kryteria (Z GORY):
  R1 vs R0     : SYGNAL+ => dekoder wnosi ponad sama kotwice.
  R1 vs CEILING: SZUM => heterogenicznosc DARMOWA (R-mild); SYGNAL- =>
                 zmierzona cena. (SANITY vs CEILING raportowane obok.)

Wymaga: data/cifar_resnet18_224_feats.pt (cache z L; jesli brak, tworzony
raz), data/glove.6B.50d.txt.

Tryb szybki:  python src/run_R1_heterogeneous.py --smoke
Pelny:        python src/run_R1_heterogeneous.py
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

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import eval_protocols, make_task_data
from mars_cl_l import ReducedBackbone, extract_or_load_cifar_feats
from mars_cl_semantic import load_word_vectors
from mars_collective_hetero import MarsCollectiveHetero

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR = 0.001
TEACHER_OFFSET = 1000   # seed nadawcy hetero = seed + OFFSET + i (rozny bb)
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
              bn_calib=False, feat_signorm=False)
VARIANTS = ("CEILING", "R0", "R1", "SANITY")


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


def run_variant(variant: str, wv, task_data, seed: int, epochs: int,
                n_dream: int, device: str) -> Dict[str, object]:
    col = build_agent(wv, seed, device)
    col.init_representation([task_data[0]], epochs=epochs, lr=LR,
                            device=device)
    col.learn_task(task_data[0], epochs=epochs, lr=LR, device=device)

    dec_mse = None
    if variant in ("R1", "SANITY"):
        dec_mse = col.train_decoder(task_data[0]["classes"], n_per=n_dream,
                                    epochs=epochs, lr=LR, device=device)

    homogeneous = variant in ("CEILING", "SANITY")
    row: List[float] = []
    for i in range(1, len(task_data)):
        teacher_seed = seed if homogeneous else seed + TEACHER_OFFSET + i
        Ai = build_agent(wv, teacher_seed, device)
        Ai.init_representation([task_data[i]], epochs=epochs, lr=LR,
                               device=device)
        Ai.learn_task(task_data[i], epochs=epochs, lr=LR, device=device)
        td = task_data[i]

        if variant == "CEILING":
            payloads = {c: Ai.export_class_stats(c, int((td["ytr"] == c).sum()))
                        for c in td["classes"]}
            col.adopt_classes(td["classes"], payloads, epochs=epochs, lr=LR,
                              device=device, n_dream=n_dream)
        elif variant in ("R1", "SANITY"):
            apl = {c: Ai.export_anchor_payload(c, n_dream, device)
                   for c in td["classes"]}
            col.adopt_classes_hetero(td["classes"], apl, epochs=epochs, lr=LR,
                                     device=device, n_dream=n_dream,
                                     stats_k=COMMON["stats_k"])
        else:  # R0
            col.adopt_classes_anchor_only(td["classes"], device=device)

        row, _ = eval_protocols(col.forward, task_data, i,
                                list(col.seen_classes))

    overall = sum(row) / len(row)
    adopted = sum(row[1:]) / len(row[1:])
    return {"final_row_class_il": row, "overall_ACC": round(overall, 4),
            "adopted_ACC": round(adopted, 4),
            "dec_mse": (round(dec_mse, 6) if dec_mse is not None else None)}


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
    print(f"R1 -- kolektyw heterogeniczny (R-mild), CIFAR "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream} | warianty={VARIANTS}")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    Ftr, ytr, Fte, yte = extract_or_load_cifar_feats(device)
    task_data = make_task_data(Ftr, ytr, Fte, yte)

    t0 = time.perf_counter()
    out: Dict[str, object] = {
        "experiment": "R1_heterogeneous", "level": "R-mild",
        "device": device, "n_seeds": n_seeds, "epochs_per_task": epochs,
        "n_dream": n_dream, "common": COMMON, "teacher_offset": TEACHER_OFFSET,
        "encoder": "resnet18_IMAGENET1K_V1 -> random frozen 512->128 "
                   "(rozny seed nadawca vs odbiorca w R0/R1)",
        "systems": {}, "verdicts": {}}

    per_seed: Dict[str, List[Dict[str, object]]] = {v: [] for v in VARIANTS}
    for seed in range(n_seeds):
        for v in VARIANTS:
            r = run_variant(v, wv, task_data, seed, epochs, n_dream, device)
            per_seed[v].append(r)
            print(f"[seed {seed}] {v:8s}: overall={r['overall_ACC']*100:5.2f}% "
                  f"| adopted={r['adopted_ACC']*100:5.2f}% "
                  f"| dec_mse={r['dec_mse']}")

    for v in VARIANTS:
        out["systems"][v] = {
            "per_seed": per_seed[v],
            "agg_overall": stats([p["overall_ACC"] for p in per_seed[v]]),
            "agg_adopted": stats([p["adopted_ACC"] for p in per_seed[v]])}

    # ---------- werdykty (na ACC klas adoptowanych) ----------
    if not args.smoke:
        r1 = [p["adopted_ACC"] for p in per_seed["R1"]]
        r0 = [p["adopted_ACC"] for p in per_seed["R0"]]
        ceil = [p["adopted_ACC"] for p in per_seed["CEILING"]]
        san = [p["adopted_ACC"] for p in per_seed["SANITY"]]

        d10 = _pairs(r1, r0)
        n10 = stats([x * 100 for x in r1])["std"] + stats([x * 100 for x in r0])["std"]
        v10, s10 = verdict_paired(d10, n10)
        out["verdicts"]["R1_vs_R0"] = {
            "opis": "czy dekoder wnosi ponad sama kotwice",
            "pairs_pp": d10, "delta": s10, "noise_pp": round(n10, 4),
            "verdict": v10}

        d1c = _pairs(r1, ceil)
        n1c = stats([x * 100 for x in r1])["std"] + stats([x * 100 for x in ceil])["std"]
        v1c, s1c = verdict_paired(d1c, n1c)
        out["verdicts"]["R1_vs_CEILING"] = {
            "opis": "cena heterogenicznosci (SZUM=darmowa, SYGNAL-=cena)",
            "pairs_pp": d1c, "delta": s1c, "noise_pp": round(n1c, 4),
            "verdict": v1c}

        dsc = _pairs(san, ceil)
        nsc = stats([x * 100 for x in san])["std"] + stats([x * 100 for x in ceil])["std"]
        vsc, ssc = verdict_paired(dsc, nsc)
        out["verdicts"]["SANITY_vs_CEILING"] = {
            "opis": "sciezka dekodera przy zerowej heterogenicznosci ~= 0",
            "pairs_pp": dsc, "delta": ssc, "noise_pp": round(nsc, 4),
            "verdict": vsc}

    # ---------- raport ----------
    print(f"\n--- R1 (n={n_seeds}) -- CIFAR-n, class-IL (ACC adoptowanych) ---")
    for v in VARIANTS:
        a = out["systems"][v]["agg_adopted"]
        o = out["systems"][v]["agg_overall"]
        print(f"  {v:8s}: adopted {a['mean']*100:5.2f}+/-{a['std']*100:.2f}% "
              f"(min {a['min']*100:.2f}) | overall {o['mean']*100:5.2f}%")
    for key in ("R1_vs_R0", "R1_vs_CEILING", "SANITY_vs_CEILING"):
        if key in out["verdicts"]:
            vd = out["verdicts"][key]
            print(f"  {key:18s} (prog {vd['noise_pp']:.2f}pp): "
                  f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
                  f"(min {vd['delta']['min']:+.2f}) | pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("R1_heterogeneous_smoke.json" if args.smoke
             else "R1_heterogeneous.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
