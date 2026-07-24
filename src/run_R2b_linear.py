"""
run_R2b_linear.py -- Droga R2b: wyrownanie LINIOWE (ridge) jako primary
+ kontrole (DROGA_R2B_PLAN.md). Poziom R-mild.

Promuje ablacje z R2 (RIDGE_HET 65% = 80% sufitu) do primary i
POTWIERDZA ja na SWIEZYCH seedach 5-9 z twarda bramka -- dyscyplina Q2c.

Split (zamrozony, 10-way class-IL, sufit ~80%):
  CAL_POOL=[0,1,2,3,4,5] -- 6 klas dzielonych uczonych realnie przez
    wszystkich; mapa fitowana na PIERWSZYCH K in {2,4,6} (reszta uczona
    tak samo -> projekcja odbiorcy stala, zmienia sie tylko budzet mapy).
  OWN=[6,7] wlasne odbiorcy; ADOPT=[8,9] adoptowane (metryka).

Auto-lambda: grid {1e-3,1e-2,1e-1,1,10}, wybor po rekonstrukcji na
held-out probkach klas kalibracyjnych (split 80/20; zero wgladu w
adoptowane).

Warianty (swieze seedy 5..9): CEILING (real), R0 (floor), oraz liniowe
R2b_SANITY_K / R2b_HET_K dla K in {2,4,6}.

BRAMKA (zamrozona): R2b_HET przy K=4 >= 70% * CEILING (~57% adopted) ->
wyrownanie liniowe POTWIERDZONE (do CLAIMS). Ponizej -> niepotwierdzone.

Wymaga: data/cifar_resnet18_224_feats.pt (cache z L), data/glove.6B.50d.txt.

Tryb szybki:  python src/run_R2b_linear.py --smoke
Pelny:        python src/run_R2b_linear.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import eval_protocols, make_task_data
from mars_cl_l import ReducedBackbone, extract_or_load_cifar_feats
from mars_cl_semantic import load_word_vectors
from mars_collective_hetero import MarsCollectiveHetero
from mars_translate import RidgeTranslator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR = 0.001
TEACHER_OFFSET = 1000
SEED_START = 5                       # swieze seedy 5..9 (R2 uzyl 0..4)
GATE_FRAC = 0.70                     # R2b_HET(K=4) >= 70% * CEILING
GATE_K = 4
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
K_SWEEP = (2, 4, 6)
CAL_POOL = [0, 1, 2, 3, 4, 5]        # 6 dzielonych; mapa na pierwszych K
OWN_TASK = [6, 7]
ADOPT_TASK = [8, 9]
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
              bn_calib=False, feat_signorm=False)


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
    torch.manual_seed(seed)
    m = MarsCollectiveHetero(wv, backbone_module=ReducedBackbone(), **COMMON)
    m.to(device)
    return m


def cal_features_512(Ftr, ytr, classes: Sequence[int], n_per: int
                     ) -> torch.Tensor:
    parts = [Ftr[(ytr == c).nonzero(as_tuple=True)[0][:n_per]]
             for c in classes]
    return torch.cat(parts)


def fit_ridge_autolam(ha: torch.Tensor, hb: torch.Tensor
                      ) -> Tuple[RidgeTranslator, float]:
    """Ridge H_A->H_B z auto-lambda: wybor po held-out (80/20) rekonstrukcji."""
    n = len(ha)
    perm = torch.randperm(n, device=ha.device)
    cut = max(int(0.8 * n), 1)
    tr, va = perm[:cut], perm[cut:]
    best_lam, best_mse = LAMBDA_GRID[0], float("inf")
    for lam in LAMBDA_GRID:
        t = RidgeTranslator(lam=lam).fit(ha[tr], hb[tr])
        ref_x, ref_y = (ha[va], hb[va]) if len(va) else (ha[tr], hb[tr])
        mse = float(((t.predict(ref_x) - ref_y) ** 2).mean())
        if mse < best_mse:
            best_lam, best_mse = lam, mse
    return RidgeTranslator(lam=best_lam).fit(ha, hb), best_lam


def run_combo(homo: bool, method: str, K: Optional[int], wv, cal_pool_task,
              own_task, adopt_task, eval_td, Ftr, ytr, seed: int,
              epochs: int, n_dream: int, device: str) -> Dict[str, object]:
    col = build_agent(wv, seed, device)
    col.init_representation([cal_pool_task], epochs=epochs, lr=LR, device=device)
    col.learn_task(cal_pool_task, epochs=epochs, lr=LR, device=device)  # 6 CAL
    col.learn_task(own_task, epochs=epochs, lr=LR, device=device)       # OWN

    teacher_seed = seed if homo else seed + TEACHER_OFFSET
    Ai = build_agent(wv, teacher_seed, device)
    Ai.init_representation([adopt_task], epochs=epochs, lr=LR, device=device)
    Ai.learn_task(adopt_task, epochs=epochs, lr=LR, device=device)

    lam, disparity = None, None
    if method == "real":
        payloads = {c: Ai.export_class_stats(c, int((adopt_task["ytr"] == c).sum()))
                    for c in ADOPT_TASK}
        col.adopt_classes(ADOPT_TASK, payloads, epochs=epochs, lr=LR,
                          device=device, n_dream=n_dream)
    elif method == "floor":
        col.adopt_classes_anchor_only(ADOPT_TASK, device=device)
    else:  # linear (ridge)
        cal_k = CAL_POOL[:K]
        f512 = cal_features_512(Ftr, ytr, cal_k, n_dream)
        hb_cal, ha_cal = col.backbone(f512), Ai.backbone(f512)
        lin, lam = fit_ridge_autolam(ha_cal, hb_cal)
        disparity = round(float(((lin.predict(ha_cal) - hb_cal) ** 2).mean()), 6)
        payloads_a = {c: Ai.export_class_stats(c, int((adopt_task["ytr"] == c).sum()))
                      for c in ADOPT_TASK}
        col.adopt_classes_maptransform(ADOPT_TASK, payloads_a, lin.predict,
                                       epochs=epochs, lr=LR, device=device,
                                       n_dream=n_dream, stats_k=COMMON["stats_k"])

    row, _ = eval_protocols(col.forward, eval_td, 1, list(col.seen_classes))
    adopted = row[1]                       # ADOPT=[8,9] to zadanie 1
    overall = sum(row) / len(row)
    return {"final_row_class_il": row, "adopted_ACC": round(adopted, 4),
            "overall_ACC": round(overall, 4), "lambda": lam,
            "disparity": disparity}


def _pairs(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [round((x - y) * 100, 2) for x, y in zip(a, b)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else 5000
    ks = (GATE_K,) if args.smoke else K_SWEEP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(SEED_START, SEED_START + n_seeds))

    # combos: (nazwa, homo, metoda, K)
    combos: List[Tuple[str, bool, str, Optional[int]]] = [
        ("CEILING", True, "real", None), ("R0", False, "floor", None)]
    for K in ks:
        combos.append((f"R2b_SANITY_K{K}", True, "linear", K))
        combos.append((f"R2b_HET_K{K}", False, "linear", K))

    print("=" * 72)
    print(f"R2b -- wyrownanie liniowe (ridge) primary "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seedy={seeds} | epok={epochs} | "
          f"n_dream={n_dream} | K={ks}")
    print(f"CAL_POOL={CAL_POOL} | OWN={OWN_TASK} | ADOPT={ADOPT_TASK} | "
          f"BRAMKA R2b_HET_K{GATE_K} >= {int(GATE_FRAC*100)}% * CEILING")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    Ftr, ytr, Fte, yte = extract_or_load_cifar_feats(device)
    cal_pool_task = make_task_data(Ftr, ytr, Fte, yte, tasks=[CAL_POOL])[0]
    own_task = make_task_data(Ftr, ytr, Fte, yte, tasks=[OWN_TASK])[0]
    adopt_task = make_task_data(Ftr, ytr, Fte, yte, tasks=[ADOPT_TASK])[0]
    eval_td = make_task_data(Ftr, ytr, Fte, yte, tasks=[OWN_TASK, ADOPT_TASK])

    t0 = time.perf_counter()
    out: Dict[str, object] = {
        "experiment": "R2b_linear", "level": "R-mild", "device": device,
        "seeds": seeds, "epochs_per_task": epochs, "n_dream": n_dream,
        "common": COMMON, "gate_frac": GATE_FRAC, "gate_K": GATE_K,
        "lambda_grid": list(LAMBDA_GRID), "cal_pool": CAL_POOL,
        "own_task": OWN_TASK, "adopt_task": ADOPT_TASK,
        "systems": {}, "verdicts": {}}

    per_seed: Dict[str, List[Dict[str, object]]] = {c[0]: [] for c in combos}
    for seed in seeds:
        for name, homo, method, K in combos:
            r = run_combo(homo, method, K, wv, cal_pool_task, own_task,
                          adopt_task, eval_td, Ftr, ytr, seed, epochs,
                          n_dream, device)
            per_seed[name].append(r)
            print(f"[seed {seed}] {name:15s}: adopted={r['adopted_ACC']*100:5.2f}% "
                  f"| overall={r['overall_ACC']*100:5.2f}% | lam={r['lambda']} "
                  f"disp={r['disparity']}")

    for name, *_ in combos:
        out["systems"][name] = {
            "per_seed": per_seed[name],
            "agg_adopted": stats([p["adopted_ACC"] for p in per_seed[name]]),
            "agg_overall": stats([p["overall_ACC"] for p in per_seed[name]])}

    # ---------- BRAMKA ----------
    ceil_mean = out["systems"]["CEILING"]["agg_adopted"]["mean"]
    gate_name = f"R2b_HET_K{GATE_K}"
    het_mean = out["systems"][gate_name]["agg_adopted"]["mean"]
    gate_abs = GATE_FRAC * ceil_mean
    gate_pass = het_mean >= gate_abs
    out["verdicts"]["GATE"] = {
        "gate_variant": gate_name,
        "adopted_mean_pp": round(het_mean * 100, 2),
        "prog_pp": round(gate_abs * 100, 2),
        "ceiling_pp": round(ceil_mean * 100, 2),
        "frac_of_ceiling": round(het_mean / ceil_mean, 3) if ceil_mean else None,
        "zdana": bool(gate_pass),
        "opis": ("bramka zdana -- wyrownanie liniowe POTWIERDZONE (do CLAIMS)"
                 if gate_pass else
                 "BRAMKA NIEZDANA -- obserwacja R2 niepotwierdzona, poza CLAIMS")}

    if not args.smoke and gate_pass:
        het = [p["adopted_ACC"] for p in per_seed[gate_name]]
        ceil = [p["adopted_ACC"] for p in per_seed["CEILING"]]
        r0 = [p["adopted_ACC"] for p in per_seed["R0"]]
        for key, base in ((f"{gate_name}_vs_CEILING", ceil),
                          (f"{gate_name}_vs_R0", r0)):
            d = _pairs(het, base)
            noise = (stats([x * 100 for x in het])["std"]
                     + stats([x * 100 for x in base])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"][key] = {"pairs_pp": d, "delta": ds,
                                    "noise_pp": round(noise, 4), "verdict": v}
        out["verdicts"]["krzywa_K_HET"] = {
            f"K{K}": out["systems"][f"R2b_HET_K{K}"]["agg_adopted"]["mean"]
            for K in ks}

    # ---------- raport ----------
    print(f"\n--- R2b (n={n_seeds}, seedy {seeds}) -- adopted ACC ---")
    for name, *_ in combos:
        a = out["systems"][name]["agg_adopted"]
        o = out["systems"][name]["agg_overall"]
        print(f"  {name:15s}: adopted {a['mean']*100:5.2f}+/-{a['std']*100:.2f}% "
              f"(min {a['min']*100:.2f}) | overall {o['mean']*100:5.2f}%")
    g = out["verdicts"]["GATE"]
    print(f"\n  BRAMKA {g['gate_variant']}: {g['adopted_mean_pp']:.2f}% vs prog "
          f"{g['prog_pp']:.2f}% ({int(GATE_FRAC*100)}% z sufitu "
          f"{g['ceiling_pp']:.2f}%) -> "
          f"{'ZDANA' if g['zdana'] else 'NIEZDANA'} "
          f"[{g['frac_of_ceiling']} sufitu]")
    for key in (f"{gate_name}_vs_CEILING", f"{gate_name}_vs_R0", "krzywa_K_HET"):
        if key in out["verdicts"]:
            print(f"  {key}: {json.dumps(out['verdicts'][key], ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("R2b_linear_smoke.json" if args.smoke else "R2b_linear.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
