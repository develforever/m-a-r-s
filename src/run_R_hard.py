"""
run_R_hard.py -- Droga R-hard: kolektyw między RÓŻNYMI reprezentacjami
(DROGA_R_HARD_PLAN.md). Prostokątna mapa liniowa (ridge) H_A(D_A)->H_B(D_B).

Krzyżujemy backbone LOSOWY-OD-ZERA (surowe piksele, CifarBackbone 3072->128)
z PRETRENOWANYM resnet18 (cache 512-d) -- RÓŻNA treść ORAZ RÓŻNE wymiary.
Rdzeń I/L i adopt_classes NIETKNIĘTE; nadawca 512-d eksportuje payload BEZ
proj/podów (feature_payload_from_feats), kalibracja per-próbka przez dwa
fronty na tych samych obrazach (paired_calib_feats), prostokątny
RidgeTranslator + adopt_classes_maptransform (dim-agnostyczny) -- bez zmian.

Ograniczenie rdzenia (mars_cl.py hardkoduje BB_H=128 w proj/podach): pełny
ODBIORCA jest zawsze 128-d; 512 tylko po stronie NADAWCY. Dlatego prostokątny
transfer jest core-safe w kierunku foundation(512)->scratch(128).

Kierunki (metryka = adopted ACC względem LOKALNEGO sufitu ODBIORCY):
  DIR-F->S (PRIMARY, prostokątny 512->128): pretrained -> scratch.
  DIR-S->F (obserwacja asymetrii, kwadratowy 128->128): scratch -> reduced.

Warianty (świeże seedy 5..9):
  CEILING_S / R0_S            -- sufit i podłoga odbiorcy-scratcha.
  CEILING_PT / R0_PT          -- sufit i podłoga odbiorcy-reduced.
  HET_FS_K{2,4,6}   (PRIMARY) -- foundation(512) -> scratch(128).
  SANITY_RECT_K{2,4,6}        -- foundation(512) -> reduced(128); mapa
                                 512->128 istnieje z konstrukcji (kontrola
                                 maszynerii prostokątnej + parowania frontów).
  HET_SF_K{2,4,6}             -- scratch(128) -> reduced(128) (asymetria).

BRAMKA (zamrożona, DROGA_R_HARD_PLAN.md, X=50% zatwierdzone):
  PRIMARY:  HET_FS_K4      >= 0.50 * CEILING_S
  KONIECZNE: HET_FS_K4 vs R0_S = SYGNAL+
  SANITY:   SANITY_RECT_K4 >= 0.65 * CEILING_PT (inaczej interpretacja
            zablokowana). DIR-S->F raportowany OSOBNO (asymetria).

Wymaga: data/cifar_resnet18_224_feats.pt (cache 512 z L), data/glove.6B.50d.txt.

Tryb szybki:  python src/run_R_hard.py --smoke
Pełny:        python src/run_R_hard.py
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
from cifar_cl import CifarBackbone
from cl_common import eval_protocols, make_task_data
from mars_cl_j import load_cifar10_norm
from mars_cl_l import ReducedBackbone, extract_or_load_cifar_feats
from mars_cl_semantic import load_word_vectors
from mars_collective_hetero import (MarsCollectiveHetero, _identity,
                                    class_indices, feature_payload_from_feats,
                                    feats_through_front, paired_calib_feats)
from mars_translate import RidgeTranslator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR = 0.001
TEACHER_OFFSET = 1000
SEED_START = 5                       # świeże seedy 5..9 (parytet z R2b)
GATE_FRAC = 0.50                     # X=50% ZATWIERDZONE (Robert, 2026-07-24)
SANITY_FRAC = 0.65                   # SANITY_RECT_K4 >= 65% * CEILING_PT
GATE_K = 4
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
K_SWEEP = (2, 4, 6)
CAL_POOL = [0, 1, 2, 3, 4, 5]        # 6 dzielonych; mapa na pierwszych K
OWN_TASK = [6, 7]
ADOPT_TASK = [8, 9]
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
              bn_calib=False, feat_signorm=False)   # config J2b (proven)


# ------------------------------------------------------------------ statystyka
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


def _pairs(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [round((x - y) * 100, 2) for x, y in zip(a, b)]


# --------------------------------------------------------------------- agenci
def make_backbone(kind: str) -> torch.nn.Module:
    """scratch = CifarBackbone (surowe piksele 3072->128, losowy zamrożony);
    reduced = ReducedBackbone (cache 512->128, losowy zamrożony)."""
    return CifarBackbone() if kind == "scratch" else ReducedBackbone()


def build_agent(kind: str, seed: int, wv, device: str) -> MarsCollectiveHetero:
    torch.manual_seed(seed)                     # seed = losowy front + proj/pody
    m = MarsCollectiveHetero(wv, backbone_module=make_backbone(kind), **COMMON)
    m.to(device)
    return m


def fit_ridge_autolam(ha: torch.Tensor, hb: torch.Tensor
                      ) -> Tuple[RidgeTranslator, float]:
    """Prostokątny ridge H_A(D_A)->H_B(D_B) z auto-lambda: wybór po held-out
    (80/20) rekonstrukcji na klasach kalibracyjnych. Zero wglądu w adoptowane."""
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


# ------------------------------------------------------------------- 1 wariant
def run_combo(cfg: Dict[str, object], wv, data, Ftr, Xtr, ytr, seed: int,
              epochs: int, n_dream: int, device: str) -> Dict[str, object]:
    """cfg: name, recv ("scratch"|"reduced"), method ("real"|"floor"|"linear"),
    sender ("homo"|"found512"|"scratch128"|None), K (int|None)."""
    recv = cfg["recv"]
    recv_space = "raw" if recv == "scratch" else "cache"
    D = data[recv_space]
    recv_X = D["X"]

    col = build_agent(recv, seed, wv, device)
    col.init_representation([D["cal"]], epochs=epochs, lr=LR, device=device)
    col.learn_task(D["cal"], epochs=epochs, lr=LR, device=device)   # 6 CAL
    col.learn_task(D["own"], epochs=epochs, lr=LR, device=device)   # OWN

    lam: Optional[float] = None
    disparity: Optional[float] = None

    if cfg["method"] == "real":                 # CEILING (homo, ten sam front+seed)
        Ai = build_agent(recv, seed, wv, device)
        Ai.init_representation([D["adopt"]], epochs=epochs, lr=LR, device=device)
        Ai.learn_task(D["adopt"], epochs=epochs, lr=LR, device=device)
        payloads = {c: Ai.export_class_stats(c, int((D["adopt"]["ytr"] == c).sum()))
                    for c in ADOPT_TASK}
        col.adopt_classes(ADOPT_TASK, payloads, epochs=epochs, lr=LR,
                          device=device, n_dream=n_dream)

    elif cfg["method"] == "floor":              # R0 anchor-only
        col.adopt_classes_anchor_only(ADOPT_TASK, device=device)

    else:                                       # linear: prostokątny/kwadratowy
        K = int(cfg["K"])
        cal_k = CAL_POOL[:K]
        if cfg["sender"] == "found512":         # foundation raw 512 (identyczność)
            src_a = (Ftr, _identity)

            def sender_feats(c: int) -> torch.Tensor:
                return Ftr[class_indices(ytr, c, n_dream)]
        else:                                   # scratch128 (CifarBackbone teacher)
            torch.manual_seed(seed + TEACHER_OFFSET)
            sender_bb = CifarBackbone().to(device)
            src_a = (Xtr, sender_bb)

            def sender_feats(c: int) -> torch.Tensor:
                return feats_through_front(
                    Xtr[class_indices(ytr, c, n_dream)], sender_bb, device)

        src_b = (recv_X, col.backbone)          # front odbiorcy (H_B)
        ha_cal, hb_cal = paired_calib_feats(cal_k, n_dream, ytr, src_a, src_b,
                                            device)
        lin, lam = fit_ridge_autolam(ha_cal, hb_cal)
        disparity = round(float(((lin.predict(ha_cal) - hb_cal) ** 2).mean()), 6)
        payloads_a = {c: feature_payload_from_feats(sender_feats(c), c,
                        COMMON["stats_k"], device) for c in ADOPT_TASK}
        col.adopt_classes_maptransform(ADOPT_TASK, payloads_a, lin.predict,
                                       epochs=epochs, lr=LR, device=device,
                                       n_dream=n_dream, stats_k=COMMON["stats_k"])

    row, _ = eval_protocols(col.forward, D["eval"], 1, list(col.seen_classes))
    adopted = row[1]                            # ADOPT=[8,9] to zadanie 1
    overall = sum(row) / len(row)
    return {"final_row_class_il": row, "adopted_ACC": round(adopted, 4),
            "overall_ACC": round(overall, 4), "lambda": lam,
            "disparity": disparity}


# ----------------------------------------------------------------------- dane
def build_space(X: torch.Tensor, Xt: torch.Tensor, ytr: torch.Tensor,
                yte: torch.Tensor) -> Dict[str, object]:
    return {
        "cal": make_task_data(X, ytr, Xt, yte, tasks=[CAL_POOL])[0],
        "own": make_task_data(X, ytr, Xt, yte, tasks=[OWN_TASK])[0],
        "adopt": make_task_data(X, ytr, Xt, yte, tasks=[ADOPT_TASK])[0],
        "eval": make_task_data(X, ytr, Xt, yte, tasks=[OWN_TASK, ADOPT_TASK]),
        "X": X}


def make_combos(ks: Sequence[int]) -> List[Dict[str, object]]:
    combos: List[Dict[str, object]] = [
        {"name": "CEILING_S", "recv": "scratch", "method": "real",
         "sender": "homo", "K": None},
        {"name": "R0_S", "recv": "scratch", "method": "floor",
         "sender": None, "K": None},
        {"name": "CEILING_PT", "recv": "reduced", "method": "real",
         "sender": "homo", "K": None},
        {"name": "R0_PT", "recv": "reduced", "method": "floor",
         "sender": None, "K": None}]
    for K in ks:
        combos.append({"name": f"HET_FS_K{K}", "recv": "scratch",
                       "method": "linear", "sender": "found512", "K": K})
        combos.append({"name": f"SANITY_RECT_K{K}", "recv": "reduced",
                       "method": "linear", "sender": "found512", "K": K})
        combos.append({"name": f"HET_SF_K{K}", "recv": "reduced",
                       "method": "linear", "sender": "scratch128", "K": K})
    return combos


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="1 seed, PEŁNA wierność (epoki 15, n_dream 5000, K=4) "
                         "-- kontrola SANITY przed FULL")
    args = ap.parse_args()

    light = args.smoke or args.diag         # tryby jednoseedowe (bez par-werdyktów)
    n_seeds = 1 if light else 5
    epochs = 4 if args.smoke else 15
    n_dream = 256 if args.smoke else 5000
    ks = (GATE_K,) if light else K_SWEEP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(SEED_START, SEED_START + n_seeds))
    combos = make_combos(ks)

    print("=" * 72)
    mode = "SMOKE" if args.smoke else "DIAG" if args.diag else "FULL"
    print(f"R-hard -- kolektyw między RÓŻNYMI reprezentacjami ({mode})")
    print(f"Device: {device} | seedy={seeds} | epok={epochs} | "
          f"n_dream={n_dream} | K={ks}")
    print(f"CAL_POOL={CAL_POOL} | OWN={OWN_TASK} | ADOPT={ADOPT_TASK}")
    print(f"BRAMKA HET_FS_K{GATE_K} >= {int(GATE_FRAC*100)}% * CEILING_S | "
          f"SANITY_RECT_K{GATE_K} >= {int(SANITY_FRAC*100)}% * CEILING_PT")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    Ftr, ytr, Fte, yte = extract_or_load_cifar_feats(device)      # cache 512
    Xtr, ytr_raw, Xte, yte_raw = load_cifar10_norm(device)        # surowe 3072
    if not (torch.equal(ytr, ytr_raw) and torch.equal(yte, yte_raw)):
        raise RuntimeError("kolejność datasetu niespójna cache vs surowe -- "
                           "parowanie po indeksach nieważne")

    data = {"raw": build_space(Xtr, Xte, ytr, yte),
            "cache": build_space(Ftr, Fte, ytr, yte)}

    t0 = time.perf_counter()
    out: Dict[str, object] = {
        "experiment": "R_hard", "level": "R-hard", "device": device,
        "seeds": seeds, "epochs_per_task": epochs, "n_dream": n_dream,
        "common": COMMON, "gate_frac": GATE_FRAC, "sanity_frac": SANITY_FRAC,
        "gate_K": GATE_K, "lambda_grid": list(LAMBDA_GRID), "cal_pool": CAL_POOL,
        "own_task": OWN_TASK, "adopt_task": ADOPT_TASK,
        "dirs": {"PRIMARY": "DIR-F->S (foundation512 -> scratch128, prostokątny)",
                 "ASYMETRIA": "DIR-S->F (scratch128 -> reduced128, kwadratowy)"},
        "systems": {}, "verdicts": {}}

    per_seed: Dict[str, List[Dict[str, object]]] = {c["name"]: [] for c in combos}
    for seed in seeds:
        for cfg in combos:
            r = run_combo(cfg, wv, data, Ftr, Xtr, ytr, seed, epochs, n_dream,
                          device)
            per_seed[cfg["name"]].append(r)
            print(f"[seed {seed}] {cfg['name']:16s}: "
                  f"adopted={r['adopted_ACC']*100:5.2f}% | "
                  f"overall={r['overall_ACC']*100:5.2f}% | lam={r['lambda']} "
                  f"disp={r['disparity']}")

    for cfg in combos:
        name = cfg["name"]
        out["systems"][name] = {
            "per_seed": per_seed[name],
            "agg_adopted": stats([p["adopted_ACC"] for p in per_seed[name]]),
            "agg_overall": stats([p["overall_ACC"] for p in per_seed[name]])}

    def adopted_list(name: str) -> List[float]:
        return [p["adopted_ACC"] for p in per_seed[name]]

    # ---------- BRAMKI ----------
    ceil_s = out["systems"]["CEILING_S"]["agg_adopted"]["mean"]
    ceil_pt = out["systems"]["CEILING_PT"]["agg_adopted"]["mean"]
    het_fs = out["systems"][f"HET_FS_K{GATE_K}"]["agg_adopted"]["mean"]
    sanity = out["systems"][f"SANITY_RECT_K{GATE_K}"]["agg_adopted"]["mean"]
    gate_abs = GATE_FRAC * ceil_s
    sanity_abs = SANITY_FRAC * ceil_pt
    gate_pass = het_fs >= gate_abs
    sanity_pass = sanity >= sanity_abs

    out["verdicts"]["GATE"] = {
        "gate_variant": f"HET_FS_K{GATE_K}", "adopted_mean_pp": round(het_fs * 100, 2),
        "prog_pp": round(gate_abs * 100, 2), "ceiling_pp": round(ceil_s * 100, 2),
        "frac_of_ceiling": round(het_fs / ceil_s, 3) if ceil_s else None,
        "zdana": bool(gate_pass),
        "opis": ("bramka zdana -- kolektyw representation-agnostic między RÓŻNYMI "
                 "reprezentacjami POTWIERDZONY (do CLAIMS)" if gate_pass else
                 "BRAMKA NIEZDANA -- patrz interpretacje miss (DROGA_R_HARD_PLAN)")}
    out["verdicts"]["SANITY_GATE"] = {
        "variant": f"SANITY_RECT_K{GATE_K}", "adopted_mean_pp": round(sanity * 100, 2),
        "prog_pp": round(sanity_abs * 100, 2), "ceiling_pp": round(ceil_pt * 100, 2),
        "frac_of_ceiling": round(sanity / ceil_pt, 3) if ceil_pt else None,
        "zdana": bool(sanity_pass),
        "opis": ("maszyneria prostokątna zdrowa (mapa 512->128 z konstrukcji "
                 "odzyskana)" if sanity_pass else
                 "SANITY NIEZDANA -- interpretacja HET zablokowana (à la R2b/R-mild)")}

    if not light:
        # KONIECZNE + CENA (DIR-F->S, primary)
        het = adopted_list(f"HET_FS_K{GATE_K}")
        for key, base_name in ((f"HET_FS_K{GATE_K}_vs_R0_S", "R0_S"),
                               (f"HET_FS_K{GATE_K}_vs_CEILING_S", "CEILING_S")):
            base = adopted_list(base_name)
            d = _pairs(het, base)
            noise = (stats([x * 100 for x in het])["std"]
                     + stats([x * 100 for x in base])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"][key] = {"pairs_pp": d, "delta": ds,
                                    "noise_pp": round(noise, 4), "verdict": v}
        out["verdicts"]["krzywa_K_HET_FS"] = {
            f"K{K}": out["systems"][f"HET_FS_K{K}"]["agg_adopted"]["mean"]
            for K in ks}

        # DIR-S->F (asymetria kierunku -- raport OSOBNY, NIE bramkowany)
        het_sf = adopted_list(f"HET_SF_K{GATE_K}")
        for key, base_name in ((f"HET_SF_K{GATE_K}_vs_R0_PT", "R0_PT"),
                               (f"HET_SF_K{GATE_K}_vs_CEILING_PT", "CEILING_PT")):
            base = adopted_list(base_name)
            d = _pairs(het_sf, base)
            noise = (stats([x * 100 for x in het_sf])["std"]
                     + stats([x * 100 for x in base])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"][key] = {"pairs_pp": d, "delta": ds,
                                    "noise_pp": round(noise, 4), "verdict": v}
        out["verdicts"]["krzywa_K_HET_SF"] = {
            f"K{K}": out["systems"][f"HET_SF_K{K}"]["agg_adopted"]["mean"]
            for K in ks}
        out["verdicts"]["ASYMETRIA_uwaga"] = (
            "DIR-F->S prostokątny (512->128) vs DIR-S->F kwadratowy (128->128) "
            "-- asymetria MIESZA kierunek treści z kształtem mapy; OBSERWACJA, "
            "nie twierdzenie (DROGA_R_HARD_PLAN, zastrzeżenie C).")

    # ---------- raport ----------
    print(f"\n--- R-hard (n={n_seeds}, seedy {seeds}) -- adopted ACC ---")
    for cfg in combos:
        name = cfg["name"]
        a = out["systems"][name]["agg_adopted"]
        o = out["systems"][name]["agg_overall"]
        print(f"  {name:16s}: adopted {a['mean']*100:5.2f}+/-{a['std']*100:.2f}% "
              f"(min {a['min']*100:.2f}) | overall {o['mean']*100:5.2f}%")
    g = out["verdicts"]["GATE"]
    s = out["verdicts"]["SANITY_GATE"]
    print(f"\n  BRAMKA {g['gate_variant']}: {g['adopted_mean_pp']:.2f}% vs prog "
          f"{g['prog_pp']:.2f}% ({int(GATE_FRAC*100)}% z sufitu-scratcha "
          f"{g['ceiling_pp']:.2f}%) -> {'ZDANA' if g['zdana'] else 'NIEZDANA'} "
          f"[{g['frac_of_ceiling']} sufitu]")
    print(f"  SANITY {s['variant']}: {s['adopted_mean_pp']:.2f}% vs prog "
          f"{s['prog_pp']:.2f}% ({int(SANITY_FRAC*100)}% z sufitu-reduced "
          f"{s['ceiling_pp']:.2f}%) -> {'ZDANA' if s['zdana'] else 'NIEZDANA'} "
          f"[{s['frac_of_ceiling']} sufitu]")
    for key in (f"HET_FS_K{GATE_K}_vs_R0_S", f"HET_FS_K{GATE_K}_vs_CEILING_S",
                "krzywa_K_HET_FS", f"HET_SF_K{GATE_K}_vs_R0_PT",
                f"HET_SF_K{GATE_K}_vs_CEILING_PT", "krzywa_K_HET_SF"):
        if key in out["verdicts"]:
            print(f"  {key}: {json.dumps(out['verdicts'][key], ensure_ascii=False)}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("R_hard_smoke.json" if args.smoke else
             "R_hard_diag.json" if args.diag else "R_hard.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
