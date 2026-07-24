"""
run_R1b_translate.py -- Droga R1b: naprawa translacji heterogenicznej
(DROGA_R1B_PLAN.md). Poziom R-mild (ten sam resnet18, rozny seed
projekcji 512->128 nadawca vs odbiorca).

Split CIFAR-10 (zamrozony): 4 klasy KALIBRACYJNE dzielone [0,1,2,3]
(realne u wszystkich, translator fitowany wylacznie na nich, rozlaczne
z metryka) + protokol [4,5]=wlasne odbiorcy, [6,7] i [8,9]=adoptowane.
Metryka = ACC klas ADOPTOWANYCH (row_class_il[1:] po protokole).

Warianty (te same seedy 0..4):
  CEILING       -- homogeniczny, payload cech (== L2). Gorna kotwica.
  R0            -- hetero, podloga anchor-only (bez translacji).
  ORACLE_SANITY -- homogeniczny, dekoder MLP na 10 realnych klasach
                   (KROK 1: czy 50-d kotwica przenosi dosc pod pelna
                   wiedza dekodera).
  ORACLE_HET    -- hetero, dekoder MLP na 10 realnych klasach.
  RBF_SANITY    -- homogeniczny, kernel-ridge (RBF) na klasach dzielonych
                   (KROK 2). *** BRAMKA: adopted >= 65% ***.
  RBF_HET       -- hetero, kernel-ridge na klasach dzielonych.
  RIDGE_HET     -- hetero, liniowy Ridge na klasach dzielonych (ablacja).

BRAMKA FALSYFIKACJI (zamrozona): RBF_SANITY < 65% => seria R na poziomie
kotwica-interlingua SFALSYFIKOWANA, NIE wchodzi do CLAIMS. Werdykty
heterogenicznosci (RBF_HET vs CEILING/R0) liczone TYLKO po zdanej bramce.

Wymaga: data/cifar_resnet18_224_feats.pt (cache z L), data/glove.6B.50d.txt.

Tryb szybki:  python src/run_R1b_translate.py --smoke
Pelny:        python src/run_R1b_translate.py
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
from mars_translate import KernelRidgeTranslator, RidgeTranslator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR = 0.001
TEACHER_OFFSET = 1000
GATE_PP = 65.0                      # bramka falsyfikacji na RBF_SANITY
LAMBDA_GRID = (1e-2, 1e-1, 1.0)     # zamrozony grid regularyzacji
CAL_CLASSES = [0, 1, 2, 3]          # dzielone kalibracyjne
OWN_TASK = [4, 5]                   # wlasne odbiorcy
ADOPT_TASKS = [[6, 7], [8, 9]]      # adoptowane (metryka)
PROTOCOL_TASKS = [OWN_TASK] + ADOPT_TASKS
COMMON = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
              bn_calib=False, feat_signorm=False)

# nazwa -> (homogeniczny, metoda)
VARIANTS: Dict[str, Tuple[bool, str]] = {
    "CEILING":       (True,  "real"),
    "R0":            (False, "floor"),
    "ORACLE_SANITY": (True,  "mlp10"),
    "ORACLE_HET":    (False, "mlp10"),
    "RBF_SANITY":    (True,  "krr"),
    "RBF_HET":       (False, "krr"),
    "RIDGE_HET":     (False, "ridge"),
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


def real_pairs(col: MarsCollectiveHetero, Ftr, ytr, classes: Sequence[int],
               n_per: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Realne pary (embed_from_feats(f), f) u odbiorcy dla podanych klas
    (cechy = backbone odbiorcy na cache 512-d)."""
    feats_list = []
    for c in classes:
        idx = (ytr == c).nonzero(as_tuple=True)[0][:n_per]
        feats_list.append(col.backbone(Ftr[idx]))
    feats = torch.cat(feats_list)
    with torch.no_grad():
        emb = col.embed_from_feats(feats)
    return emb.detach(), feats.detach()


def fit_translator(kind: str, emb: torch.Tensor, feats: torch.Tensor):
    """Wybor lambda z zamrozonego gridu po rekonstrukcji na held-out
    klas dzielonych (bez wycieku z klas adoptowanych)."""
    n = len(emb)
    perm = torch.randperm(n, device=emb.device)
    cut = max(int(0.8 * n), 1)
    tr, va = perm[:cut], perm[cut:]
    cls = KernelRidgeTranslator if kind == "krr" else RidgeTranslator
    best, best_mse = None, float("inf")
    for lam in LAMBDA_GRID:
        t = cls(lam=lam).fit(emb[tr], feats[tr])
        mse = float(((t.predict(emb[va]) - feats[va]) ** 2).mean()) \
            if len(va) else float(((t.predict(emb[tr]) - feats[tr]) ** 2).mean())
        if mse < best_mse:
            best, best_mse = lam, mse
    return cls(lam=best).fit(emb, feats), best, round(best_mse, 6)


def run_variant(name: str, wv, cal_task, protocol_td, Ftr, ytr, seed: int,
                epochs: int, n_dream: int, device: str) -> Dict[str, object]:
    homo, method = VARIANTS[name]
    col = build_agent(wv, seed, device)
    col.init_representation([cal_task], epochs=epochs, lr=LR, device=device)
    col.learn_task(cal_task, epochs=epochs, lr=LR, device=device)     # CAL real
    col.learn_task(protocol_td[0], epochs=epochs, lr=LR, device=device)  # OWN

    # przygotuj materializator
    translator, lam, tr_mse = None, None, None
    if method == "mlp10":
        emb, feats = real_pairs(col, Ftr, ytr, list(range(10)), n_dream, device)
        tr_mse = round(col.train_decoder_on(emb, feats, epochs, LR, device), 6)
    elif method in ("krr", "ridge"):
        emb, feats = real_pairs(col, Ftr, ytr, CAL_CLASSES, n_dream, device)
        translator, lam, tr_mse = fit_translator(method, emb, feats)

    row: List[float] = []
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
        elif method == "mlp10":
            apl = {c: Ai.export_anchor_payload(c, n_dream, device)
                   for c in td["classes"]}
            col.adopt_classes_hetero(td["classes"], apl, epochs=epochs, lr=LR,
                                     device=device, n_dream=n_dream,
                                     stats_k=COMMON["stats_k"])
        else:  # krr / ridge
            apl = {c: Ai.export_anchor_payload(c, n_dream, device)
                   for c in td["classes"]}
            col.adopt_classes_translate(td["classes"], apl, translator,
                                        epochs=epochs, lr=LR, device=device,
                                        n_dream=n_dream,
                                        stats_k=COMMON["stats_k"])

        row, _ = eval_protocols(col.forward, protocol_td, i,
                                list(col.seen_classes))

    overall = sum(row) / len(row)
    adopted = sum(row[1:]) / len(row[1:])
    return {"final_row_class_il": row, "overall_ACC": round(overall, 4),
            "adopted_ACC": round(adopted, 4),
            "lambda": lam, "train_mse": tr_mse}


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
    print(f"R1b -- translacja heterogeniczna (R-mild) "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"n_dream={n_dream}")
    print(f"CAL(dzielone)={CAL_CLASSES} | OWN={OWN_TASK} | "
          f"ADOPT={ADOPT_TASKS} | BRAMKA RBF_SANITY>={GATE_PP}%")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    Ftr, ytr, Fte, yte = extract_or_load_cifar_feats(device)
    cal_task = make_task_data(Ftr, ytr, Fte, yte, tasks=[CAL_CLASSES])[0]
    protocol_td = make_task_data(Ftr, ytr, Fte, yte, tasks=PROTOCOL_TASKS)

    t0 = time.perf_counter()
    out: Dict[str, object] = {
        "experiment": "R1b_translate", "level": "R-mild",
        "device": device, "n_seeds": n_seeds, "epochs_per_task": epochs,
        "n_dream": n_dream, "common": COMMON, "gate_pp": GATE_PP,
        "lambda_grid": list(LAMBDA_GRID), "cal_classes": CAL_CLASSES,
        "own_task": OWN_TASK, "adopt_tasks": ADOPT_TASKS,
        "systems": {}, "verdicts": {}}

    per_seed: Dict[str, List[Dict[str, object]]] = {v: [] for v in VARIANTS}
    for seed in range(n_seeds):
        for name in VARIANTS:
            r = run_variant(name, wv, cal_task, protocol_td, Ftr, ytr, seed,
                            epochs, n_dream, device)
            per_seed[name].append(r)
            print(f"[seed {seed}] {name:13s}: adopted={r['adopted_ACC']*100:5.2f}% "
                  f"| overall={r['overall_ACC']*100:5.2f}% "
                  f"| lam={r['lambda']} mse={r['train_mse']}")

    for name in VARIANTS:
        out["systems"][name] = {
            "per_seed": per_seed[name],
            "agg_adopted": stats([p["adopted_ACC"] for p in per_seed[name]]),
            "agg_overall": stats([p["overall_ACC"] for p in per_seed[name]])}

    # ---------- BRAMKA + werdykty ----------
    gate_val = out["systems"]["RBF_SANITY"]["agg_adopted"]["mean"] * 100
    gate_pass = gate_val >= GATE_PP
    out["verdicts"]["GATE_RBF_SANITY"] = {
        "adopted_mean_pp": round(gate_val, 2), "prog_pp": GATE_PP,
        "zdana": gate_pass,
        "opis": ("bramka zdana -- interpretacja heterogenicznosci odblokowana"
                 if gate_pass else
                 "BRAMKA NIEZDANA -- seria R sfalsyfikowana na poziomie "
                 "kotwica-interlingua; NIE wchodzi do CLAIMS")}

    if not args.smoke and gate_pass:
        rbf = [p["adopted_ACC"] for p in per_seed["RBF_HET"]]
        ceil = [p["adopted_ACC"] for p in per_seed["CEILING"]]
        r0 = [p["adopted_ACC"] for p in per_seed["R0"]]
        for key, base in (("RBF_HET_vs_CEILING", ceil),
                          ("RBF_HET_vs_R0", r0)):
            d = _pairs(rbf, base)
            noise = (stats([x * 100 for x in rbf])["std"]
                     + stats([x * 100 for x in base])["std"])
            v, ds = verdict_paired(d, noise)
            out["verdicts"][key] = {"pairs_pp": d, "delta": ds,
                                    "noise_pp": round(noise, 4), "verdict": v}

    # ---------- raport ----------
    print(f"\n--- R1b (n={n_seeds}) -- adopted ACC (class-IL) ---")
    for name in VARIANTS:
        a = out["systems"][name]["agg_adopted"]
        o = out["systems"][name]["agg_overall"]
        print(f"  {name:13s}: adopted {a['mean']*100:5.2f}+/-{a['std']*100:.2f}% "
              f"(min {a['min']*100:.2f}) | overall {o['mean']*100:5.2f}%")
    g = out["verdicts"]["GATE_RBF_SANITY"]
    print(f"\n  BRAMKA RBF_SANITY: {g['adopted_mean_pp']:.2f}% vs prog "
          f"{GATE_PP}% -> {'ZDANA' if g['zdana'] else 'NIEZDANA (R sfalsyfikowane)'}")
    for key in ("RBF_HET_vs_CEILING", "RBF_HET_vs_R0"):
        if key in out["verdicts"]:
            vd = out["verdicts"][key]
            print(f"  {key:20s} (prog {vd['noise_pp']:.2f}pp): "
                  f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
                  f"(min {vd['delta']['min']:+.2f}) | pary {vd['pairs_pp']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("R1b_translate_smoke.json" if args.smoke
             else "R1b_translate.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
