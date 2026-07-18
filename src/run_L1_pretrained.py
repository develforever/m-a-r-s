"""
run_L1_pretrained.py -- L1: Split-CIFAR-10 na zamrozonym pretrenowanym
backbone (DROGA_L_PLAN.md, sekcja L1). JAWNY FORK TOZSAMOSCI: linia
"foundation-embedding", raportowana osobno od glownej "from-scratch".

Warianty (CIFAR-n, kotwice 50d):
  l1_all : sufit zamrozonych cech pretrained (proj_train="all", jak K0)
  l1_seq : uczciwy CL, sparse_k16 (konfiguracja J2b, inny tylko backbone)

Kryteria (Z GORY, class-IL):
  Glowne: l1_seq vs J2b sparse_k16 (37.51 +/- 1.35, TE SAME seedy,
  pary): SYGNAL+/-, SYGNAL-parowy+/-, SZUM (oczekiwany SYGNAL+,
  ale werdykt symetryczny -- resize x7 moze oslabic cechy).
  Diagnostyka: gap_mech_L = l1_all - l1_seq; dystans do joint 70.24;
  jesli l1_all ~ 39.65 (sufit losowych, K0) -> fork nie dziala przez
  losowy rzut 512->128, nie przez encoder.

Wymaga: data/glove.6B.50d.txt, results/J2b_cifar_sparse.json,
results/J2_cifar_normalized.json, results/K0_cifar_ceiling.json,
internet przy pierwszym runie (wagi resnet18).

Tryb szybki:  python src/run_L1_pretrained.py --smoke
Pelny:        python src/run_L1_pretrained.py
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import make_task_data, cl_metrics, eval_protocols
from mars_cl_j import MarsCLSemanticF3J, load_cifar10_norm
from mars_cl_l import PretrainedBackbone
from mars_cl_semantic import MarsCLSemantic, load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
J2B_REF = os.path.join(RESULTS_DIR, "J2b_cifar_sparse.json")
J2_REF = os.path.join(RESULTS_DIR, "J2_cifar_normalized.json")
K0_REF = os.path.join(RESULTS_DIR, "K0_cifar_ceiling.json")
LR = 0.001
SEQ_CFG = dict(dream_model="sparse", stats_k=16, epochs_proj=15, l2sp=0.0,
               bn_calib=False, feat_signorm=False)


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


def run_one(mode, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    if mode == "all":
        m = MarsCLSemantic(wv, proj_train="all",
                           backbone_module=PretrainedBackbone())
    else:
        m = MarsCLSemanticF3J(wv, backbone_module=PretrainedBackbone(),
                              **SEQ_CFG)
    m.to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)
    R_c, R_t = [], []
    seen = []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row_c, row_t = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row_c)
        R_t.append(row_t)
    return R_c, R_t


def load_ref(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    j2b, j2, k0 = load_ref(J2B_REF), load_ref(J2_REF), load_ref(K0_REF)
    if not args.smoke and j2b is None:
        sys.exit(f"BLAD: brak {J2B_REF} (baza par).")

    print("=" * 72)
    print(f"L1 -- pretrained backbone na CIFAR "
          f"({'SMOKE' if args.smoke else 'FULL'}) [FORK TOZSAMOSCI]")
    print(f"Device: {device} | seeds={n_seeds} | epok={epochs} | "
          f"warianty=['l1_all', 'l1_seq']")
    print("=" * 72)

    wv = load_word_vectors("CIFAR-10", device=device)
    Xtr, ytr, Xte, yte = load_cifar10_norm(device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "L1_pretrained", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "encoder": "resnet18_IMAGENET1K_V1 -> random frozen 512->128",
           "seq_cfg": SEQ_CFG, "systems": {}, "verdicts": {}}

    for name, mode in (("l1_all", "all"), ("l1_seq", "seq")):
        per_seed = []
        for seed in range(n_seeds):
            R_c, R_t = run_one(mode, wv, task_data, seed, epochs, device)
            m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
            per_seed.append({"R_class_il": R_c, "class_il": m_c,
                             "task_il": m_t})
            print(f"[CIFAR-10n] {name:6s} seed {seed}: "
                  f"class-IL ACC={m_c['ACC']*100:.2f}% "
                  f"F={m_c['forgetting']*100:.1f}pp")
        agg = {"class_il_ACC": stats([p["class_il"]["ACC"]
                                      for p in per_seed]),
               "class_il_forgetting": stats(
                   [p["class_il"]["forgetting"] for p in per_seed])}
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykt + diagnostyka ----------
    if not args.smoke and j2b:
        base = [p["class_il"]["ACC"] for p in
                j2b["systems"]["sparse_k16"]["per_seed"]][:n_seeds]
        new = [p["class_il"]["ACC"] for p
               in out["systems"]["l1_seq"]["per_seed"]]
        d = [(a - b) * 100 for a, b in zip(new, base)]
        noise = (stats([x * 100 for x in base])["std"]
                 + stats([x * 100 for x in new])["std"])
        v, ds = verdict_paired(d, noise)
        allm = out["systems"]["l1_all"]["agg"]["class_il_ACC"]["mean"]
        seqm = stats(new)["mean"]
        diag = {"gap_mech_L_pp": round((allm - seqm) * 100, 2),
                "mech_pct_of_ceiling": round(seqm / allm * 100, 1)}
        if j2:
            joint = j2["systems"]["joint"]["agg"]["class_il_ACC"]["mean"]
            diag["gap_to_joint_pp"] = round((joint - seqm) * 100, 2)
        if k0:
            rand_ceil = k0["diagnosis"]["ceiling_best"]
            diag["random_ceiling_ref"] = rand_ceil
            diag["fork_lifts_ceiling_pp"] = round(
                (allm - rand_ceil) * 100, 2)
        out["verdicts"]["l1_seq_vs_random_backbone"] = {
            "base": "J2b sparse_k16 (losowy backbone, 37.51)",
            "pairs_pp": [round(x, 2) for x in d], "delta": ds,
            "noise_pp": round(noise, 4), "verdict": v,
            "diagnosis": diag}

    # ---------- raport ----------
    print(f"\n--- L1 (n={n_seeds}) -- CIFAR-n, class-IL "
          f"[linia foundation-embedding] ---")
    if j2b:
        b = j2b["systems"]["sparse_k16"]["agg"]["class_il_ACC"]
        print(f"  [J2b] losowy backbone: {b['mean']*100:.2f}"
              f"+/-{b['std']*100:.2f}%")
    if k0:
        print(f"  [K0] sufit losowych : "
              f"{k0['diagnosis']['ceiling_best']*100:.2f}%")
    for name in ("l1_all", "l1_seq"):
        a = out["systems"][name]["agg"]
        print(f"  {name:6s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
              f"+/-{a['class_il_ACC']['std']*100:.2f}% "
              f"(min {a['class_il_ACC']['min']*100:.2f}%) | "
              f"F {a['class_il_forgetting']['mean']*100:.1f}pp")
    if "l1_seq_vs_random_backbone" in out["verdicts"]:
        vd = out["verdicts"]["l1_seq_vs_random_backbone"]
        print(f"  WERDYKT vs losowy (prog {vd['noise_pp']:.2f}pp): "
              f"{vd['verdict']} | d={vd['delta']['mean']:+.2f}pp "
              f"(min {vd['delta']['min']:+.2f}) | pary {vd['pairs_pp']}")
        print(f"  DIAGNOZA: {vd['diagnosis']}")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("L1_pretrained_smoke.json" if args.smoke
             else "L1_pretrained.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
