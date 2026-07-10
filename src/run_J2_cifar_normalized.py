"""
run_J2_cifar_normalized.py -- J2: Split-CIFAR-10 z poprawnym wejsciem
+ sen k16 (DROGA_J_PLAN.md, sekcja J2).

Naprawy vs F4 (audyt 2026-07-10):
  (a) normalizacja per kanal dla WSZYSTKICH systemow (F4 mial tylko /255;
      Fashion/MNIST zawsze mialy Normalize) -- monolity tez, uczciwie;
  (b) MARS dostaje zwycieski sen k16 z H1b (F4 biegal na k4, bo kod
      zamrozono zanim H1b istnial) + kondycjonowanie J1.

Siatka MARS 2x2: stats_k {4,16} x kondycjonowanie {raw, cond};
cond = bn_calib + sigma-norm (task0, bez etykiet). epochs_proj=15,
l2sp=0 dla wszystkich (konwencja H1b). Baseline'y jak F4: finetune /
replay-200 / joint, te same seedy.

Kryteria werdyktu (Z GORY, class-IL):
  Glowne: mars_k16_cond vs replay-200 (pary per-seed, ten sam run):
    SYGNAL+ jesli sr. d > prog szumu ORAZ min per-seed > 0.
  Pytanie naprawcze: najlepszy nowy MARS vs stary F4 mars_combo
    (32.04 +/- 1.01, referencja NIESPAROWANA -- inne przygotowanie
    danych): czy sr. > 32.04 + 1.01? Jesli tak, niski wynik CIFAR byl
    czesciowo artefaktem przygotowania danych, nie tylko granica
    losowych cech.
  Dekompozycja raportowana: k4_raw->k4_cond (kondycjonowanie),
    k4_cond->k16_cond (wiernosc snu), przesuniecie sufitu joint.

Wymaga: data/glove.6B.50d.txt; results/F4_split_cifar.json (referencja).

Tryb szybki:  python src/run_J2_cifar_normalized.py --smoke
Pelny:        python src/run_J2_cifar_normalized.py
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from cl_common import (make_task_data, eval_protocols, cl_metrics,
                       balanced_buffer)
from cifar_cl import CifarBackbone, MonoCifar
from mars_cl_j import MarsCLSemanticF3J, load_cifar10_norm
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
F4_REF = os.path.join(RESULTS_DIR, "F4_split_cifar.json")
LR, BATCH = 0.001, 512
REPLAY_SIZE = 200
MARS_VARIANTS = {
    "mars_k4_raw":   dict(stats_k=4,  bn_calib=False, feat_signorm=False),
    "mars_k4_cond":  dict(stats_k=4,  bn_calib=True,  feat_signorm=True),
    "mars_k16_raw":  dict(stats_k=16, bn_calib=False, feat_signorm=False),
    "mars_k16_cond": dict(stats_k=16, bn_calib=True,  feat_signorm=True),
}
MARS_COMMON = dict(dream_model="diag", epochs_proj=15, l2sp=0.0)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def train_epochs(model, X, y, epochs, device, replay=None, replay_bs=128):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for s in range(0, len(X), BATCH):
            idx = perm[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            if replay is not None:
                bx, by = replay
                k = min(replay_bs, len(bx))
                ridx = torch.randint(0, len(bx), (k,), device=bx.device)
                xb = torch.cat([xb, bx[ridx]])
                yb = torch.cat([yb, by[ridx]])
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_mono(method, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    model = MonoCifar().to(device)
    R_c, R_t = [], []
    seen = []
    for t, td in enumerate(task_data):
        seen = seen + td["classes"]
        replay = (balanced_buffer(task_data, t - 1, REPLAY_SIZE, seed)
                  if method == "replay" and t > 0 else None)
        train_epochs(model, td["Xtr"], td["ytr"], epochs, device,
                     replay=replay)
        model.eval()
        row_c, row_t = eval_protocols(model.forward, task_data, t, seen)
        R_c.append(row_c)
        R_t.append(row_t)
    return R_c, R_t


def run_joint(task_data, seed, epochs, device):
    torch.manual_seed(seed)
    model = MonoCifar().to(device)
    X = torch.cat([td["Xtr"] for td in task_data])
    y = torch.cat([td["ytr"] for td in task_data])
    train_epochs(model, X, y, epochs * len(task_data), device)
    model.eval()
    seen = [c for td in task_data for c in td["classes"]]
    return eval_protocols(model.forward, task_data, len(task_data) - 1, seen)


def run_mars(cfg, wv, task_data, seed, epochs, device):
    torch.manual_seed(seed)
    m = MarsCLSemanticF3J(wv, backbone_module=CifarBackbone(),
                          **cfg, **MARS_COMMON)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--glove", default=None)
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f4_ref = None
    if os.path.exists(F4_REF):
        with open(F4_REF, encoding="utf-8") as f:
            f4_ref = json.load(f)
    elif not args.smoke:
        print(f"(uwaga: brak {F4_REF} -- pytanie naprawcze bez referencji)")

    print("=" * 72)
    print(f"J2 -- Split-CIFAR-10 znormalizowany  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs}")
    print("=" * 72)

    kw = {"glove_path": args.glove} if args.glove else {}
    wv = load_word_vectors("CIFAR-10", device=device, **kw)
    Xtr, ytr, Xte, yte = load_cifar10_norm(device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "J2_cifar_normalized", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "mars_variants": {k: dict(v) for k, v in MARS_VARIANTS.items()},
           "mars_common": MARS_COMMON, "systems": {}}

    systems = ([("finetune", None), ("replay", None), ("joint", None)]
               + [(k, v) for k, v in MARS_VARIANTS.items()])
    for name, cfg in systems:
        per_seed = []
        for seed in range(n_seeds):
            if name == "joint":
                row_c, row_t = run_joint(task_data, seed, epochs, device)
                m_c = {"ACC": round(sum(row_c) / len(row_c), 4),
                       "forgetting": 0.0, "BWT": 0.0,
                       "final_per_task": row_c}
                m_t = {"ACC": round(sum(row_t) / len(row_t), 4)}
                per_seed.append({"class_il": m_c, "task_il": m_t})
            else:
                if cfg is None:
                    R_c, R_t = run_mono(name, task_data, seed, epochs, device)
                else:
                    R_c, R_t = run_mars(cfg, wv, task_data, seed, epochs,
                                        device)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "class_il": m_c,
                                 "task_il": m_t})
            print(f"[CIFAR-10n] {name:13s} seed {seed}: "
                  f"class-IL ACC={m_c['ACC']*100:.2f}%"
                  + (f" F={m_c['forgetting']*100:.1f}pp"
                     if name != "joint" else " (sufit)"))
        agg = {"class_il_ACC": stats([p["class_il"]["ACC"]
                                      for p in per_seed])}
        if name != "joint":
            agg["class_il_forgetting"] = stats(
                [p["class_il"]["forgetting"] for p in per_seed])
        out["systems"][name] = {"per_seed": per_seed, "agg": agg}

    # ---------- werdykt ----------
    verdict = None
    if not args.smoke:
        rep = [p["class_il"]["ACC"] for p
               in out["systems"]["replay"]["per_seed"]]
        primary = [p["class_il"]["ACC"] for p
                   in out["systems"]["mars_k16_cond"]["per_seed"]]
        d_rep = stats([(a - b) * 100 for a, b in zip(primary, rep)])
        noise = (stats([r * 100 for r in rep])["std"]
                 + stats([m_ * 100 for m_ in primary])["std"])
        v_str = ("SYGNAL+ (nad replay)" if d_rep["mean"] > noise
                 and d_rep["min"] > 0 else
                 "SYGNAL- (pod replay)" if d_rep["mean"] < -noise
                 else "SZUM/rownowaznosc z replay")
        best = max(MARS_VARIANTS, key=lambda v:
                   out["systems"][v]["agg"]["class_il_ACC"]["mean"])
        best_acc = out["systems"][best]["agg"]["class_il_ACC"]
        verdict = {"primary": "mars_k16_cond vs replay",
                   "delta_vs_replay_pp": d_rep,
                   "noise_pp": round(noise, 4), "verdict": v_str,
                   "best_mars": best}
        # pytanie naprawcze vs stary F4 (niesparowane)
        if f4_ref:
            old = f4_ref["systems"]["mars_combo"]["agg"]["class_il_ACC"]
            thr = old["mean"] + old["std"]
            verdict["repair_question"] = {
                "old_f4_mars_combo": old,
                "new_best": {"name": best, **best_acc},
                "threshold": round(thr, 4),
                "repaired": bool(best_acc["mean"] > thr),
            }
        # dekompozycja
        def _acc(nm):
            return out["systems"][nm]["agg"]["class_il_ACC"]["mean"] * 100
        verdict["decomposition_pp"] = {
            "conditioning_k4": round(_acc("mars_k4_cond")
                                     - _acc("mars_k4_raw"), 2),
            "dream_k16_at_cond": round(_acc("mars_k16_cond")
                                       - _acc("mars_k4_cond"), 2),
            "joint_ceiling": round(_acc("joint"), 2),
        }
    out["verdict"] = verdict

    # ---------- raport ----------
    print(f"\n--- Split-CIFAR-10 znormalizowany (n={n_seeds}) -- class-IL ---")
    if f4_ref:
        o = f4_ref["systems"]
        print(f"  [F4 stare] replay {o['replay']['agg']['class_il_ACC']['mean']*100:.2f}% | "
              f"mars_combo {o['mars_combo']['agg']['class_il_ACC']['mean']*100:.2f}% | "
              f"joint {o['joint']['agg']['class_il_ACC']['mean']*100:.2f}%")
    for name, _ in systems:
        a = out["systems"][name]["agg"]
        f_str = (f" | F {a['class_il_forgetting']['mean']*100:.1f}pp"
                 if "class_il_forgetting" in a else "  (sufit)")
        print(f"  {name:13s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
              f"+/-{a['class_il_ACC']['std']*100:.2f}% "
              f"(min {a['class_il_ACC']['min']*100:.2f}%){f_str}")
    if verdict:
        print(f"  WERDYKT (mars_k16_cond vs replay, "
              f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
        print(f"    d vs replay: {verdict['delta_vs_replay_pp']['mean']:+.2f}pp"
              f" (min {verdict['delta_vs_replay_pp']['min']:+.2f})")
        if "repair_question" in verdict:
            rq = verdict["repair_question"]
            print(f"    naprawa vs stary F4 combo "
                  f"({rq['old_f4_mars_combo']['mean']*100:.2f}%): "
                  f"{rq['new_best']['name']} "
                  f"{rq['new_best']['mean']*100:.2f}% -> "
                  f"{'TAK' if rq['repaired'] else 'NIE'}")
        dec = verdict["decomposition_pp"]
        print(f"    dekompozycja: kondycjonowanie(k4) "
              f"{dec['conditioning_k4']:+.2f}pp | sen k4->k16(cond) "
              f"{dec['dream_k16_at_cond']:+.2f}pp | joint "
              f"{dec['joint_ceiling']:.2f}%")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = ("J2_cifar_normalized_smoke.json" if args.smoke
             else "J2_cifar_normalized.json")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
