"""
run_F4_split_cifar.py -- F4: Split-CIFAR-10 -- prog wiarygodnosci serii F.

Pytanie: czy glowny wynik F3b (MARS bez bufora ~ replay-200 na Fashion)
przenosi sie na naturalne obrazy? CIFAR-10: 5 zadan x 2 klasy, nazwy klas
semantycznie bogate (warunek stosowalnosci potwierdzony w G1/F3b).

Systemy (wszystko w JEDNYM runie, te same seedy -- porownanie parami):
  finetune   : monolit sekwencyjny (dolna granica)
  replay     : monolit + bufor 200 probek, probkowany per krok (przeciwnik)
  joint      : monolit na wszystkich danych (sufit; rowny budzet epok)
  mars_k4    : MARS-CL semantic (losowy zamrozony CifarBackbone, slowa
               GloVe, sen 4-centroidowy, epochs_proj=15)
  mars_combo : jw. + epochs_proj=8, l2sp=0.1
  (EWC pominiete -- F0 wykazalo bezuzytecznosc w class-IL)

RYZYKO PRE-REJESTROWANE: losowe cechy konwolucyjne na obrazach naturalnych
sa DUZO slabsze niz na Fashion, a monolity (replay/joint) trenuja backbone
end-to-end. Jesli MARS przegra wyraznie, diagnoza brzmi "reprezentacja"
(znow) -- nastepny krok to zamrozony backbone nienadzorowany/pretrenowany,
nie zmiana mechanizmu CL. To tez jest wynik: granica losowych cech.

Kryterium werdyktu (Z GORY, class-IL, per-seed):
  najlepszy mars vs replay: d >= -prog szumu => SYGNAL+ (teza przenosi sie)
  sanity: mars vs finetune > +10pp.

Tryb szybki:  python src/run_F4_split_cifar.py --smoke
Pelny:        python src/run_F4_split_cifar.py
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
from cifar_cl import load_cifar10, CifarBackbone, MonoCifar
from mars_cl_f3 import MarsCLSemanticF3
from mars_cl_semantic import load_word_vectors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LR, BATCH = 0.001, 512
REPLAY_SIZE = 200
MARS_VARIANTS = {
    "mars_k4":    dict(stats_k=4, epochs_proj=15),
    "mars_combo": dict(stats_k=4, epochs_proj=8, l2sp=0.1),
}


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
    m = MarsCLSemanticF3(wv, backbone_module=CifarBackbone(), **cfg)
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

    print("=" * 72)
    print(f"F4 -- Split-CIFAR-10  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs}")
    print("=" * 72)

    kw = {"glove_path": args.glove} if args.glove else {}
    wv = load_word_vectors("CIFAR-10", device=device, **kw)
    Xtr, ytr, Xte, yte = load_cifar10(device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    t0 = time.perf_counter()
    out = {"experiment": "F4_split_cifar", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs, "systems": {}}

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
            print(f"[CIFAR-10] {name:10s} seed {seed}: "
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
        fin = [p["class_il"]["ACC"] for p
               in out["systems"]["finetune"]["per_seed"]]
        best = max(MARS_VARIANTS, key=lambda v:
                   out["systems"][v]["agg"]["class_il_ACC"]["mean"])
        mars = [p["class_il"]["ACC"] for p
                in out["systems"][best]["per_seed"]]
        d_rep = stats([(a - b) * 100 for a, b in zip(mars, rep)])
        d_fin = stats([(a - b) * 100 for a, b in zip(mars, fin)])
        noise = (stats([r * 100 for r in rep])["std"]
                 + stats([m_ * 100 for m_ in mars])["std"])
        verdict = {"best_mars": best, "delta_vs_replay_pp": d_rep,
                   "delta_vs_finetune_pp": d_fin,
                   "noise_pp": round(noise, 4),
                   "verdict": ("SYGNAL+ (teza przenosi sie na CIFAR)"
                               if d_rep["mean"] >= -noise
                               else "PONIZEJ REPLAY (diagnoza: reprezentacja"
                                    " -- granica losowych cech)")}
    out["verdict"] = verdict

    # ---------- raport ----------
    print(f"\n--- Split-CIFAR-10 (n={n_seeds}) -- class-IL ---")
    for name, _ in systems:
        a = out["systems"][name]["agg"]
        f_str = (f" | F {a['class_il_forgetting']['mean']*100:.1f}pp"
                 if "class_il_forgetting" in a else "  (sufit)")
        print(f"  {name:10s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
              f"+/-{a['class_il_ACC']['std']*100:.2f}% "
              f"(min {a['class_il_ACC']['min']*100:.2f}%){f_str}")
    if verdict:
        print(f"  WERDYKT ({verdict['best_mars']} vs replay, "
              f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
        print(f"    d vs replay: {verdict['delta_vs_replay_pp']['mean']:+.2f}pp"
              f" | d vs finetune: "
              f"{verdict['delta_vs_finetune_pp']['mean']:+.2f}pp")

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "F4_split_cifar_smoke.json" if args.smoke else "F4_split_cifar.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
