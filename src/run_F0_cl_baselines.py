"""
run_F0_cl_baselines.py -- F0: baseline'y continual learning (multi-seed).

Cel (DROGA_F_PLAN.md, sekcja 2):
  Kalibracja pola gry PRZED wejsciem MARS-CL (F1). Bez tych liczb wyniki F1
  nic nie znacza. Cztery baseline'y na wspolnej architekturze MonoS2
  (backbone S2 + glowica 10-way -- ten sam budzet cech co przyszly MARS-CL):

  finetune : trening sekwencyjny bez ochrony -- oczekiwana katastrofa
             (dolna granica; sanity: class-IL forgetting bliski 100%).
  joint    : trening na wszystkich danych naraz -- gorna granica
             (nieosiagalna w CL; UCZCIWY sufit, bez oracle inflation).
  replay   : finetune + zbalansowany bufor 200 probek z przeszlosci --
             najprostsza silna metoda; GLOWNY przeciwnik dla F1.
  ewc      : finetune + kara EWC (online, Fisher diagonalny),
             sweep lambda {100, 1000} -- klasyczna regularyzacja.

Protokol: Split-MNIST i Split-Fashion (5 zadan x 2 klasy, TASKS5),
po kazdym zadaniu ewaluacja na zadaniach 0..t w OBU protokolach
(class-IL glowny, task-IL pomocniczy). Metryki: ACC, Forgetting, BWT
(cl_common.cl_metrics), macierze R w JSON.

To jest KALIBRACJA -- bez werdyktu SYGNAL/SZUM. Sanity checki:
  1. finetune class-IL: ACC ~ acc ostatniego zadania / T (katastrofa).
  2. joint: ACC zblizony do poziomu S2 z D6b (~91% Fashion).
  3. replay > finetune wyraznie; EWC miedzy nimi (lekcja Etapu 2: EWC
     na wspolnej malej sieci nie wystarcza).

Tryb szybki:  python src/run_F0_cl_baselines.py --smoke  (1 seed, 4 epoki/zadanie)
Pelny:        python src/run_F0_cl_baselines.py          (5 seedow, 15 epok/zadanie)
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
from cl_common import (TASKS5, MonoS2, make_task_data, eval_protocols,
                       cl_metrics, balanced_buffer, EWCState)
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EWC_LAMBDAS = [100.0, 1000.0]
REPLAY_SIZE = 200
LR, BATCH = 0.001, 512


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def train_epochs(model, X, y, epochs, device, ewc=None, lam=0.0,
                 replay=None, replay_bs=128):
    """
    replay: opcjonalna para (bx, by) -- bufor experience replay. Bufor jest
    PROBKOWANY W KAZDYM KROKU (dolaczany do kazdego mini-batcha), nie
    konkatenowany raz -- inaczej 200 probek bufora tonie w ~12k probek
    nowego zadania i replay nie dziala (lekcja ze smoke'a F0).
    """
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
            if ewc is not None and lam > 0:
                loss = loss + (lam / 2.0) * ewc.penalty(model)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_sequence(method, task_data, seed, epochs, device, lam=0.0):
    """
    Jedna sekwencja CL dla danej metody. Zwraca macierze R (class/task-IL).
    method: "finetune" | "replay" | "ewc"
    """
    torch.manual_seed(seed)
    model = MonoS2().to(device)
    ewc = EWCState(model) if method == "ewc" else None

    R_c, R_t = [], []
    seen = []
    for t, td in enumerate(task_data):
        seen = seen + td["classes"]
        Xt, yt = td["Xtr"], td["ytr"]
        replay = None
        if method == "replay" and t > 0:
            replay = balanced_buffer(task_data, t - 1, REPLAY_SIZE, seed)
        train_epochs(model, Xt, yt, epochs, device,
                     ewc=ewc if t > 0 else None, lam=lam, replay=replay)
        if method == "ewc":
            ewc.update_fisher(model, td["Xtr"], td["ytr"])
        model.eval()
        row_c, row_t = eval_protocols(model.forward, task_data, t, seen)
        R_c.append(row_c)
        R_t.append(row_t)
    return R_c, R_t


def run_joint(task_data, seed, epochs, device):
    """Gorna granica: wszystkie dane naraz (epochs * T dla rownosci budzetu)."""
    torch.manual_seed(seed)
    model = MonoS2().to(device)
    X = torch.cat([td["Xtr"] for td in task_data])
    y = torch.cat([td["ytr"] for td in task_data])
    train_epochs(model, X, y, epochs * len(task_data), device)
    model.eval()
    seen = [c for td in task_data for c in td["classes"]]
    row_c, row_t = eval_protocols(model.forward, task_data,
                                  len(task_data) - 1, seen)
    return row_c, row_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="+",
                    default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 4 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    methods = ([("finetune", 0.0), ("replay", 0.0)]
               + [("ewc", lam) for lam in EWC_LAMBDAS])

    print("=" * 72)
    print(f"F0 -- baseline'y CL (Split, 5 zadan x 2 klasy)  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epok/zadanie={epochs} | "
          f"replay={REPLAY_SIZE} | EWC lambda={EWC_LAMBDAS}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "F0_cl_baselines", "device": device,
           "n_seeds": n_seeds, "epochs_per_task": epochs,
           "tasks": TASKS5, "replay_size": REPLAY_SIZE,
           "ewc_lambdas": EWC_LAMBDAS, "datasets": {}}

    for ds_name in args.datasets:
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        task_data = make_task_data(Xtr, ytr, Xte, yte)

        res = {"methods": {}}
        for method, lam in methods:
            name = f"{method}_l{lam:g}" if method == "ewc" else method
            per_seed = []
            for seed in range(n_seeds):
                R_c, R_t = run_sequence(method, task_data, seed, epochs,
                                        device, lam=lam)
                m_c, m_t = cl_metrics(R_c), cl_metrics(R_t)
                per_seed.append({"R_class_il": R_c, "R_task_il": R_t,
                                 "class_il": m_c, "task_il": m_t})
                print(f"[{ds_name}] {name:12s} seed {seed}: "
                      f"class-IL ACC={m_c['ACC']*100:.2f}% "
                      f"F={m_c['forgetting']*100:.1f}pp | "
                      f"task-IL ACC={m_t['ACC']*100:.2f}%")
            agg = {}
            for proto in ("class_il", "task_il"):
                for metric in ("ACC", "forgetting", "BWT"):
                    agg[f"{proto}_{metric}"] = stats(
                        [p[proto][metric] for p in per_seed])
            res["methods"][name] = {"per_seed": per_seed, "agg": agg}

        # joint (gorna granica)
        per_seed_j = []
        for seed in range(n_seeds):
            row_c, row_t = run_joint(task_data, seed, epochs, device)
            per_seed_j.append({"final_class_il": row_c,
                               "final_task_il": row_t,
                               "ACC_class_il": round(sum(row_c) / len(row_c), 4),
                               "ACC_task_il": round(sum(row_t) / len(row_t), 4)})
            print(f"[{ds_name}] joint        seed {seed}: "
                  f"class-IL ACC={per_seed_j[-1]['ACC_class_il']*100:.2f}%")
        res["methods"]["joint"] = {
            "per_seed": per_seed_j,
            "agg": {"class_il_ACC": stats([p["ACC_class_il"]
                                           for p in per_seed_j]),
                    "task_il_ACC": stats([p["ACC_task_il"]
                                          for p in per_seed_j])}}

        # ---------- raport ----------
        print(f"\n--- {ds_name} (n={n_seeds}) -- class-IL (glowny protokol) ---")
        for name, r in res["methods"].items():
            a = r["agg"]
            if name == "joint":
                print(f"  {name:12s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                      f"+/-{a['class_il_ACC']['std']*100:.2f}%  (sufit)")
            else:
                print(f"  {name:12s}: ACC {a['class_il_ACC']['mean']*100:.2f}"
                      f"+/-{a['class_il_ACC']['std']*100:.2f}% | "
                      f"forgetting {a['class_il_forgetting']['mean']*100:.1f}pp | "
                      f"BWT {a['class_il_BWT']['mean']*100:+.1f}pp")
        print()
        out["datasets"][ds_name] = res

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "F0_cl_baselines_smoke.json" if args.smoke else "F0_cl_baselines.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
