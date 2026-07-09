"""
run_E1_error_anatomy.py -- E1: anatomia luki router->oracle (multi-seed).

Pytanie (DROGA_E_PLAN.md):
  Z czego DOKLADNIE sklada sie luka router->oracle (6.11pp na CNN, Fashion)?
  D4/D5/D7 pokazaly, ze algorytmy routingu jej nie zamkna. E1 pyta: KTORE
  probki ja tworza i czy maja strukture (pary klas? niska pewnosc?).

Mierzy per seed (pelny CNN D6, train_phased jak cala seria):
  1. Dekompozycja probek na 4 rozlaczne klasy:
       A: router OK,  pod OK   (sukces)
       B: router OK,  pod ZLE  ("pod miss" -- wlasny pod zawodzi)
       C: router ZLE, system OK ("odzyskanie" -- zly pod i tak trafia)
       D: router ZLE, system ZLE (strata wlasciwa)
  2. Zbior ODZYSKIWALNY: oracle OK & system ZLE (= materialna luka);
     rozklad po klasach prawdziwych.
  3. Macierz konfuzji routera + top pary myłek (symetryzowane).
  4. Blad systemu vs confidence routera (kwartyle) -- czy luka jest
     niskopewna (mechanizmy selektywne maja target) czy rozsmarowana.

To DIAGNOSTYKA (bez werdyktu SYGNAL/SZUM) -- kryterium ciekawosci w planie:
>=60% luki w <=4 parach klas => E2 (hierarchia) ma silny cel.

Tryb szybki:  python src/run_E1_error_anatomy.py --smoke   (1 seed, 8 epok)
Pelny:        python src/run_E1_error_anatomy.py           (5 seedow, 30 epok)
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import train_phased
from mars_v2_cnn import MarsV2CNNSystem
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BB_H, EMB, POD_H = 128, 32, 24
CHANNELS = (32, 64)   # pelny CNN D6 -- backbone o najwiekszej luce (6.11pp)

CLASS_NAMES = {
    "Fashion-MNIST": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot"],
    "MNIST": [str(i) for i in range(10)],
}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def analyze_seed(model, Xte, yte, names):
    """Pelna anatomia jednego wytrenowanego modelu. Zwraca dict."""
    model.eval()
    with torch.no_grad():
        feats = model.features(Xte)
        logits = model.route_logits(feats)
        probs = torch.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        ids = logits.argmax(dim=1)

        sys_pred = model.pod_forward(feats, ids).argmax(1)
        ora_pred = model.pod_forward(feats, yte).argmax(1)

    n = len(yte)
    r_ok = ids == yte
    s_ok = sys_pred == yte
    o_ok = ora_pred == yte

    # --- 1. dekompozycja A/B/C/D ---
    A = (r_ok & s_ok).sum().item()
    B = (r_ok & ~s_ok).sum().item()
    C = (~r_ok & s_ok).sum().item()
    D = (~r_ok & ~s_ok).sum().item()

    # --- 2. zbior odzyskiwalny (= luka oracle-system w probkach) ---
    rec_mask = o_ok & ~s_ok
    rec_by_class = Counter(yte[rec_mask].tolist())

    # --- 3. konfuzje routera (tylko bledy), pary symetryzowane ---
    err_mask = ~r_ok
    pairs = Counter()
    for t, p in zip(yte[err_mask].tolist(), ids[err_mask].tolist()):
        pairs[tuple(sorted((t, p)))] += 1
    top_pairs = [{"pair": f"{names[a]}<->{names[b]}", "count": c}
                 for (a, b), c in pairs.most_common(10)]
    # koncentracja luki: ile % bledow routera siedzi w top-4 parach
    total_err = max(err_mask.sum().item(), 1)
    top4_share = sum(c for _, c in pairs.most_common(4)) / total_err

    # --- 4. blad vs confidence (kwartyle confidence) ---
    q = torch.quantile(conf, torch.tensor([0.25, 0.5, 0.75], device=conf.device))
    bins = [(conf <= q[0]),
            (conf > q[0]) & (conf <= q[1]),
            (conf > q[1]) & (conf <= q[2]),
            (conf > q[2])]
    conf_bins = []
    for i, m in enumerate(bins):
        cnt = max(m.sum().item(), 1)
        conf_bins.append({
            "quartile": f"Q{i+1}",
            "conf_range": [round(conf[m].min().item(), 3),
                           round(conf[m].max().item(), 3)],
            "sys_err_pct": round((~s_ok & m).sum().item() / cnt * 100, 2),
            "recoverable_pct": round((rec_mask & m).sum().item() / cnt * 100, 2),
        })
    # ile % calego zbioru odzyskiwalnego siedzi w najnizszym kwartylu pewnosci
    rec_total = max(rec_mask.sum().item(), 1)
    rec_in_q1 = (rec_mask & bins[0]).sum().item() / rec_total

    return {
        "routing_acc": round(r_ok.float().mean().item(), 4),
        "system_acc": round(s_ok.float().mean().item(), 4),
        "oracle_acc": round(o_ok.float().mean().item(), 4),
        "decomp": {"A_ok": A, "B_pod_miss": B, "C_recovery": C, "D_loss": D,
                   "n": n},
        "recoverable_n": rec_mask.sum().item(),
        "recoverable_by_class": {names[k]: v for k, v
                                 in sorted(rec_by_class.items())},
        "top_confusion_pairs": top_pairs,
        "top4_pairs_share_of_router_errors": round(top4_share, 3),
        "conf_bins": conf_bins,
        "recoverable_in_lowest_conf_quartile": round(rec_in_q1, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, 8 epok")
    ap.add_argument("--datasets", nargs="+", default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 8 if args.smoke else 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"E1 -- anatomia bledu (pelny CNN)  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epochs={epochs}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "E1_error_anatomy", "device": device,
           "n_seeds": n_seeds, "epochs": epochs, "channels": list(CHANNELS),
           "datasets": {}}

    for ds_name in args.datasets:
        names = CLASS_NAMES[ds_name]
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)

        per_seed = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model = MarsV2CNNSystem(backbone_hidden=BB_H, emb_dim=EMB,
                                    pod_hidden=POD_H,
                                    channels=CHANNELS).to(device)
            train_phased(model, Xtr, ytr, epochs=epochs, device=device)
            r = analyze_seed(model, Xte, yte, names)
            per_seed.append(r)
            d = r["decomp"]
            print(f"[{ds_name}] seed {seed}: sys={r['system_acc']*100:.2f}% | "
                  f"A={d['A_ok']} B_podmiss={d['B_pod_miss']} "
                  f"C_recov={d['C_recovery']} D_loss={d['D_loss']} | "
                  f"odzyskiwalne={r['recoverable_n']} "
                  f"(top4 pary: {r['top4_pairs_share_of_router_errors']*100:.0f}% "
                  f"bledow routera)")

        # --- agregacja ---
        agg = {
            "system_acc": stats([p["system_acc"] for p in per_seed]),
            "oracle_acc": stats([p["oracle_acc"] for p in per_seed]),
            "recoverable_n": stats([p["recoverable_n"] for p in per_seed]),
            "top4_share": stats([p["top4_pairs_share_of_router_errors"]
                                 for p in per_seed]),
            "recoverable_in_q1_conf": stats(
                [p["recoverable_in_lowest_conf_quartile"] for p in per_seed]),
        }
        # sumaryczne pary konfuzji przez seedy
        all_pairs = Counter()
        for p in per_seed:
            for tp in p["top_confusion_pairs"]:
                all_pairs[tp["pair"]] += tp["count"]
        agg["confusion_pairs_summed"] = [
            {"pair": k, "count": v} for k, v in all_pairs.most_common(10)]
        # sumaryczny rozklad odzyskiwalnych po klasach
        rec_cls = Counter()
        for p in per_seed:
            rec_cls.update(p["recoverable_by_class"])
        agg["recoverable_by_class_summed"] = dict(rec_cls.most_common())

        print(f"\n--- {ds_name} (n={n_seeds}) ---")
        print(f"  system {agg['system_acc']['mean']*100:.2f}% | "
              f"oracle {agg['oracle_acc']['mean']*100:.2f}% | "
              f"odzyskiwalne/seed: {agg['recoverable_n']['mean']:.0f} probek")
        print(f"  koncentracja: top-4 pary = "
              f"{agg['top4_share']['mean']*100:.0f}% bledow routera "
              f"(kryterium E2: >=60%)")
        print(f"  odzyskiwalne w Q1 pewnosci: "
              f"{agg['recoverable_in_q1_conf']['mean']*100:.0f}% "
              f"(wysoko = mechanizmy selektywne maja target)")
        print("  top pary konfuzji (suma seedow):")
        for tp in agg["confusion_pairs_summed"][:6]:
            print(f"    {tp['pair']:24s} {tp['count']}")
        print("  odzyskiwalne wg klasy (suma seedow):")
        for k, v in list(agg["recoverable_by_class_summed"].items())[:6]:
            print(f"    {k:12s} {v}")
        print()

        out["datasets"][ds_name] = {"per_seed": per_seed, "agg": agg}

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "E1_error_anatomy_smoke.json" if args.smoke else "E1_error_anatomy.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
