"""
run_E2_hierarchical.py -- E2: routing hierarchiczny vs plaski (multi-seed).

Hipoteza (DROGA_E_PLAN.md, sekcja E2):
  Zmiana STRUKTURY decyzji (grupa -> klasa) omija sufit algorytmiczny
  z D4/D5/D7, bo specjalista grupowy trenuje sie na nowym zadaniu
  (rozroznianie confusables wewnatrz grupy). E1: 64% bledow routera
  w 4 parach klastra upper-body, 79% luki w Q1 pewnosci.

Kryterium werdyktu (Z GORY, Fashion, najlepszy wariant vs plaski CNN D6):
  delta system > prog szumu (std_flat + std_hier)  -> SYGNAL+
      (struktura decyzji lamie sufit -- duzy wynik: hierarchia > algorytmy)
  |delta| <= prog                                  -> SZUM
      (sufit reprezentacji obejmuje takze strukture decyzji -> zostaja
       wylacznie cechy i skala; domyka droge routingu ostatecznie)
  delta < -prog                                    -> SYGNAL-
UWAGA MAC: pody hier maja wiecej pracy (5-way vs ~1-way); pre-rejestrowany
sweep pod_hidden {24, 64}. MAC raportowany uczciwie dla kazdego wariantu.

Baseline: per-seed wyniki plaskiego CNN z results/D6_cnn_backbone.json
(seedy 0..4, ten sam protokol -- porownanie parami jak w D6b).

GRUPY: wstepnie z E1 SMOKE (1 seed). PODSTAWIC finalne po pelnym E1
(results/E1_error_anatomy.json) jesli top pary sie zmienia!

Tryb szybki:  python src/run_E2_hierarchical.py --smoke
Pelny:        python src/run_E2_hierarchical.py
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2_hier import MarsV2HierSystem, train_phased_hier, evaluate_hier
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
D6_PATH = os.path.join(RESULTS_DIR, "D6_cnn_backbone.json")

BB_H, EMB = 128, 32
CHANNELS = (32, 64)      # pelny CNN D6 (baseline 92.0% Fashion)
POD_HIDDENS = [24, 64]   # pre-rejestrowany sweep (pody grup robia wiecej)

# --- GRUPY z E1 (Fashion: klaster upper-body + obuwie; patrz E1 wyniki) ---
# 0 T-shirt, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat,
# 5 Sandal,  6 Shirt,   7 Sneaker,  8 Bag,   9 AnkleBoot
GROUPS = {
    # Fashion: POTWIERDZONE pelnym E1 (5 seedow): wszystkie top-6 par konfuzji
    # wewnatrz klastra upper-body; 86% probek odzyskiwalnych to te klasy.
    "Fashion-MNIST": [[0, 2, 3, 4, 6], [5, 7, 9], [1], [8]],
    # MNIST: koncentracja slaba (42% w top-4, E1) -- test kontrolny.
    # Grupy z danych E1 (greedy pokrycie par; hub = cyfra 9): pokrywa
    # pary 4-9(63), 2-7(41), 7-9(29), 3-5(34), 0-6(27).
    "MNIST": [[2, 4, 7, 9], [3, 5], [0, 6], [1], [8]],
}


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, 8 epok")
    ap.add_argument("--datasets", nargs="+", default=["Fashion-MNIST", "MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 5
    epochs = 8 if args.smoke else 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d6 = None
    if os.path.exists(D6_PATH):
        with open(D6_PATH, encoding="utf-8") as f:
            d6 = json.load(f)
    elif not args.smoke:
        sys.exit("BLAD: brak results/D6_cnn_backbone.json (baseline plaski).")

    print("=" * 72)
    print(f"E2 -- routing hierarchiczny vs plaski CNN  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds={n_seeds} | epochs={epochs} | "
          f"pod_hidden sweep={POD_HIDDENS}")
    print("=" * 72)

    t0 = time.perf_counter()
    out = {"experiment": "E2_hierarchical", "device": device,
           "n_seeds": n_seeds, "epochs": epochs,
           "groups": GROUPS, "pod_hiddens": POD_HIDDENS, "datasets": {}}

    for ds_name in args.datasets:
        groups = GROUPS[ds_name]
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        print(f"[{ds_name}] grupy: {groups}")

        per_seed = []
        macs = {}
        for seed in range(n_seeds):
            row = {}
            for ph in POD_HIDDENS:
                torch.manual_seed(seed)
                m = MarsV2HierSystem(groups, backbone_hidden=BB_H,
                                     emb_dim=EMB, pod_hidden=ph,
                                     channels=CHANNELS).to(device)
                train_phased_hier(m, Xtr, ytr, epochs=epochs, device=device)
                g, s, o = evaluate_hier(m, Xte, yte)
                row[f"ph{ph}"] = {"group_routing": g, "system": s, "oracle": o}
                if f"ph{ph}" not in macs:
                    macs[f"ph{ph}"] = m.mac_per_sample_top1()
                print(f"[{ds_name}] seed {seed} ph={ph:3d}: "
                      f"sys={s*100:.2f}% grupRout={g*100:.2f}% "
                      f"orac={o*100:.2f}%")
            per_seed.append(row)

        # --- agregacja + delty vs plaski CNN (per-seed z D6) ---
        agg = {}
        d6_ds = d6["datasets"][ds_name] if d6 else None
        flat_sys = ([p["cnn"]["system"] for p in d6_ds["per_seed"]]
                    if d6_ds else None)
        flat_mac = d6_ds["agg"]["cnn_mac_total"] if d6_ds else None

        for ph in POD_HIDDENS:
            key = f"ph{ph}"
            v = {m: stats([p[key][m] for p in per_seed])
                 for m in ("group_routing", "system", "oracle")}
            v["mac_total"] = macs[key]["total_top1"]
            if flat_mac:
                v["mac_ratio_vs_flat"] = round(v["mac_total"] / flat_mac, 3)
            if flat_sys and len(flat_sys) >= n_seeds:
                deltas = [(per_seed[i][key]["system"] - flat_sys[i]) * 100
                          for i in range(n_seeds)]
                v["delta_vs_flat_pp"] = stats(deltas)
            agg[key] = v

        # --- werdykt ---
        verdict = None
        if not args.smoke and flat_sys:
            flat_std_pp = stats([s * 100 for s in flat_sys])["std"]
            best = max(POD_HIDDENS, key=lambda ph:
                       agg[f"ph{ph}"]["delta_vs_flat_pp"]["mean"])
            d = agg[f"ph{best}"]["delta_vs_flat_pp"]
            noise = flat_std_pp + d["std"]
            if d["mean"] > noise:
                v_str = "SYGNAL+"
            elif d["mean"] < -noise:
                v_str = "SYGNAL-"
            else:
                v_str = "SZUM"
            verdict = {"best_variant": f"ph{best}", "delta_pp": d,
                       "noise_pp": round(noise, 4), "verdict": v_str}

        # --- raport ---
        print(f"\n--- {ds_name} (n={n_seeds}) ---")
        if d6_ds:
            a = d6_ds["agg"]
            print(f"  flat CNN (D6): sys {a['cnn_system']['mean']*100:.2f}"
                  f"+/-{a['cnn_system']['std']*100:.2f}% | "
                  f"MAC {flat_mac:,}")
        for ph in POD_HIDDENS:
            v = agg[f"ph{ph}"]
            extra = ""
            if "delta_vs_flat_pp" in v:
                d_ = v["delta_vs_flat_pp"]
                extra = f" | dSys {d_['mean']:+.2f}+/-{d_['std']:.2f}pp"
            print(f"  hier ph={ph:3d}: sys {v['system']['mean']*100:.2f}"
                  f"+/-{v['system']['std']*100:.2f}% | "
                  f"grupRout {v['group_routing']['mean']*100:.2f}% | "
                  f"orac {v['oracle']['mean']*100:.2f}% | "
                  f"MAC {v['mac_total']:,}{extra}")
        if verdict:
            print(f"  WERDYKT ({verdict['best_variant']}, "
                  f"prog {verdict['noise_pp']:.2f}pp): {verdict['verdict']}")
        print()

        out["datasets"][ds_name] = {"per_seed": per_seed, "agg": agg,
                                    "verdict": verdict}

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "E2_hierarchical_smoke.json" if args.smoke else "E2_hierarchical.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
