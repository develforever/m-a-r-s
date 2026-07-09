"""
run_D4_consultation.py -- Droga D, D4: Consultation routera.

Przy niskim confidence routera (conf < theta): zamiast top-1 poda, pytamy
top-k podow i agregujemy ich wyniki (wazone prawdopodobienstwem routingu).
Mierzymy zysk accuracy vs koszt MAC.

Mechanizm inferencji:
  conf >= theta  ->  top-1 pod   (1 * pod_mac)
  conf <  theta  ->  top-k pods  (k * pod_mac), agregacja wazona conf_k

Agregacja top-k:
  w_i = routing_prob[top_i] / sum(routing_prob[top_1..k])  (normalizacja)
  wynik = argmax( sum_i( w_i * softmax(pod_i_out) ) )

Hipoteza: router jest niepewny wlasnie wtedy, gdy probka lezy na granicy
miedzy podami. Zapytanie kilku podow moze poprawic decyzje.

Kontekst historyczny:
  - Selective Top-2 z D1b Pareto sweep: "neutralny do lekko szkodliwy" (ORACLE
    wtedy byl zawyzonyz). D1c/v2 phased to uczciwy punkt startowy.
  - D4 uogolnia to do top-k (k=2,3,5) z thresh-based activation.
  - Niezalezna galaz wzgledem D1c -- inny punkt startowy.

Protokol:
  - Base: v2 phased (jak D1c), trenowany raz per seed.
  - Sweep theta x {0.0 (base), 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01}
         k    x {2, 3, 5}
  - Brak dodatkowego treningu -- czysta inferencja.
  - 5 seedow, oba datasety.
  - Raport: accuracy vs avg_MAC per konfiguracja (baza do krzywej Pareto).

Baseline odniesienia (D1c/v2a):
  MNIST:         98.36 +/- 0.05pp
  Fashion-MNIST: 89.50 +/- 0.11pp

Uruchom:
    .venv/Scripts/python.exe src/run_D4_consultation.py
"""
import json, math, os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import MarsV2System, train_phased, evaluate, N_IN, N_CLASSES
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_SEEDS          = 5
EPOCHS           = 30
BB_H, EMB, POD_H = 384, 32, 24   # jak D1c

# Sweep: theta=0.0 odpowiada czystemu top-1 (baza), theta=1.01 = wszyscy konsultuja
THETA_GRID  = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
K_GRID      = [2, 3, 5]
BATCH_EVAL  = 2000   # batch dla inferencji (calosc testu miesci sie w RAM/VRAM)


def consult_eval(model, Xte, yte, theta, n_consult):
    """
    Ewaluacja z mechanizmem consultation.
    Zwraca (acc, avg_mac, pct_consult).
    """
    model.eval()
    mac_info = model.mac_per_sample_top1()
    fixed_mac = mac_info["backbone"] + mac_info["routing"]
    pod_mac   = mac_info["pod"]

    all_preds    = []
    n_high_total = 0
    n_low_total  = 0

    with torch.no_grad():
        for s in range(0, len(Xte), BATCH_EVAL):
            x = Xte[s:s + BATCH_EVAL]
            b = len(x)

            feats  = model.features(x)
            logits = model.route_logits(feats)
            probs  = torch.softmax(logits, dim=1)

            # Top-k wg routera
            top_probs, top_ids = probs.topk(n_consult, dim=1)  # [B, k]
            best_conf = top_probs[:, 0]                          # max prob per probke

            high_mask = best_conf >= theta
            low_mask  = ~high_mask
            n_high_total += int(high_mask.sum())
            n_low_total  += int(low_mask.sum())

            batch_preds = torch.zeros(b, dtype=torch.long, device=x.device)

            # Wysoki confidence: zwykly top-1
            if high_mask.any():
                pod_ids = top_ids[high_mask, 0]
                out = model.pod_forward(feats[high_mask], pod_ids)
                batch_preds[high_mask] = out.argmax(1)

            # Niski confidence: konsultacja top-k podow
            if low_mask.any():
                f_low = feats[low_mask]
                w = top_probs[low_mask]                            # [n_low, k]
                w = w / w.sum(dim=1, keepdim=True)                # normalizacja

                n_low_b = int(low_mask.sum())
                agg = torch.zeros(n_low_b, model.n_out, device=x.device)
                for ki in range(n_consult):
                    pod_k_ids = top_ids[low_mask, ki]
                    pod_k_out = model.pod_forward(f_low, pod_k_ids)
                    # wazona suma softmax-ow podow
                    agg += w[:, ki:ki + 1] * torch.softmax(pod_k_out, dim=1)
                batch_preds[low_mask] = agg.argmax(1)

            all_preds.append(batch_preds)

    preds = torch.cat(all_preds)
    n     = len(Xte)
    acc   = (preds == yte).float().mean().item()

    avg_mac     = fixed_mac + pod_mac * (n_high_total + n_consult * n_low_total) / n
    pct_consult = n_low_total / n * 100

    return acc, round(avg_mac), round(pct_consult, 1)


# ======================================================================= utils
def stats(vals):
    n    = len(vals)
    mean = sum(vals) / n
    var  = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    std  = math.sqrt(var)
    return {"mean": round(mean, 4), "std": round(std, 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


# ===================================================================== per-seed
def run_one_seed(Xtr, ytr, Xte, yte, seed, device):
    """
    Trenuje base phased (raz), potem ewaluuje wszystkie konfiguracje (theta, k).
    """
    torch.manual_seed(seed)
    model = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES).to(device)
    train_phased(model, Xtr, ytr, epochs=EPOCHS, device=device)

    # Baseline top-1 (przez evaluate -- dla spojnosci z D1c)
    _, base_acc, _ = evaluate(model, Xte, yte)

    configs = {}
    # theta=0.0 -> wszyscy high -> top-1 baseline (dla sprawdzenia spojnosci)
    for k in K_GRID:
        acc, avg_mac, pct = consult_eval(model, Xte, yte, theta=0.0, n_consult=k)
        configs[f"top1_k{k}"] = {"acc": acc, "avg_mac": avg_mac,
                                  "pct_consult": pct, "theta": 0.0, "k": k}

    for theta in THETA_GRID[1:]:   # pomijamy 0.0 (juz wyzej)
        for k in K_GRID:
            key = f"th{theta}_k{k}"
            acc, avg_mac, pct = consult_eval(model, Xte, yte, theta=theta, n_consult=k)
            configs[key] = {"acc": acc, "avg_mac": avg_mac,
                            "pct_consult": pct, "theta": theta, "k": k}

    return {"base_acc": base_acc, "configs": configs}


# ==================================================================== dataset
def run_dataset(ds_name, device):
    print(f"\n{'='*72}\nDataset: {ds_name}  ({N_SEEDS} seeds)\n{'='*72}")
    Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

    mac_ref = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES)
    mac_info = mac_ref.mac_per_sample_top1()
    print(f"MAC top-1 (referencja): {mac_info['total_top1']:,}  "
          f"(backbone={mac_info['backbone']:,}, pod={mac_info['pod']:,})")

    per_seed = []
    base_accs = []
    print(f"\n{'seed':>4} | {'base':>8}", end="")
    for theta in [0.7, 0.9, 0.99, 1.01]:
        for k in [2, 3]:
            print(f" | th{theta} k{k}", end="")
    print()
    print("-" * 72)

    for seed in range(N_SEEDS):
        r = run_one_seed(Xtr, ytr, Xte, yte, seed, device)
        per_seed.append(r)
        base_accs.append(r["base_acc"])
        print(f"{seed:>4} | {r['base_acc']*100:>7.2f}%", end="")
        for theta in [0.7, 0.9, 0.99, 1.01]:
            for k in [2, 3]:
                key = f"th{theta}_k{k}"
                print(f" | {r['configs'][key]['acc']*100:>7.2f}%", end="")
        print()

    base_stats = stats(base_accs)

    # Zbierz wszystkie konfiguracje i policz mean acc po seedach
    all_keys = list(per_seed[0]["configs"].keys())
    config_stats = {}
    for key in all_keys:
        accs   = [per_seed[s]["configs"][key]["acc"] for s in range(N_SEEDS)]
        macs   = [per_seed[s]["configs"][key]["avg_mac"] for s in range(N_SEEDS)]
        pcts   = [per_seed[s]["configs"][key]["pct_consult"] for s in range(N_SEEDS)]
        theta  = per_seed[0]["configs"][key]["theta"]
        k      = per_seed[0]["configs"][key]["k"]
        config_stats[key] = {
            "theta": theta, "k": k,
            "acc":   stats(accs),
            "mac":   stats(macs),
            "pct_consult": stats(pcts),
        }

    # Raport top konfiguracji (posortowane po acc mean)
    print(f"\n  Base (top-1): {base_stats['mean']*100:.2f} +/- {base_stats['std']*100:.2f}%")
    print(f"\n  Top-10 konfiguracji wg mean acc:")
    print(f"  {'config':<18} {'acc mean':>10} {'delta':>8} {'avg_mac':>10} "
          f"{'%consult':>10}")
    print("  " + "-" * 65)
    sorted_configs = sorted(config_stats.items(),
                            key=lambda x: x[1]["acc"]["mean"], reverse=True)
    for key, cs in sorted_configs[:10]:
        d      = cs["acc"]["mean"] - base_stats["mean"]
        pooled = max(base_stats["std"], cs["acc"]["std"])
        sig    = "S" if abs(d) > pooled else "."
        print(f"  {key:<18} {cs['acc']['mean']*100:>8.2f}%  "
              f"{d*100:>+7.2f}pp{sig}  {cs['mac']['mean']:>9,.0f}  "
              f"{cs['pct_consult']['mean']:>8.1f}%")

    # Pareto front (acc vs mac, per srednia)
    print(f"\n  Pareto front (accuracy vs MAC):")
    points = [{"key": k, "acc": cs["acc"]["mean"], "mac": cs["mac"]["mean"],
               "k_val": cs["k"], "theta": cs["theta"]}
              for k, cs in config_stats.items()]
    # Dodaj base
    base_mac = mac_info["total_top1"]
    points.append({"key": "base_top1", "acc": base_stats["mean"],
                   "mac": base_mac, "k_val": 1, "theta": 0.0})
    # Pareto: nie zdominowany (nie istnieje punkt o >= acc I <= mac)
    front = []
    for p in points:
        dominated = any(q["acc"] >= p["acc"] and q["mac"] <= p["mac"]
                        and (q["acc"] > p["acc"] or q["mac"] < p["mac"])
                        for q in points)
        if not dominated:
            front.append(p)
    front.sort(key=lambda x: x["mac"])
    print(f"  {'config':<18} {'acc':>10} {'avg_mac':>12}")
    for p in front:
        print(f"  {p['key']:<18} {p['acc']*100:>8.2f}%  {p['mac']:>11,.0f}")

    return {
        "dataset": ds_name, "n_seeds": N_SEEDS,
        "config": {"epochs": EPOCHS, "bb_h": BB_H, "emb": EMB, "pod_h": POD_H,
                   "theta_grid": THETA_GRID, "k_grid": K_GRID},
        "base_stats": base_stats,
        "per_seed": per_seed,
        "config_stats": config_stats,
        "pareto_front": front,
    }


# ======================================================================== main
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("DROGA D -- D4: Consultation routera")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    print(f"Seeds: {N_SEEDS}  |  Epochs: {EPOCHS}")
    print(f"Theta grid: {THETA_GRID}")
    print(f"K grid: {K_GRID}")
    print("Baseline odniesienia (D1c/v2a):")
    print("  MNIST:         98.36 +/- 0.05pp")
    print("  Fashion-MNIST: 89.50 +/- 0.11pp")
    print("=" * 72)

    results = {}
    for ds_name in ["MNIST", "Fashion-MNIST"]:
        results[ds_name] = run_dataset(ds_name, device)

    print("\n" + "=" * 72)
    print("PODSUMOWANIE")
    print("=" * 72)
    for ds_name in ["MNIST", "Fashion-MNIST"]:
        r  = results[ds_name]
        bm = r["base_stats"]["mean"] * 100
        bs = r["base_stats"]["std"]  * 100
        print(f"\n  {ds_name}  (base={bm:.2f}+/-{bs:.2f}%):")
        # Najlepsza konfiguracja
        best_key, best_cs = max(r["config_stats"].items(),
                                key=lambda x: x[1]["acc"]["mean"])
        d      = best_cs["acc"]["mean"] - r["base_stats"]["mean"]
        pooled = max(r["base_stats"]["std"], best_cs["acc"]["std"])
        sig    = "SYGNAL" if abs(d) > pooled else "SZUM"
        print(f"    Best: {best_key:<18} "
              f"acc={best_cs['acc']['mean']*100:.2f}+/-{best_cs['acc']['std']*100:.2f}%  "
              f"delta={d*100:+.2f}pp  [{sig}]")
        print(f"    Pareto front: {len(r['pareto_front'])} punktow")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "D4_consultation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")
    print("\nD4 zakonczone.")
    print("-> Jesli consultation pomaga: dokumentuj tryb jako opcja wnioskowania.")
    print("-> Jesli SZUM/ujemne: potwierdza, ze granica router osiagnal sufit.")


if __name__ == "__main__":
    main()
