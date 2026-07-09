"""
run_E4_energy.py -- E4: stos efektywnosci + energia w DZULACH (nvidia-smi).

Cel (DROGA_E_PLAN.md, sekcja E4):
  Zamienic obietnice "rewolucji energetycznej" (MAC) na liczbe nie do
  podwazenia: dzule na 10k probek, zmierzone na realnym GPU, z odjeta
  moca spoczynkowa. Jedna tabela: accuracy | MAC | J/10k | throughput.

Systemy (Fashion-MNIST -- pole gry projektu; MNIST przez --datasets):
  mono_mlp   : monolityczny MLP 784-256-128-10 (baseline z whitepapera v0.1)
  mono_s2    : backbone S2 (slim CNN 8,16) + JEDNA glowica 10-way
               (monolityczny odpowiednik przy tym samym budzecie cech)
  mars_s2    : M.A.R.S. v2 na backbone S2 (train_phased, D6b)
  mars_s2_t  : jw. + pody TERNARY (schemat B8: prog tf*mean|w|, sweep tf)
  mars_full  : M.A.R.S. v2 na pelnym CNN (referencja; acc znane z D6)

Pomiar energii -- WARIANT PROXY (GTX 1050 Ti NIE raportuje power.draw,
zweryfikowane 06.07.2026: [N/A], karta bez czujnikow INA):
  Metryka podstawowa: s/10k = sekundy GPU na 10k probek (petla inferencji
  >=20 s, batch 512, caly test w kolko, 5 powtorzen). Przy NASYCONYM GPU
  energia ~ moc_srednia * czas, a moc srednia jest zblizona miedzy systemami
  => STOSUNKI s/10k ~ stosunki energii. Nasycenie kontrolowane pomiarem
  utilization.gpu (probka/100ms, nvidia-smi -lms); raportowane per system --
  jesli systemy roznia sie utylizacja, stosunki traktowac ostroznie.
  Dodatkowo: J/10k UPPER BOUND = power.limit (75 W) * s/10k -- twarde gorne
  ograniczenie. Absolutne dzule wymagaja karty z telemetria lub watomierza
  z gniazdka (odnotowane w limitations).

Kryteria werdyktu (Z GORY):
  V1 "ternary za darmo w stosie": |d acc| <= prog szumu (std sumy, 3 seedy)
     ORAZ s/10k(ternary) <= 1.05 * s/10k(full precision).
     Uczciwa uwaga: na GPU bez sprzetu ternary NIE oczekujemy zysku czasu
     (B8: throughput 1.08x) -- zysk realizuje sie w pamieci (16x) i na
     NPU/FPGA. V1 potwierdza tylko "bez kosztu".
  V2 "efektywnosc MARS": SYGNAL+ jesli mars_s2 ma wyzsza acc niz mono_mlp
     przy s/10k <= 1.5x mono_mlp. Raport pelnego Pareto (acc, s/10k, MAC).
UCZCIWE RYZYKO (pre-rejestrowane): mono_s2 (jedna glowica na tym samym
backbone) moze osiagnac ~ta sama acc co mars_s2 taniej -- modularnosc kupuje
retencje wiedzy i skalowanie liczby zadan (Etap 2/A4), nie accuracy na 10
klasach. Jesli tak wyjdzie, raportujemy to wprost.

Tryb szybki:  python src/run_E4_energy.py --smoke   (1 seed, 8 epok, 8 s pomiaru)
Pelny:        python src/run_E4_energy.py           (3 seedy acc, 5x pomiar energii)
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import train_phased, evaluate
from mars_v2_cnn import MarsV2CNNSystem
from mars_v2_slim import MarsV2SlimSystem, SlimCNNBackbone
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BB_H, EMB, POD_H = 128, 32, 24
S2 = dict(channels=(8, 16), downsample="maxpool", depthwise=False)
TERNARY_TFS = [0.5, 0.7]   # sweep progu (B8: 0.5 optymalny)
# Dwa batche: maly (edge/latency) i duzy (throughput; amortyzuje launche
# kerneli -- fair wobec convow). Werdykty licza sie na LEPSZYM punkcie
# kazdego systemu (kazdy w swoim optimum operacyjnym).
BATCHES_INFER = [512, 4096]


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


# ------------------------------------------------------------ modele mono
class MonoMLP(nn.Module):
    """Monolityczny MLP 784-256-128-10 (baseline whitepaper v0.1)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10))
    def forward(self, x):
        return self.net(x)
    def mac_per_sample(self):
        return 784 * 256 + 256 * 128 + 128 * 10   # 234,752


class MonoS2(nn.Module):
    """Backbone S2 + jedna glowica 10-way (monolityczny odpowiednik)."""
    def __init__(self):
        super().__init__()
        self.backbone = SlimCNNBackbone(backbone_hidden=BB_H, **S2)
        self.head = nn.Linear(BB_H, 10)
    def forward(self, x):
        return self.head(self.backbone(x))
    def mac_per_sample(self):
        c1, c2 = S2["channels"]
        bb = 1 * c1 * 9 * 28 * 28 + c1 * c2 * 9 * 14 * 14 + 49 * c2 * BB_H
        return bb + BB_H * 10


def train_mono(model, Xtr, ytr, epochs, device, lr=0.001, batch=512):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            loss = crit(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def acc_mono(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()


# ------------------------------------------------------------ ternary (B8)
def ternary_quantize_tensor(w, tf):
    """Kwantyzacja do [-alpha, 0, +alpha]; prog = tf * mean|w| (jak B8)."""
    threshold = tf * w.abs().mean()
    ternary = torch.zeros_like(w)
    pos, neg = w > threshold, w < -threshold
    ternary[pos], ternary[neg] = 1.0, -1.0
    n = pos.sum() + neg.sum()
    alpha = (w[pos].sum() - w[neg].sum()) / n if n > 0 else torch.tensor(0.0)
    return ternary * alpha.abs(), (ternary == 0).float().mean().item()


def ternarize_pods(model, tf):
    """Kopia modelu z ternary pod_W1/pod_W2 (per pod; biasy full precision)."""
    m = copy.deepcopy(model)
    sparsities = []
    with torch.no_grad():
        for p in range(m.n_pods):
            for W in (m.pod_W1, m.pod_W2):
                q, sp = ternary_quantize_tensor(W.data[p], tf)
                W.data[p] = q
                sparsities.append(sp)
    return m, sum(sparsities) / len(sparsities)


# ------------------------------------------ pomiar czasu GPU + utylizacji
class UtilSampler:
    """Czyta utilization.gpu z nvidia-smi w tle (probka co ~100 ms).
    (power.draw na GTX 1050 Ti = [N/A]; utylizacja sluzy jako kontrola
    nasycenia dla proxy energetycznego s/10k.)"""
    def __init__(self):
        self.samples = []
        self._proc = None
        self._thread = None

    def start(self):
        self.samples = []
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        def _reader():
            for line in self._proc.stdout:
                try:
                    self.samples.append((time.perf_counter(),
                                         float(line.strip())))
                except ValueError:
                    pass
        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)

    def mean_util(self, t0, t1):
        vals = [u for t, u in self.samples if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else float("nan")


def gpu_power_limit_w():
    """Odczyt power.limit (gorne ograniczenie mocy karty) do bounda J/10k."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def measure_inference(sampler, forward_fn, Xte, seconds, p_limit, batch):
    """Petla inferencji >= seconds; zwraca dict metryk czasowych."""
    n_test = len(Xte)
    # warmup
    with torch.no_grad():
        for s in range(0, n_test, batch):
            forward_fn(Xte[s:s + batch])
    torch.cuda.synchronize()

    n_samples = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - t0 < seconds:
            for s in range(0, n_test, batch):
                forward_fn(Xte[s:s + batch])
                n_samples += min(batch, n_test - s)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    dur = t1 - t0
    s_per_10k = dur / n_samples * 10000
    return {
        "duration_s": round(dur, 2),
        "n_samples": n_samples,
        "throughput_sps": round(n_samples / dur),
        "util_pct": round(sampler.mean_util(t0, t1), 1),
        "s_per_10k": round(s_per_10k, 4),
        "j_per_10k_upper": round(p_limit * s_per_10k, 2),
    }


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["Fashion-MNIST"])
    args = ap.parse_args()

    n_seeds = 1 if args.smoke else 3
    epochs = 8 if args.smoke else 30
    e_seconds = 8 if args.smoke else 20
    e_repeats = 2 if args.smoke else 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("E4 wymaga GPU (pomiar nvidia-smi).")

    print("=" * 72)
    print(f"E4 -- stos efektywnosci + energia  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Device: {device} | seeds_acc={n_seeds} | epochs={epochs} | "
          f"pomiar {e_seconds}s x{e_repeats} | ternary tf={TERNARY_TFS}")
    print("=" * 72)

    t_start = time.perf_counter()
    out = {"experiment": "E4_energy", "n_seeds_acc": n_seeds, "epochs": epochs,
           "e_seconds": e_seconds, "e_repeats": e_repeats,
           "ternary_tfs": TERNARY_TFS, "datasets": {}}

    sampler = UtilSampler()
    sampler.start()
    time.sleep(1.0)
    p_limit = gpu_power_limit_w()
    print(f"power.limit = {p_limit:.1f} W (bound do J/10k; power.draw = N/A "
          f"na tej karcie -- metryka podstawowa: s/10k przy nasyceniu)")

    for ds_name in args.datasets:
        Xtr, ytr, Xte, yte = load_dataset(
            "MNIST" if ds_name == "MNIST" else "Fashion-MNIST", device)
        res = {"p_limit_w": round(p_limit, 2), "systems": {}}

        # ---------- trening + accuracy (multi-seed) ----------
        models0 = {}   # modele seed 0 do pomiaru energii
        accs = {k: [] for k in
                ("mono_mlp", "mono_s2", "mars_s2")}
        tern_accs = {tf: [] for tf in TERNARY_TFS}
        tern_sparsity = {tf: [] for tf in TERNARY_TFS}

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            mm = MonoMLP().to(device)
            train_mono(mm, Xtr, ytr, epochs, device)
            accs["mono_mlp"].append(acc_mono(mm, Xte, yte))

            torch.manual_seed(seed)
            ms = MonoS2().to(device)
            train_mono(ms, Xtr, ytr, epochs, device)
            accs["mono_s2"].append(acc_mono(ms, Xte, yte))

            torch.manual_seed(seed)
            mr = MarsV2SlimSystem(backbone_hidden=BB_H, emb_dim=EMB,
                                  pod_hidden=POD_H, **S2).to(device)
            train_phased(mr, Xtr, ytr, epochs=epochs, device=device)
            _, s_acc, _ = evaluate(mr, Xte, yte)
            accs["mars_s2"].append(s_acc)

            for tf in TERNARY_TFS:
                mt, sp = ternarize_pods(mr, tf)
                _, t_acc, _ = evaluate(mt, Xte, yte)
                tern_accs[tf].append(t_acc)
                tern_sparsity[tf].append(sp)
                if seed == 0:
                    models0[f"mars_s2_t{tf}"] = mt

            if seed == 0:
                models0.update({"mono_mlp": mm, "mono_s2": ms, "mars_s2": mr})
            print(f"[{ds_name}] seed {seed}: mono_mlp={accs['mono_mlp'][-1]*100:.2f}% "
                  f"mono_s2={accs['mono_s2'][-1]*100:.2f}% "
                  f"mars_s2={accs['mars_s2'][-1]*100:.2f}% "
                  + " ".join(f"t{tf}={tern_accs[tf][-1]*100:.2f}%"
                             for tf in TERNARY_TFS))

        # referencja: pelny CNN (1 seed -- acc znane z D6)
        torch.manual_seed(0)
        mf = MarsV2CNNSystem(backbone_hidden=BB_H, emb_dim=EMB,
                             pod_hidden=POD_H, channels=(32, 64)).to(device)
        train_phased(mf, Xtr, ytr, epochs=epochs, device=device)
        _, f_acc, _ = evaluate(mf, Xte, yte)
        models0["mars_full"] = mf

        # najlepszy prog ternary (acc)
        best_tf = max(TERNARY_TFS, key=lambda tf: sum(tern_accs[tf]))

        # ---------- pomiar energii (modele seed 0) ----------
        energy_order = ["mono_mlp", "mono_s2", "mars_s2",
                        f"mars_s2_t{best_tf}", "mars_full"]
        fwd = {
            "mono_mlp": models0["mono_mlp"].forward,
            "mono_s2": models0["mono_s2"].forward,
            "mars_s2": models0["mars_s2"].forward,
            f"mars_s2_t{best_tf}": models0[f"mars_s2_t{best_tf}"].forward,
            "mars_full": models0["mars_full"].forward,
        }
        macs = {
            "mono_mlp": models0["mono_mlp"].mac_per_sample(),
            "mono_s2": models0["mono_s2"].mac_per_sample(),
            "mars_s2": models0["mars_s2"].mac_per_sample_top1()["total_top1"],
            f"mars_s2_t{best_tf}":
                models0["mars_s2"].mac_per_sample_top1()["total_top1"],
            "mars_full": models0["mars_full"].mac_per_sample_top1()["total_top1"],
        }
        for name in energy_order:
            models0_name = models0[name]
            models0_name.eval()
            entry = {"mac": macs[name], "by_batch": {}}
            for b in BATCHES_INFER:
                runs = [measure_inference(sampler, fwd[name], Xte, e_seconds,
                                          p_limit, b)
                        for _ in range(e_repeats)]
                entry["by_batch"][str(b)] = {
                    "energy_runs": runs,
                    "s_per_10k": stats([r["s_per_10k"] for r in runs]),
                    "j_per_10k_upper": stats([r["j_per_10k_upper"]
                                              for r in runs]),
                    "util_pct": stats([r["util_pct"] for r in runs]),
                    "throughput_sps": stats([r["throughput_sps"]
                                             for r in runs]),
                }
            # najlepszy punkt operacyjny systemu (min s/10k po batchach)
            best_b = min(BATCHES_INFER, key=lambda b:
                         entry["by_batch"][str(b)]["s_per_10k"]["mean"])
            entry["best_batch"] = best_b
            entry["s_per_10k"] = entry["by_batch"][str(best_b)]["s_per_10k"]
            entry["j_per_10k_upper"] = \
                entry["by_batch"][str(best_b)]["j_per_10k_upper"]
            entry["util_pct"] = entry["by_batch"][str(best_b)]["util_pct"]
            entry["throughput_sps"] = \
                entry["by_batch"][str(best_b)]["throughput_sps"]
            res["systems"][name] = entry
            print(f"[{ds_name}] czas {name:16s}: " + " | ".join(
                f"b{b}: {entry['by_batch'][str(b)]['s_per_10k']['mean']:.4f}"
                f" s/10k (util "
                f"{entry['by_batch'][str(b)]['util_pct']['mean']:.0f}%)"
                for b in BATCHES_INFER))

        # ---------- accuracy do rekordu ----------
        res["accuracy"] = {k: stats(v) for k, v in accs.items()}
        res["accuracy"][f"mars_s2_t{best_tf}"] = stats(tern_accs[best_tf])
        res["accuracy"]["mars_full_seed0"] = round(f_acc, 4)
        res["ternary"] = {str(tf): {"acc": stats(tern_accs[tf]),
                                    "sparsity": stats(tern_sparsity[tf])}
                          for tf in TERNARY_TFS}
        res["best_tf"] = best_tf

        # ---------- werdykty ----------
        verdicts = {}
        if not args.smoke:
            # V1: ternary za darmo w stosie
            d_acc = [(t - f) * 100 for t, f
                     in zip(tern_accs[best_tf], accs["mars_s2"])]
            d_stats = stats(d_acc)
            noise = (stats([a * 100 for a in accs["mars_s2"]])["std"]
                     + d_stats["std"])
            t_ratio = (res["systems"][f"mars_s2_t{best_tf}"]["s_per_10k"]["mean"]
                       / max(res["systems"]["mars_s2"]["s_per_10k"]["mean"], 1e-9))
            v1_ok = abs(d_stats["mean"]) <= noise and t_ratio <= 1.05
            verdicts["V1_ternary_free"] = {
                "delta_acc_pp": d_stats, "noise_pp": round(noise, 4),
                "time_ratio": round(t_ratio, 3),
                "verdict": "POTWIERDZONE" if v1_ok else "NIEPOTWIERDZONE"}
            # V2: efektywnosc MARS vs mono_mlp
            acc_gain = (stats(accs["mars_s2"])["mean"]
                        - stats(accs["mono_mlp"])["mean"]) * 100
            t_ratio2 = (res["systems"]["mars_s2"]["s_per_10k"]["mean"]
                        / max(res["systems"]["mono_mlp"]["s_per_10k"]["mean"], 1e-9))
            v2_ok = acc_gain > 0 and t_ratio2 <= 1.5
            verdicts["V2_mars_vs_mono_mlp"] = {
                "acc_gain_pp": round(acc_gain, 2),
                "time_ratio": round(t_ratio2, 3),
                "verdict": "SYGNAL+" if v2_ok else
                           ("SZUM/NEGATYWNY -- patrz Pareto")}
        res["verdicts"] = verdicts

        # ---------- raport ----------
        print(f"\n--- {ds_name}: PARETO (acc | MAC | s/10k | util) ---")
        for name in energy_order:
            if name == "mars_full":
                a = f"{f_acc*100:.2f}% (seed0; D6: 91.99%)"
            elif name in res["accuracy"]:
                st = res["accuracy"][name]
                a = f"{st['mean']*100:.2f}+/-{st['std']*100:.2f}%"
            else:
                a = "?"
            sysm = res["systems"][name]
            print(f"  {name:16s}: acc {a:26s} | MAC {macs[name]:>9,} | "
                  f"s/10k {sysm['s_per_10k']['mean']:.4f} "
                  f"(b{sysm['best_batch']}) | "
                  f"util {sysm['util_pct']['mean']:.0f}%")
        for k, v in verdicts.items():
            print(f"  {k}: {v['verdict']}  ({json.dumps({kk: vv for kk, vv in v.items() if kk != 'verdict'}, default=str)[:120]})")
        print()

        out["datasets"][ds_name] = res

    sampler.stop()
    out["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = "E4_energy_smoke.json" if args.smoke else "E4_energy.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Czas: {out['elapsed_s']}s | zapisano: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
