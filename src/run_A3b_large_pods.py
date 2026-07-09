"""
run_A3b_large_pods.py — Droga A, krok A3b: duże pody (throughput crossover).

Z Etapu B wiemy, że przewaga czasowa M.A.R.S. nad monolitem pojawia się
dopiero przy dużych podach (hidden>=2048). A3 zmierzyło, że na małych podach
(hidden=24) M.A.R.S. jest 0.62× wolniejszy.

Ten skrypt odpowiada na pytanie: przy jakim hidden pełny system specjalistów
(ProtoRouter + FastPods) bije monolit na WSZYSTKICH frontach (accuracy,
MAC, throughput)?

Testujemy skalę: hidden ∈ {64, 256, 512, 1024, 2048}.
Dla każdego hidden: monolit (proporcjonalny) vs M.A.R.S. specjaliści.

Uczciwa uwaga: MNIST z hidden=2048 jest absurdalnie over-parameterized.
Accuracy obu systemów będzie ~98%. To jest test THROUGHPUT, nie accuracy —
chcemy zmierzyć, od jakiego rozmiaru poda routing się opłaca czasowo.

Uruchom:
    .venv\\Scripts\\python.exe src\\run_A3b_large_pods.py
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from routers_v2 import ProtoRouter
from mars_fast_forward import FastPods

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_PODS = 10
N_IN = 784
N_OUT = 10

# Rozmiary podów do przetestowania
HIDDEN_SIZES = [64, 256, 512, 1024, 2048]


def load_mnist_tensors(device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    Xtr = torch.stack([train[i][0].view(-1) for i in range(len(train))]).to(device)
    ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(device)
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))]).to(device)
    yte = torch.tensor([test[i][1] for i in range(len(test))]).to(device)
    return Xtr, ytr, Xte, yte


def bench_throughput(fn, x, device, n_warmup=15, n_runs=80):
    """Mierzy throughput (samples/s) z rozgrzewką i synchronizacją CUDA."""
    for _ in range(n_warmup):
        fn(x)
    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_runs):
            fn(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    return x.shape[0] / ((elapsed_ms / n_runs) / 1000)


def build_monolit(hidden, device):
    """Monolit proporcjonalny do rozmiaru poda — uczciwe porównanie.

    Żeby porównanie było uczciwe, monolit musi mieć PORÓWNYWALNĄ pojemność
    do M.A.R.S. aktywowanego poda. Jeden pod specjalisty to:
    784 → hidden → 10.
    Monolit: 784 → hidden → 10 (ten sam kształt, pełna aktywacja).

    To jest NAJLEPSZE porównanie: identyczna pojemność, identyczne wejście/wyjście,
    jedyna różnica to routing+usypianie vs pełna aktywacja.
    """
    return nn.Sequential(
        nn.Linear(N_IN, hidden), nn.ReLU(),
        nn.Linear(hidden, N_OUT)
    ).to(device)


def monolit_mac(hidden):
    """MAC monolitu: 784*hidden + hidden*10."""
    return N_IN * hidden + hidden * N_OUT


def train_model(model, Xtr, ytr, epochs=10, lr=0.001, bs=512):
    """Szybki trening modelu do pomiaru throughput (accuracy jest drugorzędna)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for s in range(0, len(Xtr), bs):
            idx = perm[s:s+bs]
            loss = crit(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def train_router(router, Xtr, ytr, epochs=15, lr=0.003):
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for s in range(0, len(Xtr), 512):
            idx = perm[s:s+512]
            loss = crit(router(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def train_specialists_into_fastpods(Xtr, ytr, device, hidden, own_ratio=0.7, epochs=10):
    """Trenuje specjalistów osobno, przenosi wagi do FastPods."""
    fast = FastPods(N_PODS, N_IN, hidden, N_OUT).to(device)
    crit = nn.CrossEntropyLoss()
    for c in range(N_PODS):
        pod = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, N_OUT)
        ).to(device)
        opt = torch.optim.Adam(pod.parameters(), lr=0.001)
        # specjalizacja: 70% swoich danych, 30% cudzych
        mask = ytr == c
        own_X, own_y = Xtr[mask], ytr[mask]
        n_other = int(len(own_X) * (1 - own_ratio) / own_ratio)
        X_pod = torch.cat([own_X, Xtr[~mask][:n_other]])
        y_pod = torch.cat([own_y, ytr[~mask][:n_other]])
        for _ in range(epochs):
            perm = torch.randperm(len(X_pod), device=device)
            for s in range(0, len(X_pod), 256):
                idx = perm[s:s+256]
                loss = crit(pod(X_pod[idx]), y_pod[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        # przenieś wagi do FastPods
        with torch.no_grad():
            fast.W1.data[c] = pod[0].weight.data.T
            fast.b1.data[c] = pod[0].bias.data
            fast.W2.data[c] = pod[2].weight.data.T
            fast.b2.data[c] = pod[2].bias.data
    return fast


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 72)
    print("DROGA A -- krok A3b: duze pody (throughput crossover)")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == 'cuda' else "")
    print("=" * 72)

    print("\\nŁadowanie MNIST...")
    Xtr, ytr, Xte, yte = load_mnist_tensors(device)

    # Router — ten sam dla wszystkich rozmiarów (ProtoRouter 16D, najlepszy z A1)
    print("Trening routera ProtoRouter 16D...")
    router = ProtoRouter(N_IN, N_PODS, enc_hidden=32, emb=16).to(device)
    train_router(router, Xtr, ytr)
    router.eval()
    with torch.no_grad():
        rout_acc = (router.route(Xte) == yte).float().mean().item()
    router_mac = router.mac_per_sample()
    print(f"  routing acc: {rout_acc*100:.1f}%, router MAC: {router_mac:,}")

    rows = []

    print(f"\\n{'hidden':>8}  {'monolit acc':>11} {'M.A.R.S. acc':>12} "
          f"{'mono t/put':>12} {'mars t/put':>12} {'mars/mono':>10} "
          f"{'MAC oszcz':>10}")
    print("-" * 90)

    for hidden in HIDDEN_SIZES:
        # --- Monolit ---
        mono = build_monolit(hidden, device)
        train_model(mono, Xtr, ytr, epochs=10)
        mono.eval()
        with torch.no_grad():
            mono_acc = (mono(Xte).argmax(1) == yte).float().mean().item()
        mono_tput = bench_throughput(lambda x: mono(x), Xte, device)
        mono_mac = monolit_mac(hidden)

        # --- M.A.R.S. specjaliści ---
        fast = train_specialists_into_fastpods(Xtr, ytr, device, hidden)
        fast.eval()

        with torch.no_grad():
            ids = router.route(Xte)
            mars_out = fast.forward_auto(Xte, ids)
            mars_acc = (mars_out.argmax(1) == yte).float().mean().item()

        mars_tput = bench_throughput(
            lambda x: fast.forward_auto(x, router.route(x)), Xte, device)
        pod_mac = N_IN * hidden + hidden * N_OUT
        mars_mac = router_mac + pod_mac
        mac_saving = (1 - mars_mac / mono_mac) * 100

        ratio = mars_tput / mono_tput
        marker = "  <<<  MARS SZYBSZY" if ratio > 1.0 else ""
        print(f"{hidden:>8}  {mono_acc*100:>10.1f}% {mars_acc*100:>11.1f}% "
              f"{mono_tput:>12,.0f} {mars_tput:>12,.0f} {ratio:>9.2f}× "
              f"{mac_saving:>9.1f}%{marker}")

        rows.append({
            "hidden": hidden,
            "monolit_acc": round(mono_acc, 4),
            "mars_acc": round(mars_acc, 4),
            "monolit_tput": round(mono_tput),
            "mars_tput": round(mars_tput),
            "mars_vs_mono": round(ratio, 3),
            "monolit_mac": mono_mac,
            "mars_mac": mars_mac,
            "mac_saving_pct": round(mac_saving, 1),
        })

        # Wyczyść pamięć GPU
        del mono, fast
        torch.cuda.empty_cache() if device == 'cuda' else None

    # --- Wniosek ---
    print("\\n--- WNIOSEK ---")
    crossover = [r for r in rows if r["mars_vs_mono"] >= 1.0]
    if crossover:
        first = crossover[0]
        print(f"Crossover throughput: hidden={first['hidden']} "
              f"(M.A.R.S. {first['mars_vs_mono']:.2f}× vs monolit)")
        print(f"Przy hidden>={first['hidden']}: M.A.R.S. BIJE monolit na throughput.")
        for r in crossover:
            print(f"  hidden={r['hidden']}: {r['mars_vs_mono']:.2f}× szybszy, "
                  f"MAC oszczędność {r['mac_saving_pct']:.1f}%, "
                  f"accuracy {r['mars_acc']*100:.1f}%")
    else:
        best = max(rows, key=lambda r: r["mars_vs_mono"])
        print(f"M.A.R.S. nie przebił monolitu na throughput (najlepszy: "
              f"hidden={best['hidden']}, {best['mars_vs_mono']:.2f}×)")
        print("Na GTX 1050 Ti próg może być wyższy niż testowane rozmiary.")

    # MAC saving zawsze
    print(f"\\nMAC oszczednosc: {rows[0]['mac_saving_pct']:.1f}% "
          f"(h={rows[0]['hidden']}) -> {rows[-1]['mac_saving_pct']:.1f}% "
          f"(h={rows[-1]['hidden']})")
    print("MAC saving rośnie z hidden (router stały, pod proporcjonalny do monolitu).")

    # Zapis JSON
    results = {
        "device": device,
        "router_mac": router_mac,
        "routing_acc": round(rout_acc, 4),
        "measurements": rows,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "A3b_large_pods.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\\nWynik zapisany: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
