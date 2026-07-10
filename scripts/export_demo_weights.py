"""export_demo_weights.py -- eksport wag MARS-CL (H1b k16, 1 seed, Fashion)
do demo przegladarkowego.

NIE modyfikuje src/ (kod zamrozony) -- tylko importuje i trenuje 1 seed
konfiguracji zwycieskiej z H1b (k16: dream diag, stats_k=16, epochs_proj=15,
l2sp=0, 15 epok/zadanie, LR=0.001, seed 0), po czym zrzuca:
  - wagi backbone'u (losowy zamrozony) i snapshoty projekcji po kazdym zadaniu,
  - pody per klasa, kotwice slowne (GloVe 50d), macierz R class-IL,
  - 12 obrazkow testowych na klase (surowe uint8, base64),
  - wektor parity (cechy/embedding obrazka nr 0) do samotestu JS vs PyTorch.

Wyjscie: demo/mars_cl_demo/mars_demo_weights.json (~4 MB)

Uzycie:   python scripts/export_demo_weights.py           # pelny (15 epok)
          python scripts/export_demo_weights.py --smoke   # szybki test (2 epoki)
Wymaga:   data/glove.6B.50d.txt (scripts/download_glove.py)
"""
import argparse
import base64
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cl_common import make_task_data, eval_protocols, cl_metrics  # noqa: E402
from mars_cl_f3 import MarsCLSemanticF3                           # noqa: E402
from mars_cl_semantic import load_word_vectors                    # noqa: E402
from run_D1_mars_v2_baseline import load_dataset, DATA_DIR        # noqa: E402

SEED, LR = 0, 0.001
MEAN, STD = 0.2860, 0.3530          # normalizacja Fashion (jak load_dataset)
IMGS_PER_CLASS = 12
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "demo",
                        "mars_cl_demo", "mars_demo_weights.json")
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def T(t, nd=6):
    """Tensor -> {shape, plaska lista} (zaokraglenie tnie rozmiar JSON-a)."""
    return {"shape": list(t.shape),
            "data": [round(float(v), nd)
                     for v in t.detach().cpu().reshape(-1).tolist()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    epochs = 2 if args.smoke else 15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    print(f"Export demo: device={device}, seed={SEED}, epok={epochs} "
          f"({'SMOKE - liczby beda gorsze' if args.smoke else 'FULL'})")
    wv = load_word_vectors("Fashion-MNIST", device=device)
    Xtr, ytr, Xte, yte = load_dataset("Fashion-MNIST", device)
    task_data = make_task_data(Xtr, ytr, Xte, yte)

    # --- trening: dokladnie konfiguracja k16 z run_H1b (COMMON + VARIANTS) ---
    torch.manual_seed(SEED)
    m = MarsCLSemanticF3(wv, dream_model="diag", stats_k=16,
                         epochs_proj=epochs, l2sp=0.0).to(device)
    m.init_representation(task_data, epochs=epochs, lr=LR, device=device)

    snapshots, seen, R_c = [], [], []
    for t, td in enumerate(task_data):
        m.learn_task(td, epochs=epochs, lr=LR, device=device)
        seen = seen + td["classes"]
        row_c, _ = eval_protocols(m.forward, task_data, t, seen)
        R_c.append(row_c)
        snapshots.append({
            "task": t, "classes": list(td["classes"]), "seen": list(seen),
            "proj_w": T(m.proj.weight), "proj_b": T(m.proj.bias),
            "pods": {str(c): {k: T(v) for k, v in m.pods[c].items()}
                     for c in td["classes"]},
            "row_class_il": row_c,
        })
        print(f"  zadanie {t}: klasy {td['classes']}, "
              f"class-IL po zadaniu: {row_c}")
    metrics = cl_metrics(R_c)
    print(f"Final ACC class-IL: {metrics['ACC']*100:.2f}% "
          f"(pelny run H1b k16: 77.57 +/- 1.02%)")

    # --- backbone (Sequential: Conv,BN,ReLU,Pool,Conv,BN,ReLU,Pool) ---
    bb = m.backbone
    conv1, bn1, conv2, bn2 = bb.conv[0], bb.conv[1], bb.conv[4], bb.conv[5]
    backbone = {
        "conv1_w": T(conv1.weight), "conv1_b": T(conv1.bias),
        "bn1": {"g": T(bn1.weight), "b": T(bn1.bias),
                "m": T(bn1.running_mean), "v": T(bn1.running_var)},
        "conv2_w": T(conv2.weight), "conv2_b": T(conv2.bias),
        "bn2": {"g": T(bn2.weight), "b": T(bn2.bias),
                "m": T(bn2.running_mean), "v": T(bn2.running_var)},
        "fc_w": T(bb.proj[0].weight), "fc_b": T(bb.proj[0].bias),
    }

    # --- obrazki testowe: surowe uint8 (base64), rowna proba per klasa ---
    raw = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False,
                                            download=True)
    per_class, pixels, labels = {c: 0 for c in range(10)}, bytearray(), []
    for img, lab in raw:
        if per_class[lab] >= IMGS_PER_CLASS:
            continue
        per_class[lab] += 1
        pixels += img.tobytes()          # 784 bajtow, wiersz po wierszu
        labels.append(int(lab))
        if all(v >= IMGS_PER_CLASS for v in per_class.values()):
            break

    # --- parity: forward obrazka nr 0 po stronie PyTorch ---
    px = torch.tensor(list(pixels[:784]), dtype=torch.float32,
                      device=device) / 255.0
    x = ((px - MEAN) / STD).unsqueeze(0)
    with torch.no_grad():
        feats = bb(x)
        emb = F.normalize(m.proj(feats), dim=1)
        pred = int(m.forward(x).argmax(dim=1).item())
    parity = {"image_index": 0, "label": labels[0],
              "feats": T(feats[0]), "emb": T(emb[0]), "pred": pred}

    out = {
        "meta": {"dataset": "Fashion-MNIST", "variant": "H1b k16",
                 "seed": SEED, "epochs_per_task": epochs, "lr": LR,
                 "smoke": args.smoke, "norm_mean": MEAN, "norm_std": STD,
                 "final_acc_class_il": metrics["ACC"],
                 "forgetting": metrics["forgetting"],
                 "reference": "results/H1b_dream_fidelity.json"},
        "class_names": CLASS_NAMES,
        "anchors": {str(c): T(m.word_vecs[c]) for c in range(10)},
        "backbone": backbone,
        "snapshots": snapshots,
        "R_class_il": R_c,
        "test_images": {"n": len(labels), "labels": labels,
                        "pixels_b64": base64.b64encode(bytes(pixels)).decode()},
        "parity": parity,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Zapisano {os.path.abspath(OUT_PATH)} ({mb:.1f} MB) "
          f"w {time.perf_counter()-t0:.0f}s")
    print("Demo: cd demo/mars_cl_demo && python -m http.server 8000 "
          "-> http://localhost:8000")


if __name__ == "__main__":
    main()
