"""
run_D5_distillation.py -- Droga D, D5: Distillation pod -> router.

Cel: czy pody (po wytrenowaniu) moga nauczyc router czegos, czego sam nie
nauczyl z CE na etykietach? Router i pod widza TE SAME cechy backbone -- pod
nie ma dodatkowej informacji wejsciowej. Otwarta kwestia empiryczna.

Trzy warianty "karmienia" routera soft targetami z podow:
  D5a -- soft:    teacher = top-1 pod wskazany przez BIEZACY router (klasyczny KD;
                  dynamiczny teacher -- zmienia sie wraz z fine-tuningiem routera)
  D5b -- oracle:  teacher = pod dla PRAWDZIWEJ klasy y (gorna granica; nie realny
                  tryb inferencji -- traktowac jak ORACLE-sufit dla distillation)
  D5c -- self:    teacher = top-1 pod wskazany przez ZAMROZONA kopie routera
                  sprzed distillation (bez ORACLE, bez feedbacku z biezacego)

Mechanizm (wspolny dla wszystkich wariantow):
  Po standardowym treningu phased (identycznym jak D1c baseline):
    1. Zamroz pody (sa nauczycielami -- nie uczymy ich ponownie).
    2. Odblokuj backbone + routing_head + protos (studenci).
    3. Loss = CE(route_logits, y) + lambda * T^2 * KL(log_p_student/T, p_teacher/T)
       gdzie student = route_logits routera, teacher = logity wybranego poda.

Protokol:
  - 5 seedow (wieloseedowa walidacja jak D1c).
  - Dla kazdego seeda: base phased -> 3x deepcopy -> 3x distillation fine-tuning.
  - Baseline: D1c/v2a (system: MNIST 98.36+/-0.05pp, Fashion 89.50+/-0.11pp).

Uruchom:
    .venv\\Scripts\\python.exe src\\run_D5_distillation.py
"""
import copy, json, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from mars_v2 import MarsV2System, train_phased, evaluate, N_IN, N_CLASSES
from run_D1_mars_v2_baseline import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_SEEDS          = 5
EPOCHS           = 30           # base phased training (jak D1c)
BB_H, EMB, POD_H = 384, 32, 24  # zrownane parametrycznie z v1 (jak D1c)

DISTILL_EPOCHS   = 10
DISTILL_LR       = 1e-4
T_TEMP           = 2.0
LAMBDA_KD        = 0.5


def distill_router_from_pods(
        model: MarsV2System, Xtr, ytr,
        variant: str,               # "soft" | "oracle" | "self"
        T: float        = T_TEMP,
        lam: float      = LAMBDA_KD,
        distill_epochs: int = DISTILL_EPOCHS,
        lr: float       = DISTILL_LR,
        batch: int      = 512,
        device: str     = "cpu") -> MarsV2System:
    """
    Fine-tunes router (backbone + routing_head + protos) uzywajac soft targetow
    z zamrozonych podow.

    Loss na probke:
        L = CE(route_logits, y) + lam * T^2 * KLDiv(log_p_student/T, p_teacher/T)

    gdzie:
        p_student = softmax(route_logits / T)    -- routing distribution studenta
        p_teacher = softmax(pod_out / T)         -- soft targets z wybranego poda

    Wybor poda-nauczyciela zalezy od wariantu:
        "soft"   -- biezacy router (argmax route_logits na biezacych cechach)
        "oracle" -- prawdziwa klasa y (idealne przypisanie, gorna granica)
        "self"   -- snapshot routera sprzed distillation (bez feedbacku)
    """
    model.train()

    # Zamroz pody (nauczyciele -- nie modyfikujemy ich)
    for p in [model.pod_W1, model.pod_b1, model.pod_W2, model.pod_b2]:
        p.requires_grad = False
    # Odblokuj backbone + routing head
    for p in (list(model.backbone.parameters())
              + list(model.routing_head.parameters())
              + [model.protos]):
        p.requires_grad = True

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    # Snapshot routera dla wariantu "self" (zamrozony przed fine-tuningiem)
    if variant == "self":
        frozen_rh_w    = model.routing_head.weight.detach().clone()
        frozen_rh_b    = model.routing_head.bias.detach().clone()
        frozen_protos  = model.protos.detach().clone()

    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()

    for _ in range(distill_epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            x, y = Xtr[idx], ytr[idx]

            # Cechy backbone -- gradient plynie przez backbone (student)
            feats = model.backbone(x)

            # Wybierz pod-nauczyciela i oblicz jego soft targets (bez gradientu)
            with torch.no_grad():
                fd = feats.detach()   # detached do obliczen nauczyciela

                if variant == "oracle":
                    teacher_ids = y
                elif variant == "self":
                    # Routing zamrozonym snapshotem routera
                    emb   = fd @ frozen_rh_w.T + frozen_rh_b
                    dists = torch.cdist(
                        emb.unsqueeze(0), frozen_protos.unsqueeze(0)).squeeze(0)
                    teacher_ids = (-dists).argmax(dim=1)
                else:   # "soft": biezacy router dynamicznie wybiera nauczyciela
                    teacher_ids = model.route(fd)

                teacher_logits = model.pod_forward(fd, teacher_ids)
                teacher_probs  = torch.softmax(teacher_logits / T, dim=1)

            # Student: route_logits -- gradient przez routing_head + backbone
            route_logits      = model.route_logits(feats)
            student_log_probs = F.log_softmax(route_logits / T, dim=1)

            loss_ce = ce_loss(route_logits, y)
            loss_kd = kl_loss(student_log_probs, teacher_probs) * (T ** 2)
            loss    = loss_ce + lam * loss_kd

            opt.zero_grad()
            loss.backward()
            opt.step()

    # Odblokuj wszystko (dla ewaluacji i ewentualnych dalszych krokow)
    for p in model.parameters():
        p.requires_grad = True
    return model


# ======================================================================= utils
def stats(vals):
    """mean / std (Bessel n-1) / min / max."""
    n    = len(vals)
    mean = sum(vals) / n
    var  = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    std  = math.sqrt(var)
    return {"mean": round(mean, 4), "std": round(std, 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


# ===================================================================== per-seed
def run_one_seed(Xtr, ytr, Xte, yte, seed, device):
    """
    Dla jednego seeda:
      1. Trenuj base phased (identyczny jak D1c/v2a).
      2. Dla kazdego wariantu: deepcopy base -> distillation -> evaluate.
    """
    torch.manual_seed(seed)
    base = MarsV2System(N_IN, BB_H, N_CLASSES, EMB, POD_H, N_CLASSES).to(device)
    train_phased(base, Xtr, ytr, epochs=EPOCHS, device=device)
    r0, s0, o0 = evaluate(base, Xte, yte)

    result = {"base": {"routing": r0, "system": s0, "oracle": o0}}

    for variant in ("soft", "oracle", "self"):
        m = copy.deepcopy(base)
        distill_router_from_pods(m, Xtr, ytr, variant=variant, device=device)
        r, s, o = evaluate(m, Xte, yte)
        result[f"d5_{variant}"] = {"routing": r, "system": s, "oracle": o}

    return result


# ==================================================================== dataset
def run_dataset(ds_name, device):
    print(f"\n{'='*72}\nDataset: {ds_name}  ({N_SEEDS} seeds)\n{'='*72}")
    Xtr, ytr, Xte, yte = load_dataset(ds_name, device)

    model_keys = ["base", "d5_soft", "d5_oracle", "d5_self"]
    per_seed   = []

    print(f"\n{'seed':>4} | {'base sys':>9} {'soft sys':>9} "
          f"{'orac sys':>9} {'self sys':>9}")
    print("-" * 58)

    for seed in range(N_SEEDS):
        m = run_one_seed(Xtr, ytr, Xte, yte, seed, device)
        per_seed.append(m)
        print(f"{seed:>4} | {m['base']['system']*100:>8.2f}% "
              f"{m['d5_soft']['system']*100:>8.2f}% "
              f"{m['d5_oracle']['system']*100:>8.2f}% "
              f"{m['d5_self']['system']*100:>8.2f}%")

    # Agregacja statystyk
    agg = {}
    for key in model_keys:
        agg[key] = {
            metric: stats([per_seed[s][key][metric] for s in range(N_SEEDS)])
            for metric in ["routing", "system", "oracle"]
        }

    # Raport mean +/- std
    print(f"\n  {'model':<16} {'system acc':>20} {'routing acc':>20}")
    print("  " + "-" * 60)
    labels = [
        ("base",      "base (phased)"),
        ("d5_soft",   "D5a soft"),
        ("d5_oracle", "D5b oracle"),
        ("d5_self",   "D5c self-distil"),
    ]
    for key, label in labels:
        s = agg[key]["system"]
        r = agg[key]["routing"]
        print(f"  {label:<16} {s['mean']*100:>6.2f}+/-{s['std']*100:.2f}%       "
              f"{r['mean']*100:>6.2f}+/-{r['std']*100:.2f}%")

    # Delty wzgledem base (sygnal vs szum)
    print("\n  Delty do base (system acc):")
    for key, label in labels[1:]:
        d      = agg[key]["system"]["mean"] - agg["base"]["system"]["mean"]
        pooled = max(agg["base"]["system"]["std"], agg[key]["system"]["std"])
        sig    = "SYGNAL" if abs(d) > pooled else "SZUM"
        print(f"    {label:<16}: {d*100:+.2f}pp  "
              f"(max_std={pooled*100:.2f}pp)  [{sig}]")

    return {
        "dataset":   ds_name,
        "n_seeds":   N_SEEDS,
        "config": {
            "epochs":         EPOCHS,
            "distill_epochs": DISTILL_EPOCHS,
            "distill_lr":     DISTILL_LR,
            "T":              T_TEMP,
            "lambda_kd":      LAMBDA_KD,
            "bb_h": BB_H, "emb": EMB, "pod_h": POD_H,
        },
        "per_seed": per_seed,
        "stats":    agg,
    }


# ======================================================================== main
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("DROGA D -- D5: Distillation pod -> router")
    print(f"Device: {device}",
          f"({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    print(f"Seeds: {N_SEEDS}  |  Base epochs: {EPOCHS}  |  "
          f"Distill epochs: {DISTILL_EPOCHS}")
    print(f"T={T_TEMP}  lambda={LAMBDA_KD}  distill_lr={DISTILL_LR}")
    print("Warianty: D5a=soft (dynamic teacher), D5b=oracle, D5c=self (frozen teacher)")
    print("Baseline odniesienia (D1c/v2a):")
    print("  MNIST:         system 98.36 +/- 0.05pp")
    print("  Fashion-MNIST: system 89.50 +/- 0.11pp")
    print("=" * 72)

    results = {}
    for ds_name in ["MNIST", "Fashion-MNIST"]:
        results[ds_name] = run_dataset(ds_name, device)

    print("\n" + "=" * 72)
    print("PODSUMOWANIE")
    print("=" * 72)
    for ds_name in ["MNIST", "Fashion-MNIST"]:
        r = results[ds_name]
        base_sys = r["stats"]["base"]["system"]["mean"] * 100
        print(f"\n  {ds_name}  (base={base_sys:.2f}%):")
        for key, label in [("d5_soft",   "D5a soft"),
                            ("d5_oracle", "D5b oracle"),
                            ("d5_self",   "D5c self-distil")]:
            d      = r["stats"][key]["system"]["mean"] - r["stats"]["base"]["system"]["mean"]
            pooled = max(r["stats"]["base"]["system"]["std"],
                         r["stats"][key]["system"]["std"])
            sig    = "SYGNAL" if abs(d) > pooled else "SZUM"
            sys_m  = r["stats"][key]["system"]["mean"] * 100
            sys_s  = r["stats"][key]["system"]["std"] * 100
            print(f"    {label:<16}: {sys_m:.2f}+/-{sys_s:.2f}pp  "
                  f"delta={d*100:+.2f}pp  [{sig}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "D5_distillation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWynik zapisany: {os.path.abspath(out)}")
    print("\nD5 zakonczone.")
    print("-> Jesli SYGNAL (>1std): analiza ktory wariant dziala i dlaczego.")
    print("-> Jesli SZUM: router osiagnal sufit tej reprezentacji, -> D4 lub CNN.")


if __name__ == "__main__":
    main()
