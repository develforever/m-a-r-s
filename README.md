# M.A.R.S. — Modular Autonomous Refinement System

**Continual learning without stored data: semantic anchors + parametric sleep
on a frozen random representation.**

Independent research project. Every experiment: 5 seeds, pre-registered
verdict criteria, negative results reported with equal weight. All runs on a
single consumer GPU (GTX 1050 Ti, 4 GB). Code at `v1.0` (July 2026); every series is preserved as a tag (`v0.3-freeze` … `v0.11`).

## Headline results (class-incremental learning, 5 tasks × 2 classes)

| Split-Fashion (class-IL) | ACC | Forgetting | Stored samples |
|---|---|---|---|
| fine-tuning | 17.96 ± 4.47% | 96.7pp | 0 |
| EWC (λ=100/1000) | ≈ fine-tuning | ~96pp | 0 |
| experience replay (200-sample buffer) | 76.97 ± 1.09% | 27.0pp | 200 |
| **MARS-CL (k16, this work)** | **77.57 ± 1.02%** | **18.8pp** | **0** (~16 KB/class stats) |
| joint training (ceiling) | 90.37 ± 0.84% | — | — |

Statistical equivalence with replay (nominally above, within the
pre-registered noise threshold), at zero stored samples, constant inference
cost (×1.0007 after 5 tasks) and ~30% less forgetting. Series J adds a
sparsity-aware dream (spike-and-slab): Fashion 78.49 ± 0.91% — nominally the
best, within the noise threshold of k16 (reported as an observation). Series K composes the levers: sparse dreams × GloVe-300d reach **79.23 ± 0.73%** (paired-SIGNAL+), 97.6% of the 81.16 frozen-feature ceiling.

| Split-CIFAR-10 (class-IL, normalized input) | ACC | Forgetting |
|---|---|---|
| experience replay (200) | 14.03 ± **4.93**% | 69.2pp |
| MARS-CL (diag k16) | 33.03 ± 1.16% | 41.9pp |
| **MARS-CL (sparse k16, this work)** | **37.51 ± 1.35%** | **32.7pp** |
| joint training (ceiling) | 70.24 ± 0.69% | — |

On natural images the ranking inverts: replay's trainable backbone drifts and
collapses (even with input normalization that lifts the joint ceiling); the
immutable representation cannot drift. Sparse spike-and-slab dreams add a
pre-registered SIGNAL+ (+4.48pp, 5/5 seeds) over diagonal dreams — the
stationarity advantage *and* the dream-fidelity advantage grow with data
difficulty. (v0.3 raw-input numbers remain in `results/F4_split_cifar.json`.)

**Measured boundaries (not hidden):** the mechanism requires class names with
visual semantics — on Split-MNIST it loses to replay by ~19pp (digit names
carry no visual meaning); absolute accuracy is capped by random frozen
features (CIFAR: 37.5% vs a 39.65% frozen-feature ceiling — the mechanism
realizes 94.6% of it; the remaining gap to the 70.2% joint ceiling is
representational — K0); compositional zero-shot from
attribute descriptions failed its pre-registered threshold (with failures
predicted 3-for-3 by a structural reachability rule).

## Collective learning by dream exchange (Series I, v0.6)

Five agents share a frozen random backbone (same seed) and the same word
space, and exchange only per-class sleep statistics (~24 KB per class; no
images, no gradients, no weights). A class learned from a single 24 KB
message reaches 94.26% where local training on 12,000 images reaches 95.55%
(full-accuracy equivalence). A collective of five agents — each having seen
only 2 of 10 classes — scores **78.87 ± 1.01%**, statistically equivalent to
one agent trained sequentially on all the data (79.23 ± 0.73%) and nominally
above replay-200 (76.97%). Eight of the collector's ten classes were learned
from dreams alone. Details: `DROGA_I_NOTATKI.md`.

## Identity fork: frozen pretrained backbone (Series L, v0.7)

Reported as a separate resource line (from-scratch vs foundation-embedding).
Swapping the random backbone for a frozen ImageNet ResNet18 — mechanism,
memory (24 KB/class) and exchange protocol untouched — lifts Split-CIFAR-10
class-IL from 37.51% to **74.69 ± 0.69%** (SIGNAL+, +37.2pp), with the
mechanism still realizing 96.7% of its all-data ceiling (77.23%). The
sequential learner and the five-agent collective (**74.13 ± 0.57%**) both
beat the trainable joint monolith (70.24%). The collective's cost versus
sequential is now measured: paired −0.56pp. Details: `DROGA_L_NOTATKI.md`.

## Long horizon: CIFAR-100, 20 tasks (Series M, v0.8)

Over 20 tasks / 100 classes (pretrained backbone) the mechanism holds
**40.70 ± 0.84%** — 85.8% of its frozen-feature ceiling (47.41%), down from
96.7% at T=5. ~79% of the raw per-task drop is the growing label space (the
ceiling drops identically); the remaining −7.8pp late-task deficit is
structural: the per-old-class dream budget is a measured stability–plasticity
knob, and per-class rehearsal stays ACC-optimal. At 100 classes, 300d word
anchors beat 50d by +7.7pp (SIGNAL+, 5/5 seeds) — anchor dimension must scale
with class count. Details: `DROGA_M_NOTATKI.md`.

## Selective forgetting with a guarantee (Series N, v0.9) — and a falsification (Series O, v0.10)

Deleting a class's entries removes access but not information (100 images
restore it near-bit-identically — the projection is the sole information
carrier). Fine-tuning the projection on remaining-class dreams erases ~84% of
recoverability; **re-initializing it and rebuilding from dreams erases 100%**
(re-learning matches a never-seen class) at ≤ zero cost to remaining classes.
Series O then falsified the tempting corollary: rebuilding the projection from
dreams of *all* classes after the sequence is paired-negative on both
benchmarks — dreams protect, transfer, and rebuild, but do not improve on real
data. Details: `DROGA_N_NOTATKI.md`, `DROGA_O_NOTATKI.md`.

## Untrusted collective (Series I4, v0.11)

A label-swap poisoned payload destroys both co-adopted classes but leaves the
recipient's own classes robust (±1pp); neither pre-registered detector
separates attacks from honest payloads on the random backbone (honest
negative — semantic detection likely needs pretrained features); and the
attack is fully reversible by forget-and-readopt of the adoption batch (noise
on all metrics vs the clean path). Protocol policy: adopt in batches, repair
at batch scope. Details: `DROGA_I4_NOTATKI.md`.

## Payload detection: gate and distance law (Series P, v1.1)

Semantic (pretrained) features alone do **not** rescue payload-attack
detection (double negative, with a paired random-backbone control). Two
pre-registered results on fresh seeds then convert the negative: a
**structural gate** (rank-consistency > 0.45) classifies 60/60
clean/structureless payloads correctly on both backbones — and also catches
mid/far-distance swaps; and a **distance law**: swap detectability grows with
the semantic distance of the forged pair (full separation at anchor-cos
0.487/0.139, none at 0.775/0.615) — close-pair swaps are undetectable *by
construction*, which retroactively explains the I4/P1 negatives. Protocol
policy, fully measured: batch adoption + entry gate + repair covers the
invisible close swaps. Details: `DROGA_P_NOTATKI.md`.

## Collective at long horizon (Series Q, v1.2)

At the default dream budget the 20-agent collective (95 of 100 CIFAR-100
classes from 24 KB messages) pays a **scale barrier**: −6.67 ± 0.88pp vs the
sequential agent — front-loaded in early adoptions, repairable by
forget-and-readopt (61%) and fully compensated by raising the adoption dream
budget to 2500/class (154%; payload unchanged). A pre-registered fairness
control cut both ways: **self-dream augmentation lifts any agent** (+4.66pp;
best single agent 45.35 ± 0.49 = 95.7% of the all-data ceiling), and at
symmetric budgets the collective is **equivalent** to the sequential agent
(44.29 vs 45.35, within noise) — the equivalence claim of Series I/L extends
10× in scale. Details: `DROGA_Q_NOTATKI.md`.

## The mechanism in one paragraph

A design law recurred across four independent experiments in this project:
*a shared representation trained under a narrow objective is worse than none*.
Catastrophic forgetting is that law in sequential form — so we make the
representation **immutable by construction** (a random frozen backbone) and
put all plasticity in per-class components. The episodic replay buffer is
replaced by two data-free resources: **semantic anchors** (class prototypes =
GloVe word vectors, which exist before any data and cannot drift) and
**parametric sleep** (per-class k-means centroids of features, ~KB/class,
"dreamed" as balanced rehearsal while a projection learns to align features
with word geometry). The final dream model is sparsity-aware (spike-and-slab:
per-dimension activation probability + conditional moments — dreamed zeros are
exact zeros; Series J). Full story: `WHITEPAPER.md` (Part III; Parts I–II cover
the earlier routing-ceiling study).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_glove.py   # GloVe 6B-50d (public word vectors)

# Part III — continual learning (the headline series)
python src/run_F0_cl_baselines.py   # finetune / EWC / replay / joint
python src/run_F1_mars_cl.py        # frozen backbones: random > trained
python src/run_F2_frozen_features.py
python src/run_G1_semantic.py       # word anchors: upper bound > replay
python src/run_F3_feature_replay.py # parametric sleep (1 Gaussian)
python src/run_F3b_drift_control.py # k-centroid dreams -> equivalence
python src/run_F4_split_cifar.py    # scale test -> ranking inversion
python src/run_H1_owm.py            # OWM elimination: gap = dream fidelity
python src/run_H1b_dream_fidelity.py# k16 -> 77.6%; full-covariance negative
python src/run_G2_compositional.py  # attribute zero-shot -> negative

# Series J — audit + sparse dreams (v0.4)
python src/run_J1_feature_conditioning.py # audit: BN-calib/σ-norm -> NOISE
python src/run_J2_cifar_normalized.py     # normalized CIFAR rerun -> SIGNAL+ vs replay
python src/run_J3_sparse_dreams.py        # spike-and-slab dreams (Fashion/MNIST)
python src/run_J2b_cifar_sparse.py        # sparse dreams on CIFAR -> SIGNAL+ (+4.5pp)
python src/run_J4_glove300.py             # GloVe-300d -> seq null; ceiling 81.16
# Series K (v0.5) -- ceilings and lever composition
python src/run_K0_cifar_ceiling.py        # frozen-feature ceiling CIFAR 39.65
python src/run_K1_sparse300.py            # sparse x 300d -> paired-SIGNAL+ 79.23
python src/run_K2_owm_sparse.py           # OWM x sparse -> eliminated
# Series I (v0.6) -- collective learning by dream exchange
python src/run_I1_transplant.py           # class from a 24 KB message
python src/run_I2_fusion.py               # payload saturation
python src/run_I3_collective.py           # 5 agents ~ 1 sequential agent
python src/run_I2b_fusion_lowdata.py      # fusion helps iff payload unsaturated
# Series L (v0.7) -- identity fork: pretrained frozen backbone
python src/run_L1_pretrained.py           # CIFAR 74.69 (+37.2pp); beats trainable joint
python src/run_L2_collective_cifar.py     # collective on strong features: 74.13
# Series M (v0.8) -- long horizon: CIFAR-100, 20 tasks
python src/run_M1_long_horizon.py         # 40.70 @T=20; anchors 300d SIGNAL+ (+7.7pp)
python src/run_M1b_balanced_dreams.py     # dream-budget knob, low end
python src/run_M1c_mid_budget.py          # mid point -> the front is sharp
# Series N (v0.9) -- selective forgetting with a guarantee
python src/run_N1_unlearning.py           # functional level: no collateral damage
python src/run_N1b_relearn_balanced.py    # information level: light 100% / scrub ~84%
python src/run_N1c_reinit.py              # re-init: full erasure guarantee
# Series O (v0.10) -- dream consolidation: FALSIFIED
python src/run_O1_consolidation.py        # deep-sleep rebuild -> paired-SIGNAL- (both)
# Series I4 (v0.11) -- untrusted collective
python src/run_I4_untrusted.py            # damage map; detection negative
python src/run_I4b_full_repair.py         # forget-and-readopt: full recovery
# Series P (v1.1) -- payload detection
python src/run_P1_detect_pretrained.py    # pretrained: double negative
python src/run_P1c_gate_distance.py       # gate 60/60; distance law
# Series Q (v1.2) -- collective at long horizon
python src/run_Q1_collective_horizon.py   # scale barrier -6.67pp
python src/run_Q2_early_repair.py         # readopt 61% / budget 154%
python src/run_Q2c_seq_selfdream.py       # self-dream lever; equivalence

# Part II — routing ceiling study
python src/run_D1_mars_v2_baseline.py
python src/run_D4_consultation.py   # ensemble channel   -> NOISE
python src/run_D5_distillation.py   # teacher channel    -> SIGNAL-
python src/run_D6_cnn_backbone.py   # features lever     -> SIGNAL+
python src/run_D6b_slim_cnn.py      # efficiency         -> SIGNAL+
python src/run_D7_predictive_coding.py # generative      -> NOISE
python src/run_E2_hierarchical.py   # oracle inflation
```

MNIST / Fashion-MNIST / CIFAR-10 download automatically (torchvision).
Every runner writes per-seed JSON to `results/` (all shipped in this repo).
Most runners accept `--smoke` for a fast sanity pass.

## Methodology

Five seeds per experiment; paired per-seed deltas; noise threshold =
std(baseline) + std(variant); verdict criteria (SIGNAL+ / NOISE / SIGNAL−)
fixed in plan documents *before* execution; minimum per-seed reported
alongside means; costs in MAC with the shared backbone counted as always-paid.

## Repository map

| Path | Content |
|---|---|
| `WHITEPAPER.md` | Full write-up (EN): PoC, routing ceiling, memory without data |
| `RELATED_WORK.md` | Verified bibliography and positioning |
| `src/` | All runners and models (PyTorch) |
| `demo/mars_cl_demo/` | Interactive browser demo of MARS-CL (pure JS, parity-checked vs PyTorch) |
| `results/*.json` | Per-seed results for every experiment, including smoke runs |
| `DROGA_*_NOTATKI.md` (D/E/F/G/H/J/K/I/L/M/N/O/I4) | Working notes with full result tables (Polish) |
| `CLAIMS.md` | Every claim of the project: series → verdict → results file (EN) |
| `PLAN_GENERALNY.md`, `PLAN_V1.md` | Stage maps: executed (K→I4) and current (v1.0 review + candidates) |
| `*_PLAN.md` | Pre-registered experiment plans, written before runs (Polish) |
| `SLOWNIK_POJEC.md` | Glossary (Polish) |
| `RAPORT_FINAL.md`, `STAN_PROJEKTU.md` | Phase-1 PoC tech report (Polish) |

Working notes are in Polish — they are the project's lab notebook and are
kept verbatim as evidence of pre-registration (file dates match run dates).
The whitepaper and this README carry the full story in English.

## License

Code: MIT (see `LICENSE`). Result JSONs and documents: CC-BY-4.0.
GloVe vectors (Pennington, Socher & Manning, EMNLP 2014) are downloaded
separately and remain under their own terms.
