# M.A.R.S. — Modular Autonomous Refinement System

**Continual learning without stored data: semantic anchors + parametric sleep
on a frozen random representation.**

Independent research project. Every experiment: 5 seeds, pre-registered
verdict criteria, negative results reported with equal weight. All runs on a
single consumer GPU (GTX 1050 Ti, 4 GB). Code frozen at `v0.3` (July 2026).

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
best, within the noise threshold of k16 (reported as an observation).

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
features (CIFAR: 32% vs 68.7% ceiling); compositional zero-shot from
attribute descriptions failed its pre-registered threshold (with failures
predicted 3-for-3 by a structural reachability rule).

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
| `DROGA_D/E/F/G/H/J_NOTATKI.md` | Working notes with full result tables (Polish) |
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
