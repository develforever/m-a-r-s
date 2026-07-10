# M.A.R.S.: Modular Autonomous Refinement System
## The Routing Ceiling, the Immutable Representation, and Memory Without Data

**Draft v0.4 — July 2026** (v0.1: June 2026, TMU routing PoC — retained as Part I; v0.2: routing ceiling study — Part II; v0.3: July 2026 code freeze; v0.4: Series J — audit cleared, sparse dreams)

---

## Abstract

We present a systematic empirical study of modular neural architectures (router + specialist pods on a shared backbone), conducted with a strict multi-seed methodology in which every experiment has a pre-registered, falsifiable verdict criterion. Part II establishes a **routing ceiling**: on a shared representation, routing accuracy cannot be improved by any modification of the routing *algorithm* — demonstrated on three independent information channels (ensemble: +0.00pp; distillation: −2.5pp; predictive coding: +0.05pp, all within noise). The ceiling is set by the representation: a small CNN backbone raises Fashion-MNIST from 89.61% to 91.99%, with the gain flowing through routing, and a slimmed variant retains 70% of it at 1.81× MLP cost. Hierarchical routing then exposes **oracle inflation**: the standard "router→oracle gap" overstates routing headroom (~0.5pp real vs 6pp apparent), because oracle assignment leaks label information.

Part III applies the design law that recurs throughout the study — *a narrowly-supervised shared representation is worse than none* (four independent instances) — to class-incremental continual learning. Freezing the representation entirely and placing all plasticity in prototypes and pods, we replace the episodic memory buffer with two data-free components: **semantic anchors** (class prototypes from public word embeddings, existing before any data) and **parametric sleep** (per-class k-centroid feature statistics, ~4 KB/class, dreamed as balanced rehearsal). On Split-Fashion (class-IL) this reaches statistical equivalence with experience replay (77.6 ± 1.0% vs 77.0 ± 1.1% — nominally above, within the pre-registered noise threshold) at zero stored samples, constant inference cost, and ~30% less forgetting. On Split-CIFAR-10 the ranking inverts: replay-200 collapses (18.9 ± 8.8%; 14.0 ± 4.9% even with per-channel input normalization that helps its trainable backbone) while the immutable-representation system remains stable, and a sparsity-aware dream model (spike-and-slab: per-dimension activation probability with moments conditional on activity, so dreamed zeros are exact zeros) lifts it to 37.5 ± 1.4% — a pre-registered SIGNAL+ over diagonal dreams (+4.5pp, all 5 paired seeds) whose effect size grows with data difficulty (Fashion +0.9pp, within noise; CIFAR +4.5pp). The harder the data, the more valuable a representation that cannot drift — and a dream that respects its geometry. Boundary conditions are measured, not hidden: the mechanism requires class names with visual semantics (digit names fail), absolute accuracy is capped by random features, and compositional zero-shot from attribute descriptions fails its pre-registered threshold while confirming a structural reachability rule 3-for-3. We frame the comparison honestly on a **resource axis**: replay consumes stored user pixels; our system consumes public, static word geometry.

---

## 1. Introduction

Modern deep learning activates the entire network for every input, regardless of difficulty. Mixture-of-Experts architectures address this with learned routing, but introduce a new question that is rarely answered rigorously: **what limits the router?** Is it the routing algorithm (how the decision is made), or the representation (what the decision is made from)?

This paper answers that question empirically for a concrete modular system, with a methodology designed to make negative results as informative as positive ones:

- every experiment runs on 5 seeds with paired per-seed comparisons;
- every experiment has a verdict criterion (SIGNAL+/NOISE/SIGNAL−) fixed *before* results are seen;
- the noise threshold is defined as std(baseline) + std(variant);
- all costs are reported in MAC (multiply-accumulate) operations per sample, with the shared backbone counted honestly as an always-paid fixed cost.

### 1.1 Contributions

1. **The routing ceiling (negative result, three channels).** On a shared backbone, router accuracy is invariant to the routing algorithm. Ensemble consultation of top-k pods, distillation from pods to router, and predictive-coding reconstruction error all fail to improve routing — because every one of these signals is a function of the same features the router already sees (Sections 5–6).
2. **Features are the lever (positive result).** A small CNN backbone raises the ceiling itself: +2.38pp system accuracy on Fashion-MNIST, with the gain demonstrably flowing through routing accuracy, not pod capability (Section 7).
3. **The lever is efficient.** 70% of the CNN gain survives at 1.81× MLP cost. The gain comes from convolutional locality/invariance, not raw compute (Section 7.2).
4. **Supporting mechanisms measured end-to-end**: zero-cost TMU texture routing on GPU (41.9M samples/s), ternary pod quantization at zero accuracy cost, catastrophic-forgetting resistance of modular pods (Part I, Section 8).
5. **A recurring design law** (Part III, Section 9): a shared representation trained under narrow supervision is worse than a random one — four independent instances (distillation, coarse labels, task-limited training, pixel reconstruction). Catastrophic forgetting is this law's sequential form; we therefore make the representation immutable by construction.
6. **Memory without data** (Part III, Sections 10–13): semantic anchors + parametric sleep achieve replay-level class-incremental accuracy with zero stored samples on Split-Fashion, and dominate replay on Split-CIFAR where trainable backbones drift; with measured boundaries (class-name semantics required; random-feature ceiling; compositional zero-shot below threshold but structurally predictable).

---

## Part I — Proof of Concept (Phase 1, v0.1 heritage)

*This part summarizes the v0.1 whitepaper. Full details in `RAPORT_FINAL.md` and `STAN_PROJEKTU.md`. These results established feasibility; Part II supersedes the architecture.*

**Local learning (Stage 1).** Forward-Forward and Contrastive Hebbian learning solve XOR without backpropagation — but cost 5–10M MAC vs 0.56M for backprop. Honest correction: biological plausibility does not imply efficiency.

**Modularity vs forgetting (Stage 2).** Separate neuron pools + router: 95% retention of task A after learning task B, vs 50% for a shared network (+45pp). Later confirmed on MNIST class-split (+25.7pp retention). Known trade-off: retention is bought with capacity.

**Routing economics (Stage 3).** Savings from waking only one specialist are *conditional*: negative at N=2 capsules (router overhead), +50% at N=10, +63% at N=50. Router accuracy degrades gracefully: routing remains profitable at N≥5 even with 50% routing errors.

**TMU texture routing (Stages 3C/4B).** Encoding routing decisions as a 2D texture (SOM label map) and sampling via GPU Texture Mapping Units: 0 MAC per routing decision, 41.9M samples/s pure-fetch throughput on a GTX 1050 Ti (13.8× a baseline MLP). Naive bilinear "semantic interpolation" was falsified on sparse grids; Kohonen SOM topology makes it work (MSE 0.85 → 0.027). Verdict: real mechanism, GPU-only.

**WebGPU deployment.** The full pipeline (WGSL compute shaders, React Three Fiber host) runs in-browser at 60 FPS with <1ms decision latency. Deployment feasibility is proven, not just simulated.

---

## Part II — M.A.R.S. v2 and the Routing Ceiling Study

## 2. Architecture (v2)

```
Input x [784]
   └─> Shared Backbone ──> features h [bb_h]        (always paid, computed ONCE)
            ├─> Routing head: h → emb [32] → −dist(emb, prototypes[10]) → argmax
            └─> Pod[i]: h → hidden [24] → logits [10]   (only the chosen pod runs)
```

- **Shared backbone**: MLP (`Linear 784→256 + ReLU`) or CNN (Section 7). Both router and pods consume the same features.
- **Prototype router**: features are projected to a 32-dim embedding; the routing logit for pod *i* is the negative Euclidean distance to a learned prototype. Hard assignment by argmax.
- **Stacked pods**: 10 two-layer specialists stored as stacked tensors `[n_pods, in, out]`, executed with batched matrix multiplication (no per-pod loop).
- **Adaptive compute** (three-tier inference: early-exit / top-1 / top-2 by router confidence) is retained from v1 but is structurally weak on a shared backbone: the pod is ~3% of total cost, so there is little to save (D1 finding).

**Training (phased, the series standard).** Phase 1: backbone + router trained with cross-entropy. Phase 2: backbone and router frozen; pods train on the samples the router *actually* sends them (not on oracle assignment). This eliminates train/test mismatch and produces honest oracle numbers (~95–98%, not 100%). End-to-end training with a combined loss is statistically indistinguishable (D1) — we use phased for comparability.

**Metrics.** `routing_acc` (router picks the pod matching the true class), `system_acc` (end-to-end accuracy — the headline metric), `oracle_acc` (accuracy under perfect routing — a diagnostic ceiling, never a headline).

## 3. Methodology

Five seeds per experiment; mean ± std with Bessel correction; deltas computed per-seed (paired). Verdict criteria are pre-registered in plan documents (`D6B_PLAN.md`, `D7_PLAN.md`) before execution. Datasets: MNIST and Fashion-MNIST (60k train / 10k test each). Hardware: single GTX 1050 Ti — deliberately modest, keeping every experiment reproducible on consumer hardware.

## 4. Baseline calibration (D1)

Parameter-matched comparison of v2 (shared backbone) against v1 (separate encoders): all differences are within noise (MNIST −0.03/−0.07pp at std 0.10pp; Fashion +0.15/−0.09pp at std 0.27pp). The apparent v2 advantage in earlier work came from training pods on oracle assignments — a methodological artifact, corrected here. **A shared backbone neither helps nor hurts accuracy; it helps cost.**

Diagnosis after D1: on Fashion-MNIST, system 89.5%, oracle ~95.5% — the router is the sole bottleneck. MNIST is saturated (98.4%, oracle ~99.9%).

## 5. The routing ceiling: three channels, one answer

Three attempts to improve routing *without changing the representation*, each using a genuinely different information channel:

| Channel | Experiment | Mechanism | Fashion Δ system | Verdict |
|---|---|---|---|---|
| Ensemble | D4 consultation | low-confidence samples ask top-k pods, weighted vote | **+0.00pp** (identical per-seed predictions) | NOISE |
| Teacher | D5 distillation | router logits trained to match pod outputs (3 teacher variants) | **−2.4 to −2.6pp** | SIGNAL− |
| Generative | D7 predictive coding | per-pod reconstruction decoders; route by reconstruction error (hard / fused / iterative) | **+0.05 ± 0.07pp** (best of 16 variants, sweep on test) | NOISE |

**Why each fails, mechanistically:**

- **D4**: pods see the same features as the router, so their "opinions" are deterministic functions of information the router already has. Even consulting *all* pods changes nothing — softmax vote preserves the argmax.
- **D5**: worse than useless. Distillation fine-tunes the backbone to make routing logits resemble pod outputs — but pod outputs also depend on the backbone. Optimizing one head's similarity to another perturbs the shared representation both are calibrated to. Frozen pods on a shifted feature space lose −2.5pp. *Design law learned: never let auxiliary objectives touch the shared representation.*
- **D7**: the cleanest test. Reconstruction error is a *generatively different* signal (Helmholtz/Friston predictive coding: the expert that best explains the input). Decoders were trained per-pod on actually-routed samples, with the backbone frozen (respecting the D5 law). Result: routing by reconstruction error alone reproduces the router's decisions *slightly worse* (90.24% vs 91.98% routing on the CNN backbone) — the reconstruction channel is a noisy copy of the logit channel, because the decoder is also a function of the same features. Fusing the channels (log-space, λ swept 0.25–4) neither helps nor hurts: the PC signal is dominated, consistent, and empty. A stress-test on the backbone with the *largest* router→oracle gap in the study (strided slim CNN: gap 8.74pp, pods at 99.4% oracle) reproduces the null exactly (+0.05 ± 0.04pp) — the ceiling holds even where the headroom is maximal.

**The ceiling, stated:** *on a shared backbone, all routing-relevant information in the features is already exploited by a trained prototype router. Algorithmic sophistication downstream of the features has zero or negative return.*

## 6. Corollary: where the gain must come from

If the ceiling is representational, only two moves remain: better features (Section 7), or information sources *outside* the current representation (input-level signals, other modalities, hierarchy — future work, Section 9).

## 7. Features are the lever (D6), and the lever is efficient (D6b)

### 7.1 CNN backbone (D6)

Replacing the MLP backbone (`784→256`) with a small CNN (2 conv blocks, 32/64 channels, projection to 128) — everything else identical, same training code:

| Dataset | MLP system | CNN system | Δ system | min per-seed Δ | Verdict |
|---|---|---|---|---|---|
| Fashion-MNIST | 89.61 ± 0.21% | **91.99 ± 0.12%** | **+2.38 ± 0.33pp** | +1.90 (5/5 positive) | **SIGNAL+** |
| MNIST | 98.34 ± 0.05% | **99.19 ± 0.06%** | +0.86 ± 0.10pp | +0.76 (5/5 positive) | **SIGNAL+** |

The gain flows through **routing** (Fashion routing 89.30 → 91.88%, +2.58pp), confirming the ceiling diagnosis: the router was representation-limited, not algorithm-limited. Pods improve too (Fashion oracle 95.49 → 98.10%). Cost: 19.7× MAC (215.6k → 4.25M) — which raises the obvious objection: is this just buying accuracy with compute?

### 7.2 Slim CNN (D6b): no, it is not

Four slimming strategies (fewer channels; strided conv replacing MaxPool; depthwise-separable convolutions), evaluated against per-seed D6 baselines:

| Variant (Fashion) | MAC | ×MLP | Δ vs MLP | Retention of D6 gain |
|---|---|---|---|---|
| Full CNN (32,64) | 4,248k | 19.7× | +2.38pp | 100% |
| S1 half (16,32) | 1,224k | 5.68× | +1.94 ± 0.36pp | 82% |
| **S2 quarter (8,16)** | **390k** | **1.81×** | **+1.66 ± 0.39pp** | **70%** |
| S3 stride (16,32) | 462k | 2.14× | +0.97 ± 0.32pp | 41% |
| S4 depthwise (16,32) | 450k | 2.09× | +1.40 ± 0.55pp | 59% |

**Verdict (pre-registered criterion: best in-budget variant ≥ +1.0pp above noise): SIGNAL+.** S2 holds 70% of the gain at 9% of the full CNN's cost. The returns diminish steeply (1.81× → +1.66pp; 19.7× → +2.38pp): most of the value of convolution is in its *structure* (locality, weight sharing, translation invariance), not its width. MNIST shows the same pattern (S2: +0.73 ± 0.07pp, retention 85%).

Curiosity worth recording: the strided variant (S3) produces the *best pods* in the entire series (oracle 99.53% on Fashion, above even the full CNN) while *hurting* the router — the largest router→oracle gap measured (8.9pp). Downsampling choices affect the two heads differently.

### 7.3 Current best operating points (Fashion-MNIST)

| Operating point | system_acc | MAC | Use case |
|---|---|---|---|
| MLP backbone | 89.61% | 215.6k | minimum cost |
| **Slim CNN S2** | **91.27%** | **390k** | **efficiency frontier** |
| Full CNN | 91.99% | 4,248k | maximum accuracy |

The remaining router→oracle gap on the full CNN is 6.11pp — real headroom that, per the ceiling result, cannot be reached by routing algorithms on this representation.

### 7.4 Decision structure: a small dividend, and an honest recalibration of the ceiling (E2)

Error anatomy (E1) showed the residual gap has structure: ~60% of router errors on Fashion-MNIST lie within four pairs of one "upper-body" cluster, and 86% of recoverable samples sit in the lowest router-confidence quartile. This motivates *restructuring the decision* rather than re-algorithmizing it: route to class groups first (easy, coarse), discriminate within the group second (hard, narrow) — with phase-1 training identical to the flat baseline, so the representation is exactly the same and only the decision structure changes. (A first attempt that trained the backbone on group labels collapsed within-group features — oracle fell from 98% to 68% — a cautionary observation on coarse supervision in its own right.)

Result (Fashion, 5 seeds): hierarchical system **92.20 ± 0.12%** vs flat 91.99 ± 0.12%, Δ = +0.21 ± 0.06pp, above the noise threshold (0.18pp), at *marginally lower* MAC — the project's best operating point, and the first positive routing-side result after three algorithmic nulls. MNIST control: noise, as expected.

The deeper finding is a recalibration. Group routing reaches 99.5% and the hierarchical system sits 0.41pp under its own oracle — routing loss is essentially eliminated — yet the absolute gain is only +0.21pp, because the bottleneck relocates to within-group discrimination on the same features. This exposes the flat oracle (98.10%) as **inflated: selecting the pod by the true label leaks label information into the pod choice.** The hierarchical oracle (92.61%, leaking only the group) is the honest ceiling of this representation — meaning the flat router's true headroom was ~0.5pp, not 6pp, and structure captured about half of it. The routing problem on this benchmark is, for practical purposes, closed: further gains must come from features and scale.

## 8. Quantization (B8): pods are free to compress

Ternary quantization ({−1, 0, +1}, threshold 0.5) of pod weights costs *nothing* in the v1 architecture: MNIST 97.98 → 98.17% (+0.19pp), Fashion 88.85 → 88.77% (−0.08pp), at 16× memory compression — quantization acts as regularization on small heads. The router, by contrast, is quantization-fragile (−18pp): it must stay full-precision.

**Caveat (E4, does not replicate in v2):** applying the same quantization to the v2 slim stack (pods 128→24→10, trained on actually-routed samples) loses −2.7 ± 2.8pp, with one seed dropping −5.9pp. Quantizability depends on head size and training regime — the smaller, routing-trained v2 pods lack the redundancy that made B8's larger, full-data pods compressible. The 16× compression claim should be read as architecture-specific, not general.

---

## Part III — Continual Learning: Memory Without Data

## 9. The design law, and why forgetting is its sequential form

Four independent experiments in this study damaged a shared representation in the same way: an auxiliary distillation objective (−2.5pp, Section 5); coarse group labels (oracle 98% → 68%, Section 7.4); training on the first task only (a *random* frozen backbone beats a task-0-trained one by 24–32pp on 5 seeds, class-IL); and unsupervised pixel reconstruction (autoencoder features at or below random, with the wider variant worse). The pattern: **a shared representation optimized for a narrow objective loses generality it cannot recover — a narrowly-supervised representation is worse than none.**

Catastrophic forgetting is exactly this phenomenon in sequential form: every task update is a narrow objective applied to shared weights. Part III therefore takes the law as an architectural axiom: *the representation is immutable after initialization; all plasticity lives in per-class prototypes and pods, which are themselves frozen after learning.* Forgetting is excluded by construction; the open question is only how much accuracy this costs.

## 10. Protocol and baselines

Split-MNIST and Split-Fashion (5 tasks × 2 classes), later Split-CIFAR-10. The primary protocol is **class-incremental** (no task label at test time — the hardest standard setting); task-IL is reported as a control. Metrics: final average accuracy (ACC), forgetting, backward transfer, and MAC as a function of task count. Baselines, calibrated on identical seeds: sequential fine-tuning collapses to the last task (ACC ≈ 18–20%, forgetting ~97pp); **EWC fails entirely in class-IL** (indistinguishable from fine-tuning, both λ settings — consistent with our Phase-1 finding that weight regularization cannot protect a shared small network); experience replay with a 200-sample balanced buffer is the serious opponent (Fashion 76.97 ± 1.09%, MNIST 88.81 ± 1.06%); joint training bounds the ceiling (90.4 / 99.0%). MARS-CL inference cost is constant in task count (×1.0007 after 5 tasks — 64 MAC of prototype growth).

## 11. Replacing the buffer: semantic anchors + parametric sleep

The base system (random frozen backbone, nearest-class-mean prototypes, per-class pods trained on actually-routed samples) reaches 60.2% Fashion class-IL — far above fine-tuning (+42pp), far below replay. Three diagnostic eliminations then localize the deficit precisely: pod calibration is *not* the lever (adding cross-task pseudo-negatives changes class-IL by +0.0pp on both datasets); random-feature width is *not* the lever (11× MAC → 0pp; the frozen-feature Pareto is flat); the deficit is inter-task routing in the embedding space.

**Semantic anchors** (G1): class prototypes taken from public GloVe word vectors — they exist *before any data* and cannot drift. The diagnostic upper bound is striking: with a projection trained on all classes, word-anchored routing reaches **80.45 ± 0.86%, above replay-200**, on a random backbone with nothing stored. But the two honest sequential variants fail: a projection trained on task 0 does not transfer (17.9%; zero-shot routing on unseen classes is *below* chance — the narrow projection actively pulls unseen classes toward known words), and sequentially fine-tuning the shared projection relocates catastrophic forgetting into the projection itself (F = 98.5pp) — the design law again.

**Parametric sleep** (F3/F3b): because the backbone is frozen, per-class feature distributions are stationary — so the past can be stored as *statistics*, not samples. Per-class diagonal Gaussians (~1 KB/class) are "dreamed" as fresh, class-balanced batches at every gradient step while the projection learns new words. Two implementation lessons carried weight: static minority rehearsal drowns (the same lesson as naive replay), and a single Gaussian is too weak a model of the past to survive 15 epochs of the present (4-epoch training retained more than 15-epoch). The final mechanism — **k-centroid dreams** (k-means, k=4, ~4 KB/class) with moderated projection epochs — reaches:

| Split-Fashion, class-IL | ACC | Forgetting | Stored data |
|---|---|---|---|
| replay-200 | 76.97 ± 1.09% | 27.0pp | 200 samples |
| MARS (k4) | 75.78 ± 1.39% | 21.9pp | 0 samples (~4 KB/class stats) |
| MARS (combo) | 75.68 ± 1.17% | **15.8pp** | 0 samples |
| **MARS (k16, final)** | **77.57 ± 1.02%** | 18.8pp | **0 samples** (~16 KB/class) |

Verdict (pre-registered): statistical equivalence with replay — the final k16 variant is nominally *above* replay (+0.60pp) but within the noise threshold, so we do not claim a win — at zero buffer, constant MAC, and 30–40% less forgetting. Two follow-ups sharpened the picture. Orthogonal weight modification (OWM) on the projection, which *guarantees* old feature mappings cannot move, changed nothing on Fashion — a clean elimination showing the residual gap to the 80.45% upper bound is **dream fidelity, not drift** (OWM did add +5.0pp on MNIST, where word anchors are weak; and OWM *alone*, without the dream, collapses to 42.9 ± 9.7% — the projector protects geometry, not decision boundaries). Raising dream resolution then confirmed the diagnosis (1 Gaussian: 70.8 → k4: 75.8 → k16: 77.6), with a design finding: **full-covariance Gaussians are worse than many local diagonal centroids** (73.9% at 4–16× the memory) — post-ReLU features are sparse, non-negative and multimodal, and a global Gaussian samples off the data manifold. **Boundary condition, measured:** on MNIST the mechanism loses decisively (−19pp to replay) — digit names carry no visual semantics, and even the all-classes upper bound sits below replay. Grounding works where language encodes appearance.

**Sparse dreams (Series J, spike-and-slab).** The full-covariance negative pointed at geometry: post-ReLU features are sparse, non-negative, multimodal. The final refinement models the sparsity explicitly — per centroid and per dimension we store P(feature > 0) plus mean and variance conditional on activity, and dream via a Bernoulli mask over a truncated Gaussian, so zeros are exact zeros rather than clamped tails. On Split-Fashion this yields 78.49 ± 0.91% (worst seed 77.72%, above replay's mean) at forgetting 16.0pp — nominally the project's best, but within the pre-registered noise threshold of diagonal k16 (+0.91pp at a 1.93pp bar; all 5 paired seeds positive), so it is reported as an observation, not a win. At equal memory, structure dominates resolution: sparse k=8 (~12 KB/class) matches or beats diagonal k=16 (~16 KB/class) on both datasets. The verdict-grade evidence arrives on CIFAR (Section 12).

**The resource axis.** This is not a *ceteris paribus* victory over replay: the system injects external knowledge. The honest framing is that every CL method consumes a resource — replay consumes *stored user data* (privacy cost, memory growing with data); MARS consumes *public, static word geometry* (free, task-data-independent, requiring only class names). The result defines a solution category: episodic memory replaced by semantic priors plus parametric sleep.

## 12. Scale test: Split-CIFAR-10 inverts the ranking

On natural images the pre-registered risk was that random features would be too weak. The risk materialized — and the comparison inverted anyway:

| Split-CIFAR-10, class-IL (per-channel normalized input, v0.4) | ACC | Forgetting |
|---|---|---|
| fine-tune | 10.16 ± 0.32% | 66.2pp |
| replay-200 | 14.03 ± **4.93**% | 69.2pp |
| MARS (diagonal k16 dreams) | 33.03 ± 1.16% | 41.9pp |
| **MARS (sparse k16 dreams, final)** | **37.51 ± 1.35%** | **32.7pp** |
| joint (ceiling) | 70.24 ± 0.69% | — |

*(v0.3 numbers on raw /255 input — fine-tune 10.14 ± 0.32%, replay 18.90 ± 8.80%, MARS combo 32.04 ± 1.01%, joint 68.73 ± 2.32% — remain in `results/F4_split_cifar.json`. Normalization helps the trainable monoliths — joint +1.5pp at 3× lower variance — is ~neutral for the frozen random backbone, and does not rescue replay.)*

Replay's trainable backbone drifts on hard data and 200 samples cannot anchor it: it collapses even under input normalization (3/5 seeds at chance level), while MARS' worst seed (36.20%) exceeds replay's mean by three of replay's standard deviations. **The stationarity advantage grows with data difficulty — and so does the dream-fidelity advantage:** sparse dreams add +4.48pp over diagonal dreams on CIFAR (min per-seed +3.50, 5/5 positive, pre-registered SIGNAL+ at a 2.51pp bar) versus +0.91pp (noise) on Fashion, and memory can simultaneously shrink (sparse k=8, ~12 KB/class: +4.06pp). Honest caveats: absolute accuracy remains representation-capped (37.5% vs a 70.2% ceiling — the road up is a stronger *frozen* backbone, not a different CL mechanism); replay's buffer size is an unswept axis (200 was pre-registered; larger buffers would help it).

## 13. Compositional zero-shot (G2): a negative result with structure

If routing operates in a space of word-attributes ("has-sleeves", "is-footwear", ...), a new class could be recognized from its *description alone*. Tested leave-one-out over 10 Fashion classes (11 hand-built binary attributes, concept learning via per-attribute BCE — the DAP recipe; a first version using class-discriminative CE scored exactly 0% zero-shot while reaching 80% on seen classes, a clean demonstration that **cross-entropy learns classes, BCE learns concepts**). Verdict: **negative** — mean zero-shot 3.2% against a pre-registered 30% threshold, with partial signal only where attribute codes are distant (Sneaker 18.2%, T-shirt 10.2%). Two structural findings survive: a reachability rule stated in advance (a class whose distinguishing attribute never varies in training is unlearnable) predicted all three failing classes exactly (0.0% on every seed); and a measured seen/unseen trade-off (longer concept training raises seen accuracy 45→70% while *collapsing* zero-shot 57→18% — bias-to-seen). The path forward is principled: attribute vocabularies need error-correcting code distance (confusable classes currently differ by one bit), decorrelated detectors, and better-than-random features.

## 14. Limitations and roadmap

**Honest limitations.**
1. Part II results (routing ceiling, feature lever) are on MNIST-scale benchmarks; Part III extends the *continual-learning* claims to Split-CIFAR-10, but Part II's ceiling characterization has not been replicated on natural images.
2. MAC is a proxy that **does not predict GPU wall-time** (E4): a monolithic MLP with 0.6× the MAC of the slim CNN runs 16× faster on CUDA (large matmuls: ~50% of peak; small convolutions: ~3%). Time-per-10k-samples at saturated utilization was used as an energy proxy (the GTX 1050 Ti does not expose power telemetry); absolute joules require instrumented hardware. The efficiency claims of small modular networks belong on hardware that honors MAC (CPU SIMD with in-cache pods, NPUs), not desktop GPUs.
3. Part III's absolute accuracy is capped by random frozen features (CIFAR: 37.5% vs a 70.2% joint ceiling). The mechanism is representation-agnostic; stronger frozen backbones (self-supervised or pretrained embeddings) are the measured road up and an explicit fork in the project's identity (from-scratch vs foundation-embedding memory layer).
4. The replay comparison fixes the buffer at 200 samples (pre-registered); buffer size is an unswept axis and larger buffers would favor replay. The semantic mechanism requires class names with visual semantics (MNIST is the measured counterexample).
5. Ternary quantization of pods does not transfer to the v2 slim stack (−2.7 ± 2.8pp; architecture-specific, see Section 8) — and the "≤ noise threshold" verdict criterion itself proved gameable by high variance, motivating an added per-seed worst-case condition in later experiment plans.
6. An engineering-audit pass (Series J, pre-registered) tested two pipeline concerns and cleared both: calibrating the frozen backbone's BatchNorm statistics and per-dimension feature scale normalization are NOISE (the latter negative on all 5 Fashion seeds), and per-channel CIFAR input normalization is ~neutral for the frozen random backbone. The dead-BatchNorm oversight was real but innocent — reported levels are properties of the mechanism, not preprocessing artifacts.

**Roadmap (Droga H and beyond).** Orthogonal weight modification (OWM) on the semantic projection — updates projected into the null space of past-class features, computed from the same statistics the parametric sleep already stores; expected to attack the residual 15.8pp forgetting with an exact guarantee on linear layers. Vector-symbolic binding as a dense-matmul replacement for physical pods (capacity of superposition to be measured, not assumed; requires orthogonalized updates). Attribute vocabularies with error-correcting code distance (G2b). Stronger frozen representations. LSH/Morton/TMU routing retained as a deployment demonstration — routing cost is measured at ~0.1% of inference and is not a bottleneck. Series J (this revision) closed the dream-fidelity thread opened by H1: sparsity-aware spike-and-slab dreams are the new default sleep model; the remaining headroom on both benchmarks is representational.

## 15. Conclusion

Modular neural systems are often motivated by intuition and validated by a single benchmark run. We offer a different template: pre-registered verdicts, paired multi-seed statistics, and equal publication weight for negative results. Under this discipline, a coherent picture emerged across two studies. In Part II: the router is the bottleneck; the bottleneck is representational, not algorithmic — three independent channels confirm it; better features raise the ceiling efficiently; and the standard oracle metric systematically inflates routing headroom. In Part III: the recurring design law — a narrowly-supervised shared representation is worse than none — turns out to be the mechanism of catastrophic forgetting itself, and taking it as an axiom (immutable representation, plasticity only in prototypes and pods) allows episodic memory to be replaced by semantic priors and parametric sleep: replay-level accuracy with zero stored samples on Split-Fashion, and dominance over replay on Split-CIFAR, where trainable representations drift. The boundaries of every claim are measured and reported — which classes of data the mechanism needs, where its accuracy is capped, and which of our own hypotheses failed their pre-registered thresholds. That, we believe, is what an evidence base for modular, continually-learning systems should look like.

---

## Appendix A: Reproducibility

All experiments run on a single consumer GPU (GTX 1050 Ti, 4GB). Every experiment is a standalone script writing a timestamped JSON to `results/`.

```bash
# Part II -- Series D/E core
python src/run_D1_mars_v2_baseline.py     # v2 vs v1, phased vs e2e
python src/run_D4_consultation.py         # ensemble channel   -> NOISE
python src/run_D5_distillation.py         # teacher channel    -> SIGNAL-
python src/run_D6_cnn_backbone.py         # features lever     -> SIGNAL+
python src/run_D6b_slim_cnn.py            # efficiency         -> SIGNAL+
python src/run_D7_predictive_coding.py    # generative channel -> NOISE
python src/run_E1_error_anatomy.py        # gap anatomy
python src/run_E2_hierarchical.py         # decision structure -> SIGNAL+ / oracle inflation
python src/run_E4_energy.py               # MAC vs wall-time
# Part III -- Series F/G (continual learning)
python src/run_F0_cl_baselines.py         # finetune/EWC/replay/joint
python src/run_F1_mars_cl.py              # frozen backbones   -> random > trained
python src/run_F2_frozen_features.py      # width/AE plateau   -> flat Pareto
python src/run_G1_semantic.py             # word anchors       -> upper bound > replay
python src/run_F3_feature_replay.py       # parametric sleep
python src/run_F3b_drift_control.py       # k-centroid dreams  -> equivalence w/ replay
python src/run_F4_split_cifar.py          # scale test         -> ranking inversion
python src/run_G2_compositional.py        # attribute zero-shot -> negative w/ structure
# Series J -- audit + sparse dreams (v0.4)
python src/run_J1_feature_conditioning.py # BN-calib / sigma-norm -> NOISE (cleared)
python src/run_J2_cifar_normalized.py     # normalized-input CIFAR -> SIGNAL+ vs replay
python src/run_J3_sparse_dreams.py        # spike-and-slab dreams  -> noise-consistent gain
python src/run_J2b_cifar_sparse.py        # sparse dreams on CIFAR -> SIGNAL+ (+4.5pp)
```

Working notes with full result tables: `DROGA_D_NOTATKI.md`, `DROGA_E_NOTATKI.md`, `DROGA_F_NOTATKI.md`, `DROGA_G_NOTATKI.md`, `DROGA_H_NOTATKI.md`, `DROGA_J_NOTATKI.md`. Pre-registered plans: `D6B_PLAN.md`, `D7_PLAN.md`, `DROGA_F_PLAN.md`, `DROGA_G_PLAN.md`, `DROGA_H_PLAN.md`, `DROGA_J_PLAN.md`. Glossary: `SLOWNIK_POJEC.md`. Word vectors: GloVe 6B-50d (public).

## Appendix B: Key metrics summary

| Metric | Value | Context |
|---|---|---|
| Best MNIST system_acc | 99.19 ± 0.06% | full CNN backbone, 5 seeds |
| Best Fashion system_acc | 92.20 ± 0.12% | hierarchical routing on full CNN, 5 seeds |
| Efficiency point | 91.27% @ 390k MAC | slim CNN S2, 1.81× MLP cost |
| Routing ceiling evidence | +0.00 / −2.5 / +0.05pp | D4 / D5 / D7 channels |
| Real routing headroom | ~0.5pp (not 6pp) | oracle inflation, E2 |
| CNN gain retention (slim) | 70% @ 9% of cost | D6b S2 vs D6 |
| TMU routing throughput | 41.9M samples/s | pure texture fetch, GTX 1050 Ti |
| Split-Fashion class-IL | 77.6% @ 0 stored samples | vs replay-200: 77.0% (equivalence, nominally above) |
| Forgetting (k16/combo vs replay) | 18.8/15.8 vs 27.0pp | 30–40% reduction, zero buffer |
| Split-CIFAR class-IL | 37.5 ± 1.4% vs 14.0 ± 4.9% | MARS sparse-k16 vs replay-200, normalized input (inversion, J2b) |
| Dream sparsity effect | +4.48pp CIFAR (SIGNAL+) / +0.91pp Fashion (noise) | spike-and-slab vs diagonal k16 — grows with data difficulty |
| Split-Fashion sparse dreams | 78.49 ± 0.91% (observation) | within noise of diag k16; forgetting 16.0pp |
| Inference cost vs task count | ×1.0007 after 5 tasks | constant-MAC thesis |
| Design-law instances | 4 | D5, E2-v1, F1, F2 (narrow supervision < none) |

---

*M.A.R.S. — Modular Autonomous Refinement System. PyTorch + WebGPU. All numbers measured, including the ones that said no.*
