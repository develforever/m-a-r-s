# M.A.R.S.: Modular Autonomous Refinement System
## The Routing Ceiling, the Immutable Representation, and Memory Without Data

**Draft v1.0 — July 2026** (v0.1: June 2026, TMU routing PoC — retained as Part I; v0.2: routing ceiling study — Part II; v0.3: July 2026 code freeze; v0.4: Series J — audit cleared, sparse dreams; v0.5: Series K — ceilings measured; v0.6: Series I — collective dream exchange; v0.7: Series L — pretrained fork; v0.8: Series M — long horizon; v0.9: Series N — selective forgetting; v0.10: Series O — consolidation falsified; v0.11: Series I4 — untrusted collective; v1.0: consolidated review, all headline numbers re-verified against per-seed JSONs)

---

## Abstract

We present a systematic empirical study of modular neural architectures (router + specialist pods on a shared backbone), conducted with a strict multi-seed methodology in which every experiment has a pre-registered, falsifiable verdict criterion. Part II establishes a **routing ceiling**: on a shared representation, routing accuracy cannot be improved by any modification of the routing *algorithm* — demonstrated on three independent information channels (ensemble: +0.00pp; distillation: −2.5pp; predictive coding: +0.05pp, all within noise). The ceiling is set by the representation: a small CNN backbone raises Fashion-MNIST from 89.61% to 91.99%, with the gain flowing through routing, and a slimmed variant retains 70% of it at 1.81× MLP cost. Hierarchical routing then exposes **oracle inflation**: the standard "router→oracle gap" overstates routing headroom (~0.5pp real vs 6pp apparent), because oracle assignment leaks label information.

Part III applies the design law that recurs throughout the study — *a narrowly-supervised shared representation is worse than none* (four independent instances) — to class-incremental continual learning. Freezing the representation entirely and placing all plasticity in prototypes and pods, we replace the episodic memory buffer with two data-free components: **semantic anchors** (class prototypes from public word embeddings, existing before any data) and **parametric sleep** (per-class k-centroid feature statistics, ~4 KB/class, dreamed as balanced rehearsal). On Split-Fashion (class-IL) this reaches statistical equivalence with experience replay (77.6 ± 1.0% vs 77.0 ± 1.1% — nominally above, within the pre-registered noise threshold) at zero stored samples, constant inference cost, and ~30% less forgetting. On Split-CIFAR-10 the ranking inverts: replay-200 collapses (18.9 ± 8.8%; 14.0 ± 4.9% even with per-channel input normalization that helps its trainable backbone) while the immutable-representation system remains stable, and a sparsity-aware dream model (spike-and-slab: per-dimension activation probability with moments conditional on activity, so dreamed zeros are exact zeros) lifts it to 37.5 ± 1.4% — a pre-registered SIGNAL+ over diagonal dreams (+4.5pp, all 5 paired seeds) whose effect size grows with data difficulty (Fashion +0.9pp, within noise; CIFAR +4.5pp). The harder the data, the more valuable a representation that cannot drift — and a dream that respects its geometry. Boundary conditions are measured, not hidden: the mechanism requires class names with visual semantics (digit names fail), absolute accuracy is capped by random features, and compositional zero-shot from attribute descriptions fails its pre-registered threshold while confirming a structural reachability rule 3-for-3. We frame the comparison honestly on a **resource axis**: replay consumes stored user pixels; our system consumes public, static word geometry.

Seven follow-up series (Sections 14–20) close the mechanism's map. Measured ceilings show the sequential learner realizes 94.6–97.6% of what the frozen features admit (K); the memory layer doubles as a **communication protocol** — five agents exchanging only ~24 KB/class of dream statistics (no images, gradients, or weights) match one agent trained sequentially on all data (I); a frozen ImageNet ResNet18 behind the *unchanged* mechanism lifts Split-CIFAR to 74.7 ± 0.7%, beating the trainable joint monolith (L); over 20 CIFAR-100 tasks the mechanism holds 85.8% of its ceiling and the anchor dimension must scale with the class count (+7.7pp for 300d, M); selective forgetting comes with a measured guarantee — projection re-init erases a class to never-seen level at ≤ zero cost to the rest (N); dream consolidation is **falsified** — dreams protect, transfer, and rebuild, but do not improve on real data (O); and a poisoned payload damages only its adoption batch, is undetectable on random features (an honest negative), and is fully reversible by forget-and-readopt (I4). Sleep learns, shares, and forgets — each leg pre-registered and measured.

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
6. **Memory without data** (Part III, Sections 10–20): semantic anchors + parametric sleep achieve replay-level class-incremental accuracy with zero stored samples on Split-Fashion, and dominate replay on Split-CIFAR where trainable backbones drift; with measured boundaries (class-name semantics required; random-feature ceiling; compositional zero-shot below threshold but structurally predictable).

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

**G2b tested the first lever in isolation, and it backfired.** A pre-registered 21-attribute vocabulary with minimum Hamming distance 4 (up from 1) and no attribute constant in any leave-one-out — designed so the reachability rule predicts all 10 classes reachable — *lowered* zero-shot to 0.18 ± 0.08% against the reproduced baseline's 3.17 ± 0.90% (paired −2.98pp, 5/5 below the noise threshold). The 10/10 reachability prediction is falsified: six classes remain exactly 0.0%, so structural reachability is necessary but not sufficient. The mechanism is clean: error-correcting codes only correct below a per-bit error threshold, and near-chance concept detectors on random features sit above it — each added orthogonal bit multiplies the decode noise, so a longer code is *harder* to hit, not easier (testing a correcting code over a channel past its capacity). This isolates the bottleneck to the features (lever c), not the vocabulary, and points the series at the same fork as the rest of the study: put strong frozen features behind the unchanged mechanism (G3, future work). A secondary observation reinforces the trade-off from G2 along a new axis: 21 attributes raised *seen* accuracy (0.72–0.81 vs 0.65–0.76) while collapsing zero-shot — more vocabulary is more capacity to memorize seen combinations.

**G3 tested the last external lever — feature quality — and it is null, which closes the series.** Swapping the random backbone for frozen ImageNet ResNet18 (512-d), with the identical mechanism, dictionaries, and leave-one-out protocol, leaves compositional zero-shot statistically unchanged: 2.77 ± 0.15% on strong features versus 3.17 ± 0.90% on random (paired −0.40 ± 0.88pp, SZUM, ratio 0.87×), still far below the 30% line — a pre-registered **G3−**. The random arm reproduces G2b exactly (harness valid), ECOC still backfires on the strong features (−2.69pp), and the structurally-unreachable classes stay at 0.0%. The prior was that features were the bottleneck (Series L moved CIFAR +37pp by this exact swap); here they are not. With code distance backfiring (G2b) and feature quality null (G3), all three levers named in the G2 diagnosis are exhausted: compositional zero-shot from a **hand-built attribute dictionary with linear concept detectors** is a limit of the *paradigm* — invariant to representation quality and code distance — not of any one component. A different paradigm (attributes learned end-to-end, non-linear detectors) would be required; the hand-dictionary route is closed.

## 14. Ceilings and lever composition (Series K, v0.5)

How much of the frozen features' potential does the sequential mechanism realize? Training the semantic projection on *all* data over the same frozen random features tops out at 39.65 ± 1.21% on CIFAR (K0) and 81.16 ± 0.87% on Fashion (300d anchors, J4). The sequential learner reaches **94.6% and 97.6% of those ceilings** respectively — the residual gap to joint training is representational, not mechanistic. Composing the two measured levers (sparse dreams × GloVe-300d anchors) gives the project's Fashion best, 79.23 ± 0.73% (paired-SIGNAL+, all 5 seeds; K1); the same composition on CIFAR is noise. OWM on the semantic projection is eliminated under the sparse sleep: its H1 MNIST gain vanishes and CIFAR shows a paired negative (K2). Verdict of the series: the mechanism thread is closed — further absolute gains must come from the representation.

## 15. Collective learning by dream exchange (Series I, v0.6)

The memory layer is also a communication protocol. Agents sharing a frozen random backbone (same seed — synchronization is free) exchange only per-class spike-and-slab statistics plus a class name: **~24 KB, no images, no gradients, no weights**. A class learned from a single such message loses 1.29pp versus local training on 12,000 images (full-accuracy equivalence at the pre-registered weak threshold; I1). A five-agent collective — each agent seeing only 2 of 10 Fashion classes, adopting the other 8 from dreams — reaches 78.87 ± 1.01%, statistically equivalent to one agent trained sequentially on all data (79.23; SZUM) and nominally above replay-200 (I3). Fusion of partial views of the same class helps exactly when the payload is unsaturated: paired-SIGNAL+ at 100 img/class, noise at 500+ — the 24 KB message saturates between 500 and 3000 images (I2/I2b). Positioning: federated learning exchanges gradients or weights; this system exchanges sleep. No sample ever leaves an agent, and the message is generative only in the feature space of a frozen backbone.

**Dropping the shared-backbone assumption (Series R).** The protocol above assumes a common frozen backbone, so a payload lives in a feature space both agents share. Series R asks whether agents with *different* backbones can still exchange sleep. The natural first idea — translate through the one space that is shared by construction, the word-anchor space — is falsified: even an oracle decoder with full class knowledge and zero heterogeneity collapses adopted accuracy to 4% versus an 81% ceiling, because the anchor space is a low-dimensional *routing* space, class-collapsed onto each word vector, and carries a class's identity but not its within-class geometry (R1b). The fix is to align in *feature* space. A regularized linear map H_A→H_B, fit on as few as **two shared calibration classes** (a public probe set passed through both backbones), lets a recipient re-express a sender's sleep payload in its own space and adopt it through the unchanged machinery: on R-mild (same ImageNet ResNet18, different random 512→128 projection) this recovers **92% of the homogeneous ceiling** (79.98 ± 2.45% vs 86.87 ± 1.09%, a −6.9pp measured cost, fresh seeds), a pre-registered gate (≥70% of ceiling) passed with wide margin (R2b). The orthogonality of a classical Procrustes alignment *hurts* (31%): the inter-representation relationship is a general linear transform, not an isometry. This is the protocol's first representation-agnostic transfer — heterogeneous agents sharing classes through a two-class calibration bridge, no raw data exchanged.

**R-hard: genuinely different representations.** The decisive test drops the shared basis entirely — a from-scratch random CNN reading raw pixels (128-d) versus a frozen ImageNet ResNet18 (512-d), different feature *content* and different *dimension*, bridged by a **rectangular** ridge H_A(D_A)→H_B(D_B) with the core untouched (the 512-d sender exports a stats-only payload; the receiver stays 128-d because the projection and pods are fixed at that width). Foundation→scratch recovers **76% of the receiver's own homogeneous ceiling** (42.28 ± 2.73% vs 55.83 ± 1.64%, +36.9pp over the anchor floor, a −13.55pp measured cost — larger than R-mild's −6.9pp, exactly as pre-registered, because no common feature basis exists), while the sanity control (a rectangular map that exists by construction) sits at 93% and confirms the machinery is sound. As pre-registered, and unlike R-mild where two calibration classes sufficed, the curve now *rises* with K — a harder cross-content alignment needs more shared probe classes. The reverse direction (scratch→foundation, a dimension-matched square map) recovers a lower 45% of its ceiling with a falling K-curve, and is reported separately because direction and map shape are confounded there. The collective is thus representation-agnostic even across genuinely different feature bases — at a measured, and larger, cost.

## 16. The identity fork: pretrained frozen backbone (Series L, v0.7)

The mechanism is representation-agnostic by construction; Series L makes that claim empirical. With a frozen ImageNet ResNet18 behind the **unchanged** mechanism and protocol, sequential Split-CIFAR class-IL jumps from 37.51 to **74.69 ± 0.69%** (+37.2pp, SIGNAL+), the mechanism still realizes 96.7% of its new all-data ceiling (77.23 ± 0.57), and the sequential learner *beats the trainable joint monolith* (70.24) — stationarity plus strong features outperform end-to-end training on this benchmark (L1). The five-agent collective transfers to the strong features with a small, now-measured protocol cost (paired −0.56pp vs sequential, SIGNAL-parowy−) while still exceeding joint training by +3.9pp (L2). The two resource regimes — from-scratch random features vs foundation embeddings — are kept as explicitly separate axes; no from-scratch number is ever compared against a foundation number.

## 17. Long horizon: 100 classes, 20 tasks (Series M, v0.8)

Over 20 CIFAR-100 tasks on the pretrained backbone, the mechanism holds 40.70 ± 0.84% — **85.8% of its frozen-feature ceiling** (47.41 ± 0.49), down from 96.7% at T=5. A decomposition attributes ~79% of the raw per-task accuracy drop to the protocol's growing label space (the all-data ceiling drops identically), leaving a real late-task deficit of −7.8pp vs ceiling. The deficit is structural, not implementational: the per-old-class dream budget is a measured stability–plasticity knob (three points on the front: 51/16/~5 dreams per old class; M1/M1b/M1c), and rich per-class rehearsal remains ACC-optimal. Anchor geometry gave its first clean SIGNAL+ here: at 100 classes, 300d word anchors beat 50d by +7.71pp (5/5 seeds) — the word-space dimension must scale with the class count.

## 18. Selective forgetting with a guarantee (Series N, v0.9)

Forgetting on demand, measured at two levels. *Functional*: deleting a class's anchor, pod, and statistics removes access with no collateral damage to remaining classes (10× removal matrix; N1). *Informational*: how much does re-learning from 100 images recover? Deleting the entries alone ("light") removes access but **not** information — the class is restored essentially bit-identically, proving the projection is the sole information carrier (pods are confirmatory). Fine-tuning the projection on remaining-class dreams ("scrub") erases ~84% of recoverability. Re-initializing the projection and rebuilding it from dreams of the remaining classes ("reinit") erases **100%**: re-learning the deleted class matches learning a never-seen class (+0.34pp, below the 0.50 threshold) at ≤ zero cost to remaining classes (N1b/N1c). A corollary with protocol weight: a class the projection was never trained on is routing-unreachable even when its anchor and pod are present.

## 19. Dream consolidation falsified (Series O, v0.10)

An N1c side observation suggested a "deep sleep": rebuild the projection from dreams of *all* classes after the sequence, hoping drift-free joint training on dreams beats the sequential path. Pre-registered and run on both benchmarks, it is **paired-negative on both** (Fashion −0.62pp, CIFAR −0.80pp, SIGNAL-parowy−; the fine-tune variant is noise). Dreams are a density model, not data: real features at learning time beat drift-free joint training on dreamed ones. This closes the sleep-function inventory — sleep protects (rehearsal), transfers (protocol), and rebuilds (N1c/I4 repair), but does not improve on reality.

## 20. The untrusted collective (Series I4, v0.11)

What if a payload is poisoned? A label-swap attack (statistics of class 8 sent under the name of class 9 and vice versa) destroys both co-adopted classes (−77.6pp and −94.4pp — contradictory anchor targets inside one adoption), while the recipient's own classes stay robust (±1pp — their own sleep defends them). Neither pre-registered detector (routing-rank consistency, canary probe) separates attacks from honest payloads on the random backbone — an honest negative; semantic detection likely needs pretrained features. The attack is, however, **fully reversible**: forget-and-readopt of the whole adoption batch returns the system to the clean path within noise on all metrics (I4b) — consistent with N1b, since light unlearning removes access and re-adoption overwrites the mapping. Practical protocol policy: adopt in batches, retain the ability to revoke a source's batch, and make repair scope cover damage scope.

## 21. Payload detection: a structural gate and a distance law (Series P, v1.1)

Series I4 left detection as an honest negative with an explicit candidate: semantic (pretrained) features. Series P falsified that candidate in its simple form — on ResNet18 features neither pre-registered detector separates clean payloads from both attacks (P1, double negative, with a paired random-backbone control) — and then converted the negative into two pre-registered positive results on fresh seeds (P1c). **The structural gate**: the rank-consistency detector with a threshold frozen in advance (0.45) classifies 60/60 clean/structureless payloads correctly on *both* backbones — a representation-independent entry gate that rejects payloads without class structure. **The distance law**: swap detectability grows with the semantic distance of the forged pair — full separation for donor classes at anchor-cosine 0.487 and 0.139, none at 0.775 (nor at 0.615 in P1), on both backbones. Close-pair swaps are undetectable *by construction*: the donor's payload satisfies the declared anchor's feature-to-word ranking, which retroactively explains the I4 and P1 negatives (both swapped the ship↔truck pair, cos 0.615) as the one structurally invisible case rather than a detector failure. The measured protocol policy is now complete: adopt in batches with revocable sources (I4); gate every payload structurally at entry — this also catches mid/far-distance swaps (P1c); accept that close-pair swaps are invisible and cover them with the fully-measured forget-and-readopt repair (I4b). A secondary observation (4–5/5 seeds, both backbones, not claimed): honest adoption costs the recipient's own classes *more* than a swapped one — the canary detector's pre-registered direction was inverted.

## 22. The scale barrier and the dream budget (Series Q, v1.2)

Does the exchange protocol survive 100 classes? At the default adoption dream budget (500/class), it does not: a 20-agent collective — the collector learns 5 classes locally and adopts 95 from 24.1 KB messages — pays −6.67 ± 0.88pp versus the sequential agent (34.02 vs 40.70, SIGNAL−, 5/5), a ~12× growth of the −0.56pp protocol cost measured at 10 classes (Q1). The structure of the cost is the opposite of the sequential agent's: it is *front-loaded*. Early adoptions reach ~37% per-task accuracy where early learning reaches ~78%, while late adoptions match late learning and nominally beat it relative to the per-task ceiling (0.913 vs 0.870) — and collective forgetting is *lower* (13.0 vs 18.3pp): adopted classes hold, they just start low (Q1b). Two pre-registered levers then dismantle the barrier. Forget-and-readopt of the five earliest batches in the now-mature projection recovers 61% of it (+4.09 ± 0.65pp; early tasks 26%→42%), confirming the immature-projection hypothesis (Q2a). Raising the adoption dream budget to 2500/class — payload unchanged — recovers 154% (+10.26 ± 0.49pp, to 44.29 ± 0.66): the barrier was a dream-budget shortage, not an information shortage in the 24 KB message (Q2b). The resulting nominal +3.59pp advantage over the sequential agent demanded a fairness control, pre-registered before running: give the sequential agent the same budget by augmenting each new task from its own first-generation statistics (500 real + 2000 dreamed). The control cut both ways — **self-dream augmentation is a lever for any agent** (+4.66 ± 0.52pp, lifting the single agent to 45.35 ± 0.49, 95.7% of the all-data ceiling — the project's best at 100 classes), and at symmetric budgets the collective advantage dissolves into **equivalence** (44.29 vs 45.35, paired −1.07 ± 0.64pp, below both thresholds; Q2c). The honest headline: the scale barrier was a budget artifact, and the collective-equivalence claim of Series I and L extends from 10 classes and 5 agents to 100 classes and 20 agents, with 95 of 100 classes learned from messages alone. A final combination run confirmed the levers are not additive — at budget 2500 early adoptions reach ~49.5% before any repair (vs 26.4% at budget 500), and readoption adds nothing beyond noise overall (Q2d) — so the recommended protocol configuration is the budget alone, with forget-and-readopt reserved for its original role: repair after adversarial payloads.

## 23. Limitations and roadmap

**Honest limitations.**
1. Part II results (routing ceiling, feature lever) are on MNIST-scale benchmarks; Part III extends the *continual-learning* claims to Split-CIFAR-10, but Part II's ceiling characterization has not been replicated on natural images.
2. MAC is a proxy that **does not predict GPU wall-time** (E4): a monolithic MLP with 0.6× the MAC of the slim CNN runs 16× faster on CUDA (large matmuls: ~50% of peak; small convolutions: ~3%). Time-per-10k-samples at saturated utilization was used as an energy proxy (the GTX 1050 Ti does not expose power telemetry); absolute joules require instrumented hardware. The efficiency claims of small modular networks belong on hardware that honors MAC (CPU SIMD with in-cache pods, NPUs), not desktop GPUs.
3. Part III's absolute accuracy is capped by random frozen features (CIFAR: 37.5% vs a 70.2% joint ceiling). The mechanism is representation-agnostic; stronger frozen backbones (self-supervised or pretrained embeddings) are the measured road up and an explicit fork in the project's identity (from-scratch vs foundation-embedding memory layer).
4. The replay comparison fixes the buffer at 200 samples (pre-registered); buffer size is an unswept axis and larger buffers would favor replay. The semantic mechanism requires class names with visual semantics (MNIST is the measured counterexample).
5. Ternary quantization of pods does not transfer to the v2 slim stack (−2.7 ± 2.8pp; architecture-specific, see Section 8) — and the "≤ noise threshold" verdict criterion itself proved gameable by high variance, motivating an added per-seed worst-case condition in later experiment plans.
6. An engineering-audit pass (Series J, pre-registered) tested two pipeline concerns and cleared both: calibrating the frozen backbone's BatchNorm statistics and per-dimension feature scale normalization are NOISE (the latter negative on all 5 Fashion seeds), and per-channel CIFAR input normalization is ~neutral for the frozen random backbone. The dead-BatchNorm oversight was real but innocent — reported levels are properties of the mechanism, not preprocessing artifacts.

**Roadmap status (v1.0): executed.** The roadmap drafted after Series H — OWM on the semantic projection, sparse dream models, stronger frozen representations, attribute vocabularies with code distance — has been run to completion and is chronicled in Sections 14–20: Series J closed the dream-fidelity thread (spike-and-slab is the default sleep model), K measured the ceilings and eliminated OWM under the sparse sleep, I turned the memory layer into a communication protocol, L executed the representation fork, M stress-tested the horizon, N delivered forgetting with a guarantee, O falsified dream consolidation, and I4 mapped the untrusted-collective failure modes. Open items now closed: payload-attack detection on semantic features (Series P — the I4 negative's candidate falsified, but a structural gate and a distance law survive as pre-registered successes), the collective at long horizon (Series Q — the scale barrier is a dream-budget artifact; collective equivalence extends to 100 classes / 20 agents), and attribute vocabularies with error-correcting code distance (Series G2b — code distance backfires on random features; the bottleneck is isolated to the representation). Now also closed: compositional zero-shot on pretrained features (Series G3 — strong ImageNet features are statistically equivalent to random ones, so the limit is the hand-dictionary *paradigm*, not the representation; the routing/zero-shot axis is exhausted), and the heterogeneous collective (Series R — the shared-word interlingua is falsified, but a two-class linear feature alignment recovers 92% of the homogeneous ceiling across different backbones on R-mild). Now also closed at R-hard: the heterogeneous collective across *genuinely different* backbones with differing feature dimensions (from-scratch raw-pixel CNN ↔ frozen ResNet18, rectangular 512↔128 alignment) recovers 76% of the receiver's own ceiling at a larger, measured cost (−13.55pp) than R-mild, with the calibration curve rising in K as pre-registered — representation-agnostic transfer holds even without a shared feature basis. Vector-symbolic binding as a dense-matmul pod replacement remains unexplored. LSH/Morton/TMU routing is retained as a deployment demonstration — routing cost is measured at ~0.1% of inference and is not a bottleneck.

## 24. Conclusion

Modular neural systems are often motivated by intuition and validated by a single benchmark run. We offer a different template: pre-registered verdicts, paired multi-seed statistics, and equal publication weight for negative results. Under this discipline, a coherent picture emerged across two studies. In Part II: the router is the bottleneck; the bottleneck is representational, not algorithmic — three independent channels confirm it; better features raise the ceiling efficiently; and the standard oracle metric systematically inflates routing headroom. In Part III: the recurring design law — a narrowly-supervised shared representation is worse than none — turns out to be the mechanism of catastrophic forgetting itself, and taking it as an axiom (immutable representation, plasticity only in prototypes and pods) allows episodic memory to be replaced by semantic priors and parametric sleep: replay-level accuracy with zero stored samples on Split-Fashion, and dominance over replay on Split-CIFAR, where trainable representations drift. The follow-up series then showed the same memory layer is more than a buffer replacement: it is a measured fraction of its representation's ceiling (94.6–97.6%), a privacy-preserving communication protocol between agents, portable without modification to a foundation backbone that then beats joint training, capable of forgetting with a guarantee, and repairable after adversarial payloads — while its one tempting free lunch, consolidation from dreams alone, was falsified and reported as such. The boundaries of every claim are measured and reported — which classes of data the mechanism needs, where its accuracy is capped, and which of our own hypotheses failed their pre-registered thresholds. That, we believe, is what an evidence base for modular, continually-learning systems should look like.

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
python src/run_J4_glove300.py             # GloVe-300d -> seq null; all-data ceiling 81.16
# Series K -- ceilings and lever composition (v0.5)
python src/run_K0_cifar_ceiling.py        # frozen-feature ceiling CIFAR: 39.65 (mechanism at 94.6%)
python src/run_K1_sparse300.py            # sparse x GloVe-300d -> paired-SIGNAL+ Fashion: 79.23
python src/run_K2_owm_sparse.py           # OWM x sparse sleep -> eliminated (paired-SIGNAL- CIFAR)
# Series I -- collective learning by dream exchange (v0.6)
python src/run_I1_transplant.py           # class from a 24 KB message: -1.29pp vs local training
python src/run_I2_fusion.py               # payload saturates at half the data -> fusion moot
python src/run_I3_collective.py           # 5 agents, 0 images exchanged ~ 1 sequential agent
python src/run_I2b_fusion_lowdata.py      # fusion helps iff payload unsaturated (paired-SIGNAL+ at n=100)
# Series L -- identity fork: pretrained frozen backbone (v0.7)
python src/run_L1_pretrained.py           # CIFAR: 74.69 (+37.2pp, SIGNAL+); mechanism at 96.7% of new ceiling
python src/run_L2_collective_cifar.py     # collective on strong features: 74.13 (> trainable joint 70.24)
# Series M -- long horizon: CIFAR-100, 20 tasks (v0.8)
python src/run_M1_long_horizon.py         # 40.70 @T=20 (85.8% of ceiling); anchors 300d SIGNAL+ (+7.7pp)
python src/run_M1b_balanced_dreams.py     # dream-budget knob, low end -> plasticity up, stability down
python src/run_M1c_mid_budget.py          # mid point -> front is sharp; per-class rehearsal stays default
# Series N -- selective forgetting with a guarantee (v0.9)
python src/run_N1_unlearning.py           # functional level: 10x removal matrix, no collateral damage
python src/run_N1b_relearn_balanced.py    # information level: light keeps 100%, scrub erases ~84%
python src/run_N1c_reinit.py              # projection re-init: full erasure guarantee at <= zero cost
# Series O -- dream consolidation (v0.10): FALSIFIED
python src/run_O1_consolidation.py        # deep-sleep rebuild: paired-SIGNAL- on both benchmarks
# Series I4 -- untrusted collective (v0.11)
python src/run_I4_untrusted.py            # damage map; detection fails on random backbone (negative)
python src/run_I4b_full_repair.py         # forget-and-readopt the batch: full recovery (noise on all metrics)
# Series P -- payload detection (v1.1)
python src/run_P1_detect_pretrained.py    # pretrained does not rescue detectors (double negative)
python src/run_P1c_gate_distance.py       # structural gate 60/60 (SUCCESS); distance law (STRONG SUCCESS)
# Series Q -- collective at long horizon (v1.2)
python src/run_Q1_collective_horizon.py   # scale barrier: -6.67pp at default budget (SIGNAL-)
python src/run_Q2_early_repair.py         # readopt: 61% recovered; budget 2500: 154% (both SIGNAL+)
python src/run_Q2c_seq_selfdream.py       # fairness control: self-dream lever (+4.66); EQUIVALENCE
```

Working notes with full result tables: `DROGA_D_NOTATKI.md`, `DROGA_E_NOTATKI.md`, `DROGA_F_NOTATKI.md`, `DROGA_G_NOTATKI.md`, `DROGA_H_NOTATKI.md`, `DROGA_J_NOTATKI.md`, `DROGA_K_NOTATKI.md`, `DROGA_I_NOTATKI.md`, `DROGA_L_NOTATKI.md`, `DROGA_M_NOTATKI.md`, `DROGA_N_NOTATKI.md`, `DROGA_O_NOTATKI.md`, `DROGA_I4_NOTATKI.md`. Pre-registered plans: `D6B_PLAN.md`, `D7_PLAN.md`, `DROGA_F_PLAN.md`, `DROGA_G_PLAN.md`, `DROGA_H_PLAN.md`, `DROGA_J_PLAN.md`, `DROGA_K_PLAN.md`, `DROGA_I_PLAN.md`, `DROGA_L_PLAN.md`; stage map: `PLAN_GENERALNY.md`. Glossary: `SLOWNIK_POJEC.md`. Word vectors: GloVe 6B-50d (public).

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
| Frozen-feature ceiling (CIFAR) | 39.65 ± 1.21% | K0: sequential mechanism at 94.6% of it |
| Split-Fashion sparse × 300d | 79.23 ± 0.73% | paired-SIGNAL+; 97.6% of the 81.16 ceiling (K1) |
| Class transplant (24 KB message) | 94.26% vs 95.55% local | −1.29pp; full-ACC equivalence (I1) |
| Collective N=5, dream exchange | 78.87 ± 1.01% | ≈ sequential 79.23; 8/10 classes from dreams (I3) |
| Pretrained-fork CIFAR (seq) | 74.69 ± 0.69% | +37.2pp vs random backbone; 96.7% of 77.23 ceiling; beats trainable joint 70.24 (L1) |
| Collective N=5 on CIFAR (pretrained) | 74.13 ± 0.57% | paired −0.56pp vs sequential — first measured protocol cost; > joint by +3.9pp (L2) |
| Message saturation (I2b) | 91.8/93.9/94.4% at 100/500/3000 img | fusion paired-SIGNAL+ only below saturation |
| Long horizon T=20 (CIFAR-100) | 40.70 ± 0.84% | 85.8% of frozen ceiling (vs 96.7% @T=5); late-task deficit −7.8pp vs ceiling (M1) |
| Anchor-dimension scaling | +7.71pp (SIGNAL+, 5/5) | 300d vs 50d at 100 classes — word-space must grow with class count (M1) |
| Stability–plasticity front | 3 points measured | single knob (dreams/old class: 51/16/5); per-class rehearsal is ACC-optimal (M1b/M1c) |
| Selective forgetting (light/scrub/reinit) | 0% / ~84% / 100% erased | re-learn from 100 img: identical / 11.2% / = never-seen (N1) |
| Full-erasure guarantee | reinit ≈ never-seen (+0.34pp, thr 0.50) | at ≤ zero cost to remaining classes (N1c) |
| Unreachability of untaught classes | exact 0.00, 5/5 seeds | anchor+pod present, projection never trained on it (N1b) |
| Dream consolidation (deep sleep) | falsified | rebuild-from-dreams paired−0.6/−0.8pp on both benchmarks; dreams protect/transfer/rebuild but do not improve on real data (O1) |
| Poisoned-payload blast radius | adoption batch only | own classes robust (±1pp); label-swap destroys both co-adopted classes (I4) |
| Payload-attack detection (random backbone) | no separation (negative) | both pre-registered detectors overlap; candidate: semantics of pretrained features |
| Attack recovery | full (noise on all metrics) | forget-and-readopt the batch — repair scope must cover damage scope (I4b) |
| Payload detection on pretrained features | no separation (double negative) | semantic features alone do not rescue either detector; paired random-backbone control (P1) |
| Structural gate (D1 > 0.45) | 60/60 correct, both backbones | rejects structureless payloads and mid/far swaps; passes only close-pair swaps (P1c-a) |
| Swap-detectability distance law | full separation at cos 0.487/0.139; none at 0.775/0.615 | close-pair swaps undetectable by construction — explains the I4/P1 negatives; threshold between cos 0.615 and 0.487 (P1c-b) |
| Scale barrier (default budget) | −6.67 ± 0.88pp at 100 classes | vs −0.56pp at 10 classes; front-loaded in early adoptions (Q1/Q1b) |
| Barrier repair | 61% (readopt) / 154% (budget 2500) | forget-and-readopt early batches; dream budget was the shortage, not the 24 KB message (Q2a/Q2b) |
| Self-dream augmentation | +4.66pp (SIGNAL+, 5/5) | 500 real + 2000 dreamed per new class; best single agent: 45.35 = 95.7% of ceiling (Q2c) |
| Collective at 100 classes, symmetric budgets | 44.29 vs 45.35 (equivalence) | 20 agents, 95/100 classes from 24 KB messages; extends I3/L2 equivalence 10× in scale (Q2c) |
| Inference cost vs task count | ×1.0007 after 5 tasks | constant-MAC thesis |
| Design-law instances | 4 | D5, E2-v1, F1, F2 (narrow supervision < none) |

---

*M.A.R.S. — Modular Autonomous Refinement System. PyTorch + WebGPU. All numbers measured, including the ones that said no.*
