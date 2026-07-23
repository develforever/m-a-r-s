# M.A.R.S. — Claims register (v1.0)

Every substantive claim of the project, with its series, pre-registered
verdict, and the per-seed results file that backs it. Negative and falsified
claims are listed with equal weight — they are results. All numbers
re-verified against `results/*.json` on 2026-07-23
(`scripts/audit_headline_numbers.py`: zero discrepancies).

Verdict vocabulary: SIGNAL+/− (paired delta beyond noise threshold
std(base)+std(variant), min per-seed same sign), NOISE (within threshold),
paired-SIGNAL (all 5 per-seed deltas one sign AND |mean| > 2×std of deltas —
reported alongside, never instead).

## Part II — the routing ceiling (MNIST-scale, random/trained MLP + pods)

| # | Claim | Series | Verdict | Results |
|---|---|---|---|---|
| 1 | Routing cannot be improved by the routing *algorithm*: ensemble consultation +0.00pp | D4 | NOISE | `D4_consultation.json` |
| 2 | Distillation from pods to router *hurts* (−2.5pp) | D5 | SIGNAL− | `D5_distillation.json` |
| 3 | Predictive-coding reconstruction error +0.05pp | D7 | NOISE | `D7_predictive_coding.json` |
| 4 | Features are the lever: small CNN backbone +2.38pp system acc, gain flows through routing | D6 | SIGNAL+ | `D6_cnn_backbone.json` |
| 5 | Slim CNN retains 70% of the gain at 9% of the added cost | D6b | SIGNAL+ | `D6b_slim_cnn.json` |
| 6 | Oracle inflation: real routing headroom ~0.5pp, not the apparent 6pp (oracle leaks labels) | E2 | measured | `E2_hierarchical.json` |
| 7 | MAC does not predict GPU wall-time (16× inversion); efficiency claims belong on MAC-honoring hardware | E4 | measured | `E4_energy.json` |
| 8 | Ternary pod quantization does not transfer to the slim stack (−2.7 ± 2.8pp) | B8 | negative (gameable criterion noted) | `B8_ternary.json` |

## Part III — memory without data (class-IL, 5 seeds, frozen random backbone unless noted)

| # | Claim | Series | Verdict | Results |
|---|---|---|---|---|
| 9 | Design law: narrowly-supervised shared representation < none (4 independent instances: D5, E2-v1, F1, F2) | D/E/F | recurring | see files above + `F1_mars_cl.json`, `F2_frozen_features.json` |
| 10 | Split-Fashion: anchors + parametric sleep ≈ replay-200 (77.57 ± 1.02 vs 76.97 ± 1.09), 0 stored samples, ~30% less forgetting | F3b/H1b | equivalence (nominally above, within threshold) | `H1b_dream_fidelity.json`, `F0_cl_baselines.json` |
| 11 | Mechanism requires visually-semantic class names: Split-MNIST loses to replay by ~19pp | F/G1 | measured boundary | `G1_semantic.json`, `F0_cl_baselines.json` |
| 12 | Split-CIFAR inverts the ranking: replay collapses (14.03 ± 4.93 normalized), MARS stable | F4/J2 | SIGNAL+ | `J2_cifar_normalized.json`, `F4_split_cifar.json` |
| 13 | Sparse (spike-and-slab) dreams beat diagonal on CIFAR: +4.48pp, 5/5 seeds; effect grows with data difficulty | J2b/J3 | SIGNAL+ (CIFAR), NOISE (Fashion +0.91) | `J2b_cifar_sparse.json`, `J3_sparse_dreams.json` |
| 14 | Pipeline audit clean: BN-calibration, σ-normalization, input normalization are NOISE for the frozen backbone | J1/J2 | NOISE (cleared) | `J1_feature_conditioning.json` |
| 15 | Compositional zero-shot from attributes fails (3.2% vs 30% threshold); reachability rule predicts failures 3/3 | G2 | negative with structure | `G2_compositional.json` |
| 16 | OWM is eliminated under the sparse sleep (MNIST gain vanishes; CIFAR paired-negative) | K2 (vs H1) | eliminated | `K2_owm_sparse.json`, `H1_owm.json` |
| 17 | Measured ceilings: frozen random features admit 81.16 ± 0.87 (Fashion, 300d) / 39.65 ± 1.21 (CIFAR); sequential mechanism realizes 97.6% / 94.6% | K0/K1/J4 | measured | `K0_cifar_ceiling.json`, `J4_glove300.json`, `K1_sparse300.json` |
| 18 | Lever composition (sparse × 300d) gives Fashion best 79.23 ± 0.73 | K1 | paired-SIGNAL+ | `K1_sparse300.json` |

## The protocol: sleep learns, shares, forgets (Series I → I4)

| # | Claim | Series | Verdict | Results |
|---|---|---|---|---|
| 19 | A class is learnable from one ~24 KB message: −1.29pp vs local training on 12k images | I1 | success (weak threshold <3pp; ACC-level SZUM) | `I1_transplant.json` |
| 20 | 5 agents × 2 classes, zero images exchanged ≈ 1 sequential agent (78.87 vs 79.23) | I3 | equivalence (SZUM) | `I3_collective.json` |
| 21 | Fusion of partial views helps iff payload unsaturated (SIGNAL+ at 100 img/class; noise at 500+; saturation between 500 and 3000) | I2/I2b | paired-SIGNAL+ / NOISE | `I2b_fusion_lowdata.json`, `I2_fusion.json` |
| 22 | Identity fork: frozen ImageNet ResNet18, mechanism unchanged → CIFAR 74.69 ± 0.69 (+37.2pp), 96.7% of new ceiling 77.23, beats trainable joint 70.24 | L1 | SIGNAL+ | `L1_pretrained.json` |
| 23 | Collective transfers to strong features at measured cost −0.56pp vs sequential; still > joint by +3.9pp | L2 | paired-SIGNAL− (cost) | `L2_collective_cifar.json` |
| 24 | Long horizon (100 classes, 20 tasks): 40.70 ± 0.84 = 85.8% of ceiling; late-task deficit −7.8pp is structural (dream budget = stability–plasticity knob, 3 points) | M1/M1b/M1c | measured front | `M1_long_horizon.json`, `M1b_balanced_dreams.json`, `M1c_mid_budget.json` |
| 25 | Anchor dimension must scale with class count: 300d > 50d by +7.71pp at 100 classes, 5/5 seeds | M1 | SIGNAL+ | `M1_long_horizon.json` |
| 26 | Deleting class entries removes access, not information (restored near-bit-identically from 100 images); projection is the sole information carrier | N1/N1b | measured | `N1_unlearning.json`, `N1b_relearn_balanced.json` |
| 27 | Erasure taxonomy: light 0% / scrub ~84% / reinit 100% of recoverability erased | N1b/N1c | measured | `N1b_relearn_balanced.json`, `N1c_reinit.json` |
| 28 | Full-erasure guarantee: reinit ≈ never-seen (+0.34pp < 0.50 threshold) at ≤ zero cost to remaining classes | N1c | PEŁNA GWARANCJA | `N1c_reinit.json` |
| 29 | Untaught classes are routing-unreachable (exact 0.00, 5/5) even with anchor+pod present | N1b | measured | `N1b_relearn_balanced.json` |
| 30 | Dream consolidation ("deep sleep" rebuild after the sequence) does NOT improve on reality: paired −0.62pp (Fashion) / −0.80pp (CIFAR) | O1 | FALSIFIED (paired-SIGNAL−) | `O1_consolidation.json` |
| 31 | Poisoned payload (label swap) destroys both co-adopted classes (−77.6 / −94.4pp); recipient's own classes robust (±1pp) | I4 | SIGNAL− (damage map) | `I4_untrusted.json` |
| 32 | Attack detection on the random backbone fails: neither pre-registered detector separates attack from honest payloads | I4 | negative (honest); candidate: pretrained features | `I4_untrusted.json` |
| 32b | The candidate is falsified: pretrained (semantic) features do NOT rescue either detector — no clean-vs-both-attacks separation on ResNet18 features (paired control on random backbone included). Post-hoc observations (not claims): D1 fully separates *structureless* (noise) payloads on both backbones; swap between semantically close classes is invisible to D1 | P1 | double NEGATIVE (pre-registered) | `P1_detect_pretrained.json` |
| 32c | Structural gate: D1 with a pre-registered threshold (0.45) rejects structureless payloads with 60/60 correct clean/noise decisions on both backbones (fresh seeds 5–9); it also catches mid/far-distance swaps, passing only close-pair swaps | P1c-a | SUCCESS (pre-registered) | `P1c_gate_distance.json` |
| 32d | Distance law: swap detectability grows with the semantic distance of the forged pair — full separation for donors at anchor-cos 0.487 and 0.139, none at 0.775 (and 0.615 in P1), on both backbones; close-pair swaps are undetectable *by construction* (the donor payload satisfies the declared anchor's ranking); detectability threshold lies between cos 0.615 and 0.487 | P1c-b | STRONG SUCCESS (pre-registered) | `P1c_gate_distance.json` |
| 33 | Attack is fully reversible: forget-and-readopt the batch returns to the clean path (noise on all metrics) | I4b | full repair (SZUM everywhere) | `I4b_full_repair.json` |

## Constant-cost thesis

| # | Claim | Series | Verdict | Results |
|---|---|---|---|---|
| 34 | Inference cost vs task count: ×1.0007 after 5 tasks (constant MAC) | F | measured | `F3b_drift_control.json` |

Full tables and verdict derivations: `DROGA_*_NOTATKI.md`. Pre-registered
plans (written before runs, dates in git history): `DROGA_*_PLAN.md`,
`D6B_PLAN.md`, `D7_PLAN.md`. Stage maps: `PLAN_GENERALNY.md` (executed),
`PLAN_V1.md` (current).
