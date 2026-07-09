# M.A.R.S. — Seria B: Przełamanie barier — WYNIKI

Data: 2026-06-16
GPU: NVIDIA GTX 1050 Ti (4GB, Pascal)

---

## Główne odkrycie: routing 89% → 92.5% dzięki CNN

**Jeden kluczowy insight**: ProtoRouter (MLP na płaskim 784-dim wektorze) NIE WIDZI
struktury przestrzennej obrazów. F-MNIST ma klasy różniące się lokalnie
(kołnierzyk, rękawy, wzory) — to wymaga konwolucji.

CNN encoder z 2 warstwami conv (32→64 kanałów) + prototype matching daje:
- **Routing: 93.0%** (vs 89.0% MLP → **+4pp**)
- **System: 92.55%** (vs 89.03% → **+3.52pp**)
- **Luka do ORACLE: 5.83pp** (z 9.31pp → **zmniejszona o 37%**)

---

## Pełna tabela wyników serii B (Fashion-MNIST)

| Eksperyment | Technika | Routing | System | Delta vs A9 |
|---|---|---|---|---|
| **A9 (baseline)** | ProtoRouter enc_h=256, emb=32 | 88.98% | 89.03% | — |
| B1 | DeepProtoRouter (2L encoder + BN) | 89.8% | 89.78% | +0.75pp |
| B1b CNN(16,32) | CNN encoder 16→32 ch | 92.3% | 91.8% | +2.77pp |
| **B1b CNN(32,64)** | **CNN encoder 32→64 ch** | **93.0%** | **92.55%** | **+3.52pp** |
| B1b LightCNN(16,32) | Lekki CNN (stride conv) | 90.3% | 90.2% | +1.17pp |
| B2 aug_router | Augmentacja na routerze | 86.5% | 86.6% | **-2.43pp** |
| B2 cosine_sched | CosineAnnealingLR | 89.1% | 89.1% | +0.07pp |
| B4 (all configs) | Głębsze pody 24→128, 2L | ~89.3% | ~89.4% | ≈0pp |
| B5 (all ratios) | own_ratio 0.3–1.0 | — | 88.6-89.3% | ≈0pp |
| B6 REINFORCE | Joint fine-tuning | 89.5% | 89.53% | +0.50pp |

---

## Kluczowe wnioski diagnostyczne

### 1. Bottleneck = routing (nie pody)
- ORACLE ≈ 98.5% dla KAŻDEGO rozmiaru podów (h=24 do h=128, 1L i 2L)
- System accuracy = routing accuracy (±0.1pp) — pody NIE kompensują błędów
- own_ratio nie wpływa na system acc (0.7 jest optymalne)
- **Wniosek: jedyna droga to lepszy routing**

### 2. MLP encoder na flat pixels = sufit ~89%
- Głębszy MLP (2L, BatchNorm, Dropout, cosine sim) → +0.7pp max
- Augmentacja SZKODZI routerowi (szum w embeddingach → gorsze prototypy)
- Problem NIE jest w capacity encodera, lecz w braku spatial features

### 3. CNN encoder przełamuje barierę
- Convolutions wyłapują lokalne wzory (kołnierzyk, guziki, faktura)
- CNN(32,64): 93.0% routing — to prostsze do nauki niż "T-shirt vs Shirt" na pikselach
- Trade-off: MAC 1.2M vs 209K (5.7× droższy router), ale gap ↓ 37%
- LightCNN(8,16): 89.7% z ~96K MAC — tańszy od MLP i lepszy

### 4. Joint fine-tuning pomaga marginalnie (+0.45pp)
- Gumbel-Softmax i REINFORCE oba dają małą poprawę
- To dlatego, że pody są silne (ORACLE 98.5%) — problem to routing, nie pods

### 5. Augmentacja: pomaga PODOM, szkodzi ROUTEROWI
- Augmented pods: ORACLE 98.7% (vs 98.3%) — lepsze
- Augmented router: routing 86.5% (vs 89.0%) — gorsze
- Implikacja: router potrzebuje CZYSTYCH, konsystentnych embeddings

---

## Zalecenia na dalszą pracę

### Priorytet 1: CNN router jako domyślny dla obrazów
- Zintegrować `CNNProtoRouter` z `SpecialistSystem`
- Dobrać optimal punkt MAC/acc (LightCNN dla mobile, full CNN dla serwera)
- Próba: CNN router + joint fine-tuning (potencjał: 93-94%)

### Priorytet 2: Ternary weights na podach
- Pody (h=24) mają mało parametrów — ternary kwantyzacja powinna dać
  ~16× kompresję z <1pp accuracy drop
- Integracja z FastPods (stacked tensory ternary)

### Priorytet 3: Hierarchiczny routing
- Zamiast 10 podów (1/klasę), grupuj confusable klasy:
  - {T-shirt, Pullover, Coat, Shirt} → 1 pod "upper-body"
  - {Sandal, Sneaker, Boot} → 1 pod "footwear"
- Router łatwiej rozróżnia kategorie → mniej błędów → specialist per category

### Priorytet 4: WebGPU deployment CNN routera
- CNN na teksturach GPU = naturalne (konwolucja to texel fetch + multiply)
- WGSL compute shader z conv2d → potencjalnie szybszy niż MLP

---

---

## B8: Ternary Kwantyzacja — 16× kompresja za darmo

| Dataset | Full | Ternary (t=0.5) | Drop | Sparsity | Kompresja |
|---|---|---|---|---|---|
| MNIST | 97.98% | **98.17%** (+0.19pp!) | 0pp | 29.5% | 16× |
| F-MNIST | 88.85% | **88.77%** | -0.08pp | 29.1% | 16× |

**KLUCZOWE**: threshold=0.5 daje ternary pody BEZ straty accuracy!
Kwantyzacja działa jak regularyzacja (usuwa drobne szumowe wagi).

- Kwantyzacja routera: KATASTROFALNA (-18pp). Router MUSI być full precision.
- Kwantyzacja podów: DARMOWA z threshold=0.5.
- Throughput: 1.08× na GPU (ograniczone brakiem dedykowanego ternary hardware).

Implikacja: pody mogą być 16× mniejsze w pamięci bez żadnego kosztu jakości.
Na dedykowanym sprzęcie (NPU, FPGA) ternary daje 10-32× speedup operacji.

---

## Pliki wynikowe

- `results/B1_deep_router_fashion.json` — DeepProtoRouter sweep
- `results/B1b_cnn_router_fashion.json` — CNN router sweep  
- `results/B2_augmentation_fashion.json` — augmentacja sweep
- `results/B4_deeper_pods_fashion.json` — deeper pods sweep
- `results/B5_own_ratio_fashion.json` — own_ratio sweep
- `results/B6_joint_finetune_fashion.json` — joint training sweep
- `results/B8_ternary.json` — ternary kwantyzacja (oba datasety)

## Pliki źródłowe (nowe)

- `src/routers_v3.py` — DeepProtoRouter (BN, cosine, temperature)
- `src/run_B1_deep_router.py` — B1 sweep
- `src/run_B1b_cnn_router.py` — B1b CNN router (BREAKTHROUGH)
- `src/run_B2_augmentation.py` — B2 augmentacja
- `src/run_B4_deeper_pods.py` — B4 głębsze pody
- `src/run_B5_own_ratio.py` — B5 own_ratio
- `src/run_B6_joint_finetune.py` — B6 joint fine-tuning
- `src/run_B8_ternary.py` — B8 ternary kwantyzacja
