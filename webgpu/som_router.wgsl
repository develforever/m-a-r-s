// ═══════════════════════════════════════════════════════════════════════════════
// M.A.R.S. — SOM-Router Compute Shader (WebGPU / WGSL)
//
// ARCHITEKTURA:
//   Krok 1: Projekcja liniowa (input_vec × projection_matrix → uv)
//           Koszt: 2×N_IN MAC (znikomy, np. 2×784 = 1568 MAC dla MNIST)
//   Krok 2: textureSampleLevel (uv → capsule_id)
//           Koszt: 0 MAC (sprzętowy TMU fetch, 1 cykl zegara)
//   Krok 3: Wynik: capsule_id do wybudzenia
//
// PORÓWNANIE:
//   Neural Router:   N_IN × HIDDEN + HIDDEN × N_PODS MAC (np. 784×64 + 64×10 = 50816 MAC)
//   SOM-Router TMU:  2×N_IN MAC (projekcja) + 0 MAC (TMU) = 1568 MAC
//   Oszczędność:     ~97% redukcji MAC dla MNIST
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Bindings ───────────────────────────────────────────────────────────────

// Group 0: Dane wejściowe i wyniki
@group(0) @binding(0) var<storage, read> input_vectors: array<f32>;     // [batch × n_in]
@group(0) @binding(1) var<storage, read_write> output_capsules: array<u32>; // [batch]
@group(0) @binding(2) var<storage, read_write> output_confidence: array<f32>; // [batch]

// Group 1: Parametry routera
@group(1) @binding(0) var<uniform> params: RouterParams;
@group(1) @binding(1) var<storage, read> projection_matrix: array<f32>; // [n_in × 2] — projekcja do UV
@group(1) @binding(2) var<storage, read> projection_bias: array<f32>;   // [2]

// Group 2: Tekstura SOM (label map)
@group(2) @binding(0) var som_texture: texture_2d<f32>;   // RGBA: R=capsule_id, G=confidence, B=unused, A=unused
@group(2) @binding(1) var som_sampler: sampler;           // bilinear sampler (TMU hardware)

struct RouterParams {
    batch_size: u32,
    n_in: u32,
    grid_size: u32,
    n_pods: u32,
}

// ─── Compute Shader: SOM Router ─────────────────────────────────────────────

@compute @workgroup_size(64)
fn som_route(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.batch_size) {
        return;
    }

    let n_in = params.n_in;
    let offset = idx * n_in;

    // ═══ KROK 1: Projekcja liniowa → UV ═══════════════════════════════════
    // input[n_in] × projection[n_in, 2] + bias[2] → uv[2]
    // Koszt: 2 × n_in MAC (dot product × 2 wymiary)
    var u: f32 = projection_bias[0];
    var v: f32 = projection_bias[1];

    for (var i: u32 = 0u; i < n_in; i = i + 1u) {
        let x_i = input_vectors[offset + i];
        u = u + x_i * projection_matrix[i * 2u + 0u];
        v = v + x_i * projection_matrix[i * 2u + 1u];
    }

    // Sigmoid → [0, 1] (normalizacja UV)
    u = 1.0 / (1.0 + exp(-u));
    v = 1.0 / (1.0 + exp(-v));

    // ═══ KROK 2: TMU FETCH (hardware, 0 MAC) ═══════════════════════════════
    // textureSampleLevel korzysta z Texture Mapping Unit karty graficznej.
    // Bilinear interpolation = 1 cykl zegara, NIE zużywa compute units.
    let uv = vec2<f32>(u, v);
    let texel = textureSampleLevel(som_texture, som_sampler, uv, 0.0);

    // ═══ KROK 3: Dekodowanie wyniku ═════════════════════════════════════════
    // R channel = capsule_id (znormalizowane: id / n_pods)
    // G channel = confidence (jak pewny jest routing)
    let capsule_id = u32(texel.r * f32(params.n_pods - 1u) + 0.5);
    let confidence = texel.g;

    output_capsules[idx] = capsule_id;
    output_confidence[idx] = confidence;
}

// ─── Compute Shader: SOM Training (online update) ───────────────────────────
// Opcjonalny shader do online adaptation SOM (Sleep v2 + Hebbian)

@group(0) @binding(0) var<storage, read> train_inputs: array<f32>;       // [batch × n_in]
@group(0) @binding(1) var<storage, read> train_labels: array<u32>;       // [batch]
@group(0) @binding(2) var<storage, read_write> som_weights: array<f32>;  // [grid × grid × n_in]

struct TrainParams {
    batch_size: u32,
    n_in: u32,
    grid_size: u32,
    learning_rate: f32,
    radius: f32,
}

@group(1) @binding(0) var<uniform> train_params: TrainParams;

@compute @workgroup_size(64)
fn som_train_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Każdy wątek aktualizuje jedną komórkę SOM na podstawie jednego sample'a
    let cell_idx = gid.x;
    let grid_total = train_params.grid_size * train_params.grid_size;

    if (cell_idx >= grid_total) {
        return;
    }

    let cell_row = cell_idx / train_params.grid_size;
    let cell_col = cell_idx % train_params.grid_size;
    let n_in = train_params.n_in;

    // Dla uproszczenia: aktualizuj na podstawie pierwszego sample'a w batchu
    // (pełna wersja: loop over batch with atomic add)
    let sample_offset = 0u * n_in;

    // Oblicz odległość komórki od BMU (w przestrzeni siatki)
    // BMU jest obliczane wcześniej (lub w osobnym upass)
    // Tu: uproszczona wersja — Hebbian update proporcjonalny do proximity

    let weight_offset = cell_idx * n_in;
    let lr = train_params.learning_rate;

    for (var i: u32 = 0u; i < n_in; i = i + 1u) {
        let x_i = train_inputs[sample_offset + i];
        let w_i = som_weights[weight_offset + i];
        // Hebbian: waga zbliża się do wejścia
        som_weights[weight_offset + i] = w_i + lr * (x_i - w_i);
    }
}

// ─── Compute Shader: Sleep v2 (Selective Decay + Prune) ─────────────────────

@group(0) @binding(0) var<storage, read_write> weights: array<f32>;      // [grid × grid × n_in]
@group(0) @binding(1) var<storage, read> activation_count: array<u32>;   // [grid × grid]

struct SleepParams {
    total_cells: u32,
    n_in: u32,
    decay_rate: f32,       // np. 0.01 — ile wag zanika per cykl
    prune_threshold: u32,  // ile razy musi być aktywowana, by przeżyć
}

@group(1) @binding(0) var<uniform> sleep_params: SleepParams;

@compute @workgroup_size(64)
fn sleep_cycle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    if (cell_idx >= sleep_params.total_cells) {
        return;
    }

    let activations = activation_count[cell_idx];
    let n_in = sleep_params.n_in;
    let weight_offset = cell_idx * n_in;

    if (activations < sleep_params.prune_threshold) {
        // PRUNE: martwa komórka → zeruj wagi (do re-assignmentu)
        for (var i: u32 = 0u; i < n_in; i = i + 1u) {
            weights[weight_offset + i] = 0.0;
        }
    } else {
        // SELECTIVE DECAY: zmniejsz wagi proporcjonalnie do ich wielkości
        let decay = sleep_params.decay_rate;
        for (var i: u32 = 0u; i < n_in; i = i + 1u) {
            let w = weights[weight_offset + i];
            weights[weight_offset + i] = w * (1.0 - decay);
        }
    }
}
