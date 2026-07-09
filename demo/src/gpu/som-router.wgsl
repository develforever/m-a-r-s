// ═══════════════════════════════════════════════════════════════════════════════
// M.A.R.S. — SOM-Router Compute Shader (WebGPU / WGSL)
// React Three Fiber integration: background compute for NPC decision-making
//
// Pipeline: Input[N_IN] → Projection → sigmoid → UV → textureSampleLevel → capsule_id
// Cost: N_IN × H_ENC + H_ENC × 2 MAC (encoder) + 0 MAC (TMU fetch)
// ═══════════════════════════════════════════════════════════════════════════════

struct Params {
    batch_size: u32,
    n_in: u32,
    encoder_hidden: u32,
    n_pods: u32,
    grid_size: u32,
}

// Group 0: I/O buffers
@group(0) @binding(0) var<storage, read> input_vectors: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_capsules: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_confidence: array<f32>;

// Group 1: Encoder weights
@group(1) @binding(0) var<uniform> params: Params;
@group(1) @binding(1) var<storage, read> enc_w1: array<f32>;  // [n_in × encoder_hidden]
@group(1) @binding(2) var<storage, read> enc_b1: array<f32>;  // [encoder_hidden]
@group(1) @binding(3) var<storage, read> enc_w2: array<f32>;  // [encoder_hidden × 2]
@group(1) @binding(4) var<storage, read> enc_b2: array<f32>;  // [2]

// Group 2: SOM texture
@group(2) @binding(0) var som_texture: texture_2d<f32>;
@group(2) @binding(1) var som_sampler: sampler;

// ─── Compute: SOM Route ──────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn som_route(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.batch_size) {
        return;
    }

    let n_in = params.n_in;
    let h = params.encoder_hidden;
    let offset = idx * n_in;

    // ═══ Layer 1: input × W1 + b1 → hidden (ReLU) ═══════════════════════════
    // Cost: n_in × encoder_hidden MAC
    for (var j: u32 = 0u; j < h; j = j + 1u) {
        var sum: f32 = enc_b1[j];
        for (var i: u32 = 0u; i < n_in; i = i + 1u) {
            sum = sum + input_vectors[offset + i] * enc_w1[i * h + j];
        }
        // ReLU
        sum = max(sum, 0.0);
        // Store in output_confidence temporarily (reuse buffer trick)
        // Actually we need intermediate storage - use local var approach
        // For workgroup_size(64), we use private memory
    }

    // Full encoder forward pass (private memory for hidden layer)
    var hidden: array<f32, 128>; // Max encoder_hidden = 128
    for (var j: u32 = 0u; j < h; j = j + 1u) {
        var sum: f32 = enc_b1[j];
        for (var i: u32 = 0u; i < n_in; i = i + 1u) {
            sum = sum + input_vectors[offset + i] * enc_w1[i * h + j];
        }
        hidden[j] = max(sum, 0.0); // ReLU
    }

    // ═══ Layer 2: hidden × W2 + b2 → UV[2] (sigmoid) ════════════════════════
    // Cost: encoder_hidden × 2 MAC
    var u: f32 = enc_b2[0];
    var v: f32 = enc_b2[1];
    for (var j: u32 = 0u; j < h; j = j + 1u) {
        u = u + hidden[j] * enc_w2[j * 2u + 0u];
        v = v + hidden[j] * enc_w2[j * 2u + 1u];
    }

    // Sigmoid → [0, 1]
    u = 1.0 / (1.0 + exp(-u));
    v = 1.0 / (1.0 + exp(-v));

    // ═══ TMU FETCH (0 MAC — hardware texture sampling) ═══════════════════════
    let uv = vec2<f32>(u, v);
    let texel = textureSampleLevel(som_texture, som_sampler, uv, 0.0);

    // Decode: R = capsule_id (normalized), G = confidence
    let capsule_id = u32(texel.r * f32(params.n_pods - 1u) + 0.5);
    let confidence = texel.g;

    output_capsules[idx] = capsule_id;
    output_confidence[idx] = confidence;
}
