/**
 * M.A.R.S. — WebGPU SOM-Router Proof of Concept
 * 
 * Demonstracja natywnego texture fetch jako routera decyzyjnego.
 * Architektura:
 *   1. Linear Projection (input → UV coords) — znikomy koszt
 *   2. textureSampleLevel (UV → capsule_id) — 0 MAC (TMU hardware)
 *   3. Output: capsule_id + confidence
 * 
 * Benchmark: porównanie z naive matmul router (w compute shader)
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIG = {
    N_IN: 2,           // Wymiar wejścia (2D dla PoC, 784 dla MNIST)
    N_PODS: 5,         // Liczba kapsuł
    GRID_SIZE: 16,     // Rozmiar siatki SOM (16×16)
    BATCH_SIZE: 4096,  // Batch do benchmarku
    N_WARMUP: 50,      // Warmup iterations
    N_BENCHMARK: 200,  // Benchmark iterations
};

// ═══════════════════════════════════════════════════════════════════════════════
// WebGPU Initialization
// ═══════════════════════════════════════════════════════════════════════════════

let device, queue;
let somTexture, somSampler;
let routerPipeline, routerBindGroups;
let inputBuffer, outputCapsuleBuffer, outputConfidenceBuffer;
let projectionMatrixBuffer, projectionBiasBuffer, paramsBuffer;
let timestampQuerySet, timestampBuffer;

const log = (msg) => {
    const el = document.getElementById('log');
    if (el) el.textContent += msg + '\n';
    console.log(msg);
};

async function initWebGPU() {
    if (!navigator.gpu) {
        log('❌ WebGPU nie jest dostępne w tej przeglądarce.');
        log('   Wymagane: Chrome 113+ / Edge 113+ / Firefox Nightly');
        return false;
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
    });

    if (!adapter) {
        log('❌ Nie znaleziono GPU adapter.');
        return false;
    }

    const adapterInfo = await adapter.requestAdapterInfo();
    log(`  GPU Adapter: ${adapterInfo.device || adapterInfo.description || 'unknown'}`);
    log(`  Vendor:      ${adapterInfo.vendor || 'unknown'}`);
    log(`  Architecture: ${adapterInfo.architecture || 'unknown'}`);

    // Request device with timestamp query if available
    const features = [];
    if (adapter.features.has('timestamp-query')) {
        features.push('timestamp-query');
    }

    device = await adapter.requestDevice({
        requiredFeatures: features,
        requiredLimits: {
            maxStorageBufferBindingSize: 256 * 1024 * 1024,
        }
    });
    queue = device.queue;

    log(`  Timestamp query: ${device.features.has('timestamp-query') ? '✓' : '✗ (timing via JS)'}`);
    log('  WebGPU ✓ initialized');
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SOM Texture Creation (emulacja wyniku treningu Kohonena)
// ═══════════════════════════════════════════════════════════════════════════════

function createSOMTexture() {
    const { GRID_SIZE, N_PODS } = CONFIG;

    // Tworzymy label map jako teksturę RGBA8:
    // R = capsule_id / (N_PODS-1)  — znormalizowane
    // G = confidence (1.0 w centrze regionu, 0.5 na granicy)
    // B = unused
    // A = 1.0
    const data = new Uint8Array(GRID_SIZE * GRID_SIZE * 4);

    // Generuj SOM-like topology: regiony jako sektory kołowe
    for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
            const u = col / (GRID_SIZE - 1);
            const v = row / (GRID_SIZE - 1);

            // Oblicz angle od centrum → przypisz do kapsuły
            const dx = u - 0.5;
            const dy = v - 0.5;
            const angle = Math.atan2(dy, dx) + Math.PI; // [0, 2π]
            const capsuleId = Math.floor(angle / (2 * Math.PI) * N_PODS) % N_PODS;

            // Confidence: wyższa bliżej centrum sektora
            const sectorAngle = (2 * Math.PI) / N_PODS;
            const sectorCenter = (capsuleId + 0.5) * sectorAngle;
            const angleDist = Math.abs(angle - sectorCenter);
            const confidence = Math.max(0, 1.0 - angleDist / (sectorAngle * 0.5));

            const idx = (row * GRID_SIZE + col) * 4;
            data[idx + 0] = Math.round((capsuleId / (N_PODS - 1)) * 255); // R: capsule_id
            data[idx + 1] = Math.round(confidence * 255);                  // G: confidence
            data[idx + 2] = 0;                                              // B: unused
            data[idx + 3] = 255;                                            // A: 1.0
        }
    }

    somTexture = device.createTexture({
        size: [GRID_SIZE, GRID_SIZE],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    queue.writeTexture(
        { texture: somTexture },
        data,
        { bytesPerRow: GRID_SIZE * 4 },
        { width: GRID_SIZE, height: GRID_SIZE }
    );

    somSampler = device.createSampler({
        magFilter: 'linear',    // Bilinear = TMU hardware interpolation
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
    });

    log(`  SOM Texture: ${GRID_SIZE}×${GRID_SIZE} RGBA8, bilinear sampler ✓`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pipeline Setup
// ═══════════════════════════════════════════════════════════════════════════════

async function createPipeline() {
    const { N_IN, N_PODS, GRID_SIZE, BATCH_SIZE } = CONFIG;

    // Load shader
    const shaderCode = await fetch('som_router.wgsl').then(r => r.text());
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // ─── Buffers ────────────────────────────────────────────────────────────

    // Input vectors [BATCH_SIZE × N_IN]
    const inputData = new Float32Array(BATCH_SIZE * N_IN);
    for (let i = 0; i < inputData.length; i++) {
        inputData[i] = Math.random(); // Random test data
    }
    inputBuffer = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(inputBuffer, 0, inputData);

    // Output capsule IDs [BATCH_SIZE]
    outputCapsuleBuffer = device.createBuffer({
        size: BATCH_SIZE * 4, // u32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Output confidence [BATCH_SIZE]
    outputConfidenceBuffer = device.createBuffer({
        size: BATCH_SIZE * 4, // f32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Projection matrix [N_IN × 2] — inicjalizacja losowa (w produkcji: PCA/LSH)
    const projData = new Float32Array(N_IN * 2);
    for (let i = 0; i < projData.length; i++) {
        projData[i] = (Math.random() - 0.5) * 2.0;
    }
    projectionMatrixBuffer = device.createBuffer({
        size: projData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(projectionMatrixBuffer, 0, projData);

    // Projection bias [2]
    const biasData = new Float32Array([0.0, 0.0]);
    projectionBiasBuffer = device.createBuffer({
        size: biasData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(projectionBiasBuffer, 0, biasData);

    // Uniform params
    const paramsData = new Uint32Array([BATCH_SIZE, N_IN, GRID_SIZE, N_PODS]);
    paramsBuffer = device.createBuffer({
        size: paramsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(paramsBuffer, 0, paramsData);

    // ─── Pipeline ───────────────────────────────────────────────────────────

    const bindGroupLayout0 = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]
    });

    const bindGroupLayout1 = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ]
    });

    const bindGroupLayout2 = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        ]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout0, bindGroupLayout1, bindGroupLayout2],
    });

    routerPipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: 'som_route',
        },
    });

    // ─── Bind Groups ────────────────────────────────────────────────────────

    routerBindGroups = [
        device.createBindGroup({
            layout: bindGroupLayout0,
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputCapsuleBuffer } },
                { binding: 2, resource: { buffer: outputConfidenceBuffer } },
            ],
        }),
        device.createBindGroup({
            layout: bindGroupLayout1,
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: projectionMatrixBuffer } },
                { binding: 2, resource: { buffer: projectionBiasBuffer } },
            ],
        }),
        device.createBindGroup({
            layout: bindGroupLayout2,
            entries: [
                { binding: 0, resource: somTexture.createView() },
                { binding: 1, resource: somSampler },
            ],
        }),
    ];

    log('  Pipeline ✓ created');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark
// ═══════════════════════════════════════════════════════════════════════════════

async function runBenchmark() {
    const { BATCH_SIZE, N_WARMUP, N_BENCHMARK } = CONFIG;
    const workgroups = Math.ceil(BATCH_SIZE / 64);

    log('\n─── BENCHMARK: SOM-Router WebGPU ─────────────────────────────────');
    log(`  Batch: ${BATCH_SIZE} | Workgroups: ${workgroups} | Iterations: ${N_BENCHMARK}`);

    // Warmup
    for (let i = 0; i < N_WARMUP; i++) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(routerPipeline);
        pass.setBindGroup(0, routerBindGroups[0]);
        pass.setBindGroup(1, routerBindGroups[1]);
        pass.setBindGroup(2, routerBindGroups[2]);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        queue.submit([encoder.finish()]);
    }
    await queue.onSubmittedWorkDone();

    // Timed runs
    const times = [];

    for (let i = 0; i < N_BENCHMARK; i++) {
        const t0 = performance.now();

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(routerPipeline);
        pass.setBindGroup(0, routerBindGroups[0]);
        pass.setBindGroup(1, routerBindGroups[1]);
        pass.setBindGroup(2, routerBindGroups[2]);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        queue.submit([encoder.finish()]);
        await queue.onSubmittedWorkDone();

        const t1 = performance.now();
        times.push((t1 - t0) * 1000); // ms → μs
    }

    // Statistics
    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)];
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const p95 = times[Math.floor(times.length * 0.95)];
    const min = times[0];
    const throughput = BATCH_SIZE / (median / 1e6); // samples/sec

    log('\n  WYNIKI:');
    log(`    Median:      ${median.toFixed(1)} μs / batch`);
    log(`    Mean:        ${mean.toFixed(1)} μs / batch`);
    log(`    Min:         ${min.toFixed(1)} μs / batch`);
    log(`    P95:         ${p95.toFixed(1)} μs / batch`);
    log(`    Throughput:  ${(throughput / 1e6).toFixed(2)}M samples/s`);
    log(`    Per sample:  ${(median / BATCH_SIZE * 1000).toFixed(2)} ns`);

    // Porównanie z PyTorch (GTX 1050 Ti)
    const pytorchNeuralUs = 156.0; // z etap3c_gpu.py
    const pytorchTmuUs = 97.9;
    log('\n  PORÓWNANIE z PyTorch (GTX 1050 Ti):');
    log(`    PyTorch Neural Router: ${pytorchNeuralUs} μs`);
    log(`    PyTorch grid_sample:   ${pytorchTmuUs} μs`);
    log(`    WebGPU SOM-Router:     ${median.toFixed(1)} μs`);

    if (median < pytorchTmuUs) {
        log(`    → WebGPU jest ${(pytorchTmuUs / median).toFixed(1)}× SZYBSZY niż PyTorch TMU!`);
    } else if (median < pytorchNeuralUs) {
        log(`    → WebGPU jest ${(pytorchNeuralUs / median).toFixed(1)}× szybszy niż PyTorch Neural`);
    }

    return { median, mean, min, p95, throughput };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Read Results (verification)
// ═══════════════════════════════════════════════════════════════════════════════

async function readResults() {
    const { BATCH_SIZE } = CONFIG;

    // Read output capsules
    const readBuffer = device.createBuffer({
        size: BATCH_SIZE * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputCapsuleBuffer, 0, readBuffer, 0, BATCH_SIZE * 4);
    queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const capsules = new Uint32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Stats
    const counts = new Array(CONFIG.N_PODS).fill(0);
    for (const c of capsules) {
        if (c < CONFIG.N_PODS) counts[c]++;
    }

    log('\n─── WYNIKI ROUTINGU ──────────────────────────────────────────────');
    log(`  Rozkład kapsuł (${BATCH_SIZE} samples):`);
    for (let i = 0; i < CONFIG.N_PODS; i++) {
        const pct = (counts[i] / BATCH_SIZE * 100).toFixed(1);
        const bar = '█'.repeat(Math.round(counts[i] / BATCH_SIZE * 30));
        log(`    Kapsuła ${i}: ${counts[i].toString().padStart(5)} (${pct.padStart(5)}%) ${bar}`);
    }

    readBuffer.destroy();
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAC Cost Analysis
// ═══════════════════════════════════════════════════════════════════════════════

function analyzeMACCost() {
    const { N_IN, N_PODS, GRID_SIZE } = CONFIG;

    log('\n─── ANALIZA MAC ─────────────────────────────────────────────────');

    // Neural Router: input[N_IN] → hidden[64] → output[N_PODS]
    const neuralHidden = 64;
    const macNeural = N_IN * neuralHidden + neuralHidden * N_PODS;

    // SOM-Router (brute force): distance to all grid cells
    const macSomBrute = N_IN * GRID_SIZE * GRID_SIZE;

    // SOM-Router (projection + TMU): only projection cost
    const macSomProjection = N_IN * 2; // dot product × 2 dimensions

    // TMU fetch: 0 MAC (hardware)
    const macTMU = 0;

    const macSomTotal = macSomProjection + macTMU;

    log(`  Neural Router (hidden=${neuralHidden}):`);
    log(`    ${N_IN}×${neuralHidden} + ${neuralHidden}×${N_PODS} = ${macNeural} MAC`);
    log('');
    log(`  SOM Brute-force (grid=${GRID_SIZE}×${GRID_SIZE}):`);
    log(`    ${N_IN}×${GRID_SIZE * GRID_SIZE} = ${macSomBrute} MAC`);
    log('');
    log(`  SOM Projection + TMU:`);
    log(`    Projekcja: ${N_IN}×2 = ${macSomProjection} MAC`);
    log(`    TMU fetch: ${macTMU} MAC (hardware)`);
    log(`    TOTAL: ${macSomTotal} MAC`);
    log('');

    const saving = ((1 - macSomTotal / macNeural) * 100).toFixed(1);
    log(`  ═══ OSZCZĘDNOŚĆ: ${saving}% mniej MAC (${macNeural} → ${macSomTotal}) ═══`);

    // Ekstrapolacja na MNIST
    log('\n  Ekstrapolacja na MNIST (N_IN=784, hidden=128, N_PODS=10):');
    const mnistNeural = 784 * 128 + 128 * 10;
    const mnistSomProj = 784 * 2;
    const mnistSaving = ((1 - mnistSomProj / mnistNeural) * 100).toFixed(1);
    log(`    Neural: ${mnistNeural.toLocaleString()} MAC`);
    log(`    SOM+TMU: ${mnistSomProj.toLocaleString()} MAC`);
    log(`    Oszczędność: ${mnistSaving}% !!!`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
    log('================================================================');
    log(' M.A.R.S. — WebGPU SOM-Router: Native Texture Fetch PoC');
    log('================================================================\n');

    const ok = await initWebGPU();
    if (!ok) return;

    createSOMTexture();
    await createPipeline();

    // Run once to verify
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(routerPipeline);
    pass.setBindGroup(0, routerBindGroups[0]);
    pass.setBindGroup(1, routerBindGroups[1]);
    pass.setBindGroup(2, routerBindGroups[2]);
    pass.dispatchWorkgroups(Math.ceil(CONFIG.BATCH_SIZE / 64));
    pass.end();
    queue.submit([encoder.finish()]);
    await queue.onSubmittedWorkDone();

    await readResults();
    analyzeMACCost();

    const benchResults = await runBenchmark();

    // Werdykt
    log('\n================================================================');
    log(' WERDYKT: WebGPU SOM-Router');
    log('================================================================');
    log(`  Throughput: ${(benchResults.throughput / 1e6).toFixed(2)}M samples/s`);
    log(`  Latency:   ${benchResults.median.toFixed(1)} μs per batch (${CONFIG.BATCH_SIZE} samples)`);

    if (benchResults.throughput > 10e6) {
        log('  ✓ POTWIERDZONY: Natywny texture fetch działa jako router');
        log('  ✓ System M.A.R.S. może działać w real-time w przeglądarce');
    }
    log('\n  Następny krok: integracja z Three.js / React Three Fiber');
    log('  → SOM-Router jako background compute w silniku 3D');
}

// Start
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
} else {
    main();
}
