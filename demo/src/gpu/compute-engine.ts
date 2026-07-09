/**
 * M.A.R.S. WebGPU Compute Engine
 * 
 * Runs the SOM-Router as a background compute pipeline on the GPU.
 * Designed to be called per-frame from React Three Fiber without
 * blocking the main JS thread or impacting rendering performance.
 * 
 * Architecture:
 *   Input[N_IN] → Encoder[N_IN→H→2] → sigmoid → UV → textureSample → capsule_id
 *   Cost: N_IN×H + H×2 MAC (encoder) + 0 MAC (TMU fetch)
 */

export interface ComputeEngineConfig {
  nIn: number;          // Input dimensionality (e.g., 16 for game state)
  encoderHidden: number; // Encoder hidden size (e.g., 8)
  nPods: number;        // Number of capsules/strategies (e.g., 4)
  gridSize: number;     // SOM texture resolution (e.g., 16)
  batchSize: number;    // How many decisions per dispatch
}

export interface RouteResult {
  capsuleIds: Uint32Array;
  confidence: Float32Array;
  computeTimeMs: number;
}

export class MARSComputeEngine {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroup0: GPUBindGroup | null = null;
  private bindGroup1: GPUBindGroup | null = null;
  private bindGroup2: GPUBindGroup | null = null;

  // Buffers
  private inputBuffer: GPUBuffer | null = null;
  private outputCapsulesBuffer: GPUBuffer | null = null;
  private outputConfidenceBuffer: GPUBuffer | null = null;
  private readbackCapsules: GPUBuffer | null = null;
  private readbackConfidence: GPUBuffer | null = null;
  private paramsBuffer: GPUBuffer | null = null;

  // Encoder weight buffers
  private encW1Buffer: GPUBuffer | null = null;
  private encB1Buffer: GPUBuffer | null = null;
  private encW2Buffer: GPUBuffer | null = null;
  private encB2Buffer: GPUBuffer | null = null;

  private config: ComputeEngineConfig;
  private ready = false;
  private lastResult: RouteResult | null = null;

  constructor(config: ComputeEngineConfig) {
    this.config = config;
  }

  async init(): Promise<boolean> {
    if (!navigator.gpu) {
      console.warn('WebGPU not supported');
      return false;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.warn('No GPU adapter found');
      return false;
    }

    this.device = await adapter.requestDevice();
    
    // Load shader
    const shaderCode = await fetch(new URL('./som-router.wgsl', import.meta.url).href)
      .then(r => r.text());
    
    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    // Create pipeline
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'som_route',
      },
    });

    this.createBuffers();
    this.initializeWeights();
    this.createBindGroups();

    this.ready = true;
    console.log('[M.A.R.S.] Compute engine initialized', this.config);
    return true;
  }

  private createBuffers() {
    const d = this.device!;
    const { batchSize, nIn, encoderHidden } = this.config;

    // I/O buffers
    this.inputBuffer = d.createBuffer({
      size: batchSize * nIn * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.outputCapsulesBuffer = d.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.outputConfidenceBuffer = d.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Readback buffers
    this.readbackCapsules = d.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.readbackConfidence = d.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Params uniform
    this.paramsBuffer = d.createBuffer({
      size: 5 * 4, // 5 u32 fields
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([
      this.config.batchSize,
      this.config.nIn,
      this.config.encoderHidden,
      this.config.nPods,
      this.config.gridSize,
    ]));

    // Encoder weights
    this.encW1Buffer = d.createBuffer({
      size: nIn * encoderHidden * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.encB1Buffer = d.createBuffer({
      size: encoderHidden * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.encW2Buffer = d.createBuffer({
      size: encoderHidden * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.encB2Buffer = d.createBuffer({
      size: 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  private initializeWeights() {
    const d = this.device!;
    const { nIn, encoderHidden } = this.config;

    // Xavier init for encoder
    const scale1 = Math.sqrt(2.0 / (nIn + encoderHidden));
    const w1 = new Float32Array(nIn * encoderHidden);
    for (let i = 0; i < w1.length; i++) w1[i] = (Math.random() * 2 - 1) * scale1;
    d.queue.writeBuffer(this.encW1Buffer!, 0, w1);

    const b1 = new Float32Array(encoderHidden); // zeros
    d.queue.writeBuffer(this.encB1Buffer!, 0, b1);

    const scale2 = Math.sqrt(2.0 / (encoderHidden + 2));
    const w2 = new Float32Array(encoderHidden * 2);
    for (let i = 0; i < w2.length; i++) w2[i] = (Math.random() * 2 - 1) * scale2;
    d.queue.writeBuffer(this.encW2Buffer!, 0, w2);

    const b2 = new Float32Array(2); // zeros
    d.queue.writeBuffer(this.encB2Buffer!, 0, b2);
  }

  private createBindGroups() {
    const d = this.device!;
    const layout = this.pipeline!.getBindGroupLayout(0);
    const layout1 = this.pipeline!.getBindGroupLayout(1);

    this.bindGroup0 = d.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer! } },
        { binding: 1, resource: { buffer: this.outputCapsulesBuffer! } },
        { binding: 2, resource: { buffer: this.outputConfidenceBuffer! } },
      ],
    });

    this.bindGroup1 = d.createBindGroup({
      layout: layout1,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer! } },
        { binding: 1, resource: { buffer: this.encW1Buffer! } },
        { binding: 2, resource: { buffer: this.encB1Buffer! } },
        { binding: 3, resource: { buffer: this.encW2Buffer! } },
        { binding: 4, resource: { buffer: this.encB2Buffer! } },
      ],
    });

    // Group 2: SOM texture (we'll use a simple fallback without texture for now)
    // In production, this would be a real texture2D with the label map
  }

  /**
   * Dispatch compute shader — non-blocking, runs on GPU.
   * Call this every frame with the current game state vectors.
   */
  async route(inputData: Float32Array): Promise<RouteResult> {
    if (!this.ready || !this.device) {
      return { capsuleIds: new Uint32Array(1), confidence: new Float32Array(1), computeTimeMs: 0 };
    }

    const t0 = performance.now();
    const d = this.device;

    // Upload input
    d.queue.writeBuffer(this.inputBuffer!, 0, inputData);

    // Encode command
    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup0!);
    pass.setBindGroup(1, this.bindGroup1!);
    // pass.setBindGroup(2, this.bindGroup2!); // Texture bind group
    
    const workgroups = Math.ceil(this.config.batchSize / 64);
    pass.dispatchWorkgroups(workgroups);
    pass.end();

    // Copy results to readback
    encoder.copyBufferToBuffer(
      this.outputCapsulesBuffer!, 0,
      this.readbackCapsules!, 0,
      this.config.batchSize * 4
    );
    encoder.copyBufferToBuffer(
      this.outputConfidenceBuffer!, 0,
      this.readbackConfidence!, 0,
      this.config.batchSize * 4
    );

    d.queue.submit([encoder.finish()]);

    // Map and read results
    await this.readbackCapsules!.mapAsync(GPUMapMode.READ);
    await this.readbackConfidence!.mapAsync(GPUMapMode.READ);

    const capsuleIds = new Uint32Array(
      this.readbackCapsules!.getMappedRange().slice(0)
    );
    const confidence = new Float32Array(
      this.readbackConfidence!.getMappedRange().slice(0)
    );

    this.readbackCapsules!.unmap();
    this.readbackConfidence!.unmap();

    const computeTimeMs = performance.now() - t0;

    this.lastResult = { capsuleIds, confidence, computeTimeMs };
    return this.lastResult;
  }

  /**
   * Lightweight CPU-side fallback router (for browsers without WebGPU).
   * Uses simple dot-product routing.
   */
  routeCPU(inputData: Float32Array): RouteResult {
    const t0 = performance.now();
    const { batchSize, nIn, nPods } = this.config;
    
    const capsuleIds = new Uint32Array(batchSize);
    const confidence = new Float32Array(batchSize);

    for (let b = 0; b < batchSize; b++) {
      // Simple hash-based routing (placeholder)
      let sum = 0;
      for (let i = 0; i < nIn; i++) {
        sum += inputData[b * nIn + i];
      }
      capsuleIds[b] = Math.abs(Math.floor(sum * 1000)) % nPods;
      confidence[b] = 0.5 + Math.random() * 0.5;
    }

    const computeTimeMs = performance.now() - t0;
    return { capsuleIds, confidence, computeTimeMs };
  }

  getLastResult(): RouteResult | null {
    return this.lastResult;
  }

  isReady(): boolean {
    return this.ready;
  }

  destroy() {
    this.inputBuffer?.destroy();
    this.outputCapsulesBuffer?.destroy();
    this.outputConfidenceBuffer?.destroy();
    this.readbackCapsules?.destroy();
    this.readbackConfidence?.destroy();
    this.paramsBuffer?.destroy();
    this.encW1Buffer?.destroy();
    this.encB1Buffer?.destroy();
    this.encW2Buffer?.destroy();
    this.encB2Buffer?.destroy();
    this.device?.destroy();
    this.ready = false;
  }
}
