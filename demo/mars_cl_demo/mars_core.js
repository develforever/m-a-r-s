/* mars_core.js — MARS-CL inference in pure JavaScript.
 *
 * Faithful replica of the PyTorch forward pass (src/mars_cl*.py):
 *   input 28x28 -> normalize -> Conv3x3(1->c1,pad=1) -> BN -> ReLU -> MaxPool2
 *   -> Conv3x3(c1->c2,pad=1) -> BN -> ReLU -> MaxPool2 -> flatten [c2*7*7]
 *   -> Linear(->128) -> ReLU  = features
 *   features -> Linear(128->50) -> L2-normalize = embedding
 *   routing: argmax cosine similarity to word anchors (seen classes only)
 *   pod of routed class: relu(f @ W1 + b1) @ W2 + b2 -> masked argmax = prediction
 *
 * Tensors arrive as {shape:[...], data:[...flat]} (see export_demo_weights.py).
 * Conv weights [O,C,3,3]; Linear weights [out,in] (PyTorch: y = x·Wᵀ + b);
 * pod W1 [128,24], W2 [24,10] used as y = x·W + b (raw tensors, not nn.Linear).
 */
"use strict";

/* ---------------- primitives ---------------- */

// Conv 3x3, stride 1, padding 1 (cross-correlation, like PyTorch).
// inp: Float32Array C*H*W, out: Float32Array O*H*W.
function conv3x3(inp, C, H, W, wt, bias, O) {
  const out = new Float32Array(O * H * W);
  for (let o = 0; o < O; o++) {
    const wo = o * C * 9;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let acc = bias[o];
        for (let c = 0; c < C; c++) {
          const wc = wo + c * 9;
          const ic = c * H * W;
          for (let ky = -1; ky <= 1; ky++) {
            const iy = y + ky;
            if (iy < 0 || iy >= H) continue;
            for (let kx = -1; kx <= 1; kx++) {
              const ix = x + kx;
              if (ix < 0 || ix >= W) continue;
              acc += wt[wc + (ky + 1) * 3 + (kx + 1)] * inp[ic + iy * W + ix];
            }
          }
        }
        out[o * H * W + y * W + x] = acc;
      }
    }
  }
  return out;
}

// BatchNorm (eval) + ReLU in place: y = max(0, g*(x-m)/sqrt(v+eps)+b)
function bnRelu(x, C, HW, g, b, m, v, eps = 1e-5) {
  for (let c = 0; c < C; c++) {
    const s = g[c] / Math.sqrt(v[c] + eps), off = b[c] - m[c] * s;
    for (let i = c * HW; i < (c + 1) * HW; i++) {
      const y = x[i] * s + off;
      x[i] = y > 0 ? y : 0;
    }
  }
  return x;
}

// MaxPool 2x2 stride 2: C×H×W -> C×(H/2)×(W/2)
function maxPool2(inp, C, H, W) {
  const h = H >> 1, w = W >> 1;
  const out = new Float32Array(C * h * w);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i0 = c * H * W + 2 * y * W + 2 * x;
        out[c * h * w + y * w + x] = Math.max(
          inp[i0], inp[i0 + 1], inp[i0 + W], inp[i0 + W + 1]);
      }
    }
  }
  return out;
}

// PyTorch Linear: W [out,in] flat, y = W·x + b
function linear(x, wt, bias, nOut, nIn) {
  const out = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let acc = bias[o];
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) acc += wt[row + i] * x[i];
    out[o] = acc;
  }
  return out;
}

// Raw matmul y = x·W + b, W [in,out] flat (pod weights)
function matVecT(x, wt, bias, nIn, nOut) {
  const out = Float32Array.from(bias);
  for (let i = 0; i < nIn; i++) {
    const xi = x[i], row = i * nOut;
    if (xi === 0) continue;
    for (let o = 0; o < nOut; o++) out[o] += xi * wt[row + o];
  }
  return out;
}

function relu(x) { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; }

function l2normalize(x) {
  let n = 0;
  for (let i = 0; i < x.length; i++) n += x[i] * x[i];
  n = Math.sqrt(n) || 1;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / n;
  return out;
}

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/* ---------------- model ---------------- */

class MarsModel {
  // weights = parsed mars_demo_weights.json
  constructor(weights) {
    this.w = weights;
    const bb = weights.backbone;
    this.c1 = bb.conv1_w.shape[0];         // 8
    this.c2 = bb.conv2_w.shape[0];         // 16
    this.bbH = bb.fc_w.shape[0];           // 128
    this.embDim = weights.anchors["0"].shape[0]; // 50
    this.anchors = {};
    for (let c = 0; c < 10; c++)
      this.anchors[c] = Float32Array.from(weights.anchors[String(c)].data);
    this.reset();
  }

  reset() { this.seen = []; this.projW = null; this.projB = null; this.pods = {}; }

  // Apply snapshots 0..upTo: projection = latest, pods accumulate, seen grows.
  setTask(upTo) {
    this.reset();
    for (let t = 0; t <= upTo; t++) {
      const s = this.w.snapshots[t];
      this.projW = Float32Array.from(s.proj_w.data);
      this.projB = Float32Array.from(s.proj_b.data);
      for (const c of s.classes) {
        const p = s.pods[String(c)];
        this.pods[c] = {
          W1: Float32Array.from(p.W1.data), b1: Float32Array.from(p.b1.data),
          W2: Float32Array.from(p.W2.data), b2: Float32Array.from(p.b2.data),
          hid: p.W1.shape[1],
        };
      }
      this.seen = this.seen.concat(s.classes);
    }
  }

  // pixels: Uint8Array/Array length 784 (0..255) -> features [128]
  features(pixels) {
    const { norm_mean: mu, norm_std: sd } = this.w.meta;
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) x[i] = (pixels[i] / 255 - mu) / sd;
    const bb = this.w.backbone;
    let t = conv3x3(x, 1, 28, 28, bb.conv1_w.data, bb.conv1_b.data, this.c1);
    bnRelu(t, this.c1, 784, bb.bn1.g.data, bb.bn1.b.data, bb.bn1.m.data, bb.bn1.v.data);
    t = maxPool2(t, this.c1, 28, 28);                      // c1×14×14
    t = conv3x3(t, this.c1, 14, 14, bb.conv2_w.data, bb.conv2_b.data, this.c2);
    bnRelu(t, this.c2, 196, bb.bn2.g.data, bb.bn2.b.data, bb.bn2.m.data, bb.bn2.v.data);
    t = maxPool2(t, this.c2, 14, 14);                      // c2×7×7
    return relu(linear(t, bb.fc_w.data, bb.fc_b.data, this.bbH, this.c2 * 49));
  }

  embed(feats) { return l2normalize(linear(feats, this.projW, this.projB, this.embDim, this.bbH)); }

  // cosine similarity to every anchor (unit vectors -> dot product)
  similarities(emb) {
    const sims = {};
    for (let c = 0; c < 10; c++) sims[c] = dot(emb, this.anchors[c]);
    return sims;
  }

  route(sims) {
    let best = this.seen[0], bs = -Infinity;
    for (const c of this.seen) if (sims[c] > bs) { bs = sims[c]; best = c; }
    return best;
  }

  podLogits(feats, c) {
    const p = this.pods[c];
    const h = relu(matVecT(feats, p.W1, p.b1, this.bbH, p.hid));
    return matVecT(h, p.W2, p.b2, p.hid, 10);
  }

  // Full pipeline. Returns everything the UI needs.
  classify(pixels) {
    const feats = this.features(pixels);
    const emb = this.embed(feats);
    const sims = this.similarities(emb);
    const routed = this.route(sims);
    const logits = this.podLogits(feats, routed);
    let pred = this.seen[0], bl = -Infinity;          // masked argmax (class-IL)
    for (const c of this.seen) if (logits[c] > bl) { bl = logits[c]; pred = c; }
    return { feats, emb, sims, routed, pred, logits };
  }

  // Self-test against PyTorch-exported vectors. Returns {pass, maxDiff, predOk}.
  parityCheck(pixelsAll) {
    const p = this.w.parity;
    const px = pixelsAll.subarray(p.image_index * 784, (p.image_index + 1) * 784);
    this.setTask(this.w.snapshots.length - 1);
    const r = this.classify(px);
    let maxDiff = 0;
    for (let i = 0; i < r.feats.length; i++)
      maxDiff = Math.max(maxDiff, Math.abs(r.feats[i] - p.feats.data[i]));
    for (let i = 0; i < r.emb.length; i++)
      maxDiff = Math.max(maxDiff, Math.abs(r.emb[i] - p.emb.data[i]));
    return { pass: maxDiff < 1e-2 && r.pred === p.pred, maxDiff, predOk: r.pred === p.pred };
  }
}

/* Node (tests) / browser */
if (typeof module !== "undefined" && module.exports) {
  module.exports = { conv3x3, bnRelu, maxPool2, linear, matVecT, relu, l2normalize, dot, MarsModel };
}
