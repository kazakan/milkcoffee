// ONNX Runtime Web – face detection worker.
// Runs entirely off the main thread so WASM compilation and inference never
// block the browser UI.
//
// Message protocol (main → worker):
//   { type:'init',       id }
//   { type:'load-model', id, modelData:ArrayBuffer, modelFileName, scoreThreshold }
//   { type:'detect',     id, modelKey, imageDataBuffer:ArrayBuffer, width, height }
//
// Message protocol (worker → main):
//   { type:'ready',        id }
//   { type:'model-loaded', id }
//   { type:'detect-result',id, detections:[] }
//   { type:'error',        id, error:string }
//   { type:'progress',     phase, ... }

'use strict';

const ORT_CDN =
  globalThis.__ORT_SCRIPT_URL_OVERRIDE ||
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js';
const ORT_WASM_CDN =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

/** @type {Map<string, {session: object, scoreThreshold: number}>} */
const sessions = new Map();

const MODEL_PROFILES = {
  generic: {
    name: 'generic',
    channelOrder: 'bgr',
    mean: [0, 0, 0],
    std: [1, 1, 1],
    letterbox: false,
  },
  scrfd: {
    name: 'scrfd',
    // InsightFace SCRFD commonly expects RGB with (x - 127.5) / 128.
    channelOrder: 'rgb',
    mean: [127.5, 127.5, 127.5],
    std: [128, 128, 128],
    letterbox: true,
  },
};

function detectModelProfile(modelFileName = '') {
  const lower = String(modelFileName).toLowerCase();
  if (lower.includes('scrfd')) return MODEL_PROFILES.scrfd;
  return MODEL_PROFILES.generic;
}

function postProgress(phase, extra = {}) {
  self.postMessage({ type: 'progress', phase, ts: Date.now(), ...extra });
}

// ── ORT initialisation ────────────────────────────────────────────────────────
async function ensureOrt() {
  if (globalThis.ort) return globalThis.ort;
  // importScripts is synchronous in workers; only blocks this thread (not UI).
  importScripts(ORT_CDN);
  const ort = globalThis.ort;
  if (!ort) throw new Error('onnxruntime-web did not expose globalThis.ort');
  // Tell ORT where to load its WASM binaries from.
  ort.env.wasm.wasmPaths = ORT_WASM_CDN;
  return ort;
}

// ── Image pre-processing ──────────────────────────────────────────────────────

/**
 * Resize RGBA pixel buffer (Uint8ClampedArray or ArrayBuffer) from srcW×srcH
 * to dstW×dstH using OffscreenCanvas for hardware-accelerated scaling.
 */
function resizeRgba(rgbaBuffer, srcW, srcH, dstW, dstH) {
  const src = new OffscreenCanvas(srcW, srcH);
  const srcCtx = src.getContext('2d');
  srcCtx.putImageData(
    new ImageData(new Uint8ClampedArray(rgbaBuffer), srcW, srcH),
    0, 0
  );
  const dst = new OffscreenCanvas(dstW, dstH);
  dst.getContext('2d').drawImage(src, 0, 0, dstW, dstH);
  return dst.getContext('2d').getImageData(0, 0, dstW, dstH).data;
}

function resizeRgbaWithLetterbox(rgbaBuffer, srcW, srcH, dstW, dstH) {
  const src = new OffscreenCanvas(srcW, srcH);
  const srcCtx = src.getContext('2d');
  srcCtx.putImageData(
    new ImageData(new Uint8ClampedArray(rgbaBuffer), srcW, srcH),
    0, 0
  );

  const scale = Math.min(dstW / srcW, dstH / srcH);
  const drawW = Math.max(1, Math.round(srcW * scale));
  const drawH = Math.max(1, Math.round(srcH * scale));
  const padX = Math.floor((dstW - drawW) / 2);
  const padY = Math.floor((dstH - drawH) / 2);

  const dst = new OffscreenCanvas(dstW, dstH);
  const dstCtx = dst.getContext('2d');
  dstCtx.fillStyle = 'black';
  dstCtx.fillRect(0, 0, dstW, dstH);
  dstCtx.drawImage(src, 0, 0, srcW, srcH, padX, padY, drawW, drawH);

  return {
    rgba: dstCtx.getImageData(0, 0, dstW, dstH).data,
    transform: { scale, padX, padY, drawW, drawH },
  };
}

/**
 * Convert RGBA Uint8ClampedArray → float32 CHW tensor in BGR channel order.
 * Values remain in the model's expected scale after profile mean/std transform.
 */
function rgbaToFloat32Chw(rgba, w, h, profile) {
  const n = w * h;
  const out = new Float32Array(3 * n);
  const channelIndices = profile.channelOrder === 'rgb' ? [0, 1, 2] : [2, 1, 0];
  const mean = profile.mean;
  const std = profile.std;
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    const c0 = rgba[p + channelIndices[0]];
    const c1 = rgba[p + channelIndices[1]];
    const c2 = rgba[p + channelIndices[2]];
    out[i] = (c0 - mean[0]) / std[0];
    out[n + i] = (c1 - mean[1]) / std[1];
    out[2 * n + i] = (c2 - mean[2]) / std[2];
  }
  return out;
}

/**
 * Build the [1, 3, modelH, modelW] float32 input tensor for a session.
 * Returns tensor + transform metadata for mapping boxes back to original image.
 */
function buildInputTensor(ort, imageDataBuffer, srcW, srcH, modelW, modelH, profile) {
  let resized;
  let transform;
  if (profile.letterbox) {
    const lb = resizeRgbaWithLetterbox(imageDataBuffer, srcW, srcH, modelW, modelH);
    resized = lb.rgba;
    transform = {
      type: 'letterbox',
      scale: lb.transform.scale,
      padX: lb.transform.padX,
      padY: lb.transform.padY,
    };
  } else {
    resized = resizeRgba(imageDataBuffer, srcW, srcH, modelW, modelH);
    transform = {
      type: 'stretch',
      scaleX: srcW / modelW,
      scaleY: srcH / modelH,
    };
  }
  const float32 = rgbaToFloat32Chw(resized, modelW, modelH, profile);
  return {
    tensor: new ort.Tensor('float32', float32, [1, 3, modelH, modelW]),
    transform,
  };
}

function mapRectFromModel(x, y, w, h, transform, origW, origH) {
  let ox;
  let oy;
  let ow;
  let oh;
  if (transform.type === 'letterbox') {
    ox = (x - transform.padX) / transform.scale;
    oy = (y - transform.padY) / transform.scale;
    ow = w / transform.scale;
    oh = h / transform.scale;
  } else {
    ox = x * transform.scaleX;
    oy = y * transform.scaleY;
    ow = w * transform.scaleX;
    oh = h * transform.scaleY;
  }
  const x1 = Math.max(0, Math.min(origW, ox));
  const y1 = Math.max(0, Math.min(origH, oy));
  const x2 = Math.max(x1, Math.min(origW, ox + ow));
  const y2 = Math.max(y1, Math.min(origH, oy + oh));
  return {
    originX: x1,
    originY: y1,
    width: x2 - x1,
    height: y2 - y1,
  };
}

// ── Detection output parsers ──────────────────────────────────────────────────

/** Normalise a score to [0,1]: pass through if already in range, else sigmoid. */
function normScore(raw) {
  if (raw >= 0 && raw <= 1) return raw;
  return 1 / (1 + Math.exp(-raw));
}

/**
 * SCRFD (buffalo_sc det_500m) multi-scale output.
 * Tensors ordered: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32
 * (optionally + kps_8/16/32 for 9-output models).
 *
 * Tensor names may be descriptive ('score_8') OR numeric ('443') depending
 * on the InsightFace export version.  Strategy:
 *   1. Try named access ('score_8', 'bbox_8', …)
 *   2. Try positional access via outputNames (InsightFace order)
 *   3. Identify by expected anchor count × last dim (shape-based fallback)
 *
 * Score values are expected to be sigmoid-activated (model bakes sigmoid in).
 * Apply normScore() as a safety net for models that output logits.
 */
function parseScrfdOutput(outputs, outputNames, modelW, modelH, transform, origW, origH, scoreThreshold) {
  const strides = [8, 16, 32];
  const numAnchorsPerPoint = 2;
  const n = strides.length;
  const result = [];

  for (let si = 0; si < n; si++) {
    const stride = strides[si];
    const featH = Math.floor(modelH / stride);
    const featW = Math.floor(modelW / stride);
    const expectedAnchors = featH * featW * numAnchorsPerPoint;

    // 1. Named lookup.
    let scoreTensor = outputs[`score_${stride}`];
    let bboxTensor  = outputs[`bbox_${stride}`];

    // 2. Positional lookup (InsightFace order: scores first, then bboxes).
    if (!scoreTensor && outputNames[si])        scoreTensor = outputs[outputNames[si]];
    if (!bboxTensor  && outputNames[si + n])    bboxTensor  = outputs[outputNames[si + n]];

    // 3. Shape-based fallback: match by anchor count × last dimension.
    if (!scoreTensor || !bboxTensor) {
      for (const name of outputNames) {
        const t = outputs[name];
        if (!t) continue;
        const lastDim = t.dims[t.dims.length - 1];
        const rows    = Math.round(t.data.length / lastDim);
        if (rows !== expectedAnchors) continue;
        if (lastDim === 1 && !scoreTensor) { scoreTensor = t; continue; }
        if (lastDim === 4 && !bboxTensor)  { bboxTensor  = t; continue; }
      }
    }

    if (!scoreTensor || !bboxTensor) continue;

    const scores     = scoreTensor.data;
    const bboxes     = bboxTensor.data;
    const scoreStride = scoreTensor.dims[scoreTensor.dims.length - 1]; // 1 or 2
    const scoreCount = scoreStride > 0 ? Math.floor(scores.length / scoreStride) : 0;
    const bboxCount = Math.floor(bboxes.length / 4);
    if (scoreCount < expectedAnchors || bboxCount < expectedAnchors) continue;

    for (let i = 0; i < expectedAnchors; i++) {
      // Some exports have 2 class columns [bg, face]; use the face (last) col.
      const rawScore = scoreStride === 1 ? scores[i] : scores[i * scoreStride + (scoreStride - 1)];
      const score    = normScore(rawScore);
      if (score < scoreThreshold) continue;
      const pointIndex = Math.floor(i / numAnchorsPerPoint);
      const x = pointIndex % featW;
      const y = Math.floor(pointIndex / featW);
      const cx = (x + 0.5) * stride;
      const cy = (y + 0.5) * stride;
      const x1 = (cx - bboxes[i * 4]     * stride);
      const y1 = (cy - bboxes[i * 4 + 1] * stride);
      const x2 = (cx + bboxes[i * 4 + 2] * stride);
      const y2 = (cy + bboxes[i * 4 + 3] * stride);
      const bb = mapRectFromModel(x1, y1, x2 - x1, y2 - y1, transform, origW, origH);
      if (bb.width <= 0 || bb.height <= 0) continue;
      result.push({
        categories: [{ score }],
        boundingBox: bb,
      });
    }
  }
  return result;
}

function parseDetections(outputs, outputNames, modelW, modelH, transform, origW, origH, scoreThreshold) {
  // SCRFD: named descriptive keys, OR 6/9 outputs (score×3 + bbox×3 [+ kps×3]).
  const n = outputNames.length;
  if ('score_8' in outputs || n === 6 || n === 9) {
    return parseScrfdOutput(outputs, outputNames, modelW, modelH, transform, origW, origH, scoreThreshold);
  }
  console.warn('[onnx-worker] Unknown model output format. Keys:', outputNames,
    '– first tensor dims:', outputs[outputNames[0]]?.dims);
  return [];
}

// ── Minimal ONNX protobuf parser – extract first graph input shape ────────────
// ONNX is protobuf-encoded.  We only need to dig down to:
//   ModelProto.graph (field 7) → GraphProto.input[0] (field 11)
//   → ValueInfoProto.type (field 2) → TypeProto.tensor_type (field 1)
//   → TensorShapeProto.dim[] (field 1 each) → dim_value (field 1, varint)
//
// The returned array is [N, C, H, W] (indices 0-3); returns null on failure.

function _readVarint(bytes, pos) {
  let result = 0, shift = 0;
  while (pos < bytes.length) {
    const b = bytes[pos++];
    result |= (b & 0x7f) << shift;
    if (!(b & 0x80)) break;
    shift += 7;
  }
  return { value: result >>> 0, pos };
}

function _skipField(bytes, pos, wireType) {
  if (wireType === 0) { // varint
    while (pos < bytes.length && (bytes[pos++] & 0x80));
    return pos;
  }
  if (wireType === 2) { // length-delimited
    const { value: len, pos: p } = _readVarint(bytes, pos);
    return p + len;
  }
  if (wireType === 5) return pos + 4;
  if (wireType === 1) return pos + 8;
  return bytes.length;
}

function _parseDims(bytes, start, end) {
  // TensorShapeProto: field 1 = dim (TensorShapeProto.Dimension)
  const dims = [];
  let pos = start;
  while (pos < end) {
    const { value: tag, pos: p } = _readVarint(bytes, pos); pos = p;
    const field = tag >> 3, wt = tag & 7;
    if (field === 1 && wt === 2) { // dim
      const { value: len, pos: p2 } = _readVarint(bytes, pos); pos = p2;
      // TensorShapeProto.Dimension: field 1 = dim_value (varint)
      let dpos = pos, dval = -1;
      while (dpos < pos + len) {
        const { value: dtag, pos: dp } = _readVarint(bytes, dpos); dpos = dp;
        const df = dtag >> 3, dwt = dtag & 7;
        if (df === 1 && dwt === 0) {
          const { value, pos: vp } = _readVarint(bytes, dpos);
          dval = value; dpos = vp;
        } else {
          dpos = _skipField(bytes, dpos, dwt);
        }
      }
      dims.push(dval);
      pos += len;
    } else {
      pos = _skipField(bytes, pos, wt);
    }
  }
  return dims;
}

function _parseTensorType(bytes, start, end) {
  // TypeProto.Tensor: field 2 = shape (TensorShapeProto)
  let pos = start;
  while (pos < end) {
    const { value: tag, pos: p } = _readVarint(bytes, pos); pos = p;
    const field = tag >> 3, wt = tag & 7;
    if (field === 2 && wt === 2) {
      const { value: len, pos: p2 } = _readVarint(bytes, pos); pos = p2;
      return _parseDims(bytes, pos, pos + len);
    }
    pos = _skipField(bytes, pos, wt);
  }
  return null;
}

function _parseValueInfo(bytes, start, end) {
  // ValueInfoProto: field 2 = type (TypeProto)
  // TypeProto: field 1 = tensor_type (TypeProto.Tensor)
  let pos = start;
  while (pos < end) {
    const { value: tag, pos: p } = _readVarint(bytes, pos); pos = p;
    const field = tag >> 3, wt = tag & 7;
    if (field === 2 && wt === 2) { // TypeProto
      const { value: len, pos: p2 } = _readVarint(bytes, pos); pos = p2;
      const typeStart = pos, typeEnd = pos + len;
      let tpos = typeStart;
      while (tpos < typeEnd) {
        const { value: ttag, pos: tp } = _readVarint(bytes, tpos); tpos = tp;
        const tf = ttag >> 3, twt = ttag & 7;
        if (tf === 1 && twt === 2) { // tensor_type
          const { value: tlen, pos: tp2 } = _readVarint(bytes, tpos); tpos = tp2;
          return _parseTensorType(bytes, tpos, tpos + tlen);
        }
        tpos = _skipField(bytes, tpos, twt);
      }
      pos += len;
    } else {
      pos = _skipField(bytes, pos, wt);
    }
  }
  return null;
}

function _parseGraph(bytes, start, end) {
  // GraphProto: field 11 = input (ValueInfoProto)
  let pos = start;
  while (pos < end) {
    const { value: tag, pos: p } = _readVarint(bytes, pos); pos = p;
    const field = tag >> 3, wt = tag & 7;
    if (field === 11 && wt === 2) {
      const { value: len, pos: p2 } = _readVarint(bytes, pos); pos = p2;
      const dims = _parseValueInfo(bytes, pos, pos + len);
      if (dims && dims.length >= 4) return dims;
      pos += len;
    } else {
      pos = _skipField(bytes, pos, wt);
    }
  }
  return null;
}

/**
 * Parse the first graph input's tensor shape from an ONNX model ArrayBuffer.
 * Returns [N, C, H, W] or null on failure.
 */
function getOnnxInputShape(buffer) {
  try {
    const bytes = new Uint8Array(buffer);
    let pos = 0;
    while (pos < bytes.length) {
      const { value: tag, pos: p } = _readVarint(bytes, pos); pos = p;
      const field = tag >> 3, wt = tag & 7;
      if (field === 7 && wt === 2) { // ModelProto.graph
        const { value: len, pos: p2 } = _readVarint(bytes, pos); pos = p2;
        const dims = _parseGraph(bytes, pos, pos + len);
        if (dims) return dims;
        pos += len;
      } else {
        pos = _skipField(bytes, pos, wt);
      }
    }
  } catch (e) {
    console.warn('[onnx-worker] Failed to parse ONNX input shape:', e);
  }
  return null;
}

// ── Message dispatch ──────────────────────────────────────────────────────────
self.addEventListener('message', async (e) => {
  const { type, id } = e.data;

  // ── init ──────────────────────────────────────────────────────────────────
  if (type === 'init') {
    try {
      postProgress('init-import-start');
      await ensureOrt();
      postProgress('init-runtime-ready');
      self.postMessage({ type: 'ready', id });
    } catch (err) {
      self.postMessage({ type: 'error', id, error: String(err) });
    }
    return;
  }

  // ── load-model ────────────────────────────────────────────────────────────
  if (type === 'load-model') {
    const { modelKey, modelData, modelFileName, scoreThreshold } = e.data;
    try {
      postProgress('load-model-start', { modelKey });
      const ort = await ensureOrt();
      // Free previous session for this key.
      if (sessions.has(modelKey)) {
        try { sessions.get(modelKey).session.release?.(); } catch {}
        sessions.delete(modelKey);
      }
      // Parse the model's expected input H×W before handing the buffer to ORT
      // (ORT may detach/transfer the buffer internally after create()).
      const shapeDims = getOnnxInputShape(modelData);
      // shapeDims is [N, C, H, W]; use positive static values only.
      const modelH = (shapeDims && shapeDims[2] > 0) ? shapeDims[2] : 320;
      const modelW = (shapeDims && shapeDims[3] > 0) ? shapeDims[3] : 320;
      const session = await ort.InferenceSession.create(modelData);
      const outputNames = [...session.outputNames];
      const profile = detectModelProfile(modelFileName);
      sessions.set(modelKey, {
        session,
        outputNames,
        scoreThreshold: scoreThreshold ?? 0.2,
        modelW,
        modelH,
        profile,
      });
      postProgress('load-model-done', { modelKey, modelW, modelH, outputNames, profile: profile.name });
      self.postMessage({ type: 'model-loaded', id });
    } catch (err) {
      self.postMessage({ type: 'error', id, error: String(err) });
    }
    return;
  }

  // ── detect ────────────────────────────────────────────────────────────────
  if (type === 'detect') {
    const { modelKey, imageDataBuffer, width, height } = e.data;
    try {
      const entry = sessions.get(modelKey);
      if (!entry) throw new Error('Model not loaded: ' + modelKey);
      const { session, outputNames, scoreThreshold, modelW, modelH, profile } = entry;
      const ort = globalThis.ort;

      const inputName = session.inputNames[0];
      const runWithProfile = async (activeProfile) => {
        const { tensor, transform } = buildInputTensor(
          ort,
          imageDataBuffer,
          width,
          height,
          modelW,
          modelH,
          activeProfile || MODEL_PROFILES.generic
        );
        const outputs = await session.run({ [inputName]: tensor });
        return parseDetections(
          outputs,
          outputNames,
          modelW,
          modelH,
          transform,
          width,
          height,
          scoreThreshold
        );
      };

      const detections = await runWithProfile(profile);
      self.postMessage({ type: 'detect-result', id, detections });
    } catch (err) {
      self.postMessage({ type: 'error', id, error: String(err) });
    }
    return;
  }
});
