/**
 * Pure validation and preset logic for detection settings.
 * No DOM dependencies – safe to import in Node.js unit tests.
 */

import {
  DETECTION_SCALES,
  DETECTION_SCORE_THRESHOLD,
  DETECTION_PADDING,
  DETECTION_TILE_SIZE,
  DETECTION_TILE_OVERLAP,
  DETECTION_TILE_THRESHOLD,
} from './face_detection.js';

// ─── Presets ──────────────────────────────────────────────────────────────────

export const DETECTION_PRESETS = {
  default: {
    scales: DETECTION_SCALES,
    scoreThreshold: DETECTION_SCORE_THRESHOLD,
    padding: DETECTION_PADDING,
    tileSize: DETECTION_TILE_SIZE,
    tileOverlap: DETECTION_TILE_OVERLAP,
    tileThreshold: DETECTION_TILE_THRESHOLD,
  },
  'big-image': {
    // Large crowd / high-resolution photos – enable tiling early, moderate scales.
    scales: [1, 1.5, 2, 2.5],
    scoreThreshold: 0.2,
    padding: 0.18,
    tileSize: 512,
    tileOverlap: 0.4,
    tileThreshold: 500,
  },
  'small-image': {
    // Portrait / selfie – a few prominent faces, no tiling needed.
    scales: [1, 1.5, 2, 2.5, 3],
    scoreThreshold: 0.3,
    padding: 0.2,
    tileSize: 0,
    tileOverlap: 0.3,
    tileThreshold: 99999,
  },
  'small-faces': {
    // Tiny or distant faces – aggressive upscaling and lower threshold.
    scales: [1, 1.5, 2, 2.5, 3, 3.5, 4],
    scoreThreshold: 0.15,
    padding: 0.15,
    tileSize: 480,
    tileOverlap: 0.4,
    tileThreshold: 400,
  },
  'big-faces': {
    // Close-up, prominent faces – high confidence threshold, no tiling.
    scales: [1, 1.5, 2],
    scoreThreshold: 0.4,
    padding: 0.25,
    tileSize: 0,
    tileOverlap: 0.3,
    tileThreshold: 99999,
  },
};

// ─── Pure validators ──────────────────────────────────────────────────────────

/**
 * Validate a comma-separated scales string.
 * @param {string} raw - The raw input string.
 * @returns {{ ok: true, values: number[] } | { ok: false, error: string }}
 */
export function validateScales(raw) {
  const parts = String(raw).split(',').map(s => s.trim()).filter(Boolean);
  if (parts.length === 0) {
    return { ok: false, error: 'Enter at least one scale value.' };
  }
  const nums = parts.map(Number);
  if (nums.some(isNaN)) {
    return { ok: false, error: 'All values must be numbers.' };
  }
  if (nums.some(n => n <= 0)) {
    return { ok: false, error: 'All scale values must be positive.' };
  }
  return { ok: true, values: nums };
}

/**
 * Validate a numeric field value.
 * @param {string|number} raw - The raw input value.
 * @param {{ min?: number, max?: number, label: string, integer?: boolean }} opts
 * @returns {{ ok: true, value: number } | { ok: false, error: string }}
 */
export function validateNumber(raw, { min, max, label, integer = false }) {
  const n = parseFloat(raw);
  if (isNaN(n)) {
    return { ok: false, error: `${label} must be a number.` };
  }
  if (integer && !Number.isInteger(n)) {
    return { ok: false, error: `${label} must be a whole number.` };
  }
  if (min !== undefined && n < min) {
    return { ok: false, error: `${label} must be ≥ ${min}.` };
  }
  if (max !== undefined && n > max) {
    return { ok: false, error: `${label} must be ≤ ${max}.` };
  }
  return { ok: true, value: n };
}
