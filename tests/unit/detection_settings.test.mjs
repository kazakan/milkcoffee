import test from 'node:test';
import assert from 'node:assert/strict';

import {
  validateScales,
  validateNumber,
  DETECTION_PRESETS,
} from '../../www/detection_settings.js';

// ─── validateScales ───────────────────────────────────────────────────────────

test('validateScales: rejects empty string', () => {
  const result = validateScales('');
  assert.equal(result.ok, false);
  assert.match(result.error, /at least one scale/i);
});

test('validateScales: rejects whitespace-only string', () => {
  const result = validateScales('   ');
  assert.equal(result.ok, false);
  assert.match(result.error, /at least one scale/i);
});

test('validateScales: rejects non-numeric values', () => {
  const result = validateScales('1, abc, 2');
  assert.equal(result.ok, false);
  assert.match(result.error, /numbers/i);
});

test('validateScales: rejects zero as a scale value', () => {
  const result = validateScales('1, 0, 2');
  assert.equal(result.ok, false);
  assert.match(result.error, /positive/i);
});

test('validateScales: rejects negative scale values', () => {
  const result = validateScales('1, -1.5, 2');
  assert.equal(result.ok, false);
  assert.match(result.error, /positive/i);
});

test('validateScales: accepts valid comma-separated numbers', () => {
  const result = validateScales('1, 1.5, 2, 2.5');
  assert.equal(result.ok, true);
  assert.deepEqual(result.values, [1, 1.5, 2, 2.5]);
});

test('validateScales: accepts a single scale value', () => {
  const result = validateScales('3');
  assert.equal(result.ok, true);
  assert.deepEqual(result.values, [3]);
});

test('validateScales: trims spaces around values', () => {
  const result = validateScales('  1 ,  2 ,  3  ');
  assert.equal(result.ok, true);
  assert.deepEqual(result.values, [1, 2, 3]);
});

// ─── validateNumber ───────────────────────────────────────────────────────────

test('validateNumber: rejects non-numeric string', () => {
  const result = validateNumber('abc', { label: 'Threshold' });
  assert.equal(result.ok, false);
  assert.match(result.error, /Threshold.*number/i);
});

test('validateNumber: rejects decimal when integer required', () => {
  const result = validateNumber('1.5', { label: 'Tile size', integer: true });
  assert.equal(result.ok, false);
  assert.match(result.error, /whole number/i);
});

test('validateNumber: rejects value below minimum', () => {
  const result = validateNumber('-0.1', { min: 0, label: 'Score threshold' });
  assert.equal(result.ok, false);
  assert.match(result.error, /≥ 0/);
});

test('validateNumber: rejects value above maximum', () => {
  const result = validateNumber('1.5', { max: 1, label: 'Score threshold' });
  assert.equal(result.ok, false);
  assert.match(result.error, /≤ 1/);
});

test('validateNumber: accepts valid value within range', () => {
  const result = validateNumber('0.5', { min: 0, max: 1, label: 'Score threshold' });
  assert.equal(result.ok, true);
  assert.equal(result.value, 0.5);
});

test('validateNumber: accepts exact boundary values', () => {
  assert.equal(validateNumber('0', { min: 0, max: 1, label: 'X' }).ok, true);
  assert.equal(validateNumber('1', { min: 0, max: 1, label: 'X' }).ok, true);
});

test('validateNumber: accepts integers when integer flag is true', () => {
  const result = validateNumber('512', { min: 0, label: 'Tile size', integer: true });
  assert.equal(result.ok, true);
  assert.equal(result.value, 512);
});

test('validateNumber: accepts 0 for integer fields (tile disabled)', () => {
  const result = validateNumber('0', { min: 0, label: 'Tile size', integer: true });
  assert.equal(result.ok, true);
  assert.equal(result.value, 0);
});

test('validateNumber: no min/max constraints when not specified', () => {
  const result = validateNumber('9999', { label: 'Tile threshold', integer: true });
  assert.equal(result.ok, true);
  assert.equal(result.value, 9999);
});

// ─── DETECTION_PRESETS structure ─────────────────────────────────────────────

const PRESET_KEYS = ['default', 'big-image', 'small-image', 'small-faces', 'big-faces'];

test('DETECTION_PRESETS contains all expected keys', () => {
  for (const key of PRESET_KEYS) {
    assert.ok(key in DETECTION_PRESETS, `missing preset: ${key}`);
  }
});

test('every preset has required fields with correct types', () => {
  for (const key of PRESET_KEYS) {
    const p = DETECTION_PRESETS[key];
    assert.ok(Array.isArray(p.scales), `${key}.scales should be an array`);
    assert.ok(p.scales.length > 0, `${key}.scales should not be empty`);
    assert.ok(p.scales.every(s => typeof s === 'number' && s > 0),
      `${key}.scales should be positive numbers`);
    assert.equal(typeof p.scoreThreshold, 'number', `${key}.scoreThreshold should be a number`);
    assert.ok(p.scoreThreshold >= 0 && p.scoreThreshold <= 1,
      `${key}.scoreThreshold should be between 0 and 1`);
    assert.equal(typeof p.padding, 'number', `${key}.padding should be a number`);
    assert.ok(p.padding >= 0 && p.padding <= 1, `${key}.padding should be between 0 and 1`);
    assert.equal(typeof p.tileSize, 'number', `${key}.tileSize should be a number`);
    assert.ok(p.tileSize >= 0, `${key}.tileSize should be non-negative`);
    assert.equal(typeof p.tileOverlap, 'number', `${key}.tileOverlap should be a number`);
    assert.ok(p.tileOverlap >= 0 && p.tileOverlap <= 0.9,
      `${key}.tileOverlap should be between 0 and 0.9`);
    assert.equal(typeof p.tileThreshold, 'number', `${key}.tileThreshold should be a number`);
    assert.ok(p.tileThreshold >= 0, `${key}.tileThreshold should be non-negative`);
  }
});

test('default preset matches face_detection.js constants', async () => {
  const { DETECTION_SCALES, DETECTION_SCORE_THRESHOLD, DETECTION_PADDING,
          DETECTION_TILE_SIZE, DETECTION_TILE_OVERLAP, DETECTION_TILE_THRESHOLD } =
    await import('../../www/face_detection.js');

  const p = DETECTION_PRESETS.default;
  assert.deepEqual(p.scales, DETECTION_SCALES);
  assert.equal(p.scoreThreshold, DETECTION_SCORE_THRESHOLD);
  assert.equal(p.padding, DETECTION_PADDING);
  assert.equal(p.tileSize, DETECTION_TILE_SIZE);
  assert.equal(p.tileOverlap, DETECTION_TILE_OVERLAP);
  assert.equal(p.tileThreshold, DETECTION_TILE_THRESHOLD);
});

test('small-faces preset has lower threshold and more scales than big-faces', () => {
  const sf = DETECTION_PRESETS['small-faces'];
  const bf = DETECTION_PRESETS['big-faces'];
  assert.ok(sf.scoreThreshold < bf.scoreThreshold,
    'small-faces should have a lower score threshold');
  assert.ok(sf.scales.length > bf.scales.length,
    'small-faces should use more scale steps');
});

test('small-image and big-faces presets disable tiling (tileSize === 0)', () => {
  assert.equal(DETECTION_PRESETS['small-image'].tileSize, 0);
  assert.equal(DETECTION_PRESETS['big-faces'].tileSize, 0);
});
