# MilkCoffee

> Client-side face anonymisation — **Rust + WebAssembly**, zero-copy single-buffer design.

![UI screenshot](https://github.com/user-attachments/assets/4c84a8c3-89c5-4997-93b7-b80d607a255a)

## What it does

Upload a JPG or PNG → faces are detected → anonymisation is applied entirely inside your browser.  
**No image data leaves your device at any point.**

## Features

| Feature | Detail |
|---|---|
| Face detection | [MediaPipe Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) (BlazeFace, runs in JS) |
| Anonymisation | Mosaic (pixelation) · Box blur · Solid mask · Cyber veil · Neon blocks |
| Strength control | 0.0 – 1.0 slider |
| Detection settings | Collapsible panel to tune all detection parameters (scales, score threshold, padding, tiling) |
| Detection presets | One-click presets for common scenarios (default, big image, small image, small faces, big faces) |
| Auto-resize | Images wider than 1280 px are resized before processing |
| Download | One-click PNG export |

## Architecture

```
Browser
 ├── app.js  (orchestration, face detection via MediaPipe CDN)
 │     │
 │     ├─ alloc(size) ──────────────────────► WASM linear memory
 │     │       ↑ one copy: decoded pixels → WASM buffer
 │     ├─ process(ptr, w, h, boxes, …) ───► in-place RGBA mutation
 │     └─ Uint8ClampedArray view ◄───────── same buffer, no copy
 │
 └── www/pkg/  (wasm-bindgen generated JS + .wasm binary)
```

### Zero-copy memory design

1. `alloc(size)` allocates the pixel buffer **inside WASM memory** and returns a pointer.  
2. JavaScript creates a `Uint8ClampedArray` **view** into `wasm.memory.buffer` — no copy.  
3. Decoded image pixels are copied **once** into that view (the single unavoidable copy).  
4. `process()` modifies the buffer **in-place** — no `Vec` cloning, no return value.  
5. The canvas is rendered by reading the **same** WASM memory address — no extra copy.

## Project layout

```
milkcoffee/
├── wasm/                   # Rust crate (compiled to WASM)
│   ├── Cargo.toml
│   └── src/lib.rs          # alloc · dealloc · process (mosaic/blur/solid/cyber/neon)
├── www/                    # Static web frontend
│   ├── index.html
│   ├── app.js              # Orchestration, UI, WASM bridge
│   ├── face_detection.js   # MediaPipe face detection + tiling logic
│   ├── detection_settings.js  # Pure validation helpers + detection presets
│   └── pkg/                # wasm-bindgen output (committed for convenience)
│       ├── milkcoffee_wasm.js
│       ├── milkcoffee_wasm_bg.wasm
│       └── *.d.ts
├── tests/
│   ├── unit/
│   │   ├── face_detection.test.mjs       # Unit tests for detection logic
│   │   └── detection_settings.test.mjs  # Unit tests for validation & presets
│   └── integration/
│       └── app.spec.mjs    # Playwright end-to-end tests
└── build.sh                # Rebuild WASM from source
```

## Running locally

### Prerequisites

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
npm install
```

### Build

```bash
./build.sh
```

### Serve

```bash
python3 -m http.server --directory www 8080
# open http://localhost:8080
```

> **Note:** The app uses ES modules (`<script type="module">`), so it must be served over HTTP (not opened as a `file://` URL).

### Tests

```bash
# JS unit tests + Playwright integration tests
npm test

# Rust unit tests only
npm run test:rust
# or equivalently:
cargo test --manifest-path wasm/Cargo.toml
```

## Anonymisation methods

| Method | WASM function | Strength effect |
|---|---|---|
| Mosaic | `apply_mosaic` | block size: 4 px → half the face dimension |
| Blur | `apply_blur` | radius: 2 px → quarter the face dimension |
| Solid | `apply_solid` | N/A — always fills with black |
| Cyber veil | `apply_cyber` | grid density and scanline spacing |
| Neon blocks | `apply_neon` | block size: 6 px → third of the face dimension |

## Detection settings

The collapsible **Detection Settings** panel (closed by default) exposes all face-detection parameters:

| Field | Default | Description |
|---|---|---|
| Scales | `1, 1.5, 2, …` | Upscale factors to run the detector at |
| Score threshold | `0.2` | Minimum confidence to accept a detection |
| Padding | `0.15` | Extra margin added around each detected box |
| Tile size | `480` | Tile dimension (px) for large-image tiling; `0` disables tiling |
| Tile overlap | `0.3` | Fraction of overlap between adjacent tiles |
| Tile threshold | `640` | Image dimension (px) above which tiling activates |

Each field is validated inline on change and invalid values block processing with a descriptive error message.

### Presets

| Button | Intended for |
|---|---|
| 🔄 Default | Module-level defaults |
| 🖼 Big image | Crowd / high-resolution photos |
| 📷 Small image | Portraits and selfies |
| 🔬 Small faces | Tiny or distant faces |
| 🙂 Big faces | Close-up, prominent faces |

## Edge cases handled

- **No faces detected** — original image is shown unchanged.
- **Face box outside image bounds** — coordinates are clamped.
- **Multiple faces** — all are processed in a single WASM call.
- **Very small faces** — minimum block/radius of 1 px enforced.
