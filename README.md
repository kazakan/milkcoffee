# MilkCoffee

> Client-side face anonymisation — **Rust + WebAssembly**, zero-copy single-buffer design.

![Current UI screenshot](https://github.com/user-attachments/assets/cf2f87bf-3700-4032-9699-96ae7d3cb3ba)

## What it does

Upload a JPG or PNG, detect faces in-browser, and apply anonymisation without uploading the image itself.

## Features

| Feature | Detail |
|---|---|
| Face detection | Selectable detector: MediaPipe (BlazeFace) or SCRFD-500M (ONNX) |
| Anonymisation | Mosaic, blur, solid mask, cyber veil, neon blocks |
| Strength control | 0.00–1.00 slider |
| Detection controls | Editable scales, threshold, padding, tiling, and presets |
| ONNX execution | Runs in a worker so inference does not block the main UI thread |
| Live preview | After the first run, changing the effect or strength re-renders instantly from cached face boxes |
| Detection cache | Re-processing the same image with the same model and settings skips detection |
| Timing | Status line reports detection and effect time in ms |
| Export | Download the processed result as PNG |

## Architecture

```text
Browser
 ├── www/index.html              # UI shell
 ├── www/app.js                  # App flow, WASM bridge, detector switching
 ├── www/face_detection.js       # MediaPipe + SCRFD detection orchestration
 ├── www/onnx-detection-worker.js# ONNX Runtime Web worker
 └── www/pkg/                    # wasm-bindgen generated JS + .wasm
        │
        ├─ alloc(size) ─────────────────────► WASM linear memory
        │      ↑ one copy: decoded pixels → WASM buffer
        ├─ process(ptr, w, h, boxes, …) ───► in-place RGBA mutation
        └─ Uint8ClampedArray view ◄──────── same buffer, no extra copy
```

### Zero-copy memory design

1. `alloc(size)` allocates the pixel buffer inside WASM memory.
2. JavaScript creates a `Uint8ClampedArray` view into `wasm.memory.buffer`.
3. Decoded image pixels are copied once into that view.
4. `process()` mutates the same buffer in place.
5. The canvas renders from the same memory region.

## Project layout

```text
milkcoffee/
├── .github/workflows/pages.yml      # CI build, tests, Pages deploy
├── wasm/
│   ├── Cargo.toml
│   └── src/lib.rs                   # alloc / dealloc / process
├── www/
│   ├── index.html
│   ├── app.js
│   ├── detection_settings.js
│   ├── face_detection.js
│   ├── models/                      # gitignored local/CI-downloaded detector assets
│   ├── onnx-detection-worker.js
│   └── pkg/                         # committed wasm-bindgen output
├── tests/
│   ├── integration/app.spec.mjs
│   ├── support/static-server.mjs
│   └── unit/
│       ├── detection_settings.test.mjs
│       └── face_detection.test.mjs
├── build.sh
├── build.bat
└── README.md
```

## Running locally

### Prerequisites

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.121 --locked
npm install
```

### Optional local model files

MilkCoffee works best when detector assets are present in `www/models/`.

```bash
mkdir -p www/models
curl -fL 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite' \
  -o www/models/face_detector.tflite
curl -fL 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip' \
  -o /tmp/buffalo_sc.zip
unzip -j /tmp/buffalo_sc.zip 'det_500m.onnx' -d /tmp/buffalo_sc/
cp /tmp/buffalo_sc/det_500m.onnx www/models/scrfd_500m_bnkps.onnx
rm -rf /tmp/buffalo_sc.zip /tmp/buffalo_sc
```

### Build

```bash
# Linux / macOS
./build.sh

# Windows
build.bat
```

### Serve

```bash
python3 -m http.server --directory www 8080
# open http://localhost:8080
```

> The app uses ES modules, so it must be served over HTTP rather than opened with `file://`.

### Tests

```bash
# Rust tests
npm run test:rust

# JS unit tests
node --test tests/unit/**/*.test.mjs

# Playwright integration tests
npx playwright install chromium
npm run test:integration

# Combined JS test suite
npm test
```

## Anonymisation methods

| Method | WASM function | Strength effect |
|---|---|---|
| Mosaic | `apply_mosaic` | Block size scales with face size |
| Blur | `apply_blur` | Blur radius scales with face size |
| Solid | `apply_solid` | Always fills the region |
| Cyber veil | `apply_cyber` | Adjusts grid density and scanlines |
| Neon blocks | `apply_neon` | Block size scales with face size |

## Detection settings

The collapsible **Detection Settings** panel exposes the current MediaPipe-oriented defaults:

| Field | Default | Description |
|---|---|---|
| Scales | `1, 1.5, 2, 2.5, 3, 3.5, 4` | Upscale passes for small-face recall |
| Score threshold | `0.2` | Minimum confidence to keep a detection |
| Padding | `0` | Extra margin around each detected box |
| Tile size | `512` | Tile dimension in pixels; `0` disables tiling |
| Tile overlap | `0.3` | Overlap fraction between adjacent tiles |
| Tile threshold | `640` | Tiling activates above this image dimension |

Invalid values are validated inline and block processing until corrected.

### Detection model selection

- `MediaPipe (BlazeFace)` is the default option.
- `SCRFD-500M (ONNX)` runs through `onnxruntime-web` in `www/onnx-detection-worker.js`.
- When SCRFD is selected, the scale and tiling fields in the UI are disabled because the ONNX path uses built-in detection settings.
- If SCRFD fails to load, the app falls back to MediaPipe automatically.

### Presets

| Button | Intended for |
|---|---|
| 🔄 Default | Module defaults |
| 🖼 Big image | High-resolution or crowd photos |
| 📷 Small image | Portraits and selfies |
| 🔬 Small faces | Distant or tiny faces |
| 🙂 Big faces | Close-up faces |

## Third-party OSS notices

MilkCoffee references or downloads third-party components for optional detector runtimes and model assets:

| Component | How it is used here | Upstream notice |
|---|---|---|
| MediaPipe Tasks Vision | Browser-loaded face detector runtime for the MediaPipe option | Apache License 2.0 (`google-ai-edge/mediapipe`) |
| ONNX Runtime Web | Browser worker runtime for ONNX inference | MIT License (`microsoft/onnxruntime`) |
| InsightFace SCRFD release asset | Source archive used to obtain `det_500m.onnx` for the SCRFD option | InsightFace repository code is published under MIT, and the `detection/scrfd` subtree includes an Apache 2.0 notice from OpenMMLab |

For the SCRFD model archive specifically, review the current upstream InsightFace terms before redistribution or commercial use. Their repository README states that some trained models and related data may carry separate usage limits.

## AI agent handoff

This section is for future AI/code agents that need a fast repo-state summary:

- The user-facing app lives entirely under `www/`, with `www/app.js` as the main browser entrypoint.
- Face detection is split between MediaPipe and SCRFD/ONNX logic in `www/face_detection.js`.
- ONNX inference runs in `www/onnx-detection-worker.js`; the UI intentionally stays responsive while the worker loads or runs models.
- Rust image processing lives in `wasm/src/lib.rs`; generated web artifacts are committed under `www/pkg/`.
- CI is defined in `.github/workflows/pages.yml`: it installs Rust + Node, pins `wasm-bindgen-cli` to `0.2.121`, downloads model files into `www/models/`, runs Rust and frontend tests, and only deploys Pages on pushes to `main`.
- Local model assets in `www/models/` are ignored by git and are expected to be recreated locally or by CI.
- Playwright integration tests use `tests/support/static-server.mjs` and network stubs instead of relying on the real external CDNs during test execution.

## Edge cases handled

- No faces detected: the original image remains unchanged.
- Out-of-bounds boxes: coordinates are clamped before processing.
- Multiple faces: all boxes are processed in a single WASM call.
- Small faces: minimum block/radius safeguards are enforced.
