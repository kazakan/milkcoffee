#!/usr/bin/env bash
# Build the Rust WASM module and place the generated bindings into www/pkg/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_DIR="$SCRIPT_DIR/wasm"
OUT_DIR="$SCRIPT_DIR/www/pkg"

echo "==> Building Rust (target: wasm32-unknown-unknown, profile: release)…"
cargo build \
  --manifest-path "$WASM_DIR/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --release

WASM_FILE="$WASM_DIR/target/wasm32-unknown-unknown/release/milkcoffee_wasm.wasm"

echo "==> Generating wasm-bindgen JS bindings…"
mkdir -p "$OUT_DIR"
wasm-bindgen "$WASM_FILE" \
  --out-dir "$OUT_DIR" \
  --target web

echo "==> Done. Artefacts:"
ls -lh "$OUT_DIR"
echo
echo "Serve the www/ directory with any static HTTP server, e.g.:"
echo "  python3 -m http.server --directory www 8080"
