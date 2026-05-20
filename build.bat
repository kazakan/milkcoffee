@echo off
setlocal enabledelayedexpansion

rem Build the Rust WASM module and place the generated bindings into www\pkg\
set "SCRIPT_DIR=%~dp0"
set "WASM_DIR=%SCRIPT_DIR%wasm"
set "OUT_DIR=%SCRIPT_DIR%www\pkg"
set "WASM_FILE=%WASM_DIR%\target\wasm32-unknown-unknown\release\milkcoffee_wasm.wasm"

echo ==^> Building Rust (target: wasm32-unknown-unknown, profile: release)...
cargo build --manifest-path "%WASM_DIR%\Cargo.toml" --target wasm32-unknown-unknown --release
if errorlevel 1 goto :error

echo ==^> Generating wasm-bindgen JS bindings...
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
wasm-bindgen "%WASM_FILE%" --out-dir "%OUT_DIR%" --target web
if errorlevel 1 goto :error

echo ==^> Done. Artefacts:
dir "%OUT_DIR%"
echo.
echo Serve the www\ directory with any static HTTP server, e.g.:
echo   python -m http.server --directory www 8080
goto :eof

:error
echo Build failed with exit code %errorlevel%.
exit /b %errorlevel%
