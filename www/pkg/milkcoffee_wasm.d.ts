/* tslint:disable */
/* eslint-disable */

/**
 * Allocate `size` bytes in WASM memory, zero-initialise them, and return a
 * pointer (as u32).  JavaScript uses this offset to create a
 * Uint8ClampedArray view.  Zero-initialisation prevents unintended
 * information leakage through uninitialised memory.
 */
export function alloc(size: number): number;

/**
 * Free previously allocated memory.
 */
export function dealloc(ptr: number, size: number): void;

/**
 * Process the image in-place.
 *
 * Parameters:
 *   ptr      – pointer (as u32) to the RGBA pixel buffer allocated via `alloc`
 *   width    – image width in pixels
 *   height   – image height in pixels
 *   boxes_js – JSON array of face bounding boxes:
 *              `[{"x":N,"y":N,"width":N,"height":N}, ...]`
 *   method   – 0 = mosaic, 1 = blur, 2 = solid
 *   strength – 0.0 .. 1.0 (controls block size / blur radius)
 */
export function process(ptr: number, width: number, height: number, boxes_js: string, method: number, strength: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly alloc: (a: number) => number;
    readonly dealloc: (a: number, b: number) => void;
    readonly process: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
