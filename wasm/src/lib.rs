use wasm_bindgen::prelude::*;

/// Allocate `size` bytes in WASM memory, zero-initialise them, and return a
/// pointer (as u32).  JavaScript uses this offset to create a
/// Uint8ClampedArray view.  Zero-initialisation prevents unintended
/// information leakage through uninitialised memory.
#[wasm_bindgen]
pub fn alloc(size: u32) -> u32 {
    let mut buf: Vec<u8> = vec![0u8; size as usize];
    let ptr = buf.as_mut_ptr() as u32;
    std::mem::forget(buf);
    ptr
}

/// Free previously allocated memory.
#[wasm_bindgen]
pub fn dealloc(ptr: u32, size: u32) {
    let n = size as usize;
    // Reconstruct with length == capacity so the Vec properly drops the
    // allocation (length 0 would leave the memory unreleased).
    unsafe {
        let _ = Vec::from_raw_parts(ptr as *mut u8, n, n);
    }
}

/// Anonymization methods.
/// 0 = mosaic (pixelation), 1 = box blur, 2 = solid mask,
/// 3 = cyber veil, 4 = neon blocks
#[repr(u8)]
enum Method {
    Mosaic = 0,
    Blur = 1,
    Solid = 2,
    Cyber = 3,
    Neon = 4,
}

impl From<u8> for Method {
    fn from(v: u8) -> Self {
        match v {
            1 => Method::Blur,
            2 => Method::Solid,
            3 => Method::Cyber,
            4 => Method::Neon,
            _ => Method::Mosaic,
        }
    }
}

fn region_bounds(img_w: u32, img_h: u32, rx: u32, ry: u32, rw: u32, rh: u32) -> Option<(u32, u32, u32, u32)> {
    let x0 = rx.min(img_w);
    let y0 = ry.min(img_h);
    let x1 = rx.saturating_add(rw).min(img_w);
    let y1 = ry.saturating_add(rh).min(img_h);
    if x0 >= x1 || y0 >= y1 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

fn blend_channel(base: u8, overlay: u8, alpha: f32) -> u8 {
    let alpha = alpha.clamp(0.0, 1.0);
    ((base as f32 * (1.0 - alpha)) + (overlay as f32 * alpha)).round() as u8
}

fn apply_overlay(pixels: &mut [u8], idx: usize, color: [u8; 3], alpha: f32) {
    pixels[idx] = blend_channel(pixels[idx], color[0], alpha);
    pixels[idx + 1] = blend_channel(pixels[idx + 1], color[1], alpha);
    pixels[idx + 2] = blend_channel(pixels[idx + 2], color[2], alpha);
    pixels[idx + 3] = 255;
}

/// Apply mosaic (pixelation) to a rectangular region.
/// `block` is the block size in pixels.
fn apply_mosaic(pixels: &mut [u8], img_w: u32, img_h: u32, rx: u32, ry: u32, rw: u32, rh: u32, block: u32) {
    let block = block.max(1);
    let Some((rx, ry, x_end, y_end)) = region_bounds(img_w, img_h, rx, ry, rw, rh) else {
        return;
    };

    let mut by = ry;
    while by < y_end {
        let block_h = (by + block).min(y_end) - by;
        let mut bx = rx;
        while bx < x_end {
            let block_w = (bx + block).min(x_end) - bx;

            // Compute average colour of the block.
            let (mut r, mut g, mut b, mut a) = (0u32, 0u32, 0u32, 0u32);
            let count = block_w * block_h;
            for dy in 0..block_h {
                for dx in 0..block_w {
                    let idx = ((by + dy) * img_w + (bx + dx)) as usize * 4;
                    r += pixels[idx] as u32;
                    g += pixels[idx + 1] as u32;
                    b += pixels[idx + 2] as u32;
                    a += pixels[idx + 3] as u32;
                }
            }
            let (r, g, b, a) = (
                (r / count) as u8,
                (g / count) as u8,
                (b / count) as u8,
                (a / count) as u8,
            );

            // Fill block with average.
            for dy in 0..block_h {
                for dx in 0..block_w {
                    let idx = ((by + dy) * img_w + (bx + dx)) as usize * 4;
                    pixels[idx] = r;
                    pixels[idx + 1] = g;
                    pixels[idx + 2] = b;
                    pixels[idx + 3] = a;
                }
            }

            bx += block;
        }
        by += block;
    }
}

/// Apply a simple box blur to a rectangular region.
/// `radius` is the blur radius in pixels.
fn apply_blur(pixels: &mut [u8], img_w: u32, img_h: u32, rx: u32, ry: u32, rw: u32, rh: u32, radius: u32) {
    let radius = radius.max(1);
    let Some((rx, ry, x_end, y_end)) = region_bounds(img_w, img_h, rx, ry, rw, rh) else {
        return;
    };

    // We need a temp buffer for the region to avoid reading already-written pixels.
    let region_w = (x_end - rx) as usize;
    let region_h = (y_end - ry) as usize;
    if region_w == 0 || region_h == 0 {
        return;
    }
    let mut tmp = vec![0u8; region_w * region_h * 4];

    for py in ry..y_end {
        for px in rx..x_end {
            let x0 = (px as i32 - radius as i32).max(rx as i32) as u32;
            let x1 = (px + radius + 1).min(x_end);
            let y0 = (py as i32 - radius as i32).max(ry as i32) as u32;
            let y1 = (py + radius + 1).min(y_end);
            let count = ((x1 - x0) * (y1 - y0)) as u32;
            let (mut r, mut g, mut b, mut a) = (0u32, 0u32, 0u32, 0u32);
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let idx = (sy * img_w + sx) as usize * 4;
                    r += pixels[idx] as u32;
                    g += pixels[idx + 1] as u32;
                    b += pixels[idx + 2] as u32;
                    a += pixels[idx + 3] as u32;
                }
            }
            let ti = ((py - ry) as usize * region_w + (px - rx) as usize) * 4;
            tmp[ti] = (r / count) as u8;
            tmp[ti + 1] = (g / count) as u8;
            tmp[ti + 2] = (b / count) as u8;
            tmp[ti + 3] = (a / count) as u8;
        }
    }

    // Write back.
    for py in ry..y_end {
        for px in rx..x_end {
            let src = ((py - ry) as usize * region_w + (px - rx) as usize) * 4;
            let dst = (py * img_w + px) as usize * 4;
            pixels[dst] = tmp[src];
            pixels[dst + 1] = tmp[src + 1];
            pixels[dst + 2] = tmp[src + 2];
            pixels[dst + 3] = tmp[src + 3];
        }
    }
}

/// Fill a rectangular region with a solid colour (black by default).
fn apply_solid(pixels: &mut [u8], img_w: u32, img_h: u32, rx: u32, ry: u32, rw: u32, rh: u32) {
    let Some((rx, ry, x_end, y_end)) = region_bounds(img_w, img_h, rx, ry, rw, rh) else {
        return;
    };
    for py in ry..y_end {
        for px in rx..x_end {
            let idx = (py * img_w + px) as usize * 4;
            pixels[idx] = 0;
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
            pixels[idx + 3] = 255;
        }
    }
}

fn apply_cyber(pixels: &mut [u8], img_w: u32, img_h: u32, rx: u32, ry: u32, rw: u32, rh: u32, strength: f32) {
    let Some((rx, ry, x_end, y_end)) = region_bounds(img_w, img_h, rx, ry, rw, rh) else {
        return;
    };
    let grid = (14.0 - strength * 8.0).round().max(4.0) as u32;
    let scanline = (6.0 - strength * 3.0).round().max(2.0) as u32;

    for py in ry..y_end {
        for px in rx..x_end {
            let idx = (py * img_w + px) as usize * 4;
            pixels[idx] = ((pixels[idx] as f32) * 0.12) as u8;
            pixels[idx + 1] = ((pixels[idx + 1] as f32) * 0.16) as u8;
            pixels[idx + 2] = ((pixels[idx + 2] as f32) * 0.22) as u8;
            pixels[idx + 3] = 255;

            let local_x = px - rx;
            let local_y = py - ry;
            if local_x % grid == 0 || local_y % grid == 0 {
                apply_overlay(pixels, idx, [0, 255, 255], 0.78);
            } else if local_y % scanline == 0 {
                apply_overlay(pixels, idx, [255, 0, 160], 0.42);
            } else if ((local_x + local_y) / scanline.max(1)) % 2 == 0 {
                apply_overlay(pixels, idx, [70, 0, 110], 0.18);
            }
        }
    }

    let border = 2u32;
    for py in ry..y_end {
        for px in rx..x_end {
            let on_border = px < rx + border || px >= x_end.saturating_sub(border)
                || py < ry + border || py >= y_end.saturating_sub(border);
            if on_border {
                let idx = (py * img_w + px) as usize * 4;
                apply_overlay(pixels, idx, [255, 64, 188], 0.85);
            }
        }
    }
}

fn apply_neon(pixels: &mut [u8], img_w: u32, img_h: u32, rx: u32, ry: u32, rw: u32, rh: u32, block: u32) {
    let Some((rx, ry, x_end, y_end)) = region_bounds(img_w, img_h, rx, ry, rw, rh) else {
        return;
    };
    let block = block.max(3);
    let palette = [
        [8u8, 12u8, 28u8],
        [0u8, 255u8, 255u8],
        [255u8, 0u8, 153u8],
        [255u8, 214u8, 10u8],
        [124u8, 58u8, 237u8],
    ];

    let mut by = ry;
    while by < y_end {
        let block_h = (by + block).min(y_end) - by;
        let mut bx = rx;
        while bx < x_end {
            let block_w = (bx + block).min(x_end) - bx;
            let mut luminance_total = 0u32;
            let mut count = 0u32;
            for dy in 0..block_h {
                for dx in 0..block_w {
                    let idx = ((by + dy) * img_w + (bx + dx)) as usize * 4;
                    luminance_total += pixels[idx] as u32 + pixels[idx + 1] as u32 + pixels[idx + 2] as u32;
                    count += 1;
                }
            }
            let avg_luma = if count == 0 { 0 } else { luminance_total / (count * 3) };
            let palette_idx = (((avg_luma / 52) + ((bx - rx) / block) + ((by - ry) / block)) as usize) % palette.len();
            let color = palette[palette_idx];

            for dy in 0..block_h {
                for dx in 0..block_w {
                    let idx = ((by + dy) * img_w + (bx + dx)) as usize * 4;
                    let edge = dx == 0 || dy == 0 || dx + 1 == block_w || dy + 1 == block_h;
                    let alpha = if edge { 0.9 } else { 0.7 };
                    apply_overlay(pixels, idx, color, alpha);
                }
            }

            bx += block;
        }
        by += block;
    }
}

/// Process the image in-place.
///
/// Parameters:
///   ptr      – pointer (as u32) to the RGBA pixel buffer allocated via `alloc`
///   width    – image width in pixels
///   height   – image height in pixels
///   boxes_js – JSON array of face bounding boxes:
///              `[{"x":N,"y":N,"width":N,"height":N}, ...]`
///   method   – 0 = mosaic, 1 = blur, 2 = solid, 3 = cyber veil, 4 = neon blocks
///   strength – 0.0 .. 1.0 (controls block size / blur radius)
#[wasm_bindgen]
pub fn process(ptr: u32, width: u32, height: u32, boxes_js: &str, method: u8, strength: f32) {
    let size = (width * height * 4) as usize;
    let pixels = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size) };
    let method = Method::from(method);
    let strength = strength.clamp(0.0, 1.0);

    // Parse face boxes from JSON without pulling in serde.
    // Expected format: [{"x":N,"y":N,"width":N,"height":N}, ...]
    let boxes = parse_boxes(boxes_js);

    for (bx, by, bw, bh) in boxes {
        match method {
            Method::Mosaic => {
                // block size: 4 .. max_dim * 0.5 mapped by strength
                let max_block = ((bw.min(bh) / 2).max(4)) as f32;
                let block = (4.0 + strength * (max_block - 4.0)).round() as u32;
                apply_mosaic(pixels, width, height, bx, by, bw, bh, block);
            }
            Method::Blur => {
                let max_radius = ((bw.min(bh) / 4).max(2)) as f32;
                let radius = (2.0 + strength * (max_radius - 2.0)).round() as u32;
                apply_blur(pixels, width, height, bx, by, bw, bh, radius);
            }
            Method::Solid => {
                apply_solid(pixels, width, height, bx, by, bw, bh);
            }
            Method::Cyber => {
                apply_cyber(pixels, width, height, bx, by, bw, bh, strength);
            }
            Method::Neon => {
                let max_block = ((bw.min(bh) / 3).max(6)) as f32;
                let block = (6.0 + strength * (max_block - 6.0)).round() as u32;
                apply_neon(pixels, width, height, bx, by, bw, bh, block);
            }
        }
    }
}

/// Minimal JSON parser for the face-box array.
/// Returns a Vec of (x, y, width, height) tuples.
fn parse_boxes(json: &str) -> Vec<(u32, u32, u32, u32)> {
    let mut result = Vec::new();
    // Strip outer brackets.
    let json = json.trim();
    if !json.starts_with('[') {
        return result;
    }
    let json = &json[1..json.len().saturating_sub(1)];

    // Split objects by "},".
    for obj in json.split("},") {
        let obj = obj.trim().trim_start_matches('{').trim_end_matches('}');
        let (mut x, mut y, mut w, mut h) = (0u32, 0u32, 0u32, 0u32);
        for kv in obj.split(',') {
            let mut parts = kv.splitn(2, ':');
            let key = parts.next().unwrap_or("").trim().trim_matches('"');
            let val: u32 = parts
                .next()
                .unwrap_or("0")
                .trim()
                .trim_matches('"')
                .parse()
                .unwrap_or(0);
            match key {
                "x" => x = val,
                "y" => y = val,
                "width" => w = val,
                "height" => h = val,
                _ => {}
            }
        }
        if w > 0 && h > 0 {
            result.push((x, y, w, h));
        }
    }
    result
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_boxes ────────────────────────────────────────────────────────────

    #[test]
    fn parse_boxes_empty_array() {
        assert!(parse_boxes("[]").is_empty());
    }

    #[test]
    fn parse_boxes_not_array() {
        assert!(parse_boxes("{}").is_empty());
        assert!(parse_boxes("").is_empty());
    }

    #[test]
    fn parse_boxes_single_box() {
        let boxes = parse_boxes(r#"[{"x":10,"y":20,"width":30,"height":40}]"#);
        assert_eq!(boxes, vec![(10, 20, 30, 40)]);
    }

    #[test]
    fn parse_boxes_multiple_boxes() {
        let boxes = parse_boxes(r#"[{"x":0,"y":0,"width":10,"height":10},{"x":50,"y":60,"width":20,"height":25}]"#);
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0], (0, 0, 10, 10));
        assert_eq!(boxes[1], (50, 60, 20, 25));
    }

    #[test]
    fn parse_boxes_skips_zero_dimension_boxes() {
        // width=0: should be skipped.
        let boxes = parse_boxes(r#"[{"x":0,"y":0,"width":0,"height":10}]"#);
        assert!(boxes.is_empty());
        // height=0: should be skipped.
        let boxes = parse_boxes(r#"[{"x":0,"y":0,"width":10,"height":0}]"#);
        assert!(boxes.is_empty());
    }

    #[test]
    fn parse_boxes_default_x_y_are_zero() {
        // x and y are optional and default to 0.
        let boxes = parse_boxes(r#"[{"width":5,"height":5}]"#);
        assert_eq!(boxes, vec![(0, 0, 5, 5)]);
    }

    // ── region_bounds ─────────────────────────────────────────────────────────

    #[test]
    fn region_bounds_normal_region() {
        let b = region_bounds(100, 100, 10, 20, 30, 40);
        assert_eq!(b, Some((10, 20, 40, 60)));
    }

    #[test]
    fn region_bounds_clamps_to_image() {
        // Region extends beyond the image boundary.
        let b = region_bounds(50, 50, 40, 40, 20, 20);
        assert_eq!(b, Some((40, 40, 50, 50)));
    }

    #[test]
    fn region_bounds_returns_none_for_empty_region() {
        // rx >= img_w after clamping → empty.
        assert_eq!(region_bounds(10, 10, 10, 0, 5, 5), None);
        assert_eq!(region_bounds(10, 10, 0, 10, 5, 5), None);
    }

    // ── blend_channel ─────────────────────────────────────────────────────────

    #[test]
    fn blend_channel_alpha_zero_returns_base() {
        assert_eq!(blend_channel(100, 200, 0.0), 100);
    }

    #[test]
    fn blend_channel_alpha_one_returns_overlay() {
        assert_eq!(blend_channel(100, 200, 1.0), 200);
    }

    #[test]
    fn blend_channel_alpha_half_blends() {
        let result = blend_channel(0, 100, 0.5);
        assert_eq!(result, 50);
    }

    #[test]
    fn blend_channel_clamps_alpha_above_one() {
        // Should behave as if alpha = 1.0.
        assert_eq!(blend_channel(100, 200, 2.0), 200);
    }

    #[test]
    fn blend_channel_clamps_alpha_below_zero() {
        // Should behave as if alpha = 0.0.
        assert_eq!(blend_channel(100, 200, -1.0), 100);
    }

    // ── apply_solid ───────────────────────────────────────────────────────────

    fn make_pixels(w: u32, h: u32, fill: u8) -> Vec<u8> {
        vec![fill; (w * h * 4) as usize]
    }

    #[test]
    fn apply_solid_fills_region_with_black() {
        let w = 10u32;
        let h = 10u32;
        let mut pixels = make_pixels(w, h, 255);
        apply_solid(&mut pixels, w, h, 2, 3, 4, 4);

        for py in 3..7 {
            for px in 2..6 {
                let idx = (py * w + px) as usize * 4;
                assert_eq!(pixels[idx],     0, "R should be 0 at ({px},{py})");
                assert_eq!(pixels[idx + 1], 0, "G should be 0 at ({px},{py})");
                assert_eq!(pixels[idx + 2], 0, "B should be 0 at ({px},{py})");
                assert_eq!(pixels[idx + 3], 255, "A should be 255 at ({px},{py})");
            }
        }
    }

    #[test]
    fn apply_solid_does_not_touch_pixels_outside_region() {
        let w = 10u32;
        let h = 10u32;
        let mut pixels = make_pixels(w, h, 200);
        apply_solid(&mut pixels, w, h, 3, 3, 2, 2);

        // Top-left corner (0,0) should be untouched.
        assert_eq!(pixels[0], 200);
        assert_eq!(pixels[1], 200);
        assert_eq!(pixels[2], 200);
        assert_eq!(pixels[3], 200);
    }

    #[test]
    fn apply_solid_zero_dimension_region_is_no_op() {
        let w = 10u32;
        let h = 10u32;
        let original = make_pixels(w, h, 128);
        let mut pixels = original.clone();
        // width = 0 → region_bounds returns None → no writes.
        apply_solid(&mut pixels, w, h, 0, 0, 0, 5);
        assert_eq!(pixels, original);
    }

    // ── apply_mosaic ──────────────────────────────────────────────────────────

    #[test]
    fn apply_mosaic_uniform_region_unchanged() {
        // A uniform colour region should look identical after mosaic.
        let w = 8u32;
        let h = 8u32;
        let mut pixels = make_pixels(w, h, 120);
        let original = pixels.clone();
        apply_mosaic(&mut pixels, w, h, 0, 0, w, h, 2);
        assert_eq!(pixels, original, "uniform image should be unchanged by mosaic");
    }

    #[test]
    fn apply_mosaic_averages_block_pixels() {
        // 2×2 image, block size 2 – the entire image is one block.
        // Pixels: (0,100,0,255) and (0,200,0,255) on row 0; same on row 1.
        let w = 2u32;
        let h = 2u32;
        let mut pixels = vec![
            0, 100, 0, 255, // (0,0)
            0, 200, 0, 255, // (1,0)
            0, 100, 0, 255, // (0,1)
            0, 200, 0, 255, // (1,1)
        ];
        apply_mosaic(&mut pixels, w, h, 0, 0, w, h, 2);
        // Average green = (100+200+100+200)/4 = 150.
        for i in 0..4 {
            let idx = i * 4;
            assert_eq!(pixels[idx],     0,   "R should be 0");
            assert_eq!(pixels[idx + 1], 150, "G should be 150 (average)");
            assert_eq!(pixels[idx + 2], 0,   "B should be 0");
            assert_eq!(pixels[idx + 3], 255, "A should be 255");
        }
    }

    // ── apply_blur ────────────────────────────────────────────────────────────

    #[test]
    fn apply_blur_uniform_region_unchanged() {
        let w = 8u32;
        let h = 8u32;
        let mut pixels = make_pixels(w, h, 80);
        let original = pixels.clone();
        apply_blur(&mut pixels, w, h, 0, 0, w, h, 2);
        assert_eq!(pixels, original, "uniform image should be unchanged by blur");
    }

    #[test]
    fn apply_blur_does_not_touch_pixels_outside_region() {
        let w = 10u32;
        let h = 10u32;
        // Fill image with 100, put a bright spot inside the blur region.
        let mut pixels = make_pixels(w, h, 100);
        // Set centre pixel to 255 in all channels.
        let cx = (5 * w + 5) as usize * 4;
        pixels[cx] = 255;
        pixels[cx + 1] = 255;
        pixels[cx + 2] = 255;

        // Blur only the centre 4×4 area.
        apply_blur(&mut pixels, w, h, 4, 4, 4, 4, 1);

        // Pixel at (0,0) must be untouched.
        assert_eq!(pixels[0], 100);
        assert_eq!(pixels[1], 100);
        assert_eq!(pixels[2], 100);
    }

    // ── Method dispatch (process-level) ──────────────────────────────────────

    #[test]
    fn process_solid_method_blackens_face_region() {
        let w = 20u32;
        let h = 20u32;
        let size = (w * h * 4) as usize;
        // Fill with white.
        let mut pixels = vec![255u8; size];

        // Apply the solid method directly (method 2).
        let boxes = parse_boxes(r#"[{"x":5,"y":5,"width":10,"height":10}]"#);
        for (bx, by, bw, bh) in boxes {
            apply_solid(&mut pixels, w, h, bx, by, bw, bh);
        }

        // Pixels inside the region (5,5)..(15,15) must be black.
        for py in 5..15 {
            for px in 5..15 {
                let idx = (py * w + px) as usize * 4;
                assert_eq!(pixels[idx],     0, "R@({px},{py})");
                assert_eq!(pixels[idx + 1], 0, "G@({px},{py})");
                assert_eq!(pixels[idx + 2], 0, "B@({px},{py})");
            }
        }

        // Pixels outside the region must remain white.
        assert_eq!(pixels[0], 255); // (0,0)
        assert_eq!(pixels[1], 255);
        assert_eq!(pixels[2], 255);
    }
}
