//! Exact lookup for the declared staged BF16 pixel conversion. The table is
//! immutable executable metadata, not a cached image or an allocated payload.

use half::bf16;

// Nonfinite entries are deliberately unused: QwenImage21RgbaOutput validates
// the complete immutable backing before any output writer can reach this code.
static RGBA8: [u8; 1 << 16] = conversion_table();

#[inline]
pub(super) fn rgba8_finite(value: bf16) -> u8 {
    RGBA8[usize::from(value.to_bits())]
}

const fn conversion_table() -> [u8; 1 << 16] {
    let mut table = [0; 1 << 16];
    let mut bits = 0;
    while bits < table.len() {
        if bits & 0x7f80 != 0x7f80 {
            let value = bf16::from_bits(bits as u16).to_f32_const();
            let clamped = clamp(value, -1.0, 1.0);
            let scaled = bf16::from_f32_const(clamped * 0.5);
            let shifted = bf16::from_f32_const(scaled.to_f32_const() + 0.5);
            let byte = clamp(shifted.to_f32_const(), 0.0, 1.0) * 255.0;
            // round_ties_even is not const on the supported toolchain. Here
            // byte is finite in [0,255], so truncation and its exact remainder
            // implement the same rounding without a general floating cast.
            let floor = byte as u8;
            let fraction = byte - floor as f32;
            table[bits] = if fraction > 0.5 || (fraction == 0.5 && floor & 1 != 0) {
                floor + 1
            } else {
                floor
            };
        }
        bits += 1;
    }
    table
}

const fn clamp(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value < minimum {
        minimum
    } else if value > maximum {
        maximum
    } else {
        value
    }
}

#[cfg(test)]
mod tests;
