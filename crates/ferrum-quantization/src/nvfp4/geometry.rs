use ferrum_types::Result;

use super::invalid;

/// Logical `[out, in]` geometry and physical scale padding. Padding is storage,
/// never extra logical rows or permission to consume padded scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Geometry {
    layout: ferrum_runtime::Nvfp4MatrixLayout,
}

impl Nvfp4Geometry {
    pub fn new(rows: usize, columns: usize) -> Result<Self> {
        Ok(Self {
            layout: ferrum_runtime::Nvfp4MatrixLayout::new(rows as u64, columns as u64)
                .map_err(invalid)?,
        })
    }

    pub const fn rows(self) -> usize {
        self.layout.rows() as usize
    }
    pub const fn columns(self) -> usize {
        self.layout.columns() as usize
    }
    pub const fn element_count(self) -> usize {
        self.layout.elements() as usize
    }
    pub const fn packed_bytes(self) -> usize {
        self.layout.packed_bytes() as usize
    }
    pub const fn scale_bytes(self) -> usize {
        self.layout.scale_bytes() as usize
    }
    pub const fn packed_shape(self) -> [usize; 2] {
        [self.rows(), self.columns() / 2]
    }
    pub const fn scale_shape(self) -> [usize; 2] {
        let shape = self.layout.scale_shape();
        [shape[0] as usize, shape[1] as usize]
    }

    /// Byte index in `[row_tile, col_tile, row%32, row/32%4, col%4]`.
    /// `group` is a logical input group, not an input element index.
    pub fn scale_index(self, row: usize, group: usize) -> Result<usize> {
        self.layout
            .scale_index(row as u64, group as u64)
            .map(|index| index as usize)
            .map_err(invalid)
    }

    /// Validate the active scale domain without dequantizing unused values.
    /// Whole physical tiles contain only active scales and can be scanned in
    /// storage order. Partial tiles retain arbitrary inactive padding bytes.
    pub fn validate_scales(self, source: &[u8]) -> Result<()> {
        if source.len() != self.scale_bytes() {
            return Err(invalid("scale byte length differs from geometry"));
        }
        if self.rows() % 128 == 0 && self.columns() % 64 == 0 {
            if source.iter().any(|&code| !valid_e4m3_scale(code)) {
                return Err(invalid("block scale must be nonnegative finite E4M3FN"));
            }
        } else {
            for row in 0..self.rows() {
                for group in 0..self.columns() / 16 {
                    if !valid_e4m3_scale(source[self.scale_index(row, group)?]) {
                        return Err(invalid("block scale must be nonnegative finite E4M3FN"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Copy active scale bytes to caller-owned row-major storage. Padding is
    /// excluded. Length errors leave the destination untouched.
    pub fn unswizzle_scales_into(self, source: &[u8], output: &mut [u8]) -> Result<()> {
        if source.len() != self.scale_bytes() || output.len() != self.element_count() / 16 {
            return Err(invalid(
                "swizzled or row-major scale byte length differs from geometry",
            ));
        }
        for row in 0..self.rows() {
            for group in 0..self.columns() / 16 {
                output[row * (self.columns() / 16) + group] =
                    source[self.scale_index(row, group)?];
            }
        }
        Ok(())
    }
}

/// Decode a single E2M1 code, preserving negative zero. Values above 15 are
/// rejected instead of silently truncating an invalid source code.
pub fn decode_e2m1(code: u8) -> Result<f32> {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if code > 15 {
        return Err(invalid("E2M1 code exceeds four bits"));
    }
    let magnitude = VALUES[(code & 7) as usize];
    Ok(if code & 8 == 0 { magnitude } else { -magnitude })
}

/// Nonnegative finite E4M3FN scale. Zero is valid for an all-zero block;
/// negative values, negative zero and both NaN encodings are rejected.
pub fn decode_e4m3_scale(code: u8) -> Result<f32> {
    if !valid_e4m3_scale(code) {
        return Err(invalid("block scale must be nonnegative finite E4M3FN"));
    }
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    Ok(if exponent == 0 {
        mantissa as f32 * 2.0_f32.powi(-9)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    })
}

const fn valid_e4m3_scale(code: u8) -> bool {
    code < 0x7f
}
