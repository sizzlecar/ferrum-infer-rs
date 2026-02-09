//! Quantization support for Apple GPU acceleration
//!
//! This module implements Q4_0 quantization optimized for Metal compute shaders.
//! Based on the analysis from llama.cpp's ggml-metal implementation.

/// Q4_0 quantization block (32 elements per block)
/// Compatible with llama.cpp's block_q4_0 format
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Q4_0Block {
    /// Scale factor for dequantization
    pub d: f32,
    /// 32 x 4-bit values packed as 16 bytes (2 nibbles per byte)
    pub qs: [u8; 16],
}

/// Number of elements per Q4_0 block
pub const QK4_0: usize = 32;

/// Pack f32 values to Q4_0 quantized block
/// Implements the quantization algorithm from llama.cpp
pub fn quantize_q4_0_block(src: &[f32]) -> Q4_0Block {
    assert_eq!(
        src.len(),
        QK4_0,
        "Q4_0 block must contain exactly 32 elements"
    );

    // Find absolute maximum for scale calculation
    let mut amax = 0.0f32;
    let mut max_val = 0.0f32;

    for &v in src {
        let abs_v = v.abs();
        if abs_v > amax {
            amax = abs_v;
            max_val = v;
        }
    }

    // Calculate scale (divide by 7.5 to map to [-8, 7] range)
    let d = if amax > 0.0 { max_val / -8.0 } else { 0.0 };
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 16];

    // Quantize to 4-bit values and pack into bytes
    for i in 0..16 {
        let x0 = src[i * 2 + 0] * id;
        let x1 = src[i * 2 + 1] * id;

        // Map to [0, 15] range (add 8 to shift from [-8, 7] to [0, 15])
        let xi0 = ((x0.round() as i32 + 8).clamp(0, 15)) as u8;
        let xi1 = ((x1.round() as i32 + 8).clamp(0, 15)) as u8;

        // Pack two 4-bit values into one byte
        qs[i] = xi0 | (xi1 << 4);
    }

    Q4_0Block { d, qs }
}

/// Pack Q4_0 matrix data for GPU upload (Metal-optimized format)
/// Each block becomes 5×u32: [scale, nibbles0, nibbles1, nibbles2, nibbles3]
pub fn pack_matrix_q4_0_for_gpu(weights: &[f32], nrows: usize, ncols: usize) -> Vec<u32> {
    assert_eq!(
        ncols % QK4_0,
        0,
        "Matrix columns must be divisible by QK4_0 ({})",
        QK4_0
    );

    let blocks_per_row = ncols / QK4_0;
    let mut gpu_data = Vec::with_capacity(nrows * blocks_per_row * 5);

    for row in 0..nrows {
        let row_start = row * ncols;
        let row_data = &weights[row_start..row_start + ncols];

        for block_idx in 0..blocks_per_row {
            let block_start = block_idx * QK4_0;
            let block_data = &row_data[block_start..block_start + QK4_0];

            let block = quantize_q4_0_block(block_data);

            // Pack block into GPU-friendly format:
            // word 0: scale (as u32 bits)
            gpu_data.push(f32::to_bits(block.d));

            // words 1-4: nibbles (pack 16 bytes into 4 u32)
            for chunk in block.qs.chunks_exact(4) {
                let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                gpu_data.push(word);
            }
        }
    }

    gpu_data
}

/// Parameters for Q4_0 matvec kernel
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MatvecParams {
    pub nrows: u32,
    pub ncols: u32,
    pub blocks_per_row: u32,
    pub _pad: u32,
}

// Safety: MatvecParams contains only u32 fields which are Pod
unsafe impl bytemuck::Pod for MatvecParams {}
unsafe impl bytemuck::Zeroable for MatvecParams {}

/// Dequantize Q4_0 block for CPU verification
pub fn dequantize_q4_0_block(block: &Q4_0Block) -> Vec<f32> {
    let mut result = vec![0.0f32; QK4_0];

    for i in 0..16 {
        let packed = block.qs[i];

        // Extract two 4-bit values
        let q0 = (packed & 0x0F) as f32;
        let q1 = ((packed >> 4) & 0x0F) as f32;

        // Dequantize to f32: map [0, 15] back to original range
        result[i * 2 + 0] = (q0 - 8.0) * block.d;
        result[i * 2 + 1] = (q1 - 8.0) * block.d;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        // Test quantization roundtrip
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let quantized = quantize_q4_0_block(&original);
        let dequantized = dequantize_q4_0_block(&quantized);

        // Check that quantization error is reasonable
        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(
                error < 0.05,
                "Quantization error too large: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_matrix_packing() {
        let nrows = 4;
        let ncols = 64; // 2 blocks per row
        let weights: Vec<f32> = (0..nrows * ncols).map(|i| i as f32 * 0.001).collect();

        let packed = pack_matrix_q4_0_for_gpu(&weights, nrows, ncols);

        // Should have 4 rows × 2 blocks × 5 words = 40 u32 words
        assert_eq!(packed.len(), 40);
    }
}
