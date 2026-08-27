//! Backend-neutral GPTQ-to-Marlin weight preparation.
//!
//! These transforms are cold-path CPU work. Keeping them outside the CUDA
//! backend lets typed checkpoint sources prepare the same physical ABI before
//! plan-owned device storage is initialized. For INT4 without activation-order
//! permutation, the packed output is byte-for-byte equivalent to vLLM's
//! `gptq_marlin_repack` 16-by-64 tile ABI.

use rayon::prelude::*;
use std::error::Error;
use std::fmt;

const FP8_E4M3_MAX: f32 = 448.0;
const FP8_F16_EXPONENT_BIAS_SCALE: f32 = 256.0;

/// Host-prepared Marlin W8A16 storage for one logical `[N, K]` F16 matrix.
///
/// `packed_values` is the 16-by-64 Marlin tile ABI, stored as little-endian
/// words but exposed as bytes. `scales` is one permuted FP16 scale per output
/// channel with the E4M3-to-F16 exponent-bias correction already folded in.
#[derive(Debug, Clone, PartialEq)]
pub struct Fp8MarlinWeight {
    packed_values: Vec<u8>,
    scales: Vec<half::f16>,
}

impl Fp8MarlinWeight {
    pub fn packed_values(&self) -> &[u8] {
        &self.packed_values
    }

    pub fn scales(&self) -> &[half::f16] {
        &self.scales
    }

    pub fn into_parts(self) -> (Vec<u8>, Vec<half::f16>) {
        (self.packed_values, self.scales)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fp8MarlinPrepareError {
    UnsupportedShape { n: usize, k: usize },
    SourceLength { actual: usize, expected: usize },
    NonFiniteWeight { output: usize, input: usize },
}

impl fmt::Display for Fp8MarlinPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedShape { n, k } => write!(
                formatter,
                "Marlin W8A16 shape [N={n}, K={k}] does not fit a supported thread-tile family"
            ),
            Self::SourceLength { actual, expected } => write!(
                formatter,
                "F16 source has {actual} bytes, expected {expected}"
            ),
            Self::NonFiniteWeight { output, input } => write!(
                formatter,
                "F16 source contains a non-finite value at [N={output}, K={input}]"
            ),
        }
    }
}

impl Error for Fp8MarlinPrepareError {}

/// Whether `[N, K]` can be dispatched by an unpadded Marlin W8A16 kernel.
pub const fn fp8_marlin_shape_supported(n: usize, k: usize) -> bool {
    n != 0
        && k != 0
        && ((n.is_multiple_of(64) && k.is_multiple_of(128))
            || (n.is_multiple_of(128) && k.is_multiple_of(64)))
}

/// Quantize a row-major logical `[N, K]` F16 matrix into the vLLM/Marlin
/// FP16-x-E4M3 W8A16 ABI.
///
/// Quantization is symmetric and channel-wise along `N`. The input is explicit
/// little-endian bytes so mmap-backed checkpoint payloads need no aligned
/// intermediate `Vec<f16>`. Preparation is cold-path CPU work and does not
/// allocate device memory.
pub fn prepare_f16_weight_for_fp8_marlin(
    source_f16_le: &[u8],
    n: usize,
    k: usize,
) -> Result<Fp8MarlinWeight, Fp8MarlinPrepareError> {
    if !fp8_marlin_shape_supported(n, k) {
        return Err(Fp8MarlinPrepareError::UnsupportedShape { n, k });
    }
    let expected = n
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<half::f16>()))
        .ok_or(Fp8MarlinPrepareError::SourceLength {
            actual: source_f16_le.len(),
            expected: usize::MAX,
        })?;
    if source_f16_le.len() != expected {
        return Err(Fp8MarlinPrepareError::SourceLength {
            actual: source_f16_le.len(),
            expected,
        });
    }

    let raw_scales = (0..n)
        .into_par_iter()
        .map(|output| {
            let mut maximum = 0.0_f32;
            for input in 0..k {
                let value = read_f16_le(source_f16_le, output * k + input);
                if !value.is_finite() {
                    return Err(Fp8MarlinPrepareError::NonFiniteWeight { output, input });
                }
                maximum = maximum.max(value.abs());
            }
            Ok(maximum / FP8_E4M3_MAX)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n_tiles = n / 64;
    let mut packed_values = vec![0_u8; n * k];
    packed_values
        .par_chunks_mut(16 * 64)
        .enumerate()
        .for_each(|(tile_index, tile)| {
            let k_tile = tile_index / n_tiles;
            let n_tile = tile_index % n_tiles;
            for thread in 0..32 {
                let tensor_core_column = thread / 4;
                let tensor_core_row = (thread % 4) * 2;
                for warp in 0..4 {
                    let column = n_tile * 64 + warp * 16 + tensor_core_column;
                    let first = fp8_marlin_word(
                        source_f16_le,
                        &raw_scales,
                        k,
                        k_tile,
                        tensor_core_row,
                        column,
                    );
                    let second = fp8_marlin_word(
                        source_f16_le,
                        &raw_scales,
                        k,
                        k_tile,
                        tensor_core_row,
                        column + 8,
                    );
                    let output_word = thread * 8 + warp * 2;
                    tile[output_word * 4..output_word * 4 + 4]
                        .copy_from_slice(&first.to_le_bytes());
                    tile[(output_word + 1) * 4..(output_word + 1) * 4 + 4]
                        .copy_from_slice(&second.to_le_bytes());
                }
            }
        });

    let scales = raw_scales
        .into_iter()
        .map(|scale| half::f16::from_f32(scale * FP8_F16_EXPONENT_BIAS_SCALE))
        .collect::<Vec<_>>();
    let scales = repack_scales_to_marlin(&scales, k, n, k);
    Ok(Fp8MarlinWeight {
        packed_values,
        scales,
    })
}

#[inline]
fn read_f16_le(bytes: &[u8], element: usize) -> f32 {
    let offset = element * 2;
    half::f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32()
}

#[inline]
fn quantized_fp8_bits(value: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        0
    } else {
        float8::F8E4M3::from_f32(value / scale).to_bits()
    }
}

#[inline]
fn fp8_marlin_word(
    source_f16_le: &[u8],
    scales: &[f32],
    k: usize,
    k_tile: usize,
    tensor_core_row: usize,
    column: usize,
) -> u32 {
    let rows = [
        tensor_core_row,
        tensor_core_row + 8,
        tensor_core_row + 1,
        tensor_core_row + 9,
    ];
    rows.into_iter()
        .enumerate()
        .fold(0_u32, |word, (byte, row)| {
            let output = column;
            let input = k_tile * 16 + row;
            let value = read_f16_le(source_f16_le, output * k + input);
            word | (u32::from(quantized_fp8_bits(value, scales[output])) << (byte * 8))
        })
}

/// Permute GPTQ INT4 rows before Marlin repacking for activation-order models.
pub fn permute_gptq_qweight_rows(
    qweight_gptq: &[i32],
    perm: &[usize],
    k: usize,
    n: usize,
) -> Vec<i32> {
    debug_assert_eq!(perm.len(), k);
    debug_assert_eq!(qweight_gptq.len(), (k / 8) * n);

    let mut unpacked = vec![0_u8; k * n];
    let packed_rows = k / 8;
    for packed_row in 0..packed_rows {
        for column in 0..n {
            let packed = qweight_gptq[packed_row * n + column] as u32;
            for lane in 0..8 {
                unpacked[(packed_row * 8 + lane) * n + column] =
                    ((packed >> (lane * 4)) & 0xF) as u8;
            }
        }
    }

    let mut sorted = vec![0_u8; k * n];
    for row in 0..k {
        let source_row = perm[row];
        for column in 0..n {
            sorted[row * n + column] = unpacked[source_row * n + column];
        }
    }

    let mut packed = vec![0_i32; packed_rows * n];
    for packed_row in 0..packed_rows {
        for column in 0..n {
            let mut word = 0_u32;
            for lane in 0..8 {
                word |= (sorted[(packed_row * 8 + lane) * n + column] as u32) << (lane * 4);
            }
            packed[packed_row * n + column] = word as i32;
        }
    }
    packed
}

/// Repack `[K/8, N]` GPTQ INT4 words into the shared IST-DASLab/vLLM Marlin
/// 16-by-64 tile ABI.
pub fn repack_gptq_to_marlin(qweight_gptq: &[i32], k: usize, n: usize) -> Vec<i32> {
    if k.is_multiple_of(16) && n.is_multiple_of(64) {
        return repack_gptq_to_marlin_tiles(qweight_gptq, k, n);
    }

    // Preserve the legacy transform for unsupported diagnostic shapes. The
    // CUDA Marlin execution contract requires 16-by-64 alignment, but keeping
    // this fallback avoids changing the behavior of format-validation callers.
    repack_gptq_to_marlin_staged(qweight_gptq, k, n)
}

/// Write final 16-by-64 Marlin tiles directly from GPTQ words.
///
/// The staged implementation below expands every INT4 value into three full
/// byte buffers before packing the final output. Direct tile emission performs
/// the same permutation in one parallel pass and bounds temporary storage to
/// eight nibbles per output word.
fn repack_gptq_to_marlin_tiles(qweight_gptq: &[i32], k: usize, n: usize) -> Vec<i32> {
    debug_assert_eq!(qweight_gptq.len(), (k / 8) * n);
    let n_tiles = n / 64;
    let mut output = vec![0_i32; qweight_gptq.len()];
    output
        .par_chunks_mut(16 * 64 / 8)
        .enumerate()
        .for_each(|(tile_index, tile)| {
            let k_tile = tile_index / n_tiles;
            let n_tile = tile_index % n_tiles;
            for thread in 0..32 {
                for warp in 0..4 {
                    tile[thread * 4 + warp] =
                        marlin_tile_word(qweight_gptq, k_tile, n_tile, thread, warp, n);
                }
            }
        });
    output
}

/// Repack directly into a byte-oriented format-adapter destination.
///
/// This avoids materializing an intermediate `Vec<i32>` and a second encoded
/// `Vec<u8>` before a component source appends the final physical bytes.
pub fn repack_gptq_to_marlin_bytes_into(
    qweight_gptq: &[i32],
    k: usize,
    n: usize,
    output: &mut [u8],
) {
    assert_eq!(qweight_gptq.len(), (k / 8) * n);
    assert_eq!(
        output.len(),
        qweight_gptq.len() * std::mem::size_of::<i32>()
    );
    if !k.is_multiple_of(16) || !n.is_multiple_of(64) {
        for (destination, value) in
            output
                .chunks_exact_mut(4)
                .zip(repack_gptq_to_marlin_staged(qweight_gptq, k, n))
        {
            destination.copy_from_slice(&value.to_le_bytes());
        }
        return;
    }

    let n_tiles = n / 64;
    output
        .par_chunks_mut(16 * 64 / 2)
        .enumerate()
        .for_each(|(tile_index, tile)| {
            let k_tile = tile_index / n_tiles;
            let n_tile = tile_index % n_tiles;
            for thread in 0..32 {
                for warp in 0..4 {
                    let offset = (thread * 4 + warp) * 4;
                    tile[offset..offset + 4].copy_from_slice(
                        &marlin_tile_word(qweight_gptq, k_tile, n_tile, thread, warp, n)
                            .to_le_bytes(),
                    );
                }
            }
        });
}

#[inline]
fn marlin_tile_word(
    qweight_gptq: &[i32],
    k_tile: usize,
    n_tile: usize,
    thread: usize,
    warp: usize,
    n: usize,
) -> i32 {
    let tensor_core_column = thread / 4;
    let tensor_core_row = (thread % 4) * 2;
    let column = n_tile * 64 + warp * 16 + tensor_core_column;
    let mut values = [0_u32; 8];
    for (slot, row_offset) in [0_usize, 1, 8, 9].into_iter().enumerate() {
        let row = k_tile * 16 + tensor_core_row + row_offset;
        let word = qweight_gptq[(row / 8) * n + column] as u32;
        values[slot] = (word >> ((row % 8) * 4)) & 0x0f;
    }
    for (slot, row_offset) in [0_usize, 1, 8, 9].into_iter().enumerate() {
        let row = k_tile * 16 + tensor_core_row + row_offset;
        let word = qweight_gptq[(row / 8) * n + column + 8] as u32;
        values[slot + 4] = (word >> ((row % 8) * 4)) & 0x0f;
    }
    [0_usize, 2, 4, 6, 1, 3, 5, 7]
        .into_iter()
        .enumerate()
        .fold(0_u32, |word, (lane, source)| {
            word | (values[source] << (lane * 4))
        }) as i32
}

fn repack_gptq_to_marlin_staged(qweight_gptq: &[i32], k: usize, n: usize) -> Vec<i32> {
    let mut unpacked = vec![0_u8; k * n];
    unpacked
        .par_chunks_mut(8 * n)
        .zip(qweight_gptq.par_chunks(n))
        .for_each(|(unpacked_block, packed_row)| {
            for column in 0..n {
                let packed = packed_row[column];
                for lane in 0..8 {
                    unpacked_block[lane * n + column] = ((packed >> (lane * 4)) & 0xF) as u8;
                }
            }
        });

    let tile = 16;
    let n_tiles = n / tile;
    let mut tiled = vec![0_u8; k * n];
    tiled
        .par_chunks_mut(n * tile)
        .enumerate()
        .for_each(|(k_tile, tile_block)| {
            for n_tile in 0..n_tiles {
                for inner_k in 0..tile {
                    for inner_n in 0..tile {
                        let source = (k_tile * tile + inner_k) * n + (n_tile * tile + inner_n);
                        let destination = n_tile * tile * tile + inner_k * tile + inner_n;
                        tile_block[destination] = unpacked[source];
                    }
                }
            }
        });
    drop(unpacked);

    let permutation = marlin_weight_permutation();
    let total = k * n;
    let mut permuted = vec![0_u8; total];
    permuted
        .par_chunks_mut(1024)
        .zip(tiled.par_chunks(1024))
        .for_each(|(output, input)| {
            for (destination, &source) in permutation.iter().enumerate() {
                output[destination] = input[source];
            }
        });

    let mut result = vec![0_i32; total / 8];
    result
        .par_iter_mut()
        .zip(permuted.par_chunks_exact(8))
        .for_each(|(output, values)| {
            let mut word = 0_u32;
            for (lane, &value) in values.iter().enumerate() {
                word |= (value as u32) << (lane * 4);
            }
            *output = word as i32;
        });
    result
}

/// Reorder GPTQ scales into the Marlin fragment access pattern.
pub fn repack_scales_to_marlin(
    scales_gptq: &[half::f16],
    k: usize,
    n: usize,
    group_size: usize,
) -> Vec<half::f16> {
    let group_count = k / group_size;
    let permutation: Vec<usize> = if group_count > 1 {
        (0..8)
            .flat_map(|row| (0..8).map(move |column| row + 8 * column))
            .collect()
    } else {
        (0..4)
            .flat_map(|row| [0, 1, 8, 9, 16, 17, 24, 25].map(move |column| 2 * row + column))
            .collect()
    };

    let total = group_count * n;
    let permutation_length = permutation.len();
    let mut result = vec![half::f16::ZERO; total];
    let remainder = (total / permutation_length) * permutation_length;
    result[..remainder]
        .par_chunks_mut(permutation_length)
        .zip(scales_gptq[..remainder].par_chunks(permutation_length))
        .for_each(|(output, input)| {
            for (destination, &source) in permutation.iter().enumerate() {
                output[destination] = input[source];
            }
        });
    result[remainder..total].copy_from_slice(&scales_gptq[remainder..total]);
    result
}

/// Reorder compressed-tensors asymmetric INT4 zero points into Marlin's
/// runtime zero-point ABI.
///
/// The checkpoint input is standard little-endian nibble packing with shape
/// `[N/8, K/G]`. The returned I32 words have shape `[K/G, N/8]` after the
/// same 64-channel permutation used by grouped Marlin scales and Marlin's
/// eight-lane interleave.
pub fn repack_compressed_tensors_zero_points_to_marlin(
    packed_zero_points: &[i32],
    group_count: usize,
    n: usize,
) -> Vec<i32> {
    assert!(n.is_multiple_of(8));
    assert_eq!(packed_zero_points.len(), (n / 8) * group_count);

    let mut logical = vec![0_u8; group_count * n];
    for packed_output in 0..n / 8 {
        for group in 0..group_count {
            let word = packed_zero_points[packed_output * group_count + group] as u32;
            for lane in 0..8 {
                logical[group * n + packed_output * 8 + lane] = ((word >> (lane * 4)) & 0x0f) as u8;
            }
        }
    }

    let scale_permutation = (0..8)
        .flat_map(|row| (0..8).map(move |column| row + 8 * column))
        .collect::<Vec<_>>();
    let interleave = [0_usize, 2, 4, 6, 1, 3, 5, 7];
    let mut result = vec![0_i32; group_count * n / 8];
    for group in 0..group_count {
        let source = &logical[group * n..(group + 1) * n];
        let mut permuted = vec![0_u8; n];
        let full = n / scale_permutation.len() * scale_permutation.len();
        for chunk_start in (0..full).step_by(scale_permutation.len()) {
            for (destination, source_offset) in scale_permutation.iter().copied().enumerate() {
                permuted[chunk_start + destination] = source[chunk_start + source_offset];
            }
        }
        permuted[full..].copy_from_slice(&source[full..]);
        for packed_output in 0..n / 8 {
            let base = packed_output * 8;
            let word =
                interleave
                    .into_iter()
                    .enumerate()
                    .fold(0_u32, |word, (lane, source_lane)| {
                        word | (u32::from(permuted[base + source_lane]) << (lane * 4))
                    });
            result[group * (n / 8) + packed_output] = word as i32;
        }
    }
    result
}

fn marlin_weight_permutation() -> Vec<usize> {
    let mut permutation = Vec::with_capacity(1024);
    for index in 0..32 {
        let column = index / 4;
        let mut fragment = Vec::with_capacity(8);
        for block in 0..2 {
            for row in [
                2 * (index % 4),
                2 * (index % 4) + 1,
                2 * (index % 4 + 4),
                2 * (index % 4 + 4) + 1,
            ] {
                fragment.push(16 * row + column + 8 * block);
            }
        }
        for outer in 0..4 {
            permutation.extend(fragment.iter().map(|entry| entry + 256 * outer));
        }
    }
    debug_assert_eq!(permutation.len(), 1024);

    let interleave = [0_usize, 2, 4, 6, 1, 3, 5, 7];
    let mut interleaved = vec![0_usize; 1024];
    for group in 0..128 {
        for index in 0..8 {
            interleaved[group * 8 + index] = permutation[group * 8 + interleave[index]];
        }
    }
    interleaved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
        values
            .into_iter()
            .flat_map(|value| half::f16::from_f32(value).to_le_bytes())
            .collect()
    }

    fn vllm_fp8_repack_reference(
        source_f16_le: &[u8],
        n: usize,
        k: usize,
        scales: &[f32],
    ) -> Vec<u8> {
        assert_eq!(k % 16, 0);
        assert_eq!(n % 64, 0);
        let mut gptq = vec![0_u8; k * n];
        for input in 0..k {
            for output in 0..n {
                let value = read_f16_le(source_f16_le, output * k + input);
                gptq[input * n + output] = quantized_fp8_bits(value, scales[output]);
            }
        }

        let mut output = vec![0_u8; k * n];
        for k_tile in 0..k / 16 {
            for n_tile in 0..n / 64 {
                let output_base = (k_tile * (n / 64) + n_tile) * 16 * 64;
                for thread in 0..32 {
                    let tensor_core_column = thread / 4;
                    let tensor_core_row = (thread % 4) * 2;
                    for warp in 0..4 {
                        for half in 0..2 {
                            let column = n_tile * 64 + warp * 16 + tensor_core_column + half * 8;
                            let rows = [
                                tensor_core_row,
                                tensor_core_row + 8,
                                tensor_core_row + 1,
                                tensor_core_row + 9,
                            ];
                            let output_word = thread * 8 + warp * 2 + half;
                            for (byte, row) in rows.into_iter().enumerate() {
                                output[output_base + output_word * 4 + byte] =
                                    gptq[(k_tile * 16 + row) * n + column];
                            }
                        }
                    }
                }
            }
        }
        output
    }

    fn unpack_fp8_marlin_for_test(packed: &[u8], n: usize, k: usize) -> Vec<u8> {
        let mut logical = vec![0_u8; n * k];
        for k_tile in 0..k / 16 {
            for n_tile in 0..n / 64 {
                let input_base = (k_tile * (n / 64) + n_tile) * 16 * 64;
                for thread in 0..32 {
                    let tensor_core_column = thread / 4;
                    let tensor_core_row = (thread % 4) * 2;
                    for warp in 0..4 {
                        for half in 0..2 {
                            let output = n_tile * 64 + warp * 16 + tensor_core_column + half * 8;
                            let rows = [
                                tensor_core_row,
                                tensor_core_row + 8,
                                tensor_core_row + 1,
                                tensor_core_row + 9,
                            ];
                            let input_word = thread * 8 + warp * 2 + half;
                            for (byte, row) in rows.into_iter().enumerate() {
                                let input = k_tile * 16 + row;
                                logical[output * k + input] =
                                    packed[input_base + input_word * 4 + byte];
                            }
                        }
                    }
                }
            }
        }
        logical
    }

    fn unpack_channel_scales_for_test(scales: &[half::f16]) -> Vec<f32> {
        let permutation = (0..4)
            .flat_map(|row| [0, 1, 8, 9, 16, 17, 24, 25].map(move |column| 2 * row + column))
            .collect::<Vec<_>>();
        let mut logical = vec![0.0_f32; scales.len()];
        for (packed, unpacked) in scales
            .chunks_exact(permutation.len())
            .zip(logical.chunks_exact_mut(permutation.len()))
        {
            for (destination, source) in permutation.iter().copied().enumerate() {
                unpacked[source] = packed[destination].to_f32() / FP8_F16_EXPONENT_BIAS_SCALE;
            }
        }
        logical
    }

    fn vllm_int4_repack_reference(qweight: &[i32], k: usize, n: usize) -> Vec<i32> {
        assert_eq!(k % 16, 0);
        assert_eq!(n % 64, 0);
        assert_eq!(qweight.len(), (k / 8) * n);

        let mut output = vec![0_i32; (k * n) / 8];
        let pack_order = [0_usize, 2, 4, 6, 1, 3, 5, 7];
        for k_tile in 0..k / 16 {
            for n_tile in 0..n / 64 {
                let output_base = (k_tile * (n / 64) + n_tile) * (16 * 64 / 8);
                for warp in 0..4 {
                    for thread in 0..32 {
                        let tensor_core_column = thread / 4;
                        let tensor_core_row = (thread % 4) * 2;
                        let column = n_tile * 64 + warp * 16 + tensor_core_column;
                        let mut values = [0_u32; 8];
                        for (slot, row_offset) in [0_usize, 1, 8, 9].into_iter().enumerate() {
                            let row = k_tile * 16 + tensor_core_row + row_offset;
                            let word = qweight[(row / 8) * n + column] as u32;
                            values[slot] = (word >> ((row % 8) * 4)) & 0x0f;
                        }
                        for (slot, row_offset) in [0_usize, 1, 8, 9].into_iter().enumerate() {
                            let row = k_tile * 16 + tensor_core_row + row_offset;
                            let word = qweight[(row / 8) * n + column + 8] as u32;
                            values[slot + 4] = (word >> ((row % 8) * 4)) & 0x0f;
                        }
                        let packed = pack_order
                            .into_iter()
                            .enumerate()
                            .fold(0_u32, |word, (lane, source)| {
                                word | (values[source] << (lane * 4))
                            });
                        output[output_base + thread * 4 + warp] = packed as i32;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn marlin_repack_preserves_expected_storage_lengths() {
        let k = 128;
        let n = 256;
        let qweight = vec![0x7654_3210_i32; (k / 8) * n];
        let scales = vec![half::f16::ONE; n];

        assert_eq!(repack_gptq_to_marlin(&qweight, k, n).len(), qweight.len());
        assert_eq!(
            repack_scales_to_marlin(&scales, k, n, k).len(),
            scales.len()
        );
    }

    #[test]
    fn marlin_repack_matches_vllm_int4_tile_abi() {
        let k = 128;
        let n = 256;
        let qweight = (0..(k / 8) * n)
            .map(|index| {
                (index as u32)
                    .wrapping_mul(0x9e37_79b9)
                    .rotate_left((index % 31) as u32) as i32
            })
            .collect::<Vec<_>>();

        assert_eq!(
            repack_gptq_to_marlin(&qweight, k, n),
            vllm_int4_repack_reference(&qweight, k, n)
        );

        let mut bytes = vec![0_u8; qweight.len() * std::mem::size_of::<i32>()];
        repack_gptq_to_marlin_bytes_into(&qweight, k, n, &mut bytes);
        assert_eq!(
            bytes,
            vllm_int4_repack_reference(&qweight, k, n)
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn compressed_tensors_zero_points_match_grouped_marlin_abi() {
        let group_count = 4;
        let n = 128;
        let zero_point = |group: usize, output: usize| -> u8 {
            ((group * 5 + output * 3 + output / 7) & 0x0f) as u8
        };

        // compressed-tensors stores `[N / 8, K / G]` words with output
        // channels in little-endian nibble order.
        let mut checkpoint = vec![0_i32; (n / 8) * group_count];
        for packed_output in 0..n / 8 {
            for group in 0..group_count {
                checkpoint[packed_output * group_count + group] =
                    (0..8).fold(0_u32, |word, lane| {
                        word | (u32::from(zero_point(group, packed_output * 8 + lane))
                            << (lane * 4))
                    }) as i32;
            }
        }

        // Derive the expected `[K / G, N / 8]` words directly from the
        // Marlin fragment coordinates, independently of the implementation's
        // unpack/permute/repack staging.
        let interleave = [0_usize, 2, 4, 6, 1, 3, 5, 7];
        let mut expected = vec![0_i32; group_count * n / 8];
        for group in 0..group_count {
            for packed_output in 0..n / 8 {
                let chunk = (packed_output * 8 / 64) * 64;
                let destination_base = packed_output * 8 % 64;
                expected[group * (n / 8) + packed_output] =
                    interleave
                        .into_iter()
                        .enumerate()
                        .fold(0_u32, |word, (lane, fragment_lane)| {
                            let destination = destination_base + fragment_lane;
                            let source = chunk + destination / 8 + 8 * (destination % 8);
                            word | (u32::from(zero_point(group, source)) << (lane * 4))
                        }) as i32;
            }
        }

        assert_eq!(
            repack_compressed_tensors_zero_points_to_marlin(&checkpoint, group_count, n),
            expected
        );
    }

    #[test]
    fn fp8_marlin_prepare_matches_vllm_w8a16_tile_abi() {
        let n = 64_usize;
        let k = 128_usize;
        let source = f16_bytes((0..n).flat_map(|output| {
            let scale = ((output % 8) + 1) as f32 / 8.0;
            (0..k).map(move |input| {
                if input == k - 1 {
                    FP8_E4M3_MAX * scale
                } else {
                    let magnitude = ((output * 11 + input * 7) % 31 + 1) as f32;
                    if (output + input).is_multiple_of(2) {
                        magnitude * scale
                    } else {
                        -magnitude * scale
                    }
                }
            })
        }));
        let raw_scales = (0..n)
            .map(|output| ((output % 8) + 1) as f32 / 8.0)
            .collect::<Vec<_>>();

        let prepared = prepare_f16_weight_for_fp8_marlin(&source, n, k).unwrap();
        assert_eq!(
            prepared.packed_values(),
            vllm_fp8_repack_reference(&source, n, k, &raw_scales)
        );

        let expected_scales = raw_scales
            .iter()
            .map(|scale| half::f16::from_f32(scale * FP8_F16_EXPONENT_BIAS_SCALE))
            .collect::<Vec<_>>();
        assert_eq!(
            prepared.scales(),
            repack_scales_to_marlin(&expected_scales, k, n, k)
        );
    }

    #[test]
    fn f16_to_fp8_marlin_materialization_is_numerically_approximate() {
        let n = 64;
        let k = 128;
        let source = f16_bytes((0..n).flat_map(|output| {
            (0..k).map(move |input| {
                let centered = ((output * 131 + input * 17) % 2_003) as f32 - 1_001.0;
                centered / 97.0
            })
        }));
        let input = (0..k)
            .map(|index| (((index * 29) % 41) as f32 - 20.0) / 19.0)
            .collect::<Vec<_>>();
        let prepared = prepare_f16_weight_for_fp8_marlin(&source, n, k).unwrap();
        let quantized = unpack_fp8_marlin_for_test(prepared.packed_values(), n, k);
        let scales = unpack_channel_scales_for_test(prepared.scales());

        let mut exact_squared = 0.0_f64;
        let mut error_squared = 0.0_f64;
        let mut maximum_error = 0.0_f32;
        for output in 0..n {
            let mut exact = 0.0_f32;
            let mut approximate = 0.0_f32;
            for (input_index, input_value) in input.iter().copied().enumerate() {
                let source_value = read_f16_le(&source, output * k + input_index);
                let quantized_value =
                    float8::F8E4M3::from_bits(quantized[output * k + input_index]).to_f32()
                        * scales[output];
                exact += input_value * source_value;
                approximate += input_value * quantized_value;
            }
            let error = approximate - exact;
            exact_squared += f64::from(exact) * f64::from(exact);
            error_squared += f64::from(error) * f64::from(error);
            maximum_error = maximum_error.max(error.abs());
        }
        let relative_l2 = (error_squared / exact_squared).sqrt();

        assert!(relative_l2 > 1.0e-4, "relative_l2={relative_l2}");
        assert!(relative_l2 < 0.1, "relative_l2={relative_l2}");
        assert!(maximum_error > 1.0e-3, "maximum_error={maximum_error}");
    }

    #[test]
    fn fp8_marlin_prepare_keeps_zero_channels_finite() {
        let n = 64;
        let k = 128;
        let prepared = prepare_f16_weight_for_fp8_marlin(&vec![0_u8; n * k * 2], n, k).unwrap();

        assert!(prepared.packed_values().iter().all(|byte| *byte == 0));
        assert!(prepared
            .scales()
            .iter()
            .all(|scale| *scale == half::f16::ZERO));
    }

    #[test]
    fn fp8_marlin_prepare_rejects_invalid_source_contracts() {
        assert_eq!(
            prepare_f16_weight_for_fp8_marlin(&[], 63, 128),
            Err(Fp8MarlinPrepareError::UnsupportedShape { n: 63, k: 128 })
        );
        assert_eq!(
            prepare_f16_weight_for_fp8_marlin(&[], 64, 128),
            Err(Fp8MarlinPrepareError::SourceLength {
                actual: 0,
                expected: 64 * 128 * 2,
            })
        );

        let n = 64;
        let k = 128;
        let mut source = vec![0_u8; n * k * 2];
        source[..2].copy_from_slice(&half::f16::NAN.to_le_bytes());
        assert_eq!(
            prepare_f16_weight_for_fp8_marlin(&source, n, k),
            Err(Fp8MarlinPrepareError::NonFiniteWeight {
                output: 0,
                input: 0,
            })
        );
    }
}
