//! Backend-neutral GPTQ-to-Marlin weight preparation.
//!
//! These transforms are cold-path CPU work. Keeping them outside the CUDA
//! backend lets typed checkpoint sources prepare the same physical ABI before
//! plan-owned device storage is initialized. For INT4 without activation-order
//! permutation, the packed output is byte-for-byte equivalent to vLLM's
//! `gptq_marlin_repack` 16-by-64 tile ABI.

use rayon::prelude::*;

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
}
