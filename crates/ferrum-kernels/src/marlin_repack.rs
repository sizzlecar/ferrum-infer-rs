//! Backend-neutral GPTQ-to-Marlin weight preparation.
//!
//! These transforms are cold-path CPU work. Keeping them outside the CUDA
//! backend lets typed checkpoint sources prepare the same physical ABI before
//! plan-owned device storage is initialized.

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

/// Repack `[K/8, N]` GPTQ INT4 words into the IST-DASLab Marlin tile ABI.
pub fn repack_gptq_to_marlin(qweight_gptq: &[i32], k: usize, n: usize) -> Vec<i32> {
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
    for block in 0..(total / permutation_length) {
        let base = block * permutation_length;
        for (destination, &source) in permutation.iter().enumerate() {
            result[base + destination] = scales_gptq[base + source];
        }
    }
    let remainder = (total / permutation_length) * permutation_length;
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
}
