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
use std::ops::Range;

const FP8_E4M3_MAX: f32 = 448.0;
const FP8_F16_EXPONENT_BIAS_SCALE: f32 = 256.0;
const MAX_BLOCK_FP8_PREPARE_WORKERS: usize = 8;

/// Marlin's one-group output-channel fragment order. The native kernels use
/// this order for per-channel scales and bias vectors alike.
pub(crate) const MARLIN_CHANNEL_PERMUTATION: [usize; 32] = [
    0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7,
    14, 15, 22, 23, 30, 31,
];

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
    UnsupportedShape {
        n: usize,
        k: usize,
    },
    UnsupportedBlockShape {
        n: usize,
        k: usize,
    },
    UnsupportedGroup128Shape {
        n: usize,
        k: usize,
    },
    SourceLength {
        actual: usize,
        expected: usize,
    },
    BlockFp8ValueLength {
        actual: usize,
        expected: usize,
    },
    BlockFp8ScaleLength {
        actual: usize,
        expected: usize,
    },
    AllocationFailed {
        buffer: &'static str,
        elements: usize,
    },
    NonFiniteWeight {
        output: usize,
        input: usize,
    },
    NonFiniteBlockFp8Value {
        output: usize,
        input: usize,
        bits: u8,
    },
    NonFiniteBlockFp8Scale {
        block_output: usize,
        block_input: usize,
    },
    NonPositiveBlockFp8Scale {
        block_output: usize,
        block_input: usize,
    },
    NonFiniteDecodedBlockFp8Weight {
        output: usize,
        input: usize,
    },
    UnrepresentableMarlinScale {
        output: usize,
    },
    InvalidBoundedWorkerPartition {
        phase: &'static str,
        work_units: usize,
        elements_per_unit: usize,
        output_elements: usize,
    },
    BoundedWorkerSpawnFailed {
        phase: &'static str,
        worker: usize,
        worker_count: usize,
        reason: String,
    },
    BoundedWorkerPanicked {
        phase: &'static str,
        worker: usize,
        reason: String,
    },
}

impl fmt::Display for Fp8MarlinPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedShape { n, k } => write!(
                formatter,
                "Marlin W8A16 shape [N={n}, K={k}] does not fit a supported thread-tile family"
            ),
            Self::UnsupportedBlockShape { n, k } => write!(
                formatter,
                "block-FP8 shape [N={n}, K={k}] must have non-zero block extents"
            ),
            Self::UnsupportedGroup128Shape { n, k } => write!(
                formatter,
                "exact block-FP8 group-128 shape [N={n}, K={k}] must be divisible by 128 on both axes"
            ),
            Self::SourceLength { actual, expected } => write!(
                formatter,
                "F16 source has {actual} bytes, expected {expected}"
            ),
            Self::BlockFp8ValueLength { actual, expected } => write!(
                formatter,
                "block-FP8 values source has {actual} bytes, expected {expected}"
            ),
            Self::BlockFp8ScaleLength { actual, expected } => write!(
                formatter,
                "block-FP8 inverse-scale source has {actual} bytes, expected {expected}"
            ),
            Self::AllocationFailed { buffer, elements } => write!(
                formatter,
                "could not reserve {elements} elements for block-FP8 {buffer}"
            ),
            Self::NonFiniteWeight { output, input } => write!(
                formatter,
                "F16 source contains a non-finite value at [N={output}, K={input}]"
            ),
            Self::NonFiniteBlockFp8Value {
                output,
                input,
                bits,
            } => write!(
                formatter,
                "block-FP8 source contains non-finite E4M3 bits 0x{bits:02x} at [N={output}, K={input}]"
            ),
            Self::NonFiniteBlockFp8Scale {
                block_output,
                block_input,
            } => write!(
                formatter,
                "block-FP8 source contains a non-finite BF16 inverse scale at block [N={block_output}, K={block_input}]"
            ),
            Self::NonPositiveBlockFp8Scale {
                block_output,
                block_input,
            } => write!(
                formatter,
                "block-FP8 source contains a non-positive BF16 inverse scale at block [N={block_output}, K={block_input}]"
            ),
            Self::NonFiniteDecodedBlockFp8Weight { output, input } => write!(
                formatter,
                "block-FP8 source decodes to a non-finite value at [N={output}, K={input}]"
            ),
            Self::UnrepresentableMarlinScale { output } => write!(
                formatter,
                "block-FP8 output channel {output} requires a scale not representable by the Marlin F16 ABI"
            ),
            Self::InvalidBoundedWorkerPartition {
                phase,
                work_units,
                elements_per_unit,
                output_elements,
            } => write!(
                formatter,
                "block-FP8 {phase} worker partition is invalid: work_units={work_units}, elements_per_unit={elements_per_unit}, output_elements={output_elements}"
            ),
            Self::BoundedWorkerSpawnFailed {
                phase,
                worker,
                worker_count,
                reason,
            } => write!(
                formatter,
                "block-FP8 {phase} worker {worker}/{worker_count} could not start: {reason}"
            ),
            Self::BoundedWorkerPanicked {
                phase,
                worker,
                reason,
            } => write!(
                formatter,
                "block-FP8 {phase} worker {worker} panicked: {reason}"
            ),
        }
    }
}

impl Error for Fp8MarlinPrepareError {}

fn bounded_block_fp8_worker_count(work_units: usize, available_parallelism: usize) -> usize {
    available_parallelism
        .max(1)
        .min(MAX_BLOCK_FP8_PREPARE_WORKERS)
        .min(work_units.max(1))
}

fn block_fp8_available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

fn bounded_worker_panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|reason| (*reason).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_owned())
}

fn fill_bounded_block_fp8_ranges<T: Send>(
    phase: &'static str,
    work_units: usize,
    elements_per_unit: usize,
    available_parallelism: usize,
    output: &mut [T],
    fill: impl Fn(Range<usize>, &mut [T]) -> Result<(), Fp8MarlinPrepareError> + Sync,
) -> Result<(), Fp8MarlinPrepareError> {
    let expected_elements = work_units.checked_mul(elements_per_unit);
    if work_units == 0 || elements_per_unit == 0 || expected_elements != Some(output.len()) {
        return Err(Fp8MarlinPrepareError::InvalidBoundedWorkerPartition {
            phase,
            work_units,
            elements_per_unit,
            output_elements: output.len(),
        });
    }
    let worker_count = bounded_block_fp8_worker_count(work_units, available_parallelism);
    let units_per_worker = work_units.div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut remaining = output;
        let mut handles = Vec::with_capacity(worker_count);
        let mut spawn_error = None;
        for worker in 0..worker_count {
            let start = worker * units_per_worker;
            let end = (start + units_per_worker).min(work_units);
            if start == end {
                break;
            }
            let chunk_elements = (end - start)
                .checked_mul(elements_per_unit)
                .expect("bounded block-FP8 chunk was preflighted");
            let (chunk, tail) = remaining.split_at_mut(chunk_elements);
            remaining = tail;
            let fill = &fill;
            let range = start..end;
            match std::thread::Builder::new()
                .name(format!("block-fp8-{phase}-{worker}"))
                .spawn_scoped(scope, move || fill(range, chunk))
            {
                Ok(handle) => handles.push((worker, handle)),
                Err(error) => {
                    spawn_error = Some(Fp8MarlinPrepareError::BoundedWorkerSpawnFailed {
                        phase,
                        worker,
                        worker_count,
                        reason: error.to_string(),
                    });
                    break;
                }
            }
        }

        let mut first_worker_error = None;
        for (worker, handle) in handles {
            let result = match handle.join() {
                Ok(result) => result,
                Err(payload) => Err(Fp8MarlinPrepareError::BoundedWorkerPanicked {
                    phase,
                    worker,
                    reason: bounded_worker_panic_reason(payload),
                }),
            };
            if first_worker_error.is_none() {
                first_worker_error = result.err();
            }
        }
        first_worker_error.or(spawn_error).map_or(Ok(()), Err)
    })
}

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

/// Pack exact row-major E4M3 bytes from `[N, K]` into the u32 staging ABI
/// `[K / 4, N]` consumed by Marlin's 8-bit repacker.
///
/// This is a small-shape/reference implementation for validating the GPU
/// static transform. It never decodes or canonicalizes E4M3 values, so every
/// source bit pattern, including non-finite encodings, is preserved verbatim.
pub fn block_fp8_group128_raw_bits_to_u32_reference(
    source_fp8_e4m3: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<u32>, Fp8MarlinPrepareError> {
    pack_block_fp8_group128_matrix_slices_to_u32_reference(&[source_fp8_e4m3], n, k)
}

/// Fuse adjacent gate/up `[N, K]` source matrices into `[2N, K]` before
/// emitting the `[K / 4, 2N]` u32 staging ABI.
///
/// Concatenating independently packed gate and up buffers is incorrect because
/// the packed-input axis is outermost. This helper is the pure Rust oracle for
/// the required GPU fusion order.
pub fn block_fp8_group128_gate_up_raw_bits_to_u32_reference(
    gate_fp8_e4m3: &[u8],
    up_fp8_e4m3: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<u32>, Fp8MarlinPrepareError> {
    pack_block_fp8_group128_matrix_slices_to_u32_reference(&[gate_fp8_e4m3, up_fp8_e4m3], n, k)
}

fn pack_block_fp8_group128_matrix_slices_to_u32_reference(
    matrices: &[&[u8]],
    n: usize,
    k: usize,
) -> Result<Vec<u32>, Fp8MarlinPrepareError> {
    validate_block_fp8_group128_shape(n, k)?;
    let values_per_matrix = n
        .checked_mul(k)
        .ok_or(Fp8MarlinPrepareError::BlockFp8ValueLength {
            actual: matrices.first().map_or(0, |matrix| matrix.len()),
            expected: usize::MAX,
        })?;
    if let Some(matrix) = matrices
        .iter()
        .find(|matrix| matrix.len() != values_per_matrix)
    {
        return Err(Fp8MarlinPrepareError::BlockFp8ValueLength {
            actual: matrix.len(),
            expected: values_per_matrix,
        });
    }
    let fused_n =
        n.checked_mul(matrices.len())
            .ok_or(Fp8MarlinPrepareError::BlockFp8ValueLength {
                actual: values_per_matrix,
                expected: usize::MAX,
            })?;
    let word_count =
        (k / 4)
            .checked_mul(fused_n)
            .ok_or(Fp8MarlinPrepareError::AllocationFailed {
                buffer: "raw-bit u32 staging",
                elements: usize::MAX,
            })?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(word_count)
        .map_err(|_| Fp8MarlinPrepareError::AllocationFailed {
            buffer: "raw-bit u32 staging",
            elements: word_count,
        })?;
    packed.resize(word_count, 0_u32);
    for packed_input in 0..k / 4 {
        for fused_output in 0..fused_n {
            let matrix = matrices[fused_output / n];
            let output = fused_output % n;
            let input = packed_input * 4;
            packed[packed_input * fused_n + fused_output] = u32::from_le_bytes([
                matrix[output * k + input],
                matrix[output * k + input + 1],
                matrix[output * k + input + 2],
                matrix[output * k + input + 3],
            ]);
        }
    }
    Ok(packed)
}

/// Expand the source BF16 128x128 inverse-scale grid into Marlin's grouped
/// F16 scale ABI and apply the exponent-bias correction (`scale * 256`).
///
/// The returned flat storage has logical shape `[K / 128, N]` before Marlin's
/// P32 (`G=1`) or P64 (`G>1`) fragment permutation.
pub fn block_fp8_group128_scales_to_marlin_f16_reference(
    source_inverse_scales_bf16_le: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<half::f16>, Fp8MarlinPrepareError> {
    validate_block_fp8_group128_shape(n, k)?;
    let block_rows = n / 128;
    let group_count = k / 128;
    let scale_count =
        block_rows
            .checked_mul(group_count)
            .ok_or(Fp8MarlinPrepareError::BlockFp8ScaleLength {
                actual: source_inverse_scales_bf16_le.len(),
                expected: usize::MAX,
            })?;
    let expected_scale_bytes =
        scale_count
            .checked_mul(2)
            .ok_or(Fp8MarlinPrepareError::BlockFp8ScaleLength {
                actual: source_inverse_scales_bf16_le.len(),
                expected: usize::MAX,
            })?;
    if source_inverse_scales_bf16_le.len() != expected_scale_bytes {
        return Err(Fp8MarlinPrepareError::BlockFp8ScaleLength {
            actual: source_inverse_scales_bf16_le.len(),
            expected: expected_scale_bytes,
        });
    }

    let expanded_count =
        group_count
            .checked_mul(n)
            .ok_or(Fp8MarlinPrepareError::AllocationFailed {
                buffer: "group-128 scales",
                elements: usize::MAX,
            })?;
    let mut expanded = Vec::new();
    expanded.try_reserve_exact(expanded_count).map_err(|_| {
        Fp8MarlinPrepareError::AllocationFailed {
            buffer: "group-128 scales",
            elements: expanded_count,
        }
    })?;
    expanded.resize(expanded_count, half::f16::ZERO);
    for group in 0..group_count {
        for output in 0..n {
            let block_output = output / 128;
            let source_index = block_output * group_count + group;
            let source_offset = source_index * 2;
            let inverse_scale = half::bf16::from_le_bytes([
                source_inverse_scales_bf16_le[source_offset],
                source_inverse_scales_bf16_le[source_offset + 1],
            ])
            .to_f32();
            if !inverse_scale.is_finite() {
                return Err(Fp8MarlinPrepareError::NonFiniteBlockFp8Scale {
                    block_output,
                    block_input: group,
                });
            }
            if !(inverse_scale > 0.0) {
                return Err(Fp8MarlinPrepareError::NonPositiveBlockFp8Scale {
                    block_output,
                    block_input: group,
                });
            }
            let marlin_scale = half::f16::from_f32(inverse_scale * FP8_F16_EXPONENT_BIAS_SCALE);
            if !marlin_scale.is_finite() || marlin_scale == half::f16::ZERO {
                return Err(Fp8MarlinPrepareError::UnrepresentableMarlinScale { output });
            }
            expanded[group * n + output] = marlin_scale;
        }
    }
    Ok(repack_scales_to_marlin(&expanded, k, n, 128))
}

/// Fuse adjacent gate/up inverse-scale grids before grouped scale expansion
/// and Marlin permutation.
pub fn block_fp8_group128_gate_up_scales_to_marlin_f16_reference(
    gate_inverse_scales_bf16_le: &[u8],
    up_inverse_scales_bf16_le: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<half::f16>, Fp8MarlinPrepareError> {
    validate_block_fp8_group128_shape(n, k)?;
    let source_scale_bytes = (n / 128)
        .checked_mul(k / 128)
        .and_then(|scales| scales.checked_mul(2))
        .ok_or(Fp8MarlinPrepareError::BlockFp8ScaleLength {
            actual: gate_inverse_scales_bf16_le.len(),
            expected: usize::MAX,
        })?;
    for source in [gate_inverse_scales_bf16_le, up_inverse_scales_bf16_le] {
        if source.len() != source_scale_bytes {
            return Err(Fp8MarlinPrepareError::BlockFp8ScaleLength {
                actual: source.len(),
                expected: source_scale_bytes,
            });
        }
    }
    let fused_n = n
        .checked_mul(2)
        .ok_or(Fp8MarlinPrepareError::UnsupportedGroup128Shape { n, k })?;
    let fused_scale_bytes =
        source_scale_bytes
            .checked_mul(2)
            .ok_or(Fp8MarlinPrepareError::AllocationFailed {
                buffer: "fused gate/up inverse-scale grid",
                elements: usize::MAX,
            })?;
    let mut fused = Vec::new();
    fused.try_reserve_exact(fused_scale_bytes).map_err(|_| {
        Fp8MarlinPrepareError::AllocationFailed {
            buffer: "fused gate/up inverse-scale grid",
            elements: fused_scale_bytes,
        }
    })?;
    fused.extend_from_slice(gate_inverse_scales_bf16_le);
    fused.extend_from_slice(up_inverse_scales_bf16_le);
    block_fp8_group128_scales_to_marlin_f16_reference(&fused, fused_n, k)
}

fn validate_block_fp8_group128_shape(n: usize, k: usize) -> Result<(), Fp8MarlinPrepareError> {
    if n == 0 || k == 0 || !n.is_multiple_of(128) || !k.is_multiple_of(128) {
        return Err(Fp8MarlinPrepareError::UnsupportedGroup128Shape { n, k });
    }
    Ok(())
}

/// Reference-only conversion from block-scaled E4M3 checkpoint storage into
/// the legacy Marlin channel-wise E4M3 W8A16 ABI.
///
/// `source_fp8_e4m3` is a row-major logical `[N, K]` byte matrix. The BF16
/// inverse-scale grid is row-major `[ceil(N / block_n), ceil(K / block_k)]` and
/// decodes each source value as `E4M3(value) * inverse_scale[block]`. The
/// conversion scans the source twice and never allocates a dense `[N, K]`
/// intermediate. Product block-FP8 initialization must not call this routine;
/// it exists for numerical comparison tests against the exact group-128 path.
pub fn prepare_block_fp8_weight_for_fp8_marlin(
    source_fp8_e4m3: &[u8],
    source_inverse_scales_bf16_le: &[u8],
    n: usize,
    k: usize,
    block_shape: [usize; 2],
) -> Result<Fp8MarlinWeight, Fp8MarlinPrepareError> {
    prepare_block_fp8_weight_for_fp8_marlin_with_parallelism(
        source_fp8_e4m3,
        source_inverse_scales_bf16_le,
        n,
        k,
        block_shape,
        block_fp8_available_parallelism(),
    )
}

fn prepare_block_fp8_weight_for_fp8_marlin_with_parallelism(
    source_fp8_e4m3: &[u8],
    source_inverse_scales_bf16_le: &[u8],
    n: usize,
    k: usize,
    block_shape: [usize; 2],
    available_parallelism: usize,
) -> Result<Fp8MarlinWeight, Fp8MarlinPrepareError> {
    if !fp8_marlin_shape_supported(n, k) {
        return Err(Fp8MarlinPrepareError::UnsupportedShape { n, k });
    }
    let [block_n, block_k] = block_shape;
    if block_n == 0 || block_k == 0 {
        return Err(Fp8MarlinPrepareError::UnsupportedBlockShape {
            n: block_n,
            k: block_k,
        });
    }

    let value_count = n
        .checked_mul(k)
        .ok_or(Fp8MarlinPrepareError::BlockFp8ValueLength {
            actual: source_fp8_e4m3.len(),
            expected: usize::MAX,
        })?;
    if source_fp8_e4m3.len() != value_count {
        return Err(Fp8MarlinPrepareError::BlockFp8ValueLength {
            actual: source_fp8_e4m3.len(),
            expected: value_count,
        });
    }

    let block_rows = n.div_ceil(block_n);
    let block_columns = k.div_ceil(block_k);
    let scale_count = block_rows.checked_mul(block_columns).ok_or(
        Fp8MarlinPrepareError::BlockFp8ScaleLength {
            actual: source_inverse_scales_bf16_le.len(),
            expected: usize::MAX,
        },
    )?;
    let expected_scale_bytes =
        scale_count
            .checked_mul(2)
            .ok_or(Fp8MarlinPrepareError::BlockFp8ScaleLength {
                actual: source_inverse_scales_bf16_le.len(),
                expected: usize::MAX,
            })?;
    if source_inverse_scales_bf16_le.len() != expected_scale_bytes {
        return Err(Fp8MarlinPrepareError::BlockFp8ScaleLength {
            actual: source_inverse_scales_bf16_le.len(),
            expected: expected_scale_bytes,
        });
    }

    let mut inverse_scales = Vec::new();
    inverse_scales.try_reserve_exact(scale_count).map_err(|_| {
        Fp8MarlinPrepareError::AllocationFailed {
            buffer: "inverse-scale grid",
            elements: scale_count,
        }
    })?;
    for block_output in 0..block_rows {
        for block_input in 0..block_columns {
            let scale_index = block_output * block_columns + block_input;
            let offset = scale_index * 2;
            let scale = half::bf16::from_le_bytes([
                source_inverse_scales_bf16_le[offset],
                source_inverse_scales_bf16_le[offset + 1],
            ])
            .to_f32();
            if !scale.is_finite() {
                return Err(Fp8MarlinPrepareError::NonFiniteBlockFp8Scale {
                    block_output,
                    block_input,
                });
            }
            if !(scale > 0.0) {
                return Err(Fp8MarlinPrepareError::NonPositiveBlockFp8Scale {
                    block_output,
                    block_input,
                });
            }
            inverse_scales.push(scale);
        }
    }

    let mut raw_scales = Vec::new();
    raw_scales
        .try_reserve_exact(n)
        .map_err(|_| Fp8MarlinPrepareError::AllocationFailed {
            buffer: "channel scales",
            elements: n,
        })?;
    raw_scales.resize(n, 0.0);
    fill_bounded_block_fp8_ranges(
        "channel-scales",
        n,
        1,
        available_parallelism,
        &mut raw_scales,
        |outputs, destination| {
            for (output, raw_scale_slot) in outputs.zip(destination.iter_mut()) {
                let mut maximum = 0.0_f32;
                for input in 0..k {
                    let bits = source_fp8_e4m3[output * k + input];
                    let source_value = float8::F8E4M3::from_bits(bits).to_f32();
                    if !source_value.is_finite() {
                        return Err(Fp8MarlinPrepareError::NonFiniteBlockFp8Value {
                            output,
                            input,
                            bits,
                        });
                    }
                    let inverse_scale =
                        inverse_scales[(output / block_n) * block_columns + input / block_k];
                    let decoded = source_value * inverse_scale;
                    if !decoded.is_finite() {
                        return Err(Fp8MarlinPrepareError::NonFiniteDecodedBlockFp8Weight {
                            output,
                            input,
                        });
                    }
                    maximum = maximum.max(decoded.abs());
                }
                let raw_scale = maximum / FP8_E4M3_MAX;
                if maximum > 0.0 && raw_scale == 0.0 {
                    return Err(Fp8MarlinPrepareError::UnrepresentableMarlinScale { output });
                }
                *raw_scale_slot = raw_scale;
            }
            Ok(())
        },
    )?;

    let mut scales = Vec::new();
    scales
        .try_reserve_exact(n)
        .map_err(|_| Fp8MarlinPrepareError::AllocationFailed {
            buffer: "Marlin scales",
            elements: n,
        })?;
    scales.resize(n, half::f16::ZERO);
    for chunk_start in (0..n).step_by(MARLIN_CHANNEL_PERMUTATION.len()) {
        for (destination, source) in MARLIN_CHANNEL_PERMUTATION.iter().copied().enumerate() {
            let output = chunk_start + source;
            let raw_scale = raw_scales[output];
            let scale = half::f16::from_f32(raw_scale * FP8_F16_EXPONENT_BIAS_SCALE);
            if !scale.is_finite() || (raw_scale != 0.0 && scale == half::f16::ZERO) {
                return Err(Fp8MarlinPrepareError::UnrepresentableMarlinScale { output });
            }
            scales[chunk_start + destination] = scale;
        }
    }

    let mut packed_values = Vec::new();
    packed_values.try_reserve_exact(value_count).map_err(|_| {
        Fp8MarlinPrepareError::AllocationFailed {
            buffer: "packed values",
            elements: value_count,
        }
    })?;
    packed_values.resize(value_count, 0_u8);
    const MARLIN_TILE_ELEMENTS: usize = 16 * 64;
    let n_tiles = n / 64;
    let tile_count = value_count / MARLIN_TILE_ELEMENTS;
    fill_bounded_block_fp8_ranges(
        "marlin-tiles",
        tile_count,
        MARLIN_TILE_ELEMENTS,
        available_parallelism,
        &mut packed_values,
        |tiles, destination| {
            for (local_tile, tile) in destination
                .chunks_exact_mut(MARLIN_TILE_ELEMENTS)
                .enumerate()
            {
                let tile_index = tiles.start + local_tile;
                let k_tile = tile_index / n_tiles;
                let n_tile = tile_index % n_tiles;
                for thread in 0..32 {
                    let tensor_core_column = thread / 4;
                    let tensor_core_row = (thread % 4) * 2;
                    for warp in 0..4 {
                        let column = n_tile * 64 + warp * 16 + tensor_core_column;
                        let first = block_fp8_marlin_word(
                            source_fp8_e4m3,
                            &inverse_scales,
                            &raw_scales,
                            k,
                            block_shape,
                            block_columns,
                            k_tile,
                            tensor_core_row,
                            column,
                        );
                        let second = block_fp8_marlin_word(
                            source_fp8_e4m3,
                            &inverse_scales,
                            &raw_scales,
                            k,
                            block_shape,
                            block_columns,
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
            }
            Ok(())
        },
    )?;

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

#[inline]
#[allow(clippy::too_many_arguments)]
fn block_fp8_marlin_word(
    source_fp8_e4m3: &[u8],
    inverse_scales: &[f32],
    scales: &[f32],
    k: usize,
    block_shape: [usize; 2],
    block_columns: usize,
    k_tile: usize,
    tensor_core_row: usize,
    column: usize,
) -> u32 {
    let [block_n, block_k] = block_shape;
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
            let source_value =
                float8::F8E4M3::from_bits(source_fp8_e4m3[output * k + input]).to_f32();
            let inverse_scale =
                inverse_scales[(output / block_n) * block_columns + input / block_k];
            let decoded = source_value * inverse_scale;
            word | (u32::from(quantized_fp8_bits(decoded, scales[output])) << (byte * 8))
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
        MARLIN_CHANNEL_PERMUTATION.to_vec()
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ActiveWorker<'a>(&'a AtomicUsize);

    impl Drop for ActiveWorker<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn enter_worker<'a>(active: &'a AtomicUsize, peak: &AtomicUsize) -> ActiveWorker<'a> {
        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(current, Ordering::SeqCst);
        ActiveWorker(active)
    }

    fn f16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
        values
            .into_iter()
            .flat_map(|value| half::f16::from_f32(value).to_le_bytes())
            .collect()
    }

    fn permute_scale_chunks_reference(
        input: &[half::f16],
        permutation: &[usize],
    ) -> Vec<half::f16> {
        let mut output = vec![half::f16::ZERO; input.len()];
        for (input_chunk, output_chunk) in input
            .chunks_exact(permutation.len())
            .zip(output.chunks_exact_mut(permutation.len()))
        {
            for (destination, source) in permutation.iter().copied().enumerate() {
                output_chunk[destination] = input_chunk[source];
            }
        }
        output
    }

    #[test]
    fn group128_raw_bit_staging_is_exact_k_over_four_by_n() {
        let n = 128;
        let k = 128;
        let source = (0..n * k)
            .map(|index| (index as u8).wrapping_mul(73).wrapping_add(0x7f))
            .collect::<Vec<_>>();

        let packed = block_fp8_group128_raw_bits_to_u32_reference(&source, n, k).unwrap();
        assert_eq!(packed.len(), (k / 4) * n);
        for packed_input in 0..k / 4 {
            for output in 0..n {
                let input = packed_input * 4;
                assert_eq!(
                    packed[packed_input * n + output].to_le_bytes(),
                    source[output * k + input..output * k + input + 4],
                );
            }
        }
    }

    #[test]
    fn group128_gate_up_is_fused_before_raw_bit_staging() {
        let n = 128;
        let k = 128;
        let gate = (0..n * k)
            .map(|index| (index as u8).wrapping_mul(17).wrapping_add(3))
            .collect::<Vec<_>>();
        let up = (0..n * k)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(11))
            .collect::<Vec<_>>();

        let fused = block_fp8_group128_gate_up_raw_bits_to_u32_reference(&gate, &up, n, k).unwrap();
        let separately_concatenated = [
            block_fp8_group128_raw_bits_to_u32_reference(&gate, n, k).unwrap(),
            block_fp8_group128_raw_bits_to_u32_reference(&up, n, k).unwrap(),
        ]
        .concat();
        assert_ne!(fused, separately_concatenated);
        assert_eq!(fused.len(), (k / 4) * (2 * n));
        for packed_input in 0..k / 4 {
            for fused_output in 0..2 * n {
                let source = if fused_output < n { &gate } else { &up };
                let output = fused_output % n;
                let input = packed_input * 4;
                assert_eq!(
                    fused[packed_input * (2 * n) + fused_output].to_le_bytes(),
                    source[output * k + input..output * k + input + 4],
                );
            }
        }
    }

    #[test]
    fn group128_gate_up_scales_are_fused_before_group_major_permutation() {
        let n = 256;
        let k = 256;
        let encode = |values: [f32; 4]| {
            values
                .into_iter()
                .flat_map(|scale| half::bf16::from_f32(scale).to_le_bytes())
                .collect::<Vec<_>>()
        };
        let gate = encode([0.5, 1.0, 2.0, 4.0]);
        let up = encode([8.0, 16.0, 32.0, 64.0]);

        let fused =
            block_fp8_group128_gate_up_scales_to_marlin_f16_reference(&gate, &up, n, k).unwrap();
        let separately_concatenated = [
            block_fp8_group128_scales_to_marlin_f16_reference(&gate, n, k).unwrap(),
            block_fp8_group128_scales_to_marlin_f16_reference(&up, n, k).unwrap(),
        ]
        .concat();
        let fused_source = [gate, up].concat();
        let expected =
            block_fp8_group128_scales_to_marlin_f16_reference(&fused_source, 2 * n, k).unwrap();

        assert_eq!(fused, expected);
        assert_ne!(fused, separately_concatenated);
        assert_eq!(fused.len(), 2 * n * (k / 128));
    }

    #[test]
    fn marlin_scales_use_p32_for_one_group_and_p64_for_multiple_groups() {
        const P32: [usize; 32] = [
            0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29,
            6, 7, 14, 15, 22, 23, 30, 31,
        ];
        let p64 = (0..8)
            .flat_map(|row| (0..8).map(move |column| row + 8 * column))
            .collect::<Vec<_>>();

        let one_group = (0..128)
            .map(|index| half::f16::from_f32(index as f32))
            .collect::<Vec<_>>();
        assert_eq!(
            repack_scales_to_marlin(&one_group, 128, 128, 128),
            permute_scale_chunks_reference(&one_group, &P32)
        );

        let multiple_groups = (0..256)
            .map(|index| half::f16::from_f32(index as f32))
            .collect::<Vec<_>>();
        assert_eq!(
            repack_scales_to_marlin(&multiple_groups, 256, 128, 128),
            permute_scale_chunks_reference(&multiple_groups, &p64)
        );
    }

    #[test]
    fn group128_inverse_bf16_scales_expand_multiply_and_permute_exactly() {
        let n = 256;
        let k = 256;
        let source_scales = [0.5_f32, 1.0, 2.0, 4.0]
            .into_iter()
            .flat_map(|scale| half::bf16::from_f32(scale).to_le_bytes())
            .collect::<Vec<_>>();
        let actual =
            block_fp8_group128_scales_to_marlin_f16_reference(&source_scales, n, k).unwrap();

        let expanded = (0..k / 128)
            .flat_map(|group| {
                (0..n).map(move |output| {
                    let source = [0.5_f32, 1.0, 2.0, 4.0][(output / 128) * (k / 128) + group];
                    half::f16::from_f32(source * 256.0)
                })
            })
            .collect::<Vec<_>>();
        let p64 = (0..8)
            .flat_map(|row| (0..8).map(move |column| row + 8 * column))
            .collect::<Vec<_>>();
        assert_eq!(actual, permute_scale_chunks_reference(&expanded, &p64));
        assert_eq!(actual.len(), n * (k / 128));
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

    #[test]
    fn block_fp8_marlin_prepare_matches_dense_reference_without_dense_staging() {
        let n = 256;
        let k = 256;
        let block_shape = [128, 128];
        let inverse_scales = [0.5_f32, 2.0, 1.0, 4.0];
        let inverse_scale_bytes = inverse_scales
            .into_iter()
            .flat_map(|scale| half::bf16::from_f32(scale).to_le_bytes())
            .collect::<Vec<_>>();
        let source = (0..n * k)
            .map(|index| {
                let centered = ((index * 17 + index / 257) % 31) as f32 - 15.0;
                float8::F8E4M3::from_f32(centered / 2.0).to_bits()
            })
            .collect::<Vec<_>>();
        let dense_reference = f16_bytes((0..n).flat_map(|output| {
            let source = &source;
            let inverse_scales = &inverse_scales;
            (0..k).map(move |input| {
                let block =
                    (output / block_shape[0]) * (k / block_shape[1]) + input / block_shape[1];
                float8::F8E4M3::from_bits(source[output * k + input]).to_f32()
                    * inverse_scales[block]
            })
        }));

        let direct = prepare_block_fp8_weight_for_fp8_marlin(
            &source,
            &inverse_scale_bytes,
            n,
            k,
            block_shape,
        )
        .unwrap();
        let reference = prepare_f16_weight_for_fp8_marlin(&dense_reference, n, k).unwrap();

        assert_eq!(direct.packed_values(), reference.packed_values());
        assert_eq!(direct.scales(), reference.scales());
    }

    #[test]
    fn block_fp8_bounded_prepare_matches_one_worker_for_both_tile_families() {
        for (n, k) in [(64_usize, 128_usize), (128, 64)] {
            let block_shape = [32, 32];
            let source = (0..n * k)
                .map(|index| {
                    let value = ((index * 17 + index / 67) % 63) as f32 - 31.0;
                    float8::F8E4M3::from_f32(value / 4.0).to_bits()
                })
                .collect::<Vec<_>>();
            let scale_count = n.div_ceil(block_shape[0]) * k.div_ceil(block_shape[1]);
            let inverse_scale_bytes = (0..scale_count)
                .flat_map(|index| half::bf16::from_f32((index % 7 + 1) as f32 / 4.0).to_le_bytes())
                .collect::<Vec<_>>();

            let one = prepare_block_fp8_weight_for_fp8_marlin_with_parallelism(
                &source,
                &inverse_scale_bytes,
                n,
                k,
                block_shape,
                1,
            )
            .unwrap();
            let eight = prepare_block_fp8_weight_for_fp8_marlin_with_parallelism(
                &source,
                &inverse_scale_bytes,
                n,
                k,
                block_shape,
                8,
            )
            .unwrap();

            assert_eq!(eight.packed_values(), one.packed_values(), "[N={n}, K={k}]");
            assert_eq!(eight.scales(), one.scales(), "[N={n}, K={k}]");
        }
    }

    #[test]
    fn block_fp8_bounded_prepare_reports_earliest_row_major_source_error() {
        let n = 64;
        let k = 128;
        let mut source = vec![float8::F8E4M3::from_f32(1.0).to_bits(); n * k];
        source[12 * k + 17] = 0x7f;
        source[12 * k + 100] = 0xff;
        source[20 * k + 2] = 0xff;
        let inverse_scale_bytes = half::bf16::ONE.to_le_bytes();
        let expected = Fp8MarlinPrepareError::NonFiniteBlockFp8Value {
            output: 12,
            input: 17,
            bits: 0x7f,
        };

        for available_parallelism in [1, 8] {
            assert_eq!(
                prepare_block_fp8_weight_for_fp8_marlin_with_parallelism(
                    &source,
                    &inverse_scale_bytes,
                    n,
                    k,
                    [128, 128],
                    available_parallelism,
                ),
                Err(expected.clone())
            );
        }
    }

    #[test]
    fn block_fp8_bounded_worker_count_has_independent_hard_cap() {
        assert_eq!(bounded_block_fp8_worker_count(usize::MAX, usize::MAX), 8);
        assert_eq!(bounded_block_fp8_worker_count(3, usize::MAX), 3);
        assert_eq!(bounded_block_fp8_worker_count(64, 4), 4);
        assert_eq!(bounded_block_fp8_worker_count(64, 0), 1);
    }

    #[test]
    fn block_fp8_bounded_workers_join_after_error_and_panic() {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let started = AtomicUsize::new(0);
        let mut error_output = [0_u8; 64];
        let error = fill_bounded_block_fp8_ranges(
            "error-fixture",
            error_output.len(),
            1,
            usize::MAX,
            &mut error_output,
            |range, _| {
                started.fetch_add(1, Ordering::SeqCst);
                let _active_worker = enter_worker(&active, &peak);
                if range.start == 16 {
                    return Err(Fp8MarlinPrepareError::NonFiniteBlockFp8Value {
                        output: range.start,
                        input: 0,
                        bits: 0x7f,
                    });
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            Fp8MarlinPrepareError::NonFiniteBlockFp8Value {
                output: 16,
                input: 0,
                bits: 0x7f,
            }
        );
        assert_eq!(started.load(Ordering::SeqCst), 8);
        assert!(peak.load(Ordering::SeqCst) <= 8);
        assert_eq!(active.load(Ordering::SeqCst), 0);

        active.store(0, Ordering::SeqCst);
        peak.store(0, Ordering::SeqCst);
        started.store(0, Ordering::SeqCst);
        let mut panic_output = [0_u8; 64];
        let error = fill_bounded_block_fp8_ranges(
            "panic-fixture",
            panic_output.len(),
            1,
            usize::MAX,
            &mut panic_output,
            |range, _| {
                started.fetch_add(1, Ordering::SeqCst);
                let _active_worker = enter_worker(&active, &peak);
                if range.start == 24 {
                    panic!("bounded worker panic fixture");
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Fp8MarlinPrepareError::BoundedWorkerPanicked {
                phase: "panic-fixture",
                worker: 3,
                reason,
            } if reason.contains("bounded worker panic fixture")
        ));
        assert_eq!(started.load(Ordering::SeqCst), 8);
        assert!(peak.load(Ordering::SeqCst) <= 8);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn block_fp8_marlin_prepare_keeps_zero_channels_finite() {
        let n = 64;
        let k = 128;
        let inverse_scale = half::bf16::from_f32(2.0).to_le_bytes();
        let prepared = prepare_block_fp8_weight_for_fp8_marlin(
            &vec![0_u8; n * k],
            &inverse_scale,
            n,
            k,
            [128, 128],
        )
        .unwrap();

        assert!(prepared.packed_values().iter().all(|byte| *byte == 0));
        assert!(prepared
            .scales()
            .iter()
            .all(|scale| *scale == half::f16::ZERO));
    }

    #[test]
    fn block_fp8_marlin_prepare_rejects_invalid_source_contracts() {
        let n = 64;
        let k = 128;
        let values = vec![0_u8; n * k];
        let one_scale = half::bf16::ONE.to_le_bytes();

        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(&[], &[], 63, k, [128, 128]),
            Err(Fp8MarlinPrepareError::UnsupportedShape { n: 63, k })
        );
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(&values, &one_scale, n, k, [0, 128]),
            Err(Fp8MarlinPrepareError::UnsupportedBlockShape { n: 0, k: 128 })
        );
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(&[], &one_scale, n, k, [128, 128]),
            Err(Fp8MarlinPrepareError::BlockFp8ValueLength {
                actual: 0,
                expected: n * k,
            })
        );
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(&values, &[], n, k, [128, 128]),
            Err(Fp8MarlinPrepareError::BlockFp8ScaleLength {
                actual: 0,
                expected: 2,
            })
        );
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(
                &values,
                &half::bf16::NAN.to_le_bytes(),
                n,
                k,
                [128, 128],
            ),
            Err(Fp8MarlinPrepareError::NonFiniteBlockFp8Scale {
                block_output: 0,
                block_input: 0,
            })
        );
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(
                &values,
                &half::bf16::from_f32(-1.0).to_le_bytes(),
                n,
                k,
                [128, 128],
            ),
            Err(Fp8MarlinPrepareError::NonPositiveBlockFp8Scale {
                block_output: 0,
                block_input: 0,
            })
        );
        for zero in [half::bf16::ZERO, half::bf16::NEG_ZERO] {
            assert_eq!(
                prepare_block_fp8_weight_for_fp8_marlin(
                    &values,
                    &zero.to_le_bytes(),
                    n,
                    k,
                    [128, 128],
                ),
                Err(Fp8MarlinPrepareError::NonPositiveBlockFp8Scale {
                    block_output: 0,
                    block_input: 0,
                })
            );
        }

        let mut non_finite = values;
        non_finite[0] = 0x7f;
        assert_eq!(float8::F8E4M3::from_bits(0x7e).to_f32(), 448.0);
        assert_eq!(float8::F8E4M3::from_bits(0xfe).to_f32(), -448.0);
        assert!(!float8::F8E4M3::from_bits(non_finite[0])
            .to_f32()
            .is_finite());
        assert!(!float8::F8E4M3::from_bits(0xff).to_f32().is_finite());
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(&non_finite, &one_scale, n, k, [128, 128],),
            Err(Fp8MarlinPrepareError::NonFiniteBlockFp8Value {
                output: 0,
                input: 0,
                bits: 0x7f,
            })
        );
        non_finite[0] = 0xff;
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(&non_finite, &one_scale, n, k, [128, 128],),
            Err(Fp8MarlinPrepareError::NonFiniteBlockFp8Value {
                output: 0,
                input: 0,
                bits: 0xff,
            })
        );

        let tiny_values = vec![0x01_u8; n * k];
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(
                &tiny_values,
                &half::bf16::MIN_POSITIVE.to_le_bytes(),
                n,
                k,
                [128, 128],
            ),
            Err(Fp8MarlinPrepareError::UnrepresentableMarlinScale { output: 0 })
        );
        let unit_values = vec![float8::F8E4M3::from_f32(1.0).to_bits(); n * k];
        assert_eq!(
            prepare_block_fp8_weight_for_fp8_marlin(
                &unit_values,
                &half::bf16::MAX.to_le_bytes(),
                n,
                k,
                [128, 128],
            ),
            Err(Fp8MarlinPrepareError::UnrepresentableMarlinScale { output: 0 })
        );
    }
}
