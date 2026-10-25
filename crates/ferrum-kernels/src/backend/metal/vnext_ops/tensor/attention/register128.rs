//! Optional Head128 SIMD path with register probability and output fragments.
use super::super::*;

pub(super) const QUERY_ROWS: u64 = 32;
// Flags, BF16 Q[32][136], and aliased K[128][24] / (V[16][136] + F32 P[32][16]).
pub(super) const SHARED_BYTES: u64 = 16 + 32 * 136 * 2 + 6400;
pub(in crate::backend::metal::vnext_ops::tensor) const FINGERPRINT_SOURCE: &str = concat!(
    include_str!("register128.rs"),
    include_str!("register128_calibration.metal"),
    include_str!("register128.metal")
);
pub(in crate::backend::metal::vnext_ops::tensor) const NUMERICAL_IDENTITY: &str =
    "metal.attention.head128.simdgroup-f32-bq32-bk16;ushort-staging-exact-bf16-decode;register-p2-o16;coordinate-calibrated-xor1-xor8-or-dynamic-four-peer-or-full-shuffle-row-reduction;f32-exp;finite-pv-guarded-explicit-f32-rescale-then-inplace-mma-c;sticky-v-above-2p93-or-o-nonfinite-above-2p124-or-p-alpha-outside-0-2-fallback-zero-pv-mma-separate-add;nonfinite-v-visible-only-scalar-pv-then-separate-rescale-add;key-runs-or-validated-all-batch-all-valid-specialization;same-causal-segment-tail-ieee-and-arithmetic-order;bf16-final-only;shared15120;fast-math=false.v4";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KeyVisibility {
    MaskRuns,
    AllValid,
}

impl KeyVisibility {
    pub(super) fn from_mask(mask: &super::StructuredAttentionBf16Mask, sequence: u32) -> Self {
        // The operation contract requires positive, covering, alternating runs.
        // All keys are valid exactly when every batch has one complete valid run.
        // This proves nothing about causal/segment visibility, which stays intact.
        if (0..mask.batch()).all(|batch| {
            mask.key_run_ends(batch) == Some([sequence].as_slice())
                && mask.key_run_valid(batch) == Some([true].as_slice())
        }) {
            Self::AllValid
        } else {
            Self::MaskRuns
        }
    }
}

#[cfg(test)]
pub(in crate::backend::metal::vnext_ops::tensor) fn compile(
    runtime: &MetalDeviceRuntime,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let library = compile_library(runtime)?;
    compile_pipeline(runtime, &library, KeyVisibility::MaskRuns)
}

pub(in crate::backend::metal::vnext_ops::tensor) fn compile_variants(
    runtime: &MetalDeviceRuntime,
) -> Result<[Option<ComputePipelineState>; 2], MetalDeviceRuntimeError> {
    let library = compile_library(runtime)?;
    Ok([
        compile_pipeline(runtime, &library, KeyVisibility::MaskRuns)?,
        compile_pipeline(runtime, &library, KeyVisibility::AllValid)?,
    ])
}

fn compile_library(
    runtime: &MetalDeviceRuntime,
) -> Result<metal::Library, MetalDeviceRuntimeError> {
    let source = concat!(
        include_str!("../bf16.metal"),
        "\n",
        include_str!("../attention.metal"),
        "\n",
        include_str!("register128_calibration.metal"),
        "\n",
        include_str!("register128.metal")
    );
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    runtime
        .device()
        .new_library_with_source(source, &options)
        .map_err(MetalDeviceRuntimeError::contract)
}

fn compile_pipeline(
    runtime: &MetalDeviceRuntime,
    library: &metal::LibraryRef,
    key_visibility: KeyVisibility,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let all_valid = u32::from(key_visibility == KeyVisibility::AllValid);
    let constants = metal::FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        (&all_valid as *const u32).cast(),
        metal::MTLDataType::UInt,
        104,
    );
    let function = library
        .get_function("ferrum_tensor_attention_register128_bf16", Some(constants))
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipeline = runtime
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let supported = pipeline.thread_execution_width() == 32
        && pipeline.max_total_threads_per_threadgroup() >= 128
        && pipeline
            .static_threadgroup_memory_length()
            .checked_add(SHARED_BYTES)
            .is_some_and(|n| n <= runtime.device().max_threadgroup_memory_length());
    tracing::info!(
        supported,
        ?key_visibility,
        query_rows = QUERY_ROWS,
        key_rows = 16,
        thread_execution_width = pipeline.thread_execution_width(),
        maximum_threads = pipeline.max_total_threads_per_threadgroup(),
        static_threadgroup_bytes = pipeline.static_threadgroup_memory_length(),
        dynamic_threadgroup_bytes = SHARED_BYTES,
        "Metal register Head128 attention capability; unsupported shapes retain the previous path"
    );
    Ok(supported.then_some(pipeline))
}

pub(super) fn supports_shape(sequence: u64, head_dim: u32) -> bool {
    // Geometry already checked tensor extents and the u32 mask domain. Short
    // sequences keep the existing 16-query path instead of an oversized tile.
    head_dim == 128 && sequence >= QUERY_ROWS && sequence <= u32::MAX as u64
}

pub(super) fn retile(params: &mut super::Params) -> Result<(), String> {
    // Only the selected register kernel changes tile geometry. A missing
    // capability retains the original 16-row MPP/SIMD grid unchanged.
    let batch_heads = params.jobs / params.query_tiles;
    let query_tiles = params.sequence.div_ceil(QUERY_ROWS);
    let jobs = batch_heads
        .checked_mul(query_tiles)
        .ok_or("Metal register attention grid overflows")?;
    params.query_tiles = query_tiles;
    params.jobs = jobs;
    params.grid_groups = jobs.min(65535);
    Ok(())
}

#[cfg(test)]
#[path = "register128_probe.rs"]
mod probe;

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{CanonicalRational, ResolvedTensorSpec, StructuredAttentionBf16Geometry};

    fn checked_mask(
        segments: &[(i64, bool)],
        runs: &[Vec<(i64, bool)>],
    ) -> Result<StructuredAttentionBf16Geometry, VNextError> {
        let sequence = segments
            .iter()
            .map(|(length, _)| *length as u64)
            .sum::<u64>();
        let tensor = ResolvedTensorSpec::new(
            vec![sequence * runs.len() as u64, 128],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let mut attributes = BTreeMap::from([
            (
                AttributeId::new("sequence_length").unwrap(),
                SemanticValue::Unsigned(sequence),
            ),
            (
                AttributeId::new("head_dim").unwrap(),
                SemanticValue::Unsigned(128),
            ),
            (
                AttributeId::new("scale").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
            ),
        ]);
        for (name, values) in [
            ("segment_lengths", segments.iter().map(|s| s.0).collect()),
            (
                "segment_bidirectional",
                segments.iter().map(|s| i64::from(s.1)).collect(),
            ),
            (
                "key_run_counts",
                runs.iter().map(|batch| batch.len() as i64).collect(),
            ),
            (
                "key_run_lengths",
                runs.iter().flatten().map(|r| r.0).collect(),
            ),
            (
                "key_run_valid",
                runs.iter().flatten().map(|r| i64::from(r.1)).collect(),
            ),
        ] {
            attributes.insert(
                AttributeId::new(name).unwrap(),
                SemanticValue::Integers(values),
            );
        }
        StructuredAttentionBf16Geometry::new(&tensor, &tensor, &tensor, &tensor, &attributes)
    }

    #[test]
    fn all_valid_keys_specialization_preserves_causal_and_separate_segment_ranges() {
        let runs = vec![vec![(33, true)], vec![(33, true)]];
        for segments in [
            vec![(33, true)],
            vec![(33, false)],
            vec![(5, false), (28, true)],
            vec![(5, true), (28, true)],
        ] {
            let checked = checked_mask(&segments, &runs).unwrap();
            let mask = checked.mask();
            assert_eq!(KeyVisibility::from_mask(mask, 33), KeyVisibility::AllValid);
            assert_eq!(
                mask.visible_end(0),
                Some(if segments[0].1 {
                    segments[0].0 as u32
                } else {
                    1
                })
            );
            assert_eq!(mask.visible_end(32), Some(33));
            assert_eq!(mask.visible_end(33), None);
        }
    }

    #[test]
    fn all_valid_keys_specialization_checks_every_batch_and_rejects_holes_or_empty_keys() {
        for invalid_batch in 0..2 {
            for invalid_runs in [vec![(8, true), (1, false), (24, true)], vec![(33, false)]] {
                let mut runs = vec![vec![(33, true)], vec![(33, true)]];
                runs[invalid_batch] = invalid_runs;
                let checked = checked_mask(&[(5, false), (28, true)], &runs).unwrap();
                assert_eq!(
                    KeyVisibility::from_mask(checked.mask(), 33),
                    KeyVisibility::MaskRuns
                );
                assert_eq!(checked.mask().key_is_valid(invalid_batch, 8), Some(false));
            }
        }
        // The native selector consumes the contract's canonical mask, never
        // unchecked attributes with redundant or incomplete key runs.
        assert!(checked_mask(&[(33, true)], &[vec![(8, true), (25, true)]]).is_err());
        assert!(checked_mask(&[(33, true)], &[vec![(32, true)]]).is_err());
    }

    #[test]
    fn register_attention_preserves_other_head_widths_and_short_sequence_fallbacks() {
        assert!(supports_shape(32, 128));
        assert!(supports_shape(65, 128));
        assert!(!supports_shape(31, 128));
        assert!(!supports_shape(65, 64));
        assert!(!supports_shape(u32::MAX as u64 + 1, 128));
    }

    #[test]
    fn retiled_attention_covers_each_batch_head_query_once_including_the_tail() {
        for sequence in [32_u64, 33, 63, 64, 65] {
            let batch = 2;
            let query_heads = 3;
            let old_tiles = sequence.div_ceil(16);
            let mut params = super::super::Params {
                jobs: batch * query_heads * old_tiles,
                grid_groups: batch * query_heads * old_tiles,
                query_tiles: old_tiles,
                sequence,
                query_heads,
                key_value_heads: 1,
                head_dim: 128,
                scale: 0.125,
            };
            retile(&mut params).unwrap();
            let mut visits = vec![0; (batch * query_heads * sequence) as usize];
            for job in 0..params.jobs {
                let batch_head = job / params.query_tiles;
                let first_query = (job % params.query_tiles) * QUERY_ROWS;
                for query in first_query..(first_query + QUERY_ROWS).min(sequence) {
                    visits[(batch_head * sequence + query) as usize] += 1;
                }
            }
            assert!(visits.iter().all(|&count| count == 1));
        }
    }
}
