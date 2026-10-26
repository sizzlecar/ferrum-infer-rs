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
    "metal.attention.head128.simdgroup-f32-bq32-bk16;ushort-staging-native-bfloat-decode-msl3.1-or-original-shift-decode-system-default-on-library-compile-rejection;original-integer-bf16-encode-and-rounded-bf16;register-p2-o16;coordinate-calibrated-xor1-xor8-or-dynamic-four-peer-or-full-shuffle-row-reduction;f32-exp;finite-pv-guarded-explicit-f32-rescale-then-inplace-mma-c;sticky-v-above-2p93-or-o-nonfinite-above-2p124-or-p-alpha-outside-0-2-fallback-zero-pv-mma-separate-add;nonfinite-v-visible-only-scalar-pv-then-separate-rescale-add;key-runs-or-validated-all-batch-all-valid-specialization;typed-structured-or-cross-independent-query-key-lengths;same-structured-causal-segment-tail-ieee-and-arithmetic-order;bf16-final-only;shared15120;fast-math=false.v5";

#[derive(Debug, Clone, Copy)]
enum Domain {
    Structured,
    Cross,
}

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

    pub(super) fn from_cross_mask(mask: &ferrum_runtime::CrossAttentionBf16Geometry) -> Self {
        let sequence = mask.key_value_length() as u32;
        if (0..mask.batch() as usize).all(|batch| {
            mask.key_run_ends(batch) == Some([sequence].as_slice())
                && mask.key_run_valid(batch) == Some([true].as_slice())
        }) {
            Self::AllValid
        } else {
            Self::MaskRuns
        }
    }
}

// Only register kernels consume this packet. The legacy structured (56 B)
// and cross (64 B) packets and dispatch grids remain independent and unchanged.
#[repr(C)]
pub(super) struct RegisterParams {
    pub(super) jobs: u64,
    pub(super) grid_groups: u64,
    pub(super) query_tiles: u64,
    pub(super) query_sequence: u64,
    pub(super) query_heads: u64,
    pub(super) key_value_heads: u64,
    pub(super) key_sequence: u64,
    pub(super) head_dim: u32,
    pub(super) scale: f32,
}
const _: () = assert!(std::mem::size_of::<RegisterParams>() == 64);

impl RegisterParams {
    pub(super) fn new(
        batch_heads: u64,
        query_sequence: u64,
        key_sequence: u64,
        query_heads: u64,
        key_value_heads: u64,
        scale: f32,
    ) -> Result<Self, String> {
        if !supports_cross_shape(query_sequence, key_sequence, 128)
            || batch_heads == 0
            || query_heads == 0
            || key_value_heads == 0
            || query_heads % key_value_heads != 0
            || batch_heads % query_heads != 0
        {
            return Err("Metal register attention requires checked sequence/head geometry".into());
        }
        let query_tiles = query_sequence.div_ceil(QUERY_ROWS);
        let jobs = batch_heads
            .checked_mul(query_tiles)
            .ok_or("Metal register attention grid overflows")?;
        Ok(Self {
            jobs,
            grid_groups: jobs.min(65535),
            query_tiles,
            query_sequence,
            query_heads,
            key_value_heads,
            key_sequence,
            head_dim: 128,
            scale,
        })
    }
}

#[cfg(test)]
pub(in crate::backend::metal::vnext_ops::tensor) fn compile(
    runtime: &MetalDeviceRuntime,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let library = compile_library(runtime)?;
    compile_pipeline(
        runtime,
        &library,
        Domain::Structured,
        KeyVisibility::MaskRuns,
    )
}

pub(in crate::backend::metal::vnext_ops::tensor) fn compile_variants(
    runtime: &MetalDeviceRuntime,
) -> Result<[Option<ComputePipelineState>; 2], MetalDeviceRuntimeError> {
    compile_domain(runtime, Domain::Structured)
}

pub(super) fn compile_cross_variants(
    runtime: &MetalDeviceRuntime,
) -> Result<[Option<ComputePipelineState>; 2], MetalDeviceRuntimeError> {
    compile_domain(runtime, Domain::Cross)
}

fn compile_domain(
    runtime: &MetalDeviceRuntime,
    domain: Domain,
) -> Result<[Option<ComputePipelineState>; 2], MetalDeviceRuntimeError> {
    let library = compile_library(runtime)?;
    Ok([
        compile_pipeline(runtime, &library, domain, KeyVisibility::MaskRuns)?,
        compile_pipeline(runtime, &library, domain, KeyVisibility::AllValid)?,
    ])
}

fn compile_library(
    runtime: &MetalDeviceRuntime,
) -> Result<metal::Library, MetalDeviceRuntimeError> {
    // Both libraries share every kernel body. Renaming only while including
    // the header preserves encode_bf16 and rounded_bf16's original arithmetic.
    macro_rules! source {
        ($prefix:literal, $suffix:literal) => {
            concat!(
                $prefix,
                include_str!("../bf16.metal"),
                $suffix,
                "\n",
                include_str!("../attention.metal"),
                "\n",
                include_str!("register128_calibration.metal"),
                "\n",
                include_str!("register128.metal")
            )
        };
    }
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    options.set_language_version(metal::MTLLanguageVersion::V3_1);
    let native_error = match runtime.device().new_library_with_source(
        source!(
            "#define decode_bf16 decode_bf16_shift\n",
            "\n#undef decode_bf16\ninline float decode_bf16(ushort bits) { return float(as_type<bfloat>(bits)); }\n"
        ),
        &options,
    ) {
        Ok(library) => {
            tracing::info!(
                selected_decode = "native_bfloat",
                requested_language = "MSL3.1",
                fast_math = false,
                "Metal register Head128 attention library compiled"
            );
            return Ok(library);
        }
        Err(reason) => reason,
    };
    // Compiler rejection can indicate an older SDK/device. Preserve the real
    // reason rather than guessing from a platform name or an error substring.
    // Function specialization and PSO creation errors remain fatal below.
    tracing::warn!(
        native_error = %native_error,
        "Metal native BF16 attention library rejected; trying original decode and system-default language"
    );
    let fallback_options = CompileOptions::new();
    fallback_options.set_fast_math_enabled(false);
    let library = runtime
        .device()
        .new_library_with_source(source!("", ""), &fallback_options)
        .map_err(|fallback_error| MetalDeviceRuntimeError::contract(format!(
            "Metal register Head128 attention library compilation failed; native BF16/MSL3.1: {native_error}; original shift decode/system-default: {fallback_error}"
        )))?;
    tracing::info!(
        selected_decode = "original_shift",
        requested_language = "system_default",
        fast_math = false,
        "Metal register Head128 attention fallback library compiled"
    );
    Ok(library)
}

fn compile_pipeline(
    runtime: &MetalDeviceRuntime,
    library: &metal::LibraryRef,
    domain: Domain,
    key_visibility: KeyVisibility,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    compile_pipeline_with_gate(runtime, library, domain, key_visibility, false)
}

pub(super) fn compile_gated(
    runtime: &MetalDeviceRuntime,
    cross: bool,
    key_visibility: KeyVisibility,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let library = compile_library(runtime)?;
    compile_pipeline_with_gate(
        runtime,
        &library,
        if cross {
            Domain::Cross
        } else {
            Domain::Structured
        },
        key_visibility,
        true,
    )
}

fn compile_pipeline_with_gate(
    runtime: &MetalDeviceRuntime,
    library: &metal::LibraryRef,
    domain: Domain,
    key_visibility: KeyVisibility,
    gated: bool,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let all_valid = u32::from(key_visibility == KeyVisibility::AllValid);
    let constants = metal::FunctionConstantValues::new();
    let gate = u32::from(gated);
    constants.set_constant_value_at_index(
        (&gate as *const u32).cast(),
        metal::MTLDataType::UInt,
        106,
    );
    constants.set_constant_value_at_index(
        (&all_valid as *const u32).cast(),
        metal::MTLDataType::UInt,
        104,
    );
    let cross = u32::from(matches!(domain, Domain::Cross));
    constants.set_constant_value_at_index(
        (&cross as *const u32).cast(),
        metal::MTLDataType::UInt,
        105,
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
        ?domain,
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

pub(super) fn supports_cross_shape(query: u64, key: u64, head_dim: u32) -> bool {
    supports_shape(query, head_dim) && key > 0 && key <= u32::MAX as u64
}

#[cfg(test)]
#[path = "register128_probe.rs"]
mod probe;

#[cfg(test)]
#[path = "register128_diagnostic.rs"]
mod diagnostic;

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
        assert!(supports_cross_shape(32, 1, 128));
        assert!(supports_cross_shape(33, 17, 128));
        assert!(!supports_cross_shape(31, 65, 128));
        assert!(!supports_cross_shape(32, 0, 128));
        assert!(!supports_cross_shape(32, u32::MAX as u64 + 1, 128));
    }

    #[test]
    fn register_packets_cover_each_batch_head_query_once_with_independent_key_length() {
        for (sequence, key_sequence) in [(32_u64, 1_u64), (33, 16), (63, 17), (64, 65), (65, 33)] {
            let batch = 2;
            let query_heads = 3;
            let old_tiles = sequence.div_ceil(16);
            let legacy = super::super::Params {
                jobs: batch * query_heads * old_tiles,
                grid_groups: batch * query_heads * old_tiles,
                query_tiles: old_tiles,
                sequence,
                query_heads,
                key_value_heads: 1,
                head_dim: 128,
                scale: 0.125,
            };
            let params = RegisterParams::new(
                legacy.jobs / legacy.query_tiles,
                sequence,
                key_sequence,
                query_heads,
                1,
                legacy.scale,
            )
            .unwrap();
            assert_eq!(legacy.query_tiles, old_tiles);
            assert_eq!(params.query_sequence, sequence);
            assert_eq!(params.key_sequence, key_sequence);
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
        assert!(RegisterParams::new(u64::MAX, 65, 33, 1, 1, 0.125).is_err());
    }
}
