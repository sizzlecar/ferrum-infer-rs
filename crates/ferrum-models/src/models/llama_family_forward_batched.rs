//! Batched-decode forward methods for `LlamaFamilyModel`.
//!
//! Split out of `llama_family.rs` to keep that file under 3000 lines.
//! These three methods are the heaviest forward paths and reference
//! many private fields on the parent struct, so they live in a separate
//! `impl` block in a peer file (Rust allows multiple `impl` blocks for
//! the same type across the crate).

use ferrum_execution::model_executor::{
    ExecutorSamplingOutput, GreedyRepetitionPenalty, LogitsReturnPolicy, TokenSelectionMask,
};
use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_kernels::backend::{
    Backend, BackendGraph, BackendMoeFused, BackendPagedKv, BackendQuantGguf, BackendQuantMarlin,
    KvCache, MoeLlmBackend, MAX_LAYERS_FOR_GRAPH,
};
use ferrum_types::{Activation, Result, TokenId};

use super::llama_family::device_execution::{
    self, FrameLayout, LlamaExecutionFrame, QuiescentContext,
};
use super::llama_family::{
    elapsed_micros_u64_floor1, llama_layer_attention_schedule, llama_qk_mode, LlamaFamilyModel,
    LlamaStageHiddenBridgeTiming, SINGLE_ITEM_GRAPH_KEY,
};
use super::llama_family_pipeline::{LlamaPipelineStageBatchOps, PipelineHidden};

// Compatibility for the retiring vector API. The numerical core records its
// actual readback kind before this conversion; direct token execution never
// interprets a one-element vector as evidence of GPU argmax.
fn legacy_sampling_values(output: ExecutorSamplingOutput) -> Result<Vec<f32>> {
    match output {
        ExecutorSamplingOutput::FullLogits(logits) => Ok(logits),
        ExecutorSamplingOutput::GreedyToken(token) => Ok(vec![token.get() as f32]),
        ExecutorSamplingOutput::NoSelectableFiniteLogit => Err(ferrum_types::FerrumError::backend(
            "No finite logits available for sampling",
        )),
    }
}

/// Selection may have encoded additional work before declining the fast
/// path. A full readback must observe that work, even if the original logits
/// were ready before the selection attempt.
fn read_full_logits_after_selection<B: MoeLlmBackend>(
    context: &mut B::Context,
    logits: &B::Buffer,
    count: usize,
    quiescent: &mut Option<QuiescentContext>,
) -> Vec<f32> {
    if quiescent.is_none() {
        *quiescent = Some(device_execution::checked_sync::<B>(context));
    }
    B::to_vec_in_context(context, logits, count)
}

/// A missing graph is an eager opportunity only before capture. After a
/// successful capture, the recorded operations have not run until this
/// first replay actually succeeds.
fn replay_new_batched_graph<B: BackendGraph>(
    context: &mut B::Context,
    graph_key: u64,
    failed: &mut bool,
) -> Result<()> {
    // SAFETY: cold construction checked projection/KV capture contracts.
    // The private model/frame owns all bindings through checked retirement;
    // storage mutation retires cached graphs before replacing dependencies.
    let result = match unsafe { B::replay_graph(context, graph_key) } {
        Ok(true) => Ok(()),
        Ok(false) => Err(ferrum_types::FerrumError::backend(
            "captured batched graph was unavailable before its first execution",
        )),
        Err(error) => Err(error),
    };
    if result.is_err() {
        *failed = true;
    }
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LlamaBatchedRuntimeConfig {
    unified_graph: bool,
    unified_graph_layers_only: bool,
    unified_graph_lm_head_eager: bool,
    batched_graph: bool,
    greedy_argmax: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum UnifiedGraphCaptureScope {
    LayersOnly,
    LmHeadEager,
    Full,
}

/// Equality checks every captured host decision; hashes only index the map.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct UnifiedGraphShape {
    scope: UnifiedGraphCaptureScope,
    rows: usize,
    sequences: usize,
    attention: ferrum_kernels::backend::PagedAttentionLaunchGeometry,
    final_indices: Vec<(usize, usize)>,
}

fn next_unified_graph_key(counter: u64) -> u64 {
    assert!(
        counter < (1u64 << 63),
        "unified graph native ID space exhausted"
    );
    (1u64 << 63) | counter
}

impl UnifiedGraphCaptureScope {
    fn from_config(config: &LlamaBatchedRuntimeConfig) -> Self {
        if config.unified_graph_layers_only {
            Self::LayersOnly
        } else if config.unified_graph_lm_head_eager {
            Self::LmHeadEager
        } else {
            Self::Full
        }
    }

    fn captures_final_pack(self) -> bool {
        matches!(self, Self::LmHeadEager | Self::Full)
    }

    fn captures_lm_head(self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Debug, Clone)]
enum DecodeLogitsReturn<'a> {
    Full,
    LegacyDefault,
    GreedyArgmax {
        token_mask: Option<&'a TokenSelectionMask>,
        repetition_penalties: Vec<Option<&'a GreedyRepetitionPenalty>>,
    },
}

#[derive(Debug, Clone, Copy)]
enum ArgmaxMode<'a> {
    Raw,
    Masked(&'a TokenSelectionMask),
    SparseRepetition {
        token_mask: Option<&'a TokenSelectionMask>,
        repetition_penalties: &'a [Option<&'a GreedyRepetitionPenalty>],
    },
}

fn has_sparse_repetition_penalties(penalties: &[Option<&GreedyRepetitionPenalty>]) -> bool {
    penalties
        .iter()
        .flatten()
        .any(|penalty| !penalty.is_empty())
}

fn resolve_unified_logits_return<'a>(
    policies: Option<&'a [LogitsReturnPolicy]>,
    item_count: usize,
    final_indices: &[(usize, usize)],
) -> DecodeLogitsReturn<'a> {
    let Some(policies) = policies else {
        return DecodeLogitsReturn::Full;
    };
    if policies.len() != item_count {
        return DecodeLogitsReturn::Full;
    }

    let mut selected_mask: Option<&'a TokenSelectionMask> = None;
    let mut saw_unmasked = false;
    // Keep the common no-penalty greedy path allocation-free. Once the first
    // live penalty appears, backfill earlier rows so device metadata remains
    // aligned with `final_indices`.
    let mut repetition_penalties = Vec::new();
    for (sampled_idx, &(orig_idx, _)) in final_indices.iter().enumerate() {
        match &policies[orig_idx] {
            LogitsReturnPolicy::FullLogits => return DecodeLogitsReturn::Full,
            LogitsReturnPolicy::GreedyArgmax {
                token_mask,
                repetition_penalty,
            } => {
                let repetition_penalty = repetition_penalty
                    .as_ref()
                    .filter(|penalty| !penalty.is_empty());
                if repetition_penalties.is_empty() {
                    if let Some(repetition_penalty) = repetition_penalty {
                        repetition_penalties.resize(sampled_idx, None);
                        repetition_penalties.push(Some(repetition_penalty));
                    }
                } else {
                    repetition_penalties.push(repetition_penalty);
                }
                match token_mask.as_ref() {
                    Some(mask) => {
                        if saw_unmasked {
                            return DecodeLogitsReturn::Full;
                        }
                        if let Some(selected) = selected_mask {
                            if selected.fingerprint != mask.fingerprint
                                || selected.len() != mask.len()
                            {
                                return DecodeLogitsReturn::Full;
                            }
                        } else {
                            selected_mask = Some(mask);
                        }
                    }
                    None => {
                        if selected_mask.is_some() {
                            return DecodeLogitsReturn::Full;
                        }
                        saw_unmasked = true;
                    }
                }
            }
        }
    }

    DecodeLogitsReturn::GreedyArgmax {
        token_mask: selected_mask,
        repetition_penalties,
    }
}

impl LlamaBatchedRuntimeConfig {
    pub(crate) fn from_execution_config(config: &ferrum_types::LlamaExecutionConfig) -> Self {
        Self {
            unified_graph: config.options.unified_graph,
            unified_graph_layers_only: config.options.unified_graph_layers_only,
            unified_graph_lm_head_eager: config.options.unified_graph_lm_head_eager,
            batched_graph: config.batched_graph,
            greedy_argmax: config.options.greedy_argmax,
        }
    }
}

/// Exact encoded shape. HashMap hashes locate candidates; complete Eq is the
/// authority for reuse. Native graph IDs are separately allocated below bit 63
/// (single decode owns 0, unified graphs own the high-bit namespace).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BatchedDecodeGraphShape {
    rows: usize,
    layout: BatchedDecodeKvLayout,
    score_capacity: usize,
    device_residual_shadow: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum BatchedDecodeKvLayout {
    Contiguous {
        capacity: usize,
    },
    Paged {
        block_size: usize,
        max_blocks_per_sequence: usize,
    },
}

fn should_use_batched_decode_graph(config_enabled: bool, use_host_residual_shadow: bool) -> bool {
    config_enabled && !use_host_residual_shadow
}

fn next_batched_decode_graph_key(installed_shapes: usize) -> u64 {
    let key = u64::try_from(installed_shapes)
        .ok()
        .and_then(|n| n.checked_add(1))
        .filter(|key| *key < (1u64 << 63))
        .expect("batched graph namespace exhausted");
    key
}

const UNIFIED_GRAPH_MAX_CACHED_KEYS: usize = 16;

fn unified_graph_capture_skip_reason(
    m_total: usize,
    num_seqs: usize,
    cached_keys: usize,
) -> Option<&'static str> {
    if num_seqs == 0 || m_total != num_seqs {
        Some("mixed_or_prefill_batch")
    } else if cached_keys >= UNIFIED_GRAPH_MAX_CACHED_KEYS {
        Some("cache_full")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        next_batched_decode_graph_key, resolve_unified_logits_return,
        should_use_batched_decode_graph, unified_graph_capture_skip_reason,
        BatchedDecodeGraphShape, BatchedDecodeKvLayout, DecodeLogitsReturn,
        LlamaBatchedRuntimeConfig, UNIFIED_GRAPH_MAX_CACHED_KEYS,
    };
    use crate::common::DecoderOnlyLLM;
    use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
    use crate::test_support::SyntheticLlamaLoader;
    use ferrum_execution::model_executor::{
        ExecutorSamplingOutput, GreedyRepetitionPenalty, LogitsReturnPolicy, TokenSelectionMask,
    };
    use ferrum_interfaces::kv_dtype::KvFp16;
    use ferrum_kernels::backend::MoeLlmBackend;

    #[cfg(feature = "cuda")]
    mod cuda_projection_tests;

    #[cfg(feature = "cuda")]
    mod cuda_batched_graph_tests;

    fn physical_batch_config() -> LlamaFamilyConfig {
        LlamaFamilyConfig {
            hidden_size: 128,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            num_layers: 1,
            vocab_size: 128,
            max_seq_len: 32,
            ..Default::default()
        }
    }

    fn physical_batch_model<B: MoeLlmBackend>() -> LlamaFamilyModel<B, KvFp16> {
        let config = physical_batch_config();
        let loader = SyntheticLlamaLoader::new(config.clone());
        physical_batch_model_with_loader::<B>(config, &loader)
    }

    fn physical_batch_model_with_loader<B: MoeLlmBackend>(
        config: LlamaFamilyConfig,
        loader: &dyn ferrum_quantization::WeightLoader<B>,
    ) -> LlamaFamilyModel<B, KvFp16> {
        let mut model = LlamaFamilyModel::<B, KvFp16>::new(config, loader).unwrap();
        model.runtime_env.metal_paged_kv = Some(false);
        model.runtime_env.prefix_cache = false;
        model.runtime_env.cuda_graph = false;
        model.runtime_env.kv_capacity = Some(32);
        model.batched_cfg.batched_graph = false;
        model.prefill_internal("a", &[1, 2, 3]);
        model.prefill_internal("b", &[3, 2, 1]);
        assert!(model.supports_batched_decode);
        model
    }

    fn physical_batch_full_logits<B: MoeLlmBackend>(
        model: &mut LlamaFamilyModel<B, KvFp16>,
        batch: &[(String, u32, u32)],
    ) -> Vec<Vec<f32>> {
        let outputs = model
            .run_device_frame(|model, frame| {
                model.decode_batch_in_frame(frame, batch, DecodeLogitsReturn::Full)
            })
            .into_legacy_result();
        assert_eq!(outputs.len(), batch.len());
        outputs
            .into_iter()
            .map(|output| match output {
                ExecutorSamplingOutput::FullLogits(values) => values,
                _ => panic!("full batch policy must preserve every logit"),
            })
            .collect()
    }

    fn assert_physical_batch_logits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            assert!(
                (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                "logit {index}: actual={actual}, expected={expected}",
            );
        }
    }

    fn physical_batch_frame_retains_scratch<B: MoeLlmBackend>(
        identity: impl Fn(&B::Buffer) -> usize,
    ) {
        use super::super::llama_family::device_execution::DeviceFrameResult;

        let mut model = physical_batch_model::<B>();
        let mut reference = physical_batch_model::<B>();
        let batch = vec![("a".to_string(), 4, 3), ("b".to_string(), 5, 3)];
        // Match the actual M=2 execution shape and keep this independent owner
        // alive through the later peer decode. Cross-shape numerical agreement
        // is a separate mandatory numerical test below; CUDA qualifies each
        // shape against an independent CPU oracle.
        let expected = physical_batch_full_logits(&mut reference, &batch);
        let original = identity(model.scratch.residual.as_ref().unwrap());
        let result: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
            let outputs = model.decode_batch_in_frame(frame, &batch, DecodeLogitsReturn::Full);
            assert_eq!(outputs.len(), 2);
            for (output, expected) in outputs.into_iter().zip(&expected) {
                let ExecutorSamplingOutput::FullLogits(actual) = output else {
                    panic!("full batch policy must preserve every logit");
                };
                assert_physical_batch_logits(&actual, expected);
            }
            assert!(model.scratch.residual.is_none());
            assert_eq!(identity(frame.residual.as_ref().unwrap()), original);
            std::panic::panic_any(ferrum_types::FerrumError::backend(
                "injected after physical batch readback",
            ));
        });
        let DeviceFrameResult::Terminal(Err(error)) = result else {
            panic!("completed physical batch must preserve its terminal failure");
        };
        assert!(error
            .to_string()
            .contains("injected after physical batch readback"));
        assert_eq!(identity(model.scratch.residual.as_ref().unwrap()), original);
        for id in ["a", "b"] {
            assert!(model.kv_caches[id].iter().all(|cache| cache.len == 4));
        }
        // This key has no captured graph, so the real replay API returns
        // Ok(false). Exercise the same first-replay check used after capture:
        // no logits/length bump may follow it, and already encoded preparation
        // must still finish before the original scratch returns to the model.
        let missing_graph: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
            frame
                .take_scratch(&mut model.scratch, super::FrameLayout::Single, false)
                .unwrap();
            B::scale_inplace(
                &mut frame.context,
                frame.residual.as_mut().unwrap(),
                2.0,
                2 * model.cfg.hidden_size,
            );
            super::replay_new_batched_graph::<B>(
                &mut frame.context,
                u64::MAX,
                &mut model.batched_graph_failed,
            )
            .unwrap_or_else(|error| std::panic::panic_any(error));
            panic!("missing first graph replay must not reach readback");
        });
        let DeviceFrameResult::Terminal(Err(error)) = missing_graph else {
            panic!("missing graph must retain the real terminal failure");
        };
        assert!(error
            .to_string()
            .contains("unavailable before its first execution"));
        assert!(model.batched_graph_failed);
        assert_eq!(identity(model.scratch.residual.as_ref().unwrap()), original);
        for id in ["a", "b"] {
            assert!(model.kv_caches[id].iter().all(|cache| cache.len == 4));
        }
        model.release("a");
        reference.release("a");
        let actual_peer = model.decode_internal("b", 6, 4);
        let expected_peer = reference.decode_internal("b", 6, 4);
        assert_physical_batch_logits(&actual_peer, &expected_peer);
        assert!(model.kv_caches["b"].iter().all(|cache| cache.len == 5));
        assert!(reference.kv_caches["b"].iter().all(|cache| cache.len == 5));
        model.release("b");
        reference.release("b");
        assert!(model.kv_caches.is_empty());
        assert!(reference.kv_caches.is_empty());
        model.close().unwrap();
        reference.close().unwrap();
    }

    #[test]
    fn physical_batch_frame_preserves_real_cpu_logits_and_scratch_on_failure() {
        physical_batch_frame_retains_scratch::<ferrum_kernels::backend::cpu::CpuBackend>(
            |buffer| buffer.as_ptr() as usize,
        );
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    #[ignore = "requires an actual Metal device"]
    fn physical_batch_frame_preserves_real_metal_logits_and_scratch_on_failure() {
        physical_batch_frame_retains_scratch::<ferrum_kernels::backend::metal::MetalBackend>(
            |buffer| buffer.raw().contents() as usize,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an actual CUDA device"]
    fn physical_batch_frame_preserves_real_cuda_logits_and_scratch_on_failure() {
        use cudarc::driver::DevicePtr;
        use ferrum_kernels::backend::cuda::CudaBackend;
        physical_batch_frame_retains_scratch::<CudaBackend>(|buffer| {
            let (pointer, _record) = buffer.as_f16().device_ptr(buffer.as_f16().stream());
            usize::try_from(pointer)
                .expect("CUDA allocation address must fit the host pointer width")
        });
    }

    fn physical_batch_and_single_logits<B: MoeLlmBackend>() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut batched = physical_batch_model::<B>();
        let mut single = physical_batch_model::<B>();
        let batch = [("a".to_string(), 4, 3), ("b".to_string(), 5, 3)];
        let expected = batch
            .iter()
            .map(|(id, token, position)| single.decode_internal(id, *token, *position))
            .collect();
        let actual = physical_batch_full_logits(&mut batched, &batch);
        for model in [&mut batched, &mut single] {
            for id in ["a", "b"] {
                assert!(model.kv_caches[id].iter().all(|cache| cache.len == 4));
                model.release(id);
            }
            assert!(model.kv_caches.is_empty());
            model.close().unwrap();
        }
        (actual, expected)
    }

    // Bounded context for both complete rows. Callers separately enforce the
    // applicable strict-equivalence or independent-reference qualification.
    fn log_physical_batch_difference(label: &str, actual: &[Vec<f32>], expected: &[Vec<f32>]) {
        let argmax = |values: &[f32]| {
            values
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .max_by(|left, right| left.1.total_cmp(&right.1))
        };
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let mut first = None;
            let (mut max_abs, mut error_sq, mut expected_sq) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                if first.is_none() && actual.to_bits() != expected.to_bits() {
                    first = Some((index, actual, expected));
                }
                let difference = f64::from(actual) - f64::from(expected);
                max_abs = max_abs.max(difference.abs());
                error_sq += difference * difference;
                expected_sq += f64::from(expected).powi(2);
            }
            let relative_l2 = if error_sq == 0.0 {
                0.0
            } else {
                (error_sq / expected_sq).sqrt()
            };
            eprintln!(
                "{label} row={row} lengths={}/{} first={first:?} max_abs={max_abs:.9e} relative_l2={relative_l2:.9e} argmax={:?}/{:?}",
                actual.len(), expected.len(), argmax(actual), argmax(expected),
            );
        }
    }

    fn physical_batch_matches_single_item<B: MoeLlmBackend>() {
        let (actual, expected) = physical_batch_and_single_logits::<B>();
        assert_eq!(actual.len(), 2);
        assert_eq!(expected.len(), 2);
        if actual.iter().zip(&expected).any(|(actual, expected)| {
            actual.len() != expected.len()
                || actual.iter().zip(expected).any(|(actual, expected)| {
                    !actual.is_finite()
                        || !expected.is_finite()
                        || (actual - expected).abs() > 1e-5 * (1.0 + expected.abs())
                })
        }) {
            eprintln!(
                "physical_batch_logits backend={}",
                std::any::type_name::<B>()
            );
            log_physical_batch_difference("M2/M1", &actual, &expected);
            let (cpu_batch, cpu_single) =
                physical_batch_and_single_logits::<ferrum_kernels::backend::cpu::CpuBackend>();
            log_physical_batch_difference("M2/CPU_M2", &actual, &cpu_batch);
            log_physical_batch_difference("M1/CPU_M1", &expected, &cpu_single);
        }
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_physical_batch_logits(actual, expected);
        }
    }

    #[test]
    fn physical_batch_logits_match_single_item_decode_on_cpu() {
        physical_batch_matches_single_item::<ferrum_kernels::backend::cpu::CpuBackend>();
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    #[ignore = "requires an actual Metal device"]
    fn physical_batch_logits_match_single_item_decode_on_metal() {
        physical_batch_matches_single_item::<ferrum_kernels::backend::metal::MetalBackend>();
    }

    #[cfg(feature = "cuda")]
    fn assert_cuda_physical_logits_qualification(batched: &[Vec<f32>], single: &[Vec<f32>]) {
        use crate::test_support::numerical::assert_full_logits_qualification;
        use ferrum_kernels::backend::cpu::CpuBackend;

        // The historical 1e-5 M2/M1 assertion was inherited from the CPU/Metal
        // lifecycle fixture. It does not establish an FP16 guarantee across
        // CUDA's different attention reductions. Qualify both real shapes
        // against independent CPU full-vocabulary results using the existing
        // managed-model envelope; retain the strict same-shape lifecycle test.
        let (cpu_batch, cpu_single) = physical_batch_and_single_logits::<CpuBackend>();
        for rows in [batched, single, cpu_batch.as_slice(), cpu_single.as_slice()] {
            assert_eq!(rows.len(), 2);
            assert!(rows
                .iter()
                .all(|row| row.len() == physical_batch_config().vocab_size));
        }
        log_physical_batch_difference("M2/CPU_M2", batched, &cpu_batch);
        log_physical_batch_difference("M1/CPU_M1", single, &cpu_single);
        for row in 0..2 {
            assert_physical_batch_logits(&cpu_batch[row], &cpu_single[row]);
            assert_full_logits_qualification(&batched[row], &cpu_batch[row]);
            assert_full_logits_qualification(&single[row], &cpu_single[row]);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an actual CUDA device"]
    fn physical_batch_and_single_logits_match_cpu_oracle_on_cuda() {
        let (batched, single) =
            physical_batch_and_single_logits::<ferrum_kernels::backend::cuda::CudaBackend>();
        assert_cuda_physical_logits_qualification(&batched, &single);
    }

    fn selection_failure_readback_observes_pending_work<
        B: ferrum_kernels::backend::MoeLlmBackend,
    >() {
        let mut context = B::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
        // Use the backend's activation dtype, including CUDA's F16 scale kernel.
        let mut logits = B::from_slice_in_context(&mut context, &[1.0f32, 2.0, 3.0, 4.0]);
        B::scale_inplace(&mut context, &mut logits, 1.0, 4);
        let mut quiescent = Some(super::device_execution::checked_sync::<B>(&mut context));
        assert_eq!(
            B::to_vec_in_context(&mut context, &logits, 4),
            [1.0, 2.0, 3.0, 4.0]
        );

        // Inject selection failure after real preparation work has been
        // encoded, but before its readback/synchronization. On Metal this
        // command buffer is still uncommitted: reading first deterministically
        // returns the old values, rather than relying on a device timing race.
        let _ = quiescent.take();
        B::scale_inplace(&mut context, &mut logits, 2.0, 4);
        let selected: ferrum_types::Result<Vec<u32>> = Err(ferrum_types::FerrumError::backend(
            "injected selection failure before readback",
        ));
        assert!(selected.is_err());
        let full =
            super::read_full_logits_after_selection::<B>(&mut context, &logits, 4, &mut quiescent);
        assert_eq!(full, [2.0, 4.0, 6.0, 8.0]);
        assert!(quiescent.is_some());
    }

    #[test]
    fn selection_failure_full_readback_observes_real_cpu_preparation() {
        selection_failure_readback_observes_pending_work::<ferrum_kernels::backend::cpu::CpuBackend>(
        );
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    #[ignore = "requires an actual Metal device"]
    fn selection_failure_full_readback_observes_real_metal_preparation() {
        selection_failure_readback_observes_pending_work::<
            ferrum_kernels::backend::metal::MetalBackend,
        >();
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an actual CUDA device"]
    fn selection_failure_full_readback_observes_real_cuda_preparation() {
        selection_failure_readback_observes_pending_work::<
            ferrum_kernels::backend::cuda::CudaBackend,
        >();
    }

    #[test]
    fn unified_logits_policy_keeps_sparse_repetition_on_greedy_path() {
        let mask = TokenSelectionMask::new(vec![1, 1, 0, 1]);
        let repetition = GreedyRepetitionPenalty::new(1.1, vec![3, 1, 3]);
        let policies = vec![
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(mask.clone()),
                repetition_penalty: None,
            },
            // A non-final item must not force full-logits readback.
            LogitsReturnPolicy::FullLogits,
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(mask.clone()),
                repetition_penalty: Some(repetition),
            },
        ];

        let resolved = resolve_unified_logits_return(Some(&policies), 3, &[(0, 0), (2, 5)]);
        let DecodeLogitsReturn::GreedyArgmax {
            token_mask,
            repetition_penalties,
        } = resolved
        else {
            panic!("compatible greedy policies should stay on the GPU argmax path");
        };
        assert_eq!(
            token_mask.map(|value| value.fingerprint),
            Some(mask.fingerprint)
        );
        assert!(repetition_penalties[0].is_none());
        let repetition = repetition_penalties[1].expect("second row repetition metadata");
        assert_eq!(repetition.penalty(), 1.1);
        assert_eq!(repetition.token_ids(), [3, 1]);
    }

    #[test]
    fn unified_logits_policy_keeps_explicit_full_logits() {
        let policies = vec![LogitsReturnPolicy::FullLogits];
        assert!(matches!(
            resolve_unified_logits_return(Some(&policies), 1, &[(0, 0)]),
            DecodeLogitsReturn::Full
        ));
    }

    #[test]
    fn unified_logits_policy_keeps_no_op_repetition_allocation_free() {
        let policies = vec![LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.0, vec![1, 2])),
        }];
        let DecodeLogitsReturn::GreedyArgmax {
            repetition_penalties,
            ..
        } = resolve_unified_logits_return(Some(&policies), 1, &[(0, 0)])
        else {
            panic!("a no-op repetition penalty should keep the greedy path");
        };
        assert!(repetition_penalties.is_empty());
    }

    #[test]
    fn unified_logits_policy_rejects_inconsistent_masks() {
        let policies = vec![
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1])),
                repetition_penalty: None,
            },
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(TokenSelectionMask::new(vec![1, 1, 0])),
                repetition_penalty: None,
            },
        ];
        assert!(matches!(
            resolve_unified_logits_return(Some(&policies), 2, &[(0, 0), (1, 1)]),
            DecodeLogitsReturn::Full
        ));
    }

    fn config_from_entries<const N: usize>(
        entries: [(&str, &str); N],
    ) -> LlamaBatchedRuntimeConfig {
        let mut engine = ferrum_types::EngineConfig::default();
        engine
            .apply_runtime_config_snapshot(&ferrum_types::RuntimeConfigSnapshot::from_env_vars(
                entries,
            ))
            .unwrap();
        LlamaBatchedRuntimeConfig::from_execution_config(&engine.llama_execution_config())
    }

    #[test]
    fn llama_batched_runtime_config_parses_startup_knobs() {
        let config = config_from_entries([
            ("FERRUM_UNIFIED_GRAPH", "1"),
            ("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY", "1"),
            ("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER", "1"),
            ("FERRUM_BATCHED_GRAPH", "1"),
            ("FERRUM_GREEDY_ARGMAX", "1"),
            ("FERRUM_SPLIT_K_ATTN", "1"),
        ]);
        assert!(config.unified_graph);
        assert!(config.unified_graph_layers_only);
        assert!(config.unified_graph_lm_head_eager);
        assert!(config.batched_graph);
        assert!(config.greedy_argmax);
    }

    #[test]
    fn llama_batched_runtime_config_preserves_opt_out_values() {
        let config = config_from_entries([
            ("FERRUM_UNIFIED_GRAPH", "true"),
            ("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY", "true"),
            ("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER", "true"),
            ("FERRUM_BATCHED_GRAPH", "0"),
            ("FERRUM_GREEDY_ARGMAX", "true"),
            ("FERRUM_SPLIT_K_ATTN", "auto"),
        ]);
        assert!(!config.unified_graph);
        assert!(!config.unified_graph_layers_only);
        assert!(!config.unified_graph_lm_head_eager);
        assert!(!config.batched_graph);
        assert!(!config.greedy_argmax);
    }

    #[test]
    fn batched_decode_graph_allows_device_shadow_not_host_shadow() {
        assert!(should_use_batched_decode_graph(true, false));
        assert!(!should_use_batched_decode_graph(true, true));
        assert!(!should_use_batched_decode_graph(false, false));
    }

    #[test]
    fn batched_decode_graph_key_separates_actual_shape_and_native_namespace() {
        let shape = BatchedDecodeGraphShape {
            rows: 3,
            layout: BatchedDecodeKvLayout::Contiguous { capacity: 32 },
            score_capacity: 8,
            device_residual_shadow: false,
        };
        for other in [
            BatchedDecodeGraphShape {
                rows: 4,
                ..shape.clone()
            },
            BatchedDecodeGraphShape {
                score_capacity: 16,
                ..shape.clone()
            },
            BatchedDecodeGraphShape {
                device_residual_shadow: true,
                ..shape.clone()
            },
            BatchedDecodeGraphShape {
                layout: BatchedDecodeKvLayout::Contiguous { capacity: 64 },
                ..shape.clone()
            },
            BatchedDecodeGraphShape {
                layout: BatchedDecodeKvLayout::Paged {
                    block_size: 16,
                    max_blocks_per_sequence: 2,
                },
                ..shape.clone()
            },
        ] {
            assert_ne!(shape, other);
        }
        assert_eq!(next_batched_decode_graph_key(0), 1);
        assert_ne!(
            next_batched_decode_graph_key(0),
            next_batched_decode_graph_key(1)
        );
        assert_eq!(next_batched_decode_graph_key(64) >> 63, 0);
    }

    #[test]
    fn unified_graph_shapes_preserve_complete_capture_identity() {
        use super::{
            next_unified_graph_key, UnifiedGraphCaptureScope, UnifiedGraphShape,
            SINGLE_ITEM_GRAPH_KEY,
        };
        use ferrum_kernels::backend::PagedAttentionLaunchGeometry as Geometry;
        use std::hash::{BuildHasherDefault, Hasher};
        #[derive(Default)]
        struct SameHash;
        impl Hasher for SameHash {
            fn finish(&self) -> u64 {
                0
            }
            fn write(&mut self, _: &[u8]) {}
        }
        let shape = UnifiedGraphShape {
            scope: UnifiedGraphCaptureScope::Full,
            rows: 5,
            sequences: 5,
            attention: Geometry::SplitK {
                num_splits: 2,
                score_capacity: 3,
            },
            final_indices: vec![(0, 0), (1, 1)],
        };
        let variants = [
            UnifiedGraphShape {
                attention: Geometry::SplitK {
                    num_splits: 2,
                    score_capacity: 4,
                },
                ..shape.clone()
            },
            UnifiedGraphShape {
                attention: Geometry::SplitK {
                    num_splits: 4,
                    score_capacity: 3,
                },
                ..shape.clone()
            },
            UnifiedGraphShape {
                attention: Geometry::SinglePass { score_capacity: 8 },
                ..shape.clone()
            },
            UnifiedGraphShape {
                rows: 6,
                ..shape.clone()
            },
            UnifiedGraphShape {
                sequences: 4,
                ..shape.clone()
            },
            UnifiedGraphShape {
                scope: UnifiedGraphCaptureScope::LayersOnly,
                ..shape.clone()
            },
            UnifiedGraphShape {
                scope: UnifiedGraphCaptureScope::LmHeadEager,
                ..shape.clone()
            },
            UnifiedGraphShape {
                final_indices: vec![(0, 0), (1, 2)],
                ..shape.clone()
            },
        ];
        let mut cache = std::collections::HashMap::<_, _, BuildHasherDefault<SameHash>>::default();
        cache.insert(shape.clone(), next_unified_graph_key(0));
        for (index, variant) in variants.iter().enumerate() {
            assert_ne!(&shape, variant);
            let key = next_unified_graph_key(index as u64 + 1);
            assert_ne!(key >> 63, 0);
            assert_ne!(key, SINGLE_ITEM_GRAPH_KEY);
            assert_ne!(key, next_batched_decode_graph_key(index));
            assert!(cache.insert(variant.clone(), key).is_none());
        }
        assert_eq!(cache.len(), variants.len() + 1);
        assert_eq!(cache.get(&shape), Some(&next_unified_graph_key(0)));
        for (index, variant) in variants.iter().enumerate() {
            assert_eq!(
                cache.get(variant),
                Some(&next_unified_graph_key(index as u64 + 1))
            );
        }
    }

    #[test]
    fn unified_graph_capture_admission_rejects_mixed_prefill_batches() {
        assert_eq!(
            unified_graph_capture_skip_reason(17, 16, 0),
            Some("mixed_or_prefill_batch")
        );
        assert_eq!(
            unified_graph_capture_skip_reason(0, 0, 0),
            Some("mixed_or_prefill_batch")
        );
        assert_eq!(unified_graph_capture_skip_reason(16, 16, 0), None);
    }

    #[test]
    fn unified_graph_capture_admission_caps_cached_shape_count() {
        assert_eq!(
            unified_graph_capture_skip_reason(16, 16, UNIFIED_GRAPH_MAX_CACHED_KEYS - 1),
            None
        );
        assert_eq!(
            unified_graph_capture_skip_reason(16, 16, UNIFIED_GRAPH_MAX_CACHED_KEYS),
            Some("cache_full")
        );
    }
}

// Batched / unified-forward paths are FP16-only. Pinning the impl to
// K = KvFp16 lets us access `KvCache<B, KvFp16>` fields directly without
// trait-method indirection. K = KvInt8 falls back to per-item decode at
// the engine level (DecoderOnlyLLM::decode_batch + unified_forward report
// Unsupported in the K=KvInt8 specialization).
impl<B: MoeLlmBackend> LlamaFamilyModel<B, KvFp16> {
    fn prepare_batched_decode_stage(
        &mut self,
        ctx: &mut B::Context,
        batch: &[(String, u32, u32)],
    ) -> (usize, usize) {
        let m = batch.len();
        debug_assert!(m > 0);
        for (cid, _, _) in batch {
            self.ensure_kv_in_context(ctx, cid, None);
            let target = self
                .cache_len(cid)
                .checked_add(1)
                .expect("stage KV length overflow");
            // Both layouts own capacity before a stage can encode a KV write.
            // Paged window metadata is only one consumer of that storage.
            self.ensure_paged_kv_capacity_for_cache_id(ctx, cid, target)
                .expect("stage KV capacity");
        }
        self.ensure_scratch_in_context(ctx, m);
        let positions: Vec<u32> = batch.iter().map(|(_, _, p)| *p).collect();
        let kv_pre: Vec<u32> = batch
            .iter()
            .map(|(cid, _, _)| self.kv_caches.get(cid).expect("kv_caches missing")[0].len as u32)
            .collect();
        let kv_post: Vec<u32> = kv_pre.iter().map(|&x| x + 1).collect();
        B::write_typed::<u32>(ctx, &mut self.scratch.batch_positions, &positions);
        B::write_typed::<u32>(ctx, &mut self.scratch.batch_kv_lens_pre, &kv_pre);
        B::write_typed::<u32>(ctx, &mut self.scratch.batch_kv_lens_post, &kv_post);
        if self.paged_pools.is_some()
            && self
                .cfg
                .requires_windowed_attention(self.source_layer_range())
        {
            self.prepare_paged_decode_metadata(ctx, batch);
        }

        (m, self.cfg.hidden_size)
    }

    /// Prepare dynamic paged views before any layer encoding or capture.
    /// Eager windowed attention and captured decode consume the same owners.
    fn prepare_paged_decode_metadata(
        &mut self,
        ctx: &mut B::Context,
        batch: &[(String, u32, u32)],
    ) {
        let m = batch.len();
        assert!(
            m <= self.scratch.paged_max_seqs,
            "batch exceeds original paged metadata capacity"
        );
        for (cid, _, _) in batch {
            let target = self
                .cache_len(cid)
                .checked_add(1)
                .expect("paged length overflow");
            self.ensure_paged_kv_capacity_for_cache_id(ctx, cid, target)
                .expect("prepare windowed paged KV capacity");
        }
        self.scratch.ensure_paged_decode_metadata(ctx);
        let cu_seqlens: Vec<u32> = (0..=m).map(|row| u32::try_from(row).unwrap()).collect();
        let stride = self.scratch.paged_max_blocks_per_seq;
        let mut tables = vec![0u32; m * stride];
        let mut lengths = Vec::with_capacity(m);
        for (row, (cid, _, position)) in batch.iter().enumerate() {
            let caches = &self.kv_caches[cid];
            let blocks = &caches[0].paged_block_indices;
            assert!(blocks.len() <= stride);
            assert!(
                caches.iter().all(|cache| cache.block_size == 16
                    && cache.capacity == stride * 16
                    && cache.paged_block_indices == *blocks
                    && cache.len == *position as usize),
                "paged layers must share original sequence mapping and frontier"
            );
            tables[row * stride..row * stride + blocks.len()].copy_from_slice(blocks);
            lengths.push(position.checked_add(1).expect("paged position overflow"));
        }
        B::write_typed::<u32>(
            ctx,
            self.scratch.paged_batch_cu_seqlens.as_mut().unwrap(),
            &cu_seqlens,
        );
        B::write_typed::<u32>(
            ctx,
            self.scratch.paged_batch_block_tables.as_mut().unwrap(),
            &tables,
        );
        B::write_typed::<u32>(
            ctx,
            self.scratch.paged_batch_context_lens.as_mut().unwrap(),
            &lengths,
        );
    }

    fn bump_local_batched_decode_kv_lengths(&mut self, batch: &[(String, u32, u32)]) {
        let local_layers = self.local_layer_count();
        for (cid, _, _) in batch {
            let caches = self.kv_caches.get_mut(cid).expect("kv_caches missing");
            for cache in caches.iter_mut().take(local_layers) {
                cache.len += 1;
            }
        }
    }

    /// One transformer layer over M items, GEMMs batched + per-item attention.
    pub(crate) fn forward_layer_batched_decode(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        batch: &[(String, u32, u32)],
        residual: &mut B::Buffer,
        m: usize,
        graph_route: bool,
        host_residual: Option<&mut [f32]>,
        device_residual: Option<&mut B::Buffer>,
        device_branch: Option<&mut B::Buffer>,
    ) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        let source_li = self.source_layer_index(li);
        let layer = &self.layers[li];
        let qk_mode: i32 = llama_qk_mode(cfg);
        let dummy_w = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy_w);

        // Match the single-sequence Gemma3 attention schedule. Local layers
        // use the local RoPE table and a sliding window; global layers use
        // the main RoPE table and full causal attention.
        let schedule = llama_layer_attention_schedule(cfg, source_li);
        let layer_window = schedule.sliding_window;
        let (rope_cos, rope_sin) = match (&self.rope_local, schedule.is_global_layer) {
            (Some(local), false) => (&local.cos, &local.sin),
            _ => (&self.rope.cos, &self.rope.sin),
        };

        // 1. rms_norm [M, H]  → norm_out

        if let Some(host) = host_residual.as_deref() {
            Self::rms_norm_host_to_activation(
                ctx,
                host,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );
        } else if let Some(device) = device_residual.as_ref() {
            B::rms_norm_f32_to_activation(
                ctx,
                &**device,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );
        } else {
            B::rms_norm(
                ctx,
                residual,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );
        }

        // 2. qkv_proj (GEMM m=M): norm_out [M, H] → qkv_out [M, QKV]

        {
            layer
                .qkv_proj
                .forward(ctx, &self.scratch.norm_out, &mut self.scratch.qkv_out, m);
        }

        // ── Paged-KV batched path (Phase 4b) ──────────────────────────
        // When paged is on, we skip the contig split_qkv + per-item
        // qk_norm_rope + kv_append + flash_attention loop entirely.
        // Instead:
        //   1. Per item: split_qkv_norm_rope_into_paged_cache with
        //      qkv_byte_offset = i * qkv_stride * 4 reads item i's
        //      slice of qkv_out, writes K/V into the shared pool at
        //      its block_table-resolved position, and stores the
        //      RoPE'd Q at paged_batch_q[i * q_dim .. (i+1) * q_dim].
        //   2. Build batched block_tables [M, max_blocks_per_seq] +
        //      context_lens [M] host-side, write to scratch device
        //      buffers.
        //   3. Single paged_decode_attention(num_seqs=M) reads all M
        //      seqs' K/V via per-seq block_tables, writes to
        //      paged_batch_o.
        //   4. Per item: copy paged_batch_o[i] → attn_flat[i * q_dim].
        //
        // This is the "real" multi-seq decode — one heavy attention
        // dispatch covering all sequences instead of M sequential ones.
        if let Some(pools) = self.paged_pools.as_mut() {
            let pool_ptr = (
                &mut pools[li].0 as *mut B::Buffer,
                &mut pools[li].1 as *mut B::Buffer,
            );
            // SAFETY: pools allocated once; not concurrently mutated.
            let (pool_k, pool_v) = unsafe { (&mut *pool_ptr.0, &mut *pool_ptr.1) };

            let max_blocks_per_seq = self.scratch.paged_max_blocks_per_seq;
            let block_size = 16;
            if graph_route {
                // All per-step metadata was uploaded before capture. This
                // existing dynamic kernel reads positions and block mappings
                // from original scratch on every replay, including new pages.
                B::split_qkv_norm_rope_into_paged_cache_varlen(
                    ctx,
                    &self.scratch.qkv_out,
                    q_norm_w,
                    k_norm_w,
                    rope_cos,
                    rope_sin,
                    self.scratch.paged_batch_q.as_mut().expect("paged Q"),
                    pool_k,
                    pool_v,
                    self.scratch
                        .paged_batch_cu_seqlens
                        .as_ref()
                        .expect("paged graph query rows"),
                    &self.scratch.batch_positions,
                    self.scratch
                        .paged_batch_block_tables
                        .as_ref()
                        .expect("paged graph tables"),
                    m,
                    m,
                    nh,
                    nkv,
                    hd,
                    eps,
                    qk_mode,
                    block_size,
                    max_blocks_per_seq,
                )
                .expect("paged graph dynamic KV write");
            } else {
                let qkv_stride = q_dim + 2 * kv_dim;

                // Step 1: per-item paged write. We collect cache_len + block_indices
                // up front for step 2. Note: this loop borrows self.kv_caches mutably
                // per iteration, so we extract the batched-write parameters first then
                // do the dispatches.
                let mut item_state: Vec<(u32, Vec<u32>)> = Vec::with_capacity(m);
                for (cache_id, _, _) in batch.iter() {
                    let caches = self
                        .kv_caches
                        .get(cache_id)
                        .expect("ensure_kv must be called before forward_layer_batched");
                    let cache = &caches[li];
                    item_state.push((cache.len as u32, cache.paged_block_indices.clone()));
                }

                let elem_size = B::activation_elem_size_bytes();
                let q_head_major_size_bytes = (q_dim * elem_size) as u64;
                let qkv_stride_bytes = (qkv_stride * elem_size) as u64;

                for (i, (cache_id, _, pos)) in batch.iter().enumerate() {
                    let pos_i = *pos as usize;
                    let caches = self
                        .kv_caches
                        .get(cache_id)
                        .expect("paged batched: cache not present");
                    let cache = &caches[li];
                    let bt = cache
                        .block_table
                        .as_ref()
                        .expect("paged batched: block_table missing");
                    let cache_len_before = cache.len;
                    let block_table_ref = bt as *const B::Buffer;
                    // SAFETY: bt is read-only in the dispatch; we don't
                    // mutate self.kv_caches between this raw deref and the
                    // call.
                    let bt_safe: &B::Buffer = unsafe { &*block_table_ref };
                    B::split_qkv_norm_rope_into_paged_cache(
                        ctx,
                        &self.scratch.qkv_out,
                        (i as u64) * qkv_stride_bytes,
                        q_norm_w,
                        k_norm_w,
                        rope_cos,
                        rope_sin,
                        self.scratch
                            .paged_batch_q
                            .as_mut()
                            .expect("paged_batch_q missing"),
                        (i as u64) * q_head_major_size_bytes,
                        pool_k,
                        pool_v,
                        bt_safe,
                        1,
                        nh,
                        nkv,
                        hd,
                        pos_i,
                        eps,
                        qk_mode,
                        cache_len_before,
                        block_size,
                        max_blocks_per_seq,
                    )
                    .expect("paged batched write");
                }

                // Step 2: build the stacked block_tables + post-append
                // context_lens host-side, then upload to device scratch. Do not
                // mutate cache.len here: decode_batch_internal does one central
                // bump after all layers so eager and graph replay stay in sync.
                let mut stacked_bt: Vec<u32> = vec![0u32; m * max_blocks_per_seq];
                let mut stacked_cl: Vec<u32> = vec![0u32; m];
                for (i, (cache_id, _, _)) in batch.iter().enumerate() {
                    let caches = self
                        .kv_caches
                        .get(cache_id)
                        .expect("paged batched: cache not present");
                    let cache = &caches[li];
                    stacked_cl[i] = (cache.len + 1) as u32;
                    let blocks = &cache.paged_block_indices;
                    let n_to_copy = blocks.len().min(max_blocks_per_seq);
                    stacked_bt[i * max_blocks_per_seq..i * max_blocks_per_seq + n_to_copy]
                        .copy_from_slice(&blocks[..n_to_copy]);
                }
                let bt_buf = self
                    .scratch
                    .paged_batch_block_tables
                    .as_mut()
                    .expect("paged_batch_block_tables missing");
                B::write_typed::<u32>(ctx, bt_buf, &stacked_bt);
                let cl_buf = self
                    .scratch
                    .paged_batch_context_lens
                    .as_mut()
                    .expect("paged_batch_context_lens missing");
                B::write_typed::<u32>(ctx, cl_buf, &stacked_cl);
            }

            // Step 3: one batched paged_decode_attention(num_seqs=m).
            let bt_ptr =
                self.scratch.paged_batch_block_tables.as_ref().unwrap() as *const B::Buffer;
            let cl_ptr =
                self.scratch.paged_batch_context_lens.as_ref().unwrap() as *const B::Buffer;
            let q_ptr = self.scratch.paged_batch_q.as_ref().unwrap() as *const B::Buffer;
            let o_ptr = self.scratch.paged_batch_o.as_mut().unwrap() as *mut B::Buffer;
            // SAFETY: the four scratch buffers above are not aliased
            // by anything else; we only deref while &mut self is held.
            let bt_safe = unsafe { &*bt_ptr };
            let cl_safe = unsafe { &*cl_ptr };
            let q_safe = unsafe { &*q_ptr };
            let o_safe = unsafe { &mut *o_ptr };

            if layer_window > 0 {
                // The encoded extent is fixed by original pool geometry, so
                // increasing device positions cannot outgrow captured scratch.
                B::paged_varlen_attention(
                    ctx,
                    q_safe,
                    pool_k,
                    pool_v,
                    o_safe,
                    self.scratch
                        .paged_batch_cu_seqlens
                        .as_ref()
                        .expect("paged query rows"),
                    &self.scratch.batch_positions,
                    bt_safe,
                    m,
                    m,
                    block_size * max_blocks_per_seq,
                    nh,
                    nkv,
                    hd,
                    layer_window,
                    block_size,
                    max_blocks_per_seq,
                )
                .expect("windowed paged batched attention");
            } else {
                B::paged_decode_attention(
                    ctx,
                    q_safe,
                    pool_k,
                    pool_v,
                    o_safe,
                    bt_safe,
                    cl_safe,
                    m,
                    nh,
                    nkv,
                    hd,
                    block_size,
                    max_blocks_per_seq,
                    1, // q_len
                    0, // full causal attention
                    None,
                )
                .expect("paged batched decode");
            }

            // Step 4: per-item copy paged_batch_o[i] → attn_flat[i * q_dim].
            // Both have q_dim floats per item; same head-major-equals-token-major
            // identity collapse used in the contig path.
            for i in 0..m {
                B::copy_slice(
                    ctx,
                    self.scratch.paged_batch_o.as_ref().unwrap(),
                    i * q_dim,
                    &mut self.scratch.attn_flat,
                    i * q_dim,
                    q_dim,
                );
            }

            // Skip the contig split_qkv + per-item loop below.
            return self.forward_layer_batched_decode_post_attn(
                ctx,
                li,
                residual,
                m,
                host_residual,
                device_residual,
                device_branch,
            );
        }

        // 3. split_qkv [M, QKV] → q_buf [M, Q], k_buf [M, KV], v_buf [M, KV]

        B::split_qkv(
            ctx,
            &self.scratch.qkv_out,
            &mut self.scratch.q_buf,
            &mut self.scratch.k_buf,
            &mut self.scratch.v_buf,
            m,
            q_dim,
            kv_dim,
        );

        // 4. Try the batched per-item qk_norm_rope path: one launch
        //    each for Q/K/V instead of M sequential per-item launches.
        //    Saves 3*(M-1) qk_norm_rope dispatches per layer (and at
        //    M=16 with 32 layers that's ~1500 launches, ~10 ms TPOT).
        //    Backends that don't implement it return Err and we drop
        //    back into the per-item loop unchanged.
        // batch_positions, batch_kv_lens_pre, batch_kv_lens_post are
        // populated by `decode_batch_internal` ONCE per step before the
        // layer loop — required so a captured CUDA graph reads from
        // stable buffer contents on replay (per-layer write_u32 inside
        // the captured region would freeze the values).
        let use_batched_qkr = if B::supports_qk_norm_rope_batched_per_item() {
            let q_batched = B::qk_norm_rope_batched_per_item(
                ctx,
                &self.scratch.q_buf,
                q_norm_w,
                rope_cos,
                rope_sin,
                &mut self.scratch.q_normed_batched,
                &self.scratch.batch_positions,
                m,
                nh,
                hd,
                eps,
                qk_mode,
            );
            let k_batched = B::qk_norm_rope_batched_per_item(
                ctx,
                &self.scratch.k_buf,
                k_norm_w,
                rope_cos,
                rope_sin,
                &mut self.scratch.k_normed_batched,
                &self.scratch.batch_positions,
                m,
                nkv,
                hd,
                eps,
                qk_mode,
            );
            // V's "qk_norm_rope" runs in mode=0 (transpose-only). For
            // tokens-per-item=1 in batched decode this is a memcpy — kept
            // for layout-equivalence with the per-item path. Cheap.
            let v_batched = B::qk_norm_rope_batched_per_item(
                ctx,
                &self.scratch.v_buf,
                dummy_w,
                rope_cos,
                rope_sin,
                &mut self.scratch.v_normed_batched,
                &self.scratch.batch_positions,
                m,
                nkv,
                hd,
                eps,
                0,
            );
            q_batched.is_ok() && k_batched.is_ok() && v_batched.is_ok()
        } else {
            false
        };

        assert!(
            !graph_route || use_batched_qkr,
            "graph decode requires dynamic batched QK/RoPE"
        );

        // 5. Batched kv_cache_append (when use_batched_qkr is on): one
        //    launch each for K and V replaces M sequential per-item
        //    kv_append calls. Reads k/v_normed_batched directly so the
        //    M K/V copy_slice dispatches into single buffers also go
        //    away. cache_lens captured BEFORE the bump.
        let mut kv_lens_host: Vec<u32> = Vec::with_capacity(m);
        let mut batched_kv_append_ok = false;

        if use_batched_qkr && B::supports_kv_cache_append_batched_per_cache() {
            let mut k_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut v_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut pre_append_lens: Vec<u32> = Vec::with_capacity(m);
            let mut capacity_first: usize = 0;
            for (cache_id, _, _) in batch.iter() {
                let caches = self
                    .kv_caches
                    .get(cache_id)
                    .expect("kv_caches must be present");
                let cache = &caches[li];
                k_caches_ref.push(&cache.k);
                v_caches_ref.push(&cache.v);
                pre_append_lens.push(cache.len as u32);
                if capacity_first == 0 {
                    capacity_first = cache.capacity;
                }
            }
            // batch_kv_lens_pre is pre-populated by decode_batch_internal.
            // Per-layer slot for K/V append. cache_ptrs is shared
            // between the K and V calls, so V uses an offset slot
            // (layer_idx + MAX_LAYERS_FOR_GRAPH) to keep its captured
            // memcpy reading from a distinct host region. See
            // ferrum-kernels::backend::cuda for the rationale (graph
            // capture records host POINTERS; same slot → all replays
            // read whichever value was written last).
            let k_append_res = B::kv_cache_append_batched_per_cache(
                ctx,
                &k_caches_ref,
                &self.scratch.k_normed_batched,
                &self.scratch.batch_kv_lens_pre,
                capacity_first,
                m,
                nkv,
                hd,
                li,
            );
            let v_append_res = B::kv_cache_append_batched_per_cache(
                ctx,
                &v_caches_ref,
                &self.scratch.v_normed_batched,
                &self.scratch.batch_kv_lens_pre,
                capacity_first,
                m,
                nkv,
                hd,
                li + MAX_LAYERS_FOR_GRAPH,
            );
            if graph_route {
                k_append_res.unwrap_or_else(|error| std::panic::panic_any(error));
                v_append_res.unwrap_or_else(|error| std::panic::panic_any(error));
                batched_kv_append_ok = true;
            } else {
                batched_kv_append_ok = k_append_res.is_ok() && v_append_res.is_ok();
            }

            // Note: cache.len bump moved to decode_batch_internal post-forward
            // (so a graph replay doesn't double-bump). No-op here.
        }

        // kv_lens_host no longer used; flash_attn reads
        // scratch.batch_kv_lens_post (also pre-populated).
        let _ = kv_lens_host;

        // 6. Per-item loop: only runs when the batched paths are NOT in
        // effect, or when batched_kv_append failed (Err fallback). Gemma
        // local-window layers still use the batched attention kernel:
        // `layer_window` is passed into flash_attention_batched_per_cache.
        for (i, (cache_id, _token, pos)) in batch.iter().enumerate() {
            if use_batched_qkr && batched_kv_append_ok {
                // Already handled by batched kv_append above. Skip
                // per-item copy + kv_append + per-item flash_attn.
                continue;
            }
            let pos_i = *pos as usize;

            if use_batched_qkr {
                // batched_kv_append fallback: still need per-item
                // copy_slice for K/V into single buffers for the
                // per-item kv_append below.
                if !batched_kv_append_ok {
                    B::copy_slice(
                        ctx,
                        &self.scratch.k_normed_batched,
                        i * kv_dim,
                        &mut self.scratch.k_head_major_single,
                        0,
                        kv_dim,
                    );
                    B::copy_slice(
                        ctx,
                        &self.scratch.v_normed_batched,
                        i * kv_dim,
                        &mut self.scratch.v_head_major_single,
                        0,
                        kv_dim,
                    );
                }
            } else {
                // Fallback: extract item i's Q/K/V then run per-item
                // qk_norm_rope. Same dispatch budget as before this
                // commit — used on backends without the batched kernel.
                B::copy_slice(
                    ctx,
                    &self.scratch.q_buf,
                    i * q_dim,
                    &mut self.scratch.q_single,
                    0,
                    q_dim,
                );
                B::copy_slice(
                    ctx,
                    &self.scratch.k_buf,
                    i * kv_dim,
                    &mut self.scratch.k_single,
                    0,
                    kv_dim,
                );
                B::copy_slice(
                    ctx,
                    &self.scratch.v_buf,
                    i * kv_dim,
                    &mut self.scratch.v_single,
                    0,
                    kv_dim,
                );

                B::qk_norm_rope(
                    ctx,
                    &self.scratch.q_single,
                    q_norm_w,
                    rope_cos,
                    rope_sin,
                    &mut self.scratch.q_head_major_single,
                    1,
                    nh,
                    hd,
                    pos_i,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    &self.scratch.k_single,
                    k_norm_w,
                    rope_cos,
                    rope_sin,
                    &mut self.scratch.k_head_major_single,
                    1,
                    nkv,
                    hd,
                    pos_i,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    &self.scratch.v_single,
                    dummy_w,
                    rope_cos,
                    rope_sin,
                    &mut self.scratch.v_head_major_single,
                    1,
                    nkv,
                    hd,
                    pos_i,
                    eps,
                    0,
                );
            }

            // KV append for item i's cache.
            let caches = self
                .kv_caches
                .get_mut(cache_id)
                .expect("ensure_kv must be called before forward_layer_batched");
            let cache = &mut caches[li];
            if !(use_batched_qkr && batched_kv_append_ok) {
                B::kv_cache_append_head_major(
                    ctx,
                    &mut cache.k,
                    &mut cache.v,
                    cache.len,
                    cache.capacity,
                    &self.scratch.k_head_major_single,
                    &self.scratch.v_head_major_single,
                    1,
                    nkv,
                    hd,
                );
            }
            // cache.len bump moved to decode_batch_internal post-forward
            // for graph-replay correctness. flash_attn below uses
            // cache.len + 1 directly.
            let kv_len = cache.len + 1;
            let kv_stride = cache.capacity;
            kv_lens_host.push(kv_len as u32);

            // Per-item flash_attn runs only when the batched qkr fallback is
            // in use. If batched qkr works, the single-launch batched
            // attention below covers both full and local-window layers.
            if !use_batched_qkr {
                let attn_cfg = ferrum_kernels::backend::AttnConfig {
                    num_heads: nh,
                    num_kv_heads: nkv,
                    head_dim: hd,
                    causal: true,
                    scale: 1.0 / (hd as f32).sqrt(),
                    kv_seq_stride: kv_stride,
                    sliding_window: layer_window,
                };
                B::flash_attention(
                    ctx,
                    &self.scratch.q_head_major_single,
                    &cache.k,
                    &cache.v,
                    &mut self.scratch.attn_head_major_single,
                    1,
                    1,
                    kv_len,
                    pos_i,
                    &attn_cfg,
                );
                // For tokens=1 head-major and token-major are
                // byte-identical, so just copy into the per-item slot
                // of attn_flat without a transpose dispatch.
                B::copy_slice(
                    ctx,
                    &self.scratch.attn_head_major_single,
                    0,
                    &mut self.scratch.attn_flat,
                    i * q_dim,
                    q_dim,
                );
            }
        }

        // 7. Batched flash_attention: one launch covers all M items.
        //    Reads q_normed_batched directly (item-major) and writes
        //    output straight into attn_flat at [m, q_dim] item-major.
        //    On Err (backend lacks the batched kernel) we fall through
        //    to a per-item flash_attn loop that mirrors the original
        //    code path.
        if use_batched_qkr {
            let mut k_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut v_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut max_kv = 0usize;
            let mut capacity_for_kernel = 0usize;
            for (cache_id, _, _) in batch.iter() {
                let caches = self
                    .kv_caches
                    .get(cache_id)
                    .expect("kv_caches must be present");
                let cache = &caches[li];
                k_caches_ref.push(&cache.k);
                v_caches_ref.push(&cache.v);
                // POST-append valid_kv_len: kv_cache_append_batched_per_cache
                // wrote position cache.len, so attention reads cache.len + 1
                // entries. Pre-append cache.len under-sized the shared mem
                // and corrupted s_scores (silent garbage tokens at m≥2).
                let post_len = cache.len + 1;
                if post_len > max_kv {
                    max_kv = post_len;
                }
                if capacity_for_kernel == 0 {
                    capacity_for_kernel = cache.capacity;
                }
            }
            let scale = 1.0 / (hd as f32).sqrt();
            // batch_kv_lens_post pre-populated by decode_batch_internal.
            // flash_attn_batched uses its own k_ptrs/v_ptrs host
            // arrays in CudaState (separate from kv_cache_append's
            // cache_ptrs), so per-layer slot = li is sufficient.

            let mut batched_attn_ok = false;
            if B::supports_flash_attention_batched_per_cache() {
                let batched_attn_res = B::flash_attention_batched_per_cache(
                    ctx,
                    &self.scratch.q_normed_batched,
                    &k_caches_ref,
                    &v_caches_ref,
                    &self.scratch.batch_kv_lens_post,
                    &mut self.scratch.attn_flat,
                    nh,
                    nkv,
                    hd,
                    scale,
                    if graph_route {
                        ferrum_kernels::backend::attention_score_capacity_bucket(max_kv)
                            .min(capacity_for_kernel)
                    } else {
                        max_kv
                    },
                    capacity_for_kernel,
                    layer_window,
                    li,
                );
                if graph_route {
                    batched_attn_res.unwrap_or_else(|error| std::panic::panic_any(error));
                    batched_attn_ok = true;
                } else {
                    batched_attn_ok = batched_attn_res.is_ok();
                }
            }

            if !batched_attn_ok {
                // Per-item flash_attn fallback for backends that
                // implement the batched qkr but not the batched attn.
                for (i, (cache_id, _, pos)) in batch.iter().enumerate() {
                    let pos_i = *pos as usize;
                    // Populate q_head_major_single from the normed Q.
                    B::copy_slice(
                        ctx,
                        &self.scratch.q_normed_batched,
                        i * q_dim,
                        &mut self.scratch.q_head_major_single,
                        0,
                        q_dim,
                    );
                    let caches = self
                        .kv_caches
                        .get(cache_id)
                        .expect("kv_caches must be present");
                    let cache = &caches[li];
                    let kv_stride = cache.capacity;
                    let attn_cfg = ferrum_kernels::backend::AttnConfig {
                        num_heads: nh,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        causal: true,
                        scale: 1.0 / (hd as f32).sqrt(),
                        kv_seq_stride: kv_stride,
                        sliding_window: layer_window,
                    };
                    B::flash_attention(
                        ctx,
                        &self.scratch.q_head_major_single,
                        &cache.k,
                        &cache.v,
                        &mut self.scratch.attn_head_major_single,
                        1,
                        1,
                        cache.len + 1,
                        pos_i,
                        &attn_cfg,
                    );
                    B::copy_slice(
                        ctx,
                        &self.scratch.attn_head_major_single,
                        0,
                        &mut self.scratch.attn_flat,
                        i * q_dim,
                        q_dim,
                    );
                }
            }
        }

        self.forward_layer_batched_decode_post_attn(
            ctx,
            li,
            residual,
            m,
            host_residual,
            device_residual,
            device_branch,
        );
    }

    pub(crate) fn forward_layer_batched_decode_post_attn(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        residual: &mut B::Buffer,
        m: usize,
        host_residual: Option<&mut [f32]>,
        device_residual: Option<&mut B::Buffer>,
        device_branch: Option<&mut B::Buffer>,
    ) {
        let layer = &self.layers[li];

        // 7. o_proj (GEMM m=M): attn_flat [M, Q] → o_proj_out [M, H]

        {
            layer.o_proj.forward(
                ctx,
                &self.scratch.attn_flat,
                &mut self.scratch.o_proj_out,
                m,
            );
        }

        self.forward_layer_post_o_proj_with_residual_shadow(
            ctx,
            li,
            None,
            residual,
            m,
            host_residual,
            device_residual,
            device_branch,
        );
    }

    /// Unified mixed-batch forward (chunked-prefill workhorse).
    ///
    /// Each item is `(cache_id, q_tokens, pos_offset, is_final_chunk)`:
    /// - `q_tokens.len() == 1` is a decode step
    /// - `q_tokens.len() >= 1` with `pos_offset > 0` is a continuing
    ///   prefill chunk
    /// - `is_final_chunk == true` ⇒ logits returned for sampling, else
    ///   `None` (intermediate prefill chunks just advance KV state)
    ///
    /// Concatenates all q_tokens into a single `[M_total, hidden]`
    /// forward, dispatches per-item `split_qkv_norm_rope_into_paged_cache`
    /// to write per-seq K/V into the paged pool with correct RoPE per
    /// token position, then a single `paged_varlen_attention` call
    /// (Step 4 kernel) handles attention for all q-tokens with per-seq
    /// causal masks.
    ///
    /// REQUIRES paged KV (self.paged_pools.is_some()). Caller (the
    /// public `unified_forward` impl on the trait) returns
    /// `Err(unsupported)` for the contig path so the engine falls
    /// back to legacy dispatch.
    pub(crate) fn unified_forward_internal(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
        logits_policies: Option<&[LogitsReturnPolicy]>,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        self.unified_sampling_internal(items, logits_policies)?
            .into_iter()
            .map(|output| output.map(legacy_sampling_values).transpose())
            .collect()
    }

    fn unified_sampling_internal(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
        logits_policies: Option<&[LogitsReturnPolicy]>,
    ) -> Result<Vec<Option<ExecutorSamplingOutput>>> {
        for (cache_id, tokens, position, _) in items {
            self.validate_prepared_legacy_write(cache_id, *position, tokens.len(), Some(tokens))?;
        }
        if items.is_empty() {
            return Ok(Vec::new());
        }
        self.run_device_frame(|model, frame| model.unified_in_frame(frame, items, logits_policies))
            .into_legacy_result()
    }

    fn unified_in_frame(
        &mut self,
        frame: &mut LlamaExecutionFrame<B>,
        items: &[(String, Vec<u32>, usize, bool)],
        logits_policies: Option<&[LogitsReturnPolicy]>,
    ) -> Result<Vec<Option<ExecutorSamplingOutput>>> {
        let prior_graph_failure = self.unified_graph_failed;
        if self.unified_graph_retirement_pending {
            // Completion recovery does not retire the original capture closure.
            // Check it once, even if capture failed before publishing a key.
            // On error, preserve both the original owners and this obligation.
            B::retire_graphs_for_storage_mutation(&mut frame.context)?;
            self.clear_graph_bookkeeping();
            self.unified_graph_failed = prior_graph_failure;
        }
        // Snapshot cfg fields into Copy locals so we can take &mut self later
        // without a long-lived `&self.cfg` borrow conflicting.
        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_heads;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let im = self.cfg.intermediate_size;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let qkv_stride = q_dim + 2 * kv_dim;
        let eps = self.cfg.rms_norm_eps;
        let qk_mode: i32 = llama_qk_mode(&self.cfg);
        let vocab = self.cfg.vocab_size;
        let num_layers = self.cfg.num_layers;
        let num_seqs = items.len();

        // Per-item bookkeeping (host) — shared helpers from
        // `common::decoder_unified` so Qwen3-MoE / future decoder
        // families don't re-implement cu_seqlens construction etc.
        let (q_lens, cu_seqlens_q, m_total) =
            crate::common::decoder_unified::compute_cu_seqlens_q(items);
        let pos_offsets = crate::common::decoder_unified::compute_pos_offsets(items);
        let max_kv_len = crate::common::decoder_unified::compute_max_kv_len(items);
        let final_indices =
            crate::common::decoder_unified::compute_final_indices(items, &cu_seqlens_q);
        let num_sampled = final_indices.len();

        // Ensure all items' KV caches exist.
        for (cid, _, _, _) in items {
            self.ensure_kv_in_context(&mut frame.context, cid, None);
        }

        // Concatenated input tokens for one embedding_lookup.
        let all_tokens = crate::common::decoder_unified::concat_q_tokens(items);
        debug_assert_eq!(all_tokens.len(), m_total);

        // Paged path requirements.
        let pools_present = self.paged_pools.is_some();
        if !pools_present {
            // Caller should have routed to fallback; defensive.
            panic!("unified_forward_internal called without paged_pools — caller must check");
        }
        let max_blocks_per_seq = self.scratch.paged_max_blocks_per_seq;
        let block_size = 16usize; // matches PAGED_BLOCK_SIZE in ensure_kv

        for (cid, tokens, pos_offset, _) in items {
            self.ensure_paged_kv_capacity_for_cache_id(
                &mut frame.context,
                cid,
                *pos_offset + tokens.len(),
            )?;
        }

        // Make sure scratch fits this batch. Use the engine-side
        // `paged_max_seqs` cap (set in `enable_paged_batch` from
        // `FERRUM_PAGED_MAX_SEQS`, default 32) so the index buffers
        // (`unified_cu_seqlens_q`, `unified_pos_offsets`,
        // `unified_block_tables`, `unified_packed_*`) are sized for the
        // largest batch the engine will ever submit. Earlier we used
        // `num_seqs` and the buffers were pinned at the FIRST call's
        // batch size — c=1 warmup → c=16 bench would write 17 ints into
        // a 2-int buffer (CUDA_ERROR_INVALID_VALUE).
        debug_assert!(
            self.scratch.paged_max_seqs >= num_seqs,
            "unified_forward batch ({} items) exceeds engine paged_max_seqs ({})",
            num_seqs,
            self.scratch.paged_max_seqs,
        );
        let max_seqs = self.scratch.paged_max_seqs.max(num_seqs);
        // Take cfg ref only for the immediate ensure call (Copy fields
        // already snapshotted above into h/nh/etc — cfg is just for the
        // shape constants ensure_unified_scratch needs).
        let cfg_for_alloc = self.cfg.clone();
        let unified_scratch_rebuilt = self.scratch.ensure_unified_scratch(
            &mut frame.context,
            &cfg_for_alloc,
            m_total,
            max_seqs,
            max_blocks_per_seq,
        );
        if unified_scratch_rebuilt {
            self.clear_graph_bookkeeping();
            self.unified_graph_failed = prior_graph_failure;
        }

        let use_device_residual_shadow =
            self.cfg.sandwich_norms && B::supports_device_f32_residual_shadow();
        frame
            .take_scratch(
                &mut self.scratch,
                FrameLayout::Unified,
                use_device_residual_shadow,
            )
            .expect("unified frame scratch");
        let mut ctx = &mut frame.context;
        let mut residual = frame.residual.as_mut().expect("frame unified residual");
        // Layers and final packing borrow the same frame-owned buffer. The
        // model scratch slot stays empty until actual context retirement.
        let mut norm_out = frame.norm_out.as_mut().expect("frame unified norm_out");
        let embed = self
            .embed
            .as_ref()
            .expect("unified_forward_internal called on backbone-only model");
        B::embedding_lookup(&mut ctx, embed, &all_tokens, &mut residual, h);
        if let Some(scale) = self.cfg.embed_scale {
            B::scale_inplace(&mut ctx, &mut residual, scale, m_total * h);
        }
        let device_residual_shadow = &mut frame.residual_f32;
        if let Some(shadow) = device_residual_shadow.as_mut() {
            B::activation_to_f32_shadow(&mut ctx, &residual, shadow, m_total * h);
        }
        let device_branch_shadow = &mut frame.branch_f32;

        // Upload index buffers.
        {
            let csq = self
                .scratch
                .unified_cu_seqlens_q
                .as_mut()
                .expect("unified_cu_seqlens_q missing");
            B::write_typed::<u32>(&mut ctx, csq, &cu_seqlens_q);
        }
        {
            let po = self
                .scratch
                .unified_pos_offsets
                .as_mut()
                .expect("unified_pos_offsets missing");
            B::write_typed::<u32>(&mut ctx, po, &pos_offsets);
        }
        // Stack per-seq block tables host-side, then upload.
        let stacked =
            crate::common::decoder_unified::stack_block_tables(items, max_blocks_per_seq, |cid| {
                self.kv_caches
                    .get(cid)
                    .expect("kv cache missing for unified item")[0]
                    .paged_block_indices
                    .clone()
            });
        {
            let bt = self
                .scratch
                .unified_block_tables
                .as_mut()
                .expect("unified_block_tables missing");
            B::write_typed::<u32>(&mut ctx, bt, &stacked);
        }

        // ── CUDA-graph capture/replay control ────────────────────────
        // `FERRUM_UNIFIED_GRAPH=1` opts in. The full graph key includes launch
        // shape plus host-side decisions that capture records as constants
        // (attention kv/shared-memory shape and final-row copy offsets).
        // `FERRUM_UNIFIED_GRAPH_LAYERS_ONLY=1` switches to a diagnostic scope
        // that captures only the layer loop, leaving final rms_norm,
        // per-item copy_slice, and lm_head eager. Pre-work (write_u32 +
        // embedding_lookup) and post-work (sync + to_vec) stay eager.
        //
        // Status (2026-05-05): WORKING. The earlier
        // `gather_columns launch: CUDA_ERROR_INVALID_VALUE` was caused
        // by `with_marlin_gather_scratch` doing an in-place
        // `stream.alloc::<f16>` when its slot was too small — runtime
        // alloc inside a captured stream is illegal. Fix:
        // `B::pregrow_marlin_gather_scratch(m_total * intermediate_size)`
        // called eagerly above, BEFORE this section.
        //
        // Bench (RTX 4090, Llama-3.1-8B GPTQ-INT4, c=16):
        //   varlen-only:     680 tok/s, TPOT 19.3 ms
        //   varlen + graph:  714 tok/s, TPOT 18.3 ms  (+5% / -5%)
        let graph_enabled = self.batched_cfg.unified_graph && !self.unified_graph_failed;
        let graph_scope = UnifiedGraphCaptureScope::from_config(&self.batched_cfg);
        let graph_shape = if graph_enabled {
            // The original backend selects the actual route, validates native
            // shared memory and prepares its own scratch before any capture.
            // No model-side copy of the split-K policy participates in keying.
            let attention =
                B::prepare_paged_varlen_attention(&mut ctx, num_seqs, m_total, max_kv_len, nh, hd)?;
            Some(UnifiedGraphShape {
                scope: graph_scope,
                rows: m_total,
                sequences: num_seqs,
                attention,
                final_indices: if graph_scope.captures_final_pack() {
                    final_indices.clone()
                } else {
                    Vec::new()
                },
            })
        } else {
            None
        };
        let cached_key = graph_shape
            .as_ref()
            .and_then(|shape| self.unified_graph_keys_seen.get(shape))
            .copied();
        let cache_has_key = cached_key.is_some();
        let graph_key =
            cached_key.unwrap_or_else(|| next_unified_graph_key(self.unified_graph_next_key));

        const UNIFIED_GRAPH_WARMUP: usize = 3;
        let should_capture = graph_enabled
            && !cache_has_key
            && self.unified_graph_warmup >= UNIFIED_GRAPH_WARMUP
            && unified_graph_capture_skip_reason(
                m_total,
                num_seqs,
                self.unified_graph_keys_seen.len(),
            )
            .is_none();
        let should_replay = graph_enabled && cache_has_key;
        if should_capture || should_replay {
            // Cover fallible pre-launch synchronization, capture-body panics,
            // and successful launches whose terminal observation later fails.
            self.unified_graph_failed = true;
            self.unified_graph_retirement_pending = true;
        }
        let mut graph_executed = false;

        // Pre-grow the marlin gather scratch slot to the worst-case
        // size for THIS call's matmul shapes. Done eagerly BEFORE
        // begin_capture: `with_marlin_gather_scratch`'s in-place grow
        // does `stream.alloc` which is forbidden inside a captured
        // stream (CUDA_ERROR_INVALID_VALUE on the next launch).
        // `down_proj` has the largest k = intermediate_size, so
        // m_total * intermediate_size is the upper bound across all
        // 4 matmuls in the layer.
        let max_marlin_required = m_total * im;
        B::pregrow_marlin_gather_scratch(&mut ctx, max_marlin_required);

        // Sync eager pre-work (write_u32 + embedding + scratch grow)
        // before either a replay or a capture starts — buffer state
        // must be settled.
        device_execution::checked_sync::<B>(&mut ctx);

        macro_rules! run_unified_layers {
            () => {
                for li in 0..num_layers {
                    self.unified_forward_layer(
                        &mut ctx,
                        li,
                        items,
                        &q_lens,
                        &cu_seqlens_q,
                        &pos_offsets,
                        &mut residual,
                        &mut norm_out,
                        device_residual_shadow.as_mut(),
                        device_branch_shadow.as_mut(),
                        m_total,
                        max_kv_len,
                        num_seqs,
                        max_blocks_per_seq,
                        block_size,
                        qkv_stride,
                        q_dim,
                        kv_dim,
                        nh,
                        nkv,
                        hd,
                        im,
                        h,
                        eps,
                        qk_mode,
                    );
                }
            };
        }

        let mut did_pure_replay = false;
        if should_replay {
            // SAFETY: cold construction checked projection/KV capture contracts.
            // The private model/frame owns all bindings through checked retirement;
            // storage mutation retires cached graphs before replacing dependencies.
            match unsafe { B::replay_graph(&mut ctx, graph_key) } {
                Ok(true) => {
                    did_pure_replay = true;
                    graph_executed = true;
                }
                Ok(false) => {
                    B::retire_graphs_for_storage_mutation(&mut ctx)?;
                    // A missing native key submitted no graph. After checked
                    // retirement this invocation runs eagerly and starts a new
                    // warmup; it is not a permanent execution failure.
                    self.clear_graph_bookkeeping();
                }
                Err(e) => {
                    self.unified_graph_failed = true;
                    std::panic::panic_any(e);
                }
            }
        }

        if !did_pure_replay {
            self.unified_graph_warmup += 1;
        }

        if should_capture {
            // SAFETY: cold construction checked projection/KV capture contracts.
            // The private model/frame owns all bindings through checked retirement;
            // storage mutation retires cached graphs before replacing dependencies.
            if let Err(e) = unsafe { B::begin_graph_capture(&mut ctx) } {
                self.unified_graph_failed = true;
                std::panic::panic_any(e);
            }
        }

        let mut capture_pending = should_capture;
        macro_rules! require_unified_capture {
            () => {
                if !B::graph_capture_in_flight(&ctx) {
                    std::panic::panic_any(ferrum_types::FerrumError::backend(
                        "unified capture ended before graph installation",
                    ));
                }
            };
        }
        if capture_pending {
            require_unified_capture!();
        }
        let mut layers_ready = did_pure_replay;
        let mut final_pack_ready = did_pure_replay && graph_scope.captures_final_pack();
        let mut lm_head_ready = did_pure_replay && graph_scope.captures_lm_head();
        macro_rules! finish_unified_graph_capture {
            () => {{
                require_unified_capture!();
                // SAFETY: cold construction checked projection/KV capture contracts.
                // The private model/frame owns all bindings through checked retirement;
                // storage mutation retires cached graphs before replacing dependencies.
                if let Err(e) = unsafe { B::end_graph_capture(&mut ctx, graph_key) } {
                    self.unified_graph_failed = true;
                    std::panic::panic_any(e);
                } else {
                    self.unified_graph_keys_seen.insert(
                        graph_shape
                            .as_ref()
                            .expect("captured unified shape")
                            .clone(),
                        graph_key,
                    );
                    self.unified_graph_next_key = self
                        .unified_graph_next_key
                        .checked_add(1)
                        .expect("unified graph native ID exhausted");

                    // SAFETY: cold construction checked projection/KV capture contracts.
                    // The private model/frame owns all bindings through checked retirement;
                    // storage mutation retires cached graphs before replacing dependencies.
                    match unsafe { B::replay_graph(&mut ctx, graph_key) } {
                        Ok(true) => {
                            graph_executed = true;
                            capture_pending = false;
                            layers_ready = true;
                            final_pack_ready = graph_scope.captures_final_pack();
                            lm_head_ready = graph_scope.captures_lm_head();
                        }
                        Ok(false) => {
                            self.unified_graph_failed = true;
                            std::panic::panic_any(ferrum_types::FerrumError::backend(
                                "captured unified graph was unavailable before its first execution",
                            ));
                        }
                        Err(e) => {
                            self.unified_graph_failed = true;
                            std::panic::panic_any(e);
                        }
                    }
                }
            }};
        }
        if !did_pure_replay {
            run_unified_layers!();
            if capture_pending {
                require_unified_capture!();
                if graph_scope == UnifiedGraphCaptureScope::LayersOnly {
                    finish_unified_graph_capture!();
                }
            } else {
                layers_ready = true;
            }
        }

        macro_rules! run_unified_final_pack {
            () => {{
                if let Some(device) = device_residual_shadow.as_ref() {
                    B::rms_norm_f32_to_activation(
                        &mut ctx,
                        device,
                        &self.final_norm_w,
                        eps,
                        &mut norm_out,
                        m_total,
                        h,
                    );
                } else {
                    B::rms_norm(
                        &mut ctx,
                        &residual,
                        &self.final_norm_w,
                        eps,
                        &mut norm_out,
                        m_total,
                        h,
                    );
                }

                if num_sampled > 0 {
                    let packed_normed = self
                        .scratch
                        .unified_packed_normed
                        .as_mut()
                        .expect("unified_packed_normed missing");
                    for (j, &(_, global)) in final_indices.iter().enumerate() {
                        B::copy_slice(&mut ctx, &norm_out, global * h, packed_normed, j * h, h);
                    }
                }
            }};
        }

        macro_rules! run_unified_lm_head {
            () => {{
                if num_sampled > 0 {
                    let packed_normed = self
                        .scratch
                        .unified_packed_normed
                        .as_ref()
                        .expect("unified_packed_normed missing");
                    let lm_head = self
                        .lm_head
                        .as_ref()
                        .expect("unified_forward_internal called on backbone-only model");
                    let packed_logits = self
                        .scratch
                        .unified_packed_logits
                        .as_mut()
                        .expect("unified_packed_logits missing");
                    {
                        lm_head.forward(&mut ctx, packed_normed, packed_logits, num_sampled);
                    }
                }
            }};
        }

        if !final_pack_ready {
            run_unified_final_pack!();
            if capture_pending {
                require_unified_capture!();
                if graph_scope == UnifiedGraphCaptureScope::LmHeadEager {
                    finish_unified_graph_capture!();
                }
            } else {
                final_pack_ready = true;
            }
        }

        if !lm_head_ready {
            run_unified_lm_head!();
            if capture_pending {
                require_unified_capture!();
                debug_assert_eq!(graph_scope, UnifiedGraphCaptureScope::Full);
                finish_unified_graph_capture!();
            } else {
                lm_head_ready = true;
            }
        }

        debug_assert!(!capture_pending);
        debug_assert!(layers_ready);
        debug_assert!(final_pack_ready);
        debug_assert!(lm_head_ready);

        // Readback. Keep full logits only for rows whose policy
        // requires CPU-side sampling. Policy-compatible greedy rows use the
        // existing GPU argmax kernels and return a one-element token sentinel.
        let mut quiescent = Some(device_execution::checked_sync::<B>(&mut ctx));
        let mut out: Vec<Option<ExecutorSamplingOutput>> = (0..items.len()).map(|_| None).collect();
        if num_sampled > 0 {
            let logits_return =
                resolve_unified_logits_return(logits_policies, items.len(), &final_indices);
            let argmax_mode = match &logits_return {
                DecodeLogitsReturn::Full => None,
                DecodeLogitsReturn::LegacyDefault if self.batched_cfg.greedy_argmax => {
                    Some(ArgmaxMode::Raw)
                }
                DecodeLogitsReturn::LegacyDefault => None,
                DecodeLogitsReturn::GreedyArgmax {
                    token_mask,
                    repetition_penalties,
                } if has_sparse_repetition_penalties(repetition_penalties) => {
                    B::supports_argmax_rows_f16_sparse_repetition_penalty().then_some(
                        ArgmaxMode::SparseRepetition {
                            token_mask: *token_mask,
                            repetition_penalties,
                        },
                    )
                }
                DecodeLogitsReturn::GreedyArgmax { token_mask, .. } => {
                    token_mask.map(ArgmaxMode::Masked).or(Some(ArgmaxMode::Raw))
                }
            };
            let mut returned_argmax = false;
            if let Some(argmax_mode) = argmax_mode {
                quiescent = None;
                let tokens = match argmax_mode {
                    ArgmaxMode::Raw => {
                        let packed_logits = self
                            .scratch
                            .unified_packed_logits
                            .as_ref()
                            .expect("unified_packed_logits missing");
                        B::argmax_rows_f16(&mut ctx, packed_logits, num_sampled, vocab)
                    }
                    ArgmaxMode::Masked(mask) => {
                        self.scratch.ensure_argmax_token_mask(&mut ctx, mask);
                        let mask_len = self.scratch.argmax_token_mask_len;
                        let device_mask = self
                            .scratch
                            .argmax_token_mask
                            .as_ref()
                            .expect("argmax token mask upload failed");
                        let packed_logits = self
                            .scratch
                            .unified_packed_logits
                            .as_ref()
                            .expect("unified_packed_logits missing");
                        B::argmax_rows_f16_masked(
                            &mut ctx,
                            packed_logits,
                            device_mask,
                            mask_len,
                            num_sampled,
                            vocab,
                        )
                    }
                    ArgmaxMode::SparseRepetition {
                        token_mask,
                        repetition_penalties,
                    } => {
                        self.scratch
                            .ensure_unified_argmax_logits(&mut ctx, &self.cfg, num_sampled);
                        if let Some(mask) = token_mask {
                            self.scratch.ensure_argmax_token_mask(&mut ctx, mask);
                        }
                        let total_token_ids = self
                            .scratch
                            .upload_argmax_repetition_penalties(&mut ctx, repetition_penalties);
                        {
                            let packed_logits = self
                                .scratch
                                .unified_packed_logits
                                .as_ref()
                                .expect("unified_packed_logits missing");
                            let selection_logits = self
                                .scratch
                                .unified_argmax_logits
                                .as_mut()
                                .expect("unified_argmax_logits missing");
                            B::copy_slice(
                                &mut ctx,
                                packed_logits,
                                0,
                                selection_logits,
                                0,
                                num_sampled * vocab,
                            );
                        }
                        let valid_token_mask = token_mask.map(|_| {
                            (
                                self.scratch
                                    .argmax_token_mask
                                    .as_ref()
                                    .expect("argmax token mask upload failed"),
                                self.scratch.argmax_token_mask_len,
                            )
                        });
                        B::argmax_rows_f16_sparse_repetition_penalty(
                            &mut ctx,
                            self.scratch
                                .unified_argmax_logits
                                .as_mut()
                                .expect("unified_argmax_logits missing"),
                            valid_token_mask,
                            self.scratch
                                .argmax_repetition_offsets
                                .as_ref()
                                .expect("argmax repetition offsets missing"),
                            self.scratch
                                .argmax_repetition_token_ids
                                .as_ref()
                                .expect("argmax repetition token ids missing"),
                            self.scratch
                                .argmax_repetition_penalties
                                .as_ref()
                                .expect("argmax repetition penalties missing"),
                            total_token_ids,
                            num_sampled,
                            vocab,
                        )
                    }
                };
                if let Ok(tokens) = tokens {
                    if tokens.iter().all(|&token| token != u32::MAX) {
                        for (j, &(orig_idx, _)) in final_indices.iter().enumerate() {
                            out[orig_idx] =
                                Some(ExecutorSamplingOutput::GreedyToken(TokenId::new(tokens[j])));
                        }
                        returned_argmax = true;
                    }
                }
            }
            let logits_flat = if returned_argmax {
                Vec::new()
            } else {
                let packed_logits = self
                    .scratch
                    .unified_packed_logits
                    .as_ref()
                    .expect("unified_packed_logits missing");
                read_full_logits_after_selection::<B>(
                    &mut ctx,
                    packed_logits,
                    num_sampled * vocab,
                    &mut quiescent,
                )
            };

            if !returned_argmax {
                for (j, &(orig_idx, _)) in final_indices.iter().enumerate() {
                    let row = logits_flat[j * vocab..(j + 1) * vocab].to_vec();
                    out[orig_idx] = Some(ExecutorSamplingOutput::FullLogits(row));
                }
            }
        }

        // Argmax can encode work after the first logits-ready observation.
        // Obtain its actual terminal proof before committing any host frontier.
        let quiescent = quiescent.unwrap_or_else(|| device_execution::checked_sync::<B>(&mut ctx));

        // Bump cache.len for each item (we wrote q_lens[i] tokens into seq i's
        // KV pool inside the layer loop). Centralised post-loop bump matches
        // the pattern in decode_batch_internal.
        for (i, (cid, _, _, _)) in items.iter().enumerate() {
            let caches = self
                .kv_caches
                .get_mut(cid)
                .expect("kv cache missing for unified item post-loop");
            for c in caches.iter_mut() {
                c.len += q_lens[i];
            }
        }

        if graph_executed {
            self.unified_graph_failed = false;
            self.unified_graph_retirement_pending = false;
        }
        frame.completed = Some(quiescent);

        Ok(out)
    }

    /// One transformer layer for the unified mixed-batch forward.
    /// Mirrors `forward_layer_batched_decode` paged path but operates
    /// on `[M_total, *]` tensors and uses `paged_varlen_attention`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn unified_forward_layer(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        items: &[(String, Vec<u32>, usize, bool)],
        q_lens: &[usize],
        cu_seqlens_q: &[u32],
        pos_offsets: &[u32],
        residual: &mut B::Buffer,
        norm_out: &mut B::Buffer,
        mut device_residual: Option<&mut B::Buffer>,
        mut device_branch: Option<&mut B::Buffer>,
        m_total: usize,
        max_kv_len: usize,
        num_seqs: usize,
        max_blocks_per_seq: usize,
        block_size: usize,
        qkv_stride: usize,
        q_dim: usize,
        kv_dim: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        im: usize,
        h: usize,
        eps: f32,
        qk_mode: i32,
    ) {
        let source_li = self.source_layer_index(li);
        let layer = &self.layers[li];
        let dummy_w = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy_w);
        let schedule = llama_layer_attention_schedule(&self.cfg, source_li);
        let layer_window = schedule.sliding_window;
        let (rope_cos, rope_sin) = match (&self.rope_local, schedule.is_global_layer) {
            (Some(local), false) => (&local.cos, &local.sin),
            _ => (&self.rope.cos, &self.rope.sin),
        };

        // 1. rms_norm [M_total, h] → unified_norm_out
        {
            if let Some(device) = device_residual.as_mut() {
                B::rms_norm_f32_to_activation(
                    ctx,
                    &**device,
                    &layer.input_ln_w,
                    eps,
                    norm_out,
                    m_total,
                    h,
                );
            } else {
                B::rms_norm(ctx, residual, &layer.input_ln_w, eps, norm_out, m_total, h);
            }
        };

        // 2. qkv_proj GEMM (m=M_total): unified_norm_out → unified_qkv_out
        {
            let qkv_out = self
                .scratch
                .unified_qkv_out
                .as_mut()
                .expect("unified_qkv_out missing");
            layer.qkv_proj.forward(ctx, norm_out, qkv_out, m_total);
        };

        // 3. Single-launch varlen split_qkv_norm_rope. Reads
        //    cu_seqlens_q / pos_offsets / block_tables from device
        //    buffers (graph-capturable; kernel scalars are stable across
        //    iters). Replaces the prior per-item dispatch loop (16 small
        //    launches per layer at c=16) with one launch.
        let pools = self
            .paged_pools
            .as_mut()
            .expect("unified_forward_layer requires paged_pools");
        let pool_ptr = (
            &mut pools[li].0 as *mut B::Buffer,
            &mut pools[li].1 as *mut B::Buffer,
        );
        // SAFETY: pools allocated once; not concurrently mutated.
        let (pool_k, pool_v) = unsafe { (&mut *pool_ptr.0, &mut *pool_ptr.1) };
        // Suppress unused warning until the legacy per-item path goes.
        let _ = items;
        let _ = q_lens;
        let _ = cu_seqlens_q;
        let _ = pos_offsets;
        let _ = qkv_stride;

        {
            let qkv_out = self
                .scratch
                .unified_qkv_out
                .as_ref()
                .expect("unified_qkv_out missing");
            let packed_q = self
                .scratch
                .unified_packed_q
                .as_mut()
                .expect("unified_packed_q missing");
            let cu_seqlens_buf = self
                .scratch
                .unified_cu_seqlens_q
                .as_ref()
                .expect("unified_cu_seqlens_q missing");
            let pos_offsets_buf = self
                .scratch
                .unified_pos_offsets
                .as_ref()
                .expect("unified_pos_offsets missing");
            let bt_buf = self
                .scratch
                .unified_block_tables
                .as_ref()
                .expect("unified_block_tables missing");
            B::split_qkv_norm_rope_into_paged_cache_varlen(
                ctx,
                qkv_out,
                q_norm_w,
                k_norm_w,
                rope_cos,
                rope_sin,
                packed_q,
                pool_k,
                pool_v,
                cu_seqlens_buf,
                pos_offsets_buf,
                bt_buf,
                num_seqs,
                m_total,
                nh,
                nkv,
                hd,
                eps,
                qk_mode,
                block_size,
                max_blocks_per_seq,
            )
            .expect("paged unified: split_qkv_norm_rope_into_paged_cache_varlen");
        };

        // 4. paged_varlen_attention: one call covering all M_total tokens.
        {
            let packed_q = self
                .scratch
                .unified_packed_q
                .as_ref()
                .expect("unified_packed_q missing");
            let cu_seqlens_buf = self
                .scratch
                .unified_cu_seqlens_q
                .as_ref()
                .expect("unified_cu_seqlens_q missing");
            let pos_offsets_buf = self
                .scratch
                .unified_pos_offsets
                .as_ref()
                .expect("unified_pos_offsets missing");
            let bt_buf = self
                .scratch
                .unified_block_tables
                .as_ref()
                .expect("unified_block_tables missing");
            let attn_out = self
                .scratch
                .unified_attn_out
                .as_mut()
                .expect("unified_attn_out missing");
            B::paged_varlen_attention(
                ctx,
                packed_q,
                pool_k,
                pool_v,
                attn_out,
                cu_seqlens_buf,
                pos_offsets_buf,
                bt_buf,
                num_seqs,
                m_total,
                max_kv_len,
                nh,
                nkv,
                hd,
                layer_window,
                block_size,
                max_blocks_per_seq,
            )
            .expect("paged_varlen_attention");
        };

        // 5. o_proj (M_total): attn_out → o_proj_out
        {
            let attn_out = self
                .scratch
                .unified_attn_out
                .as_ref()
                .expect("unified_attn_out missing");
            let o_proj_out = self
                .scratch
                .unified_o_proj_out
                .as_mut()
                .expect("unified_o_proj_out missing");
            layer.o_proj.forward(ctx, attn_out, o_proj_out, m_total);
        };

        // 6. residual + pre-MLP norm.
        //    Legacy Llama path: fused residual add + RMSNorm.
        //    Gemma sandwich path: RMSNorm(o_proj_out) -> F32 residual add,
        //    then RMSNorm(F32 residual) for pre-MLP input.
        {
            let o_proj_out = self
                .scratch
                .unified_o_proj_out
                .as_ref()
                .expect("unified_o_proj_out missing");
            if let Some(post_attn_w) = &layer.post_attn_ln_w {
                if let Some(device) = device_residual.as_mut() {
                    let branch = device_branch
                        .as_mut()
                        .expect("unified F32 sandwich branch scratch missing");
                    B::rms_norm_activation_add_to_f32(
                        ctx,
                        o_proj_out,
                        post_attn_w,
                        eps,
                        &mut **device,
                        &mut **branch,
                        m_total,
                        h,
                    );
                    B::rms_norm_f32_to_activation(
                        ctx,
                        &**device,
                        &layer.post_ln_w,
                        eps,
                        norm_out,
                        m_total,
                        h,
                    );
                } else {
                    let tmp = self
                        .scratch
                        .unified_sandwich_tmp
                        .as_mut()
                        .expect("unified sandwich tmp missing");
                    B::rms_norm(ctx, o_proj_out, post_attn_w, eps, tmp, m_total, h);
                    B::add_inplace(ctx, residual, tmp, m_total * h);
                    B::rms_norm(ctx, residual, &layer.post_ln_w, eps, norm_out, m_total, h);
                }
            } else {
                B::fused_add_rms_norm(
                    ctx,
                    residual,
                    o_proj_out,
                    &layer.post_ln_w,
                    eps,
                    norm_out,
                    m_total,
                    h,
                );
            }
        };

        // 7. gate_up_proj
        {
            let gate_up_out = self
                .scratch
                .unified_gate_up_out
                .as_mut()
                .expect("unified_gate_up_out missing");
            layer
                .gate_up_proj
                .forward(ctx, norm_out, gate_up_out, m_total);
        };

        // 8. Gated activation: SwiGLU for Llama, GeGLU for Gemma.
        {
            let gate_up_out = self
                .scratch
                .unified_gate_up_out
                .as_ref()
                .expect("unified_gate_up_out missing");
            let silu_out = self
                .scratch
                .unified_silu_out
                .as_mut()
                .expect("unified_silu_out missing");
            match self.cfg.activation {
                Activation::GeluTanh => {
                    B::fused_gelu_tanh_mul_split(ctx, gate_up_out, silu_out, m_total, im)
                }
                _ => B::fused_silu_mul_split(ctx, gate_up_out, silu_out, m_total, im),
            }
        };

        // 9. down_proj
        {
            let silu_out = self
                .scratch
                .unified_silu_out
                .as_ref()
                .expect("unified_silu_out missing");
            let mlp_out = self
                .scratch
                .unified_mlp_out
                .as_mut()
                .expect("unified_mlp_out missing");
            layer.down_proj.forward(ctx, silu_out, mlp_out, m_total);
        };

        // 10. final residual add. Gemma sandwich path normalizes the MLP
        //     branch before adding it to the F32 residual shadow.
        {
            let mlp_out = self
                .scratch
                .unified_mlp_out
                .as_ref()
                .expect("unified_mlp_out missing");
            if let Some(post_ffn_w) = &layer.post_ffn_ln_w {
                if let Some(device) = device_residual.as_mut() {
                    let branch = device_branch
                        .as_mut()
                        .expect("unified F32 sandwich branch scratch missing");
                    B::rms_norm_activation_add_to_f32(
                        ctx,
                        mlp_out,
                        post_ffn_w,
                        eps,
                        &mut **device,
                        &mut **branch,
                        m_total,
                        h,
                    );
                } else {
                    let tmp = self
                        .scratch
                        .unified_sandwich_tmp
                        .as_mut()
                        .expect("unified sandwich tmp missing");
                    B::rms_norm(ctx, mlp_out, post_ffn_w, eps, tmp, m_total, h);
                    B::add_inplace(ctx, residual, tmp, m_total * h);
                }
            } else {
                B::add_inplace(ctx, residual, mlp_out, m_total * h);
            }
        };
    }
    /// Batched decode: process M concurrent requests at potentially different
    /// positions in one forward pass. GEMM-heavy ops (qkv_proj, o_proj,
    /// gate_up, down) run with m=M for natural batching; rope + KV append +
    /// attention loop per-item (each has its own KV cache at a different
    /// kv_len, and potentially different pos).
    ///
    /// Returns M logit vectors in the same order as `batch`.
    pub fn decode_batch_internal(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        self.decode_batch_internal_with_full_logits(batch, false)
    }

    pub fn decode_batch_internal_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        let logits_return = if force_full_logits {
            DecodeLogitsReturn::Full
        } else {
            DecodeLogitsReturn::LegacyDefault
        };
        self.decode_batch_internal_with_logits_return(batch, logits_return)
    }

    pub fn decode_batch_internal_with_logits_policy(
        &mut self,
        batch: &[(String, u32, u32)],
        policies: &[LogitsReturnPolicy],
    ) -> Vec<Vec<f32>> {
        self.decode_batch_internal_with_logits_return(
            batch,
            Self::decode_token_logits_return(policies, batch.len()),
        )
    }

    fn decode_token_logits_return(
        policies: &[LogitsReturnPolicy],
        count: usize,
    ) -> DecodeLogitsReturn<'_> {
        if policies.len() != count {
            return DecodeLogitsReturn::Full;
        }

        let mut selected_mask: Option<&TokenSelectionMask> = None;
        let mut saw_unmasked = false;
        for policy in policies {
            match policy {
                LogitsReturnPolicy::FullLogits => {
                    return DecodeLogitsReturn::Full;
                }
                LogitsReturnPolicy::GreedyArgmax {
                    token_mask,
                    repetition_penalty,
                } => {
                    if repetition_penalty.is_some() {
                        return DecodeLogitsReturn::Full;
                    }
                    match token_mask.as_ref() {
                        Some(mask) => {
                            if saw_unmasked {
                                return DecodeLogitsReturn::Full;
                            }
                            if let Some(selected) = selected_mask {
                                if selected.fingerprint != mask.fingerprint
                                    || selected.len() != mask.len()
                                {
                                    return DecodeLogitsReturn::Full;
                                }
                            } else {
                                selected_mask = Some(mask);
                            }
                        }
                        None => {
                            if selected_mask.is_some() {
                                return DecodeLogitsReturn::Full;
                            }
                            saw_unmasked = true;
                        }
                    }
                }
            }
        }

        DecodeLogitsReturn::GreedyArgmax {
            token_mask: selected_mask,
            repetition_penalties: vec![None; count],
        }
    }

    pub(crate) fn execute_managed_fp16_tokens(
        &mut self,
        inputs: &[crate::common::llm::ModelTokenInput<'_>],
    ) -> crate::common::llm::ModelTokenBatchOutcome {
        use super::llama_family::device_execution::DeviceFrameResult;
        use crate::common::llm::{ModelTokenBatchOutcome, ModelTokenOutcome, ModelTokenWork};
        for input in inputs {
            if let Err(error) = self.validate_managed_token_work(input.cache_id, input.work) {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::NotSubmitted(error));
            }
        }
        if inputs.is_empty() {
            return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Terminal(Ok(Vec::new())));
        }
        let has_lora = inputs
            .iter()
            .any(|input| self.active_lora_adapter_for_cache(input.cache_id).is_some());
        let all_decode = inputs
            .iter()
            .all(|input| matches!(input.work, ModelTokenWork::Decode { .. }));
        let requests: Vec<_> = inputs
            .iter()
            .map(|input| {
                let target_len = match input.work {
                    ModelTokenWork::Prefill { tokens, position } => position + tokens.len(),
                    ModelTokenWork::Decode { position, .. } => position as usize + 1,
                };
                ferrum_execution::model_executor::KvSlotRequest {
                    cache_id: input.cache_id.to_owned(),
                    target_len,
                    admission_target_len: None,
                }
            })
            .collect();
        match self.prepare_managed_token_capacity(&requests) {
            ModelTokenOutcome::Terminal(Ok(())) => {}
            ModelTokenOutcome::CapacityDeferred(deferred) => {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::CapacityDeferred(
                    deferred,
                ));
            }
            ModelTokenOutcome::NotSubmitted(error) | ModelTokenOutcome::Terminal(Err(error)) => {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::NotSubmitted(error));
            }
            ModelTokenOutcome::Held(error) => {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Held(error))
            }
        }
        let policies: Vec<_> = inputs
            .iter()
            .map(|input| input.logits_policy.clone())
            .collect();
        if all_decode
            && self.supports_batched_decode
            && !has_lora
            && (inputs.len() > 1
                || !policies
                    .iter()
                    .any(LogitsReturnPolicy::requires_full_logits))
        {
            let batch: Vec<_> = inputs
                .iter()
                .map(|input| {
                    let ModelTokenWork::Decode { token, position } = input.work else {
                        unreachable!()
                    };
                    (input.cache_id.to_owned(), token, position)
                })
                .collect();
            let policy = Self::decode_token_logits_return(&policies, batch.len());
            self.decode_batch_stats
                .record_call(batch.len(), matches!(policy, DecodeLogitsReturn::Full));
            return ModelTokenBatchOutcome::Cohort(
                match self.run_device_frame(|model, frame| {
                    model.decode_batch_in_frame(frame, &batch, policy)
                }) {
                    DeviceFrameResult::Rejected(error) => ModelTokenOutcome::NotSubmitted(error),
                    DeviceFrameResult::Held(error) => ModelTokenOutcome::Held(error),
                    DeviceFrameResult::Terminal(result) => ModelTokenOutcome::Terminal(
                        result.map(|rows| rows.into_iter().map(Some).collect()),
                    ),
                },
            );
        }
        let reused_prefix = inputs
            .iter()
            .any(|input| self.reusable_prefix_for_work(input.cache_id, input.work) != 0);
        if !all_decode
            && self.supports_varlen_qkv
            && self.paged_pools.is_some()
            && !has_lora
            && !reused_prefix
        {
            let items: Vec<_> = inputs
                .iter()
                .map(|input| {
                    let (tokens, position) = match input.work {
                        ModelTokenWork::Prefill { tokens, position } => (tokens.to_vec(), position),
                        ModelTokenWork::Decode { token, position } => {
                            (vec![token], position as usize)
                        }
                    };
                    (input.cache_id.to_owned(), tokens, position, input.sample)
                })
                .collect();
            let outcome = self.run_device_frame(|model, frame| {
                model.unified_in_frame(frame, &items, Some(&policies))
            });
            if matches!(&outcome, DeviceFrameResult::Terminal(Ok(Ok(_)))) {
                for input in inputs {
                    if matches!(input.work, ModelTokenWork::Prefill { .. }) {
                        self.publish_prepared_prefix(input.cache_id, 0);
                    }
                }
            }
            return ModelTokenBatchOutcome::Cohort(match outcome {
                DeviceFrameResult::Rejected(error) => ModelTokenOutcome::NotSubmitted(error),
                DeviceFrameResult::Held(error) => ModelTokenOutcome::Held(error),
                DeviceFrameResult::Terminal(result) => {
                    ModelTokenOutcome::Terminal(result.and_then(|rows| rows))
                }
            });
        }
        self.execute_managed_serial_tokens(inputs)
    }

    fn decode_batch_internal_with_logits_return(
        &mut self,
        batch: &[(String, u32, u32)],
        logits_return: DecodeLogitsReturn<'_>,
    ) -> Vec<Vec<f32>> {
        self.decode_batch_sampling_with_logits_return(batch, logits_return)
            .into_iter()
            .map(legacy_sampling_values)
            .collect::<Result<Vec<_>>>()
            // The retiring Vec-only entry cannot represent a sampling error.
            // Keep the actual error payload for its existing catch boundary;
            // never invent a token or a sentinel logits vector.
            .unwrap_or_else(|error| std::panic::panic_any(error))
    }

    fn decode_batch_sampling_with_logits_return(
        &mut self,
        batch: &[(String, u32, u32)],
        logits_return: DecodeLogitsReturn<'_>,
    ) -> Vec<ExecutorSamplingOutput> {
        for (cache_id, token, position) in batch {
            self.validate_prepared_legacy_decode(cache_id, *token, *position)
                .unwrap_or_else(|error| std::panic::panic_any(error));
        }
        let force_full_logits = matches!(&logits_return, DecodeLogitsReturn::Full);
        let m = batch.len();
        if m == 0 {
            return Vec::new();
        }
        self.decode_batch_stats.record_call(m, force_full_logits);
        if m == 1 && !force_full_logits {
            let (cid, tok, pos) = &batch[0];
            self.decode_batch_stats.record_singleton_fast_path();
            return vec![ExecutorSamplingOutput::FullLogits(
                self.decode_internal(cid, *tok, *pos),
            )];
        }
        if !self.supports_batched_decode {
            // Some backends do not yet produce correct follow-up logits in
            // the optimized dense batched decode path under concurrent
            // serving. Preserve user-visible correctness by falling back to
            // the known-good per-item decode path until that backend's
            // batched kernels pass the dedicated multi-turn gate.
            self.decode_batch_stats.record_unsupported_fallback();
            return batch
                .iter()
                .map(|(cid, tok, pos)| {
                    ExecutorSamplingOutput::FullLogits(self.decode_internal(cid, *tok, *pos))
                })
                .collect();
        }
        if batch
            .iter()
            .any(|(cid, _, _)| self.active_lora_adapter_for_cache(cid).is_some())
        {
            self.decode_batch_stats.record_lora_fallback();
            return batch
                .iter()
                .map(|(cid, tok, pos)| {
                    ExecutorSamplingOutput::FullLogits(self.decode_internal(cid, *tok, *pos))
                })
                .collect();
        }

        self.run_device_frame(|model, frame| {
            model.decode_batch_in_frame(frame, batch, logits_return)
        })
        .into_legacy_result()
    }

    fn decode_batch_in_frame(
        &mut self,
        frame: &mut LlamaExecutionFrame<B>,
        batch: &[(String, u32, u32)],
        logits_return: DecodeLogitsReturn<'_>,
    ) -> Vec<ExecutorSamplingOutput> {
        if self.batched_graph_failed && self.batched_pointers_for.is_some() {
            // A failed capture/replay keeps its original cohort and graph
            // closure. Recovering stream completion alone does not authorize
            // another replay or the reuse of these bindings.
            B::retire_graphs_for_storage_mutation(&mut frame.context)
                .unwrap_or_else(|error| std::panic::panic_any(error));
            self.clear_graph_bookkeeping();
            self.batched_graph_failed = true;
        }
        let m = batch.len();
        // Ensure all caches exist and scratch is sized for M tokens.
        for (cid, _, _) in batch {
            self.ensure_kv_in_context(&mut frame.context, cid, None);
        }
        self.ensure_scratch_in_context(&mut frame.context, m);
        let grows_batch_logits = m > self.scratch.batch_logits_capacity;
        self.scratch
            .ensure_batch_logits(&mut frame.context, &self.cfg, m);
        if grows_batch_logits {
            // A failed retirement must retain the original graph/cache binding.
            self.clear_graph_bookkeeping();
        }
        // Phase 4b: when paged mode is on, ensure_kv has already
        // populated the batched scratch buffers (paged_batch_q etc.).
        // The forward path branches on `paged_pools.is_some()` inside
        // each layer.

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let num_layers = self.cfg.num_layers;
        let use_host_residual_shadow = self.use_host_residual_shadow();
        let use_device_residual_shadow = self.use_device_residual_shadow();
        frame
            .take_scratch(
                &mut self.scratch,
                FrameLayout::Single,
                use_device_residual_shadow,
            )
            .expect("batched decode frame scratch");
        let mut ctx = &mut frame.context;
        for (cid, _, _) in batch {
            let target_len = self.cache_len(cid).saturating_add(1);
            self.ensure_paged_kv_capacity_for_cache_id(&mut ctx, cid, target_len)
                .expect("paged KV dynamic grow");
        }

        let paged = self.paged_pools.is_some();
        let graph_enabled = !self.batched_graph_failed
            && should_use_batched_decode_graph(
                self.batched_cfg.batched_graph,
                use_host_residual_shadow,
            )
            && if paged {
                B::supports_varlen_qkv()
            } else {
                B::supports_qk_norm_rope_batched_per_item()
                    && B::supports_kv_cache_append_batched_per_cache()
                    && B::supports_flash_attention_batched_per_cache()
            };
        if graph_enabled {
            // A captured dynamic write requires the actual committed frontier,
            // not an unrelated caller position or a duplicate cache row.
            let mut unique = std::collections::HashSet::with_capacity(m);
            for (cid, _, position) in batch {
                assert!(unique.insert(cid.as_str()), "duplicate graph cache row");
                assert!(
                    self.kv_caches[cid]
                        .iter()
                        .all(|cache| cache.len == *position as usize),
                    "graph decode position must match every original KV frontier"
                );
            }
            if !paged {
                // The current contiguous kernel has one physical head stride.
                // Normalize differing prepared capacities before binding it;
                // growth uses the original checked storage-retirement boundary.
                let capacity = batch
                    .iter()
                    .flat_map(|(cid, _, _)| self.kv_caches[cid].iter())
                    .map(|cache| cache.capacity)
                    .max()
                    .expect("nonempty graph batch");
                for (cid, _, _) in batch {
                    self.ensure_paged_kv_capacity_for_cache_id(&mut ctx, cid, capacity)
                        .expect("contiguous graph cohort capacity");
                }
            }
            let same_owners = self.batched_pointers_for.as_ref().is_some_and(|owners| {
                owners.len() == m
                    && owners
                        .iter()
                        .zip(batch)
                        .all(|(owner, (cid, _, _))| owner == cid)
            });
            if !same_owners {
                B::retire_graphs_for_storage_mutation(&mut ctx)
                    .unwrap_or_else(|error| std::panic::panic_any(error));
                self.clear_graph_bookkeeping();
                self.batched_pointers_for =
                    Some(batch.iter().map(|(cid, _, _)| cid.clone()).collect());
            }
        }

        // Pre-step state (positions, kv_lens_pre, kv_lens_post). All
        // 32 layers' batched kernels read these from scratch device
        // buffers — written ONCE here, NEVER inside the layer loop, so
        // the captured graph below replays with stable buffer addresses
        // and the next step's content update reaches the kernels.
        let positions: Vec<u32> = batch.iter().map(|(_, _, p)| *p as u32).collect();
        let tokens: Vec<u32> = batch.iter().map(|(_, t, _)| *t).collect();
        let kv_pre: Vec<u32> = batch
            .iter()
            .map(|(cid, _, _)| self.kv_caches.get(cid).expect("kv_caches missing")[0].len as u32)
            .collect();
        let kv_post: Vec<u32> = kv_pre.iter().map(|&x| x + 1).collect();
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_positions, &positions);
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_kv_lens_pre, &kv_pre);
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_kv_lens_post, &kv_post);

        if paged
            && (graph_enabled
                || self
                    .cfg
                    .requires_windowed_attention(self.source_layer_range()))
        {
            self.prepare_paged_decode_metadata(&mut ctx, batch);
        }
        if graph_enabled {
            if !paged {
                // Refresh the canonical original-context tables every time:
                // intervening eager/stage work may have used the same slots.
                let mut keys = Vec::with_capacity(num_layers * m);
                let mut values = Vec::with_capacity(num_layers * m);
                for li in 0..num_layers {
                    for (cid, _, _) in batch {
                        keys.push(&self.kv_caches[cid][li].k);
                        values.push(&self.kv_caches[cid][li].v);
                    }
                }
                B::populate_batched_pointers(&mut ctx, &keys, &values, num_layers, m)
                    .unwrap_or_else(|error| std::panic::panic_any(error));
            }
            if paged
                && self
                    .cfg
                    .requires_windowed_attention(self.source_layer_range())
            {
                B::prepare_paged_varlen_attention(
                    &mut ctx,
                    m,
                    m,
                    16 * self.scratch.paged_max_blocks_per_seq,
                    self.cfg.num_heads,
                    self.cfg.head_dim,
                )
                .unwrap_or_else(|error| std::panic::panic_any(error));
            }
            B::pregrow_marlin_gather_scratch(&mut ctx, m * self.cfg.intermediate_size);
        }

        // 0. Embed all M tokens into residual [M, H]. Eager, OUTSIDE
        //    any captured graph (host tokens slice; embedding_lookup_dyn
        //    is single-item only).
        let mut residual = frame.residual.as_mut().expect("frame residual");
        let embed = self
            .embed
            .as_ref()
            .expect("decode_batch_internal called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, &tokens, &mut residual, h);
        if let Some(scale) = self.cfg.embed_scale {
            B::scale_inplace(&mut ctx, &mut residual, scale, m * h);
        }
        let mut host_residual = if use_host_residual_shadow {
            device_execution::checked_sync::<B>(&mut ctx);
            Some(B::to_vec_in_context(ctx, &residual, m * h))
        } else {
            None
        };
        let device_residual_shadow = &mut frame.residual_f32;
        if let Some(shadow) = device_residual_shadow.as_mut() {
            B::activation_to_f32_shadow(&mut ctx, &residual, shadow, m * h);
        }
        let device_branch_shadow = &mut frame.branch_f32;

        // ── Phase 4d: CUDA-graph replay path ─────────────────────────
        // Controlled by the original typed execution policy. Unsupported
        // capture errors stay on the original frame failure boundary.
        let graph_shape = graph_enabled.then(|| {
            let capacity = self.kv_caches[&batch[0].0][0].capacity;
            let layout = if paged {
                BatchedDecodeKvLayout::Paged {
                    block_size: 16,
                    max_blocks_per_sequence: self.scratch.paged_max_blocks_per_seq,
                }
            } else {
                BatchedDecodeKvLayout::Contiguous { capacity }
            };
            BatchedDecodeGraphShape {
                rows: m,
                layout,
                score_capacity: if paged {
                    capacity
                } else {
                    ferrum_kernels::backend::attention_score_capacity_bucket(
                        kv_post.iter().copied().max().unwrap() as usize,
                    )
                    .min(capacity)
                },
                device_residual_shadow: use_device_residual_shadow,
            }
        });
        let installed_key = graph_shape
            .as_ref()
            .and_then(|shape| self.batched_graph_keys_seen.get(shape))
            .copied();
        let cache_has_key = installed_key.is_some();
        let graph_key = installed_key
            .unwrap_or_else(|| next_batched_decode_graph_key(self.batched_graph_keys_seen.len()));

        let mut did_pure_replay = false;
        if graph_enabled && cache_has_key && !self.batched_graph_failed {
            // Deny further replay before any fallible preparation, launch or
            // completion observation. Only this complete invocation can
            // restore eligibility; failure preserves the cohort's custody.
            self.batched_graph_failed = true;
            // Sync stream first so embedding_lookup (just queued) plus
            // any null-stream cuMemcpyHtoD_v2's from write_u32 are all
            // settled before cuGraphLaunch.
            device_execution::checked_sync::<B>(&mut ctx);
            // SAFETY: cold construction checked projection/KV capture contracts.
            // The private model/frame owns all bindings through checked retirement;
            // storage mutation retires cached graphs before replacing dependencies.
            match unsafe { B::replay_graph(&mut ctx, graph_key) } {
                Ok(true) => {
                    did_pure_replay = true;
                }
                Ok(false) => {
                    B::retire_graphs_for_storage_mutation(&mut ctx)
                        .unwrap_or_else(|error| std::panic::panic_any(error));
                    self.clear_graph_bookkeeping();
                }
                Err(e) => {
                    self.batched_graph_failed = true;
                    std::panic::panic_any(e);
                }
            }
        }
        if !did_pure_replay {
            const BATCHED_GRAPH_WARMUP: usize = 3;
            let should_capture = graph_enabled
                && !self.batched_graph_failed
                && self.batched_graph_warmup >= BATCHED_GRAPH_WARMUP;
            if should_capture {
                self.batched_graph_failed = true;
                tracing::debug!("[batched-graph] BEGIN CAPTURE rows={m}");
                device_execution::checked_sync::<B>(&mut ctx);
                // SAFETY: cold construction checked projection/KV capture contracts.
                // The private model/frame owns all bindings through checked retirement;
                // storage mutation retires cached graphs before replacing dependencies.
                if let Err(e) = unsafe { B::begin_graph_capture(&mut ctx) } {
                    self.batched_graph_failed = true;
                    std::panic::panic_any(e);
                }
            }
            self.batched_graph_warmup += 1;

            // Eager forward (records into graph if capture is active).
            for li in 0..num_layers {
                self.forward_layer_batched_decode(
                    &mut ctx,
                    li,
                    batch,
                    &mut residual,
                    m,
                    graph_enabled,
                    host_residual.as_deref_mut(),
                    device_residual_shadow.as_mut(),
                    device_branch_shadow.as_mut(),
                );
            }

            if let Some(host) = host_residual.as_deref() {
                Self::rms_norm_host_to_activation(
                    &mut ctx,
                    host,
                    &self.final_norm_w,
                    self.cfg.rms_norm_eps,
                    &mut self.scratch.norm_out,
                    m,
                    h,
                );
            } else if let Some(device) = device_residual_shadow.as_ref() {
                B::rms_norm_f32_to_activation(
                    &mut ctx,
                    device,
                    &self.final_norm_w,
                    self.cfg.rms_norm_eps,
                    &mut self.scratch.norm_out,
                    m,
                    h,
                );
            } else {
                B::rms_norm(
                    &mut ctx,
                    &residual,
                    &self.final_norm_w,
                    self.cfg.rms_norm_eps,
                    &mut self.scratch.norm_out,
                    m,
                    h,
                );
            }

            let lm_head = self
                .lm_head
                .as_ref()
                .expect("decode_batch_internal called on backbone-only model (no lm_head)");

            {
                lm_head.forward(
                    &mut ctx,
                    &self.scratch.norm_out,
                    &mut self.scratch.batch_logits,
                    m,
                );
            }

            if should_capture {
                if !B::graph_capture_in_flight(&ctx) {
                    std::panic::panic_any(ferrum_types::FerrumError::backend(
                        "batched decode capture ended before graph installation",
                    ));
                }
                // SAFETY: cold construction checked projection/KV capture contracts.
                // The private model/frame owns all bindings through checked retirement;
                // storage mutation retires cached graphs before replacing dependencies.
                if let Err(e) = unsafe { B::end_graph_capture(&mut ctx, graph_key) } {
                    self.batched_graph_failed = true;
                    std::panic::panic_any(e);
                } else {
                    self.batched_graph_keys_seen
                        .insert(graph_shape.expect("enabled graph shape"), graph_key);
                    tracing::debug!("[batched-graph-capture] key={graph_key} rows={m} device_shadow={use_device_residual_shadow}");
                    replay_new_batched_graph::<B>(
                        &mut ctx,
                        graph_key,
                        &mut self.batched_graph_failed,
                    )
                    .unwrap_or_else(|error| std::panic::panic_any(error));
                }
            }
        }

        // Greedy returns explicit token products after the tiny D2H readback;
        // full sampling retains every logit. Only the retiring numerical API
        // wrapper above converts typed tokens to its historical vector form.
        let argmax_mode = match logits_return {
            DecodeLogitsReturn::Full => None,
            DecodeLogitsReturn::LegacyDefault if self.batched_cfg.greedy_argmax => {
                Some(ArgmaxMode::Raw)
            }
            DecodeLogitsReturn::LegacyDefault => None,
            DecodeLogitsReturn::GreedyArgmax { token_mask, .. } => {
                token_mask.map(ArgmaxMode::Masked).or(Some(ArgmaxMode::Raw))
            }
        };
        if let Some(argmax_mode) = argmax_mode {
            let tokens = match argmax_mode {
                ArgmaxMode::Raw => {
                    B::argmax_rows_f16(&mut ctx, &self.scratch.batch_logits, m, vocab)
                }
                ArgmaxMode::Masked(mask) => {
                    self.scratch.ensure_argmax_token_mask(&mut ctx, mask);
                    let mask_len = self.scratch.argmax_token_mask_len;
                    let device_mask = self
                        .scratch
                        .argmax_token_mask
                        .as_ref()
                        .expect("argmax token mask upload failed");
                    B::argmax_rows_f16_masked(
                        &mut ctx,
                        &self.scratch.batch_logits,
                        device_mask,
                        mask_len,
                        m,
                        vocab,
                    )
                }
                ArgmaxMode::SparseRepetition { .. } => unreachable!(
                    "legacy batched decode keeps repetition-penalty sampling on the host"
                ),
            };
            if let Ok(tokens) = tokens {
                if tokens.iter().all(|&token| token != u32::MAX) {
                    let quiescent = device_execution::checked_sync::<B>(&mut ctx);
                    let output = tokens
                        .into_iter()
                        .map(|token| ExecutorSamplingOutput::GreedyToken(TokenId::new(token)))
                        .collect();
                    self.bump_local_batched_decode_kv_lengths(batch);
                    if graph_enabled {
                        self.batched_graph_failed = false;
                    }
                    frame.completed = Some(quiescent);
                    return output;
                }
            }
        }

        let quiescent = device_execution::checked_sync::<B>(&mut ctx);
        let all = B::to_vec_in_context(ctx, &self.scratch.batch_logits, m * vocab);
        let output = (0..m)
            .map(|i| ExecutorSamplingOutput::FullLogits(all[i * vocab..(i + 1) * vocab].to_vec()))
            .collect();
        // Commit every host frontier only after this original execution
        // actually completed; failed capture/replay retains the old frontier.
        self.bump_local_batched_decode_kv_lengths(batch);
        if graph_enabled {
            self.batched_graph_failed = false;
        }
        frame.completed = Some(quiescent);
        output
    }
}

impl<B: MoeLlmBackend> LlamaPipelineStageBatchOps<B> for LlamaFamilyModel<B, KvFp16> {
    fn decode_stage_tokens_to_hidden_batch(
        &mut self,
        batch: &[(String, u32, u32)],
    ) -> PipelineHidden<B> {
        for (cache_id, token, position) in batch {
            self.validate_prepared_legacy_decode(cache_id, *token, *position)
                .unwrap_or_else(|error| std::panic::panic_any(error));
        }
        if batch.is_empty() {
            return PipelineHidden::host(Vec::new(), 0, self.cfg.hidden_size);
        }
        if batch.len() == 1 || !self.supports_batched_decode || self.cfg.sandwich_norms {
            let h = self.cfg.hidden_size;
            let mut hidden = Vec::with_capacity(batch.len() * h);
            for (cache_id, token, pos) in batch {
                hidden
                    .extend_from_slice(&self.decode_stage_token_to_hidden(cache_id, *token, *pos));
            }
            return PipelineHidden::host(hidden, batch.len(), h);
        }

        self.run_device_frame(|model, frame| {
            model.decode_stage_tokens_to_hidden_batch_in_frame(frame, batch)
        })
        .into_legacy_result()
    }

    fn decode_stage_hidden_from_host_batch(
        &mut self,
        batch: &[(String, u32, u32)],
        hidden: &PipelineHidden<B>,
    ) -> (PipelineHidden<B>, LlamaStageHiddenBridgeTiming) {
        for (cache_id, _, position) in batch {
            self.validate_prepared_legacy_write(cache_id, *position as usize, 1, None)
                .unwrap_or_else(|error| std::panic::panic_any(error));
        }
        if batch.is_empty() {
            return (
                PipelineHidden::host(Vec::new(), 0, self.cfg.hidden_size),
                LlamaStageHiddenBridgeTiming::default(),
            );
        }
        let h = self.cfg.hidden_size;
        let hidden_slice = hidden.host_slice();
        assert_eq!(
            hidden_slice.len(),
            batch.len() * h,
            "hidden length {} != batch * hidden_size {}",
            hidden_slice.len(),
            batch.len() * h
        );
        if batch.len() == 1 || !self.supports_batched_decode || self.cfg.sandwich_norms {
            let mut out = Vec::with_capacity(hidden_slice.len());
            let mut bridge_timing = LlamaStageHiddenBridgeTiming::default();
            for (row, (cache_id, _, pos)) in batch.iter().enumerate() {
                let start = row * h;
                let (row_hidden, row_timing) = self.decode_stage_hidden_from_host_with_timing(
                    cache_id,
                    &hidden_slice[start..start + h],
                    *pos,
                );
                out.extend_from_slice(&row_hidden);
                bridge_timing = bridge_timing.add(row_timing);
            }
            return (PipelineHidden::host(out, batch.len(), h), bridge_timing);
        }

        self.run_device_frame(|model, frame| {
            model.decode_stage_hidden_from_host_batch_in_frame(frame, batch, hidden)
        })
        .into_legacy_result()
    }

    fn sampling_from_hidden_batch(
        &mut self,
        hidden: &PipelineHidden<B>,
        force_full_logits: bool,
    ) -> Vec<ExecutorSamplingOutput> {
        let row_count = hidden.row_count();
        if row_count == 0 {
            return Vec::new();
        }
        let h = self.cfg.hidden_size;
        let hidden_slice = hidden.host_slice();
        assert_eq!(
            hidden_slice.len(),
            row_count * h,
            "hidden length {} != row_count * hidden_size {}",
            hidden_slice.len(),
            row_count * h
        );
        if row_count == 1 || !self.supports_batched_decode || self.cfg.sandwich_norms {
            return (0..row_count)
                .map(|row| {
                    let start = row * h;
                    ExecutorSamplingOutput::FullLogits(
                        self.logits_from_hidden(&hidden_slice[start..start + h]),
                    )
                })
                .collect();
        }

        self.run_device_frame(|model, frame| {
            model.sampling_from_hidden_batch_in_frame(frame, hidden, force_full_logits)
        })
        .into_legacy_result()
    }
}

impl<B: MoeLlmBackend> LlamaFamilyModel<B, KvFp16> {
    fn decode_stage_tokens_to_hidden_batch_in_frame(
        &mut self,
        frame: &mut LlamaExecutionFrame<B>,
        batch: &[(String, u32, u32)],
    ) -> PipelineHidden<B> {
        let (m, h) = self.prepare_batched_decode_stage(&mut frame.context, batch);
        frame
            .take_scratch(&mut self.scratch, FrameLayout::Single, false)
            .expect("stage scratch");
        let mut ctx = &mut frame.context;
        let tokens: Vec<u32> = batch.iter().map(|(_, token, _)| *token).collect();
        let mut residual = frame.residual.as_mut().expect("stage residual");
        let embed = self
            .embed
            .as_ref()
            .expect("decode_stage_tokens_to_hidden_batch called on stage without embedding");
        B::embedding_lookup(&mut ctx, embed, &tokens, &mut residual, h);

        for li in 0..self.local_layer_count() {
            self.forward_layer_batched_decode(
                &mut ctx,
                li,
                batch,
                &mut residual,
                m,
                false,
                None,
                None,
                None,
            );
        }
        let quiescent = device_execution::checked_sync::<B>(ctx);
        let out = B::to_vec_in_context(ctx, &residual, m * h);
        self.bump_local_batched_decode_kv_lengths(batch);
        frame.completed = Some(quiescent);
        PipelineHidden::host(out, m, h)
    }

    fn decode_stage_hidden_from_host_batch_in_frame(
        &mut self,
        frame: &mut LlamaExecutionFrame<B>,
        batch: &[(String, u32, u32)],
        hidden: &PipelineHidden<B>,
    ) -> (PipelineHidden<B>, LlamaStageHiddenBridgeTiming) {
        let hidden_slice = hidden.host_slice();
        let (m, h) = self.prepare_batched_decode_stage(&mut frame.context, batch);
        frame
            .take_scratch(&mut self.scratch, FrameLayout::Single, false)
            .expect("stage scratch");
        let mut ctx = &mut frame.context;
        let mut residual = frame.residual.as_mut().expect("stage residual");
        let bridge_t0 = std::time::Instant::now();
        let host_copy_t0 = std::time::Instant::now();
        B::write_f32_to_activation(&mut ctx, &mut residual, hidden_slice);
        let host_copy_us = elapsed_micros_u64_floor1(host_copy_t0);
        let bridge_timing = LlamaStageHiddenBridgeTiming {
            bridge_us: elapsed_micros_u64_floor1(bridge_t0),
            host_copy_us,
            device_copy_us: 0,
        };

        for li in 0..self.local_layer_count() {
            self.forward_layer_batched_decode(
                &mut ctx,
                li,
                batch,
                &mut residual,
                m,
                false,
                None,
                None,
                None,
            );
        }
        let quiescent = device_execution::checked_sync::<B>(ctx);
        let out = B::to_vec_in_context(ctx, &residual, m * h);
        self.bump_local_batched_decode_kv_lengths(batch);
        frame.completed = Some(quiescent);
        (PipelineHidden::host(out, m, h), bridge_timing)
    }

    fn sampling_from_hidden_batch_in_frame(
        &mut self,
        frame: &mut LlamaExecutionFrame<B>,
        hidden: &PipelineHidden<B>,
        force_full_logits: bool,
    ) -> Vec<ExecutorSamplingOutput> {
        let row_count = hidden.row_count();
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let hidden_slice = hidden.host_slice();
        self.ensure_scratch_in_context(&mut frame.context, row_count);
        let grows_batch_logits = row_count > self.scratch.batch_logits_capacity;
        self.scratch
            .ensure_batch_logits(&mut frame.context, &self.cfg, row_count);
        if grows_batch_logits {
            // A failed retirement must retain the original graph/cache binding.
            self.clear_graph_bookkeeping();
        }
        frame
            .take_scratch(&mut self.scratch, FrameLayout::Single, false)
            .expect("stage scratch");
        let mut ctx = &mut frame.context;
        let mut hidden_buf = frame.residual.as_mut().expect("stage residual");
        B::write_f32_to_activation(&mut ctx, &mut hidden_buf, hidden_slice);
        B::rms_norm(
            &mut ctx,
            &hidden_buf,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.norm_out,
            row_count,
            h,
        );
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("logits_from_hidden_batch called on stage without lm_head");
        {
            lm_head.forward(
                &mut ctx,
                &self.scratch.norm_out,
                &mut self.scratch.batch_logits,
                row_count,
            );
        }

        let greedy = self.batched_cfg.greedy_argmax && !force_full_logits;
        if greedy {
            let tokens = B::argmax_rows_f16(&mut ctx, &self.scratch.batch_logits, row_count, vocab)
                .expect("argmax_rows_f16");
            tokens
                .into_iter()
                .map(|t| ExecutorSamplingOutput::GreedyToken(TokenId(t)))
                .collect()
        } else {
            let quiescent = device_execution::checked_sync::<B>(ctx);
            let all = B::to_vec_in_context(ctx, &self.scratch.batch_logits, row_count * vocab);
            frame.completed = Some(quiescent);
            (0..row_count)
                .map(|row| {
                    ExecutorSamplingOutput::FullLogits(all[row * vocab..(row + 1) * vocab].to_vec())
                })
                .collect()
        }
    }
}
