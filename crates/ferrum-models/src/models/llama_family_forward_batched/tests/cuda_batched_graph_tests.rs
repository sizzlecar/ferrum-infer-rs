//! Actual retained LlamaFamily batch graph / eager equivalence.
//! These tests do not exercise the separate typed-runtime operator graph.

use super::{physical_batch_config, LlamaFamilyModel, SyntheticLlamaLoader};
use crate::common::DecoderOnlyLLM;
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_types::LlamaExecutionConfig;

type Model = LlamaFamilyModel<CudaBackend>;

#[test]
#[ignore = "requires an actual CUDA device"]
fn unified_graph_cuda_capture_panic_retires_before_eager_recovery() {
    unified_graph_failure_recovery(true);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn unified_graph_cuda_replay_then_head_panic_keeps_original_frontier() {
    unified_graph_failure_recovery(false);
}

// Fault only a host boundary around the original CUDA head. No successful
// graph, completion status, or numerical operation is simulated by this test.
fn unified_graph_failure_recovery(during_capture: bool) {
    use cudarc::driver::DevicePtr;
    use ferrum_kernels::backend::Backend;
    use ferrum_kernels::{Linear, LinearMetadata, LoadedLinear};
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };

    struct Fault {
        armed: AtomicBool,
        fired: AtomicUsize,
        during_capture: bool,
    }
    struct Head {
        original: LoadedLinear<CudaBackend>,
        fault: Arc<Fault>,
    }
    impl Linear<CudaBackend> for Head {
        fn in_features(&self) -> usize {
            self.original.in_features()
        }
        fn out_features(&self) -> usize {
            self.original.out_features()
        }
        fn metadata(&self) -> LinearMetadata {
            self.original.metadata()
        }
        fn forward(
            &self,
            ctx: &mut <CudaBackend as Backend>::Context,
            input: &<CudaBackend as Backend>::Buffer,
            output: &mut <CudaBackend as Backend>::Buffer,
            rows: usize,
        ) {
            self.original.forward(ctx, input, output, rows);
            if CudaBackend::graph_capture_in_flight(ctx) == self.fault.during_capture
                && self.fault.armed.swap(false, Ordering::SeqCst)
            {
                self.fault.fired.fetch_add(1, Ordering::SeqCst);
                std::panic::panic_any(ferrum_types::FerrumError::backend(
                    "controlled unified head failure after original CUDA encoding",
                ));
            }
        }
    }

    let config = physical_batch_config();
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(true);
    execution.options.unified_graph = true;
    // In the second case the graph really executes before the eager head
    // fails. A successful captured head would bypass a host fault wrapper.
    execution.options.unified_graph_lm_head_eager = !during_capture;
    execution.options.cuda_graph = false;
    execution.batched_graph = false;
    execution.prefix_cache = false;
    execution.backend_context.cuda.split_k_attn = Some(false);
    execution.paged_max_seqs = 2;
    execution.kv_capacity = Some(32);
    let mut model =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.unified_graph = false;
    let mut reference = Model::new_with_execution_config(config, &loader, execution).unwrap();
    let fault = Arc::new(Fault {
        armed: AtomicBool::new(false),
        fired: AtomicUsize::new(0),
        during_capture,
    });
    let original = model.lm_head.take().unwrap();
    assert!(original.is_capture_stable());
    // SAFETY: the wrapper owns the unchanged capture-qualified projection.
    // The only injected capture-side effect is a one-shot panic, so that
    // recording cannot install a graph. Successful recordings contain exactly
    // the original CUDA operations. The second case leaves the head outside
    // capture entirely. Neither case relies on replaying a host side effect.
    model.lm_head = Some(unsafe {
        LoadedLinear::new_capture_stable(Box::new(Head {
            original,
            fault: Arc::clone(&fault),
        }))
    });

    let one = |model: &mut Model, id: &str, tokens: Vec<u32>, position| {
        let mut rows = model
            .unified_forward(&[(id.to_owned(), tokens, position, true)])
            .unwrap();
        assert_eq!(rows.len(), 1);
        rows.remove(0).expect("full logits for final unified row")
    };
    let mut outputs = Vec::new();
    let id = "unified-failed-original";
    let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    outputs.push((
        one(&mut model, id, prompt.clone(), 0),
        one(&mut reference, id, prompt, 0),
    ));
    for position in 9..11 {
        outputs.push((
            one(&mut model, id, vec![position as u32 + 1], position),
            one(&mut reference, id, vec![position as u32 + 1], position),
        ));
    }
    // First three public calls warm up. Length 12 captures bucket 16, and
    // length 13 uses that same native graph in the second arm.
    if !during_capture {
        outputs.push((
            one(&mut model, id, vec![12], 11),
            one(&mut reference, id, vec![12], 11),
        ));
        assert!(!model.unified_graph_keys_seen.is_empty());
    }
    let prior_len = model.cache_len(id);
    let prior_warmup = model.unified_graph_warmup;
    let pointer = |model: &Model| {
        let buffer = model.scratch.unified_residual.as_ref().unwrap().as_f16();
        let (address, _record) = buffer.device_ptr(buffer.stream());
        address
    };
    let prior_scratch = pointer(&model);
    let stream = Arc::clone(
        model
            .scratch
            .unified_residual
            .as_ref()
            .unwrap()
            .as_f16()
            .stream(),
    );
    fault.armed.store(true, Ordering::SeqCst);
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        one(&mut model, id, vec![prior_len as u32 + 1], prior_len)
    }));
    // Use the existing recovery boundary; it only observes the original
    // context and must not replay the failed numerical invocation.
    assert!(model.recover_device_frame().unwrap());
    let failed_latch = model.unified_graph_failed;
    let failed_len = model.cache_len(id);
    let failed_warmup = model.unified_graph_warmup;
    let failed_keys = model.unified_graph_keys_seen.len();
    let recovered_scratch = pointer(&model);
    let same_stream = Arc::ptr_eq(
        &stream,
        model
            .scratch
            .unified_residual
            .as_ref()
            .unwrap()
            .as_f16()
            .stream(),
    );
    drop(stream);

    // A different live cache must remain usable. Five real calls exceed the
    // warmup threshold: a cleared failure latch would capture again here.
    let peer = "unified-healthy-peer";
    let prompt = vec![21, 22, 23, 24, 25, 26, 27, 28, 29];
    outputs.push((
        one(&mut model, peer, prompt.clone(), 0),
        one(&mut reference, peer, prompt, 0),
    ));
    for position in 9..13 {
        outputs.push((
            one(&mut model, peer, vec![position as u32 + 21], position),
            one(&mut reference, peer, vec![position as u32 + 21], position),
        ));
    }
    let still_disabled = model.unified_graph_failed;
    let retired_keys = model.unified_graph_keys_seen.is_empty();
    let original_len_after_peer = model.cache_len(id);
    let peer_len = model.cache_len(peer);
    for owned in [&mut model, &mut reference] {
        owned.release(id);
        owned.release(peer);
        assert!(owned.kv_caches.is_empty());
        owned.close().unwrap();
    }

    let failure = failed.expect_err("the armed original CUDA head must fail");
    assert!(failure
        .downcast_ref::<ferrum_types::FerrumError>()
        .is_some_and(|error| error
            .to_string()
            .contains("controlled unified head failure")));
    assert_eq!(fault.fired.load(Ordering::SeqCst), 1);
    assert!(failed_latch && still_disabled && retired_keys);
    assert_eq!(
        (failed_len, original_len_after_peer),
        (prior_len, prior_len)
    );
    assert_eq!(peer_len, 13);
    assert_eq!(prior_scratch, recovered_scratch);
    assert!(same_stream);
    if during_capture {
        assert_eq!(failed_keys, 0, "aborted capture cannot publish a graph");
    } else {
        assert!(
            failed_keys > 0,
            "the original installed graph remains owned"
        );
        assert_eq!(
            failed_warmup, prior_warmup,
            "failure followed a real replay"
        );
    }
    for (index, (actual, expected)) in outputs.into_iter().enumerate() {
        full_logits(
            &actual,
            &expected,
            &format!("unified failure recovery row {index}"),
        );
    }
}

/// A new unified graph shape must prepare its actual split-K workspace before
/// capture. The native split count changes from two to four at length 385.
#[test]
#[ignore = "requires an actual CUDA device"]
fn unified_graph_cuda_prepares_split_growth_before_capture() {
    use crate::test_support::numerical::assert_full_logits_qualification;
    use ferrum_kernels::backend::cpu::CpuBackend;

    let mut config = physical_batch_config();
    config.max_seq_len = 512;
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(true);
    execution.options.unified_graph = true;
    execution.options.cuda_graph = false;
    execution.batched_graph = false;
    execution.prefix_cache = false;
    execution.paged_max_seqs = 2;
    execution.kv_capacity = Some(512);
    assert_eq!(execution.options.split_k_attn, None);
    assert_eq!(execution.backend_context.cuda.split_k_attn, None);
    let mut graph =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.unified_graph = false;
    let mut eager =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.metal_paged_kv = Some(false);
    let mut cpu =
        LlamaFamilyModel::<CpuBackend>::new_with_execution_config(config, &loader, execution)
            .unwrap();

    let id = "unified-split-threshold";
    let prompt: Vec<u32> = (0..381).map(|index| 1 + (index % 97) as u32).collect();
    let items = vec![(id.to_owned(), prompt.clone(), 0, true)];
    let one_row = |model: &mut Model, items: &[(String, Vec<u32>, usize, bool)]| {
        let mut rows = model.unified_forward(items).unwrap();
        assert_eq!(rows.len(), 1);
        rows.remove(0)
            .expect("final unified item must return full logits")
    };
    let mut records = vec![(
        381,
        one_row(&mut graph, &items),
        one_row(&mut eager, &items),
        cpu.prefill_internal(id, &prompt),
    )];
    let mut graph_counts = Vec::new();
    let mut lengths = Vec::new();
    // Prefill plus lengths 382/383 are the real eager warmups. Length 384
    // captures two splits; 385 needs larger scratch and a new four-split
    // capture. Length 386 reuses that new shape's exact chunk size.
    for position in 381..386 {
        let token = 1 + (position % 97) as u32;
        let items = vec![(id.to_owned(), vec![token], position, true)];
        records.push((
            position + 1,
            one_row(&mut graph, &items),
            one_row(&mut eager, &items),
            cpu.decode_internal(id, token, position as u32),
        ));
        graph_counts.push((
            position + 1,
            graph.unified_graph_keys_seen.len(),
            graph.unified_graph_warmup,
        ));
        lengths.push((
            position + 1,
            graph.cache_len(id),
            eager.cache_len(id),
            cpu.cache_len(id),
        ));
    }
    let graph_failed = graph.unified_graph_failed;
    let eager_captured = !eager.unified_graph_keys_seen.is_empty();
    let graph_was_paged = graph.paged_pools.is_some();
    let cpu_was_contiguous = cpu.paged_pools.is_none();
    graph.close().unwrap();
    eager.close().unwrap();
    cpu.close().unwrap();

    assert!(graph_was_paged && cpu_was_contiguous);
    assert!(!graph_failed && !eager_captured);
    let count_at = |length| {
        graph_counts
            .iter()
            .find(|entry| entry.0 == length)
            .unwrap()
            .1
    };
    assert!(
        count_at(384) > 0,
        "the old split shape must actually capture"
    );
    assert!(
        count_at(385) > count_at(384),
        "the new split shape must capture"
    );
    assert_eq!(
        count_at(386),
        count_at(385),
        "same chunk shape must reuse its graph"
    );
    let warmup_at = |length| {
        graph_counts
            .iter()
            .find(|entry| entry.0 == length)
            .unwrap()
            .2
    };
    assert_eq!(
        warmup_at(386),
        warmup_at(385),
        "the original native graph must replay"
    );
    for (expected, graph_len, eager_len, cpu_len) in lengths {
        assert_eq!(
            (graph_len, eager_len, cpu_len),
            (expected, expected, expected)
        );
    }
    for (length, actual, expected, cpu) in records {
        full_logits(&actual, &expected, &format!("unified KV length {length}"));
        assert_eq!(cpu.len(), 128);
        for (route, logits) in [("eager", &expected), ("graph", &actual)] {
            // Only annotate an assertion after every device owner has closed;
            // do not catch or downgrade an execution/retirement failure.
            if let Err(failure) = std::panic::catch_unwind(|| {
                assert_full_logits_qualification(logits, &cpu);
            }) {
                eprintln!("unified split CPU oracle failed: length={length}, route={route}");
                std::panic::resume_unwind(failure);
            }
        }
    }
}

/// Independent window oracle. Record the real executions first so the original
/// models are checked-closed even when the pre-fix numerical comparison fails.
#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_paged_layer_window_matches_contiguous_and_cpu() {
    use crate::models::llama_family::llama_layer_attention_schedule;
    use crate::test_support::numerical::assert_full_logits_qualification;
    use ferrum_kernels::backend::cpu::CpuBackend;

    let mut config = physical_batch_config();
    config.num_layers = 2;
    config.sliding_window = 2;
    config.sliding_window_pattern = 2;
    config.rope_local_theta = Some(100.0);
    config.max_seq_len = 64;
    assert_eq!(llama_layer_attention_schedule(&config, 0).sliding_window, 2);
    assert_eq!(llama_layer_attention_schedule(&config, 1).sliding_window, 0);
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(true);
    execution.options.cuda_graph = false;
    execution.options.unified_graph = false;
    execution.prefix_cache = false;
    execution.batched_graph = true;
    execution.paged_max_seqs = 4;
    execution.kv_capacity = Some(64);
    let mut graph =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.batched_graph = false;
    let mut eager =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.metal_paged_kv = Some(false);
    let mut contiguous =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    let mut cpu =
        LlamaFamilyModel::<CpuBackend>::new_with_execution_config(config, &loader, execution)
            .unwrap();

    assert!(graph.supports_batched_decode && eager.supports_batched_decode);
    assert!(graph.rope_local.is_some() && eager.rope_local.is_some());

    // Each record contains every vocabulary element from four independently
    // constructed models. No reference is seeded with a paged cache or output.
    let mut records = Vec::new();
    for (id, prompt) in [("window-a", vec![1]), ("window-b", vec![21, 22])] {
        records.push((
            format!("prefill {id}"),
            graph.prefill_internal(id, &prompt),
            eager.prefill_internal(id, &prompt),
            contiguous.prefill_internal(id, &prompt),
            cpu.prefill_internal(id, &prompt),
        ));
    }
    assert!(graph.paged_pools.is_some() && eager.paged_pools.is_some());
    assert!(contiguous.paged_pools.is_none() && cpu.paged_pools.is_none());
    // Different absolute positions expose stale per-row metadata. Both rows
    // cross the local window and a physical page while the global layer stays
    // full causal. The first prefills were <=W, so they cannot seed a false
    // common paged-prefill reference.
    for step in 0..17_u32 {
        let batch = vec![
            ("window-a".to_owned(), 40 + step, 1 + step),
            ("window-b".to_owned(), 80 + step, 2 + step),
        ];
        let g = graph.decode_batch_internal_with_full_logits(&batch, true);
        let e = eager.decode_batch_internal_with_full_logits(&batch, true);
        let c = contiguous.decode_batch_internal_with_full_logits(&batch, true);
        assert_eq!(g.len(), 2);
        assert_eq!(e.len(), 2);
        assert_eq!(c.len(), 2);
        for (row, (id, token, position)) in batch.iter().enumerate() {
            let expected = cpu.decode_internal(id, *token, *position);
            records.push((
                format!("decode step={step} row={row} position={position}"),
                g[row].clone(),
                e[row].clone(),
                c[row].clone(),
                expected,
            ));
            for model in [&graph, &eager, &contiguous] {
                assert!(model.kv_caches[id]
                    .iter()
                    .all(|cache| cache.len == *position as usize + 1));
            }
            assert_eq!(cpu.cache_len(id), *position as usize + 1);
        }
    }
    assert!(!graph.batched_graph_keys_seen.is_empty());
    assert!(!graph.batched_graph_failed);
    assert!(eager.batched_graph_keys_seen.is_empty());
    for id in ["window-a", "window-b"] {
        assert!(graph.kv_caches[id]
            .iter()
            .all(|cache| cache.paged_block_indices.len() >= 2));
    }

    // The direct single path must also honor W for a prefill longer than W
    // and a subsequent decode; unified/batch correctness cannot cover it.
    let id = "window-single";
    let prompt = [101, 102, 103, 104, 105];
    records.push((
        "single prefill beyond window".to_owned(),
        graph.prefill_internal(id, &prompt),
        eager.prefill_internal(id, &prompt),
        contiguous.prefill_internal(id, &prompt),
        cpu.prefill_internal(id, &prompt),
    ));
    records.push((
        "single decode beyond window".to_owned(),
        graph.decode_internal(id, 106, 5),
        eager.decode_internal(id, 106, 5),
        contiguous.decode_internal(id, 106, 5),
        cpu.decode_internal(id, 106, 5),
    ));
    for model in [&graph, &eager, &contiguous] {
        assert_eq!(model.cache_len(id), 6);
    }
    assert_eq!(cpu.cache_len(id), 6);
    graph.close().unwrap();
    eager.close().unwrap();
    contiguous.close().unwrap();
    cpu.close().unwrap();

    // Preserve both existing contracts: strict graph/eager same-shape equality
    // and the shared CPU full-vocabulary qualification envelope. A correct
    // contiguous CUDA reference is independently checked against CPU as well.
    for (stage, graph, eager, contiguous, cpu) in records {
        full_logits(&graph, &eager, &stage);
        assert_eq!(contiguous.len(), 128, "{stage}");
        assert_eq!(cpu.len(), 128, "{stage}");
        // Annotate only a numerical assertion failure, then preserve its panic.
        // Actual device work and checked close above are never caught here.
        for (route, actual, expected) in [
            ("contiguous/CPU", &contiguous, &cpu),
            ("graph/CPU", &graph, &cpu),
            ("eager/CPU", &eager, &cpu),
            ("graph/contiguous", &graph, &contiguous),
        ] {
            if let Err(error) = std::panic::catch_unwind(|| {
                assert_full_logits_qualification(actual, expected);
            }) {
                eprintln!("paged layer-window oracle failed at {stage}: {route}");
                std::panic::resume_unwind(error);
            }
        }
    }
}

fn models() -> (Model, Model) {
    // Preserve the original counterexamples' fixed window and capacity.
    configured_models(false, 2, 1, 32)
}

fn configured_models(
    paged: bool,
    window: usize,
    layers: usize,
    max_seq_len: usize,
) -> (Model, Model) {
    let mut config = physical_batch_config();
    config.sliding_window = window;
    config.num_layers = layers;
    config.max_seq_len = max_seq_len;
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(paged);
    execution.paged_max_seqs = 4;
    execution.kv_capacity = Some(max_seq_len);
    execution.options.cuda_graph = false;
    execution.options.unified_graph = false;
    execution.batched_graph = true;
    let actual =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.batched_graph = false;
    let expected = Model::new_with_execution_config(config, &loader, execution).unwrap();
    for model in [&actual, &expected] {
        assert!(model.supports_batched_decode);
        assert!(!model.cfg.sandwich_norms);
    }
    (actual, expected)
}

fn full_logits(actual: &[f32], expected: &[f32], stage: &str) {
    assert_eq!(actual.len(), 128);
    assert_eq!(expected.len(), actual.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "{stage}: logit {index}: {a} vs {e}"
        );
        assert!(
            (a - e).abs() <= 1e-5 + 1e-5 * e.abs(),
            "{stage}: logit {index}: {a} vs {e}"
        );
    }
}

fn prefill(actual: &mut Model, expected: &mut Model, id: &str, prompt: &[u32]) {
    full_logits(
        &actual.prefill_internal(id, prompt),
        &expected.prefill_internal(id, prompt),
        &format!("prefill {id} prompt={prompt:?}"),
    );
    assert_eq!(actual.paged_pools.is_some(), expected.paged_pools.is_some());
}

fn decode(actual: &mut Model, expected: &mut Model, batch: &[(String, u32, u32)]) {
    let actual_rows = actual.decode_batch_internal_with_full_logits(batch, true);
    let expected_rows = expected.decode_batch_internal_with_full_logits(batch, true);
    assert_eq!(actual_rows.len(), batch.len());
    assert_eq!(expected_rows.len(), batch.len());
    for (row, (actual_row, expected_row)) in actual_rows.iter().zip(&expected_rows).enumerate() {
        full_logits(
            actual_row,
            expected_row,
            &format!("decode row={row} batch={batch:?}"),
        );
    }
    for (id, _, position) in batch {
        assert!(actual.kv_caches[id]
            .iter()
            .all(|cache| cache.len == *position as usize + 1));
    }
}

fn batch(ids: &[&str], position: u32, token: u32) -> Vec<(String, u32, u32)> {
    ids.iter()
        .enumerate()
        .map(|(index, id)| ((*id).to_owned(), token + index as u32, position))
        .collect()
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_same_shape_switches_original_cache_bindings() {
    let (mut actual, mut expected) = models();
    for (id, prompt) in [
        ("a0", [1, 2, 3]),
        ("a1", [11, 12, 13]),
        ("b0", [31, 32, 33]),
        ("b1", [41, 42, 43]),
    ] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..7 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1"], position, position + 50),
        );
    }
    // Set membership is published only after native end_graph_capture succeeds;
    // every capture-step output above also had to match the eager full logits.
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    decode(&mut actual, &mut expected, &batch(&["b0", "b1"], 3, 80));
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_three_then_four_rows_preserves_every_logit() {
    let (mut actual, mut expected) = models();
    // Pre-size both models before capture so m=4 does not accidentally retire
    // the m=3 graph through an unrelated scratch/logit allocation boundary.
    for model in [&mut actual, &mut expected] {
        model.ensure_scratch(4);
        model
            .run_preparation_frame(|model, frame| {
                model
                    .scratch
                    .ensure_batch_logits(&mut frame.context, &model.cfg, 4);
            })
            .into_legacy_result();
    }
    for (id, prompt) in [
        ("a0", [1, 2, 3]),
        ("a1", [11, 12, 13]),
        ("a2", [21, 22, 23]),
        ("a3", [31, 32, 33]),
    ] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..7 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1", "a2"], position, position + 50),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    let mut four = batch(&["a0", "a1", "a2"], 7, 80);
    four.push(("a3".to_owned(), 90, 3));
    decode(&mut actual, &mut expected, &four);
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_two_layers_grow_attention_score_capacity() {
    // Different synthetic layer weights and original per-layer caches exercise
    // every independent pointer-table slot. Full attention crosses two graph
    // score-capacity boundaries while physical storage remains unchanged.
    let (mut actual, mut expected) = configured_models(false, 0, 2, 64);
    for (id, prompt) in [("a0", [1, 2, 3]), ("a1", [11, 12, 13])] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..20 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1"], position, 40 + position),
        );
    }
    // Each entry is installed only after native capture succeeded. Full
    // outputs above cover capture, same-key replay, and re-capture at growth.
    assert!(actual.batched_graph_keys_seen.len() >= 3);
    assert!(!actual.batched_graph_failed);
    for id in ["a0", "a1"] {
        assert!(actual.kv_caches[id].iter().all(|cache| cache.len == 20));
    }
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_paged_replays_new_pages_and_cache_incarnations() {
    let (mut actual, mut expected) = configured_models(true, 0, 2, 64);
    for (id, prompt) in [
        ("a0", [1, 2, 3]),
        ("a1", [11, 12, 13]),
        ("b0", [21, 22, 23]),
        ("b1", [31, 32, 33]),
    ] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    assert!(actual.paged_pools.is_some());
    for position in 3..19 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1"], position, 40 + position),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    for id in ["a0", "a1"] {
        assert!(actual.kv_caches[id]
            .iter()
            .all(|cache| cache.len == 19 && cache.paged_block_indices.len() >= 2));
    }
    // A different live cohort must capture its original owners, then replay.
    for position in 3..8 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["b0", "b1"], position, 70 + position),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    for id in ["b0", "b1"] {
        actual.release(id);
        expected.release(id);
        assert!(actual.batched_graph_keys_seen.is_empty());
        assert!(!actual.kv_caches.contains_key(id));
    }
    // Reuse the same public IDs with new prompts and physical block owners.
    for (id, prompt) in [("b0", [91, 92, 93]), ("b1", [101, 102, 103])] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..8 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["b0", "b1"], position, 110 + position),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_paged_uses_alternating_layer_rope() {
    use crate::models::llama_family::llama_layer_attention_schedule;
    use crate::test_support::numerical::assert_full_logits_qualification;
    use ferrum_kernels::backend::cpu::CpuBackend;

    let mut config = physical_batch_config();
    config.num_layers = 2;
    config.sliding_window = 32;
    config.sliding_window_pattern = 2;
    config.rope_local_theta = Some(100.0);
    assert!(!llama_layer_attention_schedule(&config, 0).is_global_layer);
    assert!(llama_layer_attention_schedule(&config, 1).is_global_layer);
    // This isolates the actual local/global RoPE inputs. Every context remains
    // below the local window; it does not qualify paged window truncation or
    // the complete sandwich-norm Gemma architecture.
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(true);
    execution.options.cuda_graph = false;
    execution.options.unified_graph = false;
    execution.batched_graph = true;
    execution.paged_max_seqs = 2;
    let mut actual =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.batched_graph = false;
    let mut expected =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.metal_paged_kv = Some(false);
    let mut serial = LlamaFamilyModel::<CpuBackend>::new_with_execution_config(
        config.clone(),
        &loader,
        execution,
    )
    .unwrap();
    assert!(actual.rope_local.is_some());
    assert!(expected.rope_local.is_some());
    for (id, prompt) in [("local-a", [1, 2, 3]), ("local-b", [21, 22, 23])] {
        let a = actual.prefill_internal(id, &prompt);
        let e = expected.prefill_internal(id, &prompt);
        let oracle = serial.prefill_internal(id, &prompt);
        full_logits(&a, &e, &format!("layer-rope prefill {id}"));
        assert_eq!(oracle.len(), 128);
        assert_full_logits_qualification(&a, &oracle);
        assert_full_logits_qualification(&e, &oracle);
    }
    assert!(actual.paged_pools.is_some());
    assert!(serial.paged_pools.is_none());
    for position in 3..10 {
        let batch = batch(&["local-a", "local-b"], position, 40 + position);
        let actual_rows = actual.decode_batch_internal_with_full_logits(&batch, true);
        let expected_rows = expected.decode_batch_internal_with_full_logits(&batch, true);
        assert_eq!(actual_rows.len(), batch.len());
        assert_eq!(expected_rows.len(), batch.len());
        for (row, (id, token, pos)) in batch.iter().enumerate() {
            let oracle = serial.decode_internal(id, *token, *pos);
            full_logits(
                &actual_rows[row],
                &expected_rows[row],
                &format!("layer-rope decode row={row} batch={batch:?}"),
            );
            assert_eq!(oracle.len(), 128);
            // Reuse the existing independent CPU/full-vocabulary envelope;
            // graph/eager equivalence above keeps its original strict bound.
            assert_full_logits_qualification(&actual_rows[row], &oracle);
            assert_full_logits_qualification(&expected_rows[row], &oracle);
            assert!(actual.kv_caches[id]
                .iter()
                .all(|cache| cache.len == *pos as usize + 1));
            assert_eq!(serial.cache_len(id), *pos as usize + 1);
        }
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    actual.close().unwrap();
    expected.close().unwrap();
    serial.close().unwrap();
}

/// Backend-only typed policy must control both the actual route and graph
/// equality. M=5 is auto single-pass, but canonical Some(true) forces split-K.
#[test]
#[ignore = "requires an actual CUDA device"]
fn unified_graph_cuda_uses_canonical_split_policy_for_capture_identity() {
    use crate::test_support::numerical::assert_full_logits_qualification;
    use ferrum_kernels::backend::cpu::CpuBackend;
    let config = physical_batch_config();
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(true);
    execution.options.unified_graph = true;
    execution.backend_context.cuda.split_k_attn = Some(true);
    execution.paged_max_seqs = 5;
    execution.kv_capacity = Some(32);
    assert_eq!(execution.options.split_k_attn, None);
    let mut graph =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.unified_graph = false;
    let mut eager =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.metal_paged_kv = Some(false);
    let mut cpu =
        LlamaFamilyModel::<CpuBackend>::new_with_execution_config(config, &loader, execution)
            .unwrap();
    let ids: Vec<_> = (0..5).map(|row| format!("canonical-split-{row}")).collect();
    let mut records = Vec::new();
    let mut counts = Vec::new();
    for position in 0..7_usize {
        let items: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(row, id)| {
                (
                    id.clone(),
                    vec![(1 + row * 11 + position) as u32],
                    position,
                    true,
                )
            })
            .collect();
        let actual = graph.unified_forward(&items).unwrap();
        let expected = eager.unified_forward(&items).unwrap();
        assert_eq!(actual.len(), 5);
        assert_eq!(expected.len(), 5);
        for (row, (id, tokens, _, _)) in items.iter().enumerate() {
            let reference = if position == 0 {
                cpu.prefill_internal(id, tokens)
            } else {
                cpu.decode_internal(id, tokens[0], position as u32)
            };
            records.push((
                position + 1,
                row,
                actual[row].clone().unwrap(),
                expected[row].clone().unwrap(),
                reference,
            ));
        }
        counts.push((
            position + 1,
            graph.unified_graph_keys_seen.len(),
            graph.unified_graph_warmup,
        ));
    }
    let graph_failed = graph.unified_graph_failed;
    let eager_captured = !eager.unified_graph_keys_seen.is_empty();
    let lengths: Vec<_> = ids
        .iter()
        .map(|id| (graph.cache_len(id), eager.cache_len(id), cpu.cache_len(id)))
        .collect();
    graph.close().unwrap();
    eager.close().unwrap();
    cpu.close().unwrap();
    assert!(!graph_failed && !eager_captured);
    for lengths in lengths {
        assert_eq!(lengths, (7, 7, 7));
    }
    let at = |length| counts.iter().find(|entry| entry.0 == length).unwrap();
    assert!(at(4).1 > 0);
    assert!(at(5).1 > at(4).1, "chunk 3 must capture after chunk 2");
    assert_eq!(
        (at(6).1, at(6).2),
        (at(5).1, at(5).2),
        "same split chunk must actually replay"
    );
    assert!(
        at(7).1 > at(6).1,
        "canonical split chunk 4 must not reuse the chunk 3 graph"
    );
    for (length, row, actual, expected, reference) in records {
        full_logits(
            &actual,
            &expected,
            &format!("canonical split length={length} row={row}"),
        );
        assert_full_logits_qualification(&expected, &reference);
        assert_full_logits_qualification(&actual, &reference);
    }
}
