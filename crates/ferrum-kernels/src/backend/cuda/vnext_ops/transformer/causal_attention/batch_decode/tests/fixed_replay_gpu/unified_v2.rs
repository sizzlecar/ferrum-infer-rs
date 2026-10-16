//! Diagnostic only: current adaptive V1/production GQA2 versus the same
//! production GQA2 ABI for every positive row. No provider/default route changes.
use super::*;
use cudarc::nvrtc::Ptx;
use serde_json::json;
use std::{ffi::c_void, time::Instant};

// Reconstructed logical order of current-workload cohort 914 (27 short/5 long).
// The original CUDA row permutation is unavailable; Q/K/V values are synthetic.
const COHORT_LENGTHS: [u64; 32] = [
    462, 497, 463, 589, 515, 451, 587, 921, 256, 593, 252, 240, 226, 314, 168, 187, 272, 128, 125,
    96, 325, 98, 89, 89, 166, 74, 61, 76, 81, 59, 35, 21,
];

// This is the existing production archive, not the separate GQA2 test archive.
// Calling the fixed_masked Rust wrapper for short V2 rows would violate that
// wrapper's stronger adaptive-path safety contract, so this fixture establishes
// the native ABI's positive-length/map/bounds contract explicitly instead.
unsafe extern "C" {
    fn ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_gqa2_indexed_v1(
        out: *mut c_void,
        exp_sums: *mut f32,
        max_logits: *mut f32,
        tmp_out: *mut c_void,
        query: *const c_void,
        num_kv_heads: i32,
        scale: f32,
        block_addresses: *const u64,
        seq_lens: *const i32,
        row_map: *const i32,
        num_seqs: i32,
        original_num_seqs: i32,
        num_heads: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        max_seq_len: i32,
        stream: *mut c_void,
    ) -> i32;
}

struct Execution<'a> {
    stream: Arc<CudaStream>,
    functions: &'a CausalAttentionFunctions,
    unified_gather: CudaFunction,
    batch: FixedBatchDecode,
    launches: Vec<CausalAttentionLaunch>,
    shape: CausalAttentionShape,
    binding: BindingLayout,
    layout: ScratchLayout,
}

impl Execution<'_> {
    fn enqueue(&self, layer: &Layer, arm: usize) {
        let s = self.shape.cuda_shape().unwrap();
        let binding = layer.arms[arm].binding_pointer(&self.stream);
        let scratch = layer.arms[arm].scratch_pointer(&self.stream);
        let query_norm = layer.query_norm.pointer(&self.stream);
        let key_norm = layer.key_norm.pointer(&self.stream);
        if arm == 0 {
            assert!(self.batch.uses_paired_query_heads());
            self.batch
                .enqueue(
                    &self.stream,
                    self.functions,
                    &self.launches,
                    binding,
                    self.binding,
                    s,
                    self.layout,
                    scratch,
                    query_norm,
                    key_norm,
                    true,
                )
                .unwrap();
            return;
        }
        let p = self.layout.vllm.unwrap();
        let rows = self.launches.len() as i32;
        let geometry =
            BatchDecode::dimensions(self.launches.len(), self.binding, 2048, 24, 256).unwrap();
        self.batch
            .enqueue_with_gathered_attention(
                &self.stream,
                self.functions,
                &self.launches,
                binding,
                self.binding,
                s,
                self.layout,
                scratch,
                query_norm,
                key_norm,
                true,
                |lengths, row_map, unused_map| {
                    let mut gather = self.stream.launch_builder(&self.unified_gather);
                    gather
                        .arg(&binding)
                        .arg(&self.binding.slot_bytes)
                        .arg(&lengths)
                        .arg(&row_map)
                        .arg(&unused_map)
                        .arg(&rows);
                    unsafe { gather.launch(LaunchConfig::for_num_elems(rows as u32)) }.map_err(
                        |e| CudaDeviceRuntimeError::driver("test unified fresh gather", e),
                    )?;
                    // Host fixture proves every positive length <=2048 and retains
                    // the complete addressed pages and M*24*4 partition scratch.
                    // Same-stream gather writes i or -1 for EVERY row on each replay.
                    // Zero rows are -1 before native main/reduce inspect any page.
                    let status = unsafe {
                        ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_gqa2_indexed_v1(
                            scratch_pointer(scratch, self.layout.context)? as *mut c_void,
                            scratch_pointer(scratch, p.exp_sums)? as *mut f32,
                            scratch_pointer(scratch, p.max_logits)? as *mut f32,
                            scratch_pointer(scratch, p.temporary_output)? as *mut c_void,
                            scratch_pointer(scratch, self.layout.query)? as *const c_void,
                            4,
                            0.0625,
                            scratch_pointer(binding, BINDING_CONTROL_BYTES)? as *const u64,
                            lengths as *const i32,
                            row_map as *const i32,
                            rows,
                            rows,
                            24,
                            geometry.table_stride,
                            24 * 256,
                            4 * 256 * 16,
                            256 * 16,
                            2048,
                            self.stream.cu_stream() as *mut c_void,
                        )
                    };
                    if status != 0 {
                        return Err(CudaDeviceRuntimeError::contract(format!(
                            "test unified production GQA2 launch returned {status}"
                        )));
                    }
                    Ok(())
                },
            )
            .unwrap();
    }

    fn capture(&self, layers: &[Layer], arm: usize) -> CudaGraph {
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        for layer in layers {
            self.enqueue(layer, arm);
        }
        self.stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .unwrap()
            .unwrap()
    }

    fn input(&self, requests: &[usize], generation: usize) -> Vec<f16> {
        let mut input = poisoned_input(self.shape, self.layout, requests, generation);
        let p = self.layout.vllm.unwrap();
        let start = GUARD_WORDS + p.exp_sums as usize / 2;
        let end = GUARD_WORDS + p.sequence_lengths as usize / 2;
        input[start..end].fill(f16::from_bits(0x7e35));
        input
    }

    fn refill(
        &self,
        layers: &mut [Layer],
        lengths: &[u64],
        requests: &[usize],
        generation: usize,
    ) -> Vec<Vec<f16>> {
        assert_eq!(lengths.len(), self.launches.len());
        assert_eq!(requests.len(), lengths.len());
        assert!(lengths.iter().all(|&length| length <= 2048));
        let positive = lengths
            .iter()
            .map(|&length| length.max(1))
            .collect::<Vec<_>>();
        layers
            .iter_mut()
            .enumerate()
            .map(|(index, layer)| {
                let input = self.input(requests, generation + index * 71);
                for arm in &mut layer.arms {
                    for (&request, &length) in requests.iter().zip(&positive) {
                        arm.requests[request].grow_to(&self.stream, self.shape, length);
                    }
                    let mut words = arm.binding_words(
                        &self.stream,
                        self.shape,
                        self.binding,
                        requests,
                        &positive,
                    );
                    for (row, &length) in lengths.iter().enumerate() {
                        if length == 0 {
                            let base = GUARD_WORDS
                                + self.binding.binding_offset(row).unwrap() as usize / 8;
                            // No prepare token, no attention length, no reachable
                            // page. Keep packed row ordinal; the gate sees zero context.
                            words[base] = 0;
                            words[base + 1] = 0;
                            words[base + 3..base + self.binding.slot_bytes as usize / 8]
                                .fill(u64::MAX);
                        }
                    }
                    self.stream.memcpy_htod(&words, &mut arm.binding).unwrap();
                    self.stream.memcpy_htod(&input, &mut arm.scratch).unwrap();
                }
                input
            })
            .collect()
    }

    fn check_scratch(&self, layer: &Layer, lengths: &[u64], input: &[f16]) -> [Vec<u16>; 2] {
        let actual = layer
            .arms
            .each_ref()
            .map(|arm| self.stream.clone_dtoh(&arm.scratch).unwrap());
        let p = self.layout.vllm.unwrap();
        for base in [self.layout.query, self.layout.context] {
            let start = GUARD_WORDS + base as usize / 2;
            let end = start + lengths.len() * self.shape.query_features as usize;
            assert_eq!(
                bits(&actual[0][start..end]),
                bits(&actual[1][start..end]),
                "complete A/B query/context bits"
            );
            assert!(actual[0][start..end].iter().all(|v| v.is_finite()));
            for (row, &length) in lengths.iter().enumerate() {
                let row_start = start + row * self.shape.query_features as usize;
                let values = &actual[0][row_start..row_start + self.shape.query_features as usize];
                if length == 0 {
                    assert!(
                        values.iter().all(|v| v.to_bits() == 0),
                        "inactive query/context changed"
                    );
                } else {
                    assert!(
                        values.iter().any(|v| *v != f16::ZERO),
                        "active output not produced"
                    );
                }
            }
        }
        for (arm, words) in actual.iter().enumerate() {
            assert_eq!(
                read_i32(words, p.sequence_lengths, lengths.len()),
                lengths.iter().map(|&v| v as i32).collect::<Vec<_>>()
            );
            for map in 0..2 {
                let expected = lengths
                    .iter()
                    .enumerate()
                    .map(|(row, &length)| {
                        let active = if arm == 1 {
                            map == 0 && length > 0
                        } else if map == 0 {
                            length > 0 && length <= 512
                        } else {
                            length > 512
                        };
                        if active {
                            row as i32
                        } else {
                            -1
                        }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    read_i32(
                        words,
                        p.row_indices + (map * lengths.len() * 4) as u64,
                        lengths.len()
                    ),
                    expected,
                    "fresh map including inactive rows"
                );
            }
            let mut written = vec![false; input.len()];
            let mut allow = |base: u64, bytes: usize| {
                let start = GUARD_WORDS + base as usize / 2;
                written[start..start + bytes / 2].fill(true);
            };
            allow(p.sequence_lengths, lengths.len() * 4);
            allow(p.row_indices, lengths.len() * 8);
            // Gate touches every dense output row, including a zero placeholder.
            allow(self.layout.context, lengths.len() * 24 * 256 * 2);
            for (row, &length) in lengths.iter().enumerate() {
                if length > 0 {
                    allow(
                        self.layout.query + (row * 24 * 256 * 2) as u64,
                        24 * 256 * 2,
                    );
                }
                if length > 0 && (arm == 1 || length > 512) {
                    for head in 0..24 {
                        for partition in 0..length.div_ceil(512) as usize {
                            let scalar = (row * 24 + head) * 4 + partition;
                            allow(p.exp_sums + (scalar * 4) as u64, 4);
                            allow(p.max_logits + (scalar * 4) as u64, 4);
                            allow(p.temporary_output + (scalar * 256 * 2) as u64, 256 * 2);
                        }
                    }
                }
            }
            for (i, (&now, &before)) in words.iter().zip(input).enumerate() {
                if !written[i] {
                    assert_eq!(
                        now.to_bits(),
                        before.to_bits(),
                        "write escaped legal scratch/input/guard at {i}, arm {arm}"
                    );
                }
            }
        }
        actual.map(|values| bits(&values))
    }

    fn check_pages(&self, layer: &mut Layer, requests: &[usize], lengths: &[u64], may_write: bool) {
        let [left, right] = &mut layer.arms;
        for (request, (a, b)) in left
            .requests
            .iter_mut()
            .zip(&mut right.requests)
            .enumerate()
        {
            let target = requests.iter().position(|&r| r == request).and_then(|row| {
                (lengths[row] > 0).then(|| writable_elements(self.shape, lengths[row] - 1))
            });
            let mut changed = false;
            for page in 0..a.pages.len() {
                let av = bits(&a.pages[page].read(&self.stream));
                let bv = bits(&b.pages[page].read(&self.stream));
                assert_eq!(av, bv, "complete addressed KV bits");
                for i in 0..av.len() {
                    let writable = may_write
                        && target
                            .as_ref()
                            .is_some_and(|(p, mask)| *p == page && mask[i]);
                    if writable {
                        changed |= av[i] != a.previous[page][i];
                    } else {
                        assert_eq!(
                            av[i], a.previous[page][i],
                            "A KV write outside current token / own replay changed"
                        );
                        assert_eq!(
                            bv[i], b.previous[page][i],
                            "B KV write outside current token / own replay changed"
                        );
                    }
                }
                a.previous[page] = av;
                b.previous[page] = bv;
            }
            if may_write && target.is_some() {
                assert!(changed, "fresh KV input was ignored");
            }
        }
        layer.query_norm.assert_unchanged(&self.stream);
        layer.key_norm.assert_unchanged(&self.stream);
    }
}

fn attention_shape() -> CausalAttentionShape {
    let mut s = shape(256);
    s.hidden_size = 5120;
    s.query_heads = 24;
    s.key_value_heads = 4;
    s.query_features = 24 * 256;
    s.query_projection_features = 2 * s.query_features;
    s.kv_features = 4 * 256;
    s.output_gate = true;
    s.maximum_context_tokens = 2048;
    s.rope_dim = 64;
    s.rope_frequency_denominator = 64;
    s.rope_pair_offset = 32;
    s.rope_interleaved = false;
    s.rope_theta = 10_000_000.0;
    s.epsilon = 1e-6;
    s
}

#[test]
#[ignore = "requires CUDA and production GQA2; complete adaptive/unified V2 diagnostic"]
fn unified_v2_complete_m32_bits_freshness_and_paired_on_cuda() {
    let cold_start = Instant::now();
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.unified-v2").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    let stream = runtime.context().new_stream().unwrap();
    let module = runtime
        .context()
        .load_module(Ptx::from_src(crate::ptx::VNEXT_CAUSAL_ATTENTION))
        .unwrap();
    let shape = attention_shape();
    let binding = BindingLayout::new(shape, 32).unwrap();
    let layout = ScratchLayout::for_participants(
        shape,
        32,
        32,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    let capacity_lengths = vec![2048; 32];
    let launches = build_launches(shape, binding, layout, &capacity_lengths);
    let batch = FixedBatchDecode::for_launches(&launches, true, binding, shape, layout, true, true)
        .unwrap()
        .unwrap()
        .with_kv_reuse(
            ferrum_types::CausalDecodeKvReuse::PairedQueryHeads,
            CausalDecodeAttentionGrouping::FixedExecutionCapacity,
            shape,
        );
    assert!(
        batch.uses_paired_query_heads(),
        "A must use current production paired-query V2"
    );
    assert!(batch.packed_work().unwrap().prepare().is_some());
    assert_eq!(batch.packed_work().unwrap().gate_tokens(), Some(32));
    let exec = Execution {
        stream,
        functions: &provider.functions,
        unified_gather: module
            .load_function("vnext_causal_gather_decode_lengths_unified_v2_test")
            .unwrap(),
        batch,
        launches,
        shape,
        binding,
        layout,
    };
    let requests = (0..32).collect::<Vec<_>>();
    let mut correctness = vec![layer(
        &exec.stream,
        shape,
        binding,
        layout,
        &COHORT_LENGTHS,
        0,
    )];
    let graphs = [exec.capture(&correctness, 0), exec.capture(&correctness, 1)];
    println!(
        "{}",
        json!({"diagnostic":"unified_v2_complete", "record":"setup", "M":32,
        "query_heads":24,"kv_heads":4,"head_dim":256,"capacity":2048,"block_tokens":16,
        "rope_dim":64,"rope_frequency_denominator":64,"rope_pair_offset":32,
        "rope_interleaved":false,"rope_theta":10000000,"epsilon":1e-6,
        "actual_derived_logical_lengths":COHORT_LENGTHS,"physical_row_order_reconstructed":true,
        "qkv_values":"synthetic", "A":"adaptive V1 + production paired-query V2 + reduce",
        "B":"fresh unified map + single production paired-query V2 + reduce",
        "includes_prepare_gather_attention_reduce_gate":true, "product_dispatch_changed":false,
        "cold_setup_ns":cold_start.elapsed().as_nanos().to_string()})
    );
    for generation in 0..4 {
        let mut current_requests = requests.clone();
        if generation == 2 {
            current_requests.reverse();
        }
        if generation == 3 {
            for arm in &mut correctness[0].arms {
                for row in 0..32 {
                    let mut request = RequestPages {
                        pages: Vec::new(),
                        previous: Vec::new(),
                        seed: 1000 + row,
                    };
                    request.grow_to(&exec.stream, shape, 1);
                    arm.requests.push(request);
                }
                arm.share_allocation_with_disjoint_first_pages(&exec.stream, 32);
            }
            current_requests = (0..32).map(|r| 32 + (r ^ 1)).collect();
        }
        let lengths = match generation {
            0 => COHORT_LENGTHS.to_vec(),
            1 => (0..32)
                .map(|r| [0, 1, 16, 511, 512, 513, 1024, 1025][r % 8])
                .collect(),
            2 => (0..32)
                .map(|r| [513, 0, 17, 512, 513, 512, 1025, 2048][r % 8])
                .collect(),
            _ => COHORT_LENGTHS.to_vec(),
        };
        let inputs = exec.refill(
            &mut correctness,
            &lengths,
            &current_requests,
            generation + 1,
        );
        let bindings = correctness[0]
            .arms
            .each_ref()
            .map(|a| exec.stream.clone_dtoh(&a.binding).unwrap());
        for arm in 0..2 {
            exec.enqueue(&correctness[0], arm);
        }
        let eager = exec.check_scratch(&correctness[0], &lengths, &inputs[0]);
        exec.check_pages(&mut correctness[0], &current_requests, &lengths, true);
        for arm in &mut correctness[0].arms {
            exec.stream
                .memcpy_htod(&inputs[0], &mut arm.scratch)
                .unwrap();
        }
        for graph in &graphs {
            graph.launch().unwrap();
        }
        let replay = exec.check_scratch(&correctness[0], &lengths, &inputs[0]);
        assert_eq!(eager, replay, "each arm eager/replay whole scratch bits");
        exec.check_pages(&mut correctness[0], &current_requests, &lengths, false);
        for (arm, expected) in correctness[0].arms.iter().zip(&bindings) {
            assert_eq!(
                exec.stream.clone_dtoh(&arm.binding).unwrap(),
                *expected,
                "binding bytes/canaries changed"
            );
        }
        println!(
            "{}",
            json!({"diagnostic":"unified_v2_complete", "record":"correctness",
            "generation":generation,"lengths":lengths,"requests":current_requests,
            "complete_query_context_kv_bits":true,"own_eager_graph_bits":true,
            "input_binding_guards":true,"fresh_device_maps":true})
        );
    }
    drop(graphs);
    drop(correctness);

    let cold_start = Instant::now();
    let mut layers = (0..16)
        .map(|i| layer(&exec.stream, shape, binding, layout, &COHORT_LENGTHS, i))
        .collect::<Vec<_>>();
    let inputs = exec.refill(&mut layers, &COHORT_LENGTHS, &requests, 61);
    let graphs = [exec.capture(&layers, 0), exec.capture(&layers, 1)];
    println!(
        "{}",
        json!({"diagnostic":"unified_v2_complete","record":"timing_setup",
        "layers":16,"cold_setup_capture_ns":cold_start.elapsed().as_nanos().to_string(),
        "lengths":COHORT_LENGTHS,"excludes_binding_upload_capture_readback":true,
        "includes_prepare_fresh_map_attention_reduce_gate":true,"warm_replays_per_arm":8})
    );
    for graph in &graphs {
        for _ in 0..8 {
            graph.launch().unwrap();
        }
    }
    exec.stream.synchronize().unwrap();
    for (layer, input) in layers.iter_mut().zip(&inputs) {
        exec.check_scratch(layer, &COHORT_LENGTHS, input);
        exec.check_pages(layer, &requests, &COHORT_LENGTHS, true);
    }
    for quad in 0..8 {
        let order = if quad % 2 == 0 {
            [0, 1, 1, 0]
        } else {
            [1, 0, 0, 1]
        };
        for (position, arm) in order.into_iter().enumerate() {
            let started = Instant::now();
            let begin = exec
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            graphs[arm].launch().unwrap();
            let end = exec
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            end.synchronize().unwrap();
            let wall_ns = started.elapsed().as_nanos();
            let gpu_ms = begin.elapsed_ms(&end).unwrap();
            println!(
                "{}",
                json!({"diagnostic":"unified_v2_complete","record":"timing",
                "quad":quad,"position":position,"arm":arm,"order":order,
                "M":32,"layers":16,"gpu_ms":gpu_ms,"submit_wait_ns":wall_ns.to_string()})
            );
        }
    }
    for (layer, input) in layers.iter_mut().zip(&inputs) {
        exec.check_scratch(layer, &COHORT_LENGTHS, input);
        exec.check_pages(layer, &requests, &COHORT_LENGTHS, false);
        for arm in &layer.arms {
            let expected =
                arm.binding_words(&exec.stream, shape, binding, &requests, &COHORT_LENGTHS);
            assert_eq!(
                exec.stream.clone_dtoh(&arm.binding).unwrap(),
                expected,
                "timed binding bytes/canaries changed"
            );
        }
    }
    drop(graphs);
    println!(
        "{}",
        json!({"diagnostic":"unified_v2_complete","record":"complete",
        "all_output_state_input_guards":true,"scope":"raw backend mechanism, not model/serving qualification"})
    );
}
