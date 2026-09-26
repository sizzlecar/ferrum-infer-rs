//! Test-only paired timing of the complete prepare/attention/gate batch.
//! Projection GEMMs, allocation, uploads, capture and readback are outside this
//! boundary. This is not whole-model or whole causal-operation performance.
use super::*;
use crate::backend::cuda::vnext_ops::{
    cuda_vnext_runtime_config, transformer::test_support::Guarded,
};
use cudarc::driver::{
    sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode},
    CudaGraph,
};
use ferrum_interfaces::vnext::{CanonicalRational, DeviceId};
use half::f16;
use std::time::Instant;

const CAPACITY: u64 = 2048;
const REPLAYS: u32 = 64;
const PAIRS: usize = 7;

fn qwen9b_shape() -> CausalAttentionShape {
    // Qwen3.5-9B source: Q16/KV4/H256, hidden4096, output gate,
    // theta1e7 and rotary factor .25. The family text-RoPE getter returns
    // false when mrope_section=[11,11,10] exists, despite the source's
    // mrope_interleaved=true; use the actual program semantics here.
    let mut attributes = super::super::super::tests::attributes(true);
    for (name, value) in [
        ("hidden_size", 4096),
        ("query_heads", 16),
        ("key_value_heads", 4),
        ("head_dim", 256),
        ("query_features", 4096),
        ("query_projection_features", 8192),
        ("kv_features", 1024),
        ("rope_dim", 64),
        ("maximum_context_tokens", CAPACITY),
    ] {
        attributes.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Unsigned(value),
        );
    }
    attributes.insert(
        AttributeId::new("rope_theta").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(10_000_000, 1).unwrap()),
    );
    attributes.insert(
        AttributeId::new("rope_interleaved").unwrap(),
        SemanticValue::Bool(false),
    );
    CausalAttentionShape::from_attributes(&attributes).unwrap()
}

fn launches(
    shape: CausalAttentionShape,
    layout: ScratchLayout,
    binding: BindingLayout,
    lengths: &[u64],
) -> Vec<CausalAttentionLaunch> {
    lengths
        .iter()
        .enumerate()
        .map(|(index, &length)| {
            let path = CausalAttentionKernelPath::select(
                AttentionExecutionPolicy::NativeAdaptive,
                shape,
                1,
                length,
            )
            .unwrap();
            let offset = |base, width| layout.token_offset(base, index as u64, width).unwrap();
            CausalAttentionLaunch {
                input_region: 0,
                output_region: 1,
                binding_offset: binding.binding_offset(index).unwrap(),
                packed_token_start: index as u64,
                packed_query_raw: offset(layout.query_raw, shape.query_projection_features),
                packed_key_raw: offset(layout.key_raw, shape.kv_features),
                packed_value_raw: offset(layout.value_raw, shape.kv_features),
                packed_query: offset(layout.query, shape.query_features),
                packed_context: offset(layout.context, shape.query_features),
                tokens: 1,
                tokens_i32: 1,
                sequence_tokens: length,
                sequence_tokens_i32: length as i32,
                table_entries_i32: shape.table_entries(length).unwrap() as i32,
                replay_topology: CausalAttentionReplayTopology::new(shape, path, length).unwrap(),
                path,
            }
        })
        .collect()
}

fn representative_lengths(owners: usize, base: u64) -> Vec<u64> {
    // All owners remain within the selected production partition envelope.
    // The two cases sample the observed 618-prompt to ~1089-token range.
    (0..owners)
        .map(|owner| base + (owner % 4) as u64 * 3)
        .collect()
}

#[test]
fn causal_batch_timing_geometry_has_equal_real_envelopes_and_distinct_tail_pages() {
    let shape = qwen9b_shape();
    assert_eq!(
        (shape.query_heads, shape.key_value_heads, shape.head_dim),
        (16, 4, 256)
    );
    assert!(shape.output_gate);
    assert!(!shape.rope_interleaved);
    assert_eq!((shape.rope_dim, shape.rope_pair_offset), (64, 32));
    for owners in [4, 8] {
        let binding = BindingLayout::new(shape, owners).unwrap();
        let layout = ScratchLayout::for_participants(
            shape,
            owners as u64,
            owners,
            CausalProjection::F16,
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap();
        for base in [619, 1080] {
            let rows = launches(
                shape,
                layout,
                binding,
                &representative_lengths(owners, base),
            );
            let batch = BatchDecode::for_launches(&rows, true, binding, shape, layout)
                .unwrap()
                .unwrap();
            for row in &rows {
                assert_eq!(
                    row.replay_topology.envelope().sequence_capacity_tokens,
                    batch.maximum_sequence()
                );
            }
            let blocks = (batch.maximum_sequence() / 16) as usize;
            let order: std::collections::BTreeSet<_> = (0..blocks)
                .map(|block| physical_block(block, blocks))
                .collect();
            assert_eq!(order.len(), blocks);
            assert_eq!(physical_block(0, blocks), 0);
            for row in &rows {
                assert_ne!(
                    physical_block((row.sequence_tokens as usize - 1) / 16, blocks),
                    0
                );
            }
        }
    }
}

fn physical_block(logical: usize, blocks: usize) -> usize {
    logical * 17 % blocks
}

struct Arm {
    stream: Arc<CudaStream>,
    shape: CausalAttentionShape,
    layout: ScratchLayout,
    binding_layout: BindingLayout,
    launches: Vec<CausalAttentionLaunch>,
    batch: BatchDecode,
    binding: Guarded<u64>,
    scratch: Guarded<f16>,
    scratch_initial: Vec<f16>,
    pages: Vec<Guarded<f16>>,
    pages_initial: Vec<Vec<f16>>,
    query_norm: Guarded<f16>,
    key_norm: Guarded<f16>,
}

impl Arm {
    fn new(runtime: &CudaDeviceRuntime, owners: usize, base: u64) -> Self {
        Self::with_lengths(runtime, &representative_lengths(owners, base))
    }

    fn with_lengths(runtime: &CudaDeviceRuntime, lengths: &[u64]) -> Self {
        let owners = lengths.len();
        let stream = runtime.context().new_stream().unwrap();
        let shape = qwen9b_shape();
        let layout = ScratchLayout::for_participants(
            shape,
            owners as u64,
            owners,
            CausalProjection::F16,
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap();
        let binding_layout = BindingLayout::new(shape, owners).unwrap();
        let launches = launches(shape, layout, binding_layout, lengths);
        let batch = BatchDecode::for_launches(&launches, true, binding_layout, shape, layout)
            .unwrap()
            .unwrap();
        assert!(launches.iter().all(
            |row| row.replay_topology.envelope().sequence_capacity_tokens
                <= batch.maximum_sequence()
        ));
        let blocks = (batch.maximum_sequence() / 16) as usize;
        let block_elements = 2 * shape.kv_features as usize * 16;
        let pages_initial: Vec<Vec<_>> = (0..owners)
            .map(|owner| {
                (0..blocks * block_elements)
                    .map(|i| f16::from_f32((((i * 7 + owner * 19) % 97) as f32 - 48.0) / 512.0))
                    .collect()
            })
            .collect();
        let pages: Vec<_> = pages_initial
            .iter()
            .map(|values| Guarded::new(&stream, values, f16::from_f32(77.0)))
            .collect();
        let slot_words = binding_layout.slot_bytes as usize / 8;
        let mut words = vec![u64::MAX; binding_layout.required_bytes as usize / 8];
        for (owner, &length) in lengths.iter().enumerate() {
            let start = owner * slot_words;
            // Shader prepare reads const control[1]=position_start and [2]=1;
            // gather reads [3]=current length. Neither increments these words.
            words[start] = (length - 1) << 32 | shape.table_entries(length).unwrap();
            words[start + 1] = length << 32 | 1;
            words[start + 2] = owner as u64;
            for logical in 0..blocks {
                let source = if logical == 0 {
                    &pages[0]
                } else {
                    &pages[owner]
                };
                words[start + 3 + logical] = source.pointer(&stream)
                    + (physical_block(logical, blocks) * block_elements * 2) as u64;
            }
        }
        let binding = Guarded::new(&stream, &words, u64::MAX - 3);
        let mut scratch_initial = vec![f16::from_f32(91.0); layout.required_bytes as usize / 2];
        for (offset, width, seed) in [
            (layout.query_raw, shape.query_projection_features, 11),
            (layout.key_raw, shape.kv_features, 17),
            (layout.value_raw, shape.kv_features, 23),
        ] {
            let start = offset as usize / 2;
            for i in 0..owners * width as usize {
                scratch_initial[start + i] =
                    f16::from_f32((((i * seed + 5) % 61) as f32 - 30.0) / 64.0);
            }
        }
        for offset in [layout.query, layout.context] {
            let start = offset as usize / 2;
            scratch_initial[start..start + owners * shape.query_features as usize].fill(f16::NAN);
        }
        let scratch = Guarded::new(&stream, &scratch_initial, f16::from_f32(93.0));
        let query_norm = Guarded::new(
            &stream,
            &(0..shape.head_dim)
                .map(|i| f16::from_f32(1.0 + (i % 11) as f32 / 64.0))
                .collect::<Vec<_>>(),
            f16::from_f32(95.0),
        );
        let key_norm = Guarded::new(
            &stream,
            &(0..shape.head_dim)
                .map(|i| f16::from_f32(1.0 - (i % 7) as f32 / 64.0))
                .collect::<Vec<_>>(),
            f16::from_f32(97.0),
        );
        stream.synchronize().unwrap();
        Self {
            stream,
            shape,
            layout,
            binding_layout,
            launches,
            batch,
            binding,
            scratch,
            scratch_initial,
            pages,
            pages_initial,
            query_norm,
            key_norm,
        }
    }

    fn enqueue(&self, functions: &CausalAttentionFunctions, batched: bool) {
        let stream = &self.stream;
        let scratch = self.scratch.pointer(stream);
        let binding = self.binding.pointer(stream);
        let cuda = self.shape.cuda_shape().unwrap();
        if batched {
            self.batch
                .enqueue(
                    stream,
                    functions,
                    &self.launches,
                    binding,
                    self.binding_layout,
                    cuda,
                    self.layout,
                    scratch,
                    self.query_norm.pointer(stream),
                    self.key_norm.pointer(stream),
                    self.shape.output_gate,
                )
                .unwrap();
        } else {
            // The former packed-provider loop, with the same prepared Q/K/V
            // inputs and actual per-owner replay topology as the batch arm.
            for &launch in &self.launches {
                let control = binding + launch.binding_offset;
                launch_prepare(
                    stream,
                    &functions.prepare,
                    scratch + launch.packed_query_raw,
                    scratch + launch.packed_key_raw,
                    scratch + launch.packed_value_raw,
                    self.query_norm.pointer(stream),
                    self.key_norm.pointer(stream),
                    scratch + launch.packed_query,
                    control,
                    control + BINDING_CONTROL_BYTES,
                    launch,
                    cuda,
                    1,
                    None,
                )
                .unwrap();
                launch_selected_attention(
                    stream,
                    functions,
                    scratch + launch.packed_query,
                    scratch + launch.packed_query_raw,
                    control,
                    control + BINDING_CONTROL_BYTES,
                    scratch + launch.packed_context,
                    launch,
                    cuda,
                    self.layout,
                    scratch,
                )
                .unwrap();
                launch_attention_gate(
                    stream,
                    &functions.attention_gate,
                    scratch + launch.packed_context,
                    scratch + launch.packed_query_raw,
                    launch,
                    cuda,
                )
                .unwrap();
            }
        }
    }

    fn capture(&self, functions: &CausalAttentionFunctions, batched: bool) -> CudaGraph {
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        self.enqueue(functions, batched);
        self.stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .unwrap()
            .expect("nonempty causal batch core graph")
    }

    fn validate(&self, batched: bool) -> (Vec<f16>, Vec<Vec<f16>>) {
        self.binding.assert_unchanged(&self.stream);
        self.query_norm.assert_unchanged(&self.stream);
        self.key_norm.assert_unchanged(&self.stream);
        let scratch = self.scratch.read(&self.stream);
        let owners = self.launches.len();
        let row_elements = owners * self.shape.query_features as usize;
        let partition = self.layout.vllm.unwrap();
        let writable = [
            (
                self.layout.query,
                self.layout.query + row_elements as u64 * 2,
            ),
            (
                self.layout.context,
                self.layout.context + row_elements as u64 * 2,
            ),
            (partition.exp_sums, partition.max_logits),
            (partition.max_logits, partition.temporary_output),
            (partition.temporary_output, partition.sequence_lengths),
            (partition.sequence_lengths, self.layout.required_bytes),
        ];
        for (index, (now, old)) in scratch.iter().zip(&self.scratch_initial).enumerate() {
            if !writable
                .iter()
                .any(|&(begin, end)| begin <= index as u64 * 2 && (index as u64 * 2) < end)
            {
                assert_eq!(
                    now.to_bits(),
                    old.to_bits(),
                    "immutable scratch element {index}"
                );
            }
        }
        let mut outputs = Vec::with_capacity(row_elements * 2);
        for offset in [self.layout.query, self.layout.context] {
            let start = offset as usize / 2;
            let values = &scratch[start..start + row_elements];
            assert!(
                values.iter().all(|v| v.is_finite()),
                "unwritten/nonfinite prepared query or gated output"
            );
            outputs.extend_from_slice(values);
        }
        if batched {
            let start = partition.sequence_lengths as usize / 2;
            for (owner, row) in self.launches.iter().enumerate() {
                let low = scratch[start + owner * 2].to_bits() as u32;
                let high = scratch[start + owner * 2 + 1].to_bits() as u32;
                assert_eq!(
                    low | high << 16,
                    row.sequence_tokens as u32,
                    "gathered length differs from fixed binding"
                );
            }
        }
        let blocks = (self.batch.maximum_sequence() / 16) as usize;
        let block_elements = 2 * self.shape.kv_features as usize * 16;
        let pages = self
            .pages
            .iter()
            .map(|p| p.read(&self.stream))
            .collect::<Vec<_>>();
        for (owner, values) in pages.iter().enumerate() {
            let position = self.launches[owner].sequence_tokens as usize - 1;
            let block = physical_block(position / 16, blocks);
            let token = position % 16;
            let mut mutable = std::collections::BTreeSet::new();
            for head in 0..self.shape.key_value_heads as usize {
                for dim in 0..self.shape.head_dim as usize {
                    // Addressed-vLLM K is packed in vectors of 8 half values;
                    // V is [head,dim,token]. Only this owner's final slot is writable.
                    mutable.insert(
                        block * block_elements
                            + head * self.shape.head_dim as usize * 16
                            + (dim / 8) * 16 * 8
                            + token * 8
                            + dim % 8,
                    );
                    mutable.insert(
                        block * block_elements
                            + self.shape.kv_features as usize * 16
                            + head * self.shape.head_dim as usize * 16
                            + dim * 16
                            + token,
                    );
                }
            }
            assert_eq!(mutable.len(), 2 * self.shape.kv_features as usize);
            let mut mutable = mutable.into_iter().peekable();
            let mut changed = false;
            for (index, (now, old)) in values.iter().zip(&self.pages_initial[owner]).enumerate() {
                if mutable.peek() == Some(&index) {
                    mutable.next();
                    assert!(now.is_finite());
                    changed |= now.to_bits() != old.to_bits();
                } else {
                    assert_eq!(
                        now.to_bits(),
                        old.to_bits(),
                        "owner {owner} modified prefix/tail KV at {index}"
                    );
                }
            }
            assert!(
                changed,
                "prepare did not write owner {owner}'s admitted current token"
            );
        }
        (outputs, pages)
    }
}

fn validate_pair(serial: &Arm, batch: &Arm) {
    let (a, ka) = serial.validate(false);
    let (b, kb) = batch.validate(true);
    assert_eq!(a.len(), b.len());
    assert_eq!(ka.len(), kb.len());
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "query/output element {i}");
    }
    for (owner, (a, b)) in ka.iter().zip(&kb).enumerate() {
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "owner {owner} state element {i}");
        }
    }
}

fn measure(arm: &Arm, graph: &CudaGraph) -> (f64, f64) {
    arm.stream.synchronize().unwrap();
    let host = Instant::now();
    let start = arm
        .stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    for _ in 0..REPLAYS {
        graph.launch().unwrap();
    }
    let end = arm
        .stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    (
        f64::from(start.elapsed_ms(&end).unwrap()) * 1000.0 / f64::from(REPLAYS),
        host.elapsed().as_secs_f64() * 1e6 / f64::from(REPLAYS),
    )
}

#[test]
#[ignore = "paired full-core CUDA event timing; requires exclusive CUDA and pinned addressed-vLLM operators"]
fn causal_batched_v2_full_core_graph_timing() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.causal-batch-timing").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    let mut qualified = true;
    for owners in [4, 8] {
        for base in [619, 1080] {
            let serial = Arm::new(&runtime, owners, base);
            let batch = Arm::new(&runtime, owners, base);
            let serial_graph = serial.capture(&provider.functions, false);
            let batch_graph = batch.capture(&provider.functions, true);
            // No timing is emitted before real output/state/guard qualification.
            for _ in 0..3 {
                serial_graph.launch().unwrap();
                batch_graph.launch().unwrap();
            }
            validate_pair(&serial, &batch);
            let mut improvements = Vec::with_capacity(PAIRS);
            for pair in 0..PAIRS {
                let (a, b) = if pair % 2 == 0 {
                    (
                        measure(&serial, &serial_graph),
                        measure(&batch, &batch_graph),
                    )
                } else {
                    let b = measure(&batch, &batch_graph);
                    (measure(&serial, &serial_graph), b)
                };
                validate_pair(&serial, &batch);
                assert!(a.0.is_finite() && a.0 > 0.0 && b.0.is_finite() && b.0 > 0.0);
                let improvement = 1.0 - b.0 / a.0;
                improvements.push(improvement);
                println!(
                    "{}",
                    serde_json::json!({"kind":"causal_batch_core_timing_v1","owners":owners,"lengths":representative_lengths(owners,base),"declared_context_capacity":CAPACITY,"replay_envelope":batch.batch.maximum_sequence(),"query_heads":16,"kv_heads":4,"head_dim":256,"rope_dim":64,"rope_interleaved":false,"output_gate":true,"pair":pair,"order":if pair%2==0 {"serial_batch"} else {"batch_serial"},"replays":REPLAYS,"serial_gpu_us":a.0,"batch_gpu_us":b.0,"serial_host_inclusive_us":a.1,"batch_host_inclusive_us":b.1,"improvement":improvement,"output_and_state_bitwise":true,"guards_and_immutable":true,"boundary":"prepare_gather_v2_reduce_gate_graph_replay","projections_included":false})
                );
            }
            improvements.sort_by(f64::total_cmp);
            let median = improvements[PAIRS / 2];
            qualified &= median >= 0.05;
            println!(
                "{}",
                serde_json::json!({"kind":"causal_batch_core_timing_summary_v1","owners":owners,"base_length":base,"median_paired_improvement":median,"meets_predeclared_5pct":median>=0.05,"pairs":PAIRS,"serving_evidence":false})
            );
        }
    }
    // A negative speed result is retained, not converted to a correctness
    // failure. Product adoption requires the declared gate plus service tests.
    println!(
        "{}",
        serde_json::json!({"kind":"causal_batch_core_timing_gate_v1","all_b4_b8_cases_meet_5pct":qualified,"release_approved":false})
    );
}

#[test]
#[ignore = "requires actual CUDA and pinned addressed-vLLM operators"]
fn causal_grouped_v1_and_mixed_core_preserve_outputs_state_and_graph_replay() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.causal-grouped").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    for lengths in [
        &[32, 64, 129, 256, 400, 499, 510, 512][..],
        &[32, 256, 513, 1023, 128, 400, 1025, 1537][..],
        &[32, 513, 1025, 400][..],
    ] {
        let serial = Arm::with_lengths(&runtime, lengths);
        let batch = Arm::with_lengths(&runtime, lengths);
        serial.enqueue(&provider.functions, false);
        batch.enqueue(&provider.functions, true);
        validate_pair(&serial, &batch);
        let serial_graph = serial.capture(&provider.functions, false);
        let batch_graph = batch.capture(&provider.functions, true);
        for _ in 0..3 {
            serial_graph.launch().unwrap();
            batch_graph.launch().unwrap();
            validate_pair(&serial, &batch);
        }
    }
}
