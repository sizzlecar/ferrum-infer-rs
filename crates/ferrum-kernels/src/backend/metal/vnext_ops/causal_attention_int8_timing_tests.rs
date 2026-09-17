//! Opt-in, same-state scalar/vector GQA gather device timing.
//! This excludes prepare, projections, the rest of the model and HTTP serving.

use super::*;
use metal::objc::rc::autoreleasepool;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Instant;

struct PairedGqaReader {
    device: Device,
    queue: metal::CommandQueue,
    vector: MetalCausalAttentionPipelines,
    scalar: ComputePipelineState,
    state: State,
    query: Buffer,
    query_raw: Buffer,
    output: GuardedAttentionOutput,
    bindings: Buffer,
    params: CausalAttentionParams,
    plan: AttentionDispatchPlan,
}

impl PairedGqaReader {
    fn new(prefix: usize, tokens: usize) -> Self {
        let device = Device::system_default().expect("INT8 GQA timing requires Metal");
        let queue = device.new_command_queue();
        let vector = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
        // No product option or environment variable selects the reference.
        // The preprocessor only changes the two GQA gather loops; compilation
        // options, every matrix/softmax/tail statement and bindings are shared.
        let source =
            format!("#define VNEXT_INT8_GQA_SCALAR_GATHER_REFERENCE 1\n{INT8_SHADER_SOURCE}");
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .unwrap();
        let function = library
            .get_function("vnext_causal_attention_prefill_gqa_tiled_int8", None)
            .unwrap();
        let reflection = function.new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
        assert_eq!(reflection.encoded_length(), vector.binding_encoded_length);
        assert_eq!(reflection.alignment(), vector.binding_alignment);
        let scalar = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        assert_eq!(scalar.thread_execution_width(), SIMD_THREADS);
        assert!(u64::from(scalar.max_total_threads_per_threadgroup()) >= 256);
        let (heads, kv_heads, dim) = (16, 4, 256);
        let state = State::new(&device, prefix + tokens, kv_heads, dim);
        if prefix != 0 {
            let inputs = Inputs::new(prefix, heads, kv_heads, dim, true);
            assert_eq!(
                inputs
                    .execute(&device, &queue, &vector, &state, 0, false)
                    .error,
                0
            );
        }
        let inputs = Inputs::new(tokens, heads, kv_heads, dim, true);
        let prepared = inputs.execute(&device, &queue, &vector, &state, prefix, false);
        assert_eq!(prepared.error, 0);
        let params = inputs.params(&vector, &state, prefix);
        let plan = vector.dispatch_plan(&params);
        assert_eq!(plan.kind, AttentionDispatchKind::GqaTiledPrefill);
        let query = shared_buffer(&device, &prepared.query);
        let query_raw = shared_buffer(&device, &inputs.query);
        let output = GuardedAttentionOutput::new(&device, &params);
        let bindings = device.new_buffer(
            vector.binding_slot_bytes().unwrap(),
            MTLResourceOptions::StorageModeShared,
        );
        vector
            .with_binding_encoder(|encoder| {
                encoder.set_argument_buffer(&bindings, 0);
                encoder.set_buffers(
                    0,
                    &state.payload.iter().map(|page| &**page).collect::<Vec<_>>(),
                    &vec![0; state.payload.len()],
                );
                encoder.set_buffers(
                    MAXIMUM_KV_PAGES,
                    &state.scales.iter().map(|page| &**page).collect::<Vec<_>>(),
                    &vec![0; state.scales.len()],
                );
                Ok(())
            })
            .unwrap();
        Self {
            device,
            queue,
            vector,
            scalar,
            state,
            query,
            query_raw,
            output,
            bindings,
            params,
            plan,
        }
    }

    fn run(&self, vector: bool) -> (Vec<f32>, Value) {
        self.output.reset();
        let started = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(if vector {
            &self.vector.gqa_tiled_prefill_attention
        } else {
            &self.scalar
        });
        set_raw(encoder, 0, &self.query);
        set_raw(encoder, 1, &self.query_raw);
        set_raw(encoder, 2, &self.output.buffer);
        encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&self.bindings), 0);
        set_raw_params(encoder, 4, &self.params);
        for page in self.state.payload.iter().chain(&self.state.scales) {
            encoder.use_resource(&**page, MTLResourceUsage::Read);
        }
        encoder.set_threadgroup_memory_length(0, self.plan.threadgroup_memory_bytes[0]);
        encoder.set_threadgroup_memory_length(1, self.plan.threadgroup_memory_bytes[1]);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                self.plan.threadgroups[0],
                self.plan.threadgroups[1],
                self.plan.threadgroups[2],
            ),
            MTLSize::new(
                self.plan.threads_per_threadgroup[0],
                self.plan.threads_per_threadgroup[1],
                self.plan.threads_per_threadgroup[2],
            ),
        );
        encoder.end_encoding();
        let encode_ns = started.elapsed().as_nanos() as u64;
        let submitted = Instant::now();
        command.commit();
        command.wait_until_completed();
        let submit_wait_ns = submitted.elapsed().as_nanos() as u64;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let clock = super::super::timing::gpu_clock(command);
        let output = self.output.read_after_completion(self.plan.kind);
        (
            output,
            json!({
                "gather": if vector { "vector4" } else { "scalar" },
                "dispatches_per_command": 1,
                "host_encode_ns": encode_ns,
                "host_submit_wait_ns": submit_wait_ns,
                "gpu_clock": clock,
                "output_guard_unchanged": true,
            }),
        )
    }
}

#[test]
#[ignore = "GPU performance diagnostic: coordinate exclusive device access"]
fn int8_gqa_scalar_vector_gather_microbench() {
    for (prefix, tokens) in [(2048, 128), (4096, 512)] {
        autoreleasepool(|| {
            let case = PairedGqaReader::new(prefix, tokens);
            let payload = case.state.payload();
            let scales = case
                .state
                .scales()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            let (expected, _) = case.run(false);
            assert!(expected.iter().all(|value| value.is_finite()));
            println!(
                "{}",
                json!({
                    "kind": "metal_int8_gqa_gather_timing_config",
                    "device": case.device.name(),
                    "shader_sha256": format!("{:x}", Sha256::digest(INT8_SHADER_SOURCE.as_bytes())),
                    "prefix_tokens": prefix, "query_tokens": tokens,
                    "query_heads": 16, "kv_heads": 4, "head_dim": 256,
                    "kv_dtype": "int8_per_token_head_f32_scale_v1", "output_gate": true,
                    "page_bytes": VNEXT_KV_PAGE_BYTES,
                    "threadgroups": case.plan.threadgroups,
                    "threads_per_threadgroup": case.plan.threads_per_threadgroup,
                    "threadgroup_memory_bytes": case.plan.threadgroup_memory_bytes,
                    "warmup_pairs": 2, "measured_pairs": 3,
                    "order": "scalar/vector on even pairs; vector/scalar on odd pairs",
                    "clock_scope": "completed_single_attention_command_buffer_device_elapsed",
                    "output_oracle": "full-output scalar-GQA on identical prepared state; independent CPU conformance is separate",
                    "limitations": ["synthetic prepared tensors, not model latency", "excludes prepare, projections, other model operations and scheduling", "host submit/wait overlaps device time and must not be added to it", "requires exclusive GPU access"],
                })
            );
            for (phase, pairs) in [("warmup", 2), ("measured", 3)] {
                for round in 0..pairs {
                    let order = if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    };
                    for (position, vector) in order.into_iter().enumerate() {
                        let (actual, sample) = autoreleasepool(|| case.run(vector));
                        let max_abs =
                            assert_close("timed vector/scalar GQA", &actual, &expected, 0.001);
                        println!(
                            "{}",
                            json!({
                                "kind": "metal_int8_gqa_gather_timing_sample", "prefix_tokens": prefix,
                                "query_tokens": tokens, "phase": phase, "round": round,
                                "position_in_pair": position, "sample": sample,
                                "full_output_scalar_reference_max_abs_error": max_abs,
                            })
                        );
                    }
                }
                assert_eq!(case.state.payload(), payload);
                assert_eq!(
                    case.state
                        .scales()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    scales
                );
                println!(
                    "{}",
                    json!({"kind": "metal_int8_gqa_gather_timing_validation", "prefix_tokens": prefix, "query_tokens": tokens, "phase": phase, "payload_and_scale_slack_bits_unchanged": true})
                );
            }
        });
    }
}
