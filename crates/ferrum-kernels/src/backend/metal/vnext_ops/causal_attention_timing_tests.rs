//! Opt-in, paired timing of short prefill on the existing numerical fixture.
//!
//! Run with exclusive GPU access and the existing test target/profile:
//! `cargo test -p ferrum-kernels --features metal --lib
//! short_prefill_dispatch_microbench --locked --offline --
//! --ignored --nocapture --test-threads=1`
//! This isolates attention over prepared synthetic Q/K/V, not model throughput.

use super::*;
use metal::objc::rc::autoreleasepool;
use metal::objc::runtime::{BOOL, YES};
use metal::objc::{msg_send, sel, sel_impl};
use metal::CommandBufferRef;
use serde_json::{json, Value};
use std::time::Instant;

const WARMUP_PAIRS: usize = 2;
const MEASURED_PAIRS: usize = 8;

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 macros expand their legacy cargo-clippy feature cfg in the calling crate"
)]
fn gpu_clock(command: &CommandBufferRef) -> Value {
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: check availability before reading the documented double-valued
    // timestamps of this already-completed command. No new samples are taken.
    let (has_start, has_end): (BOOL, BOOL) = unsafe {
        (
            msg_send![command, respondsToSelector: sel!(GPUStartTime)],
            msg_send![command, respondsToSelector: sel!(GPUEndTime)],
        )
    };
    if has_start != YES || has_end != YES {
        return json!({
            "device_command_ns": null,
            "unavailable_reason": "command_buffer_timestamp_selectors_unavailable",
        });
    }
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command, GPUStartTime],
            msg_send![command, GPUEndTime],
        )
    };
    let elapsed_ns = (end - start) * 1e9;
    let valid = start.is_finite()
        && end.is_finite()
        && start > 0.0
        && end > start
        && elapsed_ns.is_finite();
    json!({
        "device_start_seconds": start.is_finite().then_some(start),
        "device_end_seconds": end.is_finite().then_some(end),
        "device_command_ns": valid.then_some(elapsed_ns),
        "unavailable_reason": (!valid).then_some("invalid_command_buffer_timestamps"),
    })
}

struct PairedAttention<'a> {
    case: &'a ValidatedPrefillCase,
    output: GuardedAttentionOutput,
    bindings: Buffer,
}

impl<'a> PairedAttention<'a> {
    fn new(case: &'a ValidatedPrefillCase) -> Self {
        let bindings = case.device.new_buffer(
            case.pipelines.binding_slot_bytes().unwrap(),
            MTLResourceOptions::StorageModeShared,
        );
        let pages = case.pages.iter().map(|page| &**page).collect::<Vec<_>>();
        case.pipelines
            .with_binding_encoder(|encoder| {
                encoder.set_argument_buffer(&bindings, 0);
                encoder.set_buffers(0, &pages, &vec![0; pages.len()]);
                Ok(())
            })
            .unwrap();
        Self {
            case,
            output: GuardedAttentionOutput::new(&case.device, &case.params),
            bindings,
        }
    }

    fn run(&self, plan: AttentionDispatchPlan, dispatches: usize) -> Value {
        // Neither timed kernel invokes prepare/RoPE or writes KV. Each dispatch
        // overwrites the same output from the same immutable Q/K/V buffers.
        self.output.reset();
        let started = Instant::now();
        let command = self.case.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        set_raw(encoder, 0, &self.case.query);
        set_raw(encoder, 1, &self.case.query_raw);
        set_raw(encoder, 2, &self.output.buffer);
        encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&self.bindings), 0);
        set_raw_params(encoder, 4, &self.case.params);
        for page in &self.case.pages {
            encoder.use_resource(&**page, MTLResourceUsage::Read);
        }
        for _ in 0..dispatches {
            encode_attention_dispatch(&self.case.pipelines, encoder, plan);
        }
        encoder.end_encoding();
        let encode_ns = started.elapsed().as_nanos() as u64;
        let submitted = Instant::now();
        command.commit();
        command.wait_until_completed();
        let submit_wait_ns = submitted.elapsed().as_nanos() as u64;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let clock = gpu_clock(command);
        // Full-output numerical validation and guard readback are outside both
        // host intervals and the device command-buffer timestamp interval.
        let output = self.output.read_after_completion(plan.kind);
        let error = assert_close("timed attention/CPU", &output, &self.case.expected, 0.001);
        json!({
            "route": format!("{:?}", plan.kind),
            "threadgroups": plan.threadgroups,
            "dispatches_per_command": dispatches,
            "host_encode_ns": encode_ns,
            "host_submit_wait_ns": submit_wait_ns,
            "gpu_clock": clock,
            "full_output_cpu_max_abs_error": error,
            "output_guard_unchanged": true,
        })
    }
}

#[test]
#[ignore = "GPU performance diagnostic: coordinate exclusive device access"]
fn short_prefill_dispatch_microbench() {
    time_short_prefill_cases(&[
        (16_384, 5, 1),
        (0, 2, 32),
        (0, 7, 32),
        (8, 5, 32),
        (32, 5, 32),
        (128, 5, 8),
        (512, 5, 4),
        (2_048, 5, 1),
    ]);
}

#[test]
#[ignore = "GPU performance diagnostic: coordinate exclusive device access"]
fn short_prefill_dispatch_crossover_microbench() {
    time_short_prefill_cases(&[
        (64, 2, 16),
        (64, 5, 16),
        (64, 7, 16),
        (128, 2, 8),
        (128, 7, 8),
    ]);
}

#[test]
#[ignore = "GPU performance diagnostic: coordinate exclusive device access"]
fn short_prefill_dispatch_tail_crossover_microbench() {
    // All cases end with seven live KV rows in the scalar value-tail block.
    // Two queries leave the most unused rows in the fixed eight-query tile.
    time_short_prefill_cases(&[
        (133, 2, 8),
        (261, 2, 8),
        (256, 7, 8),
        (517, 2, 4),
        (512, 7, 4),
    ]);
}

fn time_short_prefill_cases(cases: &[(usize, usize, usize)]) {
    for &(prefix, tokens, dispatches) in cases {
        autoreleasepool(|| {
            // The same existing CPU/general/tiled checks run before timing.
            let mut route_params = dispatch_test_params(tokens.try_into().unwrap(), 256);
            route_params.position_start = prefix.try_into().unwrap();
            let case = run_prefill_cpu_case(
                "paired short-prefill timing fixture",
                256,
                16,
                4,
                true,
                prefix,
                tokens,
                attention_dispatch_plan(&route_params).kind,
            );
            let general = general_attention_dispatch_plan(&case.params);
            let production = attention_dispatch_plan_with_memory_limit(
                &case.params,
                case.pipelines.maximum_threadgroup_memory_length,
            );
            // Compare the same two kernels even when production deliberately
            // falls back to General for a short or poorly occupied context.
            let gqa = gqa_tiled_prefill_attention_dispatch_plan(&case.params);
            assert!(
                gqa.threadgroup_memory_bytes.iter().sum::<u64>()
                    <= case.pipelines.maximum_threadgroup_memory_length
            );
            let runner = PairedAttention::new(&case);
            println!(
                "{}",
                json!({
                    "kind": "metal_short_prefill_timing_config",
                    "device": case.device.name(),
                    "prefix_tokens": prefix, "query_tokens": tokens,
                    "head_dim": 256, "query_heads": 16, "kv_heads": 4,
                    "kv_dtype": "f16", "output_gate": true,
                    "page_bytes": VNEXT_KV_PAGE_BYTES,
                    "production_route": format!("{:?}", production.kind),
                    "warmup_pairs": WARMUP_PAIRS, "measured_pairs": MEASURED_PAIRS,
                    "order": "General/explicit GQA on even pairs; explicit GQA/General on odd pairs",
                    "dispatches_per_command": dispatches,
                    "clock_scope": "completed_command_buffer_device_elapsed_no_cross_clock_anchor",
                    "limitations": [
                        "synthetic prepared Q/K/V, excludes projections, RoPE, KV upload and server scheduling",
                        "small contexts repeat dispatches over hot buffers to amortize timestamp overhead; divide each interval by dispatches_per_command",
                        "host submit/wait overlaps device elapsed and must not be added to it",
                        "debug-profile host intervals are diagnostic, not product throughput"
                    ],
                })
            );
            for (phase, pairs) in [("warmup", WARMUP_PAIRS), ("measured", MEASURED_PAIRS)] {
                for round in 0..pairs {
                    let order = if round % 2 == 0 {
                        [general, gqa]
                    } else {
                        [gqa, general]
                    };
                    for (position, plan) in order.into_iter().enumerate() {
                        let sample = autoreleasepool(|| runner.run(plan, dispatches));
                        println!(
                            "{}",
                            json!({
                                "kind": "metal_short_prefill_timing_sample",
                                "prefix_tokens": prefix, "query_tokens": tokens,
                                "phase": phase, "round": round, "position_in_pair": position,
                                "sample": sample,
                            })
                        );
                    }
                }
                case.assert_kv_unchanged(phase);
                println!(
                    "{}",
                    json!({
                        "kind": "metal_short_prefill_timing_validation",
                        "prefix_tokens": prefix, "query_tokens": tokens,
                        "phase": phase, "kv_including_nan_slack_bits_unchanged": true,
                    })
                );
            }
        });
    }
}
