//! Same-binary host diagnostic: complete fresh waves with ordinary uploads or
//! requests for successful-completion residency. TestRuntime records commands;
//! it neither runs CUDA copies nor simulates their driver cost.
use super::resident_input_tests::{neutral_inputs, InputFixture};
use super::*;

const WARM_WAVES: usize = 2;
const FORMAL_QUADS: usize = 4;
const WAVES_PER_SAMPLE: usize = 32;

struct Sample {
    elapsed_ns: u128,
    uploads: usize,
    uploaded_bytes: usize,
    submitted_commands: usize,
    provider_encodes: u64,
}

fn measure(
    fixture: &mut InputFixture,
    inputs: &[SubmissionWaveInputUpload],
    waves: usize,
    expected_uploads_per_wave: usize,
) -> Sample {
    let (uploads_before, submits_before, command_batches_before) = {
        let trace = fixture.fixture.runtime_trace.lock().unwrap();
        (
            trace.uploaded_payloads.len(),
            trace.submit_calls,
            trace.submitted_command_counts.len(),
        )
    };
    let encodes_before = fixture.fixture.provider_trace.lock().unwrap().encode_calls;
    let expected_slot = fixture.last_slot.clone();
    let mut stable_slot = true;
    let start = Instant::now();
    for _ in 0..waves {
        let handle = fixture.dispatch(black_box(inputs)).unwrap();
        fixture.finish(handle);
        if let Some(slot) = &expected_slot {
            stable_slot &= fixture.last_slot.as_ref() == Some(slot);
        }
    }
    let elapsed_ns = start.elapsed().as_nanos();

    // All observation, payload iteration and reporting stay outside the timer.
    // The runtime's own trace allocations during encode/submit remain inside.
    assert!(stable_slot);
    assert!(fixture.last_slot.is_some());
    assert_eq!(fixture.reaper.retained_count(), 0);
    assert_eq!(fixture.lane.in_flight_count(), 0);
    let (uploads, uploaded_bytes, submitted_commands) = {
        let trace = fixture.fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls - submits_before, waves as u64);
        assert_eq!(
            trace.submitted_command_counts.len() - command_batches_before,
            waves
        );
        let payloads = &trace.uploaded_payloads[uploads_before..];
        assert_eq!(payloads.len(), expected_uploads_per_wave * waves);
        let bytes = payloads.iter().map(Vec::len).sum::<usize>();
        let expected_bytes = if expected_uploads_per_wave == 0 {
            0
        } else {
            fixture.width * 12 * waves
        };
        assert_eq!(bytes, expected_bytes);
        let offsets = payloads
            .iter()
            .filter(|bytes| bytes.as_slice() == [0_u8; 8])
            .count();
        let penalties = payloads
            .iter()
            .filter(|bytes| bytes.as_slice() == 1.0_f32.to_le_bytes())
            .count();
        assert_eq!(offsets * 2, payloads.len());
        assert_eq!(penalties, offsets);
        (
            payloads.len(),
            bytes,
            trace.submitted_command_counts[command_batches_before..]
                .iter()
                .sum(),
        )
    };
    let provider_encodes =
        fixture.fixture.provider_trace.lock().unwrap().encode_calls - encodes_before;
    assert!(provider_encodes >= waves as u64);
    Sample {
        elapsed_ns,
        uploads,
        uploaded_bytes,
        submitted_commands,
        provider_encodes,
    }
}

fn report(
    phase: &str,
    width: usize,
    cache_request: bool,
    quad: Option<usize>,
    order_index: usize,
    waves: usize,
    sample: &Sample,
) {
    println!(
        "{}",
        serde_json::json!({
            "kind": "input_residency_dispatch_sample",
            "phase": phase,
            "participants": width,
            "arm": if cache_request { "request_residency" } else { "ordinary_upload" },
            "quad": quad,
            "order_index": order_index,
            "waves": waves,
            "elapsed_ns": sample.elapsed_ns,
            "ns_per_wave": sample.elapsed_ns as f64 / waves as f64,
            "actual_encode_upload_calls": sample.uploads,
            "actual_uploaded_payload_bytes": sample.uploaded_bytes,
            "actual_submitted_commands": sample.submitted_commands,
            "actual_provider_encodes": sample.provider_encodes,
            "succeeded_and_retired_waves": waves,
            "errors": 0,
            "stable_step_slot": true,
            "lane_in_flight_after": 0,
            "reaper_retained_after": 0
        })
    );
}

#[test]
#[ignore = "host timing diagnostic; run an optimized test binary with --nocapture --test-threads=1"]
fn input_residency_complete_dispatch_paired_host_bench() {
    println!(
        "{}",
        serde_json::json!({
            "kind": "input_residency_dispatch_plan",
            "participants": [1, 8, 32],
            "first_waves_per_arm": 1,
            "warm_waves_per_arm": WARM_WAVES,
            "formal_quads": FORMAL_QUADS,
            "formal_samples_per_arm": FORMAL_QUADS * 2,
            "waves_per_formal_sample": WAVES_PER_SAMPLE,
            "quad_orders": ["ABBA", "BAAB", "ABBA", "BAAB"],
            "arm_a": "ordinary_upload",
            "arm_b": "request_residency",
            "ranges_per_participant": [[0, 8], [12, 16]],
            "live_bytes_per_participant": 12,
            "scope": "fresh Step admission, provider binding, complete dispatch and submit, wait Succeeded, publish while Step is held, handle drop, Step retirement/drop",
            "excluded": "initial Plan/session/lane setup and final close; immutable upload-request construction; trace observation and JSON reporting",
            "limits": "same new binary and common core path; synthetic TestRuntime trace allocation is included but no CUDA/driver cost; repeated stable slot; not an inference throughput estimate or a baseline estimate of all newly added host work",
            "time_ratio_is_asserted": false
        })
    );
    for (shape_index, width) in [1, 8, 32].into_iter().enumerate() {
        let mut ordinary = InputFixture::new(width, true);
        let mut resident = InputFixture::new(width, true);
        let ordinary_inputs = neutral_inputs(width, false);
        let resident_inputs = neutral_inputs(width, true);
        let initial_order = if shape_index % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for phase in ["first", "warm"] {
            for (order_index, cache_request) in initial_order.into_iter().enumerate() {
                let waves = if phase == "first" { 1 } else { WARM_WAVES };
                let expected_uploads = if phase == "first" || !cache_request {
                    width * 2
                } else {
                    0
                };
                let (fixture, inputs) = if cache_request {
                    (&mut resident, &resident_inputs)
                } else {
                    (&mut ordinary, &ordinary_inputs)
                };
                let sample = measure(fixture, inputs, waves, expected_uploads);
                report(
                    phase,
                    width,
                    cache_request,
                    None,
                    order_index,
                    waves,
                    &sample,
                );
            }
        }
        for quad in 0..FORMAL_QUADS {
            let order = if quad % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            let mut ordinary_ns = 0_u128;
            let mut resident_ns = 0_u128;
            for (order_index, cache_request) in order.into_iter().enumerate() {
                let (fixture, inputs, expected_uploads) = if cache_request {
                    (&mut resident, &resident_inputs, 0)
                } else {
                    (&mut ordinary, &ordinary_inputs, width * 2)
                };
                let sample = measure(fixture, inputs, WAVES_PER_SAMPLE, expected_uploads);
                if cache_request {
                    resident_ns += sample.elapsed_ns;
                } else {
                    ordinary_ns += sample.elapsed_ns;
                }
                report(
                    "formal",
                    width,
                    cache_request,
                    Some(quad),
                    order_index,
                    WAVES_PER_SAMPLE,
                    &sample,
                );
            }
            println!(
                "{}",
                serde_json::json!({
                    "kind": "input_residency_dispatch_pair",
                    "participants": width,
                    "quad": quad,
                    "order": if quad % 2 == 0 { "ABBA" } else { "BAAB" },
                    "waves_per_arm": WAVES_PER_SAMPLE * 2,
                    "ordinary_elapsed_ns": ordinary_ns,
                    "residency_elapsed_ns": resident_ns,
                    "residency_over_ordinary": resident_ns as f64 / ordinary_ns as f64,
                    "errors": 0
                })
            );
        }
        ordinary.close();
        resident.close();
    }
}
