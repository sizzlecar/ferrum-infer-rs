use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Barrier;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Serialize)]
struct LegacyOutput<'a> {
    request: &'a CompletionReadbackRequest,
    bytes: &'a [u8],
    sha256: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
enum LegacyDisposition<'a> {
    Succeeded(LegacyOutput<'a>),
    NotAttempted(&'a CompletionReadbackRequest),
    FailedButQuiescent {
        request: &'a CompletionReadbackRequest,
        failures: &'a [IdentifiedFailure],
    },
    ContractFailedButQuiescent {
        request: &'a CompletionReadbackRequest,
        failure: &'a QuiescentCompletionContractFailure,
    },
}

fn legacy_disposition(value: &CompletionReadbackDisposition) -> LegacyDisposition<'_> {
    match value {
        CompletionReadbackDisposition::Succeeded(output) => {
            LegacyDisposition::Succeeded(LegacyOutput {
                request: output.request(),
                bytes: output.bytes(),
                sha256: digest(output.bytes()),
            })
        }
        CompletionReadbackDisposition::NotAttempted(request) => {
            LegacyDisposition::NotAttempted(request)
        }
        CompletionReadbackDisposition::FailedButQuiescent { request, failures } => {
            LegacyDisposition::FailedButQuiescent { request, failures }
        }
        CompletionReadbackDisposition::ContractFailedButQuiescent { request, failure } => {
            LegacyDisposition::ContractFailedButQuiescent { request, failure }
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
enum LegacyEvidence<'a> {
    Succeeded {
        request: &'a CompletionReadbackRequest,
        output_sha256: String,
    },
    NotAttempted {
        request: &'a CompletionReadbackRequest,
    },
    FailedButQuiescent {
        request: &'a CompletionReadbackRequest,
        failures: &'a [IdentifiedFailure],
    },
    ContractFailedButQuiescent {
        request: &'a CompletionReadbackRequest,
        failure: &'a str,
    },
}

fn legacy_evidence(value: &CompletionReadbackDisposition) -> LegacyEvidence<'_> {
    match value {
        CompletionReadbackDisposition::Succeeded(output) => LegacyEvidence::Succeeded {
            request: output.request(),
            output_sha256: digest(output.bytes()),
        },
        CompletionReadbackDisposition::NotAttempted(request) => {
            LegacyEvidence::NotAttempted { request }
        }
        CompletionReadbackDisposition::FailedButQuiescent { request, failures } => {
            LegacyEvidence::FailedButQuiescent { request, failures }
        }
        CompletionReadbackDisposition::ContractFailedButQuiescent { request, failure } => {
            LegacyEvidence::ContractFailedButQuiescent {
                request,
                failure: failure.reason(),
            }
        }
    }
}

fn legacy_batch_fingerprint(receipt: &CompletionReadbackBatchReceipt) -> String {
    #[derive(Serialize)]
    struct Input<'a> {
        domain: &'static str,
        completion_fingerprint: &'a str,
        dispositions: Vec<LegacyEvidence<'a>>,
    }
    digest(
        &serde_json::to_vec(&Input {
            domain: "ferrum.runtime-vnext.completion-readback-batch.v2",
            completion_fingerprint: receipt.completion().fingerprint(),
            dispositions: receipt.dispositions().iter().map(legacy_evidence).collect(),
        })
        .unwrap(),
    )
}

fn legacy_batch_json(receipt: &CompletionReadbackBatchReceipt) -> Vec<u8> {
    #[derive(Serialize)]
    struct Wire<'a> {
        completion: &'a OperationCompletionReceipt,
        dispositions: Vec<LegacyDisposition<'a>>,
        readback_timings: Option<&'a [DeviceTimingMeasurement<CompletionReadbackTiming>]>,
        fingerprint: String,
    }
    serde_json::to_vec(&Wire {
        completion: receipt.completion(),
        dispositions: receipt
            .dispositions()
            .iter()
            .map(legacy_disposition)
            .collect(),
        readback_timings: receipt.readback_timings(),
        fingerprint: legacy_batch_fingerprint(receipt),
    })
    .unwrap()
}

fn legacy_single_fingerprint(receipt: &CompletionReadbackReceipt) -> String {
    #[derive(Serialize)]
    struct Input<'a> {
        domain: &'static str,
        completion_fingerprint: &'a str,
        request: &'a CompletionReadbackRequest,
        output_sha256: Option<String>,
        failures: Option<&'a [IdentifiedFailure]>,
        contract_failure: Option<&'a str>,
    }
    let (request, output_sha256, failures, contract_failure) = match receipt.disposition() {
        CompletionReadbackDisposition::Succeeded(output) => {
            (output.request(), Some(digest(output.bytes())), None, None)
        }
        CompletionReadbackDisposition::NotAttempted(request) => (request, None, None, None),
        CompletionReadbackDisposition::FailedButQuiescent { request, failures } => {
            (request, None, Some(failures.as_slice()), None)
        }
        CompletionReadbackDisposition::ContractFailedButQuiescent { request, failure } => {
            (request, None, None, Some(failure.reason()))
        }
    };
    digest(
        &serde_json::to_vec(&Input {
            domain: "ferrum.runtime-vnext.completion-readback.v1",
            completion_fingerprint: receipt.completion().fingerprint(),
            request,
            output_sha256,
            failures,
            contract_failure,
        })
        .unwrap(),
    )
}

fn legacy_single_json(receipt: &CompletionReadbackReceipt) -> Vec<u8> {
    #[derive(Serialize)]
    struct Wire<'a> {
        completion: &'a OperationCompletionReceipt,
        disposition: LegacyDisposition<'a>,
        readback_timing: Option<DeviceTimingMeasurement<CompletionReadbackTiming>>,
        fingerprint: String,
    }
    serde_json::to_vec(&Wire {
        completion: receipt.completion(),
        disposition: legacy_disposition(receipt.disposition()),
        readback_timing: receipt.readback_timing(),
        fingerprint: legacy_single_fingerprint(receipt),
    })
    .unwrap()
}

fn token_requests(rows: u32) -> CompletionReadbackBatchRequest {
    CompletionReadbackBatchRequest::new(
        (0..rows)
            .map(|row| {
                CompletionReadbackRequest::new(
                    id("node.tail"),
                    row,
                    id("resource.output"),
                    0,
                    HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

fn batch_receipt(
    captured: bool,
    fail_readback: bool,
    fail_fence: bool,
) -> CompletionReadbackBatchReceipt {
    let fixture = fixture_tokens();
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.submission_readback_enabled = captured;
        trace.submission_readback_fails = fail_readback;
    }
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    lane.configure_submission_readback_staging(8).unwrap();
    let reaper = CompletionReaper::new();
    let requests = token_requests(2);
    let cohort = submit_cohort_with_spans(
        &fixture,
        &lane,
        &reaper,
        "lazy-batch",
        vec![one_token_span(); 2],
        Some(vec![53, 127]),
        captured.then(|| requests.clone()),
    );
    if fail_fence {
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .fence_behaviors
            .insert(cohort.fence, FenceBehavior::FailedButQuiescent);
    }
    let receipt = match cohort.handle.wait_with_readbacks(requests).unwrap() {
        CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
        other => panic!("expected terminal readback: {other:?}"),
    };
    if !fail_readback && !fail_fence {
        for (row, expected) in receipt.dispositions().iter().zip([54_u32, 128]) {
            let CompletionReadbackDisposition::Succeeded(output) = row else {
                panic!("token failed: {row:?}")
            };
            assert_eq!(output.bytes(), expected.to_le_bytes());
        }
    }
    drop(cohort.handle);
    cohort.step.try_retire_normal().unwrap();
    drop(cohort.batch);
    for session in &cohort.sessions {
        session.try_abort_if_quiescent().unwrap();
    }
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
    receipt
}

#[test]
fn lazy_readback_batch_real_bytes_match_legacy_for_generic_and_captured_paths() {
    for captured in [false, true] {
        let receipt = batch_receipt(captured, false, false);
        let cold = receipt.clone();
        let expected = legacy_batch_fingerprint(&receipt);
        assert_eq!(receipt.fingerprint(), expected);
        assert_eq!(receipt, cold);
        assert_eq!(
            serde_json::to_vec(&receipt).unwrap(),
            legacy_batch_json(&receipt)
        );
        assert_eq!(
            serde_json::to_vec(&cold).unwrap(),
            legacy_batch_json(&receipt)
        );
        assert_eq!(format!("{receipt:?}"), format!(
            "CompletionReadbackBatchReceipt {{ completion: {:?}, dispositions: {:?}, readback_timings: {:?}, fingerprint: {:?} }}",
            receipt.completion(), receipt.dispositions(), receipt.readback_timings(), expected));
    }
}

#[test]
fn lazy_readback_single_real_bytes_match_legacy_and_cold_clone() {
    let fixture = fixture_tokens();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    let cohort = submit_cohort_with_spans(
        &fixture,
        &lane,
        &reaper,
        "lazy-single",
        vec![one_token_span()],
        Some(vec![93]),
        None,
    );
    let request = token_requests(1).requests()[0].clone();
    let receipt = match cohort.handle.wait_with_readback(request).unwrap() {
        CompletionReadbackObservation::Terminal(receipt) => receipt,
        other => panic!("expected terminal readback: {other:?}"),
    };
    let cold = receipt.clone();
    let CompletionReadbackDisposition::Succeeded(output) = receipt.disposition() else {
        panic!("token failed")
    };
    assert_eq!(output.bytes(), 94_u32.to_le_bytes());
    let expected = legacy_single_fingerprint(&receipt);
    assert_eq!(receipt.fingerprint(), expected);
    assert_eq!(receipt, cold);
    assert_eq!(
        serde_json::to_vec(&cold).unwrap(),
        legacy_single_json(&receipt)
    );
    assert_eq!(format!("{receipt:?}"), format!(
        "CompletionReadbackReceipt {{ completion: {:?}, disposition: {:?}, readback_timing: {:?}, fingerprint: {:?} }}",
        receipt.completion(), receipt.disposition(), receipt.readback_timing(), expected));
    cohort.retire();
}

#[test]
fn lazy_readback_batch_failure_and_not_attempted_keep_exact_legacy_evidence() {
    for (readback_failure, fence_failure) in [(true, false), (false, true)] {
        let receipt = batch_receipt(true, readback_failure, fence_failure);
        if readback_failure {
            // The first device read fails and closes the lane. The second row
            // must then fail the lane contract without another device read.
            assert!(
                matches!(
                    receipt.dispositions(),
                    [CompletionReadbackDisposition::FailedButQuiescent { .. },
                     CompletionReadbackDisposition::ContractFailedButQuiescent { failure, .. }]
                        if failure.reason().contains("staged readback lane is no longer valid")
                ),
                "unexpected row failure classification: {:?}",
                receipt.dispositions()
            );
        } else {
            assert!(receipt
                .dispositions()
                .iter()
                .all(|row| matches!(row, CompletionReadbackDisposition::NotAttempted(_))));
        }
        let cold = receipt.clone();
        assert_eq!(receipt.fingerprint(), legacy_batch_fingerprint(&receipt));
        assert_eq!(receipt, cold);
        assert_eq!(
            serde_json::to_vec(&receipt).unwrap(),
            legacy_batch_json(&receipt)
        );
    }
}

#[test]
fn lazy_readback_batch_concurrent_serialize_and_fingerprint_match_legacy() {
    let receipt = Arc::new(batch_receipt(true, false, false));
    let expected_json = legacy_batch_json(&receipt);
    let expected_fingerprint = legacy_batch_fingerprint(&receipt);
    let barrier = Arc::new(Barrier::new(4));
    let threads = (0..4)
        .map(|_| {
            let receipt = Arc::clone(&receipt);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                (
                    receipt.fingerprint().to_owned(),
                    serde_json::to_vec(receipt.as_ref()).unwrap(),
                )
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        let (fingerprint, json) = thread.join().unwrap();
        assert_eq!(fingerprint, expected_fingerprint);
        assert_eq!(json, expected_json);
    }
}
