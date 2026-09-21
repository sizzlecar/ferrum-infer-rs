use super::*;
use crate::vnext::ElementType;
use std::sync::Barrier;

// Receives a terminal receipt from the existing resource/reaper fixture. This
// tests derived evidence construction; real byte routing is covered by the
// U32 generic/captured integration tests.
pub(crate) fn assert_cold_receipt_chain(completion: OperationCompletionReceipt) {
    let single = CompletionReadbackReceipt::new(
        completion.clone(),
        CompletionReadbackDisposition::Succeeded(output(vec![53, 7, 177, 253])),
    );
    let single_clone = single.clone();
    assert!(single.fingerprint.get().is_none());
    assert_eq!(single.completion(), &completion);
    assert_eq!(single.readback_timing(), None);
    let CompletionReadbackDisposition::Succeeded(value) = single.disposition() else {
        unreachable!()
    };
    assert_eq!(value.bytes(), [53, 7, 177, 253]);
    assert!(value.sha256.get().is_none());
    assert_eq!(single, single_clone);
    assert!(single.fingerprint.get().is_none());
    assert!(value.sha256.get().is_none());

    let batch = CompletionReadbackBatchReceipt::new(
        completion.clone(),
        vec![CompletionReadbackDisposition::Succeeded(output(vec![
            19, 43, 61, 97,
        ]))],
    );
    let batch_clone = batch.clone();
    assert!(batch.fingerprint.get().is_none());
    assert_eq!(batch.completion(), &completion);
    assert_eq!(batch.readback_timings(), None);
    let CompletionReadbackDisposition::Succeeded(value) = &batch.dispositions()[0] else {
        unreachable!()
    };
    assert_eq!(value.bytes(), [19, 43, 61, 97]);
    assert!(value.sha256.get().is_none());
    assert_eq!(batch, batch_clone);
    assert!(batch.fingerprint.get().is_none());
    assert!(value.sha256.get().is_none());

    single.fingerprint();
    batch.fingerprint();
    assert_eq!(single, single_clone);
    assert_eq!(batch, batch_clone);
    assert!(single_clone.fingerprint.get().is_none());
    assert!(batch_clone.fingerprint.get().is_none());
    for row in [single_clone.disposition(), &batch_clone.dispositions()[0]] {
        let CompletionReadbackDisposition::Succeeded(value) = row else {
            unreachable!()
        };
        assert!(value.sha256.get().is_none());
    }

    // A readback contract failure may follow a successful device completion.
    // The independent old v1 input retains its reason and null payload digest.
    let request = output(vec![53, 7, 177, 253]).request().clone();
    for contract_failure in [None, Some("readback extent does not match its allocation")] {
        let disposition = match contract_failure {
            Some(reason) => CompletionReadbackDisposition::ContractFailedButQuiescent {
                request: request.clone(),
                failure: QuiescentCompletionContractFailure::new(reason),
            },
            None => CompletionReadbackDisposition::NotAttempted(request.clone()),
        };
        #[derive(Serialize)]
        struct LegacyInput<'a> {
            domain: &'static str,
            completion_fingerprint: &'a str,
            request: &'a CompletionReadbackRequest,
            output_sha256: Option<&'a str>,
            failures: Option<&'a [IdentifiedFailure]>,
            contract_failure: Option<&'a str>,
        }
        let expected_hash = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&LegacyInput {
                    domain: "ferrum.runtime-vnext.completion-readback.v1",
                    completion_fingerprint: completion.fingerprint(),
                    request: &request,
                    output_sha256: None,
                    failures: None,
                    contract_failure,
                })
                .unwrap()
            )
        );
        #[derive(Serialize)]
        struct LegacyReceipt<'a> {
            completion: &'a OperationCompletionReceipt,
            disposition: &'a CompletionReadbackDisposition,
            readback_timing: Option<DeviceTimingMeasurement<CompletionReadbackTiming>>,
            fingerprint: &'a str,
        }
        let expected_wire = serde_json::to_vec(&LegacyReceipt {
            completion: &completion,
            disposition: &disposition,
            readback_timing: None,
            fingerprint: &expected_hash,
        })
        .unwrap();
        let receipt = CompletionReadbackReceipt::new(completion.clone(), disposition);
        assert!(receipt.fingerprint.get().is_none());
        let cold = receipt.clone();
        assert_eq!(receipt.fingerprint(), expected_hash);
        assert_eq!(receipt, cold);
        assert_eq!(serde_json::to_vec(&cold).unwrap(), expected_wire);
    }
}

fn output(bytes: Vec<u8>) -> CompletionReadbackOutput {
    let request = CompletionReadbackRequest::new(
        NodeId::new("node.lazy-readback").unwrap(),
        0,
        ResourceId::new("resource.lazy-readback").unwrap(),
        0,
        HostTransferLayout::new(ElementType::U8, bytes.len() as u64).unwrap(),
    )
    .unwrap();
    CompletionReadbackOutput::new(
        request,
        LaneReadback {
            bytes,
            timing: DeviceTimingMeasurement::NotRequested,
        },
    )
    .unwrap()
}

#[derive(Serialize)]
struct LegacyOutput<'a> {
    request: &'a CompletionReadbackRequest,
    bytes: &'a [u8],
    sha256: String,
}

fn legacy_json(output: &CompletionReadbackOutput) -> Vec<u8> {
    serde_json::to_vec(&LegacyOutput {
        request: output.request(),
        bytes: output.bytes(),
        sha256: format!("{:x}", Sha256::digest(output.bytes())),
    })
    .unwrap()
}

#[test]
fn lazy_readback_output_reads_and_clone_equality_do_not_require_digest() {
    let output = output(vec![0x53, 0x07, 0xb1, 0xfd]);
    let cold = output.clone();
    assert!(output.sha256.get().is_none());
    assert!(cold.sha256.get().is_none());
    assert_eq!(output.bytes(), [0x53, 0x07, 0xb1, 0xfd]);
    assert_eq!(output.request().output_layout().byte_len().unwrap(), 4);
    assert_eq!(output.timing(), DeviceTimingMeasurement::NotRequested);
    assert_eq!(output, cold);
    assert!(output.sha256.get().is_none());
    let expected = format!("{:x}", Sha256::digest(output.bytes()));
    assert_eq!(output.sha256(), expected);
    assert!(cold.sha256.get().is_none());
    assert_eq!(output, cold);
    assert!(cold.sha256.get().is_none());
    let warm = output.clone();
    assert_eq!(warm, cold);
    assert_eq!(warm.sha256.get().unwrap(), &expected);
    assert_eq!(cold.into_bytes(), [0x53, 0x07, 0xb1, 0xfd]);
}

#[test]
fn lazy_readback_output_keeps_eager_byte_validation_and_legacy_wire_debug() {
    let output = output(vec![53, 7, 177, 253]);
    assert!(CompletionReadbackOutput::new(
        output.request().clone(),
        LaneReadback {
            bytes: vec![53, 7, 177],
            timing: DeviceTimingMeasurement::NotRequested,
        }
    )
    .is_err());
    assert_eq!(serde_json::to_vec(&output).unwrap(), legacy_json(&output));
    let expected = format!(
        "CompletionReadbackOutput {{ request: {:?}, bytes: {:?}, sha256: {:?}, timing: {:?} }}",
        output.request(),
        output.bytes(),
        format!("{:x}", Sha256::digest(output.bytes())),
        output.timing()
    );
    assert_eq!(format!("{output:?}"), expected);
    let changed = self::output(vec![53, 7, 177, 252]);
    assert_ne!(output.sha256(), changed.sha256());
    assert_ne!(output, changed);
}

#[test]
fn lazy_readback_output_concurrent_digest_and_serialization_match_legacy() {
    let output = Arc::new(output((0..4096).map(|i| (i * 37 + 53) as u8).collect()));
    let expected_json = legacy_json(&output);
    let expected_hash = format!("{:x}", Sha256::digest(output.bytes()));
    let barrier = Arc::new(Barrier::new(4));
    let threads = (0..4)
        .map(|_| {
            let output = Arc::clone(&output);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                (
                    output.sha256().to_owned(),
                    serde_json::to_vec(output.as_ref()).unwrap(),
                )
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        let (hash, json) = thread.join().unwrap();
        assert_eq!(hash, expected_hash);
        assert_eq!(json, expected_json);
    }
}
