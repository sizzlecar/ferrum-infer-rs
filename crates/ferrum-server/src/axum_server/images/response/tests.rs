use super::*;
use base64::Engine;
use ferrum_output::OutputBudget;
use ferrum_runtime::{ElementType, HostMemoryClassLimits, HostMemoryLimits, ResolvedTensorLayout};
use ferrum_types::OutputConfig;

#[test]
fn exact_json_length_matches_standard_base64_and_signed_timestamp_boundaries() {
    for created in [i64::MIN, -100, -10, -1, 0, 9, 10, 99, 100, i64::MAX] {
        for length in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 63, 64, 65] {
            let payload = vec![0xff; length];
            let base64 = STANDARD.encode(payload);
            let json =
                format!("{{\"created\":{created},\"data\":[{{\"b64_json\":\"{base64}\"}}]}}");
            assert_eq!(response_bytes(length as u64, created), Some(json.len()));
        }
    }
    assert_eq!(response_bytes(u64::MAX, 0), None);
    assert_eq!(response_bytes(isize::MAX as u64, i64::MIN), None);
}

#[test]
fn four_two_k_queue_gates_are_cheap_and_only_the_available_prefix_takes_frames() {
    let host = super::super::tests::host_account(256 * 1024 * 1024);
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 4,
        ..OutputConfig::default()
    })
    .unwrap();
    let gates = (0..4)
        .map(|_| {
            ResponseGate::new(
                budget.try_open_session().unwrap(),
                host.clone(),
                PngDimensions::new(2048, 2048).unwrap(),
                PngLimits::default(),
                64 * 1024,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(budget.usage().sessions, 4);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let first = gates[0].try_admit().unwrap().unwrap();
    let second = gates[1].try_admit().unwrap().unwrap();
    assert!(gates[2].try_admit().unwrap().is_none());
    assert!(gates[3].try_admit().unwrap().is_none());
    assert_eq!(budget.usage().encoded_bytes, first.count + second.count);
    drop(first);
    let third = gates[2].try_admit().unwrap().unwrap();
    drop((second, third, gates));
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn known_host_shortage_does_not_acquire_then_refund_encoded_credit() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Counter(AtomicUsize);
    impl std::task::Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let host = super::super::tests::host_account(4096);
    let occupied = host.reserve(HostMemoryClass::Output, 4096).unwrap();
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let gate = ResponseGate::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        4,
    )
    .unwrap();
    let wakes = Arc::new(Counter(AtomicUsize::new(0)));
    let subscription = gate.subscribe_capacity(std::task::Waker::from(Arc::clone(&wakes)));
    assert!(gate.try_admit().unwrap().is_none());
    assert!(gate.try_admit().unwrap().is_none());
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
    assert_eq!(budget.usage().encoded_bytes, 0);
    drop(occupied);
    assert!(wakes.0.load(Ordering::SeqCst) > 0);
    drop(gate.try_admit().unwrap().unwrap());
    drop(subscription);
}

#[test]
fn retained_input_baseline_cannot_wait_forever_for_nonexistent_output_release() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 4096,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let input = host
        .reserve(HostMemoryClass::InputPreprocess, 4000)
        .unwrap();
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let gate = ResponseGate::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        64,
    )
    .unwrap();
    assert!(matches!(
        gate.try_acquire(),
        Err(ImageOutputGateError::Capacity(_))
    ));
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 4000);
    drop(input);
    assert!(gate.try_acquire().unwrap().is_some());
}

#[test]
fn closed_output_budget_fails_even_while_host_output_is_temporarily_full() {
    let host = super::super::tests::host_account(4096);
    let occupied = host.reserve(HostMemoryClass::Output, 4096).unwrap();
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let gate = ResponseGate::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        4,
    )
    .unwrap();
    assert!(gate.try_acquire().unwrap().is_none());
    budget.close();
    assert!(matches!(
        gate.try_acquire(),
        Err(ImageOutputGateError::Failed(_))
    ));
    drop(occupied);
}

#[tokio::test]
async fn predicted_length_mismatch_cannot_publish_or_retain_output_credit() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 4096,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let raw = host
        .allocate_zeroed(HostMemoryClass::Output, 8)
        .unwrap()
        .freeze();
    let tensor = ResolvedTensorSpec::new(
        vec![1, 4, 1, 1],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    for delta in [-1_isize, 1] {
        let mut prepared = PreparedEncoding::new(
            Admission::new(
                budget.try_open_session().unwrap(),
                host.clone(),
                PngDimensions::new(1, 1).unwrap(),
                PngLimits::default(),
                4,
            )
            .unwrap(),
            tensor.clone(),
            raw.clone(),
        )
        .unwrap();
        // Model a dependency/profile mismatch, without changing the real writer.
        // Both insufficient backing and excess prediction must fail closed.
        prepared.admission.count = prepared.admission.count.checked_add_signed(delta).unwrap();
        assert!(matches!(
            prepared.finish(),
            Err(ServerError::InternalError(_))
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), raw.len());
        assert_eq!(budget.usage().encoded_bytes, 0);
        assert_eq!(budget.usage().sessions, 0);
    }
    drop(raw);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn admitted_encoding_survives_full_parent_and_slow_body_blocks_new_admission() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 4096,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 2,
        ..OutputConfig::default()
    })
    .unwrap();
    let admitted = Admission::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        4,
    )
    .unwrap();
    let count = admitted.count;
    assert_eq!(host.usage().unwrap().committed_bytes(), count + 4);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(budget.usage().encoded_bytes, count);
    // The generator receives only its raw allocation, while unrelated users
    // consume every remaining parent byte before response encoding starts.
    let raw = host
        .allocate_zeroed(HostMemoryClass::Output, 8)
        .unwrap()
        .freeze();
    let tensor = ResolvedTensorSpec::new(
        vec![1, 4, 1, 1],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let occupied = host
        .reserve(
            HostMemoryClass::Output,
            4096 - host.usage().unwrap().committed_bytes(),
        )
        .unwrap();
    assert!(host.reserve(HostMemoryClass::Output, 1).is_err());
    let body = admitted.encode(tensor, raw).await.unwrap();
    assert_eq!(body.len(), count);
    assert_eq!(host.usage().unwrap().allocated_bytes(), count);
    let body_slice = body.slice(..1);
    drop(body);
    assert!(matches!(
        Admission::new(
            budget.try_open_session().unwrap(),
            host.clone(),
            PngDimensions::new(1, 1).unwrap(),
            PngLimits::default(),
            4,
        ),
        Err(ServerError::CapacityExceeded(_))
    ));
    assert_eq!(budget.usage().encoded_bytes, count);
    assert_eq!(budget.usage().sessions, 1);
    drop((body_slice, occupied));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}

#[tokio::test]
async fn unavailable_encoded_credit_does_not_consume_partial_credit_or_host_grant() {
    let host = super::super::tests::host_account(4096);
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 2,
        encoded_bytes_total: 512,
        max_event_bytes: 512,
        ..OutputConfig::default()
    })
    .unwrap();
    let first = budget.try_open_session().unwrap();
    let occupied = first.reserve_encoded(500).await.unwrap();
    assert!(matches!(
        Admission::new(
            budget.try_open_session().unwrap(),
            host.clone(),
            PngDimensions::new(1, 1).unwrap(),
            PngLimits::default(),
            4,
        ),
        Err(ServerError::CapacityExceeded(_))
    ));
    // The remaining 12 bytes stay available; no parked waiter or partial lease
    // survives the rejected response admission.
    assert_eq!(budget.usage().encoded_bytes, 500);
    assert_eq!(budget.usage().sessions, 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop((occupied, first));
    let admitted = Admission::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        4,
    )
    .unwrap();
    drop(admitted);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
