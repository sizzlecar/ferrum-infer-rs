use super::*;
use base64::Engine;
use ferrum_models::image_pixels::ImagePixelEncoding;
use ferrum_output::OutputBudget;
use ferrum_runtime::{
    ElementType, HostMemoryClassLimits, HostMemoryLimits, ResolvedTensorLayout, ResolvedTensorSpec,
};
use ferrum_types::OutputConfig;

#[tokio::test]
async fn f32_rgb_artifact_encodes_exact_png_and_retains_transport_owner() {
    let host = super::super::tests::host_account(4096);
    let scope = host
        .reserve_scope(&[(HostMemoryClass::Output, 24)])
        .unwrap();
    let mut raw = scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 24)
        .unwrap();
    for (dst, value) in raw
        .as_mut()
        .chunks_exact_mut(4)
        .zip([0.0_f32, 1.0, 0.5, 0.0, 1.0, 0.5])
    {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    let pixels = OwnedImagePixels::new(
        &host,
        ResolvedTensorSpec::new(
            vec![1, 3, 1, 2],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        raw.freeze(),
        ImagePixelEncoding::F32UnitRgbRne,
    )
    .unwrap();
    drop(scope);
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let admitted = Admission::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(2, 1).unwrap(),
        PngLimits::default(),
        3,
    )
    .unwrap();
    let body = admitted.encode(pixels).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let png = STANDARD
        .decode(json["data"][0]["b64_json"].as_str().unwrap())
        .unwrap();
    let mut reader = png::Decoder::new(Cursor::new(png)).read_info().unwrap();
    let mut decoded = [0; 8];
    let info = reader.next_frame(&mut decoded).unwrap();
    assert_eq!(
        (info.width, info.height, info.color_type),
        (2, 1, png::ColorType::Rgba)
    );
    assert_eq!(decoded, [0, 128, 255, 255, 255, 0, 128, 255]);
    assert_eq!(host.usage().unwrap().allocated_bytes(), body.len());
    assert_eq!(budget.usage().encoded_bytes, body.len());
    let retained = body.slice(..1);
    drop(body);
    assert_ne!(budget.usage().encoded_bytes, 0);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}

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
    let host = super::super::tests::host_account(256 * 1024 * 1024);
    let config = super::super::tests::config();
    let dimensions = PngDimensions::new(2048, 2048).unwrap();
    let budget = OutputBudget::new(config.output.clone()).unwrap();
    let gate = ResponseGate::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        dimensions,
        config.png_limits,
        config.rgba_scratch_bytes,
        0,
    )
    .unwrap();
    let declared = config.response_backing_bytes(&host, dimensions).unwrap();
    assert!(declared >= gate.physical_bytes);
    assert_eq!(
        declared,
        physical_response_bytes(
            &host,
            config.rgba_scratch_bytes,
            response_bytes(
                encoded_rgba8_png_bytes(dimensions, config.png_limits).unwrap(),
                i64::MIN
            )
            .unwrap()
        )
        .unwrap()
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn four_two_k_jobs_reserve_host_without_consuming_encoded_transport_credit() {
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
                0,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(budget.usage().sessions, 4);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let admitted = gates
        .iter()
        .map(|gate| gate.try_admit().unwrap().unwrap())
        .collect::<Vec<_>>();
    assert!(admitted.iter().map(|a| a.count).sum::<usize>() > budget.config().encoded_bytes_total);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        gates.iter().map(|gate| gate.physical_bytes).sum::<usize>()
    );
    drop((admitted, gates));
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
        0,
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
        0,
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
        0,
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

fn aligned_output_host(total: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: total,
                output_bytes: total,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}

#[test]
fn retained_output_baseline_hard_limit_is_exact_and_the_actual_frame_is_charged_once() {
    let dimensions = PngDimensions::new(1, 1).unwrap();
    let probe_host = aligned_output_host(4096);
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let probe = ResponseGate::new(
        budget.try_open_session().unwrap(),
        probe_host.clone(),
        dimensions,
        PngLimits::default(),
        4,
        0,
    )
    .unwrap();
    let frame = probe.physical_bytes;
    let fixed_bytes = probe_host.backing_bytes(17).unwrap();
    drop(probe);
    for shortfall in [0, 1] {
        let host = aligned_output_host(frame + fixed_bytes - shortfall);
        let fixed = host
            .allocate_zeroed(HostMemoryClass::Output, 17)
            .unwrap()
            .freeze();
        assert_eq!(fixed.backing_bytes(), fixed_bytes);
        let gate = ResponseGate::new(
            budget.try_open_session().unwrap(),
            host.clone(),
            dimensions,
            PngLimits::default(),
            4,
            fixed.backing_bytes() as u64,
        );
        if shortfall == 1 {
            assert!(matches!(gate, Err(ServerError::CapacityExceeded(_))));
            assert_eq!(host.usage().unwrap().committed_bytes(), fixed_bytes);
        } else {
            let gate = gate.unwrap();
            assert_eq!(gate.physical_bytes, frame);
            assert_eq!(
                gate.host_output_bytes(&host),
                frame,
                "the output gate reports only the new frame, not the already owned baseline"
            );
            let admission = gate.try_acquire().unwrap().unwrap();
            assert_eq!(host.usage().unwrap().committed_bytes(), fixed_bytes + frame);
            drop(admission);
            drop(gate);
            assert_eq!(host.usage().unwrap().committed_bytes(), fixed_bytes);
        }
        drop(fixed);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(budget.usage().encoded_bytes, 0);
}

#[test]
fn retained_output_baseline_cannot_be_treated_as_a_future_external_refund() {
    let host = aligned_output_host(4096);
    let fixed = host
        .allocate_zeroed(HostMemoryClass::Output, 17)
        .unwrap()
        .freeze();
    let fixed_bytes = fixed.backing_bytes();
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let gate = ResponseGate::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        4,
        fixed_bytes as u64,
    )
    .unwrap();
    // With no externally owned Output at all, retained Input makes the real
    // requirement one byte too large. Pretending the fixed trace will drop
    // would incorrectly return Pending with no possible capacity notification.
    let input = host
        .reserve_scope(&[(
            HostMemoryClass::InputPreprocess,
            4096 - fixed_bytes - gate.physical_bytes + 1,
        )])
        .unwrap();
    let committed = host.usage().unwrap().committed_bytes();
    assert!(matches!(
        gate.try_acquire(),
        Err(ImageOutputGateError::Capacity(_))
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), committed);
    assert_eq!(budget.usage().encoded_bytes, 0);
    drop(input);
    let admission = gate.try_acquire().unwrap().unwrap();
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        fixed_bytes + gate.physical_bytes
    );
    drop((admission, gate, fixed));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn retained_output_baseline_still_allows_waiting_for_the_last_external_output_owner() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Wakes(AtomicUsize);
    impl std::task::Wake for Wakes {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let host = aligned_output_host(4096);
    let fixed = host
        .allocate_zeroed(HostMemoryClass::Output, 17)
        .unwrap()
        .freeze();
    let fixed_bytes = fixed.backing_bytes();
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let gate = ResponseGate::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        4,
        fixed_bytes as u64,
    )
    .unwrap();
    let external = host
        .allocate_zeroed(
            HostMemoryClass::Output,
            4096 - fixed_bytes - gate.physical_bytes + 1,
        )
        .unwrap()
        .freeze();
    let last_owner = external.clone();
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let subscription = gate.subscribe_capacity(std::task::Waker::from(wakes.clone()));
    let committed = host.usage().unwrap().committed_bytes();
    assert!(gate.try_acquire().unwrap().is_none());
    assert!(gate.try_acquire().unwrap().is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), committed);
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
    drop(external);
    assert!(gate.try_acquire().unwrap().is_none());
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
    drop(last_owner);
    assert!(wakes.0.load(Ordering::SeqCst) > 0);
    let admission = gate.try_acquire().unwrap().unwrap();
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        fixed_bytes + gate.physical_bytes
    );
    assert_eq!(budget.usage().encoded_bytes, 0);
    drop((admission, gate, subscription, fixed));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(budget.usage().sessions, 0);
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
        let admission = Admission::new(
            budget.try_open_session().unwrap(),
            host.clone(),
            PngDimensions::new(1, 1).unwrap(),
            PngLimits::default(),
            4,
        )
        .unwrap();
        let lease = admission
            .session
            .reserve_encoded(admission.count)
            .await
            .unwrap();
        let pixels = OwnedImagePixels::new(
            &host,
            tensor.clone(),
            raw.clone(),
            ImagePixelEncoding::Bf16SignedRgbaStaged,
        )
        .unwrap();
        let mut prepared = PreparedEncoding::new(admission, lease, pixels).unwrap();
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
    assert_eq!(budget.usage().encoded_bytes, 0);
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
    let pixels =
        OwnedImagePixels::new(&host, tensor, raw, ImagePixelEncoding::Bf16SignedRgbaStaged)
            .unwrap();
    let body = admitted.encode(pixels).await.unwrap();
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
async fn cancelling_encoding_wait_returns_raw_host_scope_and_partial_transport_credit() {
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
    let admitted = Admission::new(
        budget.try_open_session().unwrap(),
        host.clone(),
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        4,
    )
    .unwrap();
    let count = admitted.count;
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
    let pixels =
        OwnedImagePixels::new(&host, tensor, raw, ImagePixelEncoding::Bf16SignedRgbaStaged)
            .unwrap();
    let mut encoding = Box::pin(admitted.encode(pixels));
    assert!(futures::poll!(encoding.as_mut()).is_pending());
    assert_eq!(host.usage().unwrap().committed_bytes(), count + 4 + 8);
    assert_eq!(budget.usage().sessions, 2);
    drop(encoding);
    // Cancelling the response wait releases its Host grant, raw output and
    // any partial FIFO semaphore credit; the unrelated owner remains intact.
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
