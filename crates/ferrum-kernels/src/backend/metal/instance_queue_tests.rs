//! Real queue isolation and explicitly completed cross-context buffer handoff.

use super::*;
use metal::foreign_types::ForeignType;

fn assert_full_output(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (actual - expected).abs() <= 1.0e-5 + expected.abs() * 1.0e-5,
            "index={index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
#[ignore = "requires actual Metal queues and numerical execution"]
fn metal_instance_queues_keep_peer_unsubmitted_work_through_retirement() {
    let mut first = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    let mut peer = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    assert!(Arc::ptr_eq(&first.pipes, &peer.pipes));
    assert_ne!(first.queue.as_ptr(), peer.queue.as_ptr());

    const LEN: usize = 257;
    let first_host: Vec<_> = (0..LEN).map(|i| (i as f32 - 128.0) / 16.0).collect();
    let peer_host: Vec<_> = (0..LEN)
        .map(|i| ((i * 7 % 53) as f32 - 26.0) / 8.0)
        .collect();
    let mut first_values = MetalBackend::from_slice_in_context(&mut first, &first_host);
    let mut peer_values = MetalBackend::from_slice_in_context(&mut peer, &peer_host);
    MetalBackend::begin_execution(&mut first).unwrap();
    MetalBackend::begin_execution(&mut peer).unwrap();
    MetalBackend::scale_inplace(&mut first, &mut first_values, -2.0, LEN);
    MetalBackend::scale_inplace(&mut peer, &mut peer_values, 4.0, LEN);
    let peer_command = peer.cmd.as_ref().unwrap().as_ptr();
    assert_eq!(
        peer.cmd.as_ref().unwrap().status(),
        metal::MTLCommandBufferStatus::NotEnqueued
    );
    assert!(peer.encoder.is_some() && peer.submitted.is_none());

    // Retiring A must actually execute A, without committing or replacing B's
    // original still-unsubmitted command or its open encoder.
    MetalBackend::retire_context(&mut first).unwrap();
    assert!(first.cmd.is_none() && first.encoder.is_none() && first.submitted.is_none());
    assert_full_output(
        &MetalBackend::to_vec_in_context(&mut first, &first_values, LEN),
        &first_host
            .iter()
            .map(|value| -2.0 * value)
            .collect::<Vec<_>>(),
    );
    drop(first);
    assert_eq!(peer.cmd.as_ref().unwrap().as_ptr(), peer_command);
    assert_eq!(
        peer.cmd.as_ref().unwrap().status(),
        metal::MTLCommandBufferStatus::NotEnqueued
    );
    assert!(peer.encoder.is_some() && peer.submitted.is_none() && peer.last_completion.is_none());
    // This host read is safe because the native command is still unsubmitted.
    assert_eq!(MetalBackend::to_vec(&peer_values, LEN), peer_host);

    MetalBackend::sync_checked(&mut peer).unwrap();
    assert_full_output(
        &MetalBackend::to_vec_in_context(&mut peer, &peer_values, LEN),
        &peer_host
            .iter()
            .map(|value| 4.0 * value)
            .collect::<Vec<_>>(),
    );
    MetalBackend::begin_execution(&mut peer).unwrap();
    MetalBackend::scale_inplace(&mut peer, &mut peer_values, 0.5, LEN);
    MetalBackend::sync_checked(&mut peer).unwrap();
    assert_full_output(
        &MetalBackend::to_vec_in_context(&mut peer, &peer_values, LEN),
        &peer_host
            .iter()
            .map(|value| 2.0 * value)
            .collect::<Vec<_>>(),
    );
    MetalBackend::retire_context(&mut peer).unwrap();
}

#[test]
#[ignore = "requires actual Metal GEMV, checked completion and cross-queue consumption"]
fn metal_instance_queues_consume_original_buffer_after_checked_handoff() {
    let mut producer = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    let mut consumer = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    assert!(Arc::ptr_eq(&producer.pipes, &consumer.pipes));
    assert_ne!(producer.queue.as_ptr(), consumer.queue.as_ptr());

    const K: usize = 256;
    const N: usize = 16;
    let input: Vec<_> = (0..K)
        .map(|i| ((i * 3 % 29) as f32 - 14.0) / 16.0)
        .collect();
    let weights: Vec<_> = (0..N * K)
        .map(|i| ((i * 5 % 31) as f32 - 15.0) / 32.0)
        .collect();
    let expected: Vec<_> = weights
        .chunks_exact(K)
        .map(|row| {
            row.iter()
                .zip(&input)
                .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                .sum::<f64>() as f32
        })
        .collect();
    let input = MetalBackend::from_slice_in_context(&mut producer, &input);
    let weights = MetalBackend::from_slice_in_context(&mut producer, &weights);
    let mut output = MetalBackend::from_slice_in_context(&mut producer, &[f32::NAN; N]);
    let native_buffer = output.raw.as_ptr();
    MetalBackend::begin_execution(&mut producer).unwrap();
    MetalBackend::gemm(&mut producer, &input, &weights, &mut output, 1, N, K);
    // A successful actual submission observation is the handoff boundary.
    // Sharing a catalog or a device alone would not establish this readiness.
    MetalBackend::sync_checked(&mut producer).unwrap();
    assert!(producer.cmd.is_none() && producer.submitted.is_none());
    assert_full_output(
        &MetalBackend::to_vec_in_context(&mut producer, &output, N),
        &expected,
    );
    MetalBackend::retire_context(&mut producer).unwrap();
    drop(producer);

    // Move the original native allocation; do not reconstruct it from the
    // numerical readback or rely on an implicit cross-queue ordering rule.
    let mut handed_off = output;
    assert_eq!(handed_off.raw.as_ptr(), native_buffer);
    MetalBackend::begin_execution(&mut consumer).unwrap();
    MetalBackend::scale_inplace(&mut consumer, &mut handed_off, 1.5, N);
    MetalBackend::sync_checked(&mut consumer).unwrap();
    assert_eq!(handed_off.raw.as_ptr(), native_buffer);
    assert_full_output(
        &MetalBackend::to_vec_in_context(&mut consumer, &handed_off, N),
        &expected.iter().map(|value| 1.5 * value).collect::<Vec<_>>(),
    );
    MetalBackend::retire_context(&mut consumer).unwrap();
}
