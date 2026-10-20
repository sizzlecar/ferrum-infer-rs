//! Injected completion-state boundaries and separately checked real Metal math.

use super::*;
use metal::foreign_types::ForeignType;

fn injected(status: metal::MTLCommandBufferStatus) -> ObservedMetalCommand {
    ObservedMetalCommand {
        status,
        code: Some(913),
        detail: Some("injected completion observation".to_owned()),
    }
}

fn assert_backend_panic(action: impl FnOnce(), expected: &str) {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .expect_err("failed observation must prevent the host operation");
    let error = payload
        .downcast_ref::<FerrumError>()
        .expect("preserve the backend error for the model frame");
    assert!(error.to_string().contains(expected), "{error}");
}

#[test]
#[ignore = "requires Metal buffers; the terminal failure is an injected state"]
fn metal_injected_failure_blocks_host_access_and_encoding_until_explicit_begin() {
    let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    let table = MetalBackend::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let mut destination = MetalBackend::from_slice(&[31.0, 37.0, 41.0, 43.0]);
    let sentinel = MetalBackend::to_vec(&destination, 4);
    let half_bytes: Vec<_> = [1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(|value| f16::from_f32(value).to_le_bytes())
        .collect();
    let half = MetalBuf {
        raw: buffer_from_f16_bytes(&half_bytes),
        dtype: Dtype::F16,
        n: 4,
    };
    ctx.last_completion = Some(injected(metal::MTLCommandBufferStatus::Error));

    assert_backend_panic(|| MetalBackend::sync(&mut ctx), "injected completion");
    assert_backend_panic(
        || MetalBackend::write_typed(&mut ctx, &mut destination, &[0.0f32; 4]),
        "injected completion",
    );
    assert_backend_panic(
        || {
            let _ = MetalBackend::to_vec_in_context(&mut ctx, &destination, 4);
        },
        "injected completion",
    );
    assert_backend_panic(
        || MetalBackend::embedding_lookup(&mut ctx, &table, &[0], &mut destination, 4),
        "injected completion",
    );
    for logits in [&table, &half] {
        let error = MetalBackend::argmax_rows_f16(&mut ctx, logits, 1, 4).unwrap_err();
        assert!(error.to_string().contains("injected completion"));
    }
    assert_backend_panic(
        || {
            let _ = ctx.compute_encoder();
        },
        "injected completion",
    );
    assert!(ctx.cmd.is_none() && ctx.encoder.is_none() && ctx.submitted.is_none());
    assert_eq!(MetalBackend::to_vec(&destination, 4), sentinel);
    assert_eq!(ctx.last_completion.as_ref().unwrap().code, Some(913));

    MetalBackend::begin_execution(&mut ctx).unwrap();
    MetalBackend::write_typed(&mut ctx, &mut destination, &[5.0f32, 6.0, 7.0, 8.0]);
    assert_eq!(
        MetalBackend::to_vec_in_context(&mut ctx, &destination, 4),
        [5.0, 6.0, 7.0, 8.0]
    );
    assert!(ctx.cmd.is_none() && ctx.submitted.is_none());
    MetalBackend::sync(&mut ctx); // The pristine empty no-op remains valid.
    assert!(MetalBackend::sync_checked(&mut ctx).is_err());

    // The same sticky failure during an outer unwind must not produce a
    // second panic from MetalContext::drop.
    ctx.last_completion = Some(injected(metal::MTLCommandBufferStatus::Error));
    let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _context = ctx;
        std::panic::panic_any("outer unwind");
    }))
    .unwrap_err();
    assert_eq!(outer.downcast_ref::<&str>(), Some(&"outer unwind"));
}

#[test]
#[ignore = "requires Metal execution; unresolved observation is injected after the real native wait"]
fn metal_injected_unresolved_observation_preserves_submission_and_blocks_host_access() {
    let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    let mut values = MetalBackend::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let table = MetalBackend::from_slice(&[11.0, 13.0, 17.0, 19.0]);
    let mut destination = MetalBackend::from_slice(&[31.0, 37.0, 41.0, 43.0]);
    let sentinel = MetalBackend::to_vec(&destination, 4);
    MetalBackend::scale_inplace(&mut ctx, &mut values, 2.0, 4);
    // The GPU actually completes. Only its reported observation is replaced;
    // this exercises retained-state handling, not a genuine driver failure.
    ctx.completion_observation_override = Some(injected(metal::MTLCommandBufferStatus::Committed));
    assert_backend_panic(|| MetalBackend::sync(&mut ctx), "injected completion");
    let original = ctx.submitted.as_ref().unwrap().as_ptr();
    assert!(ctx.cmd.is_none() && ctx.encoder.is_none() && ctx.last_completion.is_none());

    assert_backend_panic(
        || MetalBackend::write_typed(&mut ctx, &mut values, &[0.0f32; 4]),
        "unresolved submission",
    );
    assert_backend_panic(
        || {
            let _ = MetalBackend::to_vec_in_context(&mut ctx, &values, 4);
        },
        "unresolved submission",
    );
    assert_backend_panic(
        || {
            let _ = ctx.compute_encoder();
        },
        "unresolved submission",
    );
    assert!(MetalBackend::argmax_rows_f16(&mut ctx, &values, 1, 4).is_err());
    assert!(MetalBackend::begin_execution(&mut ctx).is_err());

    ctx.completion_observation_override = Some(injected(metal::MTLCommandBufferStatus::Committed));
    assert_backend_panic(
        || MetalBackend::embedding_lookup(&mut ctx, &table, &[0], &mut destination, 4),
        "injected completion",
    );
    assert_eq!(ctx.submitted.as_ref().unwrap().as_ptr(), original);
    assert_eq!(MetalBackend::to_vec(&destination, 4), sentinel);
    // Observe the same real command again, with no injected state. Numerical
    // work is not replayed and the original values must not have been overwritten.
    assert!(matches!(
        ctx.observe_completion(),
        crate::backend::BackendCompletion::Succeeded
    ));
    assert!(ctx.submitted.is_none());
    let actual = MetalBackend::to_vec_in_context(&mut ctx, &values, 4);
    assert!(actual.iter().all(|value| value.is_finite()));
    assert_eq!(actual, [2.0, 4.0, 6.0, 8.0]);
    MetalBackend::retire_context(&mut ctx).unwrap();
}

#[test]
#[ignore = "requires actual Metal GEMV and successful host synchronization"]
fn metal_host_synchronization_preserves_full_outputs_and_successful_continuation() {
    let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    MetalBackend::sync(&mut ctx);
    assert!(MetalBackend::sync_checked(&mut ctx).is_err());
    const N: usize = 8;
    const K: usize = 256;
    let input: Vec<_> = (0..K)
        .map(|i| ((i * 7 % 31) as f32 - 15.0) / 32.0)
        .collect();
    let weights: Vec<_> = (0..N * K)
        .map(|i| ((i * 11 % 43) as f32 - 21.0) / 32.0)
        .collect();
    let expected: Vec<_> = weights
        .chunks_exact(K)
        .map(|row| {
            row.iter()
                .zip(&input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
        })
        .collect();
    let input = MetalBackend::from_slice(&input);
    let weights = MetalBackend::from_slice(&weights);
    let mut output = MetalBackend::from_slice(&[f32::NAN; N]);
    MetalBackend::begin_execution(&mut ctx).unwrap();
    MetalBackend::gemm(&mut ctx, &input, &weights, &mut output, 1, N, K);
    MetalBackend::sync(&mut ctx);
    let actual = MetalBackend::to_vec_in_context(&mut ctx, &output, N);
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (actual - expected).abs() < 1e-4 + 1e-4 * expected.abs(),
            "row={index}: {actual} vs {expected}"
        );
    }
    // A successful intermediate sync may be followed by more encoding in the
    // same invocation. Only a terminal failure requires explicit begin/reset.
    MetalBackend::scale_inplace(&mut ctx, &mut output, 2.0, N);
    MetalBackend::sync(&mut ctx);
    let doubled = MetalBackend::to_vec_in_context(&mut ctx, &output, N);
    for (actual, expected) in doubled.iter().zip(&expected) {
        assert!(actual.is_finite());
        assert!((actual - 2.0 * expected).abs() < 2e-4 + 2e-4 * expected.abs());
    }
    MetalBackend::sync(&mut ctx); // Re-observe the original successful terminal state.
    MetalBackend::retire_context(&mut ctx).unwrap();
}
