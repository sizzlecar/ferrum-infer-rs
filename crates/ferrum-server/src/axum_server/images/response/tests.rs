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
            budget.try_open_session().unwrap(),
            host.clone(),
            tensor.clone(),
            raw.clone(),
            PngLimits::default(),
            4,
        )
        .unwrap();
        // Model a dependency/profile mismatch, without changing the real writer.
        // Both insufficient backing and excess prediction must fail closed.
        prepared.count = prepared.count.checked_add_signed(delta).unwrap();
        let lease = prepared
            .session
            .reserve_encoded(prepared.count)
            .await
            .unwrap();
        assert!(matches!(
            prepared.finish(lease),
            Err(ServerError::InternalError(_))
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), raw.len());
        assert_eq!(budget.usage().encoded_bytes, 0);
        assert_eq!(budget.usage().sessions, 0);
    }
    drop(raw);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
