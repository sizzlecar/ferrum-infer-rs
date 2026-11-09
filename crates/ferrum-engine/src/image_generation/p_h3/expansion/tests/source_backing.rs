use super::*;

#[test]
fn bounded_source_conversion_workers_preserve_host_and_device_requirements() {
    let mut fixture = Fixture::new();
    let mut expansion = fixture.take_expansion();
    let expansion = Arc::get_mut(&mut expansion).unwrap();
    assert_eq!(
        expansion.trace_identity().unwrap()["source_conversion_workers"],
        1
    );
    let parallel =
        ferrum_quantization::Bf16F16ConversionPolicy::new(std::num::NonZeroUsize::new(4).unwrap())
            .unwrap();
    assert!(expansion
        .configure_source_conversion_policy(parallel)
        .is_err());
    expansion
        .configure_source_backing(PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot)
        .unwrap();
    assert!(expansion
        .configure_source_conversion_policy(parallel)
        .is_err());
    expansion
        .configure_source_backing(PH3ImagePromptExpansionSourceBacking::BoundedStreaming)
        .unwrap();
    let (composition, runtime) = composition();
    let host = host();
    let request = prepared(expansion, 2);
    let serial = expansion
        .host_requirements(&composition, &request, &host)
        .unwrap();
    let expected = (
        serial.claims(),
        serial.startup_device_bytes,
        serial.materializer_additional_bytes(),
    );
    drop(serial);
    expansion
        .configure_source_conversion_policy(parallel)
        .unwrap();
    assert_eq!(
        expansion.trace_identity().unwrap()["source_conversion_workers"],
        4
    );
    let actual = expansion
        .host_requirements(&composition, &request, &host)
        .unwrap();
    assert_eq!(
        (actual.claims(), actual.startup_device_bytes, actual.materializer_additional_bytes()),
        expected,
        "only conversion scheduling changes; the slot and full fallback retain their original charges"
    );
    drop(actual);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
    for incompatible in [
        PH3ImagePromptExpansionSourceBacking::Fresh,
        PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot,
    ] {
        assert!(expansion.configure_source_backing(incompatible).is_err());
        assert_eq!(
            expansion.source_backing,
            PH3ImagePromptExpansionSourceBacking::BoundedStreaming
        );
        assert_eq!(
            expansion.trace_identity().unwrap()["source_conversion_workers"],
            4
        );
    }
    assert!(expansion
        .configure_process_mapped_weights(
            fixture.directory.join("unused-mapped-directory"),
            ferrum_quantization::ProcessMappedDenseLimits {
                maximum_file_bytes: 4 << 20,
                maximum_metadata_bytes: 1 << 20
            },
            ferrum_quantization::ProcessMappedDenseUploadPolicy::default(),
        )
        .is_err());
    expansion
        .configure_source_conversion_policy(Default::default())
        .unwrap();
    expansion
        .configure_source_backing(PH3ImagePromptExpansionSourceBacking::Fresh)
        .unwrap();
    assert_eq!(
        expansion.trace_identity().unwrap()["source_conversion_workers"],
        1
    );
}

#[tokio::test]
async fn reusable_single_slot_expansion_preserves_both_real_initialization_flows_and_budgets() {
    for policy in [
        PH3ImagePromptExpansionSourceBacking::Fresh,
        PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot,
    ] {
        let mut fixture = Fixture::new();
        let mut expansion = fixture.take_expansion();
        let (composition, runtime) = composition();
        let host = host();
        let before = prepared(&expansion, 2);
        let fresh = expansion
            .host_requirements(&composition, &before, &host)
            .unwrap();
        let fresh_claims = fresh.claims();
        let fresh_device = fresh.startup_device_bytes;
        drop(fresh);
        Arc::get_mut(&mut expansion)
            .unwrap()
            .configure_source_backing(policy)
            .unwrap();
        assert_eq!(
            expansion.trace_identity().unwrap()["source_backing"],
            serde_json::to_value(policy).unwrap()
        );
        for owned in [false, true] {
            let prepared = prepared(&expansion, 2);
            let requirements = expansion
                .host_requirements(&composition, &prepared, &host)
                .unwrap();
            assert_eq!(requirements.claims(), fresh_claims);
            assert_eq!(requirements.startup_device_bytes, fresh_device);
            assert_eq!(
                host.usage().unwrap().committed_bytes(),
                0,
                "compiled-plan policy validation allocates no payload or grant"
            );
            assert_eq!(runtime.resident_bytes(), 0);
            let phase = host.reserve_scope(&requirements.claims()).unwrap();
            let phase_host = phase.account();
            let owner = if owned {
                expansion
                    .initialize_owned(&composition, &prepared, phase)
                    .unwrap()
            } else {
                let executor = expansion
                    .initialize(&composition, &prepared, &phase_host)
                    .unwrap();
                let owner = ExecutorOwner::new(Box::new(executor));
                owner.shutdown_observer().retain_until_retired(phase);
                owner
            };
            assert_eq!(
                host.usage()
                    .unwrap()
                    .class_allocated_bytes(HostMemoryClass::WeightBacking),
                0,
                "the local conversion slot ends with installation, not with inference"
            );
            assert!(
                host.usage().unwrap().committed_bytes() > 0,
                "the original phase remains admitted through executor retirement"
            );
            assert!(runtime.resident_bytes() > 0);
            let scope = owner.scope().unwrap();
            let result = execute::execute(
                &scope,
                &RequestId::new(),
                &prepared.tokens,
                prepared.generation,
                prepared.prefill_chunk_tokens,
                &phase_host,
                &|| false,
            )
            .await
            .unwrap();
            assert_eq!(result.history.ids(), &[ANSWER_ID, EOS_ID]);
            assert_eq!(result.finish, FinishReason::EOS);
            assert_eq!(
                expansion
                    .extract(&prepared, result.history.ids(), result.finish)
                    .unwrap(),
                "A small red cat sits quietly."
            );
            drop(result);
            drop(scope);
            owner.begin_shutdown().wait().await.result().unwrap();
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            assert_eq!(runtime.resident_bytes(), 0);
        }
    }
}

#[test]
fn installation_local_and_process_mapped_policies_reject_both_configuration_orders() {
    let limits = ferrum_quantization::ProcessMappedDenseLimits {
        maximum_file_bytes: 4 << 20,
        maximum_metadata_bytes: 1 << 20,
    };
    for (policy, mapped_first) in [
        PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot,
        PH3ImagePromptExpansionSourceBacking::BoundedStreaming,
    ]
    .into_iter()
    .flat_map(|policy| [false, true].map(|mapped_first| (policy, mapped_first)))
    {
        let mut fixture = Fixture::new();
        let directory = fixture.directory.join("empty-mapped-directory");
        std::fs::create_dir(&directory).unwrap();
        let mut assets = fixture.take_expansion();
        let assets = Arc::get_mut(&mut assets).unwrap();
        if mapped_first {
            assets
                .configure_process_mapped_weights(
                    directory.clone(),
                    limits,
                    ferrum_quantization::ProcessMappedDenseUploadPolicy::default(),
                )
                .unwrap();
            assert!(assets.configure_source_backing(policy).is_err());
            assert_eq!(
                assets.source_backing,
                PH3ImagePromptExpansionSourceBacking::Fresh
            );
            assert!(assets.model.process_mapped_weights_enabled());
            assert!(assets
                .configure_source_conversion_policy(
                    ferrum_quantization::Bf16F16ConversionPolicy::new(
                        std::num::NonZeroUsize::new(4).unwrap(),
                    )
                    .unwrap(),
                )
                .is_err());
        } else {
            assets.configure_source_backing(policy).unwrap();
            assert!(assets
                .configure_process_mapped_weights(
                    directory.clone(),
                    limits,
                    ferrum_quantization::ProcessMappedDenseUploadPolicy::default()
                )
                .is_err());
            assert!(!assets.model.process_mapped_weights_enabled());
            assert_eq!(assets.source_backing, policy);
        }
        assert_eq!(
            std::fs::read_dir(&directory).unwrap().count(),
            0,
            "configuration never builds a mapped file"
        );
    }
}

#[test]
fn reusable_single_slot_policy_is_explicit_and_fresh_by_default() {
    assert_eq!(
        PH3ImagePromptExpansionSourceBacking::default(),
        PH3ImagePromptExpansionSourceBacking::Fresh
    );
    for (wire, policy) in [
        ("fresh", PH3ImagePromptExpansionSourceBacking::Fresh),
        (
            "reusable_single_slot",
            PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot,
        ),
        (
            "bounded_streaming",
            PH3ImagePromptExpansionSourceBacking::BoundedStreaming,
        ),
    ] {
        assert_eq!(
            serde_json::from_value::<PH3ImagePromptExpansionSourceBacking>(serde_json::json!(wire))
                .unwrap(),
            policy
        );
    }
    for invalid in [
        serde_json::json!("automatic"),
        serde_json::json!({"kind":"fresh"}),
        serde_json::json!(null),
    ] {
        assert!(serde_json::from_value::<PH3ImagePromptExpansionSourceBacking>(invalid).is_err());
    }
}
