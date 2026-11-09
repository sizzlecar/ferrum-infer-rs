use super::*;

#[tokio::test]
async fn process_mapped_expansion_reuses_exact_weights_across_both_initialization_flows() {
    check_initialization_flows(
        ferrum_quantization::ProcessMappedDenseUploadPolicy::ReadOnlyMapping,
    )
    .await;
}

#[tokio::test]
async fn process_mapped_staged_expansion_reuses_exact_weights_across_both_initialization_flows() {
    check_initialization_flows(ferrum_quantization::ProcessMappedDenseUploadPolicy::Staged).await;
}

async fn check_initialization_flows(
    upload_policy: ferrum_quantization::ProcessMappedDenseUploadPolicy,
) {
    let mut fixture = Fixture::new();
    let directory = fixture.directory.join("normalized-cache");
    std::fs::create_dir(&directory).unwrap();
    let mut expansion = fixture.take_expansion();
    Arc::get_mut(&mut expansion)
        .unwrap()
        .configure_process_mapped_weights(
            directory.clone(),
            ferrum_quantization::ProcessMappedDenseLimits {
                maximum_file_bytes: 4 << 20,
                maximum_metadata_bytes: 1 << 20,
            },
            upload_policy,
        )
        .unwrap();
    let (composition, runtime) = composition();
    let host = host();
    for owned_flow in [false, true] {
        let prepared = prepared(&expansion, 2);
        let requirements = expansion
            .host_requirements(&composition, &prepared, &host)
            .unwrap();
        let phase = host.reserve_scope(&requirements.claims()).unwrap();
        let phase_host = phase.account();
        let owner = if owned_flow {
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
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(runtime.resident_bytes(), 0);
        if !owned_flow {
            // The next request must use the retained private map, even after
            // the first phase is fully retired and no cache directory remains.
            std::fs::remove_dir(&directory).unwrap();
        }
    }
}
