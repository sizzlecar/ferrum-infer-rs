use super::*;

#[test]
fn public_adapter_family_preparer_preserves_original_program_and_source_for_all_phases() {
    let files = shared_fixture::files(&shared_fixture::config());
    let adapter =
        PH3ImageDenoiserAdapter::new(transformer(), Arc::new(files.source), limits()).unwrap();
    for phase in [
        PH3ImageDenoiserPhase::InstallFullWeights,
        PH3ImageDenoiserPhase::RefineFreshText,
        PH3ImageDenoiserPhase::DenoiseRefinedText,
    ] {
        let prepared = adapter
            .prepare(phase, 3, vec![request(&adapter, "request.family.a")])
            .unwrap();
        let original =
            TypedFamilyRegistration::new(PH3ImageDiTFamily::new(adapter.limits).unwrap())
                .prepare_with_profile(
                    &serde_json::to_value(&prepared.config).unwrap(),
                    &prepared.family().numerical_profile().id,
                )
                .unwrap();
        assert_eq!(prepared.family(), &original);
        // The original provider's inventory order differs from the order core
        // uses in a prepared family. Every physical field must still match.
        let ModelWeights::Stored(mut raw_schema) = PH3ImageDiTFamily::new(adapter.limits)
            .unwrap()
            .weights(&prepared.config)
            .unwrap()
        else {
            panic!("original DiT provider must retain its Stored schema")
        };
        raw_schema.components.sort_by(|a, b| a.id.cmp(&b.id));
        raw_schema.tensors.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(&raw_schema, original.stored_weight_schema().unwrap());
        let rebound = adapter
            .prepare_with_family(
                phase,
                3,
                vec![request(&adapter, "request.family.b")],
                |provider, config, profile| {
                    let raw = serde_json::to_value(config)?;
                    assert_eq!(
                        provider.validated_external_metadata_id(&raw, config)?,
                        *original.external_metadata_id()
                    );
                    assert_eq!(&raw, original.canonical_config());
                    assert_eq!(profile, &original.numerical_profile().id);
                    Ok(original.clone())
                },
            )
            .unwrap();
        assert_eq!(rebound.family(), &original);
        assert_eq!(rebound.compile_options(), prepared.compile_options());
        assert_eq!(rebound.request_ids()[0].as_str(), "request.family.b");
        assert!(std::ptr::eq(rebound.source(), prepared.source()));
    }
}

#[test]
fn public_adapter_family_preparer_rejects_changed_text_basis_phase_and_attention_policy() {
    let files = shared_fixture::files(&shared_fixture::config());
    let mut adapter =
        PH3ImageDenoiserAdapter::new(transformer(), Arc::new(files.source), limits()).unwrap();
    let phase = PH3ImageDenoiserPhase::DenoiseRefinedText;
    let first = adapter
        .prepare(phase, 3, vec![request(&adapter, "request.family.a")])
        .unwrap();
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let reference = PH3ImageReferenceImage::new(2, 4).unwrap();
    let tags = [PH3ImageModality::Text; 4];
    let references = [reference];
    let layout =
        PH3ImageRequestLayout::new(adapter.transformer(), &tags, target, &references).unwrap();
    let mut times = vec![0; layout.rows()];
    times[layout.reference_rows()].fill(2);
    let longer = adapter
        .request(
            RequestIdentity::new("request.family.longer").unwrap(),
            &tags,
            target,
            Some(reference),
            &times,
        )
        .unwrap();
    assert!(adapter
        .prepare_with_family(phase, 3, vec![longer], |_, _, _| Ok(first.family().clone()))
        .is_err());
    assert!(adapter
        .prepare_with_family(
            phase,
            4,
            vec![request(&adapter, "request.family.b")],
            |_, _, _| Ok(first.family().clone())
        )
        .is_err());
    assert!(adapter
        .prepare_with_family(
            PH3ImageDenoiserPhase::RefineFreshText,
            3,
            vec![request(&adapter, "request.family.b")],
            |_, _, _| Ok(first.family().clone())
        )
        .is_err());
    adapter
        .set_attention_policy(PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1)
        .unwrap();
    assert!(adapter
        .prepare_with_family(
            phase,
            3,
            vec![request(&adapter, "request.family.b")],
            |_, _, _| Ok(first.family().clone())
        )
        .is_err());
}
