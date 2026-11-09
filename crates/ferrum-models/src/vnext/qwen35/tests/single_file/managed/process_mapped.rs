use super::*;
use ferrum_quantization::{ProcessMappedDenseLimits, ProcessMappedDenseUploadPolicy};

fn limits() -> ProcessMappedDenseLimits {
    ProcessMappedDenseLimits {
        maximum_file_bytes: 1 << 20,
        maximum_metadata_bytes: 1 << 18,
    }
}

#[test]
fn process_mapped_model_preserves_every_representation_and_outlives_the_build_scope() {
    preserves_representation_and_outlives_scope(ProcessMappedDenseUploadPolicy::ReadOnlyMapping);
}

#[test]
fn process_mapped_staged_model_preserves_bytes_without_direct_upload_authority() {
    preserves_representation_and_outlives_scope(ProcessMappedDenseUploadPolicy::Staged);
}

fn preserves_representation_and_outlives_scope(upload_policy: ProcessMappedDenseUploadPolicy) {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mut mapped = prepared.managed_safetensors().unwrap();
    let directory = tempfile::tempdir().unwrap();
    assert_eq!(mapped.process_mapped_upload_policy(), None);
    match upload_policy {
        ProcessMappedDenseUploadPolicy::ReadOnlyMapping => mapped
            .configure_process_mapped_weights(directory.path().to_owned(), limits())
            .unwrap(),
        ProcessMappedDenseUploadPolicy::Staged => mapped
            .configure_process_mapped_weights_with_upload_policy(
                directory.path().to_owned(),
                limits(),
                upload_policy,
            )
            .unwrap(),
    }
    assert_eq!(mapped.process_mapped_upload_policy(), Some(upload_policy));
    let parent = host(1 << 18);
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 1 << 18)])
        .unwrap();
    let account = scope.account();
    let components = &prepared.family().stored_weight_schema().unwrap().components;
    {
        let source = mapped.weight_source(&account);
        assert!(source.component(&components[0]).is_err());
        source.prepare_process_mapped_weights().unwrap();
        for spec in components {
            let actual = source.component(spec).unwrap();
            let expected = prepared.weights().component(spec).unwrap();
            assert_eq!(actual.bytes(), expected.bytes(), "{}", spec.id);
            assert_eq!(actual.element_type(), expected.element_type());
            assert_eq!(actual.external_names(), expected.external_names());
            assert_eq!(actual.source_files(), expected.source_files());
            assert_eq!(actual.dimensions(), expected.dimensions());
            let region = actual.retained_host_memory().unwrap();
            assert_eq!(
                region.readonly_file_mapping().is_some(),
                upload_policy == ProcessMappedDenseUploadPolicy::ReadOnlyMapping
            );
            assert!(region.managed_memory_class().is_none());
        }
        assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    }
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    drop(account);
    // The anonymous file has no pathname that another request can reopen or
    // truncate, and no builder Host scope survives in its mapping owner.
    directory.close().unwrap();
    let sources = Arc::clone(prepared.sources());
    let reprofiled = Arc::new(
        crate::vnext::resolve_registered_model_from_sources(&sources)
            .unwrap()
            .into_required()
            .unwrap()
            .define_from_sources(sources)
            .unwrap()
            .prepare(&F32_MASTER_NUMERICAL_PROFILE_ID.parse().unwrap())
            .unwrap(),
    );
    let rebound = mapped.with_prepared(reprofiled).unwrap();
    assert_eq!(rebound.process_mapped_upload_policy(), Some(upload_policy));
    let no_backing = host(0);
    let owner = {
        let source = rebound.weight_source(&no_backing);
        source.prepare_process_mapped_weights().unwrap();
        let actual = source.component(&components[0]).unwrap();
        // Prediction stays at the conservative cold requirement, even though
        // this ready source needs no managed conversion allocation.
        assert!(source.component_backing_bytes(&components[0]).unwrap() > 0);
        actual.retained_host_memory().unwrap().clone()
    };
    let expected = prepared
        .weights()
        .component(&components[0])
        .unwrap()
        .bytes()
        .to_vec();
    drop(rebound);
    drop(mapped);
    assert_eq!(owner.bytes(), expected);
    assert_eq!(
        owner.readonly_file_mapping().is_some(),
        upload_policy == ProcessMappedDenseUploadPolicy::ReadOnlyMapping
    );
    assert!(owner.managed_memory_class().is_none());
    assert_eq!(no_backing.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn process_mapped_model_failed_build_publishes_nothing_and_can_retry_with_real_capacity() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mut mapped = prepared.managed_safetensors().unwrap();
    let directory = tempfile::tempdir().unwrap();
    mapped
        .configure_process_mapped_weights(directory.path().to_owned(), limits())
        .unwrap();
    let denied = host(0);
    let spec = component(&prepared, "model.language_model.embed_tokens.weight");
    let source = mapped.weight_source(&denied);
    assert!(source.prepare_process_mapped_weights().is_err());
    assert!(source.component(spec).is_err());
    assert_eq!(denied.usage().unwrap().committed_bytes(), 0);
    drop(source);
    let granted = host(1 << 18);
    let source = mapped.weight_source(&granted);
    source.prepare_process_mapped_weights().unwrap();
    assert!(source.component(spec).is_ok());
    let mut foreign = spec.clone();
    foreign.external_names[0] = "lm_head.weight".to_owned();
    assert!(source.component(&foreign).is_err());
    assert_eq!(granted.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn process_mapped_model_sharing_requires_the_same_verified_mapping_and_physical_schema() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mut mapped = prepared.managed_safetensors().unwrap();
    let mut same_source = mapped.with_prepared(Arc::clone(&prepared)).unwrap();
    let mut separate_mapping = prepared.managed_safetensors().unwrap();
    let directory = tempfile::tempdir().unwrap();
    mapped
        .configure_process_mapped_weights_with_upload_policy(
            directory.path().to_owned(),
            limits(),
            ProcessMappedDenseUploadPolicy::Staged,
        )
        .unwrap();
    same_source
        .share_process_mapped_weights_from(&mapped)
        .unwrap();
    assert_eq!(
        same_source.process_mapped_upload_policy(),
        Some(ProcessMappedDenseUploadPolicy::Staged)
    );
    assert!(separate_mapping
        .share_process_mapped_weights_from(&mapped)
        .is_err());
    let actual_host = host(1 << 18);
    mapped
        .weight_source(&actual_host)
        .prepare_process_mapped_weights()
        .unwrap();
    directory.close().unwrap();
    let no_backing = host(0);
    let source = same_source.weight_source(&no_backing);
    source.prepare_process_mapped_weights().unwrap();
    let actual = source
        .component(component(&prepared, "model.language_model.norm.weight"))
        .unwrap();
    assert!(actual
        .retained_host_memory()
        .unwrap()
        .readonly_file_mapping()
        .is_none());
    assert_eq!(
        actual.bytes(),
        prepared
            .weights()
            .component(component(&prepared, "model.language_model.norm.weight"))
            .unwrap()
            .bytes()
    );
}
