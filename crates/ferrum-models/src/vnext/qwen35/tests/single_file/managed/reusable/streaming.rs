use super::*;

#[test]
fn bounded_source_probe_leaves_fresh_derived_execution_components_unclaimed() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let model = prepared.managed_safetensors().unwrap();
    let parent = host(1 << 18);
    let source = model.weight_source(&parent);
    let original = component(&prepared, "model.language_model.embed_tokens.weight");
    let mut derived = original.clone();
    derived.id = "component.derived.execution".to_owned().try_into().unwrap();
    derived.external_names = vec!["not-a-stored-tensor".into()];
    assert!(source.streaming_component(&derived).unwrap().is_none());
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(
        source.component(original).unwrap().bytes(),
        prepared.weights().component(original).unwrap().bytes()
    );
}

#[test]
fn bounded_source_metadata_counts_slot_plus_simultaneous_fallback_peak() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let model = prepared.managed_safetensors().unwrap();
    let (_runtime, plan) = compile(&prepared);
    let parent = host(1 << 18);
    let mut serial_maximum = None;
    for workers in [None, Some(4)] {
        let source = model.weight_source(&parent);
        let ceiling = parent.backing_bytes(64).unwrap();
        if let Some(workers) = workers {
            source
                .prepare_bounded_streaming_static_backing_with_policy(
                    &plan,
                    ceiling,
                    ferrum_quantization::Bf16F16ConversionPolicy::new(
                        std::num::NonZeroUsize::new(workers).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
        } else {
            // The original public API retains its serial default.
            source
                .prepare_bounded_streaming_static_backing(&plan, ceiling)
                .unwrap();
        }
        let mut maximum_logical = 0;
        let mut full_peak = 0;
        for component in &prepared.family().stored_weight_schema().unwrap().components {
            if let Some(spec) = source.streaming_component(component).unwrap() {
                assert_eq!(spec.component(), component);
                assert_eq!(spec.source_files().len(), component.external_names.len());
                assert!(spec.maximum_range_bytes() <= 64);
                maximum_logical = maximum_logical.max(spec.maximum_range_bytes());
                assert!(
                    source.component(component).is_err(),
                    "armed streams never allocate full payloads"
                );
            } else {
                let bytes = source.component_backing_bytes(component).unwrap() as usize;
                if bytes != 0 {
                    full_peak = full_peak.max(parent.backing_bytes(bytes).unwrap());
                }
            }
        }
        assert!(maximum_logical > 0);
        assert!(
            full_peak > 0,
            "the real fixture retains affine normalization on its original route"
        );
        let maximum = source.maximum_group_backing_bytes(&plan).unwrap();
        if let Some(serial) = serial_maximum {
            assert_eq!(
                maximum, serial,
                "workers cannot add a second source payload"
            );
        } else {
            serial_maximum = Some(maximum);
        }
        assert_eq!(
            maximum,
            parent.backing_bytes(maximum_logical).unwrap() + full_peak
        );
        assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
        assert!(source.prepare_reusable_static_backing(&plan).is_err());
        assert!(source
            .prepare_bounded_streaming_static_backing(&plan, ceiling)
            .is_err());
        assert!(source
            .prepare_bounded_streaming_static_backing_with_policy(
                &plan,
                ceiling,
                ferrum_quantization::Bf16F16ConversionPolicy::default(),
            )
            .is_err());
        // Failed attempts to change an already selected policy do not replace it.
        assert!(source
            .streaming_component(component(
                &prepared,
                "model.language_model.embed_tokens.weight"
            ))
            .unwrap()
            .is_some());
    }
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn bounded_source_failed_preparation_and_prior_fresh_use_never_change_routes() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let model = prepared.managed_safetensors().unwrap();
    let (_runtime, plan) = compile(&prepared);
    let parent = host(1 << 18);
    let component = component(&prepared, "model.language_model.embed_tokens.weight");
    let failed = model.weight_source(&parent);
    assert!(failed
        .prepare_bounded_streaming_static_backing(&plan, 1)
        .is_err());
    assert!(failed.component(component).is_err());
    assert!(failed.maximum_group_backing_bytes(&plan).is_err());
    assert!(failed.prepare_reusable_static_backing(&plan).is_err());
    let fresh = model.weight_source(&parent);
    drop(fresh.component(component).unwrap());
    assert!(fresh
        .prepare_bounded_streaming_static_backing(&plan, 4096)
        .is_err());
    drop(fresh.component(component).unwrap());
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn bounded_source_actual_cpu_staging_route_is_rejected_before_payload_allocation() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let model = prepared.managed_safetensors().unwrap();
    let (runtime, plan) = compile(&prepared);
    assert!(!runtime.supports_managed_static_weight_sources());
    let parent = host(1 << 18);
    let source = model.weight_source(&parent);
    source
        .prepare_bounded_streaming_static_backing(&plan, 4096)
        .unwrap();
    let failure = match committed(&plan, runtime).initialize_static_managed(
        prepared.family(),
        &plan,
        &source,
        &parent,
        policy(),
    ) {
        Ok(_) => panic!("CPU staging cannot impersonate direct managed source upload"),
        Err(failure) => failure,
    };
    assert!(!failure.is_indeterminate());
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    assert!(source
        .component(component(
            &prepared,
            "model.language_model.embed_tokens.weight"
        ))
        .is_err());
    drop(failure.recover().unwrap());
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}
