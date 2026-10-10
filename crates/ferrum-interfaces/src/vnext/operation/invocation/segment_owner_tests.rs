//! Borrowed indexed facts from actual admitted waves, without simulated CUDA.
use super::segment_hot_tests::{compile, oracle_with_owners};
use super::*;
use crate::vnext::operation::segment_dispatch::encode_segment_wave;
use vnext_device_operation_wave_contract::{setup_with_fixture, teardown, wave_active_bindings};

#[test]
fn segment_owner_views_preserve_ordinary_ranges_order_and_exact_owners() {
    for fixture in [
        fixture_with_retained_dependencies(16, DependencyMode::Valid),
        fixture_with_token_scaled_paged_state_and_provider_behavior(
            ProviderBehavior::ProgramBinding,
        ),
    ] {
        let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture);
        assert_eq!(
            fixture.runtime.segment_binding_owner_view_mode(),
            SegmentBindingOwnerViewMode::Legacy
        );
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active = wave_active_bindings(&wave, &session);
        let identity = super::segment_authority_tests::segment_identity(&fixture, &wave, &active);
        let recipe = compile(&fixture, &wave);
        let (expected_regions, expected_owners) =
            oracle_with_owners(&fixture, &wave, &identity, &active, &recipe);
        assert!(!expected_owners.is_empty());
        let mut previous_scopes: Option<Vec<Arc<()>>> = None;
        for mode in [
            SegmentBindingOwnerViewMode::Legacy,
            SegmentBindingOwnerViewMode::Indexed,
        ] {
            {
                let mut trace = fixture.runtime_trace.lock().unwrap();
                trace.segment_metadata_enabled = true;
                trace.segment_encoder_enabled = true;
                trace.segment_capture_owners = true;
                trace.segment_owner_view_mode = mode;
            }
            let (encoded, facts) = encode_segment_wave(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                &identity,
                &wave,
                active.iter(),
                &recipe,
            )
            .unwrap()
            .unwrap();
            assert!(facts.unique_physical_buffers > 0);
            let mut trace = fixture.runtime_trace.lock().unwrap();
            assert_eq!(trace.segment_regions, expected_regions);
            assert_eq!(trace.segment_retained_owners.len(), expected_owners.len());
            for (actual, expected) in trace.segment_retained_owners.iter().zip(&expected_owners) {
                assert!(actual.same_owners(expected));
            }
            match mode {
                SegmentBindingOwnerViewMode::Legacy => assert_eq!(trace.segment_owner_count, 0),
                SegmentBindingOwnerViewMode::Indexed => assert!(trace.segment_owner_count > 0),
            }
            trace.segment_retained_owners.clear();
            drop(trace);
            if let Some(previous) = &previous_scopes {
                for (old, current) in previous.iter().zip(&encoded) {
                    assert!(!Arc::ptr_eq(old, &current.scope));
                }
            }
            previous_scopes = Some(encoded.iter().map(|node| Arc::clone(&node.scope)).collect());
            drop(encoded);
        }
        drop(previous_scopes);
        drop(expected_owners);
        drop(identity);
        drop(active);
        drop(wave);
        drop(recipe);
        teardown(fixture, sequence, session, batch, step);
    }
}

#[test]
fn segment_owner_views_keep_fresh_capability_and_descriptor_checks() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_retained_dependencies(16, DependencyMode::Valid),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = super::segment_authority_tests::segment_identity(&fixture, &wave, &active);
    let recipe = compile(&fixture, &wave);
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.segment_owner_view_mode = SegmentBindingOwnerViewMode::Indexed;
        trace.segment_encoder_enabled = true;
    }
    let run = || {
        encode_segment_wave(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &recipe,
        )
    };
    assert!(
        run().unwrap().is_none(),
        "absent capability retains fallback"
    );
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.segment_metadata_enabled = true;
        trace.segment_foreign_runtime_metadata = true;
    }
    assert!(run().is_err());
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.segment_foreign_runtime_metadata = false;
        trace.tamper_buffer_descriptor = true;
    }
    assert!(run().is_err(), "borrowed mode cannot skip a current getter");
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = false;
    assert!(run().unwrap().is_some());
    drop(identity);
    drop(active);
    drop(wave);
    drop(recipe);
    teardown(fixture, sequence, session, batch, step);
}
