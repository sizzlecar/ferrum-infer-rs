//! Actual admitted views are the oracle for the new whole-segment table.
//! TestRuntime records typed facts; it does not simulate CUDA computation.
use super::*;
use crate::vnext::operation::segment_compile::CompiledSegmentBindingRecipe;
use crate::vnext::operation::segment_dispatch::{
    encode_segment_wave, encode_segment_wave_with_timing,
};
use vnext_device_operation_wave_contract::{setup_with_fixture, teardown, wave_active_bindings};

#[derive(Default)]
struct SegmentTiming(std::sync::Mutex<Vec<SubmissionWaveDispatchStage>>);

impl SegmentTiming {
    fn take(&self) -> Vec<SubmissionWaveDispatchStage> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl DeviceSubmissionTimingSink for SegmentTiming {
    const ENABLED: bool = true;

    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("segment preparation does not submit device commands");
    }
}

impl SubmissionWaveDispatchTimingSink for SegmentTiming {
    fn record(&self, stage: SubmissionWaveDispatchStage, _: std::time::Duration) {
        self.0.lock().unwrap().push(stage);
    }
}

const SEGMENT_PHASES: [SubmissionWaveDispatchStage; 10] = [
    SubmissionWaveDispatchStage::SegmentFreshAuthorityAndWindows,
    SubmissionWaveDispatchStage::SegmentBackingSnapshotLookup,
    SubmissionWaveDispatchStage::SegmentBackingDedupReserveAndPoolResolution,
    SubmissionWaveDispatchStage::SegmentBackingLockAcquisition,
    SubmissionWaveDispatchStage::SegmentBackingLockedValidation,
    SubmissionWaveDispatchStage::SegmentBackingWindowIntersections,
    SubmissionWaveDispatchStage::SegmentBackingImmutableMetadataValidation,
    SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata,
    SubmissionWaveDispatchStage::SegmentNodeDependenciesAndRegions,
    SubmissionWaveDispatchStage::SegmentBackendEncodeAndValidate,
];

fn compile(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
) -> CompiledSegmentBindingRecipe {
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let program = OperationDispatch::reusable_execution_program_id_for_wave(
        providers.providers(),
        &fixture.resolved,
        wave,
        wave.step_resources().execution_lane(),
    )
    .unwrap()
    .unwrap();
    let declarations = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let mut regions = node
                .values()
                .iter()
                .filter(|value| value.usage() == BufferUsage::State)
                .map(|value| SegmentBindingRegionRequest {
                    selector: SegmentBindingRegionSelector::Value {
                        role: value.role(),
                        ordinal: value.ordinal(),
                        component: None,
                    },
                    offset_bytes: 0,
                    extent: SegmentBindingRegionExtent::CurrentResource,
                    element_type: value.storage().components()[0].element_type(),
                    alignment_bytes: 1,
                })
                .collect::<Vec<_>>();
            for (resource, selector) in [
                (
                    node.binding_resource(),
                    SegmentBindingRegionSelector::ProgramBinding,
                ),
                (
                    node.persistent_resource(),
                    SegmentBindingRegionSelector::Persistent,
                ),
            ] {
                if resource.is_some() {
                    regions.push(SegmentBindingRegionRequest {
                        selector,
                        offset_bytes: 0,
                        extent: SegmentBindingRegionExtent::Exact(16),
                        element_type: ElementType::U8,
                        alignment_bytes: 1,
                    });
                }
            }
            assert!(!regions.is_empty());
            (
                index,
                SegmentBindingDeclaration::new(regions, vec![], Arc::new(())).unwrap(),
            )
        })
        .collect();
    CompiledSegmentBindingRecipe::compile(&fixture.resolved, program, declarations).unwrap()
}

fn oracle(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    identity: &BatchOperationIdentity,
    active: &[TrustedActiveSequenceBinding],
    recipe: &CompiledSegmentBindingRecipe,
) -> Vec<SegmentTestRegion> {
    let mut result = Vec::new();
    for compiled in &recipe.nodes {
        let provider = fixture
            .registry
            .bind(
                &fixture.resolved,
                wave.nodes()[compiled.node_index].node_id(),
            )
            .unwrap();
        let node = identity.materialize_node(compiled.node_index).unwrap();
        let invocation = BatchedOperationInvocation::from_resources(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            identity,
            node,
            OperationInvocationResources::Wave {
                wave,
                node_index: compiled.node_index,
            },
            active.iter(),
            false,
            true,
        )
        .unwrap();
        for (participant, invocation) in invocation.participants().iter().enumerate() {
            for (region_index, region) in compiled.regions.iter().enumerate() {
                let resource = &recipe.resources[region.resource_index].resource_id;
                let view = invocation
                    .views()
                    .iter()
                    .find(|view| view.resource_id() == resource)
                    .unwrap();
                let length = region
                    .length_for_current_view(view.descriptor().size_bytes)
                    .unwrap();
                let mut descriptor = view.descriptor().clone();
                descriptor.size_bytes = length;
                let physical = view
                    .translate(region.offset_bytes, length)
                    .unwrap()
                    .iter()
                    .map(|part| {
                        let (buffer, range, _retention) = part.buffer_and_physical_range();
                        (
                            buffer as *const TestBuffer as usize,
                            range,
                            part.logical_offset_bytes(),
                        )
                    })
                    .collect();
                result.push(SegmentTestRegion {
                    node: compiled.node_index as u32,
                    participant,
                    region: region_index,
                    descriptor,
                    physical,
                });
            }
        }
    }
    result
}

#[test]
fn segment_hot_matches_real_state_binding_and_plan_workspace_views_and_fresh_scopes() {
    for fixture in [
        fixture_with_retained_dependencies(16, DependencyMode::Valid),
        fixture_with_token_scaled_paged_state_and_provider_behavior(
            ProviderBehavior::ProgramBinding,
        ),
    ] {
        let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture);
        let lane = Arc::clone(step.execution_lane());
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active = wave_active_bindings(&wave, &session);
        let identity = super::segment_authority_tests::segment_identity(&fixture, &wave, &active);
        let recipe = compile(&fixture, &wave);
        let expected = oracle(&fixture, &wave, &identity, &active, &recipe);
        {
            let mut trace = fixture.runtime_trace.lock().unwrap();
            trace.segment_metadata_enabled = true;
            trace.segment_encoder_enabled = true;
        }
        let timing = SegmentTiming::default();
        let (first, facts) = encode_segment_wave_with_timing(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &recipe,
            &timing,
        )
        .unwrap()
        .unwrap();
        assert_eq!(timing.take(), SEGMENT_PHASES);
        assert_eq!(
            fixture.runtime_trace.lock().unwrap().segment_regions,
            expected
        );
        assert_eq!(
            first.iter().map(|n| n.node_index).collect::<Vec<_>>(),
            recipe
                .nodes
                .iter()
                .map(|n| n.node_index)
                .collect::<Vec<_>>()
        );
        assert!(facts.dynamic_resource_requests > 0 && facts.unique_physical_buffers > 0);
        let (second, _) = encode_segment_wave(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &recipe,
        )
        .unwrap()
        .unwrap();
        for (a, b) in first.iter().zip(&second) {
            assert!(!Arc::ptr_eq(&a.scope, &b.scope));
        }
        drop(first);
        drop(second);
        drop(identity);
        drop(active);
        drop(wave);
        step.try_retire_normal().unwrap();
        let step = begin_single_participant_step_on_lane_with_bucket(
            &batch,
            &lane,
            fixture.reusable_execution_bucket.as_ref(),
        );
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active = wave_active_bindings(&wave, &session);
        let identity = super::segment_authority_tests::segment_identity(&fixture, &wave, &active);
        let expected = oracle(&fixture, &wave, &identity, &active, &recipe);
        let (fresh, _) = encode_segment_wave(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &recipe,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            fixture.runtime_trace.lock().unwrap().segment_regions,
            expected
        );
        drop(fresh);
        drop(identity);
        drop(active);
        drop(wave);
        drop(recipe);
        drop(lane);
        teardown(fixture, sequence, session, batch, step);
    }
}

#[test]
fn segment_hot_missing_capability_falls_back_but_foreign_capability_or_getter_drift_rejects() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_retained_dependencies(16, DependencyMode::Valid),
    );
    let lane = Arc::clone(step.execution_lane());
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = super::segment_authority_tests::segment_identity(&fixture, &wave, &active);
    let recipe = compile(&fixture, &wave);
    let timing = SegmentTiming::default();
    let run = || {
        encode_segment_wave_with_timing(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &recipe,
            &timing,
        )
    };
    assert!(run().unwrap().is_none());
    assert_eq!(timing.take(), SEGMENT_PHASES[..1]);
    assert!(!oracle(&fixture, &wave, &identity, &active, &recipe).is_empty());
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_metadata_enabled = true;
    assert!(
        run().unwrap().is_none(),
        "unsupported encoder retains old path"
    );
    assert_eq!(timing.take(), SEGMENT_PHASES);
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_encoder_enabled = true;
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_foreign_runtime_metadata = true;
    assert!(run().is_err());
    assert_eq!(timing.take(), SEGMENT_PHASES[..1]);
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_foreign_runtime_metadata = false;
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = true;
    assert!(
        run().is_err(),
        "Some capability cannot hide the actual getter"
    );
    assert_eq!(timing.take(), SEGMENT_PHASES[..8]);
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = false;
    assert!(run().unwrap().is_some());
    assert_eq!(timing.take(), SEGMENT_PHASES);
    drop(identity);
    drop(active);
    drop(wave);
    drop(recipe);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn segment_oracle_disabled_skips_reference_and_enabled_unsupported_comparison_rejects() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_token_scaled_paged_state_and_provider_behavior(
            ProviderBehavior::ProgramBinding,
        ));
    let lane = Arc::clone(step.execution_lane());
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = super::segment_authority_tests::segment_identity(&fixture, &wave, &active);
    let recipe = compile(&fixture, &wave);
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.segment_metadata_enabled = true;
        trace.segment_encoder_enabled = true;
    }
    let (nodes, _) = encode_segment_wave(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        &identity,
        &wave,
        active.iter(),
        &recipe,
    )
    .unwrap()
    .unwrap();
    let encoded = nodes
        .into_iter()
        .map(|node| (node.node_index, node))
        .collect::<BTreeMap<_, _>>();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let audit = || {
        crate::vnext::operation::segment_oracle::audit_segment_wave(
            fixture.runtime.as_ref(),
            providers.providers(),
            &fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &encoded,
        )
    };
    let before = fixture
        .provider_trace
        .lock()
        .unwrap()
        .reusable_binding_encode_calls;
    audit().unwrap();
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_binding_encode_calls,
        before
    );
    fixture.runtime_trace.lock().unwrap().segment_oracle_enabled = true;
    // TestRuntime intentionally inherits the default comparator returning None.
    assert!(matches!(
        audit(),
        Err(SubmissionWaveDispatchError::Contract(_))
    ));
    assert!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_binding_encode_calls
            > before
    );
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    drop(providers);
    drop(encoded);
    drop(recipe);
    drop(identity);
    drop(active);
    drop(wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
