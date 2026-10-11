//! Real admitted Plan observations for the segment dependency issuer.
use super::*;
use crate::vnext::operation::retained_dependency::{
    append_dependencies, issue_segment_plan_dependency, SegmentPlanDependencyInput,
};
use crate::vnext::resource::{SegmentPlanResourceRequest, SegmentPlanResourceView};
use vnext_device_operation_contract::fixture_with_padded_persistent;

fn plan_views<'a>(
    fixture: &Fixture,
    wave: &'a PreparedStepSubmissionWave<TestRuntime>,
) -> Vec<SegmentPlanResourceView<'a, TestBuffer>> {
    let requests = fixture
        .plan
        .payload()
        .memory()
        .static_allocations()
        .iter()
        .enumerate()
        .map(|(slot_index, allocation)| SegmentPlanResourceRequest {
            slot_index,
            allocation,
        })
        .collect::<Vec<_>>();
    wave.prepare_segment_plan_views(&requests).unwrap()
}

#[test]
fn segment_dependency_matches_live_invocation_and_issues_fresh_scope() {
    with_live_wave_fixture(
        fixture_with_padded_persistent(16),
        vec![1, 1],
        |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let invocation = BatchedOperationInvocation::from_resources(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                identity,
                &node,
                OperationInvocationResources::Wave {
                    wave,
                    node_index: 0,
                },
                active.iter(),
                false,
                true,
            )
            .unwrap();
            let component: WeightId = id("weight.component.left");
            let spec = || RetainedPlanDependencySpec {
                input_ordinal: 1,
                component_id: &component,
                source_offset_bytes: 0,
                source_length_bytes: 8,
                persistent_offset_bytes: 0,
                persistent_length_bytes: 4,
                alignment_bytes: 4,
                validation_identity: "fixture.segment-weight-validation.v1",
            };
            let binding = invocation.participants()[0]
                .bindings()
                .iter()
                .find(|b| b.role() == ResolvedValueRole::Input && b.ordinal() == 1)
                .unwrap();
            let resource = binding
                .storage()
                .components()
                .iter()
                .find(|c| c.component_id() == Some(&component))
                .unwrap()
                .resource_id();
            let views = plan_views(fixture, wave);
            let source = views
                .iter()
                .find(|v| v.leased.resource_id() == resource)
                .unwrap();
            let destination = views
                .iter()
                .find(|v| {
                    v.leased.resource_id()
                        == invocation.participants()[0]
                            .persistent_view()
                            .unwrap()
                            .resource_id()
                })
                .unwrap();
            let issue = |scope: &Arc<()>, spec: RetainedPlanDependencySpec<'_>| {
                let selected_binding = invocation.participants()[0]
                    .bindings()
                    .iter()
                    .find(|candidate| {
                        candidate.role() == ResolvedValueRole::Input
                            && candidate.ordinal() == spec.input_ordinal
                    })
                    .unwrap_or(binding);
                issue_segment_plan_dependency(
                    SegmentPlanDependencyInput {
                        scope,
                        node: invocation.node_id(),
                        provider: invocation.provider_id(),
                        binding: selected_binding,
                        source,
                        destination,
                        persistent_preserve: true,
                    },
                    spec,
                )
            };
            let first_scope = Arc::new(());
            let second_scope = Arc::new(());
            let legacy = invocation.retained_plan_dependency(spec()).unwrap();
            let first = issue(&first_scope, spec()).unwrap();
            let second = issue(&second_scope, spec()).unwrap();
            assert_eq!(first.identity, legacy.identity);
            assert_eq!(second.identity, legacy.identity);
            assert!(Arc::ptr_eq(&first.scope, &first_scope));
            assert!(!Arc::ptr_eq(&first.scope, &second.scope));
            assert!(append_dependencies(
                &second_scope,
                vec![first.encode(())],
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new()
            )
            .is_err());
            append_dependencies(
                &second_scope,
                vec![second.encode(())],
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
            )
            .unwrap();
            for fault in 0..8 {
                let mut invalid = spec();
                match fault {
                    0 => invalid.input_ordinal = 0,
                    1 => invalid.source_offset_bytes = 4,
                    2 => invalid.source_length_bytes = 0,
                    3 => invalid.persistent_offset_bytes = u64::MAX,
                    4 => invalid.persistent_offset_bytes = 1,
                    5 => invalid.persistent_length_bytes = 3,
                    6 => invalid.alignment_bytes = 3,
                    7 => invalid.validation_identity = "",
                    _ => unreachable!(),
                }
                let mut other = spec();
                other.input_ordinal = invalid.input_ordinal;
                other.source_offset_bytes = invalid.source_offset_bytes;
                other.source_length_bytes = invalid.source_length_bytes;
                other.persistent_offset_bytes = invalid.persistent_offset_bytes;
                other.persistent_length_bytes = invalid.persistent_length_bytes;
                other.alignment_bytes = invalid.alignment_bytes;
                other.validation_identity = invalid.validation_identity;
                let old = invocation.retained_plan_dependency(invalid).err().unwrap();
                let new = issue(&first_scope, other).err().unwrap();
                assert_eq!(new.to_string(), old.to_string());
            }
        },
    );
}

#[test]
fn segment_dependency_rejects_equal_descriptor_views_from_another_plan_lease() {
    with_live_wave_fixture(
        fixture_with_padded_persistent(16),
        vec![1, 1],
        |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let invocation = BatchedOperationInvocation::from_resources(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                identity,
                &node,
                OperationInvocationResources::Wave {
                    wave,
                    node_index: 0,
                },
                active.iter(),
                false,
                true,
            )
            .unwrap();
            let component: WeightId = id("weight.component.left");
            let binding = invocation.participants()[0]
                .bindings()
                .iter()
                .find(|b| b.role() == ResolvedValueRole::Input && b.ordinal() == 1)
                .unwrap();
            let resource = binding
                .storage()
                .components()
                .iter()
                .find(|c| c.component_id() == Some(&component))
                .unwrap()
                .resource_id();
            let views = plan_views(fixture, wave);
            let source = views
                .iter()
                .find(|v| v.leased.resource_id() == resource)
                .unwrap();
            let destination = views
                .iter()
                .find(|v| {
                    v.leased.resource_id()
                        == invocation.participants()[0]
                            .persistent_view()
                            .unwrap()
                            .resource_id()
                })
                .unwrap();
            with_live_wave_fixture(
                fixture_with_padded_persistent(16),
                vec![1, 1],
                |donor, donor_wave, _, _| {
                    let donor_views = plan_views(donor, donor_wave);
                    let foreign = donor_views
                        .iter()
                        .find(|v| v.leased.resource_id() == destination.leased.resource_id())
                        .unwrap();
                    assert_eq!(
                        foreign.leased.committed_descriptor(),
                        destination.leased.committed_descriptor()
                    );
                    assert!(!std::ptr::eq(
                        source.leased.identity(),
                        foreign.leased.identity()
                    ));
                    let scope = Arc::new(());
                    let result = issue_segment_plan_dependency(
                        SegmentPlanDependencyInput {
                            scope: &scope,
                            node: invocation.node_id(),
                            provider: invocation.provider_id(),
                            binding,
                            source,
                            destination: foreign,
                            persistent_preserve: true,
                        },
                        RetainedPlanDependencySpec {
                            input_ordinal: 1,
                            component_id: &component,
                            source_offset_bytes: 0,
                            source_length_bytes: 8,
                            persistent_offset_bytes: 0,
                            persistent_length_bytes: 4,
                            alignment_bytes: 4,
                            validation_identity: "fixture.segment-weight-validation.v1",
                        },
                    );
                    assert!(result.is_err());
                },
            );
        },
    );
}
