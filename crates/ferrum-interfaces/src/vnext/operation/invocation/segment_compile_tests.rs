//! Compile real admitted Plan schemas; these tests do not mint a live permit.
use super::*;
use crate::vnext::operation::segment_compile::{
    CompiledSegmentBindingRecipe, CompiledSegmentResourceSource,
};
use vnext_device_operation_wave_contract::{setup_with_fixture, teardown};

fn with_compilation(run: impl FnOnce(&Fixture, &DeviceReusableExecutionProgramId)) {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_retained_dependencies(16, DependencyMode::Valid),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let program = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        step.execution_lane(),
    )
    .unwrap()
    .expect("real admitted reusable fixture");
    run(&fixture, &program);
    drop(providers);
    drop(wave);
    teardown(fixture, sequence, session, batch, step);
}

fn region() -> SegmentBindingRegionRequest {
    SegmentBindingRegionRequest {
        selector: SegmentBindingRegionSelector::Persistent,
        offset_bytes: 4,
        extent: SegmentBindingRegionExtent::Exact(4),
        element_type: ElementType::U8,
        alignment_bytes: 4,
    }
}

fn declaration(region: SegmentBindingRegionRequest) -> SegmentBindingDeclaration {
    SegmentBindingDeclaration::new(vec![region], vec![], Arc::new(())).unwrap()
}

#[test]
fn segment_compile_preserves_complete_closure_and_deduplicates_real_resources() {
    with_compilation(|fixture, program| {
        let nodes = fixture.plan.payload().nodes();
        let recipe = CompiledSegmentBindingRecipe::compile(
            &fixture.resolved,
            program.clone(),
            nodes
                .iter()
                .enumerate()
                .map(|(index, _)| (index, declaration(region())))
                .collect(),
        )
        .unwrap();
        let required = nodes
            .iter()
            .flat_map(|node| {
                node.values()
                    .iter()
                    .flat_map(|b| b.storage().components())
                    .map(|c| c.resource_id().clone())
                    .chain(node.scratch_resource().cloned())
                    .chain(node.binding_resource().cloned())
                    .chain(node.persistent_resource().cloned())
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            recipe
                .resources
                .iter()
                .map(|r| r.resource_id.clone())
                .collect::<BTreeSet<_>>(),
            required
        );
        assert_eq!(recipe.resources.len(), required.len());
        for node in &recipe.nodes {
            assert!(
                node.resource_indices.len() > node.regions.len(),
                "one exported workspace must not filter captured inputs"
            );
            let selected = &node.regions[0];
            assert_eq!(selected.offset_bytes, 4);
            assert_eq!(selected.component_length, None);
            assert!(recipe.resources[selected.resource_index].required_minimum_bytes >= 16);
        }
        for resource in &recipe.resources {
            assert!(resource.required_minimum_bytes > 0);
            if let CompiledSegmentResourceSource::PlanStatic { slot_index } = resource.source {
                assert_eq!(
                    fixture.plan.payload().memory().static_allocations()[slot_index].resource_id(),
                    &resource.resource_id
                );
                assert!(recipe.plan_slots.contains(&slot_index));
            }
        }
    });
}

#[test]
fn segment_compile_rejects_selector_alignment_component_and_workspace_bounds() {
    with_compilation(|fixture, program| {
        for fault in 0..7 {
            let mut request = region();
            match fault {
                0 => {
                    request.selector = SegmentBindingRegionSelector::Value {
                        role: ResolvedValueRole::Input,
                        ordinal: u32::MAX,
                        component: None,
                    }
                }
                1 => request.selector = SegmentBindingRegionSelector::Scratch,
                2 => {
                    request.selector = SegmentBindingRegionSelector::Value {
                        role: ResolvedValueRole::Input,
                        ordinal: 1,
                        component: Some(id("absent.component")),
                    }
                }
                3 => {
                    request.selector = SegmentBindingRegionSelector::Value {
                        role: ResolvedValueRole::Input,
                        ordinal: 1,
                        component: None,
                    }
                }
                4 => request.offset_bytes = 15,
                5 => request.element_type = ElementType::F32,
                6 => {
                    request.offset_bytes = 1;
                    request.extent = SegmentBindingRegionExtent::CurrentResource;
                }
                _ => unreachable!(),
            }
            assert!(
                CompiledSegmentBindingRecipe::compile(
                    &fixture.resolved,
                    program.clone(),
                    vec![(0, declaration(request))],
                )
                .is_err(),
                "fault {fault}"
            );
        }
        let mut valid = region();
        valid.selector = SegmentBindingRegionSelector::Value {
            role: ResolvedValueRole::Input,
            ordinal: 1,
            component: Some(id("weight.component.left")),
        };
        valid.offset_bytes = 0;
        valid.extent = SegmentBindingRegionExtent::Exact(8);
        valid.element_type = fixture.plan.payload().nodes()[0]
            .values()
            .iter()
            .find(|b| b.role() == ResolvedValueRole::Input && b.ordinal() == 1)
            .unwrap()
            .storage()
            .components()
            .iter()
            .find(|c| c.component_id() == Some(&id("weight.component.left")))
            .unwrap()
            .element_type();
        let recipe = CompiledSegmentBindingRecipe::compile(
            &fixture.resolved,
            program.clone(),
            vec![(0, declaration(valid))],
        )
        .unwrap();
        assert_eq!(recipe.nodes[0].regions[0].component_length, Some(8));
        assert_eq!(
            recipe.nodes[0].regions[0]
                .length_for_current_view(64)
                .unwrap(),
            8
        );
        assert!(CompiledSegmentBindingRecipe::compile(
            &fixture.resolved,
            program.clone(),
            vec![(0, declaration(region())), (0, declaration(region()))],
        )
        .is_err());
        assert!(CompiledSegmentBindingRecipe::compile(
            &fixture.resolved,
            program.clone(),
            vec![(usize::MAX, declaration(region()))],
        )
        .is_err());
    });
}

#[test]
fn segment_compile_dependency_maps_exact_plan_slots_and_rejects_range_overrun() {
    with_compilation(|fixture, program| {
        let spec = SegmentBindingDependency {
            input_ordinal: 1,
            component_id: id("weight.component.left"),
            source_offset_bytes: 0,
            source_length_bytes: 8,
            persistent_offset_bytes: 0,
            persistent_length_bytes: 4,
            alignment_bytes: 4,
            validation_identity: "fixture.segment-validation.v1".into(),
        };
        let make = |spec| SegmentBindingDeclaration::new(vec![], vec![spec], Arc::new(())).unwrap();
        let recipe = CompiledSegmentBindingRecipe::compile(
            &fixture.resolved,
            program.clone(),
            vec![(0, make(spec.clone()))],
        )
        .unwrap();
        let dependency = &recipe.nodes[0].dependencies[0];
        assert!(recipe.nodes[0].persistent_preserve);
        for index in [
            dependency.source_resource_index,
            dependency.destination_resource_index,
        ] {
            assert!(matches!(
                recipe.resources[index].source,
                CompiledSegmentResourceSource::PlanStatic { .. }
            ));
        }
        assert_ne!(
            dependency.source_resource_index,
            dependency.destination_resource_index
        );
        assert_eq!(
            fixture.plan.payload().nodes()[0].values()[dependency.binding_index].ordinal(),
            spec.input_ordinal
        );
        for source_fault in [true, false] {
            let mut bad = spec.clone();
            if source_fault {
                bad.source_offset_bytes = 4;
            } else {
                bad.persistent_offset_bytes = 16;
            }
            assert!(CompiledSegmentBindingRecipe::compile(
                &fixture.resolved,
                program.clone(),
                vec![(0, make(bad))],
            )
            .is_err());
        }
    });
}

#[test]
fn segment_compile_current_state_extent_accepts_real_growth_beyond_canonical_minimum() {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("execution.segment-growth").unwrap(),
        ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
    )
    .unwrap();
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_token_scaled_paged_state_and_bucket(bucket));
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let program = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        step.execution_lane(),
    )
    .unwrap()
    .unwrap();
    let binding = fixture.plan.payload().nodes()[0]
        .values()
        .iter()
        .find(|b| b.usage() == BufferUsage::State)
        .unwrap();
    let component = &binding.storage().components()[0];
    let minimum = component.length_bytes();
    let resource = component.resource_id().clone();
    let request = SegmentBindingRegionRequest {
        selector: SegmentBindingRegionSelector::Value {
            role: binding.role(),
            ordinal: binding.ordinal(),
            component: None,
        },
        offset_bytes: 0,
        extent: SegmentBindingRegionExtent::CurrentResource,
        element_type: component.element_type(),
        alignment_bytes: 1,
    };
    let recipe = CompiledSegmentBindingRecipe::compile(
        &fixture.resolved,
        program.clone(),
        vec![(0, declaration(request.clone()))],
    )
    .unwrap();
    let region = &recipe.nodes[0].regions[0];
    assert_eq!(region.component_length, None);
    assert!(recipe.resources[region.resource_index].required_minimum_bytes >= minimum);
    let mut invalid = request.clone();
    invalid.offset_bytes = 1;
    assert!(CompiledSegmentBindingRecipe::compile(
        &fixture.resolved,
        program,
        vec![(0, declaration(invalid))],
    )
    .is_err());
    drop(providers);
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);

    // A second real admission in the same Plan supplies a larger authorized
    // state view. This tests the hot window rule, not a claimed cache hit.
    let span = TokenSpanWork::from_token_ids(&[1, 2, 3, 4], 0..4).unwrap();
    let sequence = logical_resources_with_work(
        &fixture.plan_resources,
        "run.segment.grown",
        "request.segment.grown",
        span.clone(),
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = begin_single_participant_step_on_lane_with_bucket_and_work(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
        span,
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let current_program = OperationDispatch::reusable_execution_program_id_for_wave(
        providers.providers(),
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .unwrap();
    let fresh_recipe = CompiledSegmentBindingRecipe::compile(
        &fixture.resolved,
        current_program,
        vec![(0, declaration(request))],
    )
    .unwrap();
    drop(providers);
    let active = vec![TrustedActiveSequenceBinding::from_session(&session).unwrap()];
    let identity = super::segment_authority_tests::segment_identity(&fixture, &wave, &active);
    let provider = fixture
        .registry
        .bind(&fixture.resolved, wave.nodes()[0].node_id())
        .unwrap();
    let node = identity.materialize_node(0).unwrap();
    let invocation = BatchedOperationInvocation::from_resources(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        provider.dispatch(),
        &identity,
        node,
        OperationInvocationResources::Wave {
            wave: &wave,
            node_index: 0,
        },
        active.iter(),
        false,
        true,
    )
    .unwrap();
    let view = invocation.participants()[0]
        .views()
        .iter()
        .find(|view| view.resource_id() == &resource)
        .unwrap();
    let current = view.descriptor().size_bytes;
    assert!(
        current > minimum,
        "real State admission must exceed its canonical component"
    );
    assert_eq!(region.length_for_current_view(current).unwrap(), current);
    assert!(region.length_for_current_view(0).is_err());
    let ranges = view.translate(0, current).unwrap();
    assert_eq!(
        ranges.iter().map(|r| r.length_bytes()).sum::<u64>(),
        current
    );
    let expected = SegmentTestRegion {
        node: 0,
        participant: 0,
        region: 0,
        descriptor: view.descriptor().clone(),
        physical: ranges
            .iter()
            .map(|part| {
                let (buffer, range, _) = part.buffer_and_physical_range();
                (
                    buffer as *const TestBuffer as usize,
                    range,
                    part.logical_offset_bytes(),
                )
            })
            .collect(),
    };
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.segment_metadata_enabled = true;
        trace.segment_encoder_enabled = true;
    }
    // New dimensions use the actual newly prepared program identity. This
    // invokes the complete hot table helper, not a claim of cross-shape reuse.
    let (encoded, _) = crate::vnext::operation::segment_dispatch::encode_segment_wave(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        &identity,
        &wave,
        active.iter(),
        &fresh_recipe,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        fixture.runtime_trace.lock().unwrap().segment_regions,
        vec![expected]
    );
    drop(encoded);
    drop(ranges);
    drop(invocation);
    drop(provider);
    drop(identity);
    drop(active);
    drop(wave);
    drop(lane);
    drop(recipe);
    drop(fresh_recipe);
    teardown(fixture, sequence, session, batch, step);
}
