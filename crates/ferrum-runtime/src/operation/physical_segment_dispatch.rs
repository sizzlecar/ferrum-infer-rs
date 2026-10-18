//! Fresh logical authorization over retained, pool-versioned physical tables.
use super::buffer_view::sequence_execution_shape;
use super::dispatch_contract::SubmissionWaveDispatchStageTimer;
use super::foundation::invalid_operation;
use super::physical_segment::PhysicalFrameResource;
use super::retained_dependency::{issue_segment_plan_dependency, SegmentPlanDependencyInput};
use super::segment_compile::{CompiledSegmentBindingRecipe, CompiledSegmentResourceSource};
use super::segment_dispatch::{
    validate_buffer, SegmentEncodedNode, SegmentEncodedProgram, SegmentPreparationFacts,
};
use crate::resource::{SegmentBackingWindowInput, SegmentPlanResourceView, SegmentWindowResource};
use crate::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;

// Inputs are cold-sorted unique buffer identities. Count their union without
// rebuilding a physical owner/region table merely for telemetry.
fn unique_buffers(rows: &[&[usize]], fresh: &[usize]) -> usize {
    let lists = rows
        .iter()
        .copied()
        .chain(std::iter::once(fresh))
        .collect::<Vec<_>>();
    let mut queue = BinaryHeap::with_capacity(lists.len());
    for (list, entries) in lists.iter().enumerate() {
        if let Some(&first) = entries.first() {
            queue.push(Reverse((first, list, 0usize)));
        }
    }
    let mut previous = None;
    let mut count = 0;
    while let Some(Reverse((key, list, index))) = queue.pop() {
        if previous != Some(key) {
            count += 1;
            previous = Some(key);
        }
        if let Some(&next) = lists[list].get(index + 1) {
            queue.push(Reverse((next, list, index + 1)));
        }
    }
    count
}

/// Called only after the ordinary exact Plan/wave/participant validation.
/// Unsupported providers/lifetimes fall back before any dependency is issued.
pub(super) fn with_prepared_physical_wave<
    R: DeviceRuntime,
    S: SubmissionWaveDispatchTimingSink,
    T,
>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    wave: &PreparedStepSubmissionWave<R>,
    recipe: &CompiledSegmentBindingRecipe,
    plan_views: &[SegmentPlanResourceView<'_, R::Buffer>],
    timing: &S,
    encode: impl FnOnce(
        &mut PreparedPhysicalSegmentWave<'_, R::Buffer>,
        Vec<Arc<()>>,
        SegmentPreparationFacts,
    ) -> Result<Option<T>, VNextError>,
) -> Result<Option<T>, VNextError> {
    if !runtime.supports_physical_segment_bindings() {
        return Ok(None);
    }
    let memory = resolved.execution_plan().payload().memory();
    for resource in &recipe.resources {
        if let CompiledSegmentResourceSource::Dynamic { descriptor_index } = resource.source {
            let descriptor = memory
                .dynamic_descriptors()
                .get(descriptor_index)
                .filter(|descriptor| descriptor.base_resource_id() == &resource.resource_id)
                .ok_or_else(|| {
                    invalid_operation("physical segment descriptor differs from its exact Plan")
                })?;
            if !matches!(
                descriptor.lifetime(),
                AllocationLifetime::Sequence
                    | AllocationLifetime::Step
                    | AllocationLifetime::Invocation
            ) {
                return Ok(None);
            }
        }
    }
    let first = wave
        .nodes()
        .first()
        .ok_or_else(|| invalid_operation("physical segment wave is empty"))?;
    let participant_count = first.participant_count() as usize;
    let backing_stage = SubmissionWaveDispatchStageTimer::start(
        timing,
        SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata,
    );
    let context = wave.segment_backing_context()?;
    let Some(tables) = context.physical_tables(runtime, timing)? else {
        return Ok(None);
    };
    if tables.len() != participant_count {
        return Err(invalid_operation(
            "physical table participant count differs",
        ));
    }
    let mut resources = Vec::with_capacity(recipe.resources.len());
    let mut sequence_bytes = Vec::new();
    let mut sequence_identities = Vec::new();
    let mut fresh_groups = Vec::new();
    let mut fresh_expected = Vec::new();
    let mut fresh_windows = Vec::new();
    for resource in &recipe.resources {
        match resource.source {
            CompiledSegmentResourceSource::PlanStatic { slot_index } => {
                let index = recipe
                    .plan_slots
                    .binary_search(&slot_index)
                    .map_err(|_| invalid_operation("physical segment Plan slot is absent"))?;
                if plan_views[index].leased.committed_descriptor().size_bytes
                    < resource.required_minimum_bytes
                {
                    return Err(invalid_operation(
                        "physical segment Plan resource is too small",
                    ));
                }
                resources.push(PhysicalFrameResource::Plan(index));
            }
            CompiledSegmentResourceSource::Dynamic { descriptor_index } => {
                let descriptor = &memory.dynamic_descriptors()[descriptor_index];
                let lifetime = descriptor.lifetime();
                if lifetime == AllocationLifetime::Sequence {
                    let first_size = sequence_bytes.len();
                    for (participant, table) in tables.iter().enumerate() {
                        let row = table
                            .resource(descriptor_index)
                            .filter(|row| row.resource_id() == &resource.resource_id)
                            .ok_or_else(|| {
                                invalid_operation(
                                    "physical Sequence resource differs from its Plan ordinal",
                                )
                            })?;
                        let tokens = &first.work_shape().participant_token_ranges()[participant];
                        let bytes = descriptor.evaluate_request_bytes_for_shape(
                            sequence_execution_shape(
                                context.sequence_shape(participant)?,
                                tokens.source_token_range().end,
                            )?,
                        )?;
                        let physical = row.descriptor();
                        sequence_identities.push(row.metadata_identities());
                        if bytes == 0
                            || bytes < resource.required_minimum_bytes
                            || bytes > physical.logical_size_bytes()
                            || physical.usage() != descriptor.usage()
                            || physical.element_type() != descriptor.element_type()
                            || physical.storage_profile() != descriptor.storage().profile()
                            || physical.alignment_bytes() != descriptor.alignment_bytes()
                        {
                            return Err(invalid_operation("physical Sequence table does not authorize the current logical view"));
                        }
                        sequence_bytes.push(bytes);
                    }
                    resources.push(PhysicalFrameResource::Sequence {
                        ordinal: descriptor_index,
                        first_size,
                    });
                } else {
                    let bytes = descriptor.evaluate_request_bytes_for_shape(match lifetime {
                        AllocationLifetime::Step => {
                            wave.step_resources().work_shape().immediate_shape()
                        }
                        AllocationLifetime::Invocation => first.work_shape().immediate_shape(),
                        _ => {
                            return Err(invalid_operation("unsupported physical segment lifetime"))
                        }
                    })?;
                    let participant_stride =
                        match (lifetime, descriptor.kind(), descriptor.demand()) {
                            (
                                AllocationLifetime::Step,
                                AllocationKind::Value,
                                DynamicResourceDemand::ActualSequences {
                                    bytes_per_sequence,
                                    maximum_sequences,
                                },
                            ) => {
                                let shape = wave.step_resources().work_shape();
                                if shape.immediate_sequences() > *maximum_sequences
                                    || participant_count > shape.participants().len()
                                {
                                    return Err(invalid_operation(
                                    "physical segment participant exceeds the current Step shape",
                                ));
                                }
                                Some(*bytes_per_sequence)
                            }
                            _ => None,
                        };
                    let index = fresh_expected.len();
                    let rows = SegmentWindowResource {
                        first_request: index,
                        shared: true,
                        participant_stride,
                    };
                    fresh_groups.push(context.resolve(
                        0,
                        &resource.resource_id,
                        descriptor_index,
                        lifetime,
                    )?);
                    fresh_expected.push(SegmentBackingExpectation {
                        logical_bytes: bytes,
                        logical_size_rule: SegmentLogicalSizeRule::Exact,
                        usage: descriptor.usage(),
                        storage_profile: descriptor.storage().profile(),
                        element_type: descriptor.element_type(),
                        alignment_bytes: descriptor.alignment_bytes(),
                    });
                    for participant in 0..participant_count {
                        if rows.row(participant, &fresh_expected)?.2
                            < resource.required_minimum_bytes
                        {
                            return Err(invalid_operation(
                                "physical segment fresh resource is too small",
                            ));
                        }
                    }
                    fresh_windows.push(SegmentBackingWindow {
                        resource_index: index,
                        offset_bytes: 0,
                        length_bytes: bytes,
                        element_type: descriptor.element_type(),
                        alignment_bytes: descriptor.alignment_bytes(),
                    });
                    resources.push(PhysicalFrameResource::Fresh(rows));
                }
            }
        }
    }
    // Only current Step/Invocation resources use the ordinary permit. Sequence
    // groups, per-leaf windows, physical dedup and owner vectors are absent.
    let fresh_backing = if fresh_expected.is_empty() {
        None
    } else {
        Some(context.prepare(
            &fresh_groups,
            &fresh_expected,
            SegmentBackingWindowInput::Slice(&fresh_windows),
            timing,
        )?)
    };
    let mut seen = BTreeMap::new();
    for plan in plan_views {
        if plan.leased.generation() == 0 {
            return Err(invalid_operation(
                "physical segment Plan resource has no generation",
            ));
        }
        if !validate_buffer(
            runtime,
            plan.leased.buffer(),
            plan.leased.committed_descriptor(),
            &mut seen,
        )? {
            return Ok(None);
        }
    }
    if let Some(backing) = &fresh_backing {
        for binding in backing.bindings().iter() {
            if !validate_buffer(runtime, binding.buffer(), binding.descriptor(), &mut seen)? {
                return Ok(None);
            }
        }
        for (index, requirement) in fresh_expected.iter().enumerate() {
            if matches!(
                requirement.storage_profile.view(),
                DynamicStorageView::Contiguous
            ) && backing
                .window(index)
                .is_none_or(|window| window.physical_regions().len() != 1)
            {
                return Err(invalid_operation(
                    "physical segment contiguous resource has multiple physical regions",
                ));
            }
        }
    }
    let fresh_keys = seen.keys().map(|key| *key as usize).collect::<Vec<_>>();
    let unique_physical_buffers = unique_buffers(&sequence_identities, &fresh_keys) as u64;
    drop(backing_stage);
    let node_stage = SubmissionWaveDispatchStageTimer::start(
        timing,
        SubmissionWaveDispatchStage::SegmentNodeDependenciesAndRegions,
    );
    let mut scopes = Vec::with_capacity(recipe.nodes.len());
    let mut dependencies = Vec::with_capacity(recipe.nodes.len());
    let mut program_bindings = Vec::with_capacity(recipe.nodes.len());
    for compiled in &recipe.nodes {
        let node = resolved
            .execution_plan()
            .payload()
            .nodes()
            .get(compiled.node_index)
            .ok_or_else(|| {
                invalid_operation("physical segment node is absent from the current Plan")
            })?;
        let scope = Arc::new(());
        let mut issued = Vec::with_capacity(compiled.dependencies.len());
        for dependency in &compiled.dependencies {
            let plan_view =
                |resource: usize| -> Result<&SegmentPlanResourceView<'_, R::Buffer>, VNextError> {
                    match resources.get(resource) {
                        Some(PhysicalFrameResource::Plan(index)) => Ok(&plan_views[*index]),
                        _ => Err(invalid_operation(
                            "physical segment dependency is not a fresh Plan resource",
                        )),
                    }
                };
            issued.push(Some(issue_segment_plan_dependency(
                SegmentPlanDependencyInput {
                    scope: &scope,
                    node: node.id(),
                    provider: node.selection().selected_provider(),
                    binding: &node.values()[dependency.binding_index],
                    source: plan_view(dependency.source_resource_index)?,
                    destination: plan_view(dependency.destination_resource_index)?,
                    persistent_preserve: compiled.persistent_preserve,
                },
                dependency.spec.as_spec(),
            )?));
        }
        dependencies.push(issued);
        program_bindings.push(
            wave.claimed_backing()
                .program_binding_node(compiled.node_index),
        );
        scopes.push(scope);
    }
    let dynamic_resource_requests = (sequence_bytes.len() + fresh_expected.len()) as u64;
    let mut physical = PreparedPhysicalSegmentWave {
        recipe,
        plan_views,
        tables: &tables,
        resources,
        sequence_bytes,
        fresh_backing: fresh_backing.as_ref(),
        fresh_expected: &fresh_expected,
        work_shape: first.work_shape(),
        program_bindings,
        dependencies,
    };
    // Check all current logical windows before the provider can consume even
    // one dependency. Physical prefix proofs remain sealed in the tables.
    for (index, compiled) in recipe.nodes.iter().enumerate() {
        let node = physical.node(index)?;
        for participant in 0..participant_count {
            for declaration in 0..compiled.regions.len() {
                node.region(participant, declaration)?;
            }
        }
    }
    drop(node_stage);
    let _backend_stage = SubmissionWaveDispatchStageTimer::start(
        timing,
        SubmissionWaveDispatchStage::SegmentBackendEncodeAndValidate,
    );
    encode(
        &mut physical,
        scopes,
        SegmentPreparationFacts {
            dynamic_resource_requests,
            unique_physical_buffers,
        },
    )
}

pub(super) fn try_encode<R: DeviceRuntime, S: SubmissionWaveDispatchTimingSink>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    wave: &PreparedStepSubmissionWave<R>,
    recipe: &CompiledSegmentBindingRecipe,
    program: &(dyn std::any::Any + Send + Sync),
    plan_views: &[SegmentPlanResourceView<'_, R::Buffer>],
    timing: &S,
) -> Result<Option<(SegmentEncodedProgram<R::Command>, SegmentPreparationFacts)>, VNextError> {
    with_prepared_physical_wave(
        runtime,
        resolved,
        wave,
        recipe,
        plan_views,
        timing,
        |physical, scopes, facts| {
            let encoded = runtime
                .encode_physical_segment_bindings(program, physical)
                .map_err(|error| {
                    invalid_operation(format!("physical segment encoding failed: {error}"))
                })?;
            let Some(encoded) = encoded else {
                if physical.dependencies.iter().flatten().any(Option::is_none) {
                    return Err(invalid_operation(
                        "physical segment backend declined after consuming fresh authority",
                    ));
                }
                return Ok(None);
            };
            if encoded.nodes.len() != recipe.nodes.len()
                || encoded
                    .nodes
                    .iter()
                    .zip(&recipe.nodes)
                    .any(|(encoded, compiled)| encoded.node_index as usize != compiled.node_index)
            {
                return Err(invalid_operation(
                    "physical segment backend returned a different node sequence",
                ));
            }
            if physical.dependencies.iter().flatten().any(Option::is_some) {
                return Err(invalid_operation(
                    "physical segment backend did not consume every fresh dependency",
                ));
            }
            Ok(Some((
                SegmentEncodedProgram {
                    nodes: encoded
                        .nodes
                        .into_iter()
                        .zip(scopes)
                        .map(|(encoded, scope)| SegmentEncodedNode {
                            node_index: encoded.node_index as usize,
                            scope,
                            program_binding: wave
                                .claimed_backing()
                                .program_binding_node(encoded.node_index as usize),
                            bindings: encoded.bindings,
                        })
                        .collect(),
                    program_bindings: encoded.program_bindings,
                    program_binding_nodes: encoded.program_binding_nodes,
                },
                facts,
            )))
        },
    )
}
