//! Fresh whole-wave facts for a closed resident binding recipe. Nothing in this
//! module caches a request, frame, dynamic buffer, or dependency authority.
use std::collections::BTreeMap;
use std::sync::Arc;

use super::buffer_view::sequence_execution_shape;
use super::dispatch_contract::SubmissionWaveDispatchStageTimer;
use super::foundation::invalid_operation;
use super::invocation::{validate_segment_wave_authority, OperationInvocationResources};
use super::preparation::SegmentPreparationMissReason;
use super::retained_dependency::{issue_segment_plan_dependency, SegmentPlanDependencyInput};
use super::segment_binding::SegmentBindingPhysicalSource;
use super::segment_compile::{CompiledSegmentBindingRecipe, CompiledSegmentResourceSource};
use crate::vnext::resource::{SegmentPlanResourceRequest, SegmentPlanResourceView};
use crate::vnext::*;

pub(super) struct SegmentEncodedNode<C> {
    pub node_index: usize,
    pub scope: Arc<()>,
    pub program_binding: Option<ProgramBindingNodeBinding>,
    pub bindings: EncodedReusableExecutionBindings<C>,
}

pub(super) struct SegmentPreparationFacts {
    pub dynamic_resource_requests: u64,
    pub unique_physical_buffers: u64,
}

#[derive(Clone, Copy)]
enum FreshResource {
    Plan(usize),
    Dynamic {
        request: usize,
        offset: u64,
        bytes: u64,
    },
}

struct RegionWindow {
    descriptor: BufferDescriptor,
    storage_kind: OperationBufferStorageKind,
    source: RegionSource,
}

enum RegionSource {
    Plan {
        index: usize,
        offset: u64,
        length: u64,
    },
    Dynamic(usize),
}

fn checked_range(offset: u64, length: u64, bound: u64) -> Result<(), VNextError> {
    if length == 0 || offset.checked_add(length).is_none_or(|end| end > bound) {
        return Err(invalid_operation(
            "segment region exceeds its current authorized extent",
        ));
    }
    Ok(())
}

/// Owner identities below are ephemeral dedup keys only. A fresh capability
/// and the original getter must agree before the owner enters this table.
fn validate_buffer<'a, R: DeviceRuntime>(
    runtime: &R,
    buffer: &'a R::Buffer,
    expected: &'a BufferDescriptor,
    seen: &mut BTreeMap<*const R::Buffer, &'a BufferDescriptor>,
) -> Result<bool, VNextError> {
    let key = buffer as *const _;
    if let Some(previous) = seen.get(&key) {
        if *previous != expected {
            return Err(invalid_operation(
                "one segment buffer has conflicting committed metadata",
            ));
        }
        return Ok(true);
    }
    let Some(capability) = runtime.immutable_buffer_metadata(buffer) else {
        return Ok(false);
    };
    let declared = capability.for_owner(runtime, buffer).ok_or_else(|| {
        invalid_operation("immutable segment buffer metadata has a foreign owner")
    })?;
    if declared != expected || runtime.buffer_descriptor(buffer) != *expected {
        return Err(invalid_operation(
            "segment runtime buffer differs from its current committed descriptor",
        ));
    }
    seen.insert(key, expected);
    Ok(true)
}

#[cfg(test)]
pub(super) fn encode_segment_wave<'binding, R, I>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    identity: &BatchOperationIdentity,
    wave: &PreparedStepSubmissionWave<R>,
    active: I,
    recipe: &CompiledSegmentBindingRecipe,
) -> Result<Option<(Vec<SegmentEncodedNode<R::Command>>, SegmentPreparationFacts)>, VNextError>
where
    R: DeviceRuntime,
    I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
{
    encode_segment_wave_with_timing(
        runtime,
        resolved,
        identity,
        wave,
        active,
        recipe,
        &super::dispatch_contract::DisabledSubmissionWaveDispatchTimingSink,
    )
}

pub(super) fn encode_segment_wave_with_timing<'binding, R, I, S>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    identity: &BatchOperationIdentity,
    wave: &PreparedStepSubmissionWave<R>,
    active: I,
    recipe: &CompiledSegmentBindingRecipe,
    timing_sink: &S,
) -> Result<Option<(Vec<SegmentEncodedNode<R::Command>>, SegmentPreparationFacts)>, VNextError>
where
    R: DeviceRuntime,
    I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    S: SubmissionWaveDispatchTimingSink,
{
    let authority_stage = SubmissionWaveDispatchStageTimer::start(
        timing_sink,
        SubmissionWaveDispatchStage::SegmentFreshAuthorityAndWindows,
    );
    let Some(capability) = runtime.immutable_runtime_metadata() else {
        identity.record_segment_miss_reason(
            SegmentPreparationMissReason::ImmutableCapabilityUnavailable,
        );
        return Ok(None);
    };
    let declared = capability.for_owner(runtime).ok_or_else(|| {
        invalid_operation("immutable segment runtime metadata has a foreign owner")
    })?;
    if declared != runtime.descriptor()
        || declared != resolved.device()
        || declared != resolved.capabilities().device()
    {
        return Err(invalid_operation(
            "segment runtime metadata differs from the exact Plan",
        ));
    }
    validate_segment_wave_authority(runtime, resolved, identity, wave, active)?;
    let first = wave
        .nodes()
        .first()
        .ok_or_else(|| invalid_operation("segment wave is empty"))?;
    if recipe.program_id.plan_hash() != identity.plan_hash()
        || recipe.program_id.lane_id() != wave.execution_lane_id()
        || recipe.program_id.runtime_implementation_fingerprint()
            != identity.runtime_implementation_fingerprint()
        || recipe.program_id.immediate_sequences() != first.work_shape().immediate_sequences()
        || recipe.program_id.immediate_tokens() != first.work_shape().immediate_tokens()
        || recipe.program_id.immediate_pages() != first.work_shape().immediate_pages()
    {
        return Err(invalid_operation(
            "segment recipe differs from its current Plan, lane, or exact work shape",
        ));
    }
    let participant_count = first.participant_count() as usize;
    let current = OperationInvocationResources::Wave {
        wave,
        node_index: 0,
    };
    let memory = resolved.execution_plan().payload().memory();
    let plan_requests = recipe
        .plan_slots
        .iter()
        .map(|&slot_index| {
            let allocation = memory
                .static_allocations()
                .get(slot_index)
                .ok_or_else(|| invalid_operation("segment Plan slot no longer exists"))?;
            Ok(SegmentPlanResourceRequest {
                slot_index,
                allocation,
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let plan_views = wave.prepare_segment_plan_views(&plan_requests)?;
    let mut requests: Vec<SegmentBackingRequest> = Vec::new();
    let mut fresh = Vec::with_capacity(recipe.resources.len());
    for resource in &recipe.resources {
        let mut participants = Vec::with_capacity(participant_count);
        match resource.source {
            CompiledSegmentResourceSource::PlanStatic { slot_index } => {
                let index = recipe.plan_slots.binary_search(&slot_index).map_err(|_| {
                    invalid_operation("segment Plan slot is absent from its schema")
                })?;
                if plan_views[index].leased.committed_descriptor().size_bytes
                    < resource.required_minimum_bytes
                {
                    return Err(invalid_operation(
                        "segment Plan resource is smaller than its compiled requirements",
                    ));
                }
                participants.resize(participant_count, FreshResource::Plan(index));
            }
            CompiledSegmentResourceSource::Dynamic { descriptor_index } => {
                let descriptor = memory
                    .dynamic_descriptors()
                    .get(descriptor_index)
                    .filter(|descriptor| descriptor.base_resource_id() == &resource.resource_id)
                    .ok_or_else(|| {
                        invalid_operation("segment descriptor differs from its exact Plan")
                    })?;
                let lifetime = descriptor.lifetime();
                let shared = matches!(
                    lifetime,
                    AllocationLifetime::Invocation | AllocationLifetime::Step
                );
                let first_request = requests.len();
                for participant_index in 0..participant_count {
                    let expected_bytes = if shared && participant_index != 0 {
                        requests[first_request].expected.logical_bytes
                    } else {
                        match lifetime {
                            AllocationLifetime::Invocation => descriptor
                                .evaluate_request_bytes_for_shape(
                                    first.work_shape().immediate_shape(),
                                )?,
                            AllocationLifetime::Step => descriptor
                                .evaluate_request_bytes_for_shape(
                                    wave.step_resources().work_shape().immediate_shape(),
                                )?,
                            AllocationLifetime::Sequence => {
                                let backing =
                                    current.participant_backing_snapshot(participant_index)?;
                                let tokens = first
                                    .work_shape()
                                    .participant_token_ranges()
                                    .get(participant_index)
                                    .ok_or_else(|| {
                                        invalid_operation(
                                            "segment participant token range is missing",
                                        )
                                    })?;
                                descriptor.evaluate_request_bytes_for_shape(
                                    sequence_execution_shape(
                                        backing.committed_shape(),
                                        tokens.source_token_range().end,
                                    )?,
                                )?
                            }
                            AllocationLifetime::Request => descriptor.evaluate_fit_request_bytes(
                                current
                                    .participant(participant_index)?
                                    .request_resources()
                                    .work_shape(),
                            )?,
                            AllocationLifetime::Plan => {
                                return Err(invalid_operation(
                                    "dynamic segment resource claims Plan lifetime",
                                ))
                            }
                        }
                    };
                    let request_index = if shared && participant_index != 0 {
                        first_request
                    } else {
                        let index = requests.len();
                        requests.push(SegmentBackingRequest {
                            participant_index,
                            resource_id: resource.resource_id.clone(),
                            lifetime,
                            expected: SegmentBackingExpectation {
                                logical_bytes: expected_bytes,
                                logical_size_rule: if lifetime == AllocationLifetime::Sequence {
                                    SegmentLogicalSizeRule::AtLeast
                                } else {
                                    SegmentLogicalSizeRule::Exact
                                },
                                usage: descriptor.usage(),
                                storage_profile: descriptor.storage().profile(),
                                element_type: descriptor.element_type(),
                                alignment_bytes: descriptor.alignment_bytes(),
                            },
                        });
                        index
                    };
                    let (offset, bytes) = match (lifetime, descriptor.kind(), descriptor.demand()) {
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
                                || participant_index >= shape.participants().len()
                            {
                                return Err(invalid_operation(
                                    "segment participant exceeds the Step work shape",
                                ));
                            }
                            (
                                bytes_per_sequence
                                    .checked_mul(participant_index as u64)
                                    .ok_or_else(|| {
                                        invalid_operation("segment participant window overflows")
                                    })?,
                                *bytes_per_sequence,
                            )
                        }
                        _ => (0, expected_bytes),
                    };
                    checked_range(offset, bytes, expected_bytes)?;
                    if bytes < resource.required_minimum_bytes {
                        return Err(invalid_operation(
                            "segment current resource is smaller than its compiled requirements",
                        ));
                    }
                    participants.push(FreshResource::Dynamic {
                        request: request_index,
                        offset,
                        bytes,
                    });
                }
            }
        }
        fresh.push(participants);
    }
    // Validate full physical coverage, including captured resources not patched
    // by the backend. Declared subwindows are consumed under the same permit.
    let mut windows = requests
        .iter()
        .enumerate()
        .map(|(resource_index, request)| SegmentBackingWindow {
            resource_index,
            offset_bytes: 0,
            length_bytes: request.expected.logical_bytes,
            element_type: request.expected.element_type,
            alignment_bytes: request.expected.alignment_bytes,
        })
        .collect::<Vec<_>>();
    let mut node_windows = Vec::with_capacity(recipe.nodes.len());
    for node in &recipe.nodes {
        let capacity = participant_count
            .checked_mul(node.regions.len())
            .ok_or_else(|| invalid_operation("segment region count overflows usize"))?;
        let mut regions = Vec::with_capacity(capacity);
        for participant in 0..participant_count {
            for region in &node.regions {
                let source = fresh
                    .get(region.resource_index)
                    .and_then(|rows| rows.get(participant))
                    .ok_or_else(|| invalid_operation("segment region resource index is invalid"))?;
                let (mut descriptor, kind, available) = match *source {
                    FreshResource::Plan(index) => {
                        let descriptor = plan_views[index].leased.committed_descriptor().clone();
                        let bytes = descriptor.size_bytes;
                        (
                            descriptor,
                            OperationBufferStorageKind::StaticContiguous,
                            bytes,
                        )
                    }
                    FreshResource::Dynamic { request, bytes, .. } => {
                        let request = &requests[request];
                        let kind = match request.expected.storage_profile.view() {
                            DynamicStorageView::Contiguous => {
                                OperationBufferStorageKind::DynamicContiguous
                            }
                            DynamicStorageView::PagedRegions { .. } => {
                                OperationBufferStorageKind::DynamicPaged
                            }
                        };
                        (
                            BufferDescriptor {
                                resource_id: request.resource_id.clone(),
                                size_bytes: bytes,
                                alignment_bytes: request.expected.alignment_bytes,
                                usage: request.expected.usage,
                                element_type: request.expected.element_type,
                            },
                            kind,
                            bytes,
                        )
                    }
                };
                let length = region.length_for_current_view(available)?;
                if descriptor.element_type != region.request.element_type
                    || descriptor.alignment_bytes < region.request.alignment_bytes
                    || !region
                        .offset_bytes
                        .is_multiple_of(region.request.alignment_bytes)
                {
                    return Err(invalid_operation(
                        "segment region type or alignment differs from its resource",
                    ));
                }
                descriptor.size_bytes = length;
                let source = match *source {
                    FreshResource::Plan(index) => RegionSource::Plan {
                        index,
                        offset: region.offset_bytes,
                        length,
                    },
                    FreshResource::Dynamic {
                        request, offset, ..
                    } => {
                        let index = windows.len();
                        windows.push(SegmentBackingWindow {
                            resource_index: request,
                            offset_bytes: offset.checked_add(region.offset_bytes).ok_or_else(
                                || invalid_operation("segment backing window overflows"),
                            )?,
                            length_bytes: length,
                            element_type: region.request.element_type,
                            alignment_bytes: region.request.alignment_bytes,
                        });
                        RegionSource::Dynamic(index)
                    }
                };
                regions.push(RegionWindow {
                    descriptor,
                    storage_kind: kind,
                    source,
                });
            }
        }
        node_windows.push(regions);
    }
    drop(authority_stage);
    let backing_stage = SubmissionWaveDispatchStageTimer::start(
        timing_sink,
        SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata,
    );
    let backing = if requests.is_empty() {
        None
    } else {
        Some(wave.prepare_segment_backings_with_timing(&requests, &windows, timing_sink)?)
    };
    let metadata_stage = SubmissionWaveDispatchStageTimer::start(
        timing_sink,
        SubmissionWaveDispatchStage::SegmentBackingImmutableMetadataValidation,
    );
    let mut seen = BTreeMap::new();
    for plan in &plan_views {
        if plan.leased.generation() == 0 {
            return Err(invalid_operation(
                "segment Plan buffer has no committed generation",
            ));
        }
        if !validate_buffer(
            runtime,
            plan.leased.buffer(),
            plan.leased.committed_descriptor(),
            &mut seen,
        )? {
            identity.record_segment_miss_reason(
                SegmentPreparationMissReason::ImmutableCapabilityUnavailable,
            );
            return Ok(None);
        }
    }
    if let Some(backing) = &backing {
        for binding in backing.bindings() {
            if !validate_buffer(runtime, binding.buffer(), binding.descriptor(), &mut seen)? {
                identity.record_segment_miss_reason(
                    SegmentPreparationMissReason::ImmutableCapabilityUnavailable,
                );
                return Ok(None);
            }
        }
        for (index, request) in requests.iter().enumerate() {
            if request.expected.storage_profile.view() == DynamicStorageView::Contiguous
                && backing.window(index).unwrap().physical_regions().len() != 1
            {
                return Err(invalid_operation(
                    "contiguous segment resource spans multiple physical bindings",
                ));
            }
        }
    }
    drop(metadata_stage);
    drop(backing_stage);
    let node_stage = SubmissionWaveDispatchStageTimer::start(
        timing_sink,
        SubmissionWaveDispatchStage::SegmentNodeDependenciesAndRegions,
    );
    let owner_view_mode = runtime.segment_binding_owner_view_mode();
    let mut owners = Vec::new();
    if owner_view_mode == SegmentBindingOwnerViewMode::Indexed {
        let owner_count = plan_views
            .len()
            .checked_add(backing.as_ref().map_or(0, |batch| batch.bindings().len()))
            .ok_or_else(|| invalid_operation("segment owner table length overflows"))?;
        owners
            .try_reserve_exact(owner_count)
            .map_err(|_| invalid_operation("segment owner view allocation failed"))?;
        owners.extend(plan_views.iter().map(|plan| SegmentBindingOwnerView {
            buffer: plan.leased.buffer(),
            retention: &plan.retention,
        }));
        if let Some(backing) = &backing {
            owners.extend(
                backing
                    .bindings()
                    .iter()
                    .map(|binding| SegmentBindingOwnerView {
                        buffer: binding.buffer(),
                        retention: binding.retention_ref(),
                    }),
            );
        }
    }
    let mut scopes = Vec::with_capacity(recipe.nodes.len());
    let mut patches = Vec::with_capacity(recipe.nodes.len());
    for (compiled, windows) in recipe.nodes.iter().zip(node_windows) {
        let node = resolved
            .execution_plan()
            .payload()
            .nodes()
            .get(compiled.node_index)
            .ok_or_else(|| invalid_operation("segment node is absent from the current Plan"))?;
        let scope = Arc::new(());
        let mut dependencies = Vec::with_capacity(compiled.dependencies.len());
        for dependency in &compiled.dependencies {
            let plan_view = |resource_index: usize| -> Result<&SegmentPlanResourceView<'_, R::Buffer>, VNextError> {
                match fresh.get(resource_index).and_then(|rows| rows.first()) {
                    Some(FreshResource::Plan(index)) => Ok(&plan_views[*index]),
                    _ => Err(invalid_operation("segment dependency is not a fresh Plan resource")),
                }
            };
            dependencies.push(Some(issue_segment_plan_dependency(
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
        let regions = windows
            .into_iter()
            .map(|window| -> Result<_, VNextError> {
                let source = match window.source {
                    RegionSource::Plan {
                        index,
                        offset,
                        length,
                    } => match owner_view_mode {
                        SegmentBindingOwnerViewMode::Legacy => {
                            SegmentBindingPhysicalSource::Legacy(vec![
                                SegmentBindingPhysicalRegion {
                                    buffer: plan_views[index].leased.buffer(),
                                    physical_range: offset..offset + length,
                                    logical_offset_bytes: 0,
                                    retention: plan_views[index].retention.clone(),
                                },
                            ])
                        }
                        SegmentBindingOwnerViewMode::Indexed => {
                            SegmentBindingPhysicalSource::Plan {
                                owner_index: SegmentBindingOwnerIndex(index),
                                buffer: plan_views[index].leased.buffer(),
                                physical_range: offset..offset + length,
                                retention: &plan_views[index].retention,
                            }
                        }
                    },
                    RegionSource::Dynamic(index) => {
                        let current = backing
                            .as_ref()
                            .and_then(|batch| batch.window(index))
                            .ok_or_else(|| invalid_operation("segment dynamic window is absent"))?;
                        match owner_view_mode {
                            SegmentBindingOwnerViewMode::Legacy => {
                                SegmentBindingPhysicalSource::Legacy(
                                    current
                                        .physical_regions()
                                        .map(|region| {
                                            let (buffer, physical_range, retention) =
                                                region.buffer_and_physical_range();
                                            SegmentBindingPhysicalRegion {
                                                buffer,
                                                physical_range,
                                                logical_offset_bytes: region.logical_offset_bytes(),
                                                retention,
                                            }
                                        })
                                        .collect(),
                                )
                            }
                            SegmentBindingOwnerViewMode::Indexed => {
                                SegmentBindingPhysicalSource::Dynamic {
                                    window: current,
                                    owner_base: plan_views.len(),
                                }
                            }
                        }
                    }
                };
                Ok(PreparedSegmentBindingRegion {
                    descriptor: window.descriptor,
                    storage_kind: window.storage_kind,
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        patches.push(PreparedSegmentBindingNode {
            node_index: u32::try_from(compiled.node_index)
                .map_err(|_| invalid_operation("segment node index exceeds u32"))?,
            declaration: &compiled.declaration,
            program_binding: wave
                .claimed_backing()
                .program_binding_node(compiled.node_index),
            work_shape: wave.nodes()[compiled.node_index].work_shape(),
            regions,
            dependencies,
        });
        scopes.push(scope);
    }
    drop(node_stage);
    let _backend_stage = SubmissionWaveDispatchStageTimer::start(
        timing_sink,
        SubmissionWaveDispatchStage::SegmentBackendEncodeAndValidate,
    );
    let Some(encoded) = runtime
        .encode_segment_bindings(PreparedSegmentBindingPatch {
            owner_view_mode,
            owners,
            nodes: patches,
        })
        .map_err(|error| {
            invalid_operation(format!("whole segment binding encode failed: {error}"))
        })?
    else {
        identity.record_segment_miss_reason(SegmentPreparationMissReason::UnsupportedEncoder);
        return Ok(None);
    };
    if encoded.len() != recipe.nodes.len()
        || encoded
            .iter()
            .zip(&recipe.nodes)
            .any(|(encoded, compiled)| encoded.node_index as usize != compiled.node_index)
    {
        return Err(invalid_operation(
            "segment encoder returned a different node sequence",
        ));
    }
    let facts = SegmentPreparationFacts {
        dynamic_resource_requests: requests.len() as u64,
        unique_physical_buffers: seen.len() as u64,
    };
    Ok(Some((
        encoded
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
        facts,
    )))
}
