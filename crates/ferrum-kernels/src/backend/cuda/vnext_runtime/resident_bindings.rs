//! Physical resident binding templates. Only a fresh Core permit can activate
//! one; templates contain neither a wave retention nor a logical slot lease.
use super::*;
use ferrum_interfaces::vnext::{
    BatchWorkShape, EncodedReusableExecutionBindings, EncodedSegmentBindingNode,
    EncodedSegmentBindingProgram, FreshResidentSegmentPermit, PreparedSegmentBindingFrame,
    SegmentBindingOwnerView,
};

pub(crate) struct ResidentControlRule {
    pub node_index: u32,
    pub values: Box<dyn Fn(&BatchWorkShape) -> Result<Vec<[i32; 6]>, String> + Send + Sync>,
}

pub(crate) struct ResidentPreparation {
    pub sink: CudaBindingProgramSink,
    pub owners: Arc<CudaSegmentOwnerTable>,
    pub dependencies: Vec<(u32, Vec<(CudaDeviceCommand, Vec<CudaBufferRegion>)>)>,
    pub controls: Vec<ResidentControlRule>,
}

struct PhysicalRegion {
    region: CudaBufferRegion,
    owner: usize,
}

impl PhysicalRegion {
    fn prepare(
        region: &CudaBufferRegion,
        owners: &[SegmentBindingOwnerView<'_, CudaDeviceBuffer>],
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let owner = owners
            .iter()
            .position(|owner| {
                let (buffer, retention) = owner.buffer_and_retention();
                Arc::ptr_eq(&buffer.allocation, &region._allocation)
                    && region
                        ._core_retention
                        .as_ref()
                        .is_some_and(|r| r.same_owners(retention))
            })
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("resident region has no exact fresh owner")
            })?;
        Ok(Self {
            region: region.physical_only(),
            owner,
        })
    }

    fn authorize(
        &self,
        permit: &FreshResidentSegmentPermit<'_>,
    ) -> Result<CudaBufferRegion, CudaDeviceRuntimeError> {
        let retention = permit.owner_retentions().get(self.owner).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("resident region owner is absent from fresh permit")
        })?;
        Ok(CudaBufferRegion {
            _core_retention: Some(retention.clone()),
            reusable_address_scope: retention.reusable_address_scope(),
            ..self.region.clone()
        })
    }
}

struct DependencyTemplate {
    operation: &'static str,
    runtime_instance: u64,
    regions: Vec<PhysicalRegion>,
    enqueue: Arc<Mutex<CudaEnqueueAction>>,
}

impl DependencyTemplate {
    fn authorize(
        &self,
        permit: &FreshResidentSegmentPermit<'_>,
    ) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
        Ok(CudaDeviceCommand {
            runtime_instance: self.runtime_instance,
            operation: self.operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 0,
            executable: Some(Arc::new(CudaCommandExecutable {
                regions: self
                    .regions
                    .iter()
                    .map(|r| r.authorize(permit))
                    .collect::<Result<_, _>>()?,
                host_storage: Vec::new(),
                work_declaration: CudaWorkDeclaration::ConditionalDependency,
                enqueue: Arc::clone(&self.enqueue),
            })),
            fence_dependencies: Vec::new(),
            binding_owner_tables: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            binding_program: None,
            reusable_execution: None,
            completion_checks: Vec::new(),
        })
    }
}

struct ResidentNode {
    node: CudaBindingProgramNode,
    recipe_ordinal: usize,
    destination: PhysicalRegion,
}

struct ControlPatch {
    rule: ResidentControlRule,
    offsets: Vec<usize>,
    entries: Vec<i32>,
}

struct CopySpan {
    source: Range<usize>,
    destination: u64,
    width: usize,
    rows: usize,
    pitch: usize,
}

pub(super) struct ResidentTemplate {
    nodes: Vec<ResidentNode>,
    dependencies: Vec<(u32, Vec<DependencyTemplate>)>,
    owners: Arc<segment_owners::CudaPhysicalOwnerTable>,
    controls: Vec<ControlPatch>,
    packet: Box<[u8]>,
    copies: Arc<Vec<CopySpan>>,
    arena_ptr: u64,
    descriptor_count: u32,
    upload_plan: DeviceProgramBindingUploadSnapshot,
    runtime_instance: u64,
}

pub(super) fn prepare(
    runtime: &CudaDeviceRuntime,
    program: &(dyn std::any::Any + Send + Sync),
    frame: &PreparedSegmentBindingFrame<'_, CudaDeviceBuffer>,
) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, CudaDeviceRuntimeError> {
    let prepared = super::super::vnext_ops::prepare_resident_segment_bindings(program, frame)?;
    let Some(scatter) = runtime.binding_scatter.as_ref() else {
        return Ok(None);
    };
    let command = prepared.sink.finish(
        Arc::clone(&prepared.owners),
        scatter,
        &runtime.allocation_stream,
        Arc::clone(&runtime.program_binding_upload_counters),
    )?;
    // Numerical-status readbacks are intentionally left to the ordinary path.
    if !command.completion_checks.is_empty() {
        return Ok(None);
    }
    let projection = command
        .binding_program
        .as_ref()
        .ok_or_else(|| CudaDeviceRuntimeError::contract("resident prelude projection missing"))?;
    let packet = command
        .executable
        .as_ref()
        .and_then(|e| e.host_storage.first())
        .ok_or_else(|| CudaDeviceRuntimeError::contract("resident prelude packet missing"))?
        .clone();
    let mut dependencies = Vec::with_capacity(prepared.dependencies.len());
    for (index, commands) in prepared.dependencies {
        let mut templates = Vec::with_capacity(commands.len());
        for (command, evidence) in commands {
            let executable = command.executable.as_ref().ok_or_else(|| {
                CudaDeviceRuntimeError::contract("resident dependency executable missing")
            })?;
            templates.push(DependencyTemplate {
                operation: command.operation,
                runtime_instance: command.runtime_instance,
                regions: evidence
                    .iter()
                    .map(|r| PhysicalRegion::prepare(r, frame.owners()))
                    .collect::<Result<_, _>>()?,
                enqueue: Arc::clone(&executable.enqueue),
            });
        }
        dependencies.push((index, templates));
    }
    let mut nodes = Vec::with_capacity(projection.nodes.len());
    for node in &projection.nodes {
        let ordinal = dependencies
            .binary_search_by_key(&(node.binding.node_index() as u32), |n| n.0)
            .map_err(|_| {
                CudaDeviceRuntimeError::contract("resident binding node absent from recipe")
            })?;
        let destination = PhysicalRegion::prepare(&node.destination, frame.owners())?;
        let mut physical = node.clone();
        physical.destination = physical.destination.physical_only();
        nodes.push(ResidentNode {
            node: physical,
            recipe_ordinal: ordinal,
            destination,
        });
    }
    let mut controls = Vec::with_capacity(prepared.controls.len());
    for rule in prepared.controls {
        let node = projection
            .nodes
            .iter()
            .find(|n| n.binding.node_index() == rule.node_index as usize)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("resident causal node missing"))?;
        let mut offsets = Vec::new();
        let mut entries = Vec::new();
        for write in &node.writes {
            if write.payload.len() < 24 {
                return Err(CudaDeviceRuntimeError::contract(
                    "resident causal controls truncated",
                ));
            }
            offsets.push(write.payload.start);
            entries.push(i32::from_ne_bytes(
                packet[write.payload.start..write.payload.start + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
        controls.push(ControlPatch {
            rule,
            offsets,
            entries,
        });
    }
    let count = projection.descriptor_count;
    let mut copies = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let field = |f: usize| -> Result<u64, CudaDeviceRuntimeError> {
            let start = index * 40 + f * 8;
            Ok(u64::from_le_bytes(
                packet
                    .get(start..start + 8)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("resident descriptor truncated")
                    })?
                    .try_into()
                    .unwrap(),
            ))
        };
        let source = checked_usize(field(0)?, "resident source")?;
        let width = checked_usize(field(2)?, "resident width")?;
        let rows = checked_usize(field(3)?, "resident rows")?;
        let end = width
            .checked_mul(rows)
            .and_then(|n| source.checked_add(n))
            .filter(|&n| n <= packet.len())
            .ok_or_else(|| CudaDeviceRuntimeError::contract("resident copy exceeds packet"))?;
        copies.push(CopySpan {
            source: source..end,
            destination: field(1)?,
            width,
            rows,
            pitch: checked_usize(field(4)?, "resident pitch")?,
        });
    }
    let first = &projection.nodes[0];
    let upload_plan = DeviceProgramBindingUploadSnapshot {
        physical_arena_bytes: first.binding.layout().physical_size_bytes(),
        logical_arena_bytes: projection
            .nodes
            .iter()
            .map(|n| n.destination.length_bytes)
            .sum(),
        live_payload_bytes: projection
            .nodes
            .iter()
            .flat_map(|n| &n.writes)
            .map(|w| w.live_payload_bytes)
            .sum(),
        ..Default::default()
    };
    let template = ResidentTemplate {
        arena_ptr: first.destination.device_ptr - first.binding.slot().physical_offset_bytes(),
        runtime_instance: runtime.runtime_instance,
        nodes,
        dependencies,
        owners: prepared.owners.physical_template(),
        controls,
        packet,
        copies: Arc::new(copies),
        descriptor_count: count,
        upload_plan,
    };
    Ok(Some(Arc::new(template)))
}

pub(super) fn encode(
    runtime: &CudaDeviceRuntime,
    template: &(dyn std::any::Any + Send + Sync),
    permit: &mut FreshResidentSegmentPermit<'_>,
) -> Result<Option<EncodedSegmentBindingProgram<CudaDeviceCommand>>, CudaDeviceRuntimeError> {
    let template = template.downcast_ref::<ResidentTemplate>().ok_or_else(|| {
        CudaDeviceRuntimeError::contract("resident template has another backend type")
    })?;
    if template.runtime_instance != runtime.runtime_instance
        || permit.node_count() != template.dependencies.len()
    {
        return Err(CudaDeviceRuntimeError::contract(
            "resident template runtime or node coverage changed",
        ));
    }
    let mut packet = template.packet.clone();
    for patch in &template.controls {
        let values =
            (patch.rule.values)(permit.work_shape()).map_err(CudaDeviceRuntimeError::contract)?;
        if values.len() != patch.offsets.len()
            || values
                .iter()
                .zip(&patch.entries)
                .any(|(v, old)| v[0] != *old)
        {
            return Ok(None); // KV table shape changed: reinstall through the full path.
        }
        for ((values, offset), _) in values.iter().zip(&patch.offsets).zip(&patch.entries) {
            patch_controls(&mut packet, *offset, *values)?;
        }
    }
    let owners = template.owners.authorize(permit.owner_frame())?;
    let mut projection_nodes = Vec::with_capacity(template.nodes.len());
    for stored in &template.nodes {
        let mut node = stored.node.clone();
        node.destination = stored.destination.authorize(permit)?;
        let current_binding = permit
            .program_binding(stored.recipe_ordinal)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("resident binding slot absent"))?;
        if current_binding.node_index() != stored.node.binding.node_index()
            || current_binding.plan_hash() != stored.node.binding.plan_hash()
            || current_binding.layout() != stored.node.binding.layout()
            || current_binding.lane_slot_identity() != stored.node.binding.lane_slot_identity()
            || current_binding.slot() != stored.node.binding.slot()
        {
            return Ok(None);
        }
        node.binding = current_binding.clone();
        node.tokens = permit.work_shape().immediate_tokens();
        projection_nodes.push(node);
    }
    let node_indices = projection_nodes
        .iter()
        .map(|n| n.binding.node_index() as u32)
        .collect();
    let command = upload(runtime, template, packet, projection_nodes, owners)?;
    let batch = permit
        .take_dependency_batch()
        .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
    if batch.len() != template.dependencies.len() {
        return Err(CudaDeviceRuntimeError::contract(
            "resident dependency batch coverage changed",
        ));
    }
    let mut encoded = Vec::with_capacity(batch.len());
    for ((node_index, commands), authorities) in template.dependencies.iter().zip(batch) {
        if commands.len() != authorities.len() {
            return Err(CudaDeviceRuntimeError::contract(
                "resident dependency count changed",
            ));
        }
        let mut bindings = EncodedReusableExecutionBindings::empty();
        for (command, authority) in commands.iter().zip(authorities) {
            bindings = bindings
                .with_retained_plan_dependency(authority.encode(command.authorize(permit)?));
        }
        encoded.push(EncodedSegmentBindingNode {
            node_index: *node_index,
            bindings,
        });
    }
    Ok(Some(EncodedSegmentBindingProgram {
        nodes: encoded,
        program_bindings: vec![command],
        program_binding_nodes: node_indices,
    }))
}

fn patch_controls(
    packet: &mut [u8],
    offset: usize,
    values: [i32; 6],
) -> Result<(), CudaDeviceRuntimeError> {
    let end = offset
        .checked_add(24)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("resident control offset overflows"))?;
    let bytes = packet
        .get_mut(offset..end)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("resident control outside packet"))?;
    for (destination, value) in bytes.chunks_exact_mut(4).zip(values) {
        destination.copy_from_slice(&value.to_ne_bytes());
    }
    Ok(())
}

fn upload(
    runtime: &CudaDeviceRuntime,
    template: &ResidentTemplate,
    packet: Box<[u8]>,
    nodes: Vec<CudaBindingProgramNode>,
    owners: Arc<CudaSegmentOwnerTable>,
) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
    let scatter = runtime
        .binding_scatter
        .as_ref()
        .ok_or_else(|| CudaDeviceRuntimeError::contract("resident scatter capability missing"))?;
    let scratch = scatter.prepare_encoded(
        &runtime.allocation_stream,
        packet.len(),
        template.descriptor_count,
    )?;
    let compact = scratch.is_some();
    let mut upload_plan = template.upload_plan;
    upload_plan.compact_scatter_preludes = u64::from(compact);
    upload_plan.compact_scatter_sparse_fallback_preludes = u64::from(!compact);
    upload_plan.planned_upload_bytes = if compact {
        packet.len() as u64
    } else {
        upload_plan.live_payload_bytes
    };
    let count = template.descriptor_count;
    let arena = template.arena_ptr;
    let copies = Arc::clone(&template.copies);
    let counters = Arc::clone(&runtime.program_binding_upload_counters);
    let participants = nodes[0].participants;
    let tokens = nodes[0].tokens;
    let regions = nodes.iter().map(|n| n.destination.clone()).collect();
    let projection = Arc::new(CudaBindingProgramProjection {
        nodes,
        owners: Arc::clone(&owners),
        descriptor_count: count,
        compact,
    });
    let executable = Arc::new(CudaCommandExecutable::static_work(
        regions,
        vec![packet],
        Box::new(move |stream, _, _, storage| {
            let [packet] = storage else {
                return Err(CudaDeviceRuntimeError::contract(
                    "resident upload packet missing",
                ));
            };
            let mut attempt = counters.attempt(upload_plan);
            if let Some(scratch) = &scratch {
                scratch.upload(stream, packet)?;
                attempt.copied(packet.len() as u64, false);
                scratch.scatter(stream, arena)?;
                attempt.scatter_dispatched();
            } else {
                for copy in copies.iter() {
                    let payload = &packet[copy.source.clone()];
                    let destination = arena + copy.destination;
                    if copy.rows == 1 {
                        unsafe {
                            cudarc::driver::result::memcpy_htod_async(
                                destination,
                                payload,
                                stream.cu_stream(),
                            )
                        }
                        .map_err(|e| CudaDeviceRuntimeError::driver("resident sparse upload", e))?;
                    } else {
                        let descriptor = cudarc::driver::sys::CUDA_MEMCPY2D {
                            srcXInBytes: 0,
                            srcY: 0,
                            srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                            srcHost: payload.as_ptr().cast(),
                            srcDevice: 0,
                            srcArray: std::ptr::null_mut(),
                            srcPitch: copy.width,
                            dstXInBytes: 0,
                            dstY: 0,
                            dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                            dstHost: std::ptr::null_mut(),
                            dstDevice: destination,
                            dstArray: std::ptr::null_mut(),
                            dstPitch: copy.pitch,
                            WidthInBytes: copy.width,
                            Height: copy.rows,
                        };
                        unsafe {
                            cudarc::driver::sys::cuMemcpy2DAsync_v2(&descriptor, stream.cu_stream())
                        }
                        .result()
                        .map_err(|e| {
                            CudaDeviceRuntimeError::driver("resident strided upload", e)
                        })?;
                    }
                    attempt.copied(payload.len() as u64, copy.rows != 1);
                }
            }
            attempt.succeeded();
            Ok(())
        }),
    ));
    Ok(CudaDeviceCommand {
        runtime_instance: runtime.runtime_instance,
        operation: if compact {
            "vnext_program_binding_compact_scatter_v1"
        } else {
            "vnext_program_binding_prelude"
        },
        batching_form: DeviceBatchingForm::ParticipantLoop,
        participant_start: 0,
        participant_count: participants,
        token_count: tokens,
        compute_dispatch_count: u64::from(compact),
        transfer_command_count: if compact { 1 } else { u64::from(count) },
        executable: Some(executable),
        fence_dependencies: Vec::new(),
        binding_owner_tables: vec![owners],
        replay_key: None,
        reusable_address_scope: None,
        replay_gap_reason: None,
        program_binding_patch: None,
        binding_program: Some(projection),
        reusable_execution: None,
        completion_checks: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::patch_controls;
    #[test]
    fn resident_controls_change_only_the_declared_row_and_reject_invalid_ranges() {
        let mut packet = vec![0x5a; 40];
        patch_controls(&mut packet, 8, [2, 17, 1, 18, 0, 0]).unwrap();
        assert_eq!(&packet[..8], &[0x5a; 8]);
        assert_eq!(&packet[32..], &[0x5a; 8]);
        assert_eq!(i32::from_ne_bytes(packet[12..16].try_into().unwrap()), 17);
        assert!(patch_controls(&mut packet, 17, [0; 6]).is_err());
        assert!(patch_controls(&mut packet, usize::MAX, [0; 6]).is_err());
    }
}
