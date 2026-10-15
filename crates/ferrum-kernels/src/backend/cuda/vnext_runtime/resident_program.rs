//! Cold packet lowering and fresh scalar patching for a complete resident use.
//! Graph ownership remains in the stream cache. No old Step/Wave is retained.
use super::*;
use ferrum_interfaces::vnext::{
    BatchWorkShape, DeviceEncodedResidentDecode, DeviceResidentDecodeRequest,
    RetainedPlanDependencyIdentity,
};

pub(crate) struct ResidentControlRule {
    pub node_index: u32,
    pub write_index: usize,
    pub values: Box<
        dyn Fn(&BatchWorkShape) -> Result<Option<[i32; 6]>, CudaDeviceRuntimeError> + Send + Sync,
    >,
}

pub(crate) struct ResidentColdParts {
    pub sink: CudaBindingProgramSink,
    pub owners: Arc<CudaSegmentOwnerTable>,
    pub controls: Vec<ResidentControlRule>,
    pub dependency_identities: Vec<RetainedPlanDependencyIdentity>,
    pub dependencies: Vec<CudaDeviceCommand>,
}

struct ControlPatch {
    offset: usize,
    rule: ResidentControlRule,
}

struct ResidentProgramData {
    runtime_instance: u64,
    participants: u32,
    tokens: u64,
    arena: u64,
    packet: Box<[u8]>,
    projection: Arc<CudaBindingProgramProjection>,
    controls: Vec<ControlPatch>,
    dependency_count: usize,
    dependencies: Vec<CudaDeviceCommand>,
    upload_plan: DeviceProgramBindingUploadSnapshot,
}

struct ResidentProgram(Arc<ResidentProgramData>);

pub(super) fn compile(
    runtime: &CudaDeviceRuntime,
    parts: ResidentColdParts,
) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, CudaDeviceRuntimeError> {
    let invalid = CudaDeviceRuntimeError::contract;
    if parts.sink.nodes.is_empty()
        || parts
            .sink
            .nodes
            .iter()
            .any(|node| node.numerical_status.is_some())
    {
        return Ok(None);
    }
    let scatter = runtime
        .binding_scatter
        .as_ref()
        .ok_or_else(|| invalid("resident scatter module is absent"))?;
    // The ordinary sink validates and coalesces exactly once. Its temporary
    // transport allocation is never enqueued and is dropped after freezing.
    let mut command = parts.sink.finish(
        parts.owners,
        scatter,
        &runtime.allocation_stream,
        Arc::clone(&runtime.program_binding_upload_counters),
    )?;
    let projection = command
        .binding_program
        .take()
        .ok_or_else(|| invalid("resident cold projection is absent"))?;
    let mut projection = Arc::try_unwrap(projection)
        .map_err(|_| invalid("resident cold projection unexpectedly shared"))?;
    if !projection.compact {
        return Ok(None);
    }
    let executable = command
        .executable
        .as_ref()
        .ok_or_else(|| invalid("resident cold executable is absent"))?;
    let [packet] = executable.host_storage.as_slice() else {
        return Err(invalid("resident cold packet count differs"));
    };
    let packet = packet.clone();
    let first = projection
        .nodes
        .first()
        .ok_or_else(|| invalid("resident cold projection is empty"))?;
    if first.tokens != u64::from(first.participants) {
        return Ok(None);
    }
    let participants = first.participants;
    let tokens = first.tokens;
    let arena = first
        .destination
        .device_ptr
        .checked_sub(first.binding.slot().physical_offset_bytes())
        .ok_or_else(|| invalid("resident arena underflows"))?;
    let mut controls = Vec::with_capacity(parts.controls.len());
    for rule in parts.controls {
        let node = projection
            .nodes
            .iter()
            .find(|n| n.binding.node_index() == rule.node_index as usize)
            .ok_or_else(|| invalid("resident control node is absent"))?;
        let write = node
            .writes
            .get(rule.write_index)
            .ok_or_else(|| invalid("resident control row is absent"))?;
        if write.payload.len() < 24 || write.payload.end > packet.len() {
            return Err(invalid("resident controls exceed the sealed packet"));
        }
        controls.push(ControlPatch {
            offset: write.payload.start,
            rule,
        });
    }
    let mut upload_plan = DeviceProgramBindingUploadSnapshot {
        physical_arena_bytes: first.binding.layout().physical_size_bytes(),
        planned_upload_bytes: packet.len() as u64,
        compact_scatter_preludes: 1,
        ..Default::default()
    };
    for node in &mut projection.nodes {
        upload_plan.logical_arena_bytes += node.destination.length_bytes;
        upload_plan.live_payload_bytes += node
            .writes
            .iter()
            .map(|w| w.live_payload_bytes)
            .sum::<u64>();
        // Allocation identity and ranges are cold facts. Current authority is
        // held by each new command's fresh-use retention, never this old scope.
        node.destination._core_retention = None;
        node.destination.reusable_address_scope = None;
        node.reusable_address_scope = None;
    }
    let data = ResidentProgramData {
        runtime_instance: command.runtime_instance,
        participants,
        tokens,
        arena,
        packet,
        projection: Arc::new(projection),
        controls,
        dependency_count: parts.dependency_identities.len(),
        dependencies: parts.dependencies,
        upload_plan,
    };
    drop(command);
    Ok(Some(Arc::new(ResidentProgram(Arc::new(data)))))
}

pub(super) fn encode(
    runtime: &CudaDeviceRuntime,
    program: &(dyn std::any::Any + Send + Sync),
    request: &mut DeviceResidentDecodeRequest<'_>,
) -> Result<Option<DeviceEncodedResidentDecode<CudaDeviceCommand>>, CudaDeviceRuntimeError> {
    let invalid = CudaDeviceRuntimeError::contract;
    let Some(program) = program.downcast_ref::<ResidentProgram>() else {
        return Ok(None);
    };
    let data = &program.0;
    if data.runtime_instance != runtime.runtime_instance {
        return Err(invalid("resident program belongs to another CUDA runtime"));
    }
    let work = request.work_shape();
    if work.participant_token_ranges().len() != data.participants as usize
        || work.immediate_tokens() != data.tokens
        || work
            .participant_token_ranges()
            .iter()
            .any(|p| p.immediate_tokens() != 1)
    {
        return Ok(None);
    }
    // Core has matched this sealed backend and its full dependency closure to
    // the current reservation before issuing this non-constructible request.
    if request
        .dependencies()
        .is_none_or(|d| d.identities().len() != data.dependency_count)
    {
        return Err(invalid("resident fresh dependency closure differs"));
    }
    let mut packet = data.packet.clone();
    for patch in &data.controls {
        let Some(values) = (patch.rule.values)(work)? else {
            return Ok(None);
        };
        for (index, value) in values.into_iter().enumerate() {
            let begin = patch.offset + index * 4;
            packet[begin..begin + 4].copy_from_slice(&value.to_ne_bytes());
        }
    }
    let scatter = runtime
        .binding_scatter
        .as_ref()
        .ok_or_else(|| invalid("resident scatter module is absent"))?;
    let Some(owner) = scatter.prepare_encoded(
        &runtime.allocation_stream,
        packet.len(),
        data.projection.descriptor_count,
    )?
    else {
        return Ok(None);
    };
    let binding_data = Arc::clone(data);
    let retention = request.retention().clone();
    let counters = Arc::clone(&runtime.program_binding_upload_counters);
    let executable = Arc::new(CudaCommandExecutable::static_work(
        Vec::new(),
        vec![packet],
        Box::new(move |stream, _blas, _regions, storage| {
            let _current_use = &retention;
            let [packet] = storage else {
                return Err(invalid("resident binding packet is absent"));
            };
            let mut attempt = counters.attempt(binding_data.upload_plan);
            owner.upload(stream, packet)?;
            attempt.copied(packet.len() as u64, false);
            owner.scatter(stream, binding_data.arena)?;
            attempt.scatter_dispatched();
            attempt.succeeded();
            Ok(())
        }),
    ));
    let bindings = CudaDeviceCommand {
        runtime_instance: data.runtime_instance,
        operation: "vnext_resident_program_binding_compact_scatter_v1",
        batching_form: DeviceBatchingForm::ParticipantLoop,
        participant_start: 0,
        participant_count: data.participants,
        token_count: data.tokens,
        compute_dispatch_count: 1,
        transfer_command_count: 1,
        executable: Some(executable),
        fence_dependencies: Vec::new(),
        binding_owner_tables: Vec::new(),
        replay_key: None,
        reusable_address_scope: None,
        replay_gap_reason: None,
        program_binding_patch: None,
        binding_program: Some(Arc::clone(&data.projection)),
        reusable_execution: None,
        completion_checks: Vec::new(),
    };
    let dependency_data = Arc::clone(data);
    let dependency_retention = request.retention().clone();
    let prefix = CudaDeviceCommand {
        runtime_instance: data.runtime_instance,
        operation: "vnext_resident_retained_plan_dependencies_v1",
        batching_form: DeviceBatchingForm::Scalar,
        participant_start: 0,
        participant_count: 0,
        token_count: 0,
        compute_dispatch_count: 0,
        transfer_command_count: 0,
        executable: Some(Arc::new(CudaCommandExecutable {
            regions: Vec::new(),
            host_storage: Vec::new(),
            work_declaration: CudaWorkDeclaration::ConditionalDependency,
            enqueue: Mutex::new(CudaEnqueueAction::ConditionalDependency(Box::new(
                move |stream, blas, _regions, _storage| {
                    let _current_use = &dependency_retention;
                    let (mut compute, mut transfer, mut waits) = (0u64, 0u64, 0u64);
                    for command in &dependency_data.dependencies {
                        let CudaEnqueuedWork::Dependency(work) = command.enqueue(stream, blas)?
                        else {
                            return Err(invalid("resident prefix contains non-dependency work"));
                        };
                        let (c, t, w) = work.counts();
                        compute = compute
                            .checked_add(c)
                            .ok_or_else(|| invalid("resident compute count overflows"))?;
                        transfer = transfer
                            .checked_add(t)
                            .ok_or_else(|| invalid("resident transfer count overflows"))?;
                        waits = waits
                            .checked_add(w)
                            .ok_or_else(|| invalid("resident wait count overflows"))?;
                    }
                    Ok(CudaEnqueuedWork::Dependency(CudaDependencyWork::Batch {
                        compute,
                        transfer,
                        waits,
                    }))
                },
            ))),
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
    };
    let dependencies = request
        .take_dependencies()
        .ok_or_else(|| invalid("resident dependency batch already consumed"))?;
    Ok(Some(DeviceEncodedResidentDecode::new(
        dependencies.encode((data.dependency_count != 0).then_some(prefix)),
        bindings,
    )))
}
