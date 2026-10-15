//! Cold packet lowering and fresh scalar patching for a complete resident use.
//! Graph ownership remains in the stream cache. No old Step/Wave is retained.
use super::*;
use ferrum_interfaces::vnext::{
    BatchWorkShape, DeviceEncodedResidentDecode, DeviceResidentDecodeRefresh,
    DeviceResidentDecodeRequest, RetainedPlanDependencyIdentity,
};

#[derive(Clone)]
pub(crate) struct ResidentControlRule {
    pub node_index: u32,
    pub write_index: usize,
    pub state_bytes: u64,
    pub values: Arc<
        dyn Fn(&BatchWorkShape) -> Result<Option<([i32; 6], u64)>, CudaDeviceRuntimeError>
            + Send
            + Sync,
    >,
    pub payload:
        Arc<dyn Fn([i32; 6], &[u64]) -> Result<Box<[u8]>, CudaDeviceRuntimeError> + Send + Sync>,
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
    // Cold node/write lookup, independent of any fresh authority or frontier.
    control_indices: Vec<Vec<Option<usize>>>,
    dependency_count: usize,
    dependencies: Arc<[CudaDeviceCommand]>,
    upload_plan: DeviceProgramBindingUploadSnapshot,
}

struct ResidentProgram(Arc<ResidentProgramData>);

pub(super) fn compile(
    runtime: &CudaDeviceRuntime,
    parts: ResidentColdParts,
) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, CudaDeviceRuntimeError> {
    seal(
        runtime,
        parts.sink,
        parts.owners,
        parts.controls,
        parts.dependency_identities.len(),
        parts.dependencies.into(),
    )
}

fn seal(
    runtime: &CudaDeviceRuntime,
    sink: CudaBindingProgramSink,
    owners: Arc<CudaSegmentOwnerTable>,
    rules: Vec<ResidentControlRule>,
    dependency_count: usize,
    dependencies: Arc<[CudaDeviceCommand]>,
) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, CudaDeviceRuntimeError> {
    let invalid = CudaDeviceRuntimeError::contract;
    if sink.nodes.is_empty()
        || sink
            .nodes
            .iter()
            .any(|node| node.numerical_status.is_some())
    {
        return Ok(None);
    }
    if runtime.binding_scatter.is_none() {
        return Err(invalid("resident scatter module is absent"));
    }
    // Freeze the same fully validated host representation as ordinary lowering.
    // The real transport is allocated once by encode for the current use; cold
    // installation and append refresh neither allocate nor synchronize a dummy.
    let FrozenBindingProgram {
        nodes,
        owners,
        packet,
        arena_ptr: arena,
        runtime_instance,
        descriptor_count,
        mut upload_plan,
        ..
    } = sink.freeze(owners)?;
    let mut projection = CudaBindingProgramProjection {
        nodes,
        owners,
        descriptor_count,
        compact: true,
    };
    let first = projection
        .nodes
        .first()
        .ok_or_else(|| invalid("resident cold projection is empty"))?;
    if first.tokens != u64::from(first.participants) {
        return Ok(None);
    }
    let participants = first.participants;
    let tokens = first.tokens;
    let mut controls = Vec::with_capacity(rules.len());
    let mut control_indices: Vec<Vec<Option<usize>>> = projection
        .nodes
        .iter()
        .map(|node| vec![None; node.writes.len()])
        .collect();
    for rule in rules {
        let node_index = projection
            .nodes
            .binary_search_by_key(&(rule.node_index as usize), |node| {
                node.binding.node_index()
            })
            .map_err(|_| invalid("resident control node is absent"))?;
        let node = &projection.nodes[node_index];
        let write = node
            .writes
            .get(rule.write_index)
            .ok_or_else(|| invalid("resident control row is absent"))?;
        if write.payload.len() < 24 || write.payload.end > packet.len() {
            return Err(invalid("resident controls exceed the sealed packet"));
        }
        let slot = &mut control_indices[node_index][rule.write_index];
        if slot.replace(controls.len()).is_some() {
            return Err(invalid("resident control row is repeated"));
        }
        controls.push(ControlPatch {
            offset: write.payload.start,
            rule,
        });
    }
    upload_plan.planned_upload_bytes = packet.len() as u64;
    upload_plan.compact_scatter_preludes = 1;
    upload_plan.compact_scatter_sparse_fallback_preludes = 0;
    for node in &mut projection.nodes {
        // Allocation identity and ranges are cold facts. Current authority is
        // held by each new command's fresh-use retention, never this old scope.
        node.destination._core_retention = None;
        node.destination.reusable_address_scope = None;
        node.reusable_address_scope = None;
    }
    let data = ResidentProgramData {
        runtime_instance,
        participants,
        tokens,
        arena,
        packet,
        projection: Arc::new(projection),
        controls,
        control_indices,
        dependency_count,
        dependencies,
        upload_plan,
    };
    Ok(Some(Arc::new(ResidentProgram(Arc::new(data)))))
}

pub(super) fn refresh(
    runtime: &CudaDeviceRuntime,
    previous: &(dyn std::any::Any + Send + Sync),
    refresh: &DeviceResidentDecodeRefresh<'_, CudaDeviceBuffer>,
) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, CudaDeviceRuntimeError> {
    let invalid = CudaDeviceRuntimeError::contract;
    let Some(previous) = previous.downcast_ref::<ResidentProgram>() else {
        return Ok(None);
    };
    let data = &previous.0;
    if data.runtime_instance != runtime.runtime_instance {
        return Err(invalid("resident refresh belongs to another CUDA runtime"));
    }
    let work = refresh.work_shape();
    if refresh.previous_tables().len() != refresh.current_tables().len()
        || work.participant_token_ranges().len() != data.participants as usize
        || work.immediate_tokens() != data.tokens
        || work
            .participant_token_ranges()
            .iter()
            .any(|p| p.immediate_tokens() != 1)
    {
        return Ok(None);
    }
    // Preserve the old projection's ordinal table order. Core has proved exact
    // physical prefix identity; we only replace its owning immutable version.
    let mut tables = Vec::with_capacity(data.projection.owners.physical_tables().len());
    let mut changed_tables = Vec::with_capacity(tables.capacity());
    for old in data.projection.owners.physical_tables() {
        let Some(index) = refresh.previous_tables().iter().position(|table| {
            table.backend_bindings().is_some_and(|backend| {
                Arc::as_ptr(backend) as *const () == Arc::as_ptr(old) as *const ()
            })
        }) else {
            return Err(invalid("resident refresh omitted a physical owner"));
        };
        let backend = refresh.current_tables()[index]
            .backend_bindings()
            .ok_or_else(|| invalid("resident refresh backend is absent"))?;
        let table = Arc::clone(backend)
            .downcast::<sequence_physical::CudaSequencePhysicalTable>()
            .map_err(|_| invalid("resident refresh has another physical provider"))?;
        changed_tables.push(!Arc::ptr_eq(old, &table));
        tables.push(table);
    }
    let owners = Arc::new(CudaSegmentOwnerTable::physical(tables));
    let mut sink = CudaBindingProgramSink::new();
    let mut rules = Vec::with_capacity(data.controls.len());
    for (node_index, node) in data.projection.nodes.iter().enumerate() {
        let CudaBindingProgramRanges::Physical(old_rows) = &node.ranges else {
            return Ok(None);
        };
        let mut rows = old_rows.clone();
        sink.begin_node(
            node.operation,
            node.binding.clone(),
            node.destination.clone(),
            node.participants,
            node.tokens,
        )?;
        for (index, write) in node.writes.iter().enumerate() {
            let patch =
                data.control_indices[node_index][index].map(|control| &data.controls[control]);
            let replacement = if let Some(patch) = patch {
                let Some((values, bytes)) = (patch.rule.values)(work)? else {
                    return Ok(None);
                };
                let row = rows
                    .get_mut(index)
                    .ok_or_else(|| invalid("resident refresh causal row is absent"))?;
                let changed = *changed_tables
                    .get(row.table_index())
                    .ok_or_else(|| invalid("resident refresh causal owner is absent"))?;
                let mut rule = patch.rule.clone();
                rule.state_bytes = bytes;
                rules.push(rule);
                // A different physical owner or live page window needs current
                // addresses. Otherwise retain the validated address payload;
                // encode still patches all six fresh controls for every row.
                if changed || bytes != patch.rule.state_bytes {
                    *row = row.refreshed(owners.physical_tables(), bytes)?;
                    let addresses = (0..row.len())
                        .map(|i| {
                            row.region(owners.physical_tables(), i)
                                .map(|region| region.device_ptr())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Some((patch.rule.payload)(values, &addresses)?)
                } else {
                    None
                }
            } else {
                None
            };
            let payload = match &replacement {
                Some(payload) => payload.as_ref(),
                None => data
                    .packet
                    .get(write.payload.clone())
                    .ok_or_else(|| invalid("resident refresh payload is outside packet"))?,
            };
            sink.write(write.destination_offset_bytes, payload.len(), |target| {
                target.extend_from_slice(payload);
                Ok(())
            })?;
        }
        sink.end_physical_node(rows, None)?;
    }
    // Re-seal only on growth. This validates packet destinations and current
    // physical ranges; it neither encodes compute nor captures a graph.
    seal(
        runtime,
        sink,
        owners,
        rules,
        data.dependency_count,
        Arc::clone(&data.dependencies),
    )
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
        let Some((values, bytes)) = (patch.rule.values)(work)? else {
            return Ok(None);
        };
        if bytes != patch.rule.state_bytes {
            return Ok(None);
        }
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
                    for command in dependency_data.dependencies.iter() {
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
