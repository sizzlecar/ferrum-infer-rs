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
    // Owned physical capacity sealed into this version, not the current
    // logical source frontier. The values closure retains topology checks.
    pub state_bytes: u64,
    pub maximum_state_bytes: u64,
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
    let scatter = runtime
        .binding_scatter
        .as_ref()
        .ok_or_else(|| invalid("resident scatter module is absent"))?;
    // Reuse the sink at cold installation or append refresh. Its temporary
    // transport allocation/synchronization is real refresh cost; it is never
    // enqueued and is dropped after freezing the immutable packet.
    let mut command = sink.finish(
        owners,
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
    let mut controls = Vec::with_capacity(rules.len());
    for rule in rules {
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
        dependency_count,
        dependencies,
        upload_plan,
    };
    drop(command);
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
        tables.push(table);
    }
    let owners = Arc::new(CudaSegmentOwnerTable::physical(tables));
    let mut sink = CudaBindingProgramSink::new();
    let mut rules = Vec::with_capacity(data.controls.len());
    for node in &data.projection.nodes {
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
            let patch = data.controls.iter().find(|patch| {
                patch.rule.node_index as usize == node.binding.node_index()
                    && patch.rule.write_index == index
            });
            let payload = if let Some(patch) = patch {
                let Some((values, bytes)) = (patch.rule.values)(work)? else {
                    return Ok(None);
                };
                let row = rows
                    .get_mut(index)
                    .ok_or_else(|| invalid("resident refresh causal row is absent"))?;
                let capacity = row.refreshed_capacity(owners.physical_tables())?;
                if bytes == 0
                    || bytes > capacity.length_bytes()
                    || capacity.length_bytes() > patch.rule.maximum_state_bytes
                {
                    return Ok(None);
                }
                *row = capacity;
                let addresses = (0..row.len())
                    .map(|i| {
                        row.region(owners.physical_tables(), i)
                            .map(|region| region.device_ptr())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut rule = patch.rule.clone();
                rule.state_bytes = row.length_bytes();
                rules.push(rule);
                (patch.rule.payload)(values, &addresses)?
            } else {
                data.packet
                    .get(write.payload.clone())
                    .ok_or_else(|| invalid("resident refresh payload is outside packet"))?
                    .into()
            };
            sink.write(write.destination_offset_bytes, payload.len(), |target| {
                target.extend_from_slice(&payload);
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
        if !patch_current_controls(
            &mut packet,
            patch.offset,
            patch.rule.state_bytes,
            bytes,
            values,
        )? {
            return Ok(None);
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

fn patch_current_controls(
    packet: &mut [u8],
    offset: usize,
    sealed_capacity: u64,
    current_required: u64,
    values: [i32; 6],
) -> Result<bool, CudaDeviceRuntimeError> {
    if current_required == 0 || current_required > sealed_capacity {
        return Ok(false);
    }
    let end = offset
        .checked_add(6 * std::mem::size_of::<i32>())
        .ok_or_else(|| CudaDeviceRuntimeError::contract("resident controls offset overflows"))?;
    let controls = packet
        .get_mut(offset..end)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("resident controls exceed packet"))?;
    for (target, value) in controls.chunks_exact_mut(4).zip(values) {
        target.copy_from_slice(&value.to_ne_bytes());
    }
    Ok(true)
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn resident_capacity_controls_preserve_addresses_and_exact_frontier() {
        const PAGE: u64 = 64 * 1024;
        let mut packet = vec![0xa5; 8 + 24];
        for address in [0x100000_u64, 0x200000, 0x300000, 0x400000] {
            packet.extend_from_slice(&address.to_ne_bytes());
        }
        let address_bytes = packet[32..].to_vec();
        for required_pages in 1..=4 {
            let current_entries = required_pages as i32;
            let values = [current_entries, 65, 1, 66, 0, 0];
            assert!(patch_current_controls(
                &mut packet,
                8,
                4 * PAGE,
                required_pages * PAGE,
                values,
            )
            .unwrap());
            assert_eq!(&packet[..8], &[0xa5; 8]);
            assert_eq!(&packet[32..], address_bytes);
            for (index, value) in values.into_iter().enumerate() {
                assert_eq!(&packet[8 + index * 4..12 + index * 4], &value.to_ne_bytes());
            }
        }
        let sealed = packet.clone();
        assert!(!patch_current_controls(&mut packet, 8, 4 * PAGE, 5 * PAGE, [9; 6]).unwrap());
        assert_eq!(packet, sealed);
        assert!(!patch_current_controls(&mut packet, 8, 4 * PAGE, 0, [9; 6]).unwrap());
        assert_eq!(packet, sealed);
        // Exact allocation policy keeps its original boundary.
        assert!(!patch_current_controls(&mut packet, 8, PAGE, 2 * PAGE, [9; 6]).unwrap());
        assert_eq!(packet, sealed);
        assert!(patch_current_controls(&mut packet, usize::MAX, PAGE, PAGE, [9; 6]).is_err());
        assert!(patch_current_controls(&mut packet, sealed.len(), PAGE, PAGE, [9; 6]).is_err());
        assert_eq!(packet, sealed);
    }
}
