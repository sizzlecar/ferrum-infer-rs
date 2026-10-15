//! A complete cold decode program plus one fresh resource use. The hot path
//! constructs neither an ordinary Step/Wave nor per-node binding frames.
use super::buffer_view::sequence_execution_shape;
use super::compiled_identity::SubmissionWaveParticipantIdentitySeed;
use super::dispatch::encode_submission_wave_inputs;
use super::dispatch_contract::{
    BoundDeviceSubmissionAttribution, SubmissionWaveDispatchStageTimer,
};
use super::foundation::invalid_operation;
use super::retained_dependency::SealedResidentPlanDependencies;
use super::segment_compile::{CompiledSegmentBindingRecipe, CompiledSegmentResourceSource};
use crate::vnext::resource::SegmentPlanResourceRequest;
use crate::vnext::*;
use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex};

// Emit the existing first-only parent callback once, excluding interleaved
// initialization/input preparation without changing execution order.
struct ResidentProviderTimer<'a, S: SubmissionWaveDispatchTimingSink> {
    sink: &'a S,
    elapsed: std::time::Duration,
    started: Option<std::time::Instant>,
}
impl<'a, S: SubmissionWaveDispatchTimingSink> ResidentProviderTimer<'a, S> {
    fn start(sink: &'a S) -> Self {
        Self {
            sink,
            elapsed: std::time::Duration::ZERO,
            started: S::ENABLED.then(std::time::Instant::now),
        }
    }
    fn pause(&mut self) {
        if let Some(start) = self.started.take() {
            self.elapsed = self.elapsed.saturating_add(start.elapsed());
        }
    }
    fn resume(&mut self) {
        self.started = S::ENABLED.then(std::time::Instant::now);
    }
}
impl<S: SubmissionWaveDispatchTimingSink> Drop for ResidentProviderTimer<'_, S> {
    fn drop(&mut self) {
        if S::ENABLED && !std::thread::panicking() {
            self.pause();
            self.sink.record(
                SubmissionWaveDispatchStage::ProviderNodeEncode,
                self.elapsed,
            );
        }
    }
}

struct SequenceWindowRule {
    ordinal: usize,
    descriptor: DynamicResourceDescriptor,
    minimum: u64,
    maximum: Option<u64>,
}

struct ResidentDecodeVersion<B> {
    backend: Arc<dyn Any + Send + Sync>,
    tables: Vec<Arc<SequencePhysicalBindingTable<B>>>,
    facts: (u64, u64, u64),
}

/// Closed constructor: the backend program and dependency closure come from
/// the same fully checked cold physical frame, using the reservation's slots.
/// No old Step, invocation, fresh scope, or completion proof is cached here.
pub struct ResidentDecodeProgram<R: DeviceRuntime> {
    reservation: Arc<ResidentDecodeReservation<R>>,
    topology: CompiledSubmissionWaveIdentity,
    replay: DeviceReusableExecutionProgram,
    _recipe: Arc<CompiledSegmentBindingRecipe>,
    version: Mutex<Arc<ResidentDecodeVersion<R::Buffer>>>,
    fixed_buffer_identities: Vec<usize>,
    dependencies: Arc<SealedResidentPlanDependencies>,
    sequence_windows: Vec<SequenceWindowRule>,
    lane: Arc<ExecutionLane<R>>,
    entry: DeviceReusableExecutionEntryIdentity,
    epoch: u64,
}

impl<R: DeviceRuntime> ResidentDecodeProgram<R> {
    pub fn reservation(&self) -> &Arc<ResidentDecodeReservation<R>> {
        &self.reservation
    }
    pub fn is_idle(&self) -> bool {
        self.reservation.is_idle()
    }
    pub fn matches_participants(&self, participants: &ExecutionBatchParticipants<R>) -> bool {
        self.reservation.matches_participants(participants)
    }
    pub fn revoke(&self) {
        self.reservation.revoke();
    }
    pub fn try_begin_use(
        &self,
        participants: &ExecutionBatchParticipants<R>,
        request: &StepResourceAdmissionRequest,
    ) -> Result<ResidentDecodeDecision<R>, VNextError> {
        let Some((entry, epoch)) = self
            .lane
            .try_segment_binding_entry(self.replay.program_id())?
        else {
            return Ok(ResidentDecodeDecision::Fallback);
        };
        if epoch != self.epoch || !entry.same_entry(&self.entry) {
            return Ok(ResidentDecodeDecision::Fallback);
        }
        self.reservation.try_begin_use(participants, request)
    }

    fn validate_current_windows(
        &self,
        use_: &ResidentDecodeUse<R>,
        tables: &[Arc<SequencePhysicalBindingTable<R::Buffer>>],
    ) -> Result<(), VNextError> {
        if tables.len() != use_.sequence_physical_tables().len()
            || tables
                .iter()
                .zip(use_.sequence_physical_tables())
                .any(|(cold, current)| !Arc::ptr_eq(cold, current))
        {
            return Err(invalid_operation(
                "resident physical table changed after use authorization",
            ));
        }
        let work = use_.work_shape();
        let snapshots = use_.backing_snapshots().collect::<Vec<_>>();
        if snapshots.len() != work.participant_token_ranges().len() {
            return Err(invalid_operation(
                "resident snapshots differ from the current cohort",
            ));
        }
        for rule in &self.sequence_windows {
            for (participant, (snapshot, tokens)) in snapshots
                .iter()
                .zip(work.participant_token_ranges())
                .enumerate()
            {
                let bytes =
                    rule.descriptor
                        .evaluate_request_bytes_for_shape(sequence_execution_shape(
                            snapshot.committed_shape(),
                            tokens.source_token_range().end,
                        )?)?;
                let row = tables[participant]
                    .resource(rule.ordinal)
                    .ok_or_else(|| invalid_operation("resident Sequence row is absent"))?;
                if row.resource_id() != rule.descriptor.base_resource_id()
                    || bytes > row.descriptor().logical_size_bytes()
                    || bytes < rule.minimum
                    || rule.maximum.is_some_and(|maximum| bytes > maximum)
                {
                    return Err(invalid_operation(
                        "resident current logical window exceeds its sealed region rules",
                    ));
                }
            }
        }
        Ok(())
    }

    /// The reservation's fresh-use gate excludes another use while this
    /// immutable binding version is replaced. Old commands/completions retain
    /// their own version; publishing an append never changes their owners.
    fn prepare_current_version(
        &self,
        use_: &ResidentDecodeUse<R>,
    ) -> Result<Option<Arc<ResidentDecodeVersion<R::Buffer>>>, VNextError> {
        let previous = {
            let version = self
                .version
                .lock()
                .map_err(|_| invalid_operation("resident binding version is poisoned"))?;
            Arc::clone(&version)
        };
        let tables = use_.sequence_physical_tables();
        if previous.tables.len() != tables.len() {
            return Ok(None);
        }
        if previous
            .tables
            .iter()
            .zip(tables)
            .all(|(old, current)| Arc::ptr_eq(old, current))
        {
            self.validate_current_windows(use_, &previous.tables)?;
            return Ok(Some(previous));
        }
        if previous
            .tables
            .iter()
            .zip(tables)
            .any(|(old, current)| !Arc::ptr_eq(old, current) && !current.is_append_of(old))
        {
            return Ok(None);
        }
        self.validate_current_windows(use_, tables)?;
        let refresh = DeviceResidentDecodeRefresh::new(use_.work_shape(), &previous.tables, tables);
        let Some(backend) = self
            .lane
            .runtime()
            .refresh_resident_decode_program(previous.backend.as_ref(), &refresh)
            .map_err(|error| {
                invalid_operation(format!("resident append refresh failed: {error}"))
            })?
        else {
            return Ok(None);
        };
        // Exact telemetry is rebuilt only at a physical append, using the
        // original recipe's Sequence rows and retained Plan/Step identities.
        let mut identities = self.fixed_buffer_identities.clone();
        for rule in &self.sequence_windows {
            for table in tables {
                let row = table.resource(rule.ordinal).ok_or_else(|| {
                    invalid_operation("refreshed resident Sequence row is absent")
                })?;
                identities.extend_from_slice(row.metadata_identities());
            }
        }
        identities.sort_unstable();
        identities.dedup();
        let next = Arc::new(ResidentDecodeVersion {
            backend,
            tables: tables.to_vec(),
            facts: (previous.facts.0, previous.facts.1, identities.len() as u64),
        });
        let mut version = self
            .version
            .lock()
            .map_err(|_| invalid_operation("resident binding version is poisoned"))?;
        if !Arc::ptr_eq(&version, &previous) {
            return Err(invalid_operation(
                "resident append raced another binding version",
            ));
        }
        *version = Arc::clone(&next);
        Ok(Some(next))
    }
}

/// Unsupported is possible only before enqueue and leaves the actual use
/// owned by the caller. Unknown submission stays in the existing reaper.
pub enum ResidentDecodeDispatchError<R: DeviceRuntime> {
    Contract(VNextError),
    Unavailable {
        use_: ResidentDecodeUse<R>,
    },
    DefinitelyNotSubmitted {
        failures: Vec<IdentifiedFailure>,
        use_: ResidentDecodeUse<R>,
    },
    SubmissionIndeterminate {
        recovery: IndeterminateSubmissionHandle<R>,
    },
    PostSubmitContract {
        error: VNextError,
        completion: CompletionHandle<R>,
    },
}
impl<R: DeviceRuntime> fmt::Debug for ResidentDecodeDispatchError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(e) => f.debug_tuple("Contract").field(e).finish(),
            Self::Unavailable { .. } => f.write_str("Unavailable"),
            Self::DefinitelyNotSubmitted { .. } => f.write_str("DefinitelyNotSubmitted"),
            Self::SubmissionIndeterminate { .. } => f.write_str("SubmissionIndeterminate"),
            Self::PostSubmitContract { error, .. } => {
                f.debug_tuple("PostSubmitContract").field(error).finish()
            }
        }
    }
}
impl<R: DeviceRuntime> fmt::Display for ResidentDecodeDispatchError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl<R: DeviceRuntime> std::error::Error for ResidentDecodeDispatchError<R> {}

impl OperationDispatch {
    /// The caller installs this only after the bootstrap's successful readback
    /// and normal retirement. Until then the reservation's use gate stays busy.
    pub fn try_compile_resident_decode<R: DeviceRuntime>(
        reservation: Arc<ResidentDecodeReservation<R>>,
        resolved: &dyn ExecutablePlanView,
        wave: &PreparedStepSubmissionWave<R>,
        topology: &CompiledSubmissionWaveIdentity,
        replay: &DeviceReusableExecutionProgram,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<Option<Arc<ResidentDecodeProgram<R>>>, VNextError> {
        if !reservation.matches_bootstrap(wave)? {
            return Err(invalid_operation(
                "resident bootstrap does not own the reserved slots",
            ));
        }
        if wave.purpose() != SubmissionWavePurpose::FullPlan
            || topology.plan_hash() != resolved.execution_plan().plan_hash()
            || topology.lane_id() != lane.id()
            || replay.program_id().plan_hash() != topology.plan_hash()
            || replay.node_count() as usize != topology.node_count()
            || wave.node_count() != topology.node_count()
            || !replay.gaps().is_empty()
            || !replay.eager_boundary_node_indices().is_empty()
        {
            return Ok(None);
        }
        let mut next = 0;
        for segment in replay.segments() {
            if segment.start_node_index() != next {
                return Ok(None);
            }
            next = segment.end_node_index();
        }
        if next != replay.node_count() {
            return Ok(None);
        }
        // Provider-declared scratch zeroing is not silently skipped. Backends
        // with such a contract keep the ordinary path until explicitly lowered.
        if resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .any(|node| {
                node.provider_resources().scratch().is_some_and(|r| {
                    r.reuse_policy() == ProviderWorkspaceReusePolicy::ZeroBeforeUse
                })
            })
        {
            return Ok(None);
        }
        let Some(recipe) = lane.lookup_segment_binding_recipe(replay.program_id())? else {
            return Ok(None);
        };
        let Some(backend_recipe) = recipe.backend_program.as_ref() else {
            return Ok(None);
        };
        if !recipe
            .nodes
            .iter()
            .map(|node| node.node_index as u32)
            .eq(replay.per_wave_binding_node_indices().iter().copied())
        {
            return Ok(None);
        };
        let Some((entry, epoch)) = lane.try_segment_binding_entry(replay.program_id())? else {
            return Ok(None);
        };
        let memory = resolved.execution_plan().payload().memory();
        let requests = recipe
            .plan_slots
            .iter()
            .map(|&slot_index| {
                let allocation = memory
                    .static_allocations()
                    .get(slot_index)
                    .ok_or_else(|| invalid_operation("resident Plan slot is absent"))?;
                Ok(SegmentPlanResourceRequest {
                    slot_index,
                    allocation,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let plans = wave.prepare_segment_plan_views(&requests)?;
        let mut sequence_windows = Vec::new();
        for (resource_index, resource) in recipe.resources.iter().enumerate() {
            let CompiledSegmentResourceSource::Dynamic { descriptor_index } = resource.source
            else {
                continue;
            };
            let descriptor = memory
                .dynamic_descriptors()
                .get(descriptor_index)
                .ok_or_else(|| invalid_operation("resident descriptor ordinal is absent"))?;
            if descriptor.lifetime() != AllocationLifetime::Sequence {
                continue;
            }
            let mut minimum = resource.required_minimum_bytes;
            let mut maximum = None;
            for region in recipe
                .nodes
                .iter()
                .flat_map(|node| &node.regions)
                .filter(|region| region.resource_index == resource_index)
            {
                if region.request.extent == SegmentBindingRegionExtent::CurrentResource {
                    minimum = minimum.max(
                        region
                            .offset_bytes
                            .checked_add(1)
                            .ok_or_else(|| invalid_operation("resident region offset overflow"))?,
                    );
                    if let Some(bound) = region.component_length {
                        let max = bound
                            .checked_sub(region.request.offset_bytes)
                            .and_then(|n| n.checked_add(region.offset_bytes))
                            .ok_or_else(|| {
                                invalid_operation("resident component bound overflow")
                            })?;
                        maximum = Some(maximum.map_or(max, |old: u64| old.min(max)));
                    }
                }
            }
            sequence_windows.push(SequenceWindowRule {
                ordinal: descriptor_index,
                descriptor: descriptor.clone(),
                minimum,
                maximum,
            });
        }
        let compiled = super::physical_segment_dispatch::with_prepared_physical_wave(
            lane.runtime(),
            resolved,
            wave,
            &recipe,
            &plans,
            &DisabledSubmissionWaveDispatchTimingSink,
            |physical, _scopes, facts| {
                let dependencies = SealedResidentPlanDependencies::seal(
                    physical
                        .dependencies
                        .iter()
                        .flatten()
                        .filter_map(Option::as_ref),
                    replay.retained_plan_dependencies(),
                )?;
                let backend = lane
                    .runtime()
                    .compile_resident_decode_program(backend_recipe.as_ref(), physical)
                    .map_err(|e| {
                        invalid_operation(format!("resident cold compilation failed: {e}"))
                    })?;
                Ok(backend.map(|backend| {
                    let mut fixed_buffer_identities = physical
                        .plan_views
                        .iter()
                        .map(|plan| plan.leased.buffer() as *const R::Buffer as usize)
                        .collect::<Vec<_>>();
                    if let Some(backing) = physical.fresh_backing {
                        fixed_buffer_identities.extend(
                            backing
                                .bindings()
                                .iter()
                                .map(|binding| binding.buffer() as *const R::Buffer as usize),
                        );
                    }
                    fixed_buffer_identities.sort_unstable();
                    fixed_buffer_identities.dedup();
                    (
                        backend,
                        dependencies,
                        physical.sequence_tables().to_vec(),
                        (
                            recipe.nodes.len() as u64,
                            facts.dynamic_resource_requests,
                            facts.unique_physical_buffers,
                        ),
                        fixed_buffer_identities,
                    )
                }))
            },
        )?;
        let Some((backend, dependencies, tables, facts, fixed_buffer_identities)) = compiled else {
            return Ok(None);
        };
        Ok(Some(Arc::new(ResidentDecodeProgram {
            reservation,
            topology: topology.clone(),
            replay: replay.clone(),
            _recipe: recipe,
            version: Mutex::new(Arc::new(ResidentDecodeVersion {
                backend,
                tables,
                facts,
            })),
            fixed_buffer_identities,
            dependencies,
            sequence_windows,
            lane: Arc::clone(lane),
            entry,
            epoch,
        })))
    }

    pub fn bind_resident_decode_identity<
        'a,
        R: DeviceRuntime,
        I: ExactSizeIterator<Item = &'a TrustedActiveSequenceBinding>,
    >(
        program: &ResidentDecodeProgram<R>,
        active: I,
        use_: &ResidentDecodeUse<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<BatchOperationIdentity, VNextError> {
        let topology = &program.topology;
        let evidence = use_.plan_evidence();
        let frames = use_.participant_frames();
        let sessions = use_.participant_sessions().collect::<Vec<_>>();
        if !Arc::ptr_eq(lane, &program.lane)
            || !Arc::ptr_eq(use_.runtime(), lane.runtime_arc())
            || use_.execution_lane_id() != topology.lane_id()
            || evidence.plan_id() != topology.plan_id()
            || evidence.plan_hash() != topology.plan_hash()
            || frames.len() != active.len()
            || frames.len() != sessions.len()
        {
            return Err(invalid_operation(
                "resident identity differs from its current Plan/lane/cohort",
            ));
        }
        let seeds = frames
            .iter()
            .copied()
            .zip(sessions)
            .zip(active)
            .map(|((frame, session), active)| {
                active.ensure_open_for_emission()?;
                if frame.sequence_authority() != active.sequence_authority()
                    || frame.request_authority() != session.request_authority()
                    || !active.matches_sequence_session(session.epoch(), session.fingerprint())
                    || active.sequence_authority() != session.sequence_authority()
                    || active.coordinator_id() != session.resources().coordinator_id()
                    || active.run_id() != session.resources().run_id()
                    || active.request_id() != session.resources().request_id()
                    || active.plan().plan_id() != topology.plan_id()
                    || active.plan().device_id() != topology.device_id()
                    || active.plan().plan_hash() != topology.plan_hash()
                    || active.runtime_implementation_fingerprint()
                        != topology.runtime_implementation_fingerprint()
                {
                    return Err(invalid_operation(
                        "resident identity differs from the live participant session",
                    ));
                }
                let count = u64::try_from(topology.node_count())
                    .map_err(|_| invalid_operation("resident node count overflows"))?;
                let completed = frame.frame_id().get() - 1;
                let last = completed
                    .checked_mul(count)
                    .and_then(|n| n.checked_add(count))
                    .ok_or_else(|| invalid_operation("resident invocation identity exhausted"))?;
                NodeInvocationId::try_from(last)?;
                let per_frame = count
                    .checked_mul(3)
                    .and_then(|n| n.checked_add(2))
                    .ok_or_else(|| invalid_operation("resident event identity exhausted"))?;
                completed
                    .checked_mul(per_frame)
                    .and_then(|n| n.checked_add((count - 1).checked_mul(3)?))
                    .and_then(|n| n.checked_add(5))
                    .ok_or_else(|| invalid_operation("resident event identity exhausted"))?;
                let span_root = format!("vnext/request/{}", active.fingerprint());
                SpanId::new(format!(
                    "{span_root}/frame/{}/node/{last}/operation",
                    frame.frame_id()
                ))?;
                let provisioning = active.static_provisioning_identity();
                Ok(SubmissionWaveParticipantIdentitySeed::new(
                    frame,
                    active.run_id().clone(),
                    active.request_id().clone(),
                    active.static_pool_id(),
                    active
                        .static_pool_identity_fingerprint_ref()
                        .map(str::to_owned),
                    provisioning.map(|p| p.run_id().clone()),
                    provisioning.map(|p| p.request_id().clone()),
                    provisioning.map(|p| p.transaction_id().clone()),
                    active.sequence_authority().sparse_id(),
                    active.sequence_authority().generation(),
                    active.activation_epoch(),
                    active.runtime_implementation_fingerprint().to_owned(),
                    active.fingerprint().to_owned(),
                    span_root,
                ))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        BatchOperationIdentity::from_compiled_wave(
            topology.clone(),
            use_.batch_step_id(),
            use_.batch_invocation_id(),
            use_.fingerprint().to_owned(),
            use_.work_shape().fingerprint().to_owned(),
            seeds,
            ferrum_types::InvocationPreparationStrategy::ResidentDecodeProgram,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_resident_decode_with_inputs<
        R: DeviceRuntime,
        S: SubmissionWaveDispatchTimingSink,
    >(
        program: &Arc<ResidentDecodeProgram<R>>,
        resolved: &dyn ExecutablePlanView,
        identity: &BatchOperationIdentity,
        uploads: &[SubmissionWaveInputUpload],
        mut use_: ResidentDecodeUse<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
        timing_mode: DeviceTimingMode,
        timing: &S,
    ) -> Result<ProfiledSubmissionHandle<R>, ResidentDecodeDispatchError<R>> {
        let contract = SubmissionWaveDispatchStageTimer::start(
            timing,
            SubmissionWaveDispatchStage::ContractValidateAndReserve,
        );
        let invalid = ResidentDecodeDispatchError::Contract;
        if !use_.belongs_to_reservation(&program.reservation)
            || !Arc::ptr_eq(lane, &program.lane)
            || resolved.execution_plan().plan_hash() != program.topology.plan_hash()
            || identity.batch_step_id() != use_.batch_step_id()
            || identity.batch_invocation_id() != use_.batch_invocation_id()
            || identity.claimed_backing_fingerprint() != use_.fingerprint()
            || identity.work_shape_fingerprint_at(0) != Some(use_.work_shape().fingerprint())
            || identity.lane_id() != lane.id()
            || identity.plan_hash() != program.topology.plan_hash()
            || identity.preparation_strategy()
                != ferrum_types::InvocationPreparationStrategy::ResidentDecodeProgram
        {
            return Err(invalid(invalid_operation(
                "resident submission identity or reservation differs from its fresh use",
            )));
        }
        let Some(version) = program.prepare_current_version(&use_).map_err(invalid)? else {
            return Err(ResidentDecodeDispatchError::Unavailable { use_ });
        };
        use_.retain_resident_program(DeviceBufferRetention::pair(
            Arc::clone(program),
            Arc::clone(&version),
        ))
        .map_err(invalid)?;
        let fresh_retention = use_.retention();
        let dependencies = program.dependencies.issue(fresh_retention.clone());
        let scope = Arc::clone(dependencies.scope());
        let mut request =
            DeviceResidentDecodeRequest::new(use_.work_shape(), fresh_retention, dependencies);
        drop(contract);
        let mut provider_stage = ResidentProviderTimer::start(timing);
        let encoded = lane
            .runtime()
            .encode_resident_decode_program(version.backend.as_ref(), &mut request)
            .map_err(|e| {
                invalid(invalid_operation(format!(
                    "resident binding encode failed: {e}"
                )))
            })?;
        let Some(encoded) = encoded else {
            if request.dependencies().is_none() {
                return Err(invalid(invalid_operation(
                    "resident backend declined after consuming fresh dependency authority",
                )));
            }
            drop(request);
            return Err(ResidentDecodeDispatchError::Unavailable { use_ });
        };
        if request.dependencies().is_some() {
            return Err(invalid(invalid_operation(
                "resident backend omitted dependency consumption",
            )));
        }
        drop(request);
        let (dependencies, bindings) = encoded.into_parts();
        let dependency = dependencies
            .into_command(&scope, &program.dependencies)
            .map_err(invalid)?;
        let participant_count = u32::try_from(use_.work_shape().participants().len())
            .map_err(|_| invalid(invalid_operation("resident participant count exceeds u32")))?;
        let immediate_tokens = use_.work_shape().immediate_tokens();
        let runtime = lane.runtime();
        provider_stage.pause();
        let backing_stage = SubmissionWaveDispatchStageTimer::start(
            timing,
            SubmissionWaveDispatchStage::BackingAndInputEncode,
        );
        use_.begin_dispatch().map_err(invalid)?;
        let mut completion =
            CompletionReaper::reserve_resident(reaper, use_, Arc::clone(lane), identity.clone())
                .map_err(invalid)?;
        let mut commands = DeviceCommandBatch::with_capacity_timing_and_compute_path(
            uploads.len() + program.replay.segments().len() + 2,
            timing_mode,
            DeviceComputePathRequirement::ReplayedOnly,
        );
        completion
            .encode_backing_initializations(runtime, &mut commands)
            .map_err(|error| match error {
                BackingInitializationEncodeError::Contract(e) => invalid(e),
                BackingInitializationEncodeError::Runtime { error, .. } => invalid(
                    invalid_operation(format!("resident initialization encode failed: {error}")),
                ),
            })?;
        encode_submission_wave_inputs(
            runtime,
            resolved,
            identity,
            &completion,
            uploads,
            &mut commands,
        )
        .map_err(|error| {
            invalid(invalid_operation(format!(
                "resident input upload failed: {error}"
            )))
        })?;
        drop(backing_stage);
        provider_stage.resume();
        if let Some(command) = dependency {
            commands.push_dynamic_binding(command)
        }
        commands.push_dynamic_binding(bindings);
        for segment in program.replay.segments() {
            let invocation = DeviceReusableExecutionInvocation::new(
                program.replay.program_id().clone(),
                segment.clone(),
                participant_count,
                immediate_tokens,
            )
            .map_err(invalid)?;
            let command = runtime
                .encode_reusable_execution(invocation)
                .map_err(|e| {
                    invalid(invalid_operation(format!(
                        "resident replay encode failed: {e}"
                    )))
                })?
                .ok_or_else(|| {
                    invalid(invalid_operation(
                        "resident runtime declined its sealed replay program",
                    ))
                })?;
            commands.push_operation_parts(
                segment.start_node_index(),
                Vec::new(),
                command,
                Vec::new(),
            );
        }
        identity.record_segment_preparation(
            true,
            version.facts.0,
            version.facts.1,
            version.facts.2,
        );
        drop(provider_stage);
        let lane_stage = SubmissionWaveDispatchStageTimer::start(
            timing,
            SubmissionWaveDispatchStage::LaneReserveSubmitAndArm,
        );
        let mut enqueue = lane.reserve_enqueue().map_err(invalid)?;
        if !enqueue
            .validate_resident_program_entry(
                program.replay.program_id(),
                &program.entry,
                program.epoch,
            )
            .map_err(invalid)?
        {
            drop(enqueue);
            drop(commands);
            let use_ = completion
                .definitely_not_submitted_resident()
                .map_err(invalid)?;
            return Err(ResidentDecodeDispatchError::Unavailable { use_ });
        }
        match completion.prepare_terminal_readbacks(&mut enqueue, &mut commands) {
            Ok(()) => {}
            Err(TerminalReadbackPrepareError::Contract(e)) => return Err(invalid(e)),
            Err(TerminalReadbackPrepareError::Runtime(error)) => {
                drop(enqueue);
                drop(commands);
                let use_ = completion
                    .definitely_not_submitted_resident()
                    .map_err(invalid)?;
                let first = identity.operation_identity_at(0, 0).map_err(invalid)?;
                let failure = classify_device_error(runtime, first, &error).map_err(invalid)?;
                return Err(ResidentDecodeDispatchError::DefinitelyNotSubmitted {
                    failures: vec![failure],
                    use_,
                });
            }
        }
        completion.mark_submission_started();
        let submit = enqueue.submit_with_timing(commands, timing);
        drop(enqueue);
        let outcome = match submit {
            LaneSubmitOutcome::DefinitelyNotSubmitted(error) => {
                let use_ = completion
                    .definitely_not_submitted_resident()
                    .map_err(invalid)?;
                let first = identity.operation_identity_at(0, 0).map_err(invalid)?;
                let failure = classify_device_error(runtime, first, &error).map_err(invalid)?;
                Err(ResidentDecodeDispatchError::DefinitelyNotSubmitted {
                    failures: vec![failure],
                    use_,
                })
            }
            LaneSubmitOutcome::PossiblySubmittedPanic => {
                Err(ResidentDecodeDispatchError::SubmissionIndeterminate {
                    recovery: completion.submission_indeterminate(),
                })
            }
            LaneSubmitOutcome::Submitted(fence) => {
                // Catch attribution panic, then arm even on failure. No
                // unowned in-flight fence or reservation may unwind here.
                let device = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    runtime.submission_attribution(&fence)
                }));
                let handle =
                    completion
                        .arm(fence, timing_mode)
                        .map_err(|(error, completion)| {
                            ResidentDecodeDispatchError::PostSubmitContract { error, completion }
                        })?;
                let device = match device {
                    Ok(device) => device,
                    Err(_) => {
                        lane.fail_closed();
                        return Err(ResidentDecodeDispatchError::PostSubmitContract {
                            error: invalid_operation(
                                "resident attribution panicked after submission",
                            ),
                            completion: handle,
                        });
                    }
                };
                if !lane.current_descriptor_matches_snapshot() {
                    lane.fail_closed();
                    return Err(ResidentDecodeDispatchError::PostSubmitContract {
                        error: invalid_operation("resident runtime drifted after submission"),
                        completion: handle,
                    });
                }
                let attribution = match device
                    .map(|device| {
                        BoundDeviceSubmissionAttribution::new(
                            identity.clone(),
                            handle.receipt().fingerprint().to_owned(),
                            device,
                        )
                    })
                    .transpose()
                {
                    Ok(attribution) => attribution,
                    Err(error) => {
                        return Err(ResidentDecodeDispatchError::PostSubmitContract {
                            error,
                            completion: handle,
                        })
                    }
                };
                Ok(ProfiledSubmissionHandle::new(handle, attribution))
            }
        };
        drop(lane_stage);
        outcome
    }
}
