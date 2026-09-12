use super::*;
use std::ops::Range;

type Runtime = MetalDeviceRuntime;

pub struct Fixture {
    _composition: CompositionParts,
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<Runtime>,
    resources: Arc<PlanRuntimeResources<Runtime>>,
    lane: Arc<ExecutionLane<Runtime>>,
    reaper: Arc<CompletionReaper<Runtime>>,
    states: Vec<StateSpec>,
    checkpoint_timing_mode: DeviceTimingMode,
}

impl Fixture {
    pub fn new(kind: AttentionKind) -> Self {
        let definition = Family::new(kind);
        let states = definition.states();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(
                &serde_json::to_value(kind).unwrap(),
                &id("fixture.attention.f32-master"),
            )
            .unwrap();
        let composition =
            MetalVNextComposition::create(id(format!("device.metal.checkpoint.{kind:?}"))).unwrap();
        let (runtime, registry, materializers, materializer_id, catalog) = composition.into_parts();
        let policy = ResolvedRuntimePolicy::new(
            "runtime-policy.metal.checkpoint-continuation",
            ContractVersion::new(1, 0),
            SchedulingDiscipline::FirstReady,
            RuntimeMemoryPolicy {
                checkpoint_capacity: Some(CheckpointCapacityPolicy::new(8 << 20).unwrap()),
                capacity_bytes: 128 << 20,
                reserve_bytes: 1 << 20,
                maximum_active_sequences: 3,
                dynamic_storage_profile_order: runtime
                    .descriptor()
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            },
            AdmissionPolicy {
                maximum_queue_depth: 3,
                maximum_scheduled_tokens: MAX_TOKENS,
                sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
                allow_defer: true,
                cancellation_check_interval_steps: 1,
            },
            ferrum_types::AttentionExecutionPolicy::NativeAdaptive,
            ExecutionDeterminismRequirement::BitwiseSameRuntime,
            None,
        )
        .unwrap();
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id("value.tokens"),
            ProgramTensorSpec {
                dimensions: vec![MAX_TOKENS],
                element_type: ElementType::U32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        options.require_weight_materializer(materializer_id);
        options.retain_completion_value(id("value.output"));
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            &catalog,
            &policy,
            &registry.planning(),
            &materializers,
            &options,
        )
        .unwrap();
        let executable = compilation.executable();
        let plan = executable.execution_plan();
        assert!(
            matches!(
                plan.sequence_checkpoint_capability(),
                SequenceCheckpointCapability::Enabled(_)
            ),
            "actual selected provider declarations must close the checkpoint contract: {:?}",
            plan.sequence_checkpoint_capability()
        );
        let providers = registry.bind_plan(executable).unwrap();
        let provisioned = plan
            .provision_static(
                Arc::clone(&runtime),
                id("request.metal.checkpoint.provision"),
            )
            .unwrap();
        let permit = match provisioned.into_provisioning() {
            StaticProvisioning::Required(permit) => permit,
            _ => panic!("real provider fixture has static weights"),
        };
        let identity = ResourceTransactionIdentity::for_admission(
            permit.binding(),
            id("run.metal.checkpoint.provision"),
            id("transaction.metal.checkpoint.provision"),
        );
        let driver = RuntimeResourceDriver::new(Arc::clone(&runtime)).unwrap();
        let reserved = ResourceTransaction::begin(driver, identity, permit)
            .unwrap()
            .reserve()
            .unwrap();
        let committed = match reserved.commit() {
            Ok(value) => value,
            Err(ResourceCommitTransitionError::Recoverable(error)) => {
                panic!("static commit failed: {:?}", error.failure())
            }
            Err(ResourceCommitTransitionError::Poisoned(error)) => {
                panic!("static commit indeterminate: {:?}", error.failure())
            }
        };
        for pool in committed
            .maintenance_controller()
            .pool_ids()
            .cloned()
            .collect::<Vec<_>>()
        {
            committed
                .maintenance_controller()
                .initialize_pool(&pool)
                .unwrap();
        }
        // Ordinary pool initialization only. Foreground admission and optional
        // checkpoint retention must request their own subsequent maintenance.
        let source = family::Weights::new(family.weight_schema());
        let initialized = committed
            .initialize_static(
                &family,
                plan,
                &source,
                StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
            )
            .unwrap();
        let resources = match initialized.into_plan_runtime() {
            Ok(resources) => resources,
            Err(error) => panic!("plan runtime handoff failed: {}", error.error()),
        };
        let lane = resources.create_execution_lane().unwrap();
        // The registry, runtime, materializers and catalog retain their real
        // composition owners; no provider descriptor is substituted in tests.
        let composition = CompositionParts {
            _runtime: runtime,
            _registry: registry,
            _materializers: materializers,
            _catalog: catalog,
        };
        Self {
            _composition: composition,
            compilation,
            providers,
            resources,
            lane,
            reaper: CompletionReaper::new(),
            states,
            checkpoint_timing_mode: DeviceTimingMode::Off,
        }
    }

    pub fn with_checkpoint_timing(mut self, mode: DeviceTimingMode) -> Self {
        self.checkpoint_timing_mode = mode;
        self
    }

    fn assert_checkpoint_timing(&self, operation: CheckpointOperationTimings, bytes: u64) {
        assert_eq!(operation.submitted_copies.samples, 1);
        assert_eq!(operation.submitted_copies.total_bytes, bytes);
        assert!(operation.submitted_copies.total_commands > 0);
        let device = operation.device_execution;
        assert_eq!(device.failed_or_unproven, 0);
        if self.checkpoint_timing_mode == DeviceTimingMode::Off {
            assert_eq!(device.not_requested, 1);
            assert_eq!(device.measured.samples, 0);
            assert_eq!(device.unavailable, 0);
        } else {
            assert_eq!(device.not_requested, 0);
            if device.unavailable == 0 {
                assert_eq!(device.measured.samples, 1);
            } else {
                assert_eq!(device.measured.samples, 0);
                assert_eq!(device.unavailable, 1);
                assert_eq!(
                    device.last_unavailable_reason,
                    Some(DeviceTimingUnavailableReason::BackendUnsupported)
                );
                eprintln!("checkpoint device timestamp unavailable: backend unsupported");
            }
            eprintln!("checkpoint terminal timing and copy ranges: {operation:?}");
        }
    }

    pub fn admit(&self, name: &str, tokens: Arc<[u32]>) -> Arc<SequenceSession<Runtime>> {
        let ceiling = tokens.len();
        self.admit_with_ceiling(name, tokens, ceiling)
    }

    pub fn admit_with_ceiling(
        &self,
        name: &str,
        tokens: Arc<[u32]>,
        ceiling: usize,
    ) -> Arc<SequenceSession<Runtime>> {
        let span = TokenSpanWork::from_token_ids_with_fit(&tokens, 0..tokens.len(), ceiling)
            .unwrap()
            .with_checkpoint_tokens(tokens)
            .unwrap();
        let work = ResourceWorkShape::single(span).unwrap();
        let request = RequestResourceAdmissionRequest::new(
            work.clone(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let binding = self.resources.trusted_runtime_binding().unwrap();
        let request_resources = loop {
            match binding
                .try_admit_request(
                    request.clone(),
                    id(format!("run.metal.checkpoint.{name}")),
                    id(format!("request.metal.checkpoint.{name}")),
                )
                .unwrap()
            {
                RequestResourceAdmissionDecision::Admitted(resources) => break resources,
                RequestResourceAdmissionDecision::BackingDeferred(deferred) => {
                    require_progress(deferred.maintain().unwrap());
                }
                RequestResourceAdmissionDecision::Deferred(reason) => {
                    require_progress(
                        self.resources
                            .maintain_for_admission_deferred(&reason)
                            .unwrap(),
                    );
                }
                RequestResourceAdmissionDecision::PermanentRejected(reason) => {
                    panic!("fixture request rejected: {reason:?}")
                }
            }
        };
        let request = SequenceResourceAdmissionRequest::new(
            work,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let sequence = loop {
            match request_resources
                .try_admit_sequence(request.clone())
                .unwrap()
            {
                SequenceResourceAdmissionDecision::Admitted(resources) => break resources,
                SequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                    require_progress(deferred.maintain().unwrap());
                }
                SequenceResourceAdmissionDecision::Deferred(reason) => {
                    require_progress(
                        self.resources
                            .maintain_for_admission_deferred(&reason)
                            .unwrap(),
                    );
                }
                SequenceResourceAdmissionDecision::PermanentRejected(reason) => {
                    panic!("fixture sequence rejected: {reason:?}")
                }
            }
        };
        sequence.open_session().unwrap()
    }

    pub fn extend(&self, session: &Arc<SequenceSession<Runtime>>, tokens: Arc<[u32]>) {
        let work =
            ResourceWorkShape::single(token_span(Arc::clone(&tokens), 0..tokens.len())).unwrap();
        let request =
            SequenceResourceExtensionRequest::new(work, AdmissionPressureAction::WaitForRelease)
                .unwrap();
        loop {
            match session.try_ensure_backing_covers(request.clone()).unwrap() {
                SequenceResourceExtensionDecision::Current(_)
                | SequenceResourceExtensionDecision::Extended(_) => break,
                SequenceResourceExtensionDecision::BackingDeferred(deferred) => {
                    require_progress(deferred.maintain().unwrap())
                }
                SequenceResourceExtensionDecision::Deferred(reason) => {
                    // A retained checkpoint can fill an otherwise growable
                    // State pool. Use the authentic foreground pressure path.
                    if self
                        .resources
                        .try_maintain_for_capacity_pressure(&reason)
                        .unwrap()
                        .is_none()
                    {
                        require_progress(
                            self.resources
                                .maintain_for_admission_deferred(&reason)
                                .unwrap(),
                        );
                    }
                }
                SequenceResourceExtensionDecision::RetryRequired(_) => {
                    panic!("fixture has no in-flight frame during extension")
                }
                SequenceResourceExtensionDecision::PermanentRejected(reason) => {
                    panic!("fixture extension rejected: {reason:?}")
                }
            }
        }
    }

    pub fn execute(
        &self,
        session: &Arc<SequenceSession<Runtime>>,
        tokens: Arc<[u32]>,
        range: Range<usize>,
    ) -> Observation {
        let span = token_span(Arc::clone(&tokens), range.clone());
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(session)]).unwrap();
        let request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(vec![span]).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let step = loop {
            match batch.try_begin_step(request.clone(), &self.lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => break step,
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    require_progress(deferred.maintain().unwrap());
                }
                StepResourceAdmissionDecision::Deferred(reason) => {
                    require_progress(
                        self.resources
                            .maintain_for_admission_deferred(&reason)
                            .unwrap(),
                    );
                }
                StepResourceAdmissionDecision::PermanentRejected(reason) => {
                    panic!("fixture step rejected: {reason:?}")
                }
            }
        };
        let wave = loop {
            match step
                .try_prepare_full_plan_submission_wave(
                    Arc::new(step.work_shape().clone()),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap()
            {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => break wave,
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    require_progress(deferred.maintain().unwrap());
                }
                StepSubmissionWaveAdmissionDecision::Deferred(reason) => {
                    require_progress(
                        self.resources
                            .maintain_for_admission_deferred(&reason)
                            .unwrap(),
                    );
                }
                StepSubmissionWaveAdmissionDecision::PermanentRejected(reason) => {
                    panic!("fixture full-plan wave rejected: {reason:?}")
                }
                other => panic!(
                    "fixture full-plan wave has a request-state hazard: {:?}",
                    std::mem::discriminant(&other)
                ),
            }
        };
        let active = TrustedActiveSequenceBinding::from_session(session).unwrap();
        let executable = self.compilation.executable();
        let plan = executable.execution_plan();
        let identity = OperationDispatch::bind_submission_wave_identity(
            executable,
            std::iter::once(&active),
            &wave,
            &self.lane,
        )
        .unwrap();
        let input = SubmissionWaveInputUpload::new(
            id("node.embedding"),
            0,
            0,
            range.start as u64 * ElementType::U32.size_bytes(),
            HostTransferLayout::new(ElementType::U32, range.len() as u64).unwrap(),
            tokens[range.clone()]
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect(),
        )
        .unwrap();
        let attention = plan
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id().as_str() == "node.attention")
            .unwrap();
        let output = attention
            .values()
            .iter()
            .find(|value| value.role() == ResolvedValueRole::Output && value.ordinal() == 0)
            .unwrap();
        let output_component = &output.storage().components()[0];
        let output_descriptor = plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .iter()
            .find(|descriptor| descriptor.base_resource_id() == output_component.resource_id())
            .unwrap();
        assert_eq!(output_descriptor.lifetime(), AllocationLifetime::Step);
        assert_eq!(output_descriptor.kind(), &AllocationKind::Value);
        assert!(matches!(
            output_descriptor.demand(),
            DynamicResourceDemand::Tokens { bytes_per_token, .. }
                if *bytes_per_token == HIDDEN * ElementType::F32.size_bytes()
        ));
        // Step-token readbacks are participant-local, unlike uploads' source
        // coordinates. Completion translates this span-local range into the
        // same packed backing that the provider's output_start selects.
        let mut requests = vec![CompletionReadbackRequest::new(
            attention.id().clone(),
            0,
            output_component.resource_id().clone(),
            output_component.offset_bytes(),
            HostTransferLayout::new(ElementType::F32, range.len() as u64 * HIDDEN).unwrap(),
        )
        .unwrap()];
        let mut names =
            BTreeMap::from([(output_component.resource_id().clone(), "output".to_owned())]);
        for state in &self.states {
            let value = attention
                .values()
                .iter()
                .find(|value| value.value_id() == &state.value_id)
                .unwrap();
            let component = &value.storage().components()[0];
            let bytes = match state.capacity_demand {
                StateCapacityDemand::FixedPerScope => state.tensor.byte_len().unwrap(),
                StateCapacityDemand::TokenScaled {
                    bytes_per_token, ..
                } => bytes_per_token * range.end as u64,
            };
            requests.push(
                CompletionReadbackRequest::new_typed(
                    attention.id().clone(),
                    0,
                    component.resource_id().clone(),
                    BufferUsage::State,
                    component.offset_bytes(),
                    HostTransferLayout::new(
                        state.tensor.element_type,
                        bytes / state.tensor.element_type.size_bytes(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
            assert!(names
                .insert(component.resource_id().clone(), state.id.to_string())
                .is_none());
        }
        let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
            self.providers.providers(),
            executable,
            &identity,
            std::iter::once(&active),
            DeviceTimingMode::Off,
            &[input],
            wave,
            &self.lane,
            &self.reaper,
        )
        .unwrap();
        let readbacks = CompletionReadbackCollectionRequest::new(
            requests
                .into_iter()
                .map(|request| CompletionReadbackBatchRequest::new(vec![request]).unwrap())
                .collect(),
        )
        .unwrap();
        let receipt = match handle.wait_with_readback_collection(readbacks).unwrap() {
            CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
            other => panic!("provider readbacks did not terminate: {other:?}"),
        };
        let values = receipt
            .dispositions()
            .iter()
            .map(|result| match result {
                CompletionReadbackDisposition::Succeeded(output) => (
                    names.remove(output.request().resource_id()).unwrap(),
                    output.bytes().to_vec(),
                ),
                other => panic!("provider readback failed: {other:?}"),
            })
            .collect();
        assert!(names.is_empty());
        drop(receipt);
        drop(handle);
        drop(identity);
        drop(active);
        step.try_retire_normal().unwrap();
        Observation { values }
    }

    pub fn capture(&self, source: &Arc<SequenceSession<Runtime>>) -> SequenceCheckpoint<Runtime> {
        let binding = self.resources.trusted_runtime_binding().unwrap();
        let before = CapacitySnapshot::observe(&self.resources);
        assert_eq!(
            before.checkpoint_claims, 0,
            "first capture must have no retained copy"
        );
        assert_eq!(before.checkpoint_bytes, 0);
        assert_eq!(before.pending_growth_bytes, 0);
        let start = self
            .reaper
            .try_capture_sequence_checkpoint_with_timing(
                self.compilation.executable().execution_plan(),
                &binding,
                Arc::clone(source),
                Arc::clone(&self.lane),
                self.checkpoint_timing_mode,
            )
            .unwrap();
        let NativeCheckpointStart::CapacityMaintenance {
            reason,
            maintenance,
        } = start
        else {
            panic!("foreground-only initialized pools must expose first-capture maintenance");
        };
        let receipt = match maintenance.try_maintain().unwrap() {
            CheckpointCapacityMaintenanceOutcome::Ready(receipt) => receipt,
            other => panic!("checkpoint maintenance could not use available capacity: {other:?}"),
        };
        let maintained = CapacitySnapshot::observe(&self.resources);
        let growth_bytes = receipt
            .growths()
            .iter()
            .map(|growth| growth.chunk_bytes())
            .sum::<u64>();
        assert!(
            growth_bytes > 0,
            "first capture must exercise real backing growth"
        );
        assert_eq!(
            maintained.resident_bytes - before.resident_bytes,
            growth_bytes
        );
        assert_eq!(maintained.free_bytes - before.free_bytes, growth_bytes);
        assert_eq!(
            maintained.budget_claimed_bytes - before.budget_claimed_bytes,
            growth_bytes
        );
        assert_eq!(maintained.pending_growth_bytes, 0);
        assert_eq!(
            maintained.checkpoint_claims, 0,
            "maintenance cannot reserve a copy"
        );
        assert_eq!(maintained.checkpoint_bytes, 0);
        assert_eq!(maintained.non_checkpoint_bytes, before.non_checkpoint_bytes);
        assert_eq!(
            maintained.non_checkpoint_claims,
            before.non_checkpoint_claims
        );
        eprintln!("first checkpoint maintenance: {reason:?}; before={before:?}; maintained={maintained:?}; growth_bytes={growth_bytes}");
        // The maintenance receipt grants no copy permission and holds no
        // source reservation. Re-enter the complete authenticated capture path.
        let start = self
            .reaper
            .try_capture_sequence_checkpoint_with_timing(
                self.compilation.executable().execution_plan(),
                &binding,
                Arc::clone(source),
                Arc::clone(&self.lane),
                self.checkpoint_timing_mode,
            )
            .unwrap();
        match finish(start) {
            NativeCheckpointResult::Captured(value) => {
                self.assert_checkpoint_timing(
                    self.reaper.checkpoint_timing_snapshot().capture,
                    value.logical_bytes(),
                );
                let captured = CapacitySnapshot::observe(&self.resources);
                assert!(captured.checkpoint_claims > 0);
                assert!(captured.checkpoint_bytes > 0);
                assert_eq!(captured.non_checkpoint_bytes, before.non_checkpoint_bytes);
                assert_eq!(captured.non_checkpoint_claims, before.non_checkpoint_claims);
                eprintln!("first checkpoint captured: {captured:?}");
                value
            }
            _ => panic!("capture did not publish an immutable checkpoint"),
        }
    }

    pub fn restore(
        &self,
        target: &Arc<SequenceSession<Runtime>>,
        checkpoint: &SequenceCheckpoint<Runtime>,
        tokens: Arc<[u32]>,
    ) {
        let start = self
            .reaper
            .try_restore_sequence_checkpoint_with_timing(
                self.compilation.executable().execution_plan(),
                Arc::clone(target),
                checkpoint,
                tokens,
                Arc::clone(&self.lane),
                self.checkpoint_timing_mode,
            )
            .unwrap();
        match finish(start) {
            NativeCheckpointResult::Restored(publication) => {
                assert!(publication.matches_target(target));
                assert_eq!(
                    publication.completed_tokens(),
                    checkpoint.completed_tokens()
                );
                publication.acknowledge().unwrap();
                self.assert_checkpoint_timing(
                    self.reaper.checkpoint_timing_snapshot().restore,
                    checkpoint.logical_bytes(),
                );
            }
            _ => panic!("restore did not publish its exact target frontier"),
        }
    }

    pub fn capture_completed_input(
        &self,
        source: &Arc<SequenceSession<Runtime>>,
    ) -> SequenceCheckpoint<Runtime> {
        let binding = self.resources.trusted_runtime_binding().unwrap();
        let plan = self.compilation.executable().execution_plan();
        let start = || {
            self.reaper
                .try_capture_sequence_checkpoint(
                    plan,
                    &binding,
                    Arc::clone(source),
                    Arc::clone(&self.lane),
                )
                .unwrap()
        };
        let initial = start();
        let submitted = match initial {
            NativeCheckpointStart::CapacityMaintenance { maintenance, .. } => {
                assert!(matches!(
                    maintenance.try_maintain().unwrap(),
                    CheckpointCapacityMaintenanceOutcome::Ready(_)
                ));
                start()
            }
            other => other,
        };
        let NativeCheckpointResult::Captured(checkpoint) = finish(submitted) else {
            panic!("completed-input capture did not publish a checkpoint");
        };
        assert_eq!(checkpoint.completed_tokens(), checkpoint.full_input().len());
        assert_eq!(checkpoint.token_prefix(), checkpoint.full_input());
        checkpoint
    }

    pub fn assert_restore_rejected(
        &self,
        target: &Arc<SequenceSession<Runtime>>,
        checkpoint: &SequenceCheckpoint<Runtime>,
        tokens: Arc<[u32]>,
    ) {
        assert!(self
            .reaper
            .try_restore_sequence_checkpoint(
                self.compilation.executable().execution_plan(),
                Arc::clone(target),
                checkpoint,
                tokens,
                Arc::clone(&self.lane),
            )
            .is_err());
    }
}

struct CompositionParts {
    // Ownership-only fields keep the complete provider composition alive for
    // this fixture; no field is dropped early after planning or provisioning.
    _runtime: Arc<Runtime>,
    _registry: OperationRuntimeRegistry<Runtime>,
    _materializers: WeightMaterializerRegistry,
    _catalog: CapabilityCatalog,
}

#[derive(Debug, Default)]
struct CapacitySnapshot {
    resident_bytes: u64,
    free_bytes: u64,
    pending_growth_bytes: u64,
    budget_claimed_bytes: u64,
    checkpoint_claims: u64,
    checkpoint_bytes: u64,
    non_checkpoint_claims: u64,
    non_checkpoint_bytes: u64,
}

impl CapacitySnapshot {
    fn observe(resources: &PlanRuntimeResources<Runtime>) -> Self {
        let status = resources.dynamic_pool_status().unwrap();
        let mut snapshot = Self {
            budget_claimed_bytes: status.budget_claimed_bytes(),
            ..Self::default()
        };
        for pool in status.pools() {
            snapshot.resident_bytes += pool.resident_bytes();
            snapshot.free_bytes += pool.free_bytes();
            snapshot.pending_growth_bytes += pool.pending_growth_bytes();
            let occupancy = pool.live_occupancy();
            for residency in [occupancy.transient(), occupancy.lane_stable()] {
                let checkpoint = residency.checkpoint();
                snapshot.checkpoint_claims += checkpoint.claim_count();
                snapshot.checkpoint_bytes += checkpoint.physical_bytes();
                snapshot.non_checkpoint_claims +=
                    residency.total().claim_count() - checkpoint.claim_count();
                snapshot.non_checkpoint_bytes +=
                    residency.total().physical_bytes() - checkpoint.physical_bytes();
            }
        }
        snapshot
    }
}

fn token_span(tokens: Arc<[u32]>, range: Range<usize>) -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&tokens, range)
        .unwrap()
        .with_checkpoint_tokens(tokens)
        .unwrap()
}

fn require_progress(outcome: DynamicDeferredMaintenanceOutcome) {
    assert!(
        !matches!(
            outcome,
            DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
        ),
        "fixture admission has no owner capable of releasing more capacity: {outcome:?}"
    );
}

fn finish(start: NativeCheckpointStart<Runtime>) -> NativeCheckpointResult<Runtime> {
    let mut transfer = match start {
        NativeCheckpointStart::Submitted(transfer) => transfer,
        NativeCheckpointStart::Skipped(reason) => panic!("checkpoint skipped: {reason:?}"),
        NativeCheckpointStart::NotSubmitted(error) => panic!("checkpoint not submitted: {error}"),
        _ => panic!("checkpoint submission did not have a definite outcome"),
    };
    assert_eq!(
        transfer.wait_for_recovery().unwrap(),
        NativeCheckpointObservation::Ready
    );
    transfer
        .take_result()
        .unwrap()
        .expect("completed checkpoint has a typed result")
}

pub struct Observation {
    values: BTreeMap<String, Vec<u8>>,
}
impl Observation {
    pub fn assert_state_nonzero(&self) {
        for (name, bytes) in &self.values {
            if name == "output" {
                continue;
            }
            let values = if name.ends_with("delta") {
                bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>()
            } else {
                bytes
                    .chunks_exact(2)
                    .map(|bytes| {
                        f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
                    })
                    .collect::<Vec<_>>()
            };
            assert!(
                !values.is_empty() && values.iter().all(|value| value.is_finite()),
                "{name}: state must be finite and nonempty"
            );
            assert!(
                values.iter().any(|value| value.abs() > 1.0e-6),
                "{name}: prefix did not produce nonzero state"
            );
        }
    }
    pub fn assert_state_changed(&self, previous: &Self, label: &str) {
        for (name, bytes) in &self.values {
            if name != "output" {
                assert!(
                    bytes != &previous.values[name],
                    "{label}: {name} did not change state"
                );
            }
        }
    }
    pub fn assert_same(&self, other: &Self, label: &str) {
        assert_eq!(
            self.values.keys().collect::<Vec<_>>(),
            other.values.keys().collect::<Vec<_>>(),
            "{label}: incomplete observation bundle"
        );
        for (name, expected) in &self.values {
            let actual = &other.values[name];
            assert_eq!(actual.len(), expected.len(), "{label} {name}: byte length");
            if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
                panic!(
                    "{label} {name}: first byte mismatch {index}: {} != {}",
                    actual[index], expected[index]
                );
            }
        }
        let output = self.values["output"]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(
            !output.is_empty() && output.iter().all(|value| value.is_finite()),
            "{label}: output must be finite and nonempty"
        );
        assert!(
            output.iter().any(|value| value.abs() > 1.0e-5),
            "{label}: degenerate output"
        );
    }
}
