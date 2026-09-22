use super::*;
use std::ops::Range;

pub type Composition = (
    Arc<Runtime>,
    OperationRuntimeRegistry<Runtime>,
    WeightMaterializerRegistry,
    WeightMaterializerSelection,
    CapabilityCatalog,
);

pub struct Fixture {
    _composition: CompositionParts,
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<Runtime>,
    resources: Arc<PlanRuntimeResources<Runtime>>,
    lane: Arc<ExecutionLane<Runtime>>,
    reaper: Arc<CompletionReaper<Runtime>>,
    states: Vec<StateSpec>,
    checkpoint_timing_mode: DeviceTimingMode,
    reusable_bucket: Option<ReusableExecutionBucketId>,
    output_type: ElementType,
}

impl Fixture {
    pub fn new(kind: AttentionKind) -> Self {
        Self::with_execution_options(kind, false, None)
    }

    pub fn with_execution_options(
        kind: AttentionKind,
        reusable: bool,
        nonfinite_token: Option<u32>,
    ) -> Self {
        Self::with_execution_capacity(kind, reusable, nonfinite_token, 1)
    }

    #[cfg(feature = "cuda")]
    pub fn with_replay_token_capacity(kind: AttentionKind, maximum_tokens: u64) -> Self {
        Self::with_execution_capacity(kind, true, None, maximum_tokens)
    }

    fn with_execution_capacity(
        kind: AttentionKind,
        reusable: bool,
        nonfinite_token: Option<u32>,
        maximum_tokens: u64,
    ) -> Self {
        let definition = Family::new(kind);
        let states = definition.states();
        let profile_id = definition.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile_id))
            .unwrap();
        let (runtime, registry, materializers, materializer, catalog) = composition(kind, &family);
        let bucket = reusable.then(|| {
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("fixture.checkpoint.decode").unwrap(),
                ReusableExecutionCapacity::new(1, maximum_tokens, 16).unwrap(),
            )
            .unwrap()
        });
        let reusable_bucket = bucket.as_ref().map(|bucket| bucket.bucket_id().clone());
        let policy = ResolvedRuntimePolicy::new(
            "runtime-policy.fixture.checkpoint-continuation",
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
            runtime.attention_execution_policy(),
            if reusable {
                ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay
            } else {
                ExecutionDeterminismRequirement::BitwiseSameRuntime
            },
            bucket.map(|bucket| ReusableExecutionPolicy::new(1, vec![bucket]).unwrap()),
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
        options.require_weight_materializer_selection(materializer);
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
                id("request.fixture.checkpoint.provision"),
            )
            .unwrap();
        let permit = match provisioned.into_provisioning() {
            StaticProvisioning::Required(permit) => permit,
            _ => panic!("real provider fixture has static weights"),
        };
        let identity = ResourceTransactionIdentity::for_admission(
            permit.binding(),
            id("run.fixture.checkpoint.provision"),
            id("transaction.fixture.checkpoint.provision"),
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
        let mut source = family::Weights::new(family.weight_schema());
        if let Some(token) = nonfinite_token {
            source.set_nonfinite_embedding(token);
        }
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
        if reusable {
            lane.configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(8).unwrap())
                .unwrap();
        }
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
            reusable_bucket,
            output_type: kind.activation_type(),
        }
    }

    pub fn with_checkpoint_timing(mut self, mode: DeviceTimingMode) -> Self {
        self.checkpoint_timing_mode = mode;
        self
    }

    #[cfg(feature = "cuda")]
    pub fn assert_q8_projection_workspace(&self, strict: &Self) {
        fn attention(fixture: &Fixture) -> &PlanNode {
            fixture
                .compilation
                .executable()
                .execution_plan()
                .payload()
                .nodes()
                .iter()
                .find(|node| node.id().as_str() == "node.attention")
                .unwrap()
        }
        let node = attention(self);
        assert_eq!(
            node.operation_id().as_str(),
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_OPERATION_ID
        );
        assert_eq!(node.operation_version(), ContractVersion::new(1, 0));
        assert_eq!(
            node.selection().selected_provider().as_str(),
            "provider.cuda.gated_delta_recurrent_attention.f32-master.q8-projections"
        );
        // The semantic node's additional requirements are distinct from the
        // capabilities of the actual registry-bound provider.
        let provider = self
            .providers
            .providers()
            .iter()
            .map(|provider| provider.descriptor())
            .find(|provider| provider.provider_id() == node.selection().selected_provider())
            .unwrap();
        assert!(provider.capabilities().contains(&id::<CapabilityId>(
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_CAPABILITY_ID
        )));
        let scratch = node.provider_resources().scratch().unwrap();
        let old = attention(strict).provider_resources().scratch().unwrap();
        let (
            ProviderWorkspaceSizeFormula::Affine {
                fixed_bytes,
                bytes_per_sequence,
                bytes_per_token,
            },
            ProviderWorkspaceSizeFormula::Affine {
                fixed_bytes: old_fixed,
                bytes_per_sequence: old_sequence,
                bytes_per_token: old_token,
            },
        ) = (scratch.size_formula(), old.size_formula())
        else {
            panic!("attention must charge its actual affine workspace");
        };
        assert_eq!(fixed_bytes, old_fixed);
        assert_eq!(bytes_per_sequence, old_sequence);
        // Both native matrices have K=256. One I8 value per input and one
        // F32 scale per K32 group, reused by the two sequential projections.
        assert_eq!(bytes_per_token - old_token, HIDDEN + HIDDEN / 32 * 4);
        assert_eq!(scratch.scope(), ProviderWorkspaceScope::Invocation);
        assert_eq!(
            scratch.reuse_policy(),
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead
        );
        assert!(node.scratch_resource().is_some() && node.binding_resource().is_some());
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
                    id(format!("run.fixture.checkpoint.{name}")),
                    id(format!("request.fixture.checkpoint.{name}")),
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
        self.execute_checked(session, tokens, range, false, false)
            .unwrap()
    }

    #[cfg(feature = "cuda")]
    pub fn execute_replayed(
        &self,
        session: &Arc<SequenceSession<Runtime>>,
        tokens: Arc<[u32]>,
        range: Range<usize>,
    ) -> Observation {
        self.execute_checked(session, tokens, range, true, false)
            .unwrap()
    }

    pub fn execute_numerical_failure(
        &self,
        session: &Arc<SequenceSession<Runtime>>,
        tokens: Arc<[u32]>,
        range: Range<usize>,
        replay: bool,
    ) {
        assert!(self
            .execute_checked(session, tokens, range, replay, true)
            .is_none());
    }

    fn execute_checked(
        &self,
        session: &Arc<SequenceSession<Runtime>>,
        tokens: Arc<[u32]>,
        range: Range<usize>,
        replay: bool,
        expect_failure: bool,
    ) -> Option<Observation> {
        let catalog = replay.then(|| self.lane.reusable_execution_catalog().unwrap());
        let span = token_span(Arc::clone(&tokens), range.clone());
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(session)]).unwrap();
        let mut request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(vec![span]).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        if let Some(bucket) = &self.reusable_bucket {
            request = request.with_reusable_execution_bucket(bucket.clone());
        }
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
                if *bytes_per_token == HIDDEN * self.output_type.size_bytes()
        ));
        // Step-token readbacks are participant-local, unlike uploads' source
        // coordinates. Completion translates this span-local range into the
        // same packed backing that the provider's output_start selects.
        let mut requests = vec![CompletionReadbackRequest::new(
            attention.id().clone(),
            0,
            output_component.resource_id().clone(),
            output_component.offset_bytes(),
            HostTransferLayout::new(self.output_type, range.len() as u64 * HIDDEN).unwrap(),
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
        let handle = if let Some(catalog) = catalog {
            let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
                self.providers.providers(),
                executable,
                &wave,
                &self.lane,
            )
            .unwrap()
            .expect("typed slots must authorize a reusable topology");
            let program = catalog
                .programs()
                .iter()
                .find(|program| program.program_id() == &program_id)
                .expect("real backend must publish the warmed topology, without eager fallback");
            assert!(
                program.is_determinism_ready(),
                "incomplete replay program: {program:?}"
            );
            assert!(
                attention.binding_resource().is_some(),
                "dual-state attention must declare a typed binding for replay"
            );
            // CUDA embedding uses direct kernel arguments; Metal embedding
            // declares its own binding workspace. Require the exact selected
            // providers' binding nodes, rather than imposing either topology.
            let declared_binding_nodes = plan
                .payload()
                .nodes()
                .iter()
                .enumerate()
                .filter(|(_, node)| node.binding_resource().is_some())
                .map(|(index, _)| u32::try_from(index).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                program.per_wave_binding_node_indices(),
                declared_binding_nodes.as_slice(),
                "every provider-declared typed binding must update before replay"
            );
            OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_policy(
                self.providers.providers(),
                executable,
                &identity,
                std::iter::once(&active),
                DeviceTimingMode::Off,
                &[input],
                program,
                SubmissionExecutionPolicy::determinism_replayed(1),
                wave,
                &self.lane,
                &self.reaper,
            )
            .unwrap()
        } else {
            OperationDispatch::encode_and_submit_wave_with_inputs(
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
            .unwrap()
        };
        if expect_failure {
            let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
                panic!("numerical failure must reach a quiescent terminal");
            };
            let OperationCompletionDisposition::FailedButQuiescent(failures) =
                receipt.disposition()
            else {
                panic!("non-finite input must report a device numerical failure: {receipt:?}");
            };
            // A backend may submit embedding and attention under one fence.
            // The completion layer then attributes that device error to every
            // participant in the batch rather than only the attention node.
            assert!(
                !failures.is_empty(),
                "missing numerical failure attribution"
            );
            for failure in failures {
                assert_eq!(failure.failure().domain(), FailureDomain::Device);
                assert!(
                    failure.failure().message().contains("non-finite"),
                    "unexpected device failure: {failures:?}"
                );
            }
            drop((receipt, handle, identity, active));
            // Retiring a fresh frame can succeed without a completed FullPlan
            // proof: that retires ownership and leaves the frontier Unproven.
            // The safety boundary is capture authorization, not the retirement
            // receipt's label. Exercise it before closing the failed request.
            match step.try_retire_normal() {
                Ok(_) => {
                    self.assert_failed_sequence_has_no_checkpoint(session);
                    session.try_abort_if_quiescent().unwrap();
                }
                Err(failure) => {
                    failure.into_step().try_abort().unwrap();
                    self.assert_failed_sequence_has_no_checkpoint(session);
                }
            }
            return None;
        }
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
        Some(Observation {
            values,
            state_types: self
                .states
                .iter()
                .map(|s| (s.id.to_string(), s.tensor.element_type))
                .chain(std::iter::once(("output".to_owned(), self.output_type)))
                .collect(),
        })
    }

    fn assert_failed_sequence_has_no_checkpoint(&self, session: &Arc<SequenceSession<Runtime>>) {
        let binding = self.resources.trusted_runtime_binding().unwrap();
        let result = self.reaper.try_capture_sequence_checkpoint(
            self.compilation.executable().execution_plan(),
            &binding,
            Arc::clone(session),
            Arc::clone(&self.lane),
        );
        assert!(
            matches!(result, Err(_) | Ok(NativeCheckpointStart::Skipped(_))),
            "a failed full-plan wave must never authorize checkpoint allocation or publication"
        );
        assert_eq!(
            CapacitySnapshot::observe(&self.resources).checkpoint_claims,
            0
        );
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
    state_types: BTreeMap<String, ElementType>,
}
impl Observation {
    #[cfg(feature = "cuda")]
    pub fn assert_different_output(&self, other: &Self) {
        assert_ne!(
            self.values["output"], other.values["output"],
            "changed token input must produce different output, not stale graph data"
        );
    }

    pub fn assert_state_nonzero(&self) {
        for (name, bytes) in &self.values {
            if name == "output" {
                continue;
            }
            let values = match self.state_types[name] {
                ElementType::F32 => bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                ElementType::F16 => bytes
                    .chunks_exact(2)
                    .map(|bytes| {
                        f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
                    })
                    .collect::<Vec<_>>(),
                ElementType::I8 => bytes
                    .iter()
                    .map(|byte| i8::from_le_bytes([*byte]) as f32)
                    .collect::<Vec<_>>(),
                other => panic!("fixture does not define state observations for {other:?}"),
            };
            assert!(
                !values.is_empty() && values.iter().all(|value| value.is_finite()),
                "{name}: state must be finite and nonempty"
            );
            assert!(
                values.iter().any(|value| *value != 0.0),
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
        let output = match self.state_types["output"] {
            ElementType::F32 => self.values["output"]
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
            ElementType::F16 => self.values["output"]
                .chunks_exact(2)
                .map(|bytes| f16::from_le_bytes(bytes.try_into().unwrap()).to_f32())
                .collect::<Vec<_>>(),
            other => panic!("unexpected output dtype {other:?}"),
        };
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
