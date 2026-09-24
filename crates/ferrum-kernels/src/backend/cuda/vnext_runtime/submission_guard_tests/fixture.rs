use super::*;

struct Scale {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
    program_binding: bool,
    pub(super) enqueues: Arc<AtomicU64>,
    pub(super) encoded: Arc<AtomicU64>,
}
impl OperationResourceEstimator for Scale {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().fingerprint()? != self.descriptor.operation_fingerprint() {
            return Err(invalid("scale estimator operation mismatch"));
        }
        let estimate = OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            None,
        );
        Ok(if self.program_binding {
            estimate.with_binding(ProviderWorkspaceRequirement::new(
                16,
                16,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?)
        } else {
            estimate
        })
    }
}
impl OperationProvider<CudaDeviceRuntime> for Scale {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        // No graph authority is asserted for this eager-only fixture.
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        assert_eq!(invocation.participants().len(), 1);
        assert_eq!(invocation.work_shape().immediate_tokens(), 1);
        let first = &invocation.participants()[0];
        let input = first
            .bindings()
            .iter()
            .find(|b| b.role() == ResolvedValueRole::Input && b.ordinal() == 0)
            .unwrap();
        let output = first
            .bindings()
            .iter()
            .find(|b| b.role() == ResolvedValueRole::Output && b.ordinal() == 0)
            .unwrap();
        assert_eq!(
            input.storage(),
            output.storage(),
            "standard scale alias contract"
        );
        assert_eq!(
            first.attributes().get(&id("hidden_size")),
            Some(&SemanticValue::Unsigned(4))
        );
        assert_eq!(
            first.attributes().get(&id("scale")),
            Some(&SemanticValue::Rational(
                CanonicalRational::new(2, 1).unwrap()
            ))
        );
        let component = &input.storage().components()[0];
        let view = first
            .views()
            .iter()
            .find(|v| v.resource_id() == component.resource_id())
            .unwrap();
        let regions = view.translate(component.offset_bytes(), 8).unwrap();
        let mut iter = regions.iter();
        let physical = iter.next().unwrap();
        assert!(iter.next().is_none());
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let region = buffer.retained_region(range, retention).unwrap();
        let function = self.function.clone();
        let enqueues = Arc::clone(&self.enqueues);
        self.encoded.fetch_add(1, Ordering::Relaxed);
        let command = CudaDeviceCommand::operation(
            "cuda_guard_fixture_scale",
            vec![region],
            move |stream, regions| {
                enqueues.fetch_add(1, Ordering::Relaxed);
                let pointer = regions[0].device_ptr();
                let scale = 2.0_f32;
                let elements = 4_i32;
                let mut launch = stream.launch_builder(&function);
                launch.arg(&pointer);
                launch.arg(&scale);
                launch.arg(&elements);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| CudaDeviceRuntimeError::driver("guard fixture scale", e))
            },
        )
        .unwrap()
        .with_work_attribution(DeviceBatchingForm::Packed, 1, 1, 1, 0)
        .unwrap();
        let operation = EncodedDeviceOperation::compute(command);
        Ok(if self.program_binding {
            operation.with_program_binding(self.binding_command(&invocation))
        } else {
            operation
        })
    }
}

impl Scale {
    fn binding_command(
        &self,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> CudaDeviceCommand {
        let binding = invocation.program_binding().cloned().unwrap();
        let view = invocation.participants()[0].binding_view().unwrap();
        let regions = view.translate(0, 16).unwrap();
        let mut pieces = regions.iter();
        let physical = pieces.next().unwrap();
        assert!(pieces.next().is_none());
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let destination = buffer.retained_region(range, retention).unwrap();
        CudaDeviceCommand::program_binding_patch(
            "cuda_guard_fixture_binding",
            binding,
            destination,
            vec![CudaProgramBindingWrite::new(0, vec![0x5a; 16].into_boxed_slice()).unwrap()],
            vec![],
        )
        .unwrap()
        .with_work_attribution(DeviceBatchingForm::Packed, 1, 1, 0, 1)
        .unwrap()
    }
}

pub(super) struct Fixture {
    pub(super) runtime: Arc<CudaDeviceRuntime>,
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<CudaDeviceRuntime>,
    resources: Arc<PlanRuntimeResources<CudaDeviceRuntime>>,
    pub(super) session: Arc<SequenceSession<CudaDeviceRuntime>>,
    batch: ExecutionBatchParticipants<CudaDeviceRuntime>,
    bucket: Option<ReusableExecutionBucketSpec>,
    pub(super) lane: Arc<ExecutionLane<CudaDeviceRuntime>>,
    pub(super) reaper: Arc<CompletionReaper<CudaDeviceRuntime>>,
    pub(super) enqueues: Arc<AtomicU64>,
    pub(super) encoded: Arc<AtomicU64>,
}

impl Fixture {
    pub(super) fn new() -> Self {
        Self::configured(false)
    }
    pub(super) fn new_program_binding() -> Self {
        Self::configured(true)
    }
    fn configured(program_binding: bool) -> Self {
        let runtime = Arc::new(
            CudaDeviceRuntime::new(CudaDeviceRuntimeConfig {
                ordinal: 0,
                device_id: id("device.cuda-guard-fixture"),
                attention_execution_policy: AttentionExecutionPolicy::Portable,
                runtime_implementation_fingerprint: digest(include_bytes!("../submission.rs")),
                capabilities: BTreeSet::from([id(CONSTANT_SCALE_F16_CAPABILITY_ID)]),
                dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                    DynamicStorageAllocator::LinearArena,
                    DynamicStorageView::Contiguous,
                )
                .unwrap()]),
            })
            .expect("requires an actual configured CUDA device"),
        );
        let contract = family::scale_contract(program_binding);
        let descriptor = OperationProviderDescriptor::new(
            id("provider.cuda-guard-fixture.scale"),
            contract.descriptor().id.clone(),
            contract.descriptor().fingerprint().unwrap(),
            digest(crate::ptx::FUSED_SILU_MUL.as_bytes()),
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
            BTreeSet::new(),
            BTreeSet::new(),
            vec![
                ProviderStorageBindingRequirement::new(
                    ResolvedValueRole::Input,
                    0,
                    DynamicStorageRequirement::contiguous(),
                ),
                ProviderStorageBindingRequirement::new(
                    ResolvedValueRole::Output,
                    0,
                    DynamicStorageRequirement::contiguous(),
                ),
            ],
            "resource-estimator.cuda-guard-fixture",
            ContractVersion::new(1, 0),
            digest(if program_binding {
                b"scale-invocation-binding-v1"
            } else {
                b"scale-no-workspace-v1"
            }),
        )
        .unwrap();
        let function = runtime
            .context
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .unwrap()
            .load_function("scale_inplace_f16")
            .unwrap();
        let enqueues = Arc::new(AtomicU64::new(0));
        let encoded = Arc::new(AtomicU64::new(0));
        let registry = OperationRuntimeRegistry::new(
            vec![contract],
            vec![Box::new(Scale {
                descriptor,
                function,
                program_binding,
                enqueues: Arc::clone(&enqueues),
                encoded: Arc::clone(&encoded),
            })],
        )
        .unwrap();
        let catalog = registry
            .capability_catalog(
                runtime.descriptor().clone(),
                vec![EngineProviderDescriptor::new(
                    id("provider.engine.cuda-guard-fixture"),
                    ContractVersion::new(1, 0),
                    digest(b"guard-fixture-engine-v1"),
                    runtime.descriptor().id.clone(),
                    runtime.descriptor().capabilities.clone(),
                )
                .unwrap()],
            )
            .unwrap();
        let family = TypedFamilyRegistration::new(if program_binding {
            family::Family::with_program_binding()
        } else {
            family::Family::default()
        })
        .prepare_with_profile(&serde_json::json!({"width": 4}), &id("fixture.f16"))
        .unwrap();
        // Workspace reuse is deliberately independent from graph execution.
        let bucket = program_binding.then(|| {
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("fixture.decode").unwrap(),
                ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
            )
            .unwrap()
        });
        let reusable = bucket
            .as_ref()
            .map(|bucket| ReusableExecutionPolicy::new(1, vec![bucket.clone()]).unwrap());
        assert!(reusable
            .as_ref()
            .is_none_or(|policy| policy.program_policy().is_none()));
        let policy = ResolvedRuntimePolicy::new(
            "runtime-policy.cuda-guard-fixture",
            ContractVersion::new(1, 0),
            SchedulingDiscipline::FirstReady,
            RuntimeMemoryPolicy {
                checkpoint_capacity: None,
                capacity_bytes: 1 << 20,
                reserve_bytes: 4096,
                maximum_active_sequences: 1,
                dynamic_storage_profile_order: runtime
                    .descriptor()
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            },
            AdmissionPolicy {
                maximum_queue_depth: 1,
                maximum_scheduled_tokens: 1,
                sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
                allow_defer: true,
                cancellation_check_interval_steps: 1,
            },
            AttentionExecutionPolicy::Portable,
            ExecutionDeterminismRequirement::BitwiseSameRuntime,
            reusable,
        )
        .unwrap();
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id("value.input"),
            ProgramTensorSpec {
                dimensions: vec![1, 4],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        options.retain_completion_value(id("value.output"));
        let compilation = ProgramPlanCompiler::compile(
            &family,
            &catalog,
            &policy,
            &registry.planning(),
            &options,
        )
        .unwrap();
        let providers = registry.bind_plan(compilation.executable()).unwrap();
        let resources = match compilation
            .executable()
            .execution_plan()
            .provision_static(Arc::clone(&runtime), id("request.provision"))
            .unwrap()
            .into_provisioning()
        {
            StaticProvisioning::Required(permit) => {
                let identity = ResourceTransactionIdentity::for_admission(
                    permit.binding(),
                    id("run.provision"),
                    id("transaction.provision"),
                );
                let committed = ResourceTransaction::begin(
                    RuntimeResourceDriver::new(Arc::clone(&runtime)).unwrap(),
                    identity,
                    permit,
                )
                .unwrap()
                .reserve()
                .unwrap()
                .commit()
                .unwrap_or_else(|_| panic!("real static commit failed"));
                for pool in committed.maintenance_controller().pool_ids() {
                    committed
                        .maintenance_controller()
                        .initialize_pool(pool)
                        .unwrap();
                }
                committed
                    .into_plan_runtime()
                    .unwrap_or_else(|error| panic!("plan handoff: {}", error.error()))
            }
            StaticProvisioning::NoStatic(value) => value.into_plan_runtime(),
        };
        let work = ResourceWorkShape::single(span()).unwrap();
        let request = RequestResourceAdmissionRequest::new(
            work.clone(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let binding = resources.trusted_runtime_binding().unwrap();
        let admitted = (0..4)
            .find_map(|_| {
                match binding
                    .try_admit_request(request.clone(), id("run.guard"), id("request.guard"))
                    .unwrap()
                {
                    RequestResourceAdmissionDecision::Admitted(value) => Some(value),
                    RequestResourceAdmissionDecision::BackingDeferred(value) => {
                        value.maintain().unwrap();
                        None
                    }
                    _ => panic!("unexpected request rejection"),
                }
            })
            .expect("bounded real request maintenance");
        let request = SequenceResourceAdmissionRequest::new(
            work,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let sequence = (0..4)
            .find_map(
                |_| match admitted.try_admit_sequence(request.clone()).unwrap() {
                    SequenceResourceAdmissionDecision::Admitted(value) => Some(value),
                    SequenceResourceAdmissionDecision::BackingDeferred(value) => {
                        value.maintain().unwrap();
                        None
                    }
                    _ => panic!("unexpected sequence rejection"),
                },
            )
            .expect("bounded real sequence maintenance");
        let session = sequence.open_session().unwrap();
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
        let lane = resources.create_execution_lane().unwrap();
        lane.configure_submission_readback_staging(8).unwrap();
        Self {
            runtime,
            compilation,
            providers,
            resources,
            session,
            batch,
            bucket,
            lane,
            reaper: CompletionReaper::new(),
            enqueues,
            encoded,
        }
    }

    pub(super) fn has_program_binding(&self) -> bool {
        self.bucket.is_some()
    }

    pub(super) fn prepare(
        &self,
    ) -> (
        Arc<StepResourceLease<CudaDeviceRuntime>>,
        PreparedStepSubmissionWave<CudaDeviceRuntime>,
    ) {
        let mut request = StepResourceAdmissionRequest::new(
            self.batch.bind_work_shape(vec![span()]).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        if let Some(bucket) = &self.bucket {
            request = request.with_reusable_execution_bucket(bucket.bucket_id().clone());
        }
        let step = (0..4)
            .find_map(|_| {
                match self
                    .batch
                    .try_begin_step(request.clone(), &self.lane)
                    .unwrap()
                {
                    StepResourceAdmissionDecision::Admitted(value) => Some(value),
                    StepResourceAdmissionDecision::BackingDeferred(value) => {
                        value.maintain().unwrap();
                        None
                    }
                    _ => panic!("unexpected step rejection"),
                }
            })
            .expect("bounded real step maintenance");
        let wave = (0..4)
            .find_map(|_| {
                match step
                    .try_prepare_full_plan_submission_wave(
                        Arc::new(step.work_shape().clone()),
                        AdmissionFitPolicy::ImmediateOnly,
                        AdmissionPressureAction::WaitForRelease,
                    )
                    .unwrap()
                {
                    StepSubmissionWaveAdmissionDecision::Prepared(value) => Some(value),
                    StepSubmissionWaveAdmissionDecision::BackingDeferred(value) => {
                        value.maintain().unwrap();
                        None
                    }
                    _ => panic!("unexpected wave rejection"),
                }
            })
            .expect("bounded real wave maintenance");
        (
            step,
            wave.with_submission_readbacks(self.readback()).unwrap(),
        )
    }
    pub(super) fn readback(&self) -> CompletionReadbackBatchRequest {
        let node = &self
            .compilation
            .executable()
            .execution_plan()
            .payload()
            .nodes()[0];
        let output = node
            .values()
            .iter()
            .find(|v| v.role() == ResolvedValueRole::Output && v.ordinal() == 0)
            .unwrap();
        let part = &output.storage().components()[0];
        CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new(
            node.id().clone(),
            0,
            part.resource_id().clone(),
            part.offset_bytes(),
            HostTransferLayout::new(ElementType::F16, 4).unwrap(),
        )
        .unwrap()])
        .unwrap()
    }
    pub(super) fn dispatch(
        &self,
        wave: PreparedStepSubmissionWave<CudaDeviceRuntime>,
        guard: &Guard,
    ) -> GuardedWaveSubmissionOutcome<CudaDeviceRuntime> {
        let active = TrustedActiveSequenceBinding::from_session(&self.session).unwrap();
        let identity = OperationDispatch::bind_submission_wave_identity(
            self.compilation.executable(),
            [&active].into_iter(),
            &wave,
            &self.lane,
        )
        .unwrap();
        let input = SubmissionWaveInputUpload::new(
            id("node.scale"),
            0,
            0,
            0,
            HostTransferLayout::new(ElementType::F16, 4).unwrap(),
            [1.0_f32, -2.0, 0.5, 4.0]
                .into_iter()
                .flat_map(|v| f16::from_f32(v).to_le_bytes())
                .collect(),
        )
        .unwrap();
        OperationDispatch::encode_and_submit_guarded_wave(
            self.providers.providers(),
            self.compilation.executable(),
            &identity,
            [&active].into_iter(),
            &[input],
            guard,
            wave,
            &self.lane,
            &self.reaper,
        )
    }
    pub(super) fn close(self, completed: bool) {
        assert_eq!(self.reaper.retained_count(), 0);
        assert_eq!(self.reaper.quarantined_count(), 0);
        if completed {
            self.session.try_complete().unwrap();
        } else {
            self.session.try_abort_if_quiescent().unwrap();
        }
        drop((
            self.batch,
            self.session,
            self.lane,
            self.reaper,
            self.providers,
            self.compilation,
        ));
        assert!(matches!(
            PlanRuntimeResources::close(self.resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}
