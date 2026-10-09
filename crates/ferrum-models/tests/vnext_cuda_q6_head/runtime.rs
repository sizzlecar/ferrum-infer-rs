use super::*;
use std::ops::Range;
#[path = "dispatch.rs"]
mod dispatch;
pub use dispatch::{Path, SubmissionRendezvous};

pub struct Fixture {
    _owners: (
        Arc<Runtime>,
        OperationRuntimeRegistry<Runtime>,
        WeightMaterializerRegistry,
        CapabilityCatalog,
    ),
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<Runtime>,
    pub resources: Arc<PlanRuntimeResources<Runtime>>,
    pub lane: Arc<ExecutionLane<Runtime>>,
    pub reaper: Arc<CompletionReaper<Runtime>>,
    reusable_bucket: ReusableExecutionBucketId,
    pub source: Weights,
    pub head: Head,
}
impl Fixture {
    pub fn new(head: Head, participants: u32, bad_token: Option<u32>, bad_weight: bool) -> Self {
        let family = TypedFamilyRegistration::new(Family::new(head))
            .prepare_with_profile(&serde_json::to_value(head).unwrap(), &id(PROFILE))
            .unwrap();
        let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
            0,
            id("device.cuda.q6-head-fixture"),
            ferrum_types::AttentionExecutionPolicy::Portable,
        )
        .unwrap()
        .into_parts();
        let materializer = cuda_weight_materializer_selection(&family).unwrap();
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("fixture.q6-head.decode").unwrap(),
            ReusableExecutionCapacity::new(
                participants,
                u64::from(participants) * MAX_TOKENS,
                u64::from(participants) * MAX_TOKENS,
            )
            .unwrap(),
        )
        .unwrap();
        let reusable_bucket = bucket.bucket_id().clone();
        let policy = ResolvedRuntimePolicy::new(
            "runtime-policy.fixture.q6-head",
            ContractVersion::new(1, 0),
            SchedulingDiscipline::FirstReady,
            RuntimeMemoryPolicy {
                checkpoint_capacity: None,
                capacity_bytes: 256 << 20,
                reserve_bytes: 1 << 20,
                maximum_active_sequences: participants,
                dynamic_storage_profile_order: runtime
                    .descriptor()
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            },
            AdmissionPolicy {
                maximum_queue_depth: participants,
                maximum_scheduled_tokens: u64::from(participants) * MAX_TOKENS,
                sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
                allow_defer: true,
                cancellation_check_interval_steps: 1,
            },
            runtime.attention_execution_policy(),
            ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay,
            Some(ReusableExecutionPolicy::new(1, vec![bucket]).unwrap()),
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
        let plan = compilation.executable().execution_plan();
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|n| n.id().as_str() == "node.head")
            .unwrap();
        let requirement = node.provider_resources().persistent().unwrap();
        assert_eq!(
            requirement.fixed_bytes(),
            Some(4),
            "one admitted validation flag per physical leaf"
        );
        let allocation = plan
            .payload()
            .memory()
            .static_allocations()
            .iter()
            .find(|a| Some(a.resource_id()) == node.persistent_resource())
            .unwrap();
        assert_eq!(allocation.lifetime(), AllocationLifetime::Plan);
        assert_eq!(allocation.usage(), BufferUsage::Persistent);
        let providers = registry.bind_plan(compilation.executable()).unwrap();
        let provisioning = plan
            .provision_static(Arc::clone(&runtime), id("request.q6-head.provision"))
            .unwrap();
        let StaticProvisioning::Required(permit) = provisioning.into_provisioning() else {
            panic!("fixture has static weights")
        };
        let identity = ResourceTransactionIdentity::for_admission(
            permit.binding(),
            id("run.q6-head.provision"),
            id("transaction.q6-head.provision"),
        );
        let reserved = ResourceTransaction::begin(
            RuntimeResourceDriver::new(Arc::clone(&runtime)).unwrap(),
            identity,
            permit,
        )
        .unwrap()
        .reserve()
        .unwrap();
        let committed = match reserved.commit() {
            Ok(v) => v,
            Err(ResourceCommitTransitionError::Recoverable(e)) => {
                panic!("static commit: {:?}", e.failure())
            }
            Err(ResourceCommitTransitionError::Poisoned(e)) => {
                panic!("indeterminate static commit: {:?}", e.failure())
            }
        };
        let source = Weights::new(family.weight_schema(), bad_token, bad_weight);
        let initialized = committed
            .initialize_static(
                &family,
                plan,
                &source,
                StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
            )
            .unwrap();
        let resources = match initialized.into_plan_runtime() {
            Ok(v) => v,
            Err(e) => panic!("runtime handoff: {}", e.error()),
        };
        let lane = resources.create_execution_lane().unwrap();
        lane.configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(8).unwrap())
            .unwrap();
        Self {
            _owners: (runtime, registry, materializers, catalog),
            compilation,
            providers,
            resources,
            lane,
            reaper: CompletionReaper::new(),
            reusable_bucket,
            source,
            head,
        }
    }
    pub fn admit(&self, name: &str, tokens: Arc<[u32]>) -> Arc<SequenceSession<Runtime>> {
        let work = ResourceWorkShape::single(token_span(tokens.clone(), 0..tokens.len())).unwrap();
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
                    id(format!("run.q6-head.{name}")),
                    id(format!("request.q6-head.{name}")),
                )
                .unwrap()
            {
                RequestResourceAdmissionDecision::Admitted(r) => break r,
                RequestResourceAdmissionDecision::BackingDeferred(d) => {
                    require_progress(d.maintain().unwrap())
                }
                RequestResourceAdmissionDecision::Deferred(r) => {
                    require_progress(self.resources.maintain_for_admission_deferred(&r).unwrap())
                }
                RequestResourceAdmissionDecision::PermanentRejected(r) => {
                    panic!("request rejected: {r:?}")
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
                SequenceResourceAdmissionDecision::Admitted(r) => break r,
                SequenceResourceAdmissionDecision::BackingDeferred(d) => {
                    require_progress(d.maintain().unwrap())
                }
                SequenceResourceAdmissionDecision::Deferred(r) => {
                    require_progress(self.resources.maintain_for_admission_deferred(&r).unwrap())
                }
                SequenceResourceAdmissionDecision::PermanentRejected(r) => {
                    panic!("sequence rejected: {r:?}")
                }
            }
        };
        sequence.open_session().unwrap()
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
        "no owner can release capacity: {outcome:?}"
    );
}
