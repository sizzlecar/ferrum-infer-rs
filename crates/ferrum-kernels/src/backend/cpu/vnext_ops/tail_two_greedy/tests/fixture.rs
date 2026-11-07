//! Genuine Stored CPU token plan; only lifecycle plumbing is reused from the
//! GDN fixture. Arithmetic is the production tail-head provider.
use crate::backend::cpu::{vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime};
use ferrum_runtime::*;
use half::f16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const H: usize = 4;
pub const V: usize = 5;
pub const X: &str = "value.tail.x";
pub const IDS: &str = "value.tail.ids";
pub const CARRIER: &str = "value.tail.hidden";
const W: &str = "value.tail.weight";
const PROFILE: &str = "profile.fixture.tail";
pub fn id<T: TryFrom<String>>(v: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(v.into()).unwrap()
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
}
impl Config {
    fn inputs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([(id(X), tensor(vec![self.rows, H as u64], ElementType::F16))])
    }
}
struct Family {
    id: ModelFamilyId,
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.tail")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(format!("{e}")))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if c.rows == 0 || serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("invalid tail fixture configuration"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.tail"))
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fixture.tail"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.tail.head"),
                role: WeightComponentRole::Values,
                external_names: vec!["head".into()],
                dimensions: vec![V as u64, H as u64],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.tail.head"),
                dimensions: vec![V as u64, H as u64],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("component.tail.head"),
                },
                required: true,
            }],
        }))
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(CARRIER),
                boundaries: BTreeMap::from([
                    (id(IDS), ElementType::U32),
                    (id(CARRIER), ElementType::F16),
                ]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(TAIL_TWO_GREEDY_F16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.id.clone(),
            vec![id(X)],
            vec![ProgramBlock {
                id: "block.tail".into(),
                nodes: vec![ProgramNode {
                    id: id("node.tail"),
                    operation_id: id(TAIL_TWO_GREEDY_F16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::tokens(id(X), 0),
                    inputs: vec![id(X), id(W)],
                    outputs: vec![id(IDS), id(CARRIER)],
                    attributes: BTreeMap::from([
                        (id("hidden_size"), SemanticValue::Unsigned(H as u64)),
                        (id("vocab_size"), SemanticValue::Unsigned(V as u64)),
                    ]),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.tail.head"),
                value_id: id(W),
                tensor: tensor(vec![V as u64, H as u64], ElementType::F16),
            }],
            vec![id(IDS), id(CARRIER)],
        )?
        .with_input_tensor_specs(c.inputs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(CausalLanguageMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".into(),
                source_file: "fixture.json".into(),
                sha256: format!("{:x}", Sha256::digest(b"{{ messages }}")),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: Some(1),
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: Some(0),
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        }
        .into())
    }
}
fn prepare(rows: u64) -> Result<PreparedModelFamily, VNextError> {
    TypedFamilyRegistration::new(Family {
        id: id("family.fixture.tail"),
    })
    .prepare_with_profile(&serde_json::json!({"rows":rows}), &id(PROFILE))
}
pub fn compile(
    rows: u64,
    composition: &CpuVNextComposition,
    capacity: u64,
) -> Result<ProgramPlanCompilation, VNextError> {
    let family = prepare(rows)?;
    let descriptor = composition.runtime().descriptor();
    let policy = ResolvedRuntimePolicy::new(
        "runtime-policy.fixture.tail",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            checkpoint_capacity: None,
            capacity_bytes: capacity,
            reserve_bytes: 0,
            maximum_active_sequences: 1,
            dynamic_storage_profile_order: descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        },
        AdmissionPolicy {
            maximum_queue_depth: 1,
            maximum_scheduled_tokens: rows,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            sequence_backing_growth: Default::default(),
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntime,
        None,
    )?;
    let mut options = ProgramPlanCompileOptions::new(Config { rows }.inputs())?;
    options.retain_completion_value(id(IDS));
    options.retain_completion_value(id(CARRIER));
    ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        &WeightMaterializerRegistry::identity_only()?,
        &options,
    )
}
pub fn encode(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_le_bytes())
        .collect()
}
struct Parameters {
    bytes: Vec<u8>,
}
impl Parameters {
    fn new() -> Self {
        Self {
            bytes: encode([
                1.,
                0.,
                0.,
                0.,
                1.,
                0.000244140625,
                0.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                0.,
                0.,
                1.,
                0.,
            ]),
        }
    }
}
impl WeightComponentSource for Parameters {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if c.id != id("component.tail.head") || c.dimensions != [V as u64, H as u64] {
            return Err(invalid("unexpected tail weight"));
        }
        WeightComponentPayload::new(
            c,
            "head",
            "tail.typed-fixture",
            c.dimensions.clone(),
            ElementType::F16,
            &self.bytes,
        )
    }
}
pub fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
pub struct Session {
    pub compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<CpuDeviceRuntime>,
    resources: Arc<PlanRuntimeResources<CpuDeviceRuntime>>,
    lane: Arc<ExecutionLane<CpuDeviceRuntime>>,
    reaper: Arc<CompletionReaper<CpuDeviceRuntime>>,
}
pub struct Outputs {
    pub terminal: Vec<(ProgramValueId, ManagedHostBytes)>,
}
pub fn session(rows: u64, composition: &CpuVNextComposition, host: &HostMemoryAccount) -> Session {
    let family = prepare(rows).unwrap();
    let compilation = compile(
        rows,
        composition,
        composition.runtime().descriptor().total_memory_bytes,
    )
    .unwrap();
    let executable = compilation.executable();
    let plan = executable.execution_plan();
    let providers = composition.registry().bind_plan(executable).unwrap();
    let StaticProvisioning::Required(permit) = plan
        .provision_static(composition.runtime().clone(), id("request.tail.init"))
        .unwrap()
        .into_provisioning()
    else {
        panic!("Stored tail requires real initialization")
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.tail.init"),
        id("transaction.tail.init"),
    );
    let reserved = ResourceTransaction::begin(
        RuntimeResourceDriver::new(composition.runtime().clone()).unwrap(),
        identity,
        permit,
    )
    .unwrap()
    .reserve()
    .unwrap();
    let committed = match reserved.commit() {
        Ok(v) => v,
        Err(ResourceCommitTransitionError::Recoverable(e)) => panic!("commit: {:?}", e.failure()),
        Err(ResourceCommitTransitionError::Poisoned(e)) => {
            panic!("unknown commit: {:?}", e.failure())
        }
    };
    // Provision the declared maximum once: this arithmetic/lifetime fixture
    // does not claim to qualify elastic fragmentation recovery.
    let growth = plan
        .payload()
        .memory()
        .dynamic_pools()
        .iter()
        .map(|pool| {
            DynamicPoolGrowthRequest::new(
                pool.pool_id().clone(),
                pool.provisioning().maximum_resident_bytes(),
            )
            .unwrap()
        })
        .collect();
    committed
        .maintenance_controller()
        .grow_pools(growth)
        .unwrap();
    let initialized = committed
        .initialize_static_managed(
            &family,
            plan,
            &Parameters::new(),
            host,
            StaticInitializationPolicy::new(4096, 16).unwrap(),
        )
        .unwrap();
    let resources = initialized
        .into_plan_runtime()
        .unwrap_or_else(|e| panic!("handoff: {:?}", e.failure()));
    let lane = resources.create_execution_lane().unwrap();
    Session {
        compilation,
        providers,
        resources,
        lane,
        reaper: CompletionReaper::new(),
    }
}
// Real token admission, with bounded backing maintenance; no fabricated permits.
fn admit_sequence(
    resources: &Arc<PlanRuntimeResources<CpuDeviceRuntime>>,
    work: &ResourceWorkShape,
) -> Arc<AdmittedSequenceResources<CpuDeviceRuntime>> {
    let request = RequestResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let binding = resources.trusted_runtime_binding().unwrap();
    let request_resources = loop {
        match binding
            .try_admit_request(
                request.clone(),
                id("run.tail.step"),
                id("request.tail.step"),
            )
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(resources) => break resources,
            RequestResourceAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap());
            }
            RequestResourceAdmissionDecision::Deferred(reason) => {
                require_progress(resources.maintain_for_admission_deferred(&reason).unwrap());
            }
            RequestResourceAdmissionDecision::PermanentRejected(reason) => {
                panic!("tail head request rejected: {reason:?}");
            }
        }
    };
    let request = SequenceResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    loop {
        match request_resources
            .try_admit_sequence(request.clone())
            .unwrap()
        {
            SequenceResourceAdmissionDecision::Admitted(resources) => break resources,
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap());
            }
            SequenceResourceAdmissionDecision::Deferred(reason) => {
                require_progress(resources.maintain_for_admission_deferred(&reason).unwrap());
            }
            SequenceResourceAdmissionDecision::PermanentRejected(reason) => {
                panic!("tail head sequence rejected: {reason:?}");
            }
        }
    }
}

fn require_progress(outcome: DynamicDeferredMaintenanceOutcome) {
    assert!(
        !matches!(
            outcome,
            DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
        ),
        "tail head admission has no owner capable of releasing more capacity: {outcome:?}"
    );
}

pub fn run(fixture: &mut Session, host: &HostMemoryAccount, rows: u64, x: &[u8]) -> Outputs {
    // Tokens supply actual row work; the program's numerical inputs are the
    // separately charged hidden bytes, never token IDs as floats.
    let tokens = vec![1_u32; rows as usize];
    let span = TokenSpanWork::from_token_ids(&tokens, 0..tokens.len()).unwrap();
    let work = ResourceWorkShape::single(span.clone()).unwrap();
    let sequence = admit_sequence(&fixture.resources, &work);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![session.clone()]).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![span]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = loop {
        match batch
            .try_begin_step(request.clone(), &fixture.lane)
            .unwrap()
        {
            StepResourceAdmissionDecision::Admitted(step) => break step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                let outcome = deferred.maintain().unwrap();
                assert!(
                    !matches!(outcome, DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }),
                    "tail head step backing blocked: config={rows}, evidence={:?}, pools={:?}, outcome={outcome:?}",
                    deferred.evidence(),
                    fixture.resources.dynamic_pool_status().unwrap()
                );
                require_progress(outcome);
            }
            StepResourceAdmissionDecision::Deferred(reason) => {
                let outcome = fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap();
                assert!(
                    !matches!(
                        outcome,
                        DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
                    ),
                    "tail head step blocked: config={rows}, deferred={reason:?}, pools={:?}",
                    fixture.resources.dynamic_pool_status().unwrap()
                );
                require_progress(outcome);
            }
            StepResourceAdmissionDecision::PermanentRejected(reason) => {
                panic!("tail head step rejected: {reason:?}");
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
                    fixture
                        .resources
                        .maintain_for_admission_deferred(&reason)
                        .unwrap(),
                );
            }
            StepSubmissionWaveAdmissionDecision::PermanentRejected(reason) => {
                panic!("tail head wave rejected: {reason:?}");
            }
            other => panic!(
                "tail head wave has a request-state hazard: {:?}",
                std::mem::discriminant(&other)
            ),
        }
    };
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let executable = fixture.compilation.executable();
    let plan = executable.execution_plan();
    let identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        std::iter::once(&active),
        &wave,
        &fixture.lane,
    )
    .unwrap();
    assert_eq!(x.len(), rows as usize * H * 2);
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, x.len())
        .unwrap();
    allocation.as_mut().copy_from_slice(x);
    let uploads = vec![SubmissionWaveInputUpload::managed_range(
        id("node.tail"),
        0,
        0,
        0,
        HostTransferLayout::new(ElementType::F16, x.len() as u64 / 2).unwrap(),
        allocation.freeze(),
    )
    .unwrap()];
    let names = &[IDS, CARRIER];
    // Reserve every Host destination before submitting the actual compute wave.
    let readbacks = names
        .iter()
        .map(|name| {
            let request = plan
                .completion_checkpoint_readback_for_work(&id(*name), 0, &work)
                .unwrap();
            let output = host
                .allocate_zeroed(
                    HostMemoryClass::Output,
                    request.output_layout().byte_len().unwrap() as usize,
                )
                .unwrap();
            (request, output)
        })
        .collect();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
        fixture.providers.providers(),
        executable,
        &identity,
        std::iter::once(&active),
        DeviceTimingMode::Off,
        &uploads,
        wave,
        &fixture.lane,
        &fixture.reaper,
    )
    .unwrap();
    let (completion, outputs) = handle
        .wait_with_managed_readbacks(readbacks)
        .unwrap()
        .into_parts();
    let CompletionObservation::Terminal(receipt) = completion else {
        panic!("tail head completion not terminal")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    let terminal = names
        .iter()
        .zip(outputs.unwrap().unwrap())
        .map(|(name, bytes)| (id(*name), bytes))
        .collect();
    drop((receipt, handle, identity, active, uploads));
    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop((session, sequence));
    Outputs { terminal }
}
pub fn bytes<'a>(outputs: &'a Outputs, name: &str) -> &'a [u8] {
    outputs
        .terminal
        .iter()
        .find(|(value, _)| value.as_str() == name)
        .unwrap()
        .1
        .bytes()
}
pub fn close(session: Session) {
    let Session {
        compilation,
        providers,
        resources,
        lane,
        reaper,
    } = session;
    drop((compilation, providers, lane, reaper));
    assert!(matches!(
        PlanRuntimeResources::close(resources)
            .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
}
