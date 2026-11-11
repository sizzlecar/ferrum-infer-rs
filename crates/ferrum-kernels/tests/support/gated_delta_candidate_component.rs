//! Real Token Core CUDA fixture, adapted from the CPU GDN fixture.
//! No FixedTensor execution, fabricated permits, alternate arithmetic or driver.
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_and_dense_f16_visual_causal_runtime_config,
        cuda_vnext_dense_f16_visual_causal_contracts, cuda_vnext_dense_f16_visual_causal_providers,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    CudaDenseF16CausalProfile, CudaTensorAndCausalRuntimeConfig,
};
use ferrum_runtime::*;
use half::f16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
const DEVICE_CAPACITY_BYTES: u64 = 1 << 30;
pub struct ManagedComposition {
    runtime: Arc<CudaDeviceRuntime>,
    registry: OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: CapabilityCatalog,
}
impl ManagedComposition {
    fn runtime(&self) -> &Arc<CudaDeviceRuntime> {
        &self.runtime
    }
    fn registry(&self) -> &OperationRuntimeRegistry<CudaDeviceRuntime> {
        &self.registry
    }
    fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }
}

/// The actual H3 bounded F16 causal factory, without replay or an unmanaged
/// fallback. Its closed profile deliberately does not admit F32Master.
pub fn composition() -> ManagedComposition {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_and_causal_eager(CudaTensorAndCausalRuntimeConfig {
            tensor: CudaTensorRuntimeConfig {
                device: cuda_tensor_and_dense_f16_visual_causal_runtime_config(
                    0,
                    id("device.cuda.0"),
                )
                .unwrap(),
                usable_memory_bytes: DEVICE_CAPACITY_BYTES,
                bootstrap_memory_bytes: 128 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 3,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 1024,
                opaque_memory_profile: "unqualified-gdn-candidate-runtime-v1".into(),
            },
            causal: CudaDenseF16CausalProfile::new(
                64 << 20,
                "unqualified-gdn-candidate-modules-v1",
            )
            .unwrap()
            .with_visual_prompt_v1(),
        })
        .unwrap(),
    );
    assert!(runtime.supports_managed_tensor_io());
    assert!(runtime.supports_managed_static_weight_sources());
    let registry = OperationRuntimeRegistry::new(
        cuda_vnext_dense_f16_visual_causal_contracts().unwrap(),
        cuda_vnext_dense_f16_visual_causal_providers(&runtime).unwrap(),
    )
    .unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.gdn-candidates.fixture"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = WeightMaterializerRegistry::identity_only()
        .unwrap()
        .augment_catalog(
            registry
                .capability_catalog(descriptor.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    ferrum_kernels::native_ops::validate_compiled_native_operator_provider_catalog_for_units(
        &catalog
            .native_operator_provider_catalog(ferrum_types::NativeOperatorBackend::Cuda)
            .unwrap(),
        ferrum_kernels::native_ops::compiled_native_operator_artifacts(),
        &[],
    )
    .unwrap();
    ManagedComposition {
        runtime,
        registry,
        catalog,
    }
}
pub struct TokenSession {
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<CudaDeviceRuntime>,
    resources: Arc<PlanRuntimeResources<CudaDeviceRuntime>>,
    lane: Arc<ExecutionLane<CudaDeviceRuntime>>,
    reaper: Arc<CompletionReaper<CudaDeviceRuntime>>,
}
pub struct Outputs {
    pub terminal: Vec<(ProgramValueId, ManagedHostBytes)>,
}
pub fn compile(
    c: Config,
    composition: &ManagedComposition,
) -> Result<ProgramPlanCompilation, VNextError> {
    compile_with_capacity(c, composition, DEVICE_CAPACITY_BYTES)
}

pub const X: &str = "value.gdn.x";
pub const C: &str = "value.gdn.convolution";
pub const D: &str = "value.gdn.recurrent";
pub const Y: &str = "value.gdn.y";
pub const CC: &str = "value.gdn.candidate_convolution";
pub const DD: &str = "value.gdn.candidate_recurrent";
pub const H: usize = 8;
pub const QKV: usize = 20; // 2 key heads * 3 key dim * 2 + 4 value heads * 2 value dim.
pub const MIXED: usize = 36;
pub const CONV_BYTES: usize = QKV * 2 * 2;
pub const DELTA_BYTES: usize = 4 * 2 * 3 * 4;
const PROFILE: &str = "profile.fixture.gdn.candidates";

pub fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
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
pub struct Config {
    pub rows: u64,
    pub candidates: bool,
    pub f32_hidden: bool,
    pub log_rate: bool,
    pub grouped: bool,
}
impl Config {
    pub fn dtype(self) -> ElementType {
        if self.f32_hidden {
            ElementType::F32
        } else {
            ElementType::F16
        }
    }
    pub fn operation(self) -> &'static str {
        match (self.candidates, self.f32_hidden) {
            (true, false) => GATED_DELTA_CANDIDATE_SCAN_F16_OPERATION_ID,
            (true, true) => GATED_DELTA_CANDIDATE_SCAN_F32_MASTER_OPERATION_ID,
            (false, false) => GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            (false, true) => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
        }
    }
    fn version(self) -> ContractVersion {
        ContractVersion::new(
            if !self.candidates && !self.f32_hidden {
                6
            } else {
                1
            },
            0,
        )
    }
    pub fn inputs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([
            (id(X), tensor(vec![self.rows, H as u64], self.dtype())),
            (id(C), tensor(vec![QKV as u64, 2], ElementType::F16)),
            (id(D), tensor(vec![4, 2, 3], ElementType::F32)),
        ])
    }
    fn attributes(self) -> BTreeMap<AttributeId, SemanticValue> {
        let mut attributes: BTreeMap<_, _> = [
            ("hidden_size", H as u64),
            ("key_heads", 2),
            ("value_heads", 4),
            ("key_head_dim", 3),
            ("value_head_dim", 2),
            ("qkv_features", QKV as u64),
            ("value_features", 8),
            ("qkvz_features", 28),
            ("ba_features", 8),
            ("qkvzba_features", MIXED as u64),
            ("conv_kernel", 3),
            ("conv_state_width", 2),
            ("layer_index", 0),
        ]
        .into_iter()
        .map(|(k, v)| (id(k), SemanticValue::Unsigned(v)))
        .collect();
        attributes.insert(
            id("epsilon"),
            SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
        );
        attributes.insert(
            id("decay_parameterization"),
            SemanticValue::Text(
                if self.log_rate {
                    "log_rate"
                } else {
                    "negative_rate"
                }
                .into(),
            ),
        );
        attributes.insert(
            id("value_head_mapping"),
            SemanticValue::Text(
                if self.grouped {
                    "grouped_by_key_head"
                } else {
                    "interleaved_by_key_head"
                }
                .into(),
            ),
        );
        attributes
    }
}

fn weights() -> Vec<(&'static str, Vec<u64>, ElementType)> {
    vec![
        ("norm", vec![H as u64], ElementType::F16),
        ("projection", vec![MIXED as u64, H as u64], ElementType::F16),
        ("conv", vec![QKV as u64, 3], ElementType::F16),
        ("rate", vec![4], ElementType::F32),
        ("bias", vec![4], ElementType::F32),
        ("gated_norm", vec![2], ElementType::F32),
        ("out", vec![H as u64, 8], ElementType::F16),
    ]
}
fn weight_id(name: &str) -> WeightId {
    id(format!("weight.gdn.{name}"))
}
fn component_id(name: &str) -> WeightId {
    id(format!("component.gdn.{name}"))
}
fn weight_value(name: &str) -> ProgramValueId {
    id(format!("value.gdn.weight.{name}"))
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
        BTreeSet::from([id("metadata.fixture.gdn.candidates")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|e| invalid(format!("{e}")))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if c.rows == 0 || serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("invalid GDN fixture config"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.gdn.candidates"))
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fixture.gdn.candidates"),
            version: ContractVersion::new(1, 0),
            components: weights()
                .into_iter()
                .map(|(name, dimensions, dtype)| WeightComponentSpec {
                    id: component_id(name),
                    role: WeightComponentRole::Values,
                    external_names: vec![name.into()],
                    dimensions,
                    encoding: WeightEncoding::Dense {
                        element_type: dtype,
                    },
                    required: true,
                })
                .collect(),
            tensors: weights()
                .into_iter()
                .map(|(name, dimensions, dtype)| WeightTensorSpec {
                    id: weight_id(name),
                    dimensions,
                    logical_element_type: dtype,
                    physical_layout: PhysicalWeightLayout::Dense {
                        component_id: component_id(name),
                    },
                    required: true,
                })
                .collect(),
        }))
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut boundaries = BTreeMap::from([(id(Y), c.dtype())]);
        if c.candidates {
            boundaries.extend([(id(CC), ElementType::F16), (id(DD), ElementType::F32)]);
        }
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Y),
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(c.operation()),
                    version: c.version(),
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
        let mut inputs = vec![id(X)];
        inputs.extend(weights().iter().map(|(name, _, _)| weight_value(name)));
        inputs.extend([id(C), id(D)]);
        let mut node_outputs = vec![id(Y)];
        if c.candidates {
            node_outputs.extend([id(CC), id(DD)]);
        }
        // A test provider observer retains the original state regions through
        // completion; public terminal ports are genuine operation outputs.
        let mut terminal = vec![id(Y)];
        if c.candidates {
            terminal.extend([id(CC), id(DD)]);
        }
        ModelProgram::new(
            self.id.clone(),
            vec![id(X), id(C), id(D)],
            vec![ProgramBlock {
                id: "block.gdn.candidates".into(),
                nodes: vec![ProgramNode {
                    id: id("node.gdn.candidates"),
                    operation_id: id(c.operation()),
                    required_version: c.version(),
                    work: ProgramNodeWorkSpec::tokens(id(X), 0),
                    inputs,
                    outputs: node_outputs,
                    attributes: c.attributes(),
                }],
            }],
            vec![],
            weights()
                .into_iter()
                .map(|(name, dimensions, dtype)| WeightReference {
                    weight_id: weight_id(name),
                    value_id: weight_value(name),
                    tensor: tensor(dimensions, dtype),
                })
                .collect(),
            terminal,
        )?
        .with_input_tensor_specs(c.inputs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(CausalLanguageMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".into(),
                source_file: "fixture-template.json".into(),
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

fn prepare(c: Config) -> Result<PreparedModelFamily, VNextError> {
    TypedFamilyRegistration::new(Family {
        id: id("family.fixture.gdn.candidates"),
    })
    .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(PROFILE))
}
pub fn compile_with_capacity(
    c: Config,
    composition: &ManagedComposition,
    capacity: u64,
) -> Result<ProgramPlanCompilation, VNextError> {
    let family = prepare(c)?;
    let descriptor = composition.runtime().descriptor();
    let policy = ResolvedRuntimePolicy::new(
        "runtime-policy.fixture.gdn.candidates",
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
            maximum_scheduled_tokens: c.rows,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            sequence_backing_growth: Default::default(),
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntime,
        None,
    )?;
    let mut options = ProgramPlanCompileOptions::new(c.inputs())?;
    for name in if c.candidates {
        &[Y, CC, DD][..]
    } else {
        &[Y][..]
    } {
        options.retain_completion_value(id(*name));
    }
    ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        &WeightMaterializerRegistry::identity_only()?,
        &options,
    )
}
pub fn encode(values: impl IntoIterator<Item = f32>, dtype: ElementType) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        match dtype {
            ElementType::F16 => out.extend(f16::from_f32(value).to_le_bytes()),
            ElementType::F32 => out.extend(value.to_le_bytes()),
            _ => panic!("fixture float"),
        }
    }
    out
}
pub struct Parameters {
    values: BTreeMap<String, Vec<u8>>,
}
impl Parameters {
    pub fn new(c: Config) -> Self {
        Self {
            values: weights()
                .into_iter()
                .enumerate()
                .map(|(ordinal, (name, shape, dtype))| {
                    let values = (0..shape.iter().product::<u64>() as usize).map(|i| match name {
                        "norm" => 0.9 + i as f32 * 0.03125,
                        "gated_norm" => 0.75 + i as f32 * 0.125,
                        "rate" => {
                            if c.log_rate {
                                -0.4 + i as f32 * 0.13
                            } else {
                                -(-0.4 + i as f32 * 0.13).exp()
                            }
                        }
                        "bias" => i as f32 * 0.1 - 0.15,
                        _ => ((i + ordinal * 17) as f32 * 0.37).sin() * 0.2,
                    });
                    (name.into(), encode(values, dtype))
                })
                .collect(),
        }
    }
}
impl WeightComponentSource for Parameters {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let [name] = c.external_names.as_slice() else {
            return Err(invalid("invalid parameter names"));
        };
        let bytes = self
            .values
            .get(name)
            .ok_or_else(|| invalid("unknown parameter"))?;
        let WeightEncoding::Dense { element_type } = c.encoding else {
            return Err(invalid("dense parameters only"));
        };
        WeightComponentPayload::new(
            c,
            name.clone(),
            "gdn.typed_fixture",
            c.dimensions.clone(),
            element_type,
            bytes,
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
pub fn session(
    c: Config,
    composition: &ManagedComposition,
    host: &HostMemoryAccount,
) -> TokenSession {
    session_with_provisioning(c, composition, host, false)
}

/// Qualify variable-row arithmetic in one plan, independently of elastic
/// fragmentation recovery. The compiled ceiling is reserved through ordinary
/// Core growth before any minimum chunk exists; no capacity limit is changed.
pub fn session_preprovisioned(
    c: Config,
    composition: &ManagedComposition,
    host: &HostMemoryAccount,
) -> TokenSession {
    session_with_provisioning(c, composition, host, true)
}

fn session_with_provisioning(
    c: Config,
    composition: &ManagedComposition,
    host: &HostMemoryAccount,
    preprovisioned: bool,
) -> TokenSession {
    let family = prepare(c).unwrap();
    let compilation = compile(c, composition).unwrap();
    let executable = compilation.executable();
    let plan = executable.execution_plan();
    eprintln!(
        "gdn_plan {}",
        serde_json::json!({
            "config": c,
            "preprovisioned": preprovisioned,
            "dynamic_pools": plan.payload().memory().dynamic_pools(),
            "dynamic_descriptors": plan.payload().memory().dynamic_descriptors(),
        })
    );
    assert!(plan
        .payload()
        .execution_weights()
        .static_weight_transforms()
        .is_empty());
    let providers = composition.registry().bind_plan(executable).unwrap();
    let permit = match plan
        .provision_static(composition.runtime().clone(), id("request.gdn.init"))
        .unwrap()
        .into_provisioning()
    {
        StaticProvisioning::Required(permit) => permit,
        _ => panic!("GDN fixture has real Stored weights"),
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.gdn.init"),
        id("transaction.gdn.init"),
    );
    let driver = RuntimeResourceDriver::new(composition.runtime().clone()).unwrap();
    let reserved = ResourceTransaction::begin(driver, identity, permit)
        .unwrap()
        .reserve()
        .unwrap();
    let committed = match reserved.commit() {
        Ok(value) => value,
        Err(ResourceCommitTransitionError::Recoverable(error)) => {
            panic!("commit: {:?}", error.failure())
        }
        Err(ResourceCommitTransitionError::Poisoned(error)) => {
            panic!("unknown commit: {:?}", error.failure())
        }
    };
    if preprovisioned {
        let requests = plan
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
        // This is a fallible global reservation plus all-or-nothing allocation
        // and publication, not a proof that later elastic growth will succeed.
        committed
            .maintenance_controller()
            .grow_pools(requests)
            .unwrap();
    } else {
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
    }
    let initialized = committed
        .initialize_static_managed(
            &family,
            plan,
            &Parameters::new(c),
            host,
            StaticInitializationPolicy::new(4096, 16).unwrap(),
        )
        .unwrap();
    let resources = initialized
        .into_plan_runtime()
        .unwrap_or_else(|error| panic!("runtime handoff: {:?}", error.failure()));
    let lane = resources.create_execution_lane().unwrap();
    TokenSession {
        compilation,
        providers,
        resources,
        lane,
        reaper: CompletionReaper::new(),
    }
}

// Real token admission, with bounded backing maintenance; no fabricated permits.
fn admit_sequence(
    resources: &Arc<PlanRuntimeResources<CudaDeviceRuntime>>,
    work: &ResourceWorkShape,
) -> Arc<AdmittedSequenceResources<CudaDeviceRuntime>> {
    let request = RequestResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let binding = resources.trusted_runtime_binding().unwrap();
    let request_resources = loop {
        match binding
            .try_admit_request(request.clone(), id("run.gdn.step"), id("request.gdn.step"))
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
                panic!("GDN request rejected: {reason:?}");
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
                panic!("GDN sequence rejected: {reason:?}");
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
        "GDN admission has no owner capable of releasing more capacity: {outcome:?}"
    );
}

pub fn run(
    fixture: &mut TokenSession,
    host: &HostMemoryAccount,
    c: Config,
    x: &[u8],
    conv: &[u8],
    delta: &[u8],
) -> Outputs {
    // Tokens supply actual row work; the program's numerical inputs are the
    // separately charged hidden/conv/recurrent bytes, never token IDs as floats.
    let tokens = vec![1_u32; c.rows as usize];
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
                    "GDN step backing blocked: config={c:?}, evidence={:?}, pools={:?}, outcome={outcome:?}",
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
                    "GDN step blocked: config={c:?}, deferred={reason:?}, pools={:?}",
                    fixture.resources.dynamic_pool_status().unwrap()
                );
                require_progress(outcome);
            }
            StepResourceAdmissionDecision::PermanentRejected(reason) => {
                panic!("GDN step rejected: {reason:?}");
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
                panic!("GDN wave rejected: {reason:?}");
            }
            other => panic!(
                "GDN wave has a request-state hazard: {:?}",
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
    let uploads = [
        (0, c.dtype(), x),
        (8, ElementType::F16, conv),
        (9, ElementType::F32, delta),
    ]
    .into_iter()
    .map(|(ordinal, dtype, bytes)| {
        let mut allocation = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .unwrap();
        allocation.as_mut().copy_from_slice(bytes);
        SubmissionWaveInputUpload::managed_range(
            id("node.gdn.candidates"),
            0,
            ordinal,
            0,
            HostTransferLayout::new(dtype, bytes.len() as u64 / dtype.size_bytes()).unwrap(),
            allocation.freeze(),
        )
        .unwrap()
    })
    .collect::<Vec<_>>();
    let names = if c.candidates {
        &[Y, CC, DD][..]
    } else {
        &[Y][..]
    };
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
        panic!("GDN completion not terminal")
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
pub fn close(session: TokenSession) {
    let TokenSession {
        resources,
        lane,
        reaper,
        providers,
        compilation,
    } = session;
    drop((lane, reaper, providers, compilation));
    assert!(matches!(
        PlanRuntimeResources::close(resources)
            .unwrap_or_else(|e| panic!("GDN close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
}
