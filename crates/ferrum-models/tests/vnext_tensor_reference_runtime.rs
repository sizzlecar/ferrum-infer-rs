//! Real F16/BF16 safetensors through the tensor-component runtime path.
//! Tiny independent arithmetic fixtures do not establish image-model parity.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

use ferrum_execution::{PreparedTensorExecutor, TensorExecutionError, TensorExecutorConfig};
use ferrum_kernels::backend::reference::{
    ReferenceDeviceRuntime, ReferenceVNextComposition, REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use half::f16;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const INPUT_ID: &str = "value.tensor.input";
const OUTPUT_ID: &str = "value.tensor.output";
const PROFILE_ID: &str = "reference.tensor.f16";
const INPUT_VALUES: [f32; 6] = [1.0, 2.0, 3.0, -1.0, 0.0, 2.0];
const WEIGHT_VALUES: [f32; 6] = [2.0, -1.0, 1.0, 1.0, 3.0, -2.0];
// Independently calculated X * W^T, rather than calling a second matmul.
const EXPECTED_VALUES: [f32; 4] = [3.0, 1.0, 0.0, -5.0];

#[path = "vnext_tensor_reference_runtime/tensor_faults.rs"]
mod tensor_faults;

#[path = "vnext_tensor_reference_runtime/bf16.rs"]
mod bf16;

#[path = "vnext_tensor_reference_runtime/layer_norm.rs"]
mod layer_norm;

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TinyTensorConfig {
    rows: u64,
    in_features: u64,
    out_features: u64,
}

#[derive(Clone, Copy)]
enum TinyTensorPrecision {
    F16,
    Bf16,
}

impl TinyTensorPrecision {
    fn element_type(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::Bf16 => ElementType::Bf16,
        }
    }

    fn operation_id(self) -> &'static str {
        match self {
            Self::F16 => DENSE_LINEAR_OPERATION_ID,
            Self::Bf16 => DENSE_LINEAR_BF16_OPERATION_ID,
        }
    }

    fn profile_id(self) -> &'static str {
        match self {
            Self::F16 => PROFILE_ID,
            Self::Bf16 => "reference.tensor.bf16",
        }
    }
}

struct TinyTensorFamily(TinyTensorPrecision);

impl ModelFamilyProvider for TinyTensorFamily {
    type Config = TinyTensorConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static F16: OnceLock<ModelFamilyId> = OnceLock::new();
        static BF16: OnceLock<ModelFamilyId> = OnceLock::new();
        match self.0 {
            TinyTensorPrecision::F16 => F16.get_or_init(|| id("family.reference.tensor-dense")),
            TinyTensorPrecision::Bf16 => {
                BF16.get_or_init(|| id("family.reference.tensor-dense.bf16"))
            }
        }
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.reference.tensor-dense")])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        if (config.rows, config.in_features, config.out_features) != (2, 3, 2)
            || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".to_owned(),
                reason: "tensor fixture requires its exact rectangular matrix dimensions"
                    .to_owned(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.reference.tensor-dense"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID),
            layout_id: id("weight-layout.reference.tensor-dense"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("weight.component.tensor.linear"),
                role: WeightComponentRole::Values,
                external_names: vec!["linear.weight".to_owned()],
                dimensions: vec![config.out_features, config.in_features],
                encoding: WeightEncoding::Dense {
                    element_type: self.0.element_type(),
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.tensor.linear"),
                dimensions: vec![config.out_features, config.in_features],
                logical_element_type: self.0.element_type(),
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("weight.component.tensor.linear"),
                },
                required: true,
            }],
        })
    }

    fn numerical_profiles(&self, _: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new(self.0.profile_id()).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(OUTPUT_ID),
                boundaries: BTreeMap::from([(id(OUTPUT_ID), self.0.element_type())]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(self.0.operation_id()),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile_id],
        )
    }

    fn semantic_program(
        &self,
        config: &Self::Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT_ID)],
            vec![ProgramBlock {
                id: "block.tensor.dense".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.tensor.dense"),
                    operation_id: id(self.0.operation_id()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT_ID), id("value.tensor.weight")],
                    outputs: vec![id(OUTPUT_ID)],
                    attributes: BTreeMap::from([
                        (
                            id("in_features"),
                            SemanticValue::Unsigned(config.in_features),
                        ),
                        (
                            id("out_features"),
                            SemanticValue::Unsigned(config.out_features),
                        ),
                    ]),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.tensor.linear"),
                value_id: id("value.tensor.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.out_features, config.in_features],
                    element_type: self.0.element_type(),
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }],
            vec![id(OUTPUT_ID)],
        )
    }

    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

/// A tensor policy carries no text scheduling ceiling or placeholder token.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorPolicy {
    capacity_bytes: u64,
    storage_profiles: Vec<DynamicStorageProfile>,
}

impl RuntimePolicy for TensorPolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    fn memory_reserve_bytes(&self) -> u64 {
        0
    }
    fn maximum_active_sequences(&self) -> u32 {
        1
    }
    fn work_limits(&self) -> RuntimeWorkLimits {
        RuntimeWorkLimits::FixedTensor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement {
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    }
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.storage_profiles
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.capacity_bytes == 0 || self.storage_profiles.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "tensor fixture requires positive capacity and a storage profile"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

fn host_account(total_bytes: usize, output_bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            total_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: total_bytes,
                transfer_staging_bytes: total_bytes,
                output_bytes,
                weight_backing_bytes: total_bytes,
            },
        )
        .unwrap(),
    )
}

fn managed_input(account: &HostMemoryAccount) -> ManagedHostBytes {
    let mut input = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 12)
        .unwrap();
    input.as_mut().copy_from_slice(&f16_bytes(&INPUT_VALUES));
    input.freeze()
}

fn input_spec(dimensions: Vec<u64>, element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dimensions, element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

fn typed_input(bytes: ManagedHostBytes) -> ManagedHostTensor {
    ManagedHostTensor::new(
        id(INPUT_ID),
        input_spec(vec![2, 3], ElementType::F16),
        bytes,
    )
    .unwrap()
}

fn admission_request(work: &ResourceWorkShape) -> RequestResourceAdmissionRequest {
    RequestResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
}

fn admission_sequence(work: &ResourceWorkShape) -> SequenceResourceAdmissionRequest {
    SequenceResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
}

fn admit_initial_sequence(
    binding: &TensorExecutionBinding<ReferenceDeviceRuntime>,
    work: &ResourceWorkShape,
) -> Arc<AdmittedSequenceResources<ReferenceDeviceRuntime>> {
    // This fixture has one request input, one completion output, and no state.
    // Bound retries diagnose a broken backing transition rather than hanging.
    const MAX_BACKING_GROWTH_ATTEMPTS: usize = 3;
    for attempt in 0..=MAX_BACKING_GROWTH_ATTEMPTS {
        match binding
            .runtime_binding()
            .try_admit_initial_sequence(
                admission_request(work),
                admission_sequence(work),
                id("run.tensor.initial"),
                id("request.tensor.initial"),
            )
            .unwrap()
        {
            InitialSequenceResourceAdmissionDecision::Admitted(resources) => return resources,
            InitialSequenceResourceAdmissionDecision::BackingDeferred(deferred)
                if attempt < MAX_BACKING_GROWTH_ATTEMPTS =>
            {
                deferred.maintain().unwrap();
            }
            _ => panic!("tensor fixture initial admission did not converge"),
        }
    }
    unreachable!("the final failed admission attempt reports its failure")
}

struct TensorFixture {
    composition: ReferenceVNextComposition,
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<ReferenceDeviceRuntime>,
    resources: Arc<PlanRuntimeResources<ReferenceDeviceRuntime>>,
    host: HostMemoryAccount,
    // Keep the real tiny source alive through static initialization and close.
    _source_directory: tempfile::TempDir,
}

fn source_and_family() -> (tempfile::TempDir, SafetensorsArchive, PreparedModelFamily) {
    source_and_family_with_precision(TinyTensorPrecision::F16, &f16_bytes(&WEIGHT_VALUES))
}

fn source_and_family_with_precision(
    precision: TinyTensorPrecision,
    weight_bytes: &[u8],
) -> (tempfile::TempDir, SafetensorsArchive, PreparedModelFamily) {
    let source_directory = tempfile::tempdir().unwrap();
    let dtype = match precision {
        TinyTensorPrecision::F16 => Dtype::F16,
        TinyTensorPrecision::Bf16 => Dtype::BF16,
    };
    let tensor = TensorView::new(dtype, vec![2, 3], weight_bytes).unwrap();
    serialize_to_file(
        vec![("linear.weight".to_owned(), tensor)],
        &None::<HashMap<String, String>>,
        &source_directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = SafetensorsArchive::open(source_directory.path()).unwrap();
    let family = TypedFamilyRegistration::new(TinyTensorFamily(precision))
        .prepare_with_profile(
            &json!({"rows":2,"in_features":3,"out_features":2}),
            &NumericalProfileId::new(precision.profile_id()).unwrap(),
        )
        .unwrap();
    assert_eq!(family.metadata(), &ModelSemanticMetadata::TensorComponent);
    (source_directory, source, family)
}

fn build_fixture(host: HostMemoryAccount) -> TensorFixture {
    build_fixture_with_precision(host, TinyTensorPrecision::F16, &f16_bytes(&WEIGHT_VALUES))
}

fn build_fixture_with_precision(
    host: HostMemoryAccount,
    precision: TinyTensorPrecision,
    weight_bytes: &[u8],
) -> TensorFixture {
    let (source_directory, source, family) =
        source_and_family_with_precision(precision, weight_bytes);
    let composition = ReferenceVNextComposition::create(id("device.reference.tensor")).unwrap();
    let policy = TensorPolicy {
        capacity_bytes: composition.runtime().descriptor().total_memory_bytes,
        storage_profiles: composition
            .runtime()
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(INPUT_ID),
        ProgramTensorSpec {
            dimensions: vec![2, 3],
            element_type: precision.element_type(),
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    options.require_weight_materializer(composition.weight_materializer_id().clone());
    assert!(options.retain_completion_value(id(OUTPUT_ID)));
    let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    let executable = compilation.executable();
    let plan = executable.execution_plan();
    let providers = composition.registry().bind_plan(executable).unwrap();
    let resources = initialize_resources(Arc::clone(composition.runtime()), &family, plan, &source);
    TensorFixture {
        composition,
        compilation,
        providers,
        resources,
        host,
        _source_directory: source_directory,
    }
}

fn initialize_resources<R: DeviceRuntime>(
    runtime: Arc<R>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: &SafetensorsArchive,
) -> Arc<PlanRuntimeResources<R>> {
    initialize_resources_with_weight_bytes(runtime, family, plan, source, 12)
}

fn initialize_resources_with_weight_bytes<R: DeviceRuntime>(
    runtime: Arc<R>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: &SafetensorsArchive,
    expected_weight_bytes: u64,
) -> Arc<PlanRuntimeResources<R>> {
    let provisioned = plan
        .provision_static(runtime.clone(), id("request.tensor.provision"))
        .unwrap();
    let permit = match provisioned.into_provisioning() {
        StaticProvisioning::Required(permit) => permit,
        StaticProvisioning::NoStatic(_) => panic!("real dense weights require static provisioning"),
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.tensor.provision"),
        id("transaction.tensor.provision"),
    );
    let driver = RuntimeResourceDriver::new(runtime).unwrap();
    let reserved = ResourceTransaction::begin(driver, identity, permit)
        .unwrap()
        .reserve()
        .unwrap();
    let committed = match reserved.commit() {
        Ok(committed) => committed,
        Err(ResourceCommitTransitionError::Recoverable(error)) => {
            panic!("static commit failed: {:?}", error.failure())
        }
        Err(ResourceCommitTransitionError::Poisoned(error)) => {
            panic!("static commit indeterminate: {:?}", error.failure())
        }
    };
    let pool_ids = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool_id in pool_ids {
        committed
            .maintenance_controller()
            .initialize_pool(&pool_id)
            .unwrap();
    }
    let initialized = committed
        .initialize_static(
            family,
            plan,
            source,
            StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(
        initialized.receipt().uploaded_bytes(),
        expected_weight_bytes
    );
    match initialized.into_plan_runtime() {
        Ok(resources) => resources,
        Err(error) => panic!(
            "tensor fixture runtime handoff failed: {}",
            error.failure().message()
        ),
    }
}

fn close_fixture(fixture: TensorFixture) {
    let TensorFixture {
        composition,
        compilation,
        providers,
        resources,
        host: _host,
        _source_directory,
    } = fixture;
    drop((providers, compilation));
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), 1)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("tensor fixture retains {strong_count} plan roots")
        }
        Err(error) => panic!("tensor fixture close failed: {:?}", error.failure()),
    }
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}

fn executor(fixture: &TensorFixture) -> PreparedTensorExecutor<ReferenceDeviceRuntime> {
    let executable = fixture.compilation.executable().clone();
    let providers = fixture
        .composition
        .registry()
        .bind_plan(&executable)
        .unwrap();
    PreparedTensorExecutor::new(
        executable,
        providers,
        Arc::clone(&fixture.resources),
        fixture.host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap()
}

#[test]
fn tensor_family_prepares_real_f16_weights_without_text_metadata_or_work() {
    let fixture = build_fixture(host_account(32, 8));
    assert_eq!(
        fixture
            .composition
            .runtime()
            .snapshot()
            .dense_linear_launches,
        0
    );
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    close_fixture(fixture);
}

#[test]
fn same_numel_shape_and_dtype_mismatches_fail_before_device_work() {
    let fixture = build_fixture(host_account(32, 8));
    let binding = fixture
        .resources
        .bind_tensor_execution(
            fixture.compilation.executable().execution_plan(),
            fixture.host.clone(),
        )
        .unwrap();
    let bytes = managed_input(&fixture.host);
    let before = fixture.composition.runtime().snapshot();

    // Both declarations have exactly twelve bytes, so a byte-count-only check
    // would admit them. BF16 here is invalid metadata, not a numerical test.
    for spec in [
        input_spec(vec![3, 2], ElementType::F16),
        input_spec(vec![2, 3], ElementType::Bf16),
    ] {
        let input = ManagedHostTensor::new(id(INPUT_ID), spec, bytes.clone()).unwrap();
        assert!(binding.bind_inputs(vec![input]).is_err());
        assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 12);
    }
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.dense_linear_launches, 0);
    drop((binding, bytes));
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    close_fixture(fixture);
}

#[test]
fn foreign_host_account_with_identical_limits_cannot_supply_inputs() {
    let fixture = build_fixture(host_account(32, 8));
    let foreign = host_account(32, 8);
    assert_eq!(foreign.limits(), fixture.host.limits());
    let bytes = managed_input(&foreign);
    let binding = fixture
        .resources
        .bind_tensor_execution(
            fixture.compilation.executable().execution_plan(),
            fixture.host.clone(),
        )
        .unwrap();
    let before = fixture.composition.runtime().snapshot();

    assert!(binding
        .bind_inputs(vec![typed_input(bytes.clone())])
        .is_err());
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 12);
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.dense_linear_launches, 0);
    drop((binding, bytes));
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    close_fixture(fixture);
}

#[test]
fn tensor_input_set_requires_exact_ports_without_duplicates() {
    let fixture = build_fixture(host_account(32, 8));
    let binding = fixture
        .resources
        .bind_tensor_execution(
            fixture.compilation.executable().execution_plan(),
            fixture.host.clone(),
        )
        .unwrap();
    let bytes = managed_input(&fixture.host);
    let input = typed_input(bytes.clone());
    let foreign_port = ManagedHostTensor::new(
        id("value.tensor.undeclared"),
        input_spec(vec![2, 3], ElementType::F16),
        bytes.clone(),
    )
    .unwrap();
    let before = fixture.composition.runtime().snapshot();

    assert!(binding.bind_inputs(vec![]).is_err());
    assert!(binding
        .bind_inputs(vec![input.clone(), input.clone()])
        .is_err());
    assert!(binding.bind_inputs(vec![foreign_port.clone()]).is_err());
    assert!(binding
        .bind_inputs(vec![input.clone(), foreign_port.clone()])
        .is_err());
    // A valid set produces owned evidence, with no device admission or launch.
    let work = binding.bind_inputs(vec![input.clone()]).unwrap();
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.dense_linear_launches, 0);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 12);
    drop((binding, input, foreign_port, bytes));
    // The evidence retains its source allocation until it is retired.
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 12);
    drop(work);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    close_fixture(fixture);
}

#[test]
fn tensor_evidence_cannot_cross_distinct_roots_with_the_same_compiled_plan() {
    let host = host_account(32, 8);
    let first = build_fixture(host.clone());
    let second = build_fixture(host.clone());
    let first_plan = first.compilation.executable().execution_plan();
    let second_plan = second.compilation.executable().execution_plan();
    assert_eq!(first_plan.plan_hash(), second_plan.plan_hash());
    let first_binding = first
        .resources
        .bind_tensor_execution(first_plan, host.clone())
        .unwrap();
    let second_binding = second
        .resources
        .bind_tensor_execution(second_plan, host.clone())
        .unwrap();
    let work = first_binding
        .bind_inputs(vec![typed_input(managed_input(&host))])
        .unwrap();
    let before = second.composition.runtime().snapshot();

    assert!(second_binding
        .runtime_binding()
        .try_admit_initial_sequence(
            admission_request(&work),
            admission_sequence(&work),
            id("run.tensor.foreign-root"),
            id("request.tensor.foreign-root"),
        )
        .is_err());
    assert!(second_binding
        .runtime_binding()
        .try_admit_request(
            admission_request(&work),
            id("run.tensor.foreign-root-request"),
            id("request.tensor.foreign-root-request"),
        )
        .is_err());
    let after = second.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    drop((work, first_binding, second_binding));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    close_fixture(first);
    close_fixture(second);
}

#[test]
fn rebinding_inputs_cannot_replace_the_admitted_work_identity() {
    let fixture = build_fixture(host_account(48, 8));
    let binding = fixture
        .resources
        .bind_tensor_execution(
            fixture.compilation.executable().execution_plan(),
            fixture.host.clone(),
        )
        .unwrap();
    let bytes = managed_input(&fixture.host);
    let original = binding
        .bind_inputs(vec![typed_input(bytes.clone())])
        .unwrap();
    let same_bytes = binding
        .bind_inputs(vec![typed_input(bytes.clone())])
        .unwrap();
    let mut changed = fixture
        .host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 12)
        .unwrap();
    changed.as_mut().copy_from_slice(&f16_bytes(&[0.0; 6]));
    let different_bytes = binding
        .bind_inputs(vec![typed_input(changed.freeze())])
        .unwrap();
    assert_eq!(original, original.clone());
    let before = fixture.composition.runtime().snapshot();

    for replacement in [&same_bytes, &different_bytes] {
        assert_ne!(original, *replacement);
        assert!(binding
            .runtime_binding()
            .try_admit_initial_sequence(
                admission_request(&original),
                admission_sequence(replacement),
                id("run.tensor.rebound"),
                id("request.tensor.rebound"),
            )
            .is_err());
    }
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    let sequence = admit_initial_sequence(&binding, &original);
    let session = sequence.open_session().unwrap();
    let participants = ExecutionBatchParticipants::new(vec![session.clone()]).unwrap();
    let admitted_batch = participants.bind_tensor_work_shape(&original).unwrap();
    assert!(admitted_batch.token_view().is_err());
    assert!(participants.bind_tensor_work_shape(&same_bytes).is_err());
    assert!(participants
        .bind_tensor_work_shape(&different_bytes)
        .is_err());
    drop((admitted_batch, participants, session, sequence));
    drop((binding, original, same_bytes, different_bytes, bytes));
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    close_fixture(fixture);
}

#[test]
fn closing_root_keeps_old_evidence_alive_but_rejects_new_admission() {
    let fixture = build_fixture(host_account(32, 8));
    let binding = fixture
        .resources
        .bind_tensor_execution(
            fixture.compilation.executable().execution_plan(),
            fixture.host.clone(),
        )
        .unwrap();
    let bytes = managed_input(&fixture.host);
    let work = binding
        .bind_inputs(vec![typed_input(bytes.clone())])
        .unwrap();
    let before = fixture.composition.runtime().snapshot();
    match PlanRuntimeResources::close(Arc::clone(&fixture.resources)) {
        Ok(PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            ..
        }) => {
            assert!(strong_count > 1);
            assert!(Arc::ptr_eq(&resources, &fixture.resources));
            assert!(resources.is_closing());
        }
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("live tensor evidence must retain its root")
        }
        Err(error) => panic!("close initiation failed: {:?}", error.failure()),
    }

    assert!(binding
        .bind_inputs(vec![typed_input(bytes.clone())])
        .is_err());
    assert!(binding
        .runtime_binding()
        .try_admit_initial_sequence(
            admission_request(&work),
            admission_sequence(&work),
            id("run.tensor.closing"),
            id("request.tensor.closing"),
        )
        .is_err());
    assert!(fixture.resources.trusted_runtime_binding().is_err());
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.live_allocations, before.live_allocations);
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 12);
    drop((binding, work, bytes));
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    close_fixture(fixture);
}

#[test]
fn input_slice_retains_the_entire_managed_backing_charge() {
    let fixture = build_fixture(host_account(32, 8));
    let binding = fixture
        .resources
        .bind_tensor_execution(
            fixture.compilation.executable().execution_plan(),
            fixture.host.clone(),
        )
        .unwrap();
    let mut backing = fixture
        .host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 24)
        .unwrap();
    backing.as_mut()[6..18].copy_from_slice(&f16_bytes(&INPUT_VALUES));
    let backing = backing.freeze();
    let view = backing.slice(6..18).unwrap();
    let work = binding.bind_inputs(vec![typed_input(view)]).unwrap();
    drop(backing);

    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 24);
    assert_eq!(
        work.tensor_evidence().unwrap().inputs()[0].bytes().bytes(),
        f16_bytes(&INPUT_VALUES)
    );
    drop(work);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    drop(binding);
    close_fixture(fixture);
}

#[test]
fn tensor_executor_computes_f16_and_output_owner_outlives_the_plan() {
    // Exact input/output backing boundary: twelve input bytes plus eight output
    // bytes. No allowance for silently duplicating either managed allocation.
    let fixture = build_fixture(host_account(20, 8));
    let host = fixture.host.clone();
    let mut executor = executor(&fixture);
    let input = managed_input(&host);
    let outputs = executor
        .execute(
            id("run.tensor.execute"),
            id("request.tensor.execute"),
            vec![typed_input(input.clone())],
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].value_id(), &id::<ProgramValueId>(OUTPUT_ID));
    assert_eq!(outputs[0].tensor().dimensions(), [2, 2]);
    assert_eq!(outputs[0].tensor().element_type(), ElementType::F16);
    assert_eq!(outputs[0].bytes().bytes(), f16_bytes(&EXPECTED_VALUES));
    assert!(host.owns(outputs[0].bytes()));
    assert_eq!(host.usage().unwrap().committed_bytes(), 20);
    let snapshot = fixture.composition.runtime().snapshot();
    assert_eq!(snapshot.dense_linear_launches, 1);
    assert_eq!(snapshot.uploaded_bytes, 24);
    assert_eq!(snapshot.readback_bytes, 8);

    let output_owner = outputs[0].bytes().clone();
    let output_view = output_owner.slice(2..6).unwrap();
    drop((input, outputs, executor));
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    close_fixture(fixture);
    assert_eq!(output_owner.bytes(), f16_bytes(&EXPECTED_VALUES));
    assert_eq!(output_view.bytes(), f16_bytes(&[1.0, 0.0]));
    drop(output_owner);
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    drop(output_view);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn output_total_and_class_capacity_fail_before_device_admission() {
    for (host, expected_scope, expected_committed, expected_limit) in [
        (host_account(19, 8), HostMemoryPressureScope::Total, 12, 19),
        (
            host_account(32, 7),
            HostMemoryPressureScope::Class(HostMemoryClass::Output),
            0,
            7,
        ),
    ] {
        let fixture = build_fixture(host.clone());
        let mut executor = executor(&fixture);
        let input = managed_input(&host);
        let before = fixture.composition.runtime().snapshot();
        let result = executor.execute(
            id("run.tensor.output-capacity"),
            id("request.tensor.output-capacity"),
            vec![typed_input(input.clone())],
        );
        assert!(
            matches!(
                &result,
                Err(TensorExecutionError::Host(HostMemoryError::Capacity {
                    scope,
                    requested: 8,
                    committed,
                    limit,
                })) if *scope == expected_scope
                    && *committed == expected_committed
                    && *limit == expected_limit
            ),
            "expected exact host output capacity failure, received {result:?}"
        );
        let after = fixture.composition.runtime().snapshot();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
        assert_eq!(after.dense_linear_launches, 0);
        assert_eq!(after.readback_bytes, 0);
        drop(result);
        assert_eq!(host.usage().unwrap().committed_bytes(), 12);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output),
            0
        );
        drop((input, executor));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        close_fixture(fixture);
    }
}
