//! Real weightless SiLU through the managed tensor path. No checkpoint, source,
//! materializer, or static-initialization receipt participates in successful work.

use super::*;
use ferrum_execution::{TensorAdmissionFailure, TensorAdmissionStage};
use ferrum_kernels::backend::reference::{
    reference_vnext_silu_bf16_provider, ReferenceDeviceBuffer, ReferenceDeviceCommand,
};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};

// Reuse the existing observation-only proxy, including its real allocations,
// uploads and computation. This is a second module instance, not copied code.
#[path = "fault_runtime.rs"]
mod fault_runtime;
use fault_runtime::{FaultRuntime, FenceMode};

#[path = "fixed_liveness.rs"]
mod fixed_liveness;

const PROFILE: &str = "reference.tensor.weightless.silu.bf16";
const BYTE_LEN: usize = 24;
const INPUT_BITS: [u16; 12] = [
    0x0000, 0x8000, 0x3f80, 0xc2b4, 0x0001, 0x007f, 0x8001, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0x42b4,
];
// Prepared independently before candidate execution: signed zero, SiLU(1),
// expf overflow at -90, tiny exp=1 followed by FP32 division and BF16 RNE,
// infinities/NaN, and saturation at +90. Full primitive qualification is in
// ferrum-kernels; these selected singleton cases do not claim Torch parity.
const EXPECTED_BITS: [u16; 12] = [
    0x0000, 0x8000, 0x3f3b, 0x8000, 0x0000, 0x0040, 0x8000, 0x8040, 0x7f80, 0x7fc0, 0x7fc0, 0x42b4,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
    width: u64,
}

impl Config {
    fn standard() -> Self {
        Self { rows: 2, width: 6 }
    }

    fn spec(&self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![self.rows, self.width],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
}

struct Family;

impl ModelFamilyProvider for Family {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.reference.tensor.weightless.silu"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.reference.tensor.weightless.silu")])
    }

    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows == 0
            || config.width == 0
            || config.rows.checked_mul(config.width) != Some(INPUT_BITS.len() as u64)
            || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "weightless fixture requires a canonical twelve-element matrix".into(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.reference.tensor.weightless.silu"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }

    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(OUTPUT_ID),
                boundaries: BTreeMap::from([(id(OUTPUT_ID), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(SILU_BF16_OPERATION_ID),
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
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT_ID)],
            vec![ProgramBlock {
                id: "block.weightless.silu".into(),
                nodes: vec![ProgramNode {
                    id: id("node.weightless.silu"),
                    operation_id: id(SILU_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT_ID)],
                    outputs: vec![id(OUTPUT_ID)],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT_ID)],
        )
    }

    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn family(config: &Config) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Family)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap()
}

fn options(config: &Config) -> ProgramPlanCompileOptions {
    let mut options =
        ProgramPlanCompileOptions::new(BTreeMap::from([(id(INPUT_ID), config.spec())])).unwrap();
    options.retain_completion_value(id(OUTPUT_ID));
    options
}

fn policy<R: DeviceRuntime>(runtime: &R) -> TensorPolicy {
    TensorPolicy {
        capacity_bytes: runtime.descriptor().total_memory_bytes,
        storage_profiles: runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    }
}

fn account(total: usize, output: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: BYTE_LEN,
                transfer_staging_bytes: 0,
                output_bytes: output,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn input(host: &HostMemoryAccount, config: &Config) -> ManagedHostTensor {
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, BYTE_LEN)
        .unwrap();
    for (destination, bits) in allocation.as_mut().chunks_exact_mut(2).zip(INPUT_BITS) {
        destination.copy_from_slice(&bits.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(INPUT_ID),
        input_spec(vec![config.rows, config.width], ElementType::Bf16),
        allocation.freeze(),
    )
    .unwrap()
}

fn check_output(bytes: &[u8]) {
    assert_eq!(bytes.len(), BYTE_LEN);
    for (index, (bytes, expected)) in bytes.chunks_exact(2).zip(EXPECTED_BITS).enumerate() {
        let actual = u16::from_le_bytes([bytes[0], bytes[1]]);
        if expected & 0x7f80 == 0x7f80 && expected & 0x007f != 0 {
            assert!(
                actual & 0x7f80 == 0x7f80 && actual & 0x007f != 0,
                "index {index}"
            );
        } else {
            assert_eq!(actual, expected, "index {index}");
        }
    }
}

fn no_static<R: DeviceRuntime>(
    runtime: Arc<R>,
    plan: &ExecutionPlan,
) -> Arc<PlanRuntimeResources<R>> {
    match plan
        .provision_static(runtime, id("request.weightless.provision"))
        .unwrap()
        .into_provisioning()
    {
        StaticProvisioning::NoStatic(provisioned) => provisioned.into_plan_runtime(),
        StaticProvisioning::Required(_) => panic!("weightless SiLU requires no static allocation"),
    }
}

fn close_resources<R: DeviceRuntime>(resources: Arc<PlanRuntimeResources<R>>) {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), 0)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("weightless root retains {strong_count} owners")
        }
        Err(error) => panic!("weightless close failed: {:?}", error.failure()),
    }
}

struct Fixture {
    config: Config,
    family: PreparedModelFamily,
    composition: ReferenceVNextComposition,
    policy: TensorPolicy,
    compilation: ProgramPlanCompilation,
    resources: Arc<PlanRuntimeResources<ReferenceDeviceRuntime>>,
    host: HostMemoryAccount,
}

impl Fixture {
    fn new(config: Config, host: HostMemoryAccount) -> Self {
        let family = family(&config);
        let composition =
            ReferenceVNextComposition::create(id("device.reference.weightless.silu")).unwrap();
        let policy = policy(composition.runtime().as_ref());
        // The default compile entrypoint must derive absence itself. No call to
        // require_weight_materializer, no source and no dummy weight schema.
        let compilation = ProgramPlanCompiler::compile(
            &family,
            composition.catalog(),
            &policy,
            &composition.registry().planning(),
            &options(&config),
        )
        .unwrap();
        let resources = no_static(
            composition.runtime().clone(),
            compilation.executable().execution_plan(),
        );
        Self {
            config,
            family,
            composition,
            policy,
            compilation,
            resources,
            host,
        }
    }

    fn plan(&self) -> &ExecutionPlan {
        self.compilation.executable().execution_plan()
    }

    fn executor(&self) -> PreparedTensorExecutor<ReferenceDeviceRuntime> {
        PreparedTensorExecutor::new(
            self.compilation.executable().clone(),
            self.composition
                .registry()
                .bind_plan(self.compilation.executable())
                .unwrap(),
            self.resources.clone(),
            self.host.clone(),
            TensorExecutorConfig::default(),
        )
        .unwrap()
    }

    fn rehydrate(
        &self,
        bytes: &[u8],
        family: &PreparedModelFamily,
    ) -> Result<ExecutionPlan, VNextError> {
        ExecutionPlan::from_json_validated_with_completion_retention(
            bytes,
            family,
            self.composition.catalog(),
            &self.policy,
            self.compilation.node_resolutions().to_vec(),
            self.compilation.completion_retention().clone(),
        )
    }

    fn close(self) {
        let Self {
            composition,
            compilation,
            resources,
            ..
        } = self;
        drop(compilation);
        close_resources(resources);
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    }
}

#[test]
fn weightless_silu_has_no_stored_contract_and_output_outlives_the_root() {
    let fixture = Fixture::new(Config::standard(), account(BYTE_LEN * 2, BYTE_LEN));
    let payload = fixture.plan().payload();
    assert!(matches!(fixture.family.weights(), ModelWeights::None));
    assert!(fixture.family.program().weights().is_empty());
    assert!(matches!(
        payload.execution_weights(),
        ExecutionWeights::None
    ));
    assert!(payload.execution_weights().schema().is_none());
    assert!(payload.execution_weights().stored().is_err());
    assert!(payload.weight_format().is_none());
    assert!(payload.quantization_formats().is_empty());
    assert!(payload.memory().static_allocations().is_empty());
    assert_eq!(payload.memory().static_bytes(), 0);
    for node in payload.nodes() {
        for value in node.values() {
            assert_ne!(value.usage(), BufferUsage::Weights);
            assert!(value.weight().is_none());
        }
    }
    let evidence = fixture
        .resources
        .trusted_runtime_binding()
        .unwrap()
        .evidence();
    assert!(evidence.static_provisioning_identity().is_none());
    assert!(evidence.static_pool_identity().is_none());
    assert_eq!(fixture.composition.runtime().snapshot().allocations, 0);
    let host = fixture.host.clone();
    let mut executor = fixture.executor();
    let outputs = executor
        .execute(
            id("run.weightless.execute"),
            id("request.weightless.execute"),
            vec![input(&host, &fixture.config)],
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].value_id(), &id::<ProgramValueId>(OUTPUT_ID));
    assert_eq!(outputs[0].tensor().dimensions(), [2, 6]);
    assert_eq!(outputs[0].tensor().element_type(), ElementType::Bf16);
    assert!(host.owns(outputs[0].bytes()));
    check_output(outputs[0].bytes().bytes());
    let snapshot = fixture.composition.runtime().snapshot();
    assert_eq!(snapshot.silu_launches, 1);
    assert_eq!(snapshot.dense_linear_launches, 0);
    assert_eq!(snapshot.uploaded_bytes, BYTE_LEN as u64);
    assert_eq!(snapshot.readback_bytes, BYTE_LEN as u64);
    let owner = outputs[0].bytes().clone();
    let slice = owner.slice(4..6).unwrap();
    drop((outputs, executor));
    assert_eq!(host.usage().unwrap().committed_bytes(), BYTE_LEN);
    fixture.close();
    check_output(owner.bytes());
    assert_eq!(slice.bytes(), &0x3f3b_u16.to_le_bytes());
    drop(owner);
    assert_eq!(host.usage().unwrap().committed_bytes(), BYTE_LEN);
    drop(slice);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn weightless_host_capacity_fails_before_admission_or_upload() {
    for (host, expected_scope) in [
        (
            account(2 * BYTE_LEN - 1, BYTE_LEN),
            HostMemoryPressureScope::Total,
        ),
        (
            account(2 * BYTE_LEN, BYTE_LEN - 1),
            HostMemoryPressureScope::Class(HostMemoryClass::Output),
        ),
    ] {
        let fixture = Fixture::new(Config::standard(), host.clone());
        let mut executor = fixture.executor();
        let input = input(&host, &fixture.config);
        let before = fixture.composition.runtime().snapshot();
        let error = executor
            .execute(
                id("run.weightless.host-capacity"),
                id("request.weightless.host-capacity"),
                vec![input.clone()],
            )
            .unwrap_err();
        assert!(
            matches!(error, TensorExecutionError::Host(HostMemoryError::Capacity {
            scope, requested: BYTE_LEN, ..
        }) if scope == expected_scope)
        );
        let after = fixture.composition.runtime().snapshot();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.uploaded_bytes, 0);
        assert_eq!(after.silu_launches, 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), BYTE_LEN);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output),
            0
        );
        drop((input, executor));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        fixture.close();
    }
}

#[test]
fn weightless_dynamic_bytes_and_live_sequence_capacity_are_enforced() {
    let fixture = Fixture::new(Config::standard(), account(2 * BYTE_LEN, BYTE_LEN));
    let minimum = fixture
        .plan()
        .payload()
        .memory()
        .minimum_runnable_request_bytes();
    assert!(minimum > 1);
    let mut too_small = fixture.policy.clone();
    too_small.capacity_bytes = minimum - 1;
    let error = ProgramPlanCompiler::compile(
        &fixture.family,
        fixture.composition.catalog(),
        &too_small,
        &fixture.composition.registry().planning(),
        &options(&fixture.config),
    )
    .unwrap_err();
    assert!(matches!(error, VNextError::PlanCapacityExceeded {
        required_bytes, usable_capacity_bytes,
    } if required_bytes == minimum && usable_capacity_bytes == minimum - 1));

    let binding = fixture
        .resources
        .bind_tensor_execution(fixture.plan(), fixture.host.clone())
        .unwrap();
    let input = input(&fixture.host, &fixture.config);
    let work = binding.bind_inputs(vec![input.clone()]).unwrap();
    let sequence = admit_initial_sequence(&binding, &work);
    let session = sequence.open_session().unwrap();
    let mut executor = fixture.executor();
    let before = fixture.composition.runtime().snapshot();
    let error = executor
        .execute(
            id("run.weightless.occupied"),
            id("request.weightless.occupied"),
            vec![input.clone()],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        TensorExecutionError::Admission {
            stage: TensorAdmissionStage::Request,
            kind: TensorAdmissionFailure::Deferred,
            ..
        }
    ));
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.uploaded_bytes, 0);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), BYTE_LEN);
    session.try_abort_if_quiescent().unwrap();
    drop((session, sequence, work, binding));
    let output = executor
        .execute(
            id("run.weightless.released"),
            id("request.weightless.released"),
            vec![input],
        )
        .unwrap();
    check_output(output[0].bytes().bytes());
    drop((output, executor));
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    fixture.close();
}

#[test]
fn weightless_work_cannot_cross_roots_or_different_plans() {
    let host = account(2 * BYTE_LEN, BYTE_LEN);
    let first = Fixture::new(Config::standard(), host.clone());
    let second = Fixture::new(Config::standard(), host.clone());
    let different = Fixture::new(Config { rows: 3, width: 4 }, host.clone());
    assert_eq!(first.plan().plan_hash(), second.plan().plan_hash());
    assert_ne!(first.plan().plan_hash(), different.plan().plan_hash());
    assert!(first
        .resources
        .bind_tensor_execution(different.plan(), host.clone())
        .is_err());
    let first_binding = first
        .resources
        .bind_tensor_execution(first.plan(), host.clone())
        .unwrap();
    let second_binding = second
        .resources
        .bind_tensor_execution(second.plan(), host.clone())
        .unwrap();
    let work = first_binding
        .bind_inputs(vec![input(&host, &first.config)])
        .unwrap();
    let before = second.composition.runtime().snapshot();
    assert!(second_binding
        .runtime_binding()
        .try_admit_initial_sequence(
            admission_request(&work),
            admission_sequence(&work),
            id("run.weightless.foreign-root"),
            id("request.weightless.foreign-root"),
        )
        .is_err());
    assert!(second_binding.input_uploads(&work).is_err());
    let after = second.composition.runtime().snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    drop((work, first_binding, second_binding));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    first.close();
    second.close();
    different.close();
}

fn canonical(value: Value) -> Value {
    match value {
        Value::Object(fields) => {
            let ordered = fields
                .into_iter()
                .map(|(key, value)| (key, canonical(value)))
                .collect::<BTreeMap<_, _>>();
            serde_json::to_value(ordered).unwrap()
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        scalar => scalar,
    }
}

// Recompute caller-controlled integrity fields: rejection must be semantic,
// not an accidental stale-hash check. Sort independently of serde_json features.
fn rehash(wire: &mut Value) {
    let payload = wire["payload"].as_object_mut().unwrap();
    payload.remove("plan_id");
    let material = canonical(Value::Object(payload.clone()));
    let digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&material).unwrap())
    );
    wire["payload"]["plan_id"] = json!(format!("plan/sha256/{digest}"));
    wire["plan_hash"] = json!(digest);
}

#[test]
fn weightless_plan_wire_is_strict_and_rebuilt_from_trusted_family() {
    let fixture = Fixture::new(Config::standard(), account(2 * BYTE_LEN, BYTE_LEN));
    let bytes = fixture.plan().to_json().unwrap();
    assert_eq!(
        fixture.rehydrate(&bytes, &fixture.family).unwrap(),
        *fixture.plan()
    );
    let wire = serde_json::to_value(fixture.plan()).unwrap();
    let mut rehashed = wire.clone();
    rehash(&mut rehashed);
    assert_eq!(rehashed, wire);
    // Even null cannot sneak a Stored-only field into the empty None variant.
    for field in [
        "plan",
        "schema",
        "materializer_id",
        "materializer_version",
        "materializer_implementation_fingerprint",
        "source_schema_fingerprint",
        "component_sources",
        "static_weight_transforms",
        "approximate_quality_approval",
        "artifact_abi",
    ] {
        let mut forged = wire.clone();
        forged["payload"]["execution_weights"][field] = Value::Null;
        rehash(&mut forged);
        assert!(
            ExecutionPlan::decode_untrusted(&serde_json::to_vec(&forged).unwrap()).is_err(),
            "accepted None.{field}"
        );
    }
    for (field, value) in [
        (
            "weight_format",
            json!(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID),
        ),
        (
            "quantization_formats",
            json!(["quantization.weightless.forged"]),
        ),
    ] {
        let mut forged = wire.clone();
        forged["payload"][field] = value;
        rehash(&mut forged);
        assert!(
            fixture
                .rehydrate(&serde_json::to_vec(&forged).unwrap(), &fixture.family)
                .is_err(),
            "accepted false {field} summary"
        );
    }
    let foreign = family(&Config { rows: 3, width: 4 });
    assert!(fixture.rehydrate(&bytes, &foreign).is_err());
    let mut missing = wire.clone();
    missing["payload"]
        .as_object_mut()
        .unwrap()
        .remove("execution_weights");
    assert!(ExecutionPlan::decode_untrusted(&serde_json::to_vec(&missing).unwrap()).is_err());
    let mut old = wire;
    old["payload"]["schema"] = json!({"major": EXECUTION_PLAN_SCHEMA.major - 1, "minor": 0});
    assert!(matches!(
        ExecutionPlan::decode_untrusted(&serde_json::to_vec(&old).unwrap()),
        Err(VNextError::UnsupportedPlanSchema { .. })
    ));
    fixture.close();
}

struct MaterializerObserver {
    descriptor: WeightMaterializerDescriptor,
    calls: Arc<AtomicUsize>,
}

impl WeightMaterializer for MaterializerObserver {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        _: &PreparedModelFamily,
        _: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(VNextError::InvalidExecutionPlan {
            reason: "unexpected materializer call for weightless work".into(),
        })
    }

    fn materialize_component<'source>(
        &self,
        _: &'source dyn WeightComponentSource,
        _: &[&WeightComponentSpec],
        _: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(VNextError::InvalidExecutionPlan {
            reason: "unexpected materialization for weightless work".into(),
        })
    }
}

// A fully schema-valid, internally hashed numeric selection is a negative-test
// input, never a successful weightless artifact or a checkpoint/source object.
fn quality_artifact() -> (WeightMaterializerDescriptor, Vec<u8>) {
    let reference = format!("{:x}", Sha256::digest(1.0_f32.to_bits().to_le_bytes()));
    let actual = format!("{:x}", Sha256::digest(0x3c00_u16.to_le_bytes()));
    let checkpoint = json!({"id":"checkpoint.numeric-selection", "repository":"ferrum/test-vector", "revision":"cccccccccccccccccccccccccccccccccccccccc"});
    let payload = json!({
        "activation_batches": [1, 4], "activation_contract": {"dtype":"F16"},
        "cases": (0..4).map(|n| json!({"case_id":format!("case-{n}"), "reference_f32le_sha256":reference})).collect::<Vec<_>>(),
        "checkpoint": checkpoint, "fixture_id":"weightless-negative-selection",
        "generator":{"algorithm":"constant-v1"}, "reference_contract":{"output_dtype":"F32 little-endian"},
        "schema_version":1, "source_contract":{"values_dtype":"F16"}, "weight_shapes":[[1,1],[1,1]],
    });
    let digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical(payload.clone())).unwrap())
    );
    let contract = ApproximateWeightQualityContract::new(
        "a".repeat(64),
        digest,
        4,
        CanonicalRational::new(1, 20).unwrap(),
        0,
        0,
    )
    .unwrap();
    let descriptor = WeightMaterializerDescriptor::new(
        id("weight-materializer.weightless.approximate-observer"),
        ContractVersion::new(1, 0),
        "b".repeat(64),
        WeightMaterializationFidelity::Approximate,
        BTreeSet::new(),
    )
    .unwrap()
    .with_approximate_quality_contract(contract.clone())
    .unwrap();
    let artifact = json!({
        "authority": {"id":NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID, "version":{"major":1,"minor":0}, "implementation_fingerprint":numeric_weight_quality_authority_implementation_fingerprint().unwrap()},
        "cases": (0..4).map(|n| json!({
            "actual_f16_bits":[0x3c00_u16], "actual_f16le_sha256":actual,
            "case_id":format!("case-{n}"), "inf_count":0, "nan_count":0,
            "reference_f32_bits":[1.0_f32.to_bits()], "reference_f32le_sha256":reference,
            "relative_l2_upper_bound":{"denominator":1,"numerator":0},
        })).collect::<Vec<_>>(),
        "checkpoint": checkpoint,
        "contract":{"execution_contract_fingerprint":contract.execution_contract_fingerprint(), "quality_vector_digest":contract.quality_vector_digest()},
        "execution":{"quantization_format_ids":[], "weight_format_id":"weight-format.dense", "weight_layout_id":"weight-layout.numeric-selection"},
        "materializer":{"fidelity":"approximate", "id":descriptor.id(), "implementation_fingerprint":descriptor.implementation_fingerprint(), "version":descriptor.version()},
        "quality_vector_payload":payload, "schema_id":NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID,
        "source":{"weight_format_id":"weight-format.dense"},
    });
    (
        descriptor,
        serde_json::to_vec(&canonical(artifact)).unwrap(),
    )
}

#[test]
fn weightless_auto_skips_materializers_but_every_explicit_require_is_rejected() {
    let fixture = Fixture::new(Config::standard(), account(2 * BYTE_LEN, BYTE_LEN));
    let calls = Arc::new(AtomicUsize::new(0));
    let exact = WeightMaterializerDescriptor::new(
        id("weight-materializer.weightless.exact-observer"),
        ContractVersion::new(1, 0),
        "d".repeat(64),
        WeightMaterializationFidelity::Exact,
        BTreeSet::new(),
    )
    .unwrap();
    let (approximate, artifact) = quality_artifact();
    let selection = WeightMaterializerSelection::numeric_quality_artifact(
        approximate.id().clone(),
        artifact.clone(),
    )
    .unwrap();
    assert!(selection.has_numeric_quality_artifact());
    let materializers = WeightMaterializerRegistry::new(vec![
        Box::new(MaterializerObserver {
            descriptor: exact.clone(),
            calls: calls.clone(),
        }),
        Box::new(MaterializerObserver {
            descriptor: approximate.clone(),
            calls: calls.clone(),
        }),
    ])
    .unwrap();
    let catalog = materializers
        .augment_catalog(fixture.composition.catalog().clone())
        .unwrap();
    let compile = |options: &ProgramPlanCompileOptions| {
        ProgramPlanCompiler::compile_with_weight_materializers(
            &fixture.family,
            &catalog,
            &fixture.policy,
            &fixture.composition.registry().planning(),
            &materializers,
            options,
        )
    };
    assert!(compile(&options(&fixture.config)).is_ok());
    let identity = id::<WeightMaterializerId>(IDENTITY_WEIGHT_MATERIALIZER_ID);
    for selection in [
        WeightMaterializerSelection::exact(identity.clone()),
        WeightMaterializerSelection::exact(exact.id().clone()),
        selection,
    ] {
        let mut options = options(&fixture.config);
        options.require_weight_materializer_selection(selection);
        assert!(compile(&options).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    let mut explicit_identity = options(&fixture.config);
    explicit_identity.require_weight_materializer(identity.clone());
    assert!(compile(&explicit_identity).is_err());
    let mut explicit_quality = options(&fixture.config);
    explicit_quality
        .require_weight_materializer_with_numeric_quality_artifact(
            approximate.id().clone(),
            artifact.clone(),
        )
        .unwrap();
    assert!(compile(&explicit_quality).is_err());
    for materializer in [&identity, exact.id()] {
        assert!(materializers
            .select_exact(&fixture.family, &catalog, materializer)
            .is_err());
    }
    assert!(materializers
        .select_with_numeric_quality_artifact(
            &fixture.family,
            &catalog,
            approximate.id(),
            &artifact
        )
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.composition.runtime().snapshot().allocations, 0);
    fixture.close();
}

struct ForwardSilu(Box<dyn OperationProvider<ReferenceDeviceRuntime>>);

impl OperationResourceEstimator for ForwardSilu {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        self.0.descriptor()
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        self.0.estimate_resources(request)
    }
}

impl OperationProvider<FaultRuntime> for ForwardSilu {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        self.0.reusable_execution_topology(request)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        self.0.encode_selected(invocation)
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        self.0.encode_tensor_selected(invocation)
    }
}

#[test]
fn weightless_pending_retains_owners_until_quiescent_abort_then_reuses_capacity() {
    let config = Config::standard();
    let family = family(&config);
    let composition =
        ReferenceVNextComposition::create(id("device.reference.weightless.fault")).unwrap();
    let runtime = Arc::new(FaultRuntime::new(composition.runtime().clone()));
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(silu_bf16_contract().unwrap())],
        vec![Box::new(ForwardSilu(
            reference_vnext_silu_bf16_provider(&runtime.inner).unwrap(),
        )) as Box<dyn OperationProvider<FaultRuntime>>],
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(
            runtime.descriptor().clone(),
            vec![EngineProviderDescriptor::new(
                id("provider.engine.reference.weightless.fault-observer"),
                ContractVersion::new(1, 0),
                runtime
                    .descriptor()
                    .runtime_implementation_fingerprint
                    .clone(),
                runtime.descriptor().id.clone(),
                runtime.descriptor().capabilities.clone(),
            )
            .unwrap()],
        )
        .unwrap();
    let compilation = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options(&config),
    )
    .unwrap();
    let resources = no_static(runtime.clone(), compilation.executable().execution_plan());
    let host = account(2 * BYTE_LEN, BYTE_LEN);
    let mut executor = PreparedTensorExecutor::new(
        compilation.executable().clone(),
        registry.bind_plan(compilation.executable()).unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap();
    runtime.set_fence(FenceMode::Indeterminate);
    let mut error = executor
        .execute(
            id("run.weightless.indeterminate"),
            id("request.weightless.indeterminate"),
            vec![input(&host, &config)],
        )
        .unwrap_err();
    let TensorExecutionError::NonTerminal {
        completion,
        retirement,
        reaper,
        ..
    } = &mut error
    else {
        panic!("expected retained completion authority, received {error:?}");
    };
    assert_eq!(reaper.retained_count(), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), BYTE_LEN);
    assert_eq!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output),
        0
    );
    assert_eq!(runtime.inner.snapshot().silu_launches, 1);
    assert_eq!(runtime.readback_calls.load(Ordering::SeqCst), 0);
    assert!(retirement.try_abort().is_err());
    runtime.set_fence(FenceMode::Pending);
    assert!(matches!(
        completion.poll().unwrap(),
        CompletionObservation::Pending
    ));
    assert!(retirement.try_abort().is_err());
    assert_eq!(reaper.retained_count(), 1);
    runtime.set_fence(FenceMode::Real);
    assert!(matches!(
        completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    retirement.try_abort().unwrap();
    retirement.try_abort().unwrap();
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    // The retained diagnostic error cannot keep a retired slot or input alive.
    // Recovery intentionally aborts; it does not promise recovered output.
    let output = executor
        .execute(
            id("run.weightless.after-recovery"),
            id("request.weightless.after-recovery"),
            vec![input(&host, &config)],
        )
        .unwrap();
    check_output(output[0].bytes().bytes());
    drop((output, executor, compilation));
    close_resources(resources);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.inner.snapshot().live_allocations, 0);
    drop(error);
}
