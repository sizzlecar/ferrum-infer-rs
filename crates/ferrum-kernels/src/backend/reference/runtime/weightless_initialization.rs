//! Initialization-only evidence, not a cached-copy or model execution provider.
//! A metadata fixture requests a real Plan persistent allocation. The probe
//! delegates allocation and zero commands to Reference and observes only weak
//! allocation references, so it cannot keep retired memory artificially alive.

use super::*;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{OnceLock, Weak};

const OPERATION: &str = "operation.test.initialize-persistent";
const INPUT: &str = "value.initialize.input";
const OUTPUT: &str = "value.initialize.output";
const PROFILE: &str = "profile.initialize.bf16";
const PERSISTENT_BYTES: u64 = 3072;
const CAPACITY_BYTES: u64 = 4096;

fn id<T: TryFrom<String, Error = VNextError>>(value: &str) -> T {
    T::try_from(value.to_owned()).unwrap()
}

fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.to_owned(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    persistent_bytes: u64,
}

struct Family;

impl ModelFamilyProvider for Family {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.initialize-persistent"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.initialize-persistent")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.persistent_bytes != PERSISTENT_BYTES
            || *raw != json!({"persistent_bytes": PERSISTENT_BYTES})
        {
            return Err(invalid(
                "initialization fixture requires its declared persistent extent",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.test.initialize-persistent"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone())
            .map_err(|_| invalid("invalid initialization fixture config"))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(OPERATION),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT)],
            vec![ProgramBlock {
                id: "block.initialize".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.initialize"),
                    operation_id: id(OPERATION),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT)],
                    outputs: vec![id(OUTPUT)],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )
    }
}

struct Contract(OperationDescriptor);
impl Contract {
    fn new() -> Self {
        let tensor = |access| {
            TensorContract::new(
                vec![DimensionConstraint::Exact(1), DimensionConstraint::Exact(4)],
                BTreeSet::from([ElementType::Bf16]),
                vec![LayoutConstraint::Contiguous],
                access,
                AliasPolicy::NoAlias,
            )
            .unwrap()
        };
        let descriptor = OperationDescriptor {
            id: id(OPERATION),
            version: ContractVersion::new(1, 0),
            inputs: vec![tensor(TensorAccess::Read)],
            outputs: vec![tensor(TensorAccess::Write)],
            attributes: AttributeSchema::empty(),
            resources: ResourceRequirements {
                minimum_value_alignment_bytes: 16,
                scratch: ResourcePresenceRequirement::Forbidden,
                binding: ResourcePresenceRequirement::Forbidden,
                persistent: ResourcePresenceRequirement::Required,
            },
            oracle: OracleSpec::Exact,
            provider: ProviderRequirement {
                minimum_version: ContractVersion::new(1, 0),
                required_capabilities: BTreeSet::new(),
            },
            profile_phase: ProfilePhase::Prepare,
        };
        descriptor.validate().unwrap();
        Self(descriptor)
    }
}
impl OperationContract for Contract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.0
    }
    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if inputs != self.0.inputs || outputs != self.0.outputs {
            return Err(invalid("initialization signature mismatch"));
        }
        Ok(())
    }
}

/// Used only to exercise the compiler's real resource estimates. Dispatch is
/// deliberately rejected: these tests make no claim about executing this node.
struct InitializationOnlyProvider(OperationProviderDescriptor);
impl InitializationOnlyProvider {
    fn new(runtime: &ProbeRuntime) -> Self {
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(include_bytes!("weightless_initialization.rs"))
        );
        Self(
            OperationProviderDescriptor::new(
                id("provider.test.initialize-persistent"),
                id(OPERATION),
                Contract::new().0.fingerprint().unwrap(),
                fingerprint.clone(),
                ProviderExecutionSemantics::bitwise_eager_only(),
                ContractVersion::new(1, 0),
                runtime.descriptor().id.clone(),
                BTreeSet::new(),
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
                "estimator.test.initialize-persistent",
                ContractVersion::new(1, 0),
                fingerprint,
            )
            .unwrap()
            .with_fixed_tensor_work(true),
        )
    }
}
impl OperationResourceEstimator for InitializationOnlyProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.0
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().fingerprint()? != self.0.operation_fingerprint() {
            return Err(invalid("initialization estimator operation mismatch"));
        }
        Ok(OperationResourceEstimate::new(
            self.0.resource_estimator_id(),
            self.0.resource_estimator_version(),
            self.0.resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            Some(ProviderWorkspaceRequirement::new(
                PERSISTENT_BYTES,
                16,
                ProviderWorkspaceScope::Plan,
                ProviderWorkspaceReusePolicy::Preserve,
                DynamicStorageRequirement::contiguous(),
            )?),
        ))
    }
}
impl OperationProvider<ProbeRuntime> for InitializationOnlyProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(OperationFailure::new(
            invocation.participants()[0].identity().clone(),
            ProfilePhase::Prepare,
            "test.initialization-only",
            "fixture covers initialization, not execution",
            false,
        )
        .unwrap())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Policy;
impl RuntimePolicy for Policy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        CAPACITY_BYTES
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
        static PROFILES: OnceLock<Vec<DynamicStorageProfile>> = OnceLock::new();
        PROFILES.get_or_init(|| {
            DynamicStorageRequirement::contiguous()
                .accepted_profiles()
                .to_vec()
        })
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> Result<(), VNextError> {
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    NotSubmitted,
    SubmitPanic,
    Indeterminate,
    FailedButQuiescent,
}

struct Observation {
    allocation: Weak<ReferenceAllocation>,
    range: Range<usize>,
}
struct ProbeRuntime {
    inner: ReferenceDeviceRuntime,
    descriptor: DeviceDescriptor,
    observations: Mutex<Vec<Observation>>,
    fault: Mutex<Fault>,
    synchronize_fails: AtomicBool,
    import_calls: AtomicUsize,
    upload_calls: AtomicUsize,
}
impl ProbeRuntime {
    fn new(name: &str) -> Arc<Self> {
        let mut config =
            super::super::composition::reference_vnext_runtime_config(id(name)).unwrap();
        config.descriptor.total_memory_bytes = CAPACITY_BYTES;
        let inner = ReferenceDeviceRuntime::new(config).unwrap();
        let mut descriptor = inner.descriptor().clone();
        descriptor.runtime_implementation_fingerprint = format!(
            "{:x}",
            Sha256::digest(
                [
                    descriptor.runtime_implementation_fingerprint.as_bytes(),
                    include_bytes!("weightless_initialization.rs"),
                ]
                .concat()
            )
        );
        Arc::new(Self {
            inner,
            descriptor,
            observations: Mutex::new(vec![]),
            fault: Mutex::new(Fault::None),
            synchronize_fails: AtomicBool::new(false),
            import_calls: AtomicUsize::new(0),
            upload_calls: AtomicUsize::new(0),
        })
    }
    fn assert_live_bytes(&self, expected: u8) {
        let observations = self.observations.lock().unwrap();
        assert!(!observations.is_empty());
        for observation in observations.iter() {
            let allocation = observation
                .allocation
                .upgrade()
                .expect("transaction must retain its allocation");
            assert!(allocation.lock()[observation.range.clone()]
                .iter()
                .all(|&byte| byte == expected));
        }
    }
    fn assert_released(&self) {
        assert_eq!(self.inner.snapshot().live_allocations, 0);
        assert!(self
            .observations
            .lock()
            .unwrap()
            .iter()
            .all(|observation| observation.allocation.upgrade().is_none()));
    }
}
impl DeviceRuntime for ProbeRuntime {
    type Buffer = ReferenceDeviceBuffer;
    type Stream = ReferenceDeviceStream;
    type Command = ReferenceDeviceCommand;
    type Fence = ReferenceDeviceFence;
    type Error = ReferenceDeviceRuntimeError;
    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        self.inner.attention_execution_policy()
    }
    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let buffer = self.inner.allocate(permit)?;
        let range = buffer.allocation.logical_offset_bytes
            ..buffer.allocation.logical_offset_bytes + buffer.allocation.logical_size_bytes;
        buffer.allocation.lock()[range.clone()].fill(0xa5);
        self.observations.lock().unwrap().push(Observation {
            allocation: Arc::downgrade(&buffer.allocation),
            range,
        });
        Ok(buffer)
    }
    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        self.inner.buffer_descriptor(buffer)
    }
    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        self.inner.create_stream()
    }
    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        self.inner.stream_state(stream)
    }
    fn encode_copy(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        self.inner.encode_copy(source, destination, region)
    }
    fn encode_upload(
        &self,
        source: &[u8],
        layout: HostTransferLayout,
        destination: &Self::Buffer,
        offset: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.upload_calls.fetch_add(1, Ordering::Relaxed);
        self.inner
            .encode_upload(source, layout, destination, offset)
    }
    fn begin_static_weight_import(
        &self,
    ) -> Option<
        Result<Box<dyn StaticWeightImportSession<Self::Buffer, Self::Error> + '_>, Self::Error>,
    > {
        self.import_calls.fetch_add(1, Ordering::Relaxed);
        None
    }
    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        offset: u64,
        length: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.inner.encode_zero(destination, offset, length)
    }
    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        let fault = *self.fault.lock().unwrap();
        if fault == Fault::NotSubmitted {
            return Err(DefinitelyNotSubmitted::new(
                ReferenceDeviceRuntimeError::contract("injected no submission"),
            ));
        }
        if fault == Fault::SubmitPanic {
            // Work really ran before the observer lost its fence. The caller
            // must retain the original stream and allocation until recovery.
            let _fence = self.inner.submit(stream, commands).unwrap();
            panic!("injected lost submit result");
        }
        self.inner.submit(stream, commands)
    }
    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        self.inner.submission_attribution(fence)
    }
    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        self.inner.query_fence(fence)
    }
    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        match *self.fault.lock().unwrap() {
            Fault::Indeterminate => Err(FenceIndeterminate::new(
                ReferenceDeviceRuntimeError::contract("injected unknown completion"),
            )),
            Fault::FailedButQuiescent => Ok(DeviceTerminalReceipt::unprofiled(
                DeviceTerminal::FailedButQuiescent(ReferenceDeviceRuntimeError::contract(
                    "injected terminal failure",
                )),
            )),
            _ => self.inner.wait_fence(fence),
        }
    }
    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        if self.synchronize_fails.load(Ordering::Relaxed) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "injected recovery failure",
            ));
        }
        self.inner.synchronize(stream)
    }
    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        self.inner.readback(stream, source, region, layout)
    }
    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        self.inner.describe_error(error)
    }
}

type Transaction = ResourceTransaction<RuntimeResourceDriver<ProbeRuntime>, TransactionCommitted>;

struct Fixture {
    runtime: Arc<ProbeRuntime>,
    family: PreparedModelFamily,
    plan: ExecutionPlan,
}
impl Fixture {
    fn new(name: &str) -> Self {
        let runtime = ProbeRuntime::new(name);
        let family = TypedFamilyRegistration::new(Family)
            .prepare_with_profile(
                &json!({"persistent_bytes": PERSISTENT_BYTES}),
                &NumericalProfileId::new(PROFILE).unwrap(),
            )
            .unwrap();
        let registry = OperationRuntimeRegistry::new(
            vec![Box::new(Contract::new())],
            vec![Box::new(InitializationOnlyProvider::new(&runtime))
                as Box<dyn OperationProvider<ProbeRuntime>>],
        )
        .unwrap();
        let engines = vec![EngineProviderDescriptor::new(
            id("provider.engine.initialize-test"),
            ContractVersion::new(1, 0),
            runtime
                .descriptor()
                .runtime_implementation_fingerprint
                .clone(),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .unwrap()];
        // No weight materializer registry, archive, checkpoint, or source.
        let catalog = registry
            .capability_catalog(runtime.descriptor().clone(), engines)
            .unwrap();
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(INPUT),
            ProgramTensorSpec {
                dimensions: vec![1, 4],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        options.retain_completion_value(id(OUTPUT));
        let compilation = ProgramPlanCompiler::compile(
            &family,
            &catalog,
            &Policy,
            &registry.planning(),
            &options,
        )
        .unwrap();
        let plan = compilation.executable().execution_plan().clone();
        assert!(matches!(family.weights(), ModelWeights::None));
        assert!(matches!(
            plan.payload().execution_weights(),
            ExecutionWeights::None
        ));
        let [allocation] = plan.payload().memory().static_allocations() else {
            panic!("exactly one real persistent allocation required")
        };
        assert_eq!(allocation.usage(), BufferUsage::Persistent);
        assert_eq!(allocation.element_type(), ElementType::U8);
        assert_eq!(allocation.size_bytes(), PERSISTENT_BYTES);
        Self {
            runtime,
            family,
            plan,
        }
    }
    fn commit(&self) -> Transaction {
        let provisioning = self
            .plan
            .provision_static(self.runtime.clone(), id("request.initialize"))
            .unwrap()
            .into_provisioning();
        let StaticProvisioning::Required(permit) = provisioning else {
            panic!("weightless persistent storage still requires a transaction")
        };
        let identity = ResourceTransactionIdentity::for_admission(
            permit.binding(),
            id("run.initialize"),
            id("transaction.initialize"),
        );
        let reserved = ResourceTransaction::begin(
            RuntimeResourceDriver::new(self.runtime.clone()).unwrap(),
            identity,
            permit,
        )
        .unwrap()
        .reserve()
        .unwrap();
        match reserved.commit() {
            Ok(transaction) => transaction,
            Err(_) => panic!("real reference static allocation failed"),
        }
    }
    fn initialize(
        &self,
        transaction: Transaction,
    ) -> Result<
        InitializedResourceTransaction<RuntimeResourceDriver<ProbeRuntime>>,
        StaticInitializationFailure<RuntimeResourceDriver<ProbeRuntime>>,
    > {
        transaction.initialize_static_without_weights(
            &self.family,
            &self.plan,
            StaticInitializationPolicy::new(16, 8).unwrap(),
        )
    }
    fn assert_capacity_held(&self) {
        assert!(matches!(
            self.plan
                .provision_static(self.runtime.clone(), id("request.initialize.contender")),
            Err(VNextError::DeviceCapacityUnavailable(_))
        ));
        assert_eq!(self.runtime.inner.snapshot().live_allocations, 1);
    }
    fn assert_capacity_reusable(&self) {
        let provisioned = self
            .plan
            .provision_static(self.runtime.clone(), id("request.initialize.reuse"))
            .unwrap();
        assert!(matches!(
            provisioned.into_provisioning(),
            StaticProvisioning::Required(_)
        ));
    }
}

#[test]
fn weightless_required_initialization_zeroes_real_bytes_without_weight_activity() {
    let fixture = Fixture::new("device.reference.weightless-initialize-success");
    let transaction = fixture.commit();
    fixture.runtime.assert_live_bytes(0xa5);
    fixture.assert_capacity_held();
    let initialized = fixture.initialize(transaction).unwrap();
    fixture.runtime.assert_live_bytes(0);
    let receipt = initialized.receipt();
    assert_eq!(receipt.initialized_resource_count(), 1);
    assert_eq!(receipt.submission_batch_count(), 1);
    assert_eq!(receipt.uploaded_component_count(), 0);
    assert_eq!(receipt.imported_component_count(), 0);
    assert_eq!(receipt.transformed_component_count(), 0);
    assert_eq!(
        receipt.uploaded_bytes() + receipt.imported_bytes() + receipt.transformed_bytes(),
        0
    );
    assert_eq!(
        receipt.upload_command_count() + receipt.transform_command_count(),
        0
    );
    assert!(receipt.source_files().is_empty());
    assert_eq!(fixture.runtime.upload_calls.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.runtime.import_calls.load(Ordering::Relaxed), 0);
    let root = initialized.into_plan_runtime().unwrap();
    fixture.assert_capacity_held();
    match PlanRuntimeResources::close(root)
        .unwrap_or_else(|_| panic!("static fixture root must close"))
    {
        PlanRuntimeCloseOutcome::Closed(receipt) => {
            assert_eq!(receipt.released_static_resources(), 1)
        }
        _ => panic!("initialization fixture unexpectedly retained root authority"),
    }
    fixture.runtime.assert_released();
    fixture.assert_capacity_reusable();
}

#[test]
fn weightless_not_submitted_and_quiescent_failures_retire_the_real_allocation() {
    for (name, fault, expected_byte) in [
        (
            "device.reference.weightless-not-submitted",
            Fault::NotSubmitted,
            0xa5,
        ),
        (
            "device.reference.weightless-terminal-failure",
            Fault::FailedButQuiescent,
            0,
        ),
    ] {
        let fixture = Fixture::new(name);
        let transaction = fixture.commit();
        *fixture.runtime.fault.lock().unwrap() = fault;
        let mut failure = fixture
            .initialize(transaction)
            .err()
            .expect("injected initialization must fail");
        assert!(!failure.is_indeterminate());
        fixture.runtime.assert_live_bytes(expected_byte);
        fixture.assert_capacity_held();
        failure.retire_in_place().unwrap();
        assert!(failure.is_retired());
        fixture.runtime.assert_released();
        fixture.assert_capacity_reusable();
        // Keep the diagnostic failure alive while proving its resources retired.
        failure.retire_in_place().unwrap();
    }
}

#[test]
fn weightless_indeterminate_and_lost_submit_result_hold_capacity_until_recovery() {
    for (name, fault) in [
        (
            "device.reference.weightless-indeterminate",
            Fault::Indeterminate,
        ),
        (
            "device.reference.weightless-submit-panic",
            Fault::SubmitPanic,
        ),
    ] {
        let fixture = Fixture::new(name);
        let transaction = fixture.commit();
        *fixture.runtime.fault.lock().unwrap() = fault;
        let failure = fixture
            .initialize(transaction)
            .err()
            .expect("injected initialization must fail");
        assert!(failure.is_indeterminate());
        let mut failure = failure
            .into_transaction()
            .err()
            .expect("unknown completion cannot return an active transaction");
        fixture.runtime.assert_live_bytes(0);
        fixture.assert_capacity_held();
        fixture
            .runtime
            .synchronize_fails
            .store(true, Ordering::Relaxed);
        assert!(failure.retire_in_place().is_err());
        fixture.assert_capacity_held();
        fixture.runtime.assert_live_bytes(0);
        fixture
            .runtime
            .synchronize_fails
            .store(false, Ordering::Relaxed);
        failure.retire_in_place().unwrap();
        assert!(failure.is_retired());
        fixture.runtime.assert_released();
        fixture.assert_capacity_reusable();
    }
}

#[test]
fn weightless_recovery_reinitializes_the_same_admitted_allocation() {
    let fixture = Fixture::new("device.reference.weightless-retry");
    let transaction = fixture.commit();
    *fixture.runtime.fault.lock().unwrap() = Fault::Indeterminate;
    let failure = fixture
        .initialize(transaction)
        .err()
        .expect("injected wait must fail");
    fixture
        .runtime
        .synchronize_fails
        .store(true, Ordering::Relaxed);
    let failure = failure
        .recover()
        .err()
        .expect("failed recovery must keep the original transaction");
    fixture.assert_capacity_held();
    fixture
        .runtime
        .synchronize_fails
        .store(false, Ordering::Relaxed);
    *fixture.runtime.fault.lock().unwrap() = Fault::None;
    let transaction = failure.recover().unwrap();
    {
        let observations = fixture.runtime.observations.lock().unwrap();
        let [observation] = observations.as_slice() else {
            panic!("recovery must not reallocate")
        };
        let allocation = observation.allocation.upgrade().unwrap();
        allocation.lock()[observation.range.clone()].fill(0xa5);
    }
    fixture.runtime.assert_live_bytes(0xa5);
    let initialized = fixture.initialize(transaction).unwrap();
    fixture.runtime.assert_live_bytes(0);
    assert_eq!(initialized.receipt().submission_batch_count(), 1);
    let mut retired = initialized.abort_activation(invalid("test stops after cold initialization"));
    retired.retire_in_place().unwrap();
    fixture.runtime.assert_released();
    fixture.assert_capacity_reusable();
}

struct RejectedSource(AtomicUsize);
impl WeightComponentSource for RejectedSource {
    fn component<'a>(
        &'a self,
        _: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Err(invalid(
            "a weightless initializer must never request a source component",
        ))
    }
}

#[test]
fn passing_a_source_to_none_is_rejected_before_zero_or_import() {
    let fixture = Fixture::new("device.reference.weightless-source-mismatch");
    let transaction = fixture.commit();
    let source = RejectedSource(AtomicUsize::new(0));
    let mut failure = transaction
        .initialize_static(
            &fixture.family,
            &fixture.plan,
            &source,
            StaticInitializationPolicy::new(16, 8).unwrap(),
        )
        .err()
        .expect("a source cannot be silently ignored");
    assert!(!failure.is_indeterminate());
    assert_eq!(source.0.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.runtime.import_calls.load(Ordering::Relaxed), 0);
    fixture.runtime.assert_live_bytes(0xa5);
    fixture.assert_capacity_held();
    failure.retire_in_place().unwrap();
    fixture.runtime.assert_released();
    fixture.assert_capacity_reusable();
}
