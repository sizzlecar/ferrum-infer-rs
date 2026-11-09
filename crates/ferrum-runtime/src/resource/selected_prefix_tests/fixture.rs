//! A byte-backed Reference backend and one real token program. No test-created
//! allocation permits, wave seals, or accepted-frontier proofs are used here.
use crate::test_support::resource_support::{
    contiguous_storage_bindings, contiguous_storage_profile, fixture_f32_profiles, id, sha,
    PrepareFixture, TestConfig, TestFamily, TestRuntimeError,
};
use crate::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, Weak,
    },
};

const X: &str = "value.input";
const A: &str = "value.state.a";
const B: &str = "value.state.b";
const KV: &str = "value.state.kv";
const Y: &str = "value.output";
const CA: &str = "value.candidate.a";
const CB: &str = "value.candidate.b";
const NODE: &str = "node.selected";
const OP: &str = "operation.selected.fixture";
const WEIGHT: &str = "value.weight";
pub const VISUAL: &str = "value.visual";
const SELECTOR: &str = "value.output_selector";
const FINAL: &str = "value.final";
const OUTPUT_OP: &str = "operation.selected.output.fixture";

#[derive(Clone)]
pub struct Buffer(Arc<Allocation>);
struct Allocation {
    descriptor: BufferDescriptor,
    bytes: Mutex<Vec<u8>>,
    live_bytes: Arc<AtomicU64>,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        self.live_bytes
            .fetch_sub(self.descriptor.size_bytes, Ordering::SeqCst);
    }
}
#[derive(Clone)]
pub(super) struct Region {
    buffer: Buffer,
    range: Range<usize>,
    // Commands retain the actual Core translation owner, not only the backing.
    _retention: DeviceBufferRetention,
}
impl Region {
    fn read(&self) -> Vec<u8> {
        self.buffer.0.bytes.lock().unwrap()[self.range.clone()].to_vec()
    }
    fn write(&self, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.range.len());
        self.buffer.0.bytes.lock().unwrap()[self.range.clone()].copy_from_slice(bytes);
    }
}
#[derive(Clone)]
pub struct Probe {
    owner: Weak<Allocation>,
    range: Range<usize>,
}
impl Probe {
    fn new(region: &Region) -> Self {
        Self {
            owner: Arc::downgrade(&region.buffer.0),
            range: region.range.clone(),
        }
    }
    pub fn values(&self) -> Vec<f32> {
        let owner = self.owner.upgrade().expect("state still admitted");
        let values = floats(&owner.bytes.lock().unwrap()[self.range.clone()]);
        values
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyOutcome {
    #[default]
    Succeed,
    FailAfterFirst,
    Indeterminate,
}
pub struct Runtime {
    descriptor: DeviceDescriptor,
    pub live_bytes: Arc<AtomicU64>,
    pub copied: AtomicU64,
    pub submitted: AtomicU64,
    pub next_copy: Mutex<CopyOutcome>,
    pub drain_fails: Mutex<bool>,
    pub replay_address_checks: Arc<AtomicU64>,
    pub eager_single_rows: Arc<AtomicU64>,
    probes: Arc<Mutex<Vec<[Probe; 3]>>>,
}
pub struct Stream {
    state: StreamState,
}
pub struct Fence {
    outcome: CopyOutcome,
    _commands: Vec<Command>,
}
pub enum Command {
    Copy(Buffer, Buffer, CopyRegion),
    Upload(ManagedHostBytes, Buffer, usize),
    Zero(Buffer, Range<usize>),
    Binding(Vec<(Region, [u8; 16])>),
    Compute(Vec<Compute>),
    Output(Vec<OutputCompute>),
}
pub struct OutputCompute {
    input: Region,
    selector: Region,
    output: Region,
}
pub struct Compute {
    input: Region,
    states: [Region; 3],
    candidates: [Region; 2],
    output: Region,
    weight: Region,
    visual: Option<Region>,
    start: usize,
    binding: Option<Region>,
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Compute {
    fn execute(&self) {
        let input = self.input.read();
        let tokens: Vec<_> = input
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let start = if let Some(binding) = &self.binding {
            let metadata = binding.read();
            let start = u64::from_le_bytes(metadata[..8].try_into().unwrap()) as usize;
            let count = u64::from_le_bytes(metadata[8..].try_into().unwrap()) as usize;
            assert_eq!(count, tokens.len(), "fresh binding must describe this wave");
            start
        } else {
            self.start
        };
        let mut a = floats(&self.states[0].read());
        let mut b = floats(&self.states[1].read());
        let mut kv = floats(&self.states[2].read());
        let mut previous_a = a.clone();
        let mut previous_b = b.clone();
        let mut output = Vec::new();
        let weight = floats(&self.weight.read());
        let visual = self.visual.as_ref().map(|v| floats(&v.read()));
        for (i, token) in tokens.iter().enumerate() {
            previous_a.clone_from(&a);
            previous_b.clone_from(&b);
            let t = *token as f32 * weight[0]
                + visual.as_ref().map_or(0., |v| v[2 * i] + 2. * v[2 * i + 1]);
            // Two different recurrences; small integer-valued F32 stays exact.
            a = vec![a[1], a[0] + 2.0 * t];
            b = vec![b[0] + t, b[1] + a[1], b[2] + b[0] + 3.0 * t];
            kv[start + i] = a[1] + b[2];
            output.push(kv[start + i]);
        }
        previous_a.extend(a);
        previous_b.extend(b);
        self.candidates[0].write(&bytes(&previous_a));
        self.candidates[1].write(&bytes(&previous_b));
        self.states[2].write(&bytes(&kv));
        self.output.write(&bytes(&output));
    }
}

impl Runtime {
    fn new() -> Arc<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Arc::new(Self {
            descriptor: DeviceDescriptor {
                id: id(format!(
                    "device.selected.reference.{}",
                    NEXT.fetch_add(1, Ordering::SeqCst)
                )),
                class: DeviceClass::Reference,
                ordinal: 0,
                total_memory_bytes: 1 << 20,
                runtime_implementation_fingerprint: sha('a'),
                capabilities: BTreeSet::from([id("capability.compute")]),
                dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
            },
            live_bytes: Arc::new(AtomicU64::new(0)),
            copied: AtomicU64::new(0),
            submitted: AtomicU64::new(0),
            next_copy: Mutex::new(CopyOutcome::Succeed),
            drain_fails: Mutex::new(false),
            replay_address_checks: Arc::new(AtomicU64::new(0)),
            eager_single_rows: Arc::new(AtomicU64::new(0)),
            probes: Arc::new(Mutex::new(Vec::new())),
        })
    }
    pub fn probes(&self) -> Vec<[Probe; 3]> {
        self.probes.lock().unwrap().clone()
    }
    fn receipt(outcome: CopyOutcome) -> DeviceTerminalReceipt<TestRuntimeError> {
        DeviceTerminalReceipt::unprofiled(match outcome {
            CopyOutcome::Succeed => DeviceTerminal::Succeeded,
            _ => DeviceTerminal::FailedButQuiescent(TestRuntimeError(
                "injected selected copy failure",
            )),
        })
    }
}
impl DeviceRuntime for Runtime {
    type Buffer = Buffer;
    type Stream = Stream;
    type Command = Command;
    type Fence = Fence;
    type Error = TestRuntimeError;
    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Buffer, TestRuntimeError> {
        let request = permit.into_request();
        let descriptor = BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        };
        self.live_bytes
            .fetch_add(descriptor.size_bytes, Ordering::SeqCst);
        Ok(Buffer(Arc::new(Allocation {
            bytes: Mutex::new(vec![0xcd; descriptor.size_bytes as usize]),
            descriptor,
            live_bytes: self.live_bytes.clone(),
        })))
    }
    fn buffer_descriptor(&self, buffer: &Buffer) -> BufferDescriptor {
        buffer.0.descriptor.clone()
    }
    fn create_stream(&self) -> Result<Stream, TestRuntimeError> {
        Ok(Stream {
            state: StreamState::Ready,
        })
    }
    fn stream_state(&self, stream: &Stream) -> StreamState {
        stream.state
    }
    fn encode_copy(
        &self,
        source: &Buffer,
        destination: &Buffer,
        region: CopyRegion,
    ) -> Result<Command, TestRuntimeError> {
        region
            .validate_bounds(&source.0.descriptor, &destination.0.descriptor)
            .map_err(|_| TestRuntimeError("copy bounds"))?;
        Ok(Command::Copy(source.clone(), destination.clone(), region))
    }
    fn encode_upload(
        &self,
        _: &[u8],
        _: HostTransferLayout,
        _: &Buffer,
        _: u64,
    ) -> Result<Command, TestRuntimeError> {
        Err(TestRuntimeError("fixture requires managed input upload"))
    }
    fn supports_managed_tensor_io(&self) -> bool {
        true
    }
    fn supports_managed_static_weight_sources(&self) -> bool {
        true
    }
    fn encode_managed_upload(
        &self,
        source: ManagedHostBytes,
        layout: HostTransferLayout,
        destination: &Buffer,
        offset: u64,
    ) -> Result<Option<Command>, TestRuntimeError> {
        assert_eq!(source.len() as u64, layout.byte_len().unwrap());
        assert!(offset + source.len() as u64 <= destination.0.descriptor.size_bytes);
        Ok(Some(Command::Upload(
            source,
            destination.clone(),
            offset as usize,
        )))
    }
    fn encode_zero(
        &self,
        destination: &Buffer,
        offset: u64,
        length: u64,
    ) -> Result<Command, TestRuntimeError> {
        assert!(offset + length <= destination.0.descriptor.size_bytes);
        Ok(Command::Zero(
            destination.clone(),
            offset as usize..(offset + length) as usize,
        ))
    }
    fn submit(
        &self,
        stream: &mut Stream,
        commands: DeviceCommandBatch<Command>,
    ) -> Result<Fence, DefinitelyNotSubmitted<TestRuntimeError>> {
        assert!(!commands.is_empty());
        self.submitted.fetch_add(1, Ordering::SeqCst);
        let commands = commands.into_commands();
        let has_copy = commands.iter().any(|c| matches!(c, Command::Copy(..)));
        let outcome = if has_copy {
            std::mem::take(&mut *self.next_copy.lock().unwrap())
        } else {
            CopyOutcome::Succeed
        };
        let mut copies = 0;
        for command in &commands {
            match command {
                Command::Copy(src, dst, r) => {
                    if outcome != CopyOutcome::Succeed && copies > 0 {
                        break;
                    }
                    let source = src.0.bytes.lock().unwrap()[r.source_offset_bytes() as usize
                        ..(r.source_offset_bytes() + r.length_bytes()) as usize]
                        .to_vec();
                    dst.0.bytes.lock().unwrap()[r.destination_offset_bytes() as usize
                        ..(r.destination_offset_bytes() + r.length_bytes()) as usize]
                        .copy_from_slice(&source);
                    copies += 1;
                    self.copied.fetch_add(1, Ordering::SeqCst);
                }
                Command::Upload(src, dst, offset) => dst.0.bytes.lock().unwrap()
                    [*offset..*offset + src.len()]
                    .copy_from_slice(src.as_ref()),
                Command::Zero(dst, range) => dst.0.bytes.lock().unwrap()[range.clone()].fill(0),
                Command::Binding(writes) => {
                    for (destination, bytes) in writes {
                        destination.write(bytes);
                    }
                }
                Command::Compute(members) => {
                    for member in members {
                        member.execute();
                    }
                }
                Command::Output(members) => {
                    for member in members {
                        let input = floats(&member.input.read());
                        let selector = floats(&member.selector.read());
                        member.output.write(&bytes(
                            &input
                                .iter()
                                .zip(selector)
                                .map(|(a, b)| a + b)
                                .collect::<Vec<_>>(),
                        ));
                    }
                }
            }
        }
        stream.state = StreamState::Submitted;
        Ok(Fence {
            outcome,
            _commands: commands,
        })
    }
    fn query_fence(&self, fence: &Fence) -> FenceQuery<TestRuntimeError> {
        if fence.outcome == CopyOutcome::Indeterminate {
            FenceQuery::Pending
        } else {
            FenceQuery::Terminal(Self::receipt(fence.outcome))
        }
    }
    fn wait_fence(
        &self,
        fence: &Fence,
    ) -> Result<DeviceTerminalReceipt<TestRuntimeError>, FenceIndeterminate<TestRuntimeError>> {
        if fence.outcome == CopyOutcome::Indeterminate {
            Err(FenceIndeterminate::new(TestRuntimeError(
                "unknown selected copy",
            )))
        } else {
            Ok(Self::receipt(fence.outcome))
        }
    }
    fn synchronize(&self, stream: &mut Stream) -> Result<(), TestRuntimeError> {
        if *self.drain_fails.lock().unwrap() {
            return Err(TestRuntimeError("unknown drain"));
        }
        stream.state = StreamState::Ready;
        Ok(())
    }
    fn readback(
        &self,
        _: &mut Stream,
        source: &Buffer,
        region: CopyRegion,
        layout: HostTransferLayout,
    ) -> Result<Vec<u8>, TestRuntimeError> {
        assert_eq!(layout.byte_len().unwrap(), region.length_bytes());
        Ok(
            source.0.bytes.lock().unwrap()[region.source_offset_bytes() as usize
                ..(region.source_offset_bytes() + region.length_bytes()) as usize]
                .to_vec(),
        )
    }
    fn readback_managed_into(
        &self,
        stream: &mut Stream,
        source: &Buffer,
        region: CopyRegion,
        layout: HostTransferLayout,
        destination: &mut ManagedHostBuffer,
    ) -> Result<bool, TestRuntimeError> {
        let data = self.readback(stream, source, region, layout)?;
        destination.as_mut()[region.destination_offset_bytes() as usize
            ..region.destination_offset_bytes() as usize + data.len()]
            .copy_from_slice(&data);
        Ok(true)
    }
    fn describe_error(&self, error: &TestRuntimeError) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("selected_fixture", error.to_string(), false)
    }
}

fn tensor(dimensions: &[u64], element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: dimensions.to_vec(),
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn states() -> Vec<StateSpec> {
    [(A, 2, false), (B, 3, false), (KV, 1, true)]
        .into_iter()
        .map(|(value, width, prefix)| StateSpec {
            id: id(format!("state.{value}")),
            value_id: id(value),
            tensor: tensor(&[width], ElementType::F32),
            lifetime: StateLifetime::Sequence,
            capacity_demand: if prefix {
                StateCapacityDemand::TokenScaled {
                    bytes_per_token: 4,
                    maximum_tokens: 16,
                }
            } else {
                StateCapacityDemand::FixedPerScope
            },
            initialization: StateInitialization::Zero,
            checkpoint: StateCheckpointCapability::CompletedBoundary(StateCheckpointContract::new(
                if prefix {
                    StateCheckpointContents::PrefixPositions
                } else {
                    StateCheckpointContents::BoundaryValue
                },
                CheckpointInputDependency::ExactTokenPrefix,
            )),
        })
        .collect()
}
struct Family {
    visual: bool,
    output_only: Option<bool>,
    terminal: bool,
}
impl ModelFamilyProvider for Family {
    type Config = TestConfig;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        static VISUAL_ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        static OUTPUT_ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        static BAD_OUTPUT_ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        if let Some(state_dependency) = self.output_only {
            if state_dependency {
                BAD_OUTPUT_ID.get_or_init(|| id("family.selected.fixture.output_state"))
            } else {
                OUTPUT_ID.get_or_init(|| id("family.selected.fixture.output_only"))
            }
        } else if self.visual {
            VISUAL_ID.get_or_init(|| id("family.selected.fixture.visual"))
        } else {
            ID.get_or_init(|| id("family.selected.fixture"))
        }
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }
    fn parse_config(&self, raw: &Value) -> Result<TestConfig, VNextError> {
        TestFamily.parse_config(raw)
    }
    fn validate_config_identity(&self, raw: &Value, config: &TestConfig) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &TestConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, config)
    }
    fn weights(&self, config: &TestConfig) -> Result<ModelWeights, VNextError> {
        TestFamily.weights(config)
    }
    fn semantic_metadata(&self, config: &TestConfig) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
    fn numerical_profiles(&self, _: &TestConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        if self.output_only.is_some() {
            fixture_f32_profiles(
                self.family_id(),
                &[CA, CB, Y, FINAL],
                &[OP, OUTPUT_OP],
                states(),
            )
        } else {
            fixture_f32_profiles(self.family_id(), &[CA, CB, Y], &[OP], states())
        }
    }
    fn semantic_program(
        &self,
        _: &TestConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let mut external_inputs = vec![id(X)];
        let mut node_inputs = [X, A, B, KV, WEIGHT].map(id).to_vec();
        let mut conditioning = BTreeSet::new();
        if self.visual {
            external_inputs.push(id(VISUAL));
            node_inputs.push(id(VISUAL));
            conditioning.insert(id(VISUAL));
        }
        if self.output_only.is_some() {
            external_inputs.push(id(SELECTOR));
        }
        if self.output_only == Some(true) {
            node_inputs.push(id(if self.terminal { FINAL } else { SELECTOR }));
        }
        let mut nodes = vec![ProgramNode {
            id: id(NODE),
            operation_id: id(OP),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(id(X), 0),
            inputs: node_inputs,
            outputs: [Y, CA, CB].map(id).to_vec(),
            attributes: BTreeMap::new(),
        }];
        if self.output_only.is_some() {
            let node = ProgramNode {
                id: id("node.output"),
                operation_id: id(OUTPUT_OP),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::tokens(
                    id(if self.terminal && self.output_only == Some(true) {
                        SELECTOR
                    } else {
                        Y
                    }),
                    0,
                ),
                inputs: if self.terminal && self.output_only == Some(true) {
                    vec![id(SELECTOR)]
                } else {
                    vec![id(Y), id(SELECTOR)]
                },
                outputs: vec![id(FINAL)],
                attributes: BTreeMap::new(),
            };
            if self.terminal && self.output_only == Some(true) {
                // Legal acyclic dataflow, but a terminal reduction is used by
                // the persistent-state node. The effects proof must reject it.
                nodes.insert(0, node);
            } else {
                nodes.push(node);
            }
        }
        if self.terminal {
            conditioning.insert(id(SELECTOR));
        }
        ModelProgram::new(
            self.family_id().clone(),
            external_inputs,
            vec![ProgramBlock {
                id: "block.selected".into(),
                nodes,
            }],
            states(),
            vec![WeightReference {
                weight_id: id("weight.matrix"),
                value_id: id(WEIGHT),
                tensor: tensor(&[4], ElementType::F32),
            }],
            vec![id(if self.output_only.is_some() { FINAL } else { Y })],
        )?
        .with_checkpoint_inputs(
            ProgramCheckpointInputs::new(id(X), conditioning)?.with_output_only_inputs(
                if self.output_only.is_some() && !self.terminal {
                    BTreeSet::from([id(SELECTOR)])
                } else {
                    BTreeSet::new()
                },
            )?,
        )
    }
}

struct Contract(OperationDescriptor);
impl OperationContract for Contract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.0
    }
    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if inputs == self.0.inputs && outputs == self.0.outputs {
            Ok(())
        } else {
            Err(VNextError::InvalidExecutionPlan {
                reason: "selected fixture signature".into(),
            })
        }
    }
}
fn contract(visual: bool) -> OperationDescriptor {
    let t = |dimensions, dtype, access| {
        TensorContract::new(
            dimensions,
            BTreeSet::from([dtype]),
            vec![LayoutConstraint::Contiguous],
            access,
            AliasPolicy::NoAlias,
        )
        .unwrap()
    };
    let mut op = crate::test_support::resource_support::operation();
    op.id = id(OP);
    op.inputs = vec![
        t(
            vec![DimensionConstraint::Symbol("tokens".into())],
            ElementType::U32,
            TensorAccess::Read,
        ),
        t(
            vec![DimensionConstraint::Exact(2)],
            ElementType::F32,
            TensorAccess::Read,
        ),
        t(
            vec![DimensionConstraint::Exact(3)],
            ElementType::F32,
            TensorAccess::Read,
        ),
        t(
            vec![DimensionConstraint::Exact(1)],
            ElementType::F32,
            TensorAccess::ReadWrite,
        ),
        t(
            vec![DimensionConstraint::Exact(4)],
            ElementType::F32,
            TensorAccess::Read,
        ),
    ];
    op.outputs = vec![
        t(
            vec![DimensionConstraint::Symbol("tokens".into())],
            ElementType::F32,
            TensorAccess::Write,
        ),
        t(
            vec![DimensionConstraint::Exact(2), DimensionConstraint::Exact(2)],
            ElementType::F32,
            TensorAccess::Write,
        ),
        t(
            vec![DimensionConstraint::Exact(2), DimensionConstraint::Exact(3)],
            ElementType::F32,
            TensorAccess::Write,
        ),
    ];
    if visual {
        op.inputs.push(t(
            vec![
                DimensionConstraint::Symbol("tokens".into()),
                DimensionConstraint::Exact(2),
            ],
            ElementType::F32,
            TensorAccess::Read,
        ));
    }
    op.resources.scratch = ResourcePresenceRequirement::Forbidden;
    op.resources.persistent = ResourcePresenceRequirement::Forbidden;
    op
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayProbe {
    Checked,
    UncheckedStatic,
    UncheckedDynamic,
    CapturedCandidate,
    OmittedCandidate,
    MissingBindingDeclaration,
    MissingBindingResource,
    ErrorAfterValid,
    UnavailableAfterValid,
    UncheckedAfterPriorWave,
}

struct Provider {
    descriptor: OperationProviderDescriptor,
    probes: Arc<Mutex<Vec<[Probe; 3]>>>,
    visual: bool,
    output_only: bool,
    replay_probe: Option<ReplayProbe>,
    replay_address_checks: Arc<AtomicU64>,
    eager_single_rows: Arc<AtomicU64>,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
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
        Ok(
            if self.replay_probe.is_some()
                && self.replay_probe != Some(ReplayProbe::MissingBindingResource)
            {
                estimate.with_binding(ProviderWorkspaceRequirement::new(
                    128,
                    16,
                    ProviderWorkspaceScope::Invocation,
                    ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                    DynamicStorageRequirement::contiguous(),
                )?)
            } else {
                estimate
            },
        )
    }
}
fn region(
    p: &OperationInvocation<'_, Buffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    offset: u64,
    length: u64,
) -> Region {
    let binding = p
        .bindings()
        .iter()
        .find(|b| b.role() == role && b.ordinal() == ordinal)
        .unwrap();
    let [component] = binding.storage().components() else {
        panic!("fixture contiguous binding")
    };
    let view = p
        .views()
        .iter()
        .find(|v| v.resource_id() == component.resource_id())
        .unwrap();
    let translated = view
        .translate(component.offset_bytes() + offset, length)
        .unwrap();
    let mut iter = translated.iter();
    let (buffer, range, retention) = iter.next().unwrap().buffer_and_physical_range();
    assert!(
        iter.next().is_none(),
        "preprovisioned fixture uses one physical extent"
    );
    Region {
        buffer: buffer.clone(),
        range: range.start as usize..range.end as usize,
        _retention: retention,
    }
}
impl OperationProvider<Runtime> for Provider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        let Some(probe) = self.replay_probe else {
            return Ok(ReusableExecutionTopology::EagerBoundary);
        };
        if request
            .work_shape()
            .participant_token_ranges()
            .iter()
            .all(|span| span.immediate_tokens() == 1)
        {
            self.eager_single_rows.fetch_add(1, Ordering::SeqCst);
            return Ok(ReusableExecutionTopology::EagerBoundary);
        }
        let dynamic = ReusableExecutionTopology::Dynamic(
            DeviceReusableExecutionTopologyFingerprint::static_program(),
        );
        if probe == ReplayProbe::UncheckedStatic {
            return Ok(ReusableExecutionTopology::Static);
        }
        if probe == ReplayProbe::UncheckedDynamic
            || (probe == ReplayProbe::UncheckedAfterPriorWave
                && request
                    .work_shape()
                    .participant_token_ranges()
                    .iter()
                    .any(|span| span.source_token_range().start > 0))
        {
            return Ok(dynamic);
        }
        let mut values = request
            .bindings()
            .iter()
            .map(|binding| {
                ReusableExecutionValueAddress::program_binding(binding.role(), binding.ordinal())
            })
            .collect::<Vec<_>>();
        let binding = [ReusableExecutionWorkspaceAddress::Binding];
        if matches!(
            probe,
            ReplayProbe::Checked
                | ReplayProbe::ErrorAfterValid
                | ReplayProbe::UnavailableAfterValid
                | ReplayProbe::UncheckedAfterPriorWave
        ) {
            assert!(matches!(
                request.reusable_address_scope(&values, &binding)?,
                Some(DeviceReusableAddressScope::ExecutionLane(_))
            ));
            self.replay_address_checks.fetch_add(1, Ordering::SeqCst);
            if matches!(
                probe,
                ReplayProbe::Checked | ReplayProbe::UncheckedAfterPriorWave
            ) {
                return Ok(dynamic);
            }
        }
        let mut workspaces = binding.as_slice();
        match probe {
            ReplayProbe::CapturedCandidate => {
                let value = values
                    .iter_mut()
                    .find(|value| {
                        **value
                            == ReusableExecutionValueAddress::program_binding(
                                ResolvedValueRole::Output,
                                1,
                            )
                    })
                    .unwrap();
                *value = ReusableExecutionValueAddress::captured(ResolvedValueRole::Output, 1);
            }
            ReplayProbe::OmittedCandidate | ReplayProbe::ErrorAfterValid => {
                values.retain(|value| {
                    *value
                        != ReusableExecutionValueAddress::program_binding(
                            ResolvedValueRole::Output,
                            1,
                        )
                });
            }
            ReplayProbe::MissingBindingDeclaration => workspaces = &[],
            ReplayProbe::MissingBindingResource => {}
            ReplayProbe::UnavailableAfterValid => {
                let value = values
                    .iter_mut()
                    .find(|value| {
                        **value
                            == ReusableExecutionValueAddress::program_binding(
                                ResolvedValueRole::Input,
                                3,
                            )
                    })
                    .unwrap();
                *value = ReusableExecutionValueAddress::captured(ResolvedValueRole::Input, 3);
            }
            _ => unreachable!(),
        }
        let rejected = request.reusable_address_scope(&values, workspaces);
        if probe == ReplayProbe::UnavailableAfterValid {
            assert_eq!(rejected?, None);
        } else {
            assert!(rejected.is_err());
        }
        // Deliberately ignore the failed/absent address proof. Core, rather
        // than this untrusted provider's cooperation, must reject the topology.
        Ok(dynamic)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, Buffer>,
    ) -> Result<EncodedDeviceOperation<Command>, OperationFailure> {
        if self.output_only {
            let mut members = Vec::new();
            for (i, p) in invocation.participants().iter().enumerate() {
                let span = &invocation.participant_token_ranges()[i];
                let mapped = |role, ordinal| {
                    if invocation
                        .binding_uses_packed_batch_coordinates(role, ordinal)
                        .unwrap()
                    {
                        span.immediate_token_range().start
                    } else {
                        span.source_token_range().start
                    }
                };
                members.push(OutputCompute {
                    input: region(
                        p,
                        ResolvedValueRole::Input,
                        0,
                        mapped(ResolvedValueRole::Input, 0) * 4,
                        span.immediate_tokens() * 4,
                    ),
                    selector: region(
                        p,
                        ResolvedValueRole::Input,
                        1,
                        mapped(ResolvedValueRole::Input, 1) * 4,
                        span.immediate_tokens() * 4,
                    ),
                    output: region(
                        p,
                        ResolvedValueRole::Output,
                        0,
                        mapped(ResolvedValueRole::Output, 0) * 4,
                        span.immediate_tokens() * 4,
                    ),
                });
            }
            return Ok(EncodedDeviceOperation::compute(Command::Output(members)));
        }
        let mut members = Vec::new();
        let mut probes = Vec::new();
        let mut binding_writes = Vec::new();
        for (i, p) in invocation.participants().iter().enumerate() {
            let span = &invocation.participant_token_ranges()[i];
            let source = span.source_token_range();
            let packed = span.immediate_token_range();
            let input_start = if invocation
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
                .unwrap()
            {
                packed.start
            } else {
                source.start
            };
            let output_start = if invocation
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
                .unwrap()
            {
                packed.start
            } else {
                source.start
            };
            let s = [
                region(p, ResolvedValueRole::Input, 1, 0, 8),
                region(p, ResolvedValueRole::Input, 2, 0, 12),
                region(p, ResolvedValueRole::Input, 3, 0, source.end * 4),
            ];
            probes.push(s.each_ref().map(Probe::new));
            let binding = invocation.participants()[0].binding_view().map(|view| {
                let translated = view.translate(i as u64 * 16, 16).unwrap();
                let mut regions = translated.iter();
                let (buffer, range, retention) =
                    regions.next().unwrap().buffer_and_physical_range();
                assert!(regions.next().is_none());
                let region = Region {
                    buffer: buffer.clone(),
                    range: range.start as usize..range.end as usize,
                    _retention: retention,
                };
                let mut metadata = [0; 16];
                metadata[..8].copy_from_slice(&source.start.to_le_bytes());
                metadata[8..].copy_from_slice(&span.immediate_tokens().to_le_bytes());
                binding_writes.push((region.clone(), metadata));
                region
            });
            members.push(Compute {
                input: region(
                    p,
                    ResolvedValueRole::Input,
                    0,
                    input_start * 4,
                    span.immediate_tokens() * 4,
                ),
                candidates: [
                    region(p, ResolvedValueRole::Output, 1, 0, 16),
                    region(p, ResolvedValueRole::Output, 2, 0, 24),
                ],
                output: region(
                    p,
                    ResolvedValueRole::Output,
                    0,
                    output_start * 4,
                    span.immediate_tokens() * 4,
                ),
                states: s,
                weight: region(p, ResolvedValueRole::Input, 4, 0, 16),
                visual: self.visual.then(|| {
                    let start = if invocation
                        .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 5)
                        .unwrap()
                    {
                        packed.start
                    } else {
                        source.start
                    };
                    region(
                        p,
                        ResolvedValueRole::Input,
                        5,
                        start * 8,
                        span.immediate_tokens() * 8,
                    )
                }),
                start: source.start as usize,
                binding,
            });
        }
        *self.probes.lock().unwrap() = probes;
        let operation = EncodedDeviceOperation::compute(Command::Compute(members));
        Ok(if binding_writes.is_empty() {
            operation
        } else {
            invocation.attach_binding_command(operation, Command::Binding(binding_writes))
        })
    }
}

pub struct Fixture {
    pub runtime: Arc<Runtime>,
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<Runtime>,
    pub resources: Arc<PlanRuntimeResources<Runtime>>,
    lane: Arc<ExecutionLane<Runtime>>,
    pub reaper: Arc<CompletionReaper<Runtime>>,
    pub host: HostMemoryAccount,
    replay_probe: Option<ReplayProbe>,
}
pub struct Wave {
    pub step: Arc<StepResourceLease<Runtime>>,
    pub batch: ExecutionBatchParticipants<Runtime>,
}
struct Parameters;
impl WeightComponentSource for Parameters {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        static ONES: [u8; 16] = [0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63];
        WeightComponentPayload::new(
            c,
            "weight.bin",
            "selected.fixture",
            vec![4],
            ElementType::F32,
            &ONES,
        )
    }
}
impl Fixture {
    pub fn new() -> Self {
        Self::with_effects(true).unwrap()
    }
    pub fn with_effects(include_kv: bool) -> Result<Self, VNextError> {
        Self::build(include_kv, false, None, false, None, None)
    }
    pub fn visual() -> Self {
        Self::build(true, true, None, false, None, None).unwrap()
    }
    pub fn with_output_only(state_dependency: bool) -> Result<Self, VNextError> {
        Self::build(true, false, Some(state_dependency), false, None, None)
    }
    pub fn with_terminal(state_dependency: bool) -> Result<Self, VNextError> {
        Self::build(true, false, Some(state_dependency), true, None, None)
    }
    pub fn with_workspace_policy() -> Result<Self, VNextError> {
        let workspace = ReusableExecutionPolicy::new(
            1,
            vec![ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("workspace.selected.eager")?,
                ReusableExecutionCapacity::new(2, 32, 32)?,
            )?],
        )?;
        Self::build(true, false, None, false, Some(workspace), None)
    }

    pub fn with_program_policy(
        preparation: ReusableExecutionProgramPreparation,
    ) -> Result<Self, VNextError> {
        Self::with_replay_probe(preparation, None)
    }

    pub fn with_replay_probe(
        preparation: ReusableExecutionProgramPreparation,
        replay_probe: Option<ReplayProbe>,
    ) -> Result<Self, VNextError> {
        let class = ReusableExecutionClassId::new("workspace.selected.compute")?;
        let programs = vec![ReusableExecutionProgramSpec::new(
            class.clone(),
            ReusableExecutionProgramShape::uniform_decode(2, 2)?,
        )?];
        let programs = match preparation {
            ReusableExecutionProgramPreparation::OnDemand => {
                ReusableExecutionProgramPolicy::exact_on_demand(programs)?
            }
            ReusableExecutionProgramPreparation::BeforeReadiness => {
                ReusableExecutionProgramPolicy::exact_startup_sealed(1, 1, 1, programs)?
            }
        };
        let workspace = ReusableExecutionPolicy::new(
            1,
            vec![ReusableExecutionBucketSpec::new(
                class,
                ReusableExecutionCapacity::new(2, 32, 32)?,
            )?],
        )?
        .with_program_policy(programs)?;
        Self::build(true, false, None, false, Some(workspace), replay_probe)
    }
    fn build(
        include_kv: bool,
        visual: bool,
        output_only: Option<bool>,
        terminal: bool,
        workspace: Option<ReusableExecutionPolicy>,
        replay_probe: Option<ReplayProbe>,
    ) -> Result<Self, VNextError> {
        let runtime = Runtime::new();
        let mut operation = contract(visual);
        if replay_probe.is_some() {
            operation.resources.binding = ResourcePresenceRequirement::Optional;
        }
        let token_float = |access| {
            TensorContract::new(
                vec![DimensionConstraint::Symbol("tokens".into())],
                BTreeSet::from([ElementType::F32]),
                vec![LayoutConstraint::Contiguous],
                access,
                AliasPolicy::NoAlias,
            )
            .unwrap()
        };
        if output_only == Some(true) {
            operation.inputs.push(token_float(TensorAccess::Read));
        }
        let mut ports = vec![
            ProviderCandidateStatePort::TailTwo {
                input_ordinal: 1,
                output_ordinal: 1,
                storage_profile: contiguous_storage_profile(),
            },
            ProviderCandidateStatePort::TailTwo {
                input_ordinal: 2,
                output_ordinal: 2,
                storage_profile: contiguous_storage_profile(),
            },
        ];
        if include_kv {
            ports.push(ProviderCandidateStatePort::PrefixAppendOnly {
                role: ResolvedValueRole::Input,
                ordinal: 3,
                storage_profile: contiguous_storage_profile(),
            });
        }
        let descriptor = OperationProviderDescriptor::new(
            id("provider.selected.fixture"),
            operation.id.clone(),
            operation.fingerprint()?,
            sha('b'),
            if replay_probe.is_some() {
                ProviderExecutionSemantics::bitwise_eager_and_replay()
            } else {
                ProviderExecutionSemantics::bitwise_eager_only()
            },
            ContractVersion::new(1, 0),
            runtime.descriptor.id.clone(),
            runtime.descriptor.capabilities.clone(),
            BTreeSet::from([id("weight-format.resource-dense")]),
            BTreeSet::new(),
            contiguous_storage_bindings(&operation),
            "resource-estimator.selected.fixture",
            ContractVersion::new(1, 0),
            sha('c'),
        )?
        .with_candidate_state_effects(ProviderCandidateStateEffects::PrefixCausal { ports });
        let mut contracts = vec![Box::new(Contract(operation)) as Box<dyn OperationContract>];
        let mut providers = vec![Box::new(Provider {
            descriptor,
            probes: runtime.probes.clone(),
            visual,
            output_only: false,
            replay_probe,
            replay_address_checks: runtime.replay_address_checks.clone(),
            eager_single_rows: runtime.eager_single_rows.clone(),
        }) as Box<dyn OperationProvider<Runtime>>];
        if output_only.is_some() {
            let mut output = contract(false);
            output.id = id(OUTPUT_OP);
            output.inputs = vec![
                token_float(TensorAccess::Read),
                token_float(TensorAccess::Read),
            ];
            if terminal && output_only == Some(true) {
                output.inputs.truncate(1);
            }
            output.outputs = vec![token_float(TensorAccess::Write)];
            let descriptor = OperationProviderDescriptor::new(
                id("provider.selected.output.fixture"),
                output.id.clone(),
                output.fingerprint()?,
                sha('e'),
                ProviderExecutionSemantics::bitwise_eager_only(),
                ContractVersion::new(1, 0),
                runtime.descriptor.id.clone(),
                runtime.descriptor.capabilities.clone(),
                BTreeSet::from([id("weight-format.resource-dense")]),
                BTreeSet::new(),
                contiguous_storage_bindings(&output),
                "resource-estimator.selected.output.fixture",
                ContractVersion::new(1, 0),
                sha('f'),
            )?
            .with_candidate_state_effects(if terminal {
                ProviderCandidateStateEffects::TerminalOnly
            } else {
                ProviderCandidateStateEffects::PrefixCausal { ports: vec![] }
            });
            contracts.push(Box::new(Contract(output)));
            providers.push(Box::new(Provider {
                descriptor,
                probes: runtime.probes.clone(),
                visual: false,
                output_only: true,
                replay_probe: None,
                replay_address_checks: runtime.replay_address_checks.clone(),
                eager_single_rows: runtime.eager_single_rows.clone(),
            }));
        }
        let registry = OperationRuntimeRegistry::new(contracts, providers)?;
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.selected.fixture"),
            ContractVersion::new(1, 0),
            sha('d'),
            runtime.descriptor.id.clone(),
            runtime.descriptor.capabilities.clone(),
        )?;
        let catalog = WeightMaterializerRegistry::identity_only()?.augment_catalog(
            registry.capability_catalog(runtime.descriptor.clone(), vec![engine])?,
        )?;
        let family = TypedFamilyRegistration::new(Family {
            visual,
            output_only,
            terminal,
        })
        .prepare_fixture(&json!({"width":4}))?;
        let policy = ResolvedRuntimePolicy::new(
            "runtime-policy.selected.fixture",
            ContractVersion::new(1, 0),
            SchedulingDiscipline::FirstReady,
            RuntimeMemoryPolicy {
                checkpoint_capacity: None,
                capacity_bytes: 1 << 20,
                reserve_bytes: 0,
                maximum_active_sequences: 2,
                dynamic_storage_profile_order: vec![contiguous_storage_profile()],
            },
            AdmissionPolicy {
                maximum_queue_depth: 2,
                maximum_scheduled_tokens: 32,
                sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
                sequence_backing_growth: Default::default(),
                allow_defer: true,
                cancellation_check_interval_steps: 1,
            },
            ferrum_types::AttentionExecutionPolicy::Portable,
            ExecutionDeterminismRequirement::BitwiseSameRuntime,
            workspace,
        )?;
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(X),
            tensor(&[16], ElementType::U32),
        )]))?;
        options.insert_tensor_spec(id(CA), tensor(&[2, 2], ElementType::F32))?;
        options.insert_tensor_spec(id(CB), tensor(&[2, 3], ElementType::F32))?;
        if visual {
            options.insert_tensor_spec(id(VISUAL), tensor(&[16, 2], ElementType::F32))?;
        }
        if output_only.is_some() {
            options.insert_tensor_spec(id(SELECTOR), tensor(&[16], ElementType::F32))?;
        }
        // Candidate liveness must come from the production candidate contract,
        // never retain_completion_value calls supplied by this fixture.
        let compilation = ProgramPlanCompiler::compile(
            &family,
            &catalog,
            &policy,
            &registry.planning(),
            &options,
        )?;
        let executable = compilation.executable();
        let providers = registry.bind_plan(executable)?;
        let plan = executable.execution_plan();
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                1 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 1 << 18,
                    transfer_staging_bytes: 1 << 18,
                    output_bytes: 1 << 18,
                    weight_backing_bytes: 1 << 18,
                },
            )
            .unwrap(),
        );
        let StaticProvisioning::Required(permit) = plan
            .provision_static(runtime.clone(), id("request.selected.provision"))?
            .into_provisioning()
        else {
            panic!("causal fixture requires real Stored weights")
        };
        let identity = ResourceTransactionIdentity::for_admission(
            permit.binding(),
            id("run.selected.init"),
            id("transaction.selected.init"),
        );
        let committed = ResourceTransaction::begin(
            RuntimeResourceDriver::new(runtime.clone())?,
            identity,
            permit,
        )?
        .reserve()
        .unwrap()
        .commit()
        .unwrap_or_else(|_| panic!("fixture static allocation failed"));
        // Upfront compiled maxima isolate settlement correctness from elastic
        // fragmentation. This neither changes ceilings nor claims growth tests.
        let growth = plan
            .payload()
            .memory()
            .dynamic_pools()
            .iter()
            .map(|p| {
                DynamicPoolGrowthRequest::new(
                    p.pool_id().clone(),
                    p.provisioning().maximum_resident_bytes(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        committed.maintenance_controller().grow_pools(growth)?;
        let initialized = committed
            .initialize_static_managed(
                &family,
                plan,
                &Parameters,
                &host,
                StaticInitializationPolicy::new(4096, 16)?,
            )
            .unwrap();
        let resources = initialized
            .into_plan_runtime()
            .unwrap_or_else(|_| panic!("fixture static handoff"));
        let lane = resources.create_execution_lane().unwrap();
        Ok(Self {
            runtime,
            compilation,
            providers,
            resources,
            lane,
            reaper: CompletionReaper::new(),
            host,
            replay_probe,
        })
    }
    pub fn session(&self, suffix: &str) -> Arc<SequenceSession<Runtime>> {
        let tokens = [1_u32; 16];
        let work =
            ResourceWorkShape::single(TokenSpanWork::from_token_ids(&tokens, 0..16).unwrap())
                .unwrap();
        let request = RequestResourceAdmissionRequest::new(
            work.clone(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let binding = self.resources.trusted_runtime_binding().unwrap();
        let RequestResourceAdmissionDecision::Admitted(request) = binding
            .try_admit_request(
                request,
                id(format!("run.selected.{suffix}")),
                id(format!("request.selected.{suffix}")),
            )
            .unwrap()
        else {
            panic!("preprovisioned request")
        };
        let SequenceResourceAdmissionDecision::Admitted(sequence) = request
            .try_admit_sequence(
                SequenceResourceAdmissionRequest::new(
                    work,
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
            )
            .unwrap()
        else {
            panic!("preprovisioned sequence")
        };
        sequence.open_session().unwrap()
    }
    pub fn try_step(
        &self,
        sessions: &[Arc<SequenceSession<Runtime>>],
        histories: &[&[u32]],
        starts: &[usize],
    ) -> Result<Wave, VNextError> {
        self.try_step_inner(sessions, histories, starts, None)
    }

    pub fn try_reusable_step(
        &self,
        sessions: &[Arc<SequenceSession<Runtime>>],
        histories: &[&[u32]],
        starts: &[usize],
    ) -> Result<Wave, VNextError> {
        let bucket = self
            .compilation
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .reusable_execution()
            .unwrap()
            .buckets()[0]
            .bucket()
            .bucket_id()
            .clone();
        self.try_step_inner(sessions, histories, starts, Some(bucket))
    }

    fn try_step_inner(
        &self,
        sessions: &[Arc<SequenceSession<Runtime>>],
        histories: &[&[u32]],
        starts: &[usize],
        bucket: Option<ReusableExecutionBucketId>,
    ) -> Result<Wave, VNextError> {
        let batch = ExecutionBatchParticipants::new(sessions.to_vec())?;
        let spans = histories
            .iter()
            .zip(starts)
            .map(|(tokens, start)| {
                TokenSpanWork::from_token_ids(tokens, *start..tokens.len())?
                    .with_checkpoint_tokens(Arc::from(*tokens))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(spans)?,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )?;
        if let Some(bucket) = bucket {
            request = request.with_reusable_execution_bucket(bucket);
        }
        match batch.try_begin_step(request, &self.lane)? {
            StepResourceAdmissionDecision::Admitted(step) => Ok(Wave { step, batch }),
            other => panic!("preprovisioned step: {:?}", std::mem::discriminant(&other)),
        }
    }
    pub fn compute(&self, wave: &Wave, histories: &[&[u32]]) {
        self.try_compute_with_visual(wave, histories, None).unwrap();
    }

    pub fn try_compute_with_visual(
        &self,
        wave: &Wave,
        histories: &[&[u32]],
        visual: Option<&ManagedHostBytes>,
    ) -> Result<(), String> {
        self.try_compute_with_visual_row(wave, histories, visual, None)
    }
    pub fn try_compute_with_visual_row(
        &self,
        wave: &Wave,
        histories: &[&[u32]],
        visual: Option<&ManagedHostBytes>,
        row: Option<u64>,
    ) -> Result<(), String> {
        let step = &wave.step;
        let StepSubmissionWaveAdmissionDecision::Prepared(prepared) = step
            .try_prepare_full_plan_submission_wave(
                Arc::new(step.work_shape().clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        else {
            panic!("preprovisioned wave")
        };
        if self.replay_probe == Some(ReplayProbe::MissingBindingResource) {
            assert_eq!(
                OperationDispatch::reusable_execution_program_id_for_wave(
                    self.providers.providers(),
                    self.compilation.executable(),
                    &prepared,
                    &self.lane,
                )
                .unwrap(),
                None,
                "without a Binding resource, this original wave cannot replay"
            );
        }
        let active = wave
            .batch
            .sessions()
            .iter()
            .map(|session| TrustedActiveSequenceBinding::from_session(session))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let executable = self.compilation.executable();
        let identity = OperationDispatch::bind_submission_wave_identity(
            executable,
            active.iter(),
            &prepared,
            &self.lane,
        )
        .unwrap();
        let mut uploads = histories
            .iter()
            .enumerate()
            .map(|(index, tokens)| {
                let span = step.work_shape().participant_token_ranges().unwrap()[index]
                    .source_token_range();
                let selected_tokens = &tokens[span.start as usize..span.end as usize];
                let mut storage = self
                    .host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, selected_tokens.len() * 4)
                    .unwrap();
                for (dst, token) in storage.as_mut().chunks_exact_mut(4).zip(selected_tokens) {
                    dst.copy_from_slice(&token.to_le_bytes());
                }
                SubmissionWaveInputUpload::managed_range(
                    id(NODE),
                    index as u32,
                    0,
                    span.start * 4,
                    HostTransferLayout::new(ElementType::U32, selected_tokens.len() as u64)
                        .unwrap(),
                    storage.freeze(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        if let Some(visual) = visual {
            assert_eq!(
                histories.len(),
                1,
                "visual-like fixture is one actual participant"
            );
            let span =
                step.work_shape().participant_token_ranges().unwrap()[0].source_token_range();
            let start = row.unwrap_or(span.start);
            let end = start + (span.end - span.start);
            uploads.push(
                SubmissionWaveInputUpload::managed_range(
                    id(NODE),
                    0,
                    5,
                    start * 8,
                    HostTransferLayout::new(ElementType::F32, (span.end - span.start) * 2).unwrap(),
                    visual.slice(start as usize * 8..end as usize * 8).unwrap(),
                )
                .unwrap(),
            );
        }
        let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
            self.providers.providers(),
            executable,
            &identity,
            active.iter(),
            DeviceTimingMode::Off,
            &uploads,
            prepared,
            &self.lane,
            &self.reaper,
        )
        .map_err(|e| format!("{e:?}"))?;
        let observation = handle.wait().unwrap();
        assert_succeeded(&observation);
        drop((observation, handle, identity, active, uploads));
        Ok(())
    }
    pub fn select(&self, wave: Wave, selections: Vec<SelectedPrefix>) -> StepRetirementReceipt {
        let prepared = wave
            .step
            .try_prepare_selected_state_commit(selections)
            .unwrap();
        let handle = match self.reaper.submit_selected_state_commit(prepared).unwrap() {
            SelectedStateSubmission::Submitted(handle) => handle,
            _ => panic!("selected submission unexpected"),
        };
        let observation = handle.wait().unwrap();
        assert_succeeded(&observation);
        drop((observation, handle));
        let Wave { step, batch } = wave;
        let receipt = step.try_retire_selected().unwrap();
        drop(batch);
        receipt
    }
    pub fn close(self, sessions: Vec<Arc<SequenceSession<Runtime>>>) {
        for session in sessions {
            session
                .try_abort_if_quiescent()
                .or_else(|_| session.try_abort())
                .unwrap();
            drop(session);
        }
        assert_eq!(self.reaper.retained_count(), 0);
        let Self {
            resources,
            lane,
            reaper,
            runtime,
            host,
            compilation,
            providers,
            replay_probe: _,
        } = self;
        drop((lane, reaper, compilation, providers));
        assert!(matches!(
            PlanRuntimeResources::close(resources)
                .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(
            runtime.live_bytes.load(Ordering::SeqCst),
            0,
            "all actual backing released"
        );
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            0,
            "managed input backing returned"
        );
    }
}
pub fn assert_succeeded(observation: &CompletionObservation) {
    let CompletionObservation::Terminal(receipt) = observation else {
        panic!("expected successful terminal: {observation:?}")
    };
    assert!(
        matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ),
        "{observation:?}"
    );
}
