pub(crate) use ferrum_interfaces::vnext::*;
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use serde_json::{json, Value};
pub(crate) use std::collections::{BTreeMap, BTreeSet};
pub(crate) use std::error::Error;
pub(crate) use std::fmt;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
pub(crate) use std::sync::{mpsc, Arc, Barrier, Mutex};
pub(crate) use std::time::{Duration, Instant};

pub(crate) const EXPECTED_LEGACY_AUTHORITY_CASES: usize = 13;
pub(crate) const EXPECTED_CANCEL_DISPATCH_CASES: usize = 16;
pub(crate) const EXPECTED_COMPLETION_CASES: usize = 200;
pub(crate) const EXPECTED_CASES: usize = 299;
pub(crate) const COMPLETION_DROP_TEST_WORKERS: usize = 1;
pub(crate) const MAX_COMPLETION_DROP_TEST_WORKERS: usize = 2;
pub(crate) const _: () = assert!(
    COMPLETION_DROP_TEST_WORKERS == 1
        && COMPLETION_DROP_TEST_WORKERS <= MAX_COMPLETION_DROP_TEST_WORKERS
);

pub(crate) fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

pub(crate) fn sha(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

pub(crate) fn one_token_span() -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[1], 0..1).unwrap()
}

pub(crate) fn one_token_work() -> ResourceWorkShape {
    ResourceWorkShape::single(one_token_span()).unwrap()
}

pub(crate) fn contiguous_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

pub(crate) const TEST_PAGED_BLOCK_BYTES: u64 = 16;

pub(crate) fn paged_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena {
            block_bytes: TEST_PAGED_BLOCK_BYTES,
        },
        DynamicStorageView::PagedRegions {
            block_bytes: TEST_PAGED_BLOCK_BYTES,
        },
    )
    .unwrap()
}

pub(crate) fn contiguous_storage_bindings(
    operation: &OperationDescriptor,
) -> Vec<ProviderStorageBindingRequirement> {
    storage_bindings(operation, TestStateProfile::none())
}

fn storage_bindings(
    operation: &OperationDescriptor,
    state_profile: TestStateProfile,
) -> Vec<ProviderStorageBindingRequirement> {
    operation
        .inputs
        .iter()
        .enumerate()
        .map(|(ordinal, _)| {
            let storage = if state_profile.token_scaled_state && ordinal == 2 {
                DynamicStorageRequirement::new(vec![paged_storage_profile()]).unwrap()
            } else {
                DynamicStorageRequirement::contiguous()
            };
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                ordinal as u32,
                storage,
            )
        })
        .chain(operation.outputs.iter().enumerate().map(|(ordinal, _)| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Output,
                ordinal as u32,
                DynamicStorageRequirement::contiguous(),
            )
        }))
        .collect()
}

pub(crate) fn check(passed: &mut usize, condition: bool) {
    assert!(condition);
    *passed += 1;
}

pub(crate) fn suppress_expected_panic_hook<T>(action: impl FnOnce() -> T) -> T {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action));
    std::panic::set_hook(previous);
    match outcome {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TestConfig {
    pub(crate) width: u64,
    #[serde(default)]
    pub(crate) zero_state: bool,
    #[serde(default)]
    pub(crate) token_scaled_state: bool,
    #[serde(default)]
    pub(crate) recurrent_state: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TestStateProfile {
    pub(crate) zero_state: bool,
    pub(crate) token_scaled_state: bool,
    pub(crate) recurrent_state: bool,
}

impl TestStateProfile {
    pub(crate) const fn none() -> Self {
        Self {
            zero_state: false,
            token_scaled_state: false,
            recurrent_state: false,
        }
    }

    pub(crate) const fn fixed_sequence() -> Self {
        Self {
            zero_state: true,
            token_scaled_state: false,
            recurrent_state: false,
        }
    }

    pub(crate) const fn token_scaled_sequence() -> Self {
        Self {
            zero_state: true,
            token_scaled_state: true,
            recurrent_state: false,
        }
    }

    pub(crate) const fn hybrid_sequence() -> Self {
        Self {
            zero_state: true,
            token_scaled_state: true,
            recurrent_state: true,
        }
    }

    pub(crate) const fn from_zero_state(zero_state: bool) -> Self {
        if zero_state {
            Self::fixed_sequence()
        } else {
            Self::none()
        }
    }
}

#[derive(Default)]
pub(crate) struct TestFamily;

impl ModelFamilyProvider for TestFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY.get_or_init(|| id("family.device-operation-contract"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.device-operation")])
    }

    fn validate_config_identity(
        &self,
        _raw: &Value,
        _config: &Self::Config,
    ) -> Result<(), VNextError> {
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.device-operation"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: TestConfig = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        if config.width != 4 {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "width".to_owned(),
                reason: "fixture requires width 4".to_owned(),
            });
        }
        Ok(config)
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.device-operation-composite"),
            layout_id: id("weight-layout.device-operation-composite"),
            version: ContractVersion::new(1, 0),
            components: vec![
                WeightComponentSpec {
                    id: id("weight.component.left"),
                    role: WeightComponentRole::Values,
                    external_names: vec!["weight.left.bin".to_owned()],
                    dimensions: vec![2],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                    required: true,
                },
                WeightComponentSpec {
                    id: id("weight.component.right"),
                    role: WeightComponentRole::Values,
                    external_names: vec!["weight.right.bin".to_owned()],
                    dimensions: vec![2],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                    required: true,
                },
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.matrix"),
                dimensions: vec![4],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Composite {
                    parts: vec![
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::Dense {
                                component_id: id("weight.component.left"),
                            }),
                            logical_offsets: vec![0],
                            extents: vec![2],
                        },
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::Dense {
                                component_id: id("weight.component.right"),
                            }),
                            logical_offsets: vec![2],
                            extents: vec![2],
                        },
                    ],
                },
                required: true,
            }],
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        if config.token_scaled_state && !config.zero_state {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "token-scaled test state requires the state binding".to_owned(),
            });
        }
        if config.recurrent_state && !config.token_scaled_state {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "recurrent test state requires token-scaled KV state".to_owned(),
            });
        }
        let mut main_inputs = vec![id("value.input"), id("value.weight")];
        let mut tail_inputs = vec![id("value.intermediate"), id("value.weight")];
        let mut states = if config.zero_state {
            main_inputs.push(id("value.state"));
            tail_inputs.push(id("value.state"));
            vec![StateSpec {
                id: id("state.device-operation"),
                value_id: id("value.state"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
                lifetime: StateLifetime::Sequence,
                capacity_demand: if config.token_scaled_state {
                    StateCapacityDemand::TokenScaled {
                        bytes_per_token: config.width,
                        maximum_tokens: 16,
                    }
                } else {
                    StateCapacityDemand::FixedPerScope
                },
                initialization: StateInitialization::Zero,
            }]
        } else {
            Vec::new()
        };
        if config.recurrent_state {
            main_inputs.push(id("value.recurrent-state"));
            tail_inputs.push(id("value.recurrent-state"));
            states.push(StateSpec {
                id: id("state.device-operation.recurrent"),
                value_id: id("value.recurrent-state"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
                lifetime: StateLifetime::Sequence,
                capacity_demand: StateCapacityDemand::FixedPerScope,
                initialization: StateInitialization::Zero,
            });
        }
        let main_work = if config.token_scaled_state {
            ProgramNodeWorkSpec::tokens(id("value.input"), 0)
        } else {
            ProgramNodeWorkSpec::Fixed
        };
        let tail_work = if config.token_scaled_state {
            ProgramNodeWorkSpec::tokens(id("value.intermediate"), 0)
        } else {
            ProgramNodeWorkSpec::Fixed
        };
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.main"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        work: main_work,
                        inputs: main_inputs,
                        outputs: vec![id("value.intermediate")],
                        attributes: BTreeMap::new(),
                    },
                    ProgramNode {
                        id: id("node.tail"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        work: tail_work,
                        inputs: tail_inputs,
                        outputs: vec![id("value.output")],
                        attributes: BTreeMap::new(),
                    },
                ],
            }],
            states,
            vec![WeightReference {
                weight_id: id("weight.matrix"),
                value_id: id("value.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }],
            vec![id("value.output")],
        )
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".to_owned(),
                source_file: "template.json".to_owned(),
                sha256: sha('a'),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: Some(1),
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: Some(0),
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        })
    }
}

pub(crate) fn tensor_contract(access: TensorAccess) -> TensorContract {
    typed_tensor_contract(ElementType::F32, access)
}

pub(crate) fn typed_tensor_contract(
    element_type: ElementType,
    access: TensorAccess,
) -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Exact(4)],
        BTreeSet::from([element_type]),
        vec![LayoutConstraint::Contiguous],
        access,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}

fn token_tensor_contract(access: TensorAccess) -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Symbol("tokens".to_owned())],
        BTreeSet::from([ElementType::F32]),
        vec![LayoutConstraint::Contiguous],
        access,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}

pub(crate) fn operation() -> OperationDescriptor {
    operation_with_zero_state(false)
}

pub(crate) fn operation_with_zero_state(zero_state: bool) -> OperationDescriptor {
    operation_with_resource_options(zero_state, ResourcePresenceRequirement::Forbidden)
}

fn operation_with_resource_options(
    zero_state: bool,
    scratch: ResourcePresenceRequirement,
) -> OperationDescriptor {
    operation_with_resource_options_and_work(zero_state, scratch, false)
}

fn operation_with_resource_options_and_work(
    zero_state: bool,
    scratch: ResourcePresenceRequirement,
    token_scaled_state: bool,
) -> OperationDescriptor {
    operation_with_resource_profile(
        TestStateProfile {
            zero_state,
            token_scaled_state,
            recurrent_state: false,
        },
        scratch,
    )
}

fn operation_with_resource_profile(
    state_profile: TestStateProfile,
    scratch: ResourcePresenceRequirement,
) -> OperationDescriptor {
    let mut inputs = vec![
        if state_profile.token_scaled_state {
            token_tensor_contract(TensorAccess::Read)
        } else {
            tensor_contract(TensorAccess::Read)
        },
        tensor_contract(TensorAccess::Read),
    ];
    if state_profile.zero_state {
        inputs.push(typed_tensor_contract(
            ElementType::U8,
            TensorAccess::ReadWrite,
        ));
    }
    if state_profile.recurrent_state {
        inputs.push(typed_tensor_contract(
            ElementType::U8,
            TensorAccess::ReadWrite,
        ));
    }
    OperationDescriptor {
        id: id("operation.main"),
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![if state_profile.token_scaled_state {
            token_tensor_contract(TensorAccess::Write)
        } else {
            tensor_contract(TensorAccess::Write)
        }],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch,
            binding: ResourcePresenceRequirement::Optional,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::from([id("capability.compute")]),
        },
        profile_phase: ProfilePhase::Decode,
    }
}

pub(crate) fn catalog() -> CapabilityCatalog {
    catalog_with_zero_state(false)
}

pub(crate) fn catalog_with_zero_state(zero_state: bool) -> CapabilityCatalog {
    catalog_with_resource_options(zero_state, ResourcePresenceRequirement::Forbidden)
}

fn catalog_with_resource_options(
    zero_state: bool,
    scratch: ResourcePresenceRequirement,
) -> CapabilityCatalog {
    catalog_with_resource_options_and_execution_semantics(
        zero_state,
        scratch,
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
    )
}

fn catalog_with_resource_options_and_execution_semantics(
    zero_state: bool,
    scratch: ResourcePresenceRequirement,
    execution_semantics: ProviderExecutionSemantics,
) -> CapabilityCatalog {
    catalog_with_resource_options_execution_semantics_and_storage(
        TestStateProfile::from_zero_state(zero_state),
        scratch,
        execution_semantics,
    )
}

fn catalog_with_resource_options_execution_semantics_and_storage(
    state_profile: TestStateProfile,
    scratch: ResourcePresenceRequirement,
    execution_semantics: ProviderExecutionSemantics,
) -> CapabilityCatalog {
    catalog_with_resource_options_execution_semantics_storage_and_operation_version(
        state_profile,
        scratch,
        execution_semantics,
        ContractVersion::new(1, 0),
    )
}

fn catalog_with_resource_options_execution_semantics_storage_and_operation_version(
    state_profile: TestStateProfile,
    scratch: ResourcePresenceRequirement,
    execution_semantics: ProviderExecutionSemantics,
    operation_version: ContractVersion,
) -> CapabilityCatalog {
    let mut operation = operation_with_resource_profile(state_profile, scratch);
    operation.version = operation_version;
    operation.validate().unwrap();
    let device_id: DeviceId = id("device.device-operation.0");
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.device-operation"),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        sha('c'),
        execution_semantics,
        operation_version,
        device_id.clone(),
        capabilities.clone(),
        BTreeSet::from([id("weight-format.device-operation-composite")]),
        BTreeSet::new(),
        storage_bindings(&operation, state_profile),
        "resource-estimator.device-operation",
        ContractVersion::new(1, 0),
        sha('b'),
    )
    .unwrap();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id.clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint: sha('d'),
            capabilities: capabilities.clone(),
            dynamic_storage_profiles: if state_profile.token_scaled_state {
                BTreeSet::from([contiguous_storage_profile(), paged_storage_profile()])
            } else {
                BTreeSet::from([contiguous_storage_profile()])
            },
        },
        vec![operation.clone()],
        BTreeMap::from([(operation.id.clone(), vec![provider])]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.device-operation"),
            ContractVersion::new(1, 0),
            sha('e'),
            device_id,
            capabilities,
        )
        .unwrap()],
    )
    .unwrap()
}

pub(crate) struct TestOperationContract {
    pub(crate) descriptor: OperationDescriptor,
}

impl OperationContract for TestOperationContract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.descriptor
    }

    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if inputs != self.descriptor.inputs || outputs != self.descriptor.outputs {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "test operation signature mismatch".to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderBehavior {
    Success,
    SplitPhases,
    ProgramBinding,
    ProgramBindingFirstNodeEagerBoundary,
    ProgramBindingWithScratchTail,
    ScratchOverwrite,
    ScratchZeroed,
    WrongIdentity,
    WrongPhase,
}

impl ProviderBehavior {
    fn uses_program_binding(self) -> bool {
        matches!(
            self,
            Self::ProgramBinding
                | Self::ProgramBindingFirstNodeEagerBoundary
                | Self::ProgramBindingWithScratchTail
        )
    }
}

#[derive(Default)]
pub(crate) struct ProviderTrace {
    pub(crate) reusable_topology_calls: u64,
    pub(crate) encode_calls: u64,
    pub(crate) reusable_binding_encode_calls: u64,
    pub(crate) last_participant_count: usize,
    pub(crate) last_work_sequences: u32,
    pub(crate) component_resources: BTreeSet<ResourceId>,
    pub(crate) view_resources: BTreeSet<ResourceId>,
    pub(crate) program_binding_slots: BTreeMap<usize, ResourceId>,
    pub(crate) program_binding_plan_hashes: BTreeSet<String>,
    pub(crate) program_binding_layout_fingerprints: BTreeSet<String>,
    pub(crate) program_binding_lane_slot_ids: BTreeSet<u64>,
    pub(crate) program_binding_lifetimes: Vec<AllocationLifetime>,
}

pub(crate) struct TestProvider {
    pub(crate) descriptor: OperationProviderDescriptor,
    pub(crate) behavior: Arc<Mutex<ProviderBehavior>>,
    pub(crate) trace: Arc<Mutex<ProviderTrace>>,
}

impl OperationResourceEstimator for TestProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        let behavior = *self.behavior.lock().unwrap();
        let scratch_policy = match behavior {
            ProviderBehavior::ProgramBindingWithScratchTail
                if request.node_id().as_str() == "node.tail" =>
            {
                Some(ProviderWorkspaceReusePolicy::OverwriteBeforeRead)
            }
            ProviderBehavior::ScratchOverwrite => {
                Some(ProviderWorkspaceReusePolicy::OverwriteBeforeRead)
            }
            ProviderBehavior::ScratchZeroed => Some(ProviderWorkspaceReusePolicy::ZeroBeforeUse),
            _ => None,
        };
        let scratch = scratch_policy
            .map(|reuse_policy| {
                ProviderWorkspaceRequirement::from_formula(
                    ProviderWorkspaceSizeFormula::tokens(16)?,
                    16,
                    ProviderWorkspaceScope::Invocation,
                    reuse_policy,
                    DynamicStorageRequirement::contiguous(),
                )
            })
            .transpose()?;
        let estimate = OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            scratch,
            None,
        );
        let node_uses_program_binding = behavior.uses_program_binding()
            && (!matches!(behavior, ProviderBehavior::ProgramBindingWithScratchTail)
                || request.node_id().as_str() == "node.main");
        if node_uses_program_binding {
            Ok(
                estimate.with_binding(ProviderWorkspaceRequirement::from_formula(
                    ProviderWorkspaceSizeFormula::actual_sequences(16)?,
                    16,
                    ProviderWorkspaceScope::Invocation,
                    ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                    DynamicStorageRequirement::contiguous(),
                )?),
            )
        } else {
            Ok(estimate)
        }
    }
}

impl OperationProvider<TestRuntime> for TestProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        self.trace.lock().unwrap().reusable_topology_calls += 1;
        match *self.behavior.lock().unwrap() {
            ProviderBehavior::ProgramBindingWithScratchTail
                if request.node_id().as_str() == "node.tail" =>
            {
                return if request.scratch_reusable_address_scope()?.is_some() {
                    Ok(ReusableExecutionTopology::Static)
                } else {
                    Ok(ReusableExecutionTopology::EagerBoundary)
                };
            }
            ProviderBehavior::ScratchOverwrite | ProviderBehavior::ScratchZeroed => {
                return if request.scratch_reusable_address_scope()?.is_some() {
                    Ok(ReusableExecutionTopology::Static)
                } else {
                    Ok(ReusableExecutionTopology::EagerBoundary)
                };
            }
            ProviderBehavior::ProgramBinding | ProviderBehavior::ProgramBindingWithScratchTail => {
                let program_bound_identity = if request.node_id().as_str() == "node.main" {
                    (ResolvedValueRole::Input, 0)
                } else {
                    (ResolvedValueRole::Output, 0)
                };
                let addresses = request
                    .bindings()
                    .iter()
                    .map(|binding| {
                        if (binding.role(), binding.ordinal()) == program_bound_identity {
                            ReusableExecutionValueAddress::program_binding(
                                binding.role(),
                                binding.ordinal(),
                            )
                        } else {
                            ReusableExecutionValueAddress::captured(
                                binding.role(),
                                binding.ordinal(),
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                let scope = request
                    .reusable_address_scope(
                        &addresses,
                        &[ReusableExecutionWorkspaceAddress::Binding],
                    )?
                    .ok_or_else(|| VNextError::InvalidExecutionPlan {
                        reason: "program-binding test captured address lacks reusable authority"
                            .to_owned(),
                    })?;
                if matches!(scope, DeviceReusableAddressScope::Plan) {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: "program-binding test unexpectedly used plan scope".to_owned(),
                    });
                }
            }
            ProviderBehavior::ProgramBindingFirstNodeEagerBoundary
                if request.node_id().as_str() == "node.main" =>
            {
                return Ok(ReusableExecutionTopology::EagerBoundary);
            }
            _ => return Ok(ReusableExecutionTopology::Static),
        }
        let mut fingerprint = [0_u8; 32];
        fingerprint[..8].copy_from_slice(&request.work_shape().immediate_tokens().to_le_bytes());
        let source_frontier = request
            .work_shape()
            .participant_token_ranges()
            .iter()
            .map(|range| range.source_token_range().end)
            .max()
            .unwrap_or_default();
        fingerprint[8..16].copy_from_slice(&source_frontier.to_le_bytes());
        fingerprint[16] = 2;
        Ok(ReusableExecutionTopology::Dynamic(
            DeviceReusableExecutionTopologyFingerprint::from_sha256(fingerprint),
        ))
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<EncodedDeviceOperation<TestCommand>, OperationFailure> {
        let mut trace = self.trace.lock().unwrap();
        trace.encode_calls += 1;
        trace.last_participant_count = invocation.participants().len();
        trace.last_work_sequences = invocation.work_shape().immediate_sequences();
        if let Some(binding) = invocation.program_binding() {
            trace
                .program_binding_slots
                .insert(binding.node_index(), binding.slot().resource_id().clone());
            trace
                .program_binding_plan_hashes
                .insert(binding.plan_hash().as_str().to_owned());
            trace
                .program_binding_layout_fingerprints
                .insert(binding.layout().fingerprint().to_owned());
            trace
                .program_binding_lane_slot_ids
                .insert(binding.lane_slot_identity().slot_id());
            trace
                .program_binding_lifetimes
                .push(binding.lane_slot_identity().lifetime());
        }
        let participant = &invocation.participants()[0];
        trace.component_resources = participant
            .bindings()
            .iter()
            .find(|binding| binding.value_id().as_str() == "value.weight")
            .unwrap()
            .storage()
            .components()
            .iter()
            .map(|component| component.resource_id().clone())
            .collect();
        trace.view_resources = participant
            .views()
            .iter()
            .map(|view| view.resource_id().clone())
            .collect();
        drop(trace);
        let participant_count = u32::try_from(invocation.participants().len()).unwrap();
        let token_count = invocation.work_shape().immediate_tokens();
        match *self.behavior.lock().unwrap() {
            ProviderBehavior::Success => Ok(EncodedDeviceOperation::compute(TestCommand::Provider)),
            ProviderBehavior::SplitPhases => {
                Ok(EncodedDeviceOperation::compute(TestCommand::Provider)
                    .with_dynamic_binding(TestCommand::DynamicBinding)
                    .with_result_binding(TestCommand::ResultBinding))
            }
            ProviderBehavior::ProgramBinding
            | ProviderBehavior::ProgramBindingFirstNodeEagerBoundary => Ok(invocation
                .attach_binding_command(
                    EncodedDeviceOperation::compute(TestCommand::Provider),
                    TestCommand::ProgramBinding,
                )),
            ProviderBehavior::ProgramBindingWithScratchTail => {
                if participant
                    .identity()
                    .parts()
                    .node_id
                    .as_ref()
                    .map(NodeId::as_str)
                    == Some("node.main")
                {
                    Ok(invocation.attach_binding_command(
                        EncodedDeviceOperation::compute(TestCommand::Provider),
                        TestCommand::ProgramBinding,
                    ))
                } else {
                    Ok(EncodedDeviceOperation::compute(
                        TestCommand::ScratchProvider,
                    ))
                }
            }
            ProviderBehavior::ScratchOverwrite | ProviderBehavior::ScratchZeroed => {
                let command = if participant_count == 1 {
                    TestCommand::ScratchProvider
                } else {
                    TestCommand::ScratchProviderWork(participant_count, token_count)
                };
                Ok(EncodedDeviceOperation::compute(command))
            }
            ProviderBehavior::WrongIdentity => {
                let mut parts = participant.identity().parts().clone();
                parts.request_id = id("request.provider.wrong");
                let identity = ExecutionIdentityEnvelope::new(parts).unwrap();
                Err(OperationFailure::new(
                    identity,
                    ProfilePhase::Decode,
                    "provider_failure",
                    "injected provider failure",
                    false,
                )
                .unwrap())
            }
            ProviderBehavior::WrongPhase => Err(OperationFailure::new(
                participant.identity().clone(),
                ProfilePhase::Prefill,
                "provider_failure",
                "injected provider failure",
                false,
            )
            .unwrap()),
        }
    }

    fn encode_reusable_execution_bindings(
        &self,
        invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<EncodedReusableExecutionBindings<TestCommand>, OperationFailure> {
        if !self.behavior.lock().unwrap().uses_program_binding() {
            return self
                .encode_selected(invocation)
                .map(EncodedReusableExecutionBindings::from_operation);
        }
        let mut trace = self.trace.lock().unwrap();
        trace.reusable_binding_encode_calls += 1;
        let binding = invocation
            .program_binding()
            .expect("program-binding behavior requires a compiled binding slot");
        trace
            .program_binding_slots
            .insert(binding.node_index(), binding.slot().resource_id().clone());
        trace
            .program_binding_plan_hashes
            .insert(binding.plan_hash().as_str().to_owned());
        trace
            .program_binding_layout_fingerprints
            .insert(binding.layout().fingerprint().to_owned());
        trace
            .program_binding_lane_slot_ids
            .insert(binding.lane_slot_identity().slot_id());
        trace
            .program_binding_lifetimes
            .push(binding.lane_slot_identity().lifetime());
        Ok(EncodedReusableExecutionBindings::empty()
            .with_program_binding(TestCommand::ProgramBinding))
    }
}

fn policy_with_reusable_execution(
    reusable_execution: Option<ReusableExecutionPolicy>,
) -> ResolvedRuntimePolicy {
    policy_with_reusable_execution_and_determinism(
        reusable_execution,
        ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay,
    )
}

fn policy_with_reusable_execution_and_determinism(
    reusable_execution: Option<ReusableExecutionPolicy>,
    execution_determinism: ExecutionDeterminismRequirement,
) -> ResolvedRuntimePolicy {
    policy_with_reusable_execution_determinism_and_storage(
        reusable_execution,
        execution_determinism,
        false,
    )
}

fn policy_with_reusable_execution_determinism_and_storage(
    reusable_execution: Option<ReusableExecutionPolicy>,
    execution_determinism: ExecutionDeterminismRequirement,
    paged_state: bool,
) -> ResolvedRuntimePolicy {
    ResolvedRuntimePolicy::new(
        "runtime-policy.device-operation",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 65_536,
            reserve_bytes: 128,
            maximum_active_sequences: 64,
            dynamic_storage_profile_order: if paged_state {
                vec![contiguous_storage_profile(), paged_storage_profile()]
            } else {
                vec![contiguous_storage_profile()]
            },
        },
        AdmissionPolicy {
            maximum_queue_depth: 8,
            maximum_scheduled_tokens: 4096,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        execution_determinism,
        reusable_execution,
    )
    .unwrap()
}

pub(crate) fn policy() -> ResolvedRuntimePolicy {
    policy_with_reusable_execution(None)
}

fn reusable_policy() -> (ResolvedRuntimePolicy, ReusableExecutionBucketSpec) {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("execution.device-operation").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let reusable_execution = ReusableExecutionPolicy::new(1, vec![bucket.clone()]).unwrap();
    (
        policy_with_reusable_execution(Some(reusable_execution)),
        bucket,
    )
}

pub(crate) fn resolved_tensor() -> ResolvedTensorSpec {
    resolved_tensor_for(ElementType::F32)
}

pub(crate) fn resolved_tensor_for(element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![4], element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

fn resolved_weight() -> ResolvedWeightBinding {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    ResolvedWeightBinding::from_schema(family.weight_schema(), &id("weight.matrix")).unwrap()
}

pub(crate) fn single_binding(
    value: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    usage: BufferUsage,
    resource: &str,
) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id(value),
        role,
        ordinal,
        resolved_tensor(),
        if role == ResolvedValueRole::Output {
            TensorAccess::Write
        } else {
            TensorAccess::Read
        },
        AliasPolicy::NoAlias,
        usage,
        None,
        ResolvedValueStorage::single(id(resource), 0, 16, ElementType::F32).unwrap(),
    )
    .unwrap()
}

pub(crate) fn node_values_for(
    input_value: &str,
    input_resource: &str,
    output_value: &str,
    output_resource: &str,
) -> Vec<ResolvedValueBinding> {
    vec![
        single_binding(
            input_value,
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            input_resource,
        ),
        ResolvedValueBinding::new(
            id("value.weight"),
            ResolvedValueRole::Input,
            1,
            resolved_tensor(),
            TensorAccess::Read,
            AliasPolicy::NoAlias,
            BufferUsage::Weights,
            Some(resolved_weight()),
            ResolvedValueStorage::composite(vec![
                ResolvedStorageComponent::new(
                    Some(id("weight.component.left")),
                    id("resource.weight.left"),
                    0,
                    8,
                    ElementType::F32,
                )
                .unwrap(),
                ResolvedStorageComponent::new(
                    Some(id("weight.component.right")),
                    id("resource.weight.right"),
                    0,
                    8,
                    ElementType::F32,
                )
                .unwrap(),
            ])
            .unwrap(),
        )
        .unwrap(),
        single_binding(
            output_value,
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            output_resource,
        ),
    ]
}

pub(crate) fn node_values_with_zero_state_for(
    input_value: &str,
    input_resource: &str,
    output_value: &str,
    output_resource: &str,
) -> Vec<ResolvedValueBinding> {
    node_values_with_state_profile_for(
        input_value,
        input_resource,
        output_value,
        output_resource,
        TestStateProfile::fixed_sequence(),
    )
}

pub(crate) fn node_values_with_state_profile_for(
    input_value: &str,
    input_resource: &str,
    output_value: &str,
    output_resource: &str,
    state_profile: TestStateProfile,
) -> Vec<ResolvedValueBinding> {
    let mut values = node_values_for(input_value, input_resource, output_value, output_resource);
    if state_profile.zero_state {
        values.insert(
            2,
            ResolvedValueBinding::new(
                id("value.state"),
                ResolvedValueRole::Input,
                2,
                resolved_tensor_for(ElementType::U8),
                TensorAccess::ReadWrite,
                AliasPolicy::NoAlias,
                BufferUsage::State,
                None,
                ResolvedValueStorage::single(id("resource.state"), 0, 4, ElementType::U8).unwrap(),
            )
            .unwrap(),
        );
    }
    if state_profile.recurrent_state {
        values.insert(
            3,
            ResolvedValueBinding::new(
                id("value.recurrent-state"),
                ResolvedValueRole::Input,
                3,
                resolved_tensor_for(ElementType::U8),
                TensorAccess::ReadWrite,
                AliasPolicy::NoAlias,
                BufferUsage::State,
                None,
                ResolvedValueStorage::single(id("resource.recurrent-state"), 0, 4, ElementType::U8)
                    .unwrap(),
            )
            .unwrap(),
        );
    }
    values
}

pub(crate) fn node_values() -> Vec<ResolvedValueBinding> {
    node_values_for(
        "value.input",
        "resource.input",
        "value.intermediate",
        "resource.intermediate",
    )
}

pub(crate) fn tail_node_values() -> Vec<ResolvedValueBinding> {
    node_values_for(
        "value.intermediate",
        "resource.intermediate",
        "value.output",
        "resource.output",
    )
}

#[derive(Debug)]
pub(crate) struct TestBuffer {
    pub(crate) descriptor: BufferDescriptor,
}

#[derive(Debug, Default)]
pub(crate) struct TestStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TestCommand {
    Provider,
    ScratchProvider,
    ScratchProviderWork(u32, u64),
    ReusableExecution,
    ProgramBinding,
    CoalescedProgramBinding,
    DynamicBinding,
    ResultBinding,
    Copy,
    Upload(u8, BufferUsage),
    Zero,
}

#[derive(Debug, Clone)]
pub(crate) struct TestFence(u64, DeviceTimingMode, Option<DeviceSubmissionAttribution>);

impl TestFence {
    fn terminal_receipt(
        &self,
        terminal: DeviceTerminal<TestRuntimeError>,
    ) -> DeviceTerminalReceipt<TestRuntimeError> {
        match self.1 {
            DeviceTimingMode::Off => DeviceTerminalReceipt::unprofiled(terminal),
            DeviceTimingMode::Completion => DeviceTerminalReceipt::profiled(
                terminal,
                DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(
                    1_000_000,
                )),
            ),
            DeviceTimingMode::Replay
            | DeviceTimingMode::Kernel
            | DeviceTimingMode::Verification => {
                let submission_timing = self
                    .2
                    .as_ref()
                    .map(|attribution| {
                        attribution
                            .commands()
                            .iter()
                            .map(|command| {
                                let start = u64::from(command.command_index()) * 100;
                                DeviceCommandExecutionTiming::new(
                                    command.command_index(),
                                    vec![DeviceExecutionInterval::new(
                                        DeviceExecutionIntervalKind::Compute,
                                        start,
                                        start + 10,
                                    )
                                    .unwrap()],
                                )
                                .unwrap()
                            })
                            .collect::<Vec<_>>()
                    })
                    .and_then(DeviceSubmissionExecutionTiming::new)
                    .map_or(
                        DeviceTimingMeasurement::Unavailable(
                            DeviceTimingUnavailableReason::BackendMeasurementFailed,
                        ),
                        DeviceTimingMeasurement::Measured,
                    );
                DeviceTerminalReceipt::profiled_with_submission_timing(
                    terminal,
                    DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(
                        1_000_000,
                    )),
                    submission_timing,
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TestRuntimeError(pub(crate) &'static str);

impl fmt::Display for TestRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for TestRuntimeError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum SubmitBehavior {
    #[default]
    Success,
    DefinitelyNotSubmitted,
    Panic,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum FenceBehavior {
    Pending,
    #[default]
    Succeeded,
    FailedButQuiescent,
    Indeterminate,
    Panic,
}

#[derive(Default)]
pub(crate) struct RuntimeTrace {
    pub(crate) allocation_calls: u64,
    pub(crate) submit_calls: u64,
    pub(crate) submitted_command_counts: Vec<usize>,
    pub(crate) submitted_command_phases: Vec<Vec<DeviceCommandPhase>>,
    pub(crate) submitted_command_node_indices: Vec<Vec<Option<u32>>>,
    pub(crate) submitted_commands: Vec<Vec<TestCommand>>,
    pub(crate) submitted_compute_path_requirements: Vec<DeviceComputePathRequirement>,
    pub(crate) submitted_attribution_requirements: Vec<DeviceSubmissionAttributionRequirement>,
    pub(crate) uploaded_payloads: Vec<Vec<u8>>,
    pub(crate) submitted_reusable_captures: Vec<Option<DeviceReusableExecutionCapture>>,
    pending_reusable_invocations: Vec<DeviceReusableExecutionInvocation>,
    pub(crate) scratch_bytes: BTreeMap<u32, u8>,
    pub(crate) scratch_observations: Vec<(u32, u8)>,
    pub(crate) program_binding_coalesce_calls: u64,
    pub(crate) program_binding_input_counts: Vec<usize>,
    pub(crate) readback_calls: u64,
    pub(crate) readback_lengths: Vec<u64>,
    pub(crate) readback_fill_pattern: Vec<u8>,
    pub(crate) readback_fill_index: usize,
    pub(crate) synchronize_calls: u64,
    pub(crate) wait_fence_calls: u64,
    pub(crate) tamper_buffer_descriptor: bool,
    pub(crate) drift_on_submit: bool,
    pub(crate) next_fence: u64,
    pub(crate) submit_behavior: SubmitBehavior,
    pub(crate) fence_behavior: FenceBehavior,
    pub(crate) fence_behaviors: BTreeMap<u64, FenceBehavior>,
    pub(crate) wait_fence_block: Option<(Arc<Barrier>, Arc<Barrier>)>,
    pub(crate) synchronize_fails: bool,
    pub(crate) stream_failed: bool,
    pub(crate) describe_error_panics: bool,
    pub(crate) static_weight_import_enabled: bool,
    pub(crate) static_weight_import_begin_calls: u64,
    pub(crate) static_weight_import_seal_calls: u64,
    pub(crate) imported_component_count: usize,
    pub(crate) imported_bytes: u64,
}

pub(crate) struct TestRuntime {
    pub(crate) descriptor: DeviceDescriptor,
    pub(crate) alternate_descriptor: DeviceDescriptor,
    pub(crate) use_alternate_descriptor: AtomicBool,
    pub(crate) descriptor_reads_until_drift: AtomicU64,
    pub(crate) trace: Arc<Mutex<RuntimeTrace>>,
}

fn test_submission_attribution(
    timing_mode: DeviceTimingMode,
    requirement: DeviceSubmissionAttributionRequirement,
    entries: &[DeviceCommandEntry<TestCommand>],
    reusable_invocations: Vec<DeviceReusableExecutionInvocation>,
) -> Result<Option<DeviceSubmissionAttribution>, TestRuntimeError> {
    if !timing_mode.kernel_attribution_enabled() && !requirement.logical_execution_path_required() {
        return Ok(None);
    }
    let mut reusable_invocations = reusable_invocations.into_iter();
    let mut rows = Vec::with_capacity(entries.len());
    let mut replayed_segments = Vec::new();
    for (command_index, entry) in entries.iter().enumerate() {
        let command_index = u32::try_from(command_index)
            .map_err(|_| TestRuntimeError("command index exceeds u32"))?;
        let invocation = matches!(entry.command(), TestCommand::ReusableExecution)
            .then(|| reusable_invocations.next())
            .flatten();
        let (native_op_id, execution_path, compute_dispatch_count, transfer_command_count) =
            match entry.command() {
                TestCommand::Provider => ("test_provider", DeviceExecutionPath::Eager, 1, 0),
                TestCommand::ScratchProvider | TestCommand::ScratchProviderWork(_, _) => {
                    ("test_scratch_provider", DeviceExecutionPath::Eager, 1, 0)
                }
                TestCommand::ReusableExecution => (
                    "test_reusable_execution",
                    DeviceExecutionPath::Replayed,
                    1,
                    0,
                ),
                TestCommand::ProgramBinding => {
                    ("test_program_binding", DeviceExecutionPath::Eager, 0, 1)
                }
                TestCommand::CoalescedProgramBinding => (
                    "test_coalesced_program_binding",
                    DeviceExecutionPath::Eager,
                    0,
                    1,
                ),
                TestCommand::DynamicBinding => {
                    ("test_dynamic_binding", DeviceExecutionPath::Eager, 0, 1)
                }
                TestCommand::ResultBinding => {
                    ("test_result_binding", DeviceExecutionPath::Eager, 0, 1)
                }
                TestCommand::Copy => ("test_copy", DeviceExecutionPath::Eager, 0, 1),
                TestCommand::Upload(_, _) => ("test_upload", DeviceExecutionPath::Eager, 0, 1),
                TestCommand::Zero => ("test_zero", DeviceExecutionPath::Eager, 0, 1),
            };
        let logical_work = entry.logical_work();
        let batching_form = invocation.as_ref().map_or_else(
            || logical_work.map_or(DeviceBatchingForm::Scalar, |work| work.batching_form()),
            |_| DeviceBatchingForm::ParticipantLoop,
        );
        let encoded_work = match entry.command() {
            TestCommand::ScratchProviderWork(participants, tokens) => {
                Some((*participants, *tokens))
            }
            _ => None,
        };
        let participant_count = invocation.as_ref().map_or_else(
            || {
                logical_work.map_or_else(
                    || encoded_work.map_or(u32::from(entry.node_index().is_some()), |work| work.0),
                    |work| work.participant_count(),
                )
            },
            DeviceReusableExecutionInvocation::participant_count,
        );
        let participant_start = invocation.as_ref().map_or_else(
            || logical_work.map_or(0, |work| work.participant_start()),
            |_| 0,
        );
        let token_count = invocation.as_ref().map_or_else(
            || {
                logical_work.map_or_else(
                    || encoded_work.map_or(u64::from(entry.node_index().is_some()), |work| work.1),
                    |work| work.token_count(),
                )
            },
            DeviceReusableExecutionInvocation::token_count,
        );
        let reusable_graph_node_count = invocation
            .as_ref()
            .map(|invocation| u64::from(invocation.segment().logical_command_count()));
        rows.push(
            DeviceNativeWorkAttribution::with_participant_range(
                command_index,
                entry.node_index(),
                entry.phase(),
                DeviceNativeOperationId::new(native_op_id)
                    .ok_or(TestRuntimeError("non-portable test native attribution"))?,
                execution_path,
                batching_form,
                participant_start,
                participant_count,
                token_count,
                compute_dispatch_count,
                transfer_command_count,
                reusable_graph_node_count,
            )
            .ok_or(TestRuntimeError("invalid test native attribution"))?,
        );

        if let Some(invocation) = invocation {
            let logical_commands = (0..invocation.segment().logical_command_count())
                .map(|ordinal| {
                    DeviceReplayedLogicalCommandAttribution::new(
                        ordinal,
                        invocation
                            .segment()
                            .start_node_index()
                            .checked_add(ordinal)?,
                        DeviceNativeOperationId::new("test_replayed_logical_command").unwrap(),
                        DeviceBatchingForm::ParticipantLoop,
                        invocation.participant_count(),
                        invocation.token_count(),
                        1,
                        0,
                        1,
                    )
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(TestRuntimeError("invalid test logical replay attribution"))?;
            let fingerprint = format!(
                "{:064x}",
                u64::from(invocation.segment().ordinal()).saturating_add(1)
            );
            replayed_segments.push(
                DeviceReplayedSegmentAttribution::new(
                    command_index,
                    invocation.program_id().clone(),
                    invocation.segment().clone(),
                    fingerprint,
                    logical_commands,
                )
                .ok_or(TestRuntimeError("invalid test replay segment attribution"))?,
            );
        }
    }
    if reusable_invocations.next().is_some() {
        return Err(TestRuntimeError("unused reusable invocation metadata"));
    }
    DeviceSubmissionAttribution::with_replayed_segments(rows, replayed_segments)
        .map(Some)
        .ok_or(TestRuntimeError("invalid test submission attribution"))
}

struct TestStaticWeightImport {
    trace: Arc<Mutex<RuntimeTrace>>,
    components: Vec<(ResourceId, u64, u64)>,
}

impl StaticWeightImportSession<TestBuffer, TestRuntimeError> for TestStaticWeightImport {
    fn import_component(
        &mut self,
        payload: &WeightComponentPayload<'_>,
        destination: &TestBuffer,
        destination_offset_bytes: u64,
    ) -> Result<(), TestRuntimeError> {
        let length_bytes = u64::try_from(payload.bytes().len())
            .map_err(|_| TestRuntimeError("import payload exceeds u64"))?;
        let end = destination_offset_bytes
            .checked_add(length_bytes)
            .ok_or(TestRuntimeError("import range overflows"))?;
        if destination.descriptor.usage != BufferUsage::Weights
            || destination.descriptor.element_type != payload.element_type()
            || end > destination.descriptor.size_bytes
        {
            return Err(TestRuntimeError("import range differs from destination"));
        }
        self.components.push((
            destination.descriptor.resource_id.clone(),
            destination_offset_bytes,
            length_bytes,
        ));
        Ok(())
    }

    fn seal(self: Box<Self>) -> Result<(), TestRuntimeError> {
        let imported_bytes = self
            .components
            .iter()
            .try_fold(0_u64, |total, (_, _, length)| total.checked_add(*length))
            .ok_or(TestRuntimeError("import byte count overflows"))?;
        let mut trace = self.trace.lock().unwrap();
        trace.static_weight_import_seal_calls += 1;
        trace.imported_component_count += self.components.len();
        trace.imported_bytes += imported_bytes;
        Ok(())
    }
}

impl DeviceRuntime for TestRuntime {
    type Buffer = TestBuffer;
    type Stream = TestStream;
    type Command = TestCommand;
    type Fence = TestFence;
    type Error = TestRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        if self
            .descriptor_reads_until_drift
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok_and(|remaining| remaining == 1)
        {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        if self.use_alternate_descriptor.load(Ordering::Acquire) {
            &self.alternate_descriptor
        } else {
            &self.descriptor
        }
    }

    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        self.trace.lock().unwrap().allocation_calls += 1;
        let request = permit.into_request();
        Ok(TestBuffer {
            descriptor: BufferDescriptor {
                resource_id: request.resource_id().clone(),
                size_bytes: request.size_bytes(),
                alignment_bytes: request.alignment_bytes(),
                usage: request.usage(),
                element_type: request.element_type(),
            },
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        let mut descriptor = buffer.descriptor.clone();
        if self.trace.lock().unwrap().tamper_buffer_descriptor {
            descriptor.size_bytes += 1;
        }
        descriptor
    }

    fn begin_static_weight_import(
        &self,
    ) -> Option<
        Result<Box<dyn StaticWeightImportSession<Self::Buffer, Self::Error> + '_>, Self::Error>,
    > {
        let mut trace = self.trace.lock().unwrap();
        if !trace.static_weight_import_enabled {
            return None;
        }
        trace.static_weight_import_begin_calls += 1;
        Some(Ok(Box::new(TestStaticWeightImport {
            trace: Arc::clone(&self.trace),
            components: Vec::new(),
        })))
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        Ok(TestStream)
    }

    fn stream_state(&self, _stream: &Self::Stream) -> StreamState {
        if self.trace.lock().unwrap().stream_failed {
            StreamState::Failed
        } else {
            StreamState::Ready
        }
    }

    fn encode_copy(
        &self,
        _source: &Self::Buffer,
        _destination: &Self::Buffer,
        _region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        Ok(TestCommand::Copy)
    }

    fn encode_upload(
        &self,
        source: &[u8],
        _source_layout: HostTransferLayout,
        destination: &Self::Buffer,
        _destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.trace
            .lock()
            .unwrap()
            .uploaded_payloads
            .push(source.to_vec());
        Ok(TestCommand::Upload(
            source.first().copied().unwrap_or(0),
            destination.descriptor.usage,
        ))
    }

    fn encode_zero(
        &self,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
        _length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(TestCommand::Zero)
    }

    fn coalesce_program_bindings(
        &self,
        commands: Vec<Self::Command>,
    ) -> Result<Vec<Self::Command>, Self::Error> {
        let mut trace = self.trace.lock().unwrap();
        trace.program_binding_coalesce_calls += 1;
        trace.program_binding_input_counts.push(commands.len());
        drop(trace);
        if commands.is_empty() {
            Ok(commands)
        } else if commands
            .iter()
            .all(|command| *command == TestCommand::ProgramBinding)
        {
            Ok(vec![TestCommand::CoalescedProgramBinding])
        } else {
            Err(TestRuntimeError("unexpected program binding command"))
        }
    }

    fn encode_reusable_execution(
        &self,
        invocation: DeviceReusableExecutionInvocation,
    ) -> Result<Option<Self::Command>, Self::Error> {
        self.trace
            .lock()
            .unwrap()
            .pending_reusable_invocations
            .push(invocation);
        Ok(Some(TestCommand::ReusableExecution))
    }

    fn submit(
        &self,
        _stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        assert!(!commands.is_empty(), "core must not submit an empty batch");
        let timing_mode = commands.timing_mode();
        let compute_path_requirement = commands.compute_path_requirement();
        let declared_eager_compute_node_count =
            commands.declared_eager_compute_node_indices().len();
        let declared_eager_compute_node_indices = commands
            .declared_eager_compute_node_indices()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let attribution_requirement = commands.attribution_requirement();
        let reusable_execution_capture = commands.reusable_execution_capture().cloned();
        let entries = commands.into_entries();
        let mut compute_command_count = 0_usize;
        let mut replayed_compute_command_count = 0_usize;
        let mut observed_eager_compute_node_indices = BTreeSet::new();
        let mut exact_boundary_shape = true;
        for entry in &entries {
            if entry.phase() == DeviceCommandPhase::Compute {
                compute_command_count += 1;
                if *entry.command() == TestCommand::ReusableExecution {
                    replayed_compute_command_count += 1;
                } else if compute_path_requirement
                    == DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
                {
                    exact_boundary_shape &= entry.node_index().is_some_and(|node_index| {
                        declared_eager_compute_node_indices.contains(&node_index)
                            && observed_eager_compute_node_indices.insert(node_index)
                    });
                }
            }
        }
        let reusable_invocations = {
            let mut trace = self.trace.lock().unwrap();
            if trace.pending_reusable_invocations.len() != replayed_compute_command_count {
                trace.pending_reusable_invocations.clear();
                return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                    "reusable invocation metadata differs from encoded commands",
                )));
            }
            std::mem::take(&mut trace.pending_reusable_invocations)
        };
        let compute_path_matches = match compute_path_requirement {
            DeviceComputePathRequirement::Adaptive => true,
            DeviceComputePathRequirement::EagerOnly => replayed_compute_command_count == 0,
            DeviceComputePathRequirement::ReplayedOnly => {
                compute_command_count > 0 && replayed_compute_command_count == compute_command_count
            }
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => {
                !declared_eager_compute_node_indices.is_empty()
                    && declared_eager_compute_node_count
                        == declared_eager_compute_node_indices.len()
                    && exact_boundary_shape
                    && replayed_compute_command_count > 0
                    && replayed_compute_command_count < compute_command_count
                    && observed_eager_compute_node_indices == declared_eager_compute_node_indices
            }
        };
        if !compute_path_matches {
            return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                "compute-path requirement mismatch",
            )));
        }
        let command_phases = entries.iter().map(DeviceCommandEntry::phase).collect();
        let command_node_indices = entries.iter().map(DeviceCommandEntry::node_index).collect();
        let attribution = test_submission_attribution(
            timing_mode,
            attribution_requirement,
            &entries,
            reusable_invocations,
        )
        .map_err(DefinitelyNotSubmitted::new)?;
        let scratch_events = entries
            .iter()
            .filter_map(|entry| {
                let node_index = entry.node_index()?;
                match (entry.phase(), entry.command()) {
                    (DeviceCommandPhase::Initialization, TestCommand::Zero) => {
                        Some((node_index, 0, false))
                    }
                    (
                        DeviceCommandPhase::Initialization,
                        TestCommand::Upload(value, BufferUsage::Scratch),
                    ) => Some((node_index, *value, false)),
                    (_, TestCommand::ScratchProvider | TestCommand::ScratchProviderWork(_, _)) => {
                        Some((node_index, 0xa5, true))
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        let commands = entries
            .into_iter()
            .map(DeviceCommandEntry::into_parts)
            .map(|(_, _, _, command)| command)
            .collect::<Vec<_>>();
        let command_count = commands.len();
        let (drift, behavior, fence) = {
            let mut trace = self.trace.lock().unwrap();
            trace.submit_calls += 1;
            trace.submitted_command_counts.push(command_count);
            trace.submitted_command_phases.push(command_phases);
            trace
                .submitted_command_node_indices
                .push(command_node_indices);
            trace.submitted_commands.push(commands);
            trace
                .submitted_compute_path_requirements
                .push(compute_path_requirement);
            trace
                .submitted_attribution_requirements
                .push(attribution_requirement);
            for (node_index, value, observe_before_write) in scratch_events {
                if observe_before_write {
                    let observed = *trace.scratch_bytes.get(&node_index).unwrap_or(&0xa5);
                    trace.scratch_observations.push((node_index, observed));
                }
                trace.scratch_bytes.insert(node_index, value);
            }
            trace
                .submitted_reusable_captures
                .push(reusable_execution_capture);
            trace.next_fence += 1;
            (
                trace.drift_on_submit,
                trace.submit_behavior,
                TestFence(trace.next_fence, timing_mode, attribution),
            )
        };
        match behavior {
            SubmitBehavior::DefinitelyNotSubmitted => {
                return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                    "definitely-not-submitted",
                )));
            }
            SubmitBehavior::Panic => panic!("injected submit panic"),
            SubmitBehavior::Success => {}
        }
        if drift {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        Ok(fence)
    }

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        fence.2.clone()
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        assert!(fence.0 > 0);
        let trace = self.trace.lock().unwrap();
        let behavior = trace
            .fence_behaviors
            .get(&fence.0)
            .copied()
            .unwrap_or(trace.fence_behavior);
        drop(trace);
        match behavior {
            FenceBehavior::Pending => FenceQuery::Pending,
            FenceBehavior::Succeeded => {
                FenceQuery::Terminal(fence.terminal_receipt(DeviceTerminal::Succeeded))
            }
            FenceBehavior::FailedButQuiescent => FenceQuery::Terminal(fence.terminal_receipt(
                DeviceTerminal::FailedButQuiescent(TestRuntimeError("terminal-failure")),
            )),
            FenceBehavior::Indeterminate => {
                FenceQuery::Indeterminate(TestRuntimeError("fence-indeterminate"))
            }
            FenceBehavior::Panic => panic!("injected query panic"),
        }
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        assert!(fence.0 > 0);
        let (behavior, block) = {
            let mut trace = self.trace.lock().unwrap();
            trace.wait_fence_calls += 1;
            let behavior = trace
                .fence_behaviors
                .get(&fence.0)
                .copied()
                .unwrap_or(trace.fence_behavior);
            (behavior, trace.wait_fence_block.take())
        };
        if let Some((entered, release)) = block {
            entered.wait();
            release.wait();
        }
        match behavior {
            FenceBehavior::Succeeded => Ok(fence.terminal_receipt(DeviceTerminal::Succeeded)),
            FenceBehavior::FailedButQuiescent => Ok(fence.terminal_receipt(
                DeviceTerminal::FailedButQuiescent(TestRuntimeError("terminal-failure")),
            )),
            FenceBehavior::Pending | FenceBehavior::Indeterminate => Err(FenceIndeterminate::new(
                TestRuntimeError("fence-indeterminate"),
            )),
            FenceBehavior::Panic => panic!("injected wait panic"),
        }
    }

    fn synchronize(&self, _stream: &mut Self::Stream) -> Result<(), Self::Error> {
        let mut trace = self.trace.lock().unwrap();
        trace.synchronize_calls += 1;
        if trace.synchronize_fails {
            Err(TestRuntimeError("synchronize-failed"))
        } else {
            Ok(())
        }
    }

    fn readback(
        &self,
        _stream: &mut Self::Stream,
        _source: &Self::Buffer,
        region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        let mut trace = self.trace.lock().unwrap();
        trace.readback_calls += 1;
        trace.readback_lengths.push(region.length_bytes());
        let fill_byte = if trace.readback_fill_pattern.is_empty() {
            0
        } else {
            let index = trace.readback_fill_index % trace.readback_fill_pattern.len();
            trace.readback_fill_index += 1;
            trace.readback_fill_pattern[index]
        };
        Ok(vec![fill_byte; output_layout.byte_len().unwrap() as usize])
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        assert!(
            !self.trace.lock().unwrap().describe_error_panics,
            "injected describe_error panic"
        );
        DeviceErrorReport::new("test_runtime", error.to_string(), false)
    }
}

#[derive(Default)]
pub(crate) struct DriverTrace {
    pub(crate) calls: u64,
}

pub(crate) struct TestDriver {
    pub(crate) runtime: Arc<TestRuntime>,
    pub(crate) trace: Arc<Mutex<DriverTrace>>,
}

impl fmt::Debug for TestDriver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TestDriver")
            .field("device", &self.runtime.descriptor().id)
            .finish_non_exhaustive()
    }
}

impl ResourceTransactionDriver for TestDriver {
    type Buffer = TestBuffer;
    type Runtime = TestRuntime;

    fn runtime(&self) -> &Arc<Self::Runtime> {
        &self.runtime
    }

    fn device_id(&self) -> &DeviceId {
        &self.runtime.descriptor.id
    }

    fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime.descriptor.runtime_implementation_fingerprint
    }

    fn device_capacity_bytes(&self) -> u64 {
        self.runtime.descriptor.total_memory_bytes
    }

    fn reserve_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.trace.lock().unwrap().calls += 1;
        Ok(())
    }

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure> {
        self.trace.lock().unwrap().calls += 1;
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        context
            .allocate(&request)
            .map_err(|_| resource_failure("allocation"))
    }

    fn compensate_reserve_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn compensate_commit_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
        _buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn rollback_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn release_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
        _buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn reconcile_commit_outcome(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _expected: &ResourceReservation,
        _actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn quarantine_transaction(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        drop(ownership);
        Ok(())
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        drop(ownership);
    }
}

pub(crate) fn resource_failure(code: &str) -> ResourceDriverFailure {
    ResourceDriverFailure::new(
        FailureEnvelope::new(FailureDomain::Resource, code, "resource failure", false).unwrap(),
    )
    .unwrap()
}

pub(crate) fn runtime(catalog: &CapabilityCatalog) -> (Arc<TestRuntime>, Arc<Mutex<RuntimeTrace>>) {
    let trace = Arc::new(Mutex::new(RuntimeTrace::default()));
    let descriptor = catalog.device().clone();
    let mut alternate_descriptor = descriptor.clone();
    alternate_descriptor.runtime_implementation_fingerprint = sha('f');
    (
        Arc::new(TestRuntime {
            descriptor,
            alternate_descriptor,
            use_alternate_descriptor: AtomicBool::new(false),
            descriptor_reads_until_drift: AtomicU64::new(0),
            trace: Arc::clone(&trace),
        }),
        trace,
    )
}

mod planning;
pub(crate) use planning::*;
