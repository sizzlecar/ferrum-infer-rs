//! Only schema/planning fixtures: the existing resource device implements no
//! tensor numerics. Tests deliberately observe its managed-IO refusal.
use super::{f, *};
use serde_json::Value;
use std::collections::BTreeMap;

const OP: &str = "operation.component.lifecycle_fixture";
const INPUT: &str = "value.input";
const OUTPUT: &str = "value.output";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Config {
    pub stored: bool,
    pub persistent: bool,
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        ID.get_or_init(|| f::id("family.component.lifecycle_fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([f::id("metadata.component.lifecycle_fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw {
            return Err(invalid("noncanonical fixture config"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(f::id("metadata.component.lifecycle_fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone())
            .map_err(|e| invalid(format!("fixture config: {e}")))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, config: &Config) -> Result<ModelWeights, VNextError> {
        if config.stored {
            f::TestFamily.weights(&f::TestConfig { width: 4 })
        } else {
            Ok(ModelWeights::None)
        }
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        f::fixture_f32_profiles(self.family_id(), &[OUTPUT], &[OP], vec![])
    }
    fn semantic_program(
        &self,
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let mut inputs = vec![f::id(INPUT)];
        let mut weights = vec![];
        if config.stored {
            inputs.push(f::id("value.weight"));
            weights.push(WeightReference {
                weight_id: f::id("weight.matrix"),
                value_id: f::id("value.weight"),
                tensor: spec(),
            });
        }
        ModelProgram::new(
            self.family_id().clone(),
            vec![f::id(INPUT)],
            vec![ProgramBlock {
                id: "block.component.fixture".into(),
                nodes: vec![ProgramNode {
                    id: f::id("node.component.fixture"),
                    operation_id: f::id(OP),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: vec![f::id(OUTPUT)],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            weights,
            vec![f::id(OUTPUT)],
        )?
        .with_input_tensor_specs(BTreeMap::from([(f::id(INPUT), spec())]))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn spec() -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![4],
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
pub(super) fn options() -> ProgramPlanCompileOptions {
    ProgramPlanCompileOptions::new(BTreeMap::from([(f::id(INPUT), spec())])).unwrap()
}
pub(super) fn policy() -> TensorComponentPolicy {
    TensorComponentPolicy::new(4096, vec![f::contiguous_storage_profile()]).unwrap()
}
pub(super) fn family(config: &Config) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Family)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new("fixture.f32").unwrap(),
        )
        .unwrap()
}
pub(super) fn compile(config: &Config) -> CompiledTensorComponent<f::TestRuntime> {
    let (registry, catalog) = registry(config);
    CompiledTensorComponent::compile(
        family(config),
        &registry,
        &catalog,
        &policy(),
        &WeightMaterializerRegistry::identity_only().unwrap(),
        &options(),
    )
    .unwrap()
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
        if inputs != self.0.inputs || outputs != self.0.outputs {
            return Err(invalid("fixture signature"));
        }
        Ok(())
    }
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    persistent: bool,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        let persistent = if self.persistent {
            Some(ProviderWorkspaceRequirement::new(
                16,
                16,
                ProviderWorkspaceScope::Plan,
                ProviderWorkspaceReusePolicy::Preserve,
                DynamicStorageRequirement::contiguous(),
            )?)
        } else {
            None
        };
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            persistent,
        ))
    }
}
impl OperationProvider<f::TestRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        _: BatchedOperationInvocation<'_, f::TestBuffer>,
    ) -> Result<EncodedDeviceOperation<()>, OperationFailure> {
        unreachable!("fixture stops before execution")
    }
}

pub(super) fn registry(
    config: &Config,
) -> (OperationRuntimeRegistry<f::TestRuntime>, CapabilityCatalog) {
    // The existing fixture supplies a fresh isolated device capacity identity.
    let device = f::catalog().device().clone();
    let mut inputs = vec![f::tensor_contract(
        ElementType::F32,
        TensorAccess::Read,
        AliasPolicy::NoAlias,
    )];
    if config.stored {
        inputs.push(inputs[0].clone());
    }
    let descriptor = OperationDescriptor {
        id: f::id(OP),
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![f::tensor_contract(
            ElementType::F32,
            TensorAccess::Write,
            AliasPolicy::NoAlias,
        )],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Forbidden,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: if config.persistent {
                ResourcePresenceRequirement::Required
            } else {
                ResourcePresenceRequirement::Forbidden
            },
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::from([f::id("capability.compute")]),
        },
        profile_phase: ProfilePhase::Forward,
    };
    let provider = OperationProviderDescriptor::new(
        f::id("provider.component.lifecycle_fixture"),
        descriptor.id.clone(),
        descriptor.fingerprint().unwrap(),
        f::sha('c'),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        device.capabilities.clone(),
        if config.stored {
            BTreeSet::from([f::id("weight-format.resource-dense")])
        } else {
            BTreeSet::new()
        },
        BTreeSet::new(),
        f::contiguous_storage_bindings(&descriptor),
        "resource-estimator.component.fixture",
        ContractVersion::new(1, 0),
        f::sha('b'),
    )
    .unwrap()
    .with_fixed_tensor_work(true);
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(Contract(descriptor))],
        vec![Box::new(Provider {
            descriptor: provider,
            persistent: config.persistent,
        })],
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        f::id("provider.engine.component.fixture"),
        ContractVersion::new(1, 0),
        device.runtime_implementation_fingerprint.clone(),
        device.id.clone(),
        device.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry.capability_catalog(device, vec![engine]).unwrap();
    (registry, catalog)
}
