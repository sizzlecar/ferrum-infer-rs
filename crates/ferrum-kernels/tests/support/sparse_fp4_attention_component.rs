//! One sparse-attention node through ordinary family/prepare/session APIs.
use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
pub const INPUTS: [&str; 6] = [
    "value.vsa.q",
    "value.vsa.k",
    "value.vsa.v",
    "value.vsa.sq",
    "value.vsa.sk",
    "value.vsa.sv",
];
pub const OUTPUTS: [&str; 1] = ["value.vsa.y"];
pub const SCORES: &str = "value.vsa.scores";
const PROFILE: &str = "profile.test.vsa.checked_host.two_level_p";
pub fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).expect("fixture ID")
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub batches: u32,
    pub heads: u32,
    pub query_blocks: u32,
    pub key_blocks: u32,
    pub pattern_sha256: String,
    pub producer: Producer,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Producer {
    CheckedHost,
    RegionTopK,
}
impl Config {
    pub fn inputs(&self) -> Vec<&'static str> {
        let mut inputs = INPUTS.to_vec();
        if matches!(self.producer, Producer::RegionTopK) {
            inputs.push(SCORES);
        }
        inputs
    }
    fn operation(&self) -> &'static str {
        match self.producer {
            Producer::CheckedHost => SPARSE_FP4_ATTENTION_CHECKED_HOST_OPERATION_ID,
            Producer::RegionTopK => SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID,
        }
    }
    pub fn attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([
            (
                id("scale"),
                SemanticValue::Rational(CanonicalRational::new(1, 8)?),
            ),
            (
                id("pattern_sha256"),
                SemanticValue::Text(self.pattern_sha256.clone()),
            ),
        ]))
    }
    pub fn specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let [b, h, q, k] = [
            u64::from(self.batches),
            u64::from(self.heads),
            u64::from(self.query_blocks) * 128,
            u64::from(self.key_blocks) * 128,
        ];
        let shapes = [
            vec![b, h, q, 64],
            vec![b, h, k, 64],
            vec![b, h, 128, k / 2],
            vec![b, h, q, 8],
            vec![b, h, k, 8],
            vec![b, h, 128, k / 16],
            vec![b, h, q, 128],
        ];
        let mut specs: BTreeMap<_, _> = INPUTS
            .into_iter()
            .chain(OUTPUTS)
            .zip(shapes)
            .zip([ElementType::U8; 6].into_iter().chain([ElementType::Bf16]))
            .map(|((name, dimensions), element_type)| {
                (
                    id(name),
                    ProgramTensorSpec {
                        dimensions,
                        element_type,
                        layout: ResolvedTensorLayout::Contiguous,
                    },
                )
            })
            .collect();
        if matches!(self.producer, Producer::RegionTopK) {
            specs.insert(
                id(SCORES),
                ProgramTensorSpec {
                    dimensions: vec![b, h, k / 128, k / 128],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            );
        }
        specs
    }
    fn input_specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        self.specs()
            .into_iter()
            .filter(|(id, _)| self.inputs().contains(&id.as_str()))
            .collect()
    }
    pub fn managed(&self, host: &HostMemoryAccount, name: &str, bytes: &[u8]) -> ManagedHostTensor {
        let spec = self
            .input_specs()
            .remove(&id(name))
            .expect("declared VSA input");
        let mut owner = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .unwrap();
        owner.as_mut().copy_from_slice(bytes);
        ManagedHostTensor::new(
            id(name),
            ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap(),
            owner.freeze(),
        )
        .unwrap()
    }
    pub fn try_compile<R: DeviceRuntime>(
        &self,
        runtime: &R,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        materializers: &WeightMaterializerRegistry,
    ) -> Result<CompiledTensorComponent<R>, VNextError> {
        let family = TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(self).unwrap(), &id(PROFILE))?;
        let policy = TensorComponentPolicy::new(
            runtime.descriptor().total_memory_bytes,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )?;
        CompiledTensorComponent::compile(
            family,
            registry,
            catalog,
            &policy,
            materializers,
            &ProgramPlanCompileOptions::new(self.specs())?,
        )
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.vsa.checked_host"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.vsa.checked_host")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("noncanonical VSA fixture"));
        }
        let specs = c
            .specs()
            .into_iter()
            .map(|(id, s)| {
                Ok((
                    id,
                    ResolvedTensorSpec::new(s.dimensions, s.element_type, s.layout)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
        let inputs = INPUTS.map(|s| &specs[&id::<ProgramValueId>(s)]);
        let outputs = OUTPUTS.map(|s| &specs[&id::<ProgramValueId>(s)]);
        match c.producer {
            Producer::CheckedHost => {
                SparseFp4AttentionGeometry::new(inputs, outputs, &c.attributes()?)
            }
            Producer::RegionTopK => {
                let [q, k, v, sq, sk, sv] = inputs;
                SparseFp4AttentionGeometry::with_region_scores(
                    [q, k, v, sq, sk, sv, &specs[&id::<ProgramValueId>(SCORES)]],
                    outputs,
                    &c.attributes()?,
                )
            }
        }?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.test.vsa.checked_host"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUTS[0]),
                boundaries: BTreeMap::from([(id(OUTPUTS[0]), ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(c.operation()),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
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
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown VSA fixture profile"));
        }
        ModelProgram::new(
            self.family_id().clone(),
            c.inputs().into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.vsa".into(),
                nodes: vec![ProgramNode {
                    id: id("node.vsa"),
                    operation_id: id(c.operation()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: c.inputs().into_iter().map(id).collect(),
                    outputs: OUTPUTS.map(id).to_vec(),
                    attributes: c.attributes()?,
                }],
            }],
            vec![],
            vec![],
            OUTPUTS.map(id).to_vec(),
        )?
        .with_input_tensor_specs(c.input_specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
