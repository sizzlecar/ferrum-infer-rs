use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
pub const INPUT: &str = "value.qk_single_round.x";
pub const COSINE: &str = "value.qk_single_round.cosine";
pub const GAMMA: &str = "value.qk_single_round.gamma";
pub const SINE: &str = "value.qk_single_round.sine";
pub const OUTPUT: &str = "value.qk_single_round.output";
const PROFILE: &str = "profile.test.qk_single_round.block8192_vec8_warp16_fma";
pub fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: e.to_string(),
    }
}
fn tensor(shape: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: shape,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub rows: u64,
    pub width: u64,
    pub head: u64,
    pub rot: u64,
}
impl Config {
    pub fn input_specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([
            (id(INPUT), tensor(vec![self.rows, self.width])),
            (id(GAMMA), tensor(vec![self.head])),
            (id(COSINE), tensor(vec![self.rows, self.rot])),
            (id(SINE), tensor(vec![self.rows, self.rot])),
        ])
    }
    pub fn specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let mut s = self.input_specs();
        s.insert(id(OUTPUT), tensor(vec![self.rows, self.width]));
        s
    }
    pub fn attrs(self) -> BTreeMap<AttributeId, SemanticValue> {
        let mut a = HeadRmsPartialRopeSingleRoundBf16Profile::Block128Warp4Fma
            .attributes()
            .unwrap();
        for (key, value) in [
            ("head_dim", SemanticValue::Unsigned(self.head)),
            ("rotary_dim", SemanticValue::Unsigned(self.rot)),
            (
                "epsilon",
                SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
            ),
        ] {
            a.insert(id(key), value);
        }
        a
    }
    pub fn managed(self, host: &HostMemoryAccount, name: &str, bytes: &[u8]) -> ManagedHostTensor {
        let s = self.input_specs().remove(&id(name)).unwrap();
        let mut b = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .unwrap();
        b.as_mut().copy_from_slice(bytes);
        ManagedHostTensor::new(
            id(name),
            ResolvedTensorSpec::new(s.dimensions, s.element_type, s.layout).unwrap(),
            b.freeze(),
        )
        .unwrap()
    }
    pub fn compile<R: DeviceRuntime>(
        self,
        runtime: &R,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        materializers: &WeightMaterializerRegistry,
    ) -> CompiledTensorComponent<R> {
        let family = TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(self).unwrap(), &id(PROFILE))
            .unwrap();
        let policy = TensorComponentPolicy::new(
            runtime.descriptor().total_memory_bytes,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        CompiledTensorComponent::compile(
            family,
            registry,
            catalog,
            &policy,
            materializers,
            &ProgramPlanCompileOptions::new(self.specs()).unwrap(),
        )
        .unwrap()
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.qk_single_round"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.qk_single_round")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw
            || c.rows == 0
            || c.width == 0
            || c.head == 0
            || c.head > 128
            || c.width % c.head != 0
            || c.rot > c.head
            || c.rot == 0
            || c.rot % 2 != 0
        {
            return Err(invalid("invalid fixture geometry"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.test.qk_single_round"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let boundaries = BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]);
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT),
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID),
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
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown fixture profile"));
        }
        let inputs = vec![id(INPUT), id(GAMMA), id(COSINE), id(SINE)];
        let outputs = vec![id(OUTPUT)];
        ModelProgram::new(
            self.family_id().clone(),
            inputs.clone(),
            vec![ProgramBlock {
                id: "block.qk_single_round".into(),
                nodes: vec![ProgramNode {
                    id: id("node.qk_single_round"),
                    operation_id: id(HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: outputs.clone(),
                    attributes: c.attrs(),
                }],
            }],
            vec![],
            vec![],
            outputs,
        )?
        .with_input_tensor_specs(c.input_specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
