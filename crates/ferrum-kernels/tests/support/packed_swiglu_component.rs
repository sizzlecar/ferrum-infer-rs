//! One packed value-first SwiGLU component, shared by CPU and CUDA tests.
use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

pub const INPUT: &str = "value.packed_swiglu.x";
pub const OUTPUT: &str = "value.packed_swiglu.output";
fn profile_id(profile: PackedSwiGluBf16Profile) -> &'static str {
    match profile {
        PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound => {
            "profile.test.packed_swiglu.exp2_sigmoid.single_bf16.v1"
        }
        PackedSwiGluBf16Profile::SiluBf16ThenMultiply => {
            "profile.test.packed_swiglu.staged_silu.v1"
        }
    }
}
pub fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).expect("fixture ID")
}
fn invalid(s: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan { reason: s.into() }
}
fn tensor(shape: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: shape,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub rows: u64,
    pub width: u64,
    pub profile: PackedSwiGluBf16Profile,
}
impl Config {
    pub fn specs(self) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let packed = self
            .width
            .checked_mul(2)
            .ok_or_else(|| invalid("packed width overflow"))?;
        let result = BTreeMap::from([
            (
                id(INPUT),
                tensor(vec![self.rows, packed], ElementType::Bf16),
            ),
            (
                id(OUTPUT),
                tensor(vec![self.rows, self.width], ElementType::Bf16),
            ),
        ]);
        for spec in result.values() {
            spec.validate("packed SwiGLU fixture")?;
        }
        Ok(result)
    }

    pub fn input_specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        self.specs()
            .unwrap()
            .into_iter()
            .filter(|(v, _)| v == &id(INPUT))
            .collect()
    }
    pub fn managed(self, host: &HostMemoryAccount, name: &str, bytes: &[u8]) -> ManagedHostTensor {
        let spec = self
            .input_specs()
            .remove(&id(name))
            .expect("declared packed SwiGLU input");
        let mut owned = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .unwrap();
        owned.as_mut().copy_from_slice(bytes);
        ManagedHostTensor::new(
            id(name),
            ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap(),
            owned.freeze(),
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
        self.try_compile(runtime, registry, catalog, materializers)
            .unwrap()
    }
    pub fn try_compile<R: DeviceRuntime>(
        self,
        runtime: &R,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        materializers: &WeightMaterializerRegistry,
    ) -> Result<CompiledTensorComponent<R>, VNextError> {
        let family = TypedFamilyRegistration::new(Family).prepare_with_profile(
            &serde_json::to_value(self).unwrap(),
            &id(profile_id(self.profile)),
        )?;
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
            &ProgramPlanCompileOptions::new(self.specs()?)?,
        )
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.packed_swiglu"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.packed_swiglu")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("noncanonical packed SwiGLU fixture"));
        }
        for s in c.specs()?.values() {
            s.validate("packed SwiGLU fixture")?;
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.test.packed_swiglu"))
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
                id: id(profile_id(c.profile)),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(PACKED_SWIGLU_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![id(profile_id(c.profile))],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown packed SwiGLU fixture profile"));
        }
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT)],
            vec![ProgramBlock {
                id: "block.packed_swiglu".into(),
                nodes: vec![ProgramNode {
                    id: id("node.packed_swiglu"),
                    operation_id: id(PACKED_SWIGLU_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT)],
                    outputs: vec![id(OUTPUT)],
                    attributes: c.profile.attributes()?,
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(c.input_specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
