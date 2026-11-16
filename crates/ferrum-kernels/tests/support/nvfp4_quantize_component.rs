//! One ordinary managed quantization node. Source and reference providers may
//! be selected by separate explicit registries without changing this graph.
use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

pub const INPUT: &str = "value.nvfp4.quantize.x";
pub const GLOBAL: &str = "value.nvfp4.quantize.global";
pub const PACKED: &str = "value.nvfp4.quantize.packed";
pub const SCALES: &str = "value.nvfp4.quantize.scales";
const PROFILE: &str = "profile.test.nvfp4.quantize.cuda_rcp_approx_ftz_v1";
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
    pub k: u64,
}
impl Config {
    pub fn specs(self) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let g = Nvfp4MatrixLayout::new(self.rows, self.k)?;
        Ok(BTreeMap::from([
            (
                id(INPUT),
                tensor(vec![self.rows, self.k], ElementType::Bf16),
            ),
            (id(GLOBAL), tensor(vec![1], ElementType::F32)),
            (
                id(PACKED),
                tensor(g.packed_shape().to_vec(), ElementType::U8),
            ),
            (
                id(SCALES),
                tensor(g.scale_shape().to_vec(), ElementType::U8),
            ),
        ]))
    }
    pub fn input_specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        self.specs()
            .unwrap()
            .into_iter()
            .filter(|(v, _)| v == &id(INPUT) || v == &id(GLOBAL))
            .collect()
    }
    pub fn managed(self, host: &HostMemoryAccount, name: &str, bytes: &[u8]) -> ManagedHostTensor {
        let spec = self
            .input_specs()
            .remove(&id(name))
            .expect("declared quantizer input");
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
            &ProgramPlanCompileOptions::new(self.specs()?)?,
        )
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.nvfp4.quantize"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.nvfp4.quantize")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("noncanonical quantizer fixture"));
        }
        for s in c.specs()?.values() {
            s.validate("quantizer fixture")?;
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.test.nvfp4.quantize"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(PACKED),
                boundaries: BTreeMap::from([
                    (id(PACKED), ElementType::U8),
                    (id(SCALES), ElementType::U8),
                ]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(NVFP4_QUANTIZE_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: None,
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
            return Err(invalid("unknown quantizer fixture profile"));
        }
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT), id(GLOBAL)],
            vec![ProgramBlock {
                id: "block.nvfp4.quantize".into(),
                nodes: vec![ProgramNode {
                    id: id("node.nvfp4.quantize"),
                    operation_id: id(NVFP4_QUANTIZE_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT), id(GLOBAL)],
                    outputs: vec![id(PACKED), id(SCALES)],
                    attributes: Nvfp4QuantizeProfile::CudaRcpApproxFtzV1.attributes()?,
                }],
            }],
            vec![],
            vec![],
            vec![id(PACKED), id(SCALES)],
        )?
        .with_input_tensor_specs(c.input_specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
