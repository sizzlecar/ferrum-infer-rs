//! One head FP8 roundtrip component, shared by CPU and CUDA tests.
use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

pub const INPUT: &str = "value.head_fp8_roundtrip.x";
pub const INDICES: &str = "value.head_fp8_roundtrip.indices";
pub const OUTPUT: &str = "value.head_fp8_roundtrip.output";
const PROFILE: &str = "profile.test.head_fp8_roundtrip.amax_div448.zero_inverse.e4m3fn_rne_bf16.v1";
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
    pub batches: u64,
    pub heads: u64,
    pub rows: u64,
    pub width: u64,
    pub selected_rows: Option<u64>,
}
impl Config {
    pub fn operation(self) -> &'static str {
        if self.selected_rows.is_some() {
            INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID
        } else {
            HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID
        }
    }
    pub fn indices(self) -> Vec<i32> {
        (0..self.selected_rows.unwrap_or(0))
            .map(|i| match i % 9 {
                0 => self.rows as i32 - 1,
                1 | 4 => 2,
                2 => -1,
                3 => 0,
                5 => self.rows as i32,
                6 => 1,
                7 => i32::MIN,
                _ => (i % self.rows) as i32,
            })
            .collect()
    }
    pub fn specs(self) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let shape = vec![
            self.batches,
            self.rows,
            self.width
                .checked_mul(self.heads)
                .ok_or_else(|| invalid("head channels overflow"))?,
        ];
        let mut result = BTreeMap::from([
            (id(INPUT), tensor(shape.clone(), ElementType::Bf16)),
            (id(OUTPUT), tensor(shape, ElementType::Bf16)),
        ]);
        if let Some(rows) = self.selected_rows {
            result.insert(id(INDICES), tensor(vec![rows], ElementType::I32));
            result
                .get_mut(&id::<ProgramValueId>(OUTPUT))
                .unwrap()
                .dimensions[1] = rows;
        }
        let t = &result[&id::<ProgramValueId>(INPUT)];
        let r = ResolvedTensorSpec::new(t.dimensions.clone(), t.element_type, t.layout.clone())?;
        HeadFp8RoundtripBf16Geometry::new(
            &r,
            &r,
            &HeadFp8RoundtripBf16Geometry::attributes(self.width)?,
        )?;
        for spec in result.values() {
            spec.validate("head FP8 roundtrip fixture")?;
        }
        Ok(result)
    }

    pub fn input_specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        self.specs()
            .unwrap()
            .into_iter()
            .filter(|(v, _)| v != &id(OUTPUT))
            .collect()
    }
    pub fn managed(self, host: &HostMemoryAccount, name: &str, bytes: &[u8]) -> ManagedHostTensor {
        let spec = self
            .input_specs()
            .remove(&id(name))
            .expect("declared head FP8 roundtrip input");
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
        ID.get_or_init(|| id("family.test.head_fp8_roundtrip"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.head_fp8_roundtrip")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("noncanonical head FP8 roundtrip fixture"));
        }
        for s in c.specs()?.values() {
            s.validate("head FP8 roundtrip fixture")?;
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.test.head_fp8_roundtrip"))
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
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(c.operation()),
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
            return Err(invalid("unknown head FP8 roundtrip fixture profile"));
        }
        ModelProgram::new(
            self.family_id().clone(),
            c.input_specs().into_keys().collect(),
            vec![ProgramBlock {
                id: "block.head_fp8_roundtrip".into(),
                nodes: vec![ProgramNode {
                    id: id("node.head_fp8_roundtrip"),
                    operation_id: id(c.operation()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: if c.selected_rows.is_some() {
                        vec![id(INPUT), id(INDICES)]
                    } else {
                        vec![id(INPUT)]
                    },
                    outputs: vec![id(OUTPUT)],
                    attributes: HeadFp8RoundtripBf16Geometry::attributes(c.width)?,
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
