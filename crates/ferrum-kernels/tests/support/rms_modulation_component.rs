use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
pub const INPUT: &str = "value.rms_modulation.x";
pub const TABLE: &str = "value.rms_modulation.table";
pub const GAMMA: &str = "value.rms_modulation.gamma";
pub const BRANCH: &str = "value.rms_modulation.branch";
pub const HIDDEN: &str = "value.rms_modulation.hidden";
pub const OUTPUT: &str = "value.rms_modulation.output";
const PROFILE: &str = "profile.test.rms_modulation.block8192_vec8_warp16_fma";
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
    pub residual: bool,
}
impl Config {
    pub fn kind(self) -> RowSelectedRmsModulationKind {
        if self.residual {
            RowSelectedRmsModulationKind::Residual
        } else {
            RowSelectedRmsModulationKind::Normalize
        }
    }
    pub fn input_specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let mut s = BTreeMap::from([
            (id(INPUT), tensor(vec![self.rows, self.width])),
            (id(TABLE), tensor(vec![3, 6 * self.width])),
            (id(GAMMA), tensor(vec![self.width])),
        ]);
        if self.residual {
            s.insert(id(BRANCH), tensor(vec![self.rows, self.width]));
        }
        s
    }
    pub fn specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let mut s = self.input_specs();
        s.insert(id(OUTPUT), tensor(vec![self.rows, self.width]));
        if self.residual {
            s.insert(id(HIDDEN), tensor(vec![self.rows, self.width]));
        }
        s
    }
    pub fn attrs(self) -> BTreeMap<AttributeId, SemanticValue> {
        let mut a = RowSelectedRmsModulationProfile::Block8192Vec8Warp16Fma
            .attributes()
            .unwrap();
        for (key, value) in [
            (
                "epsilon",
                SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
            ),
            (
                "run_lengths",
                SemanticValue::Integers(vec![1; self.rows as usize]),
            ),
            (
                "parameter_rows",
                SemanticValue::Integers(
                    (0..self.rows)
                        .map(|r| if r == 0 { 2 } else { (r % 3) as i64 })
                        .collect(),
                ),
            ),
            ("shift_column", SemanticValue::Unsigned(0)),
            ("scale_column", SemanticValue::Unsigned(self.width)),
        ] {
            a.insert(id(key), value);
        }
        if self.residual {
            a.insert(id("gate_column"), SemanticValue::Unsigned(2 * self.width));
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
        ID.get_or_init(|| id("family.test.rms_modulation"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.rms_modulation")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw || c.rows < 2 || c.width == 0 || c.width > 8192
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
        Ok(id("metadata.test.rms_modulation"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut boundaries = BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]);
        if c.residual {
            boundaries.insert(id(HIDDEN), ElementType::Bf16);
        }
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
                    operation_id: id(c.kind().operation_id()),
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
        let mut inputs = vec![id(INPUT), id(TABLE), id(GAMMA)];
        let mut outputs = vec![id(OUTPUT)];
        if c.residual {
            inputs.push(id(BRANCH));
            outputs.insert(0, id(HIDDEN));
        }
        ModelProgram::new(
            self.family_id().clone(),
            inputs.clone(),
            vec![ProgramBlock {
                id: "block.rms_modulation".into(),
                nodes: vec![ProgramNode {
                    id: id("node.rms_modulation"),
                    operation_id: id(c.kind().operation_id()),
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
