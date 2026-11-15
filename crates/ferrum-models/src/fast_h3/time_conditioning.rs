//! One block's cold-path time MLP and AdaLN projection over managed tensors.
//! The input is explicit F32 sinusoidal features. The output retains all three
//! modalities and six modulation fields, in `[time, modality, field, hidden]`
//! order. No sampling cursor, table cache, or device allocator lives here.
use super::{FastH3TransformerPackage, FastH3WeightStorage};
use ferrum_quantization::ManagedDenseArchiveSource;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod program;
mod table;
mod time_rows;
pub use table::FastH3ConditioningTable;
pub use time_rows::{FastH3ConditioningClock, FastH3ConditioningSchedule};

#[derive(Clone, Copy, Debug)]
pub struct FastH3TimeConditioningLimits {
    pub maximum_time_rows: u64,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3TimeConditioningConfig {
    block: usize,
    rows: u64,
    frequency: u64,
    intermediate: u64,
    embedding: u64,
    hidden: u64,
    components: [WeightComponentSpec; 6],
}

impl FastH3TimeConditioningConfig {
    /// The archive's original representation is normalized through the existing
    /// managed dense source. Time MLP parameters compute in F32 even when the
    /// downloaded source is BF16; AdaLN parameters and its output stay BF16.
    pub fn from_package(
        package: &FastH3TransformerPackage,
        block: usize,
        rows: u64,
        limits: FastH3TimeConditioningLimits,
    ) -> Result<Self, VNextError> {
        let c = package.config();
        if block >= c.layers() || c.adaln_rank().is_some() {
            return Err(invalid("requires a declared full-rank block"));
        }
        let mut components = Vec::with_capacity(6);
        for (index, name) in parameter_names(block).into_iter().enumerate() {
            let weight = package
                .weight(&name)
                .ok_or_else(|| invalid("missing time parameter"))?;
            let FastH3WeightStorage::Dense(source) = &weight.storage else {
                return Err(invalid("time conditioning parameters must remain dense"));
            };
            let dtype = if index < 4 {
                ElementType::F32
            } else {
                ElementType::Bf16
            };
            if weight.compute_element_type != dtype {
                return Err(invalid("time conditioning compute precision differs"));
            }
            let mut component = source.clone();
            component.encoding = WeightEncoding::Dense {
                element_type: dtype,
            };
            components.push(component);
        }
        let config = Self {
            block,
            rows,
            frequency: c
                .frequency_dim()
                .try_into()
                .map_err(|_| invalid("frequency exceeds u64"))?,
            intermediate: c
                .time_embed_hidden_dim()
                .try_into()
                .map_err(|_| invalid("MLP width exceeds u64"))?,
            embedding: c
                .time_embed_dim()
                .try_into()
                .map_err(|_| invalid("time width exceeds u64"))?,
            hidden: c
                .hidden_size()
                .try_into()
                .map_err(|_| invalid("hidden width exceeds u64"))?,
            components: components
                .try_into()
                .map_err(|_| invalid("time parameter inventory differs"))?,
        };
        config.validate(limits)?;
        Ok(config)
    }
    pub fn block(&self) -> usize {
        self.block
    }
    pub fn time_rows(&self) -> u64 {
        self.rows
    }
    pub fn frequency_dim(&self) -> u64 {
        self.frequency
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn embedding_dim(&self) -> u64 {
        self.embedding
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    fn validate(&self, limits: FastH3TimeConditioningLimits) -> Result<(), VNextError> {
        if limits.maximum_time_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
            || self.rows == 0
            || self.rows > limits.maximum_time_rows
            || [
                self.rows,
                self.frequency,
                self.intermediate,
                self.embedding,
                self.hidden,
            ]
            .iter()
            .any(|&d| d == 0 || d > i32::MAX as u64)
            || self.frequency % 2 != 0
        {
            return Err(invalid("invalid time geometry or capacity"));
        }
        let modulation = self
            .hidden
            .checked_mul(18)
            .filter(|&n| n <= i32::MAX as u64)
            .ok_or_else(|| invalid("modality/field width exceeds projection extent"))?;
        for (i, (component, dimensions)) in self
            .components
            .iter()
            .zip([
                vec![self.intermediate, self.frequency],
                vec![self.intermediate],
                vec![self.embedding, self.intermediate],
                vec![self.embedding],
                vec![modulation, self.embedding],
                vec![modulation],
            ])
            .enumerate()
        {
            if component.dimensions != dimensions
                || component.role != WeightComponentRole::Values
                || component.external_names.len() != 1
                || !component.required
                || component.encoding
                    != (WeightEncoding::Dense {
                        element_type: if i < 4 {
                            ElementType::F32
                        } else {
                            ElementType::Bf16
                        },
                    })
            {
                return Err(invalid("time source dimensions, role or precision differ"));
            }
        }
        if serde_json::to_vec(self)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > limits.maximum_metadata_bytes
        {
            return Err(invalid("time metadata exceeds capacity"));
        }
        let schema = self.schema()?;
        let bytes = schema.components.iter().try_fold(0u64, |sum, c| {
            sum.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("time stored extent overflows"))
        })?;
        if bytes > limits.maximum_stored_bytes {
            return Err(invalid("time stored capacity exceeded"));
        }
        for tensor in program::build(self)?.1.values() {
            tensor.byte_len()?;
        }
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new(
                "weight-layout.fast_h3.time_conditioning.f32_mlp_bf16_adaln.v1",
            )?,
            version: ContractVersion::new(1, 0),
            components: self.components.to_vec(),
            tensors: self
                .components
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    Ok(WeightTensorSpec {
                        id: weight(i)?,
                        dimensions: c.dimensions.clone(),
                        logical_element_type: c.physical_element_type(),
                        physical_layout: PhysicalWeightLayout::Dense {
                            component_id: c.id.clone(),
                        },
                        required: true,
                    })
                })
                .collect::<Result<_, VNextError>>()?,
        };
        schema.validate(&ModelFamilyId::new(
            FastH3TimeConditioningFamily::FAMILY_ID,
        )?)?;
        Ok(schema)
    }
}

/// Source selection is checked against the same validated package again before
/// conversion. Converted bytes use its caller's Host WeightBacking account.
pub struct FastH3TimeConditioningSource<'a> {
    source: ManagedDenseArchiveSource<'a>,
    components: [WeightComponentSpec; 6],
}
impl<'a> FastH3TimeConditioningSource<'a> {
    pub fn new(
        package: &'a FastH3TransformerPackage,
        config: &FastH3TimeConditioningConfig,
        host: &'a HostMemoryAccount,
        limits: FastH3TimeConditioningLimits,
    ) -> Result<Self, VNextError> {
        if FastH3TimeConditioningConfig::from_package(package, config.block, config.rows, limits)?
            != *config
        {
            return Err(invalid("time source differs from its validated package"));
        }
        Ok(Self {
            source: ManagedDenseArchiveSource::new(package.dense_archive(), host),
            components: config.components.clone(),
        })
    }
}
impl WeightComponentSource for FastH3TimeConditioningSource<'_> {
    fn component<'s>(
        &'s self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        if !self.components.contains(component) {
            return Err(invalid("undeclared time component"));
        }
        self.source.component(component)
    }
}

pub struct FastH3TimeConditioningFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3TimeConditioningLimits,
}
impl FastH3TimeConditioningFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.time_conditioning";
    pub const PROFILE_ID: &'static str = "fast_h3.time_conditioning.f32_mlp_bf16_adaln.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.time_conditioning.features";
    pub const EMBEDDING_ID: &'static str = "value.fast_h3.time_conditioning.embedding";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.time_conditioning.modulation";
    pub fn new(limits: FastH3TimeConditioningLimits) -> Result<Self, VNextError> {
        if limits.maximum_time_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
        {
            return Err(invalid("time conditioning capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.fast_h3.time_conditioning.full_rank")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3TimeConditioningFamily {
    type Config = FastH3TimeConditioningConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("time config is not canonical"));
        }
        c.validate(self.limits)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::Stored(c.schema()?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        let (program, specs) = program::build(c)?;
        let operations = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|operation_id| {
                let name = operation_id.as_str();
                NumericalOperationContract {
                    multiplication_type: match name {
                        AFFINE_LINEAR_BF16_OPERATION_ID => Some(ElementType::Bf16),
                        AFFINE_LINEAR_F32_OPERATION_ID | SILU_F32_OPERATION_ID => {
                            Some(ElementType::F32)
                        }
                        _ => None,
                    },
                    accumulation_type: matches!(
                        name,
                        AFFINE_LINEAR_F32_OPERATION_ID | AFFINE_LINEAR_BF16_OPERATION_ID
                    )
                    .then_some(ElementType::F32),
                    operation_id,
                    version: ContractVersion::new(1, 0),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }
            })
            .collect();
        let boundaries = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .map(|id| (id.clone(), specs[id].element_type))
            .collect();
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations,
            }],
            vec![NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown time conditioning profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn parameter_names(block: usize) -> [String; 6] {
    [
        "time_embedder.fc_in.weight".into(),
        "time_embedder.fc_in.bias".into(),
        "time_embedder.fc_out.weight".into(),
        "time_embedder.fc_out.bias".into(),
        format!("transformer_blocks.{block}.adaln_proj.linear.weight"),
        format!("transformer_blocks.{block}.adaln_proj.linear.bias"),
    ]
}
fn weight(index: usize) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.fast_h3.time_conditioning.{index}"))
}
fn value(suffix: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.fast_h3.time_conditioning.{suffix}"))
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 time conditioning: {}", reason.into()),
    }
}
