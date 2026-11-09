//! Independent Comfy F32 visual component. The original BF16 visual family
//! remains separate. This component does not enable H3 visual token injection.
use super::{
    Qwen3VLVisionArchitecture, Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid,
};
use anyhow::{ensure, Context, Result};
use ferrum_quantization::{ManagedDenseArchiveSource, SafetensorsArchive};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod position;
mod preprocess;
mod program;
#[cfg(any(test, feature = "test-support"))]
pub(crate) mod test_fixture;
mod weights;
pub use position::QWEN3_VL_COMFY_VISION_POSITION_PROFILE;
pub use preprocess::{
    Qwen3VLVisionComfyImagePlan, Qwen3VLVisionComfyPixelPolicy,
    QWEN3_VL_COMFY_IMAGE_PREPROCESS_PROFILE,
};
pub use weights::Qwen3VLVisionComfyWeightAdapter;

#[derive(Clone, Copy, Debug)]
pub struct Qwen3VLVisionComfyLimits {
    pub max_metadata_bytes: usize,
    pub max_plan_nodes: usize,
    pub max_stored_bytes: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Qwen3VLVisionComfyConfig {
    base: Qwen3VLVisionConfig,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    base: Qwen3VLVisionConfig,
}
impl<'de> Deserialize<'de> for Qwen3VLVisionComfyConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::from_base(w.base).map_err(serde::de::Error::custom)
    }
}
impl Qwen3VLVisionComfyConfig {
    /// Each invocation processes one image. Different grids rebind one static
    /// snapshot; this does not place independent images in a causal sequence.
    pub fn new(
        architecture: Qwen3VLVisionArchitecture,
        grid: Qwen3VLVisionGrid,
    ) -> std::result::Result<Self, VNextError> {
        Self::from_base(Qwen3VLVisionConfig::new(architecture, vec![grid])?)
    }
    fn from_base(base: Qwen3VLVisionConfig) -> std::result::Result<Self, VNextError> {
        base.validate()?;
        if base.grids().len() != 1 {
            return Err(invalid("this fixed component invokes one image at a time"));
        }
        Ok(Self { base })
    }
    pub fn architecture(&self) -> &Qwen3VLVisionArchitecture {
        self.base.architecture()
    }
    pub fn grid(&self) -> &Qwen3VLVisionGrid {
        &self.base.grids()[0]
    }
    pub fn rows(&self) -> std::result::Result<u64, VNextError> {
        self.base.rows()
    }
    pub fn merged_rows(&self) -> std::result::Result<u64, VNextError> {
        self.base.merged_rows()
    }
    pub fn compile_options(&self) -> std::result::Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
}
#[derive(Clone, Debug)]
pub struct Qwen3VLVisionComfyFamily {
    id: ModelFamilyId,
    limits: Qwen3VLVisionComfyLimits,
}
impl Qwen3VLVisionComfyFamily {
    pub const FAMILY_ID: &'static str = "family.qwen3_vl.vision_comfy";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "qwen3vl.vision_comfy.reference.f32.staged_bf16_positions.v1";
    pub const CUDA_PROFILE_ID: &'static str =
        "qwen3vl.vision_comfy.cuda.f32.staged_bf16_positions.v1";
    pub const PATCHES_ID: &'static str = "value.qwen3_vl.vision_comfy.patches";
    pub const POSITIONS_ID: &'static str = "value.qwen3_vl.vision_comfy.positions";
    pub const COSINE_ID: &'static str = "value.qwen3_vl.vision_comfy.cosine";
    pub const SINE_ID: &'static str = "value.qwen3_vl.vision_comfy.sine";
    pub const OUTPUT_ID: &'static str = "value.qwen3_vl.vision_comfy.merged";
    pub const DEEPSTACK_IDS: [&'static str; 3] = [
        "value.qwen3_vl.vision_comfy.deepstack.0",
        "value.qwen3_vl.vision_comfy.deepstack.1",
        "value.qwen3_vl.vision_comfy.deepstack.2",
    ];
    pub fn new(limits: Qwen3VLVisionComfyLimits) -> std::result::Result<Self, VNextError> {
        if limits.max_metadata_bytes == 0
            || limits.max_plan_nodes == 0
            || limits.max_stored_bytes == 0
        {
            return Err(invalid("positive explicit component capacities required"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            limits,
        })
    }
    fn check(&self, c: &Qwen3VLVisionComfyConfig) -> std::result::Result<(), VNextError> {
        c.base.validate()?;
        if c.base.grids().len() != 1 {
            return Err(invalid("one original image grid required"));
        }
        if c.architecture()
            .depth()
            .checked_mul(15)
            .and_then(|n| n.checked_add(22))
            .is_none_or(|n| n > self.limits.max_plan_nodes as u64)
        {
            return Err(invalid("visual topology exceeds declared node capacity"));
        }
        let (g, _) = program::build(c)?;
        let schema = program::schema(c)?;
        if g.blocks().iter().map(|b| b.nodes.len()).sum::<usize>() > self.limits.max_plan_nodes {
            return Err(invalid("vision plan exceeds declared node capacity"));
        }
        let metadata = serde_json::to_vec(&(c, &schema, &g))
            .map_err(|e| invalid(e.to_string()))?
            .len();
        if metadata > self.limits.max_metadata_bytes {
            return Err(invalid("vision metadata exceeds declared capacity"));
        }
        let bytes = schema.components.iter().try_fold(0u64, |n, w| {
            n.checked_add(w.physical_bytes()?)
                .ok_or_else(|| invalid("weight extent overflows"))
        })?;
        if bytes > self.limits.max_stored_bytes {
            return Err(invalid(
                "F32 installed vision weights exceed declared Stored capacity",
            ));
        }
        Ok(())
    }
}
impl ModelFamilyProvider for Qwen3VLVisionComfyFamily {
    type Config = Qwen3VLVisionComfyConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([
            ExternalModelMetadataId::new("metadata.qwen3_vl.vision_comfy")
                .expect("constant metadata"),
        ])
    }
    fn parse_config(
        &self,
        raw: &serde_json::Value,
    ) -> std::result::Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> std::result::Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("noncanonical F32 vision configuration"));
        }
        self.check(c)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> std::result::Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        ExternalModelMetadataId::new("metadata.qwen3_vl.vision_comfy")
    }
    fn weights(&self, c: &Self::Config) -> std::result::Result<ModelWeights, VNextError> {
        self.check(c)?;
        Ok(ModelWeights::Stored(program::schema(c)?))
    }
    fn numerical_profiles(
        &self,
        c: &Self::Config,
    ) -> std::result::Result<FamilyNumericalProfiles, VNextError> {
        self.check(c)?;
        let (g, _) = program::build(c)?;
        let boundaries = g
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .cloned()
            .map(|v| (v, ElementType::F32))
            .collect::<BTreeMap<_, _>>();
        let ids = g
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.clone())
            .collect::<BTreeSet<_>>();
        let operations = ids
            .into_iter()
            .map(|id| {
                let copy = matches!(
                    id.as_str(),
                    RESHAPE_COPY_F32_OPERATION_ID | GROUPED_COLUMN_SLICE_F32_OPERATION_ID
                );
                let add = id.as_str() == PLAIN_RESIDUAL_ADD_F32_OPERATION_ID;
                NumericalOperationContract {
                    operation_id: id,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (!copy && !add).then_some(ElementType::F32),
                    accumulation_type: (!copy).then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }
            })
            .collect::<Vec<_>>();
        let profiles = [Self::REFERENCE_PROFILE_ID, Self::CUDA_PROFILE_ID]
            .into_iter()
            .map(|id| {
                Ok(NumericalExecutionProfile {
                    id: NumericalProfileId::new(id).map_err(invalid)?,
                    family_id: self.id.clone(),
                    version: ContractVersion::new(1, 0),
                    primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                    boundaries: boundaries.clone(),
                    states: vec![],
                    kv_storage: vec![],
                    operations: operations.clone(),
                })
            })
            .collect::<std::result::Result<Vec<_>, VNextError>>()?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> std::result::Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("F32 vision numerical profile differs"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(
        &self,
        _: &Self::Config,
    ) -> std::result::Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
/// Original checkpoint validation. Conversion exists only in an explicitly
/// borrowed source bound to the current Host account; no dense Vec mirror.
pub struct Qwen3VLVisionComfyCheckpoint<'a> {
    archive: &'a SafetensorsArchive,
    schema: WeightSchema,
    index: BTreeMap<WeightId, usize>,
}
impl<'a> Qwen3VLVisionComfyCheckpoint<'a> {
    pub fn new(
        c: &Qwen3VLVisionComfyConfig,
        archive: &'a SafetensorsArchive,
        limits: Qwen3VLVisionComfyLimits,
    ) -> Result<Self> {
        let family = Qwen3VLVisionComfyFamily::new(limits)?;
        family.check(c)?;
        let schema = program::schema(c)?;
        let mut index = BTreeMap::new();
        for (i, w) in schema.components.iter().enumerate() {
            let [name] = w.external_names.as_slice() else {
                anyhow::bail!("vision original tensor mapping is not one-to-one")
            };
            let raw = archive.tensor(name)?;
            ensure!(
                raw.shape() == w.dimensions && raw.element_type() == Some(ElementType::Bf16),
                "original visual dtype/shape differs: {name}"
            );
            index.insert(w.id.clone(), i);
        }
        let p = archive.tensor(Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)?;
        ensure!(
            p.element_type() == Some(ElementType::Bf16)
                && p.shape()
                    == [
                        c.architecture().position_embeddings(),
                        c.architecture().hidden_size()
                    ],
            "original learned position table differs"
        );
        Ok(Self {
            archive,
            schema,
            index,
        })
    }
    pub fn weight_source<'s>(
        &'s self,
        host: &'s HostMemoryAccount,
    ) -> Qwen3VLVisionComfyWeightSource<'s> {
        Qwen3VLVisionComfyWeightSource {
            schema: &self.schema,
            index: &self.index,
            dense: ManagedDenseArchiveSource::new(self.archive, host),
        }
    }
}
pub struct Qwen3VLVisionComfyWeightSource<'a> {
    schema: &'a WeightSchema,
    index: &'a BTreeMap<WeightId, usize>,
    dense: ManagedDenseArchiveSource<'a>,
}
impl WeightComponentSource for Qwen3VLVisionComfyWeightSource<'_> {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'s>, VNextError> {
        let i = self
            .index
            .get(&c.id)
            .ok_or_else(|| invalid("foreign vision weight component"))?;
        if self.schema.components[*i] != *c {
            return Err(invalid(
                "vision component differs from checked original projection contract",
            ));
        }
        self.dense.component(c)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: Qwen3VLVisionComfyFamily::FAMILY_ID.into(),
        field: "vision_comfy".into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
pub(crate) mod tests;
