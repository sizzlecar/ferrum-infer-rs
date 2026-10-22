use super::*;
use serde::Serialize;
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21VaeDecoderWeight {
    external_name: String,
    dimensions: Vec<u64>,
    bytes: u64,
}
impl QwenImage21VaeDecoderWeight {
    pub(super) fn new(name: &str, dimensions: &[u64]) -> Result<Self, VNextError> {
        Ok(Self {
            external_name: name.into(),
            dimensions: dimensions.to_vec(),
            bytes: tensor(dimensions)?.minimum_storage_bytes()?,
        })
    }
    pub fn external_name(&self) -> &str {
        &self.external_name
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn weight_id(&self) -> Result<WeightId, VNextError> {
        WeightId::new(format!("weight.qwen_image21.vae.{}", self.external_name))
    }
    pub(super) fn value_id(&self) -> Result<ProgramValueId, VNextError> {
        ProgramValueId::new(format!(
            "value.qwen_image21.vae.weight.{}",
            self.external_name
        ))
    }
}
pub(super) fn schema(
    weights: &[QwenImage21VaeDecoderWeight],
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut total = 0_u64;
    for weight in weights {
        total = total
            .checked_add(weight.bytes)
            .ok_or_else(|| invalid("weights", "total bytes overflow"))?;
        let component_id = WeightId::new(format!(
            "weight.component.qwen_image21.vae.{}",
            weight.external_name
        ))?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![weight.external_name.clone()],
            dimensions: weight.dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: weight.weight_id()?,
            dimensions: weight.dimensions.clone(),
            logical_element_type: ElementType::Bf16,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new("weight-layout.qwen_image21.vae.decoder.bf16")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(family)?;
    Ok(schema)
}
