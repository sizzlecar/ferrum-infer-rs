use super::*;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21VaeStemWeight {
    external_name: String,
    dimensions: Vec<u64>,
    bytes: u64,
}
impl QwenImage21VaeStemWeight {
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
        value_id(&self.external_name)
    }
}
pub(super) fn value_id(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen_image21.vae.weight.{name}"))
}
pub(super) fn expected(
    c: &QwenImage21VaeStemConfig,
) -> Result<Vec<QwenImage21VaeStemWeight>, VNextError> {
    let z = c.latent_channels();
    let h = c.hidden_channels();
    let shapes = [
        vec![z, z, 1, 1],
        vec![z],
        vec![h, z, 3, 3],
        vec![h],
        vec![h, 1, 1, 1],
        vec![h, h, 3, 3],
        vec![h],
        vec![h, 1, 1, 1],
        vec![h, h, 3, 3],
        vec![h],
    ];
    let mut total = 0_u64;
    QwenImage21VaeStemFamily::WEIGHT_NAMES
        .into_iter()
        .zip(shapes)
        .map(|(name, dimensions)| {
            let tensor = ResolvedTensorSpec::new(
                dimensions.clone(),
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            let bytes = tensor.minimum_storage_bytes()?;
            total = total
                .checked_add(bytes)
                .ok_or_else(|| invalid("weights", "prefix weight bytes overflow"))?;
            Ok(QwenImage21VaeStemWeight {
                external_name: name.to_string(),
                dimensions,
                bytes,
            })
        })
        .collect()
}
pub(super) fn schema(
    c: &QwenImage21VaeStemConfig,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    for weight in expected(c)? {
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
            dimensions: weight.dimensions,
            logical_element_type: ElementType::Bf16,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
    }
    let out = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new("weight-layout.qwen_image21.vae.decoder_stem.bf16")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    out.validate(family)?;
    Ok(out)
}
