use super::{invalid, Qwen3VLTextArchitecture, Qwen3VLTextConfig};
use ferrum_runtime::*;
use serde::Serialize;

/// Required physical BF16 weight metadata; it is not payload verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen3VLTextExpectedWeight {
    external_name: String,
    dimensions: Vec<u64>,
    bytes: u64,
}
impl Qwen3VLTextExpectedWeight {
    pub fn external_name(&self) -> &str {
        &self.external_name
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    pub const fn element_type(&self) -> ElementType {
        ElementType::Bf16
    }
    pub fn weight_id(&self) -> Result<WeightId, VNextError> {
        weight_id(&self.external_name)
    }
    pub(super) fn value_id(&self) -> Result<ProgramValueId, VNextError> {
        value_id(&self.external_name)
    }
}
pub(super) fn external_name(layer: u64, suffix: &str) -> String {
    format!("model.language_model.layers.{layer}.{suffix}")
}
pub(super) fn weight_id(name: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.qwen3_vl.text.{name}"))
}
pub(super) fn value_id(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen3_vl.text.weight.{name}"))
}

pub(super) fn layer_shapes(
    a: &Qwen3VLTextArchitecture,
) -> Result<[(&'static str, Vec<u64>); 11], VNextError> {
    let h = a.hidden_size();
    let m = a.intermediate_size();
    let d = a.head_dim();
    let q = a.query_columns()?;
    let kv = a.key_value_columns()?;
    Ok([
        ("input_layernorm.weight", vec![h]),
        ("post_attention_layernorm.weight", vec![h]),
        ("self_attn.q_norm.weight", vec![d]),
        ("self_attn.k_norm.weight", vec![d]),
        ("self_attn.q_proj.weight", vec![q, h]),
        ("self_attn.k_proj.weight", vec![kv, h]),
        ("self_attn.v_proj.weight", vec![kv, h]),
        ("self_attn.o_proj.weight", vec![h, q]),
        ("mlp.gate_proj.weight", vec![m, h]),
        ("mlp.up_proj.weight", vec![m, h]),
        ("mlp.down_proj.weight", vec![h, m]),
    ])
}
pub(super) fn expected(
    config: &Qwen3VLTextConfig,
) -> Result<Vec<Qwen3VLTextExpectedWeight>, VNextError> {
    config.validate()?;
    let shapes = layer_shapes(config.architecture())?;
    let mut per_layer = 0_u64;
    for (_, dims) in &shapes {
        let bytes = dims
            .iter()
            .try_fold(2_u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("weights", "physical tensor byte size overflows"))?;
        per_layer = per_layer
            .checked_add(bytes)
            .ok_or_else(|| invalid("weights", "layer bytes overflow"))?;
    }
    per_layer
        .checked_mul(config.architecture().layers())
        .ok_or_else(|| invalid("weights", "total layer bytes overflow"))?;
    let mut out = Vec::new();
    for layer in 0..config.architecture().layers() {
        for (suffix, dims) in &shapes {
            let tensor = ResolvedTensorSpec::new(
                dims.clone(),
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            out.push(Qwen3VLTextExpectedWeight {
                external_name: external_name(layer, suffix),
                dimensions: dims.clone(),
                bytes: tensor.minimum_storage_bytes()?,
            });
        }
    }
    Ok(out)
}
pub(super) fn schema(
    config: &Qwen3VLTextConfig,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    for w in expected(config)? {
        let component_id = WeightId::new(format!(
            "weight.component.qwen3_vl.text.{}",
            w.external_name
        ))?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![w.external_name.clone()],
            dimensions: w.dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: w.weight_id()?,
            dimensions: w.dimensions,
            logical_element_type: ElementType::Bf16,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new("weight-layout.qwen3_vl.text_encoder.bf16")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(family)?;
    Ok(schema)
}
