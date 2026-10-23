use super::{invalid, Qwen3VLVisionConfig};
use ferrum_runtime::*;
use serde::Serialize;

/// Physical BF16 metadata only. The learned position table is consumed by the
/// position-input adapter, so is not an unused static allocation in this graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen3VLVisionExpectedWeight {
    external_name: String,
    dimensions: Vec<u64>,
    bytes: u64,
}
impl Qwen3VLVisionExpectedWeight {
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
        WeightId::new(format!("weight.qwen3_vl.vision.{}", self.external_name))
    }
    pub(super) fn value_id(&self) -> Result<ProgramValueId, VNextError> {
        value_id(&self.external_name)
    }
}
pub(super) fn value_id(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen3_vl.vision.weight.{name}"))
}
pub(super) fn expected(
    c: &Qwen3VLVisionConfig,
) -> Result<Vec<Qwen3VLVisionExpectedWeight>, VNextError> {
    c.validate()?;
    let a = c.architecture();
    let h = a.hidden_size();
    let m = a.intermediate_size();
    let merged = a.merger_width()?;
    let qkv = h
        .checked_mul(3)
        .ok_or_else(|| invalid("weights", "QKV width overflows"))?;
    let mut out = Vec::new();
    let mut add = |name: String, dimensions: Vec<u64>| -> Result<(), VNextError> {
        let tensor = ResolvedTensorSpec::new(
            dimensions.clone(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        out.push(Qwen3VLVisionExpectedWeight {
            external_name: name,
            dimensions,
            bytes: tensor.minimum_storage_bytes()?,
        });
        Ok(())
    };
    add(
        "model.visual.patch_embed.proj.weight".into(),
        vec![
            h,
            a.channels(),
            a.temporal_patch_size(),
            a.patch_size(),
            a.patch_size(),
        ],
    )?;
    add("model.visual.patch_embed.proj.bias".into(), vec![h])?;
    for layer in 0..a.depth() {
        for (suffix, shape) in [
            ("norm1.weight", vec![h]),
            ("norm1.bias", vec![h]),
            ("norm2.weight", vec![h]),
            ("norm2.bias", vec![h]),
            ("attn.qkv.weight", vec![qkv, h]),
            ("attn.qkv.bias", vec![qkv]),
            ("attn.proj.weight", vec![h, h]),
            ("attn.proj.bias", vec![h]),
            ("mlp.linear_fc1.weight", vec![m, h]),
            ("mlp.linear_fc1.bias", vec![m]),
            ("mlp.linear_fc2.weight", vec![h, m]),
            ("mlp.linear_fc2.bias", vec![h]),
        ] {
            add(format!("model.visual.blocks.{layer}.{suffix}"), shape)?;
        }
    }
    for (prefix, norm) in std::iter::once(("model.visual.merger".to_string(), h))
        .chain((0..3).map(|i| (format!("model.visual.deepstack_merger_list.{i}"), merged)))
    {
        for (suffix, shape) in [
            ("norm.weight", vec![norm]),
            ("norm.bias", vec![norm]),
            ("linear_fc1.weight", vec![merged, merged]),
            ("linear_fc1.bias", vec![merged]),
            ("linear_fc2.weight", vec![a.output_size(), merged]),
            ("linear_fc2.bias", vec![a.output_size()]),
        ] {
            add(format!("{prefix}.{suffix}"), shape)?;
        }
    }
    out.iter().try_fold(0_u64, |n, w| {
        n.checked_add(w.bytes)
            .ok_or_else(|| invalid("weights", "total physical bytes overflow"))
    })?;
    Ok(out)
}
pub(super) fn schema(
    c: &Qwen3VLVisionConfig,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    for w in expected(c)? {
        let component_id = WeightId::new(format!(
            "weight.component.qwen3_vl.vision.{}",
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
        layout_id: WeightLayoutId::new("weight-layout.qwen3_vl.vision.bf16")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(family)?;
    Ok(schema)
}
