use super::*;
use ferrum_quantization::comfy_convrot::COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID;

pub(super) fn projection_shapes(
    text: &Qwen3VLTextConfig,
) -> Result<Vec<(String, u64, u64)>, VNextError> {
    let a = text.architecture();
    let h = a.hidden_size();
    let q = a.query_columns()?;
    let kv = a.key_value_columns()?;
    let m = a.intermediate_size();
    let mut out = Vec::new();
    for layer in 0..a.layers() {
        for (suffix, input, output) in [
            ("self_attn.q_proj", h, q),
            ("self_attn.k_proj", h, kv),
            ("self_attn.v_proj", h, kv),
            ("self_attn.o_proj", q, h),
            ("mlp.gate_proj", h, m),
            ("mlp.up_proj", h, m),
            ("mlp.down_proj", m, h),
        ] {
            out.push((
                super::super::weights::external_name(layer, suffix),
                input,
                output,
            ));
        }
    }
    Ok(out)
}
pub(super) fn norm_shapes(text: &Qwen3VLTextConfig) -> Vec<(String, Vec<u64>)> {
    let a = text.architecture();
    let mut out = Vec::new();
    for layer in 0..a.layers() {
        for (suffix, width) in [
            ("input_layernorm.weight", a.hidden_size()),
            ("post_attention_layernorm.weight", a.hidden_size()),
            ("self_attn.q_norm.weight", a.head_dim()),
            ("self_attn.k_norm.weight", a.head_dim()),
        ] {
            out.push((
                super::super::weights::external_name(layer, suffix),
                vec![width],
            ));
        }
    }
    if text.hidden_state_indices().last() == Some(&a.layers()) {
        out.push((
            DenseFamily::FINAL_NORM_EXTERNAL_NAME.into(),
            vec![a.hidden_size()],
        ));
    }
    out
}
pub(super) fn entries(
    c: &Qwen3VLTextComfyConfig,
) -> Result<Vec<(String, Vec<u64>, ElementType)>, VNextError> {
    // These small vectors are promoted exactly once by the ordinary dense
    // source materializer. Canonical checkpoint validation still requires BF16.
    let mut out = norm_shapes(c.text())
        .into_iter()
        .map(|(name, shape)| (name, shape, ElementType::F32))
        .collect::<Vec<_>>();
    for (module, input, output) in projection_shapes(c.text())? {
        let quantization = c.projections()[&module];
        let shapes = match quantization {
            Qwen3VLTextComfyProjectionQuantization::W4A8 => vec![
                ("weight", vec![output, input / 2], ElementType::I8),
                ("weight_s_rel", vec![output, input / 16], ElementType::U8),
                ("weight_s_channel", vec![output], ElementType::F32),
                ("weight_codebook", vec![16], ElementType::F32),
            ],
            Qwen3VLTextComfyProjectionQuantization::W8A8 => vec![
                ("weight", vec![output, input], ElementType::I8),
                ("weight_scale", vec![output, 1], ElementType::F32),
            ],
        };
        for (suffix, dimensions, dtype) in shapes {
            out.push((format!("{module}.{suffix}"), dimensions, dtype));
        }
    }
    Ok(out)
}
pub(super) fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen3_vl.text.comfy.weight.{name}"))
}
fn weight(name: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.qwen3_vl.text.comfy.{name}"))
}
pub(super) fn schema(
    c: &Qwen3VLTextComfyConfig,
    capacity: u64,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut bytes = 0u64;
    for (name, dimensions, element_type) in entries(c)? {
        let tensor = ProgramTensorSpec {
            dimensions: dimensions.clone(),
            element_type,
            layout: ResolvedTensorLayout::Contiguous,
        };
        bytes = bytes
            .checked_add(tensor.byte_len()?)
            .filter(|n| *n <= capacity)
            .ok_or_else(|| invalid("weights", "weights exceed declared capacity"))?;
        let component_id = WeightId::new(format!("component.qwen3_vl.text.comfy.{name}"))?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![name.clone()],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense { element_type },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: weight(&name)?,
            dimensions,
            logical_element_type: element_type,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new(COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID)?,
        layout_id: WeightLayoutId::new("weight-layout.qwen3_vl.text.comfy.canonical_w4_w8_f32.v1")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(family)?;
    Ok(schema)
}
pub(super) fn references(c: &Qwen3VLTextComfyConfig) -> Result<Vec<WeightReference>, VNextError> {
    entries(c)?
        .into_iter()
        .map(|(name, dimensions, element_type)| {
            Ok(WeightReference {
                weight_id: weight(&name)?,
                value_id: value(&name)?,
                tensor: ProgramTensorSpec {
                    dimensions,
                    element_type,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            })
        })
        .collect()
}
