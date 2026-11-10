//! Closed lowering of the shared visual topology to independent F32 contracts.
//! Original tensor ranks and external names survive; every intermediate has
//! an explicit compile shape. Unsupported future topology changes fail closed.
use super::*;
fn renamed(s: &str) -> String {
    s.replace("qwen3_vl.vision", "qwen3_vl.vision_comfy")
}
fn v(id: &ProgramValueId) -> std::result::Result<ProgramValueId, VNextError> {
    ProgramValueId::new(renamed(id.as_str()))
}
pub(super) fn schema(
    c: &Qwen3VLVisionComfyConfig,
) -> std::result::Result<WeightSchema, VNextError> {
    let mut s = super::super::weights::schema(
        &c.base,
        &ModelFamilyId::new(Qwen3VLVisionFamily::FAMILY_ID)?,
    )?;
    s.layout_id = WeightLayoutId::new("weight-layout.qwen3_vl.vision_comfy.original_bf16_to_f32")?;
    for w in &mut s.components {
        w.id = WeightId::new(renamed(w.id.as_str()))?;
        w.encoding = WeightEncoding::Dense {
            element_type: ElementType::F32,
        };
    }
    for w in &mut s.tensors {
        w.id = WeightId::new(renamed(w.id.as_str()))?;
        w.logical_element_type = ElementType::F32;
        let PhysicalWeightLayout::Dense { component_id } = &mut w.physical_layout else {
            return Err(invalid("visual source layout is not dense"));
        };
        *component_id = WeightId::new(renamed(component_id.as_str()))?;
    }
    s.validate(&ModelFamilyId::new(Qwen3VLVisionComfyFamily::FAMILY_ID)?)?;
    Ok(s)
}
pub(super) fn build(
    c: &Qwen3VLVisionComfyConfig,
) -> std::result::Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    build_with_attention(c, STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID)
}
pub(super) fn build_with_attention(
    c: &Qwen3VLVisionComfyConfig,
    attention_operation: &str,
) -> std::result::Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let (base, specs) = super::super::program::build(
        &c.base,
        &ModelFamilyId::new(Qwen3VLVisionFamily::FAMILY_ID)?,
    )?;
    let specs = specs
        .into_iter()
        .map(|(id, mut t)| {
            t.element_type = ElementType::F32;
            Ok((v(&id)?, t))
        })
        .collect::<std::result::Result<BTreeMap<_, _>, VNextError>>()?;
    let mut blocks = base.blocks().to_vec();
    for b in &mut blocks {
        b.id = renamed(&b.id);
        for n in &mut b.nodes {
            n.id = NodeId::new(renamed(n.id.as_str()))?;
            n.inputs = n
                .inputs
                .iter()
                .map(v)
                .collect::<std::result::Result<_, _>>()?;
            n.outputs = n
                .outputs
                .iter()
                .map(v)
                .collect::<std::result::Result<_, _>>()?;
            let mapped = match n.operation_id.as_str() {
                super::super::program::PATCH => PATCH_PROJECTION_F32_OPERATION_ID,
                super::super::program::RESHAPE => RESHAPE_COPY_F32_OPERATION_ID,
                super::super::program::AFFINE => AFFINE_LINEAR_F32_OPERATION_ID,
                super::super::program::NORM => {
                    n.attributes.remove(&AttributeId::new("hidden_size")?);
                    LAYER_NORM_F32_OPERATION_ID
                }
                super::super::program::GELU => GELU_ERF_F32_OPERATION_ID,
                GELU_TANH_BF16_OPERATION_ID => GELU_TANH_F32_OPERATION_ID,
                PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID => PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
                super::super::program::ROTARY => HALF_SPLIT_ROPE_F32_OPERATION_ID,
                ROW_GATHER_BF16_OPERATION_ID => {
                    let get = |name: &str| {
                        n.attributes
                            .get(&AttributeId::new(name)?)
                            .cloned()
                            .ok_or_else(|| invalid("shared QKV split lacks a column attribute"))
                    };
                    let start = get("column_start")?;
                    let count = get("column_count")?;
                    n.attributes = BTreeMap::from([
                        (
                            AttributeId::new("group_width")?,
                            SemanticValue::Unsigned(c.architecture().hidden_size() * 3),
                        ),
                        (AttributeId::new("column_start")?, start),
                        (AttributeId::new("column_count")?, count),
                    ]);
                    GROUPED_COLUMN_SLICE_F32_OPERATION_ID
                }
                super::super::program::ATTENTION => {
                    let a = c.architecture();
                    let r = c.rows()?;
                    let scale = n
                        .attributes
                        .get(&AttributeId::new("scale")?)
                        .cloned()
                        .ok_or_else(|| invalid("shared visual attention lacks explicit scale"))?;
                    n.attributes = BTreeMap::from([
                        (
                            AttributeId::new("sequence_length")?,
                            SemanticValue::Unsigned(r),
                        ),
                        (
                            AttributeId::new("head_dim")?,
                            SemanticValue::Unsigned(a.head_dim()),
                        ),
                        (AttributeId::new("scale")?, scale),
                        (
                            AttributeId::new("segment_lengths")?,
                            SemanticValue::Integers(vec![i64::try_from(r)
                                .map_err(|_| invalid("attention rows exceed attribute range"))?]),
                        ),
                        (
                            AttributeId::new("segment_bidirectional")?,
                            SemanticValue::Integers(vec![1]),
                        ),
                        (
                            AttributeId::new("key_run_counts")?,
                            SemanticValue::Integers(vec![1]),
                        ),
                        (
                            AttributeId::new("key_run_lengths")?,
                            SemanticValue::Integers(vec![r as i64]),
                        ),
                        (
                            AttributeId::new("key_run_valid")?,
                            SemanticValue::Integers(vec![1]),
                        ),
                    ]);
                    attention_operation
                }
                other => {
                    return Err(invalid(format!(
                        "shared visual topology added unsupported operation {other}"
                    )))
                }
            };
            n.operation_id = OperationId::new(mapped)?;
        }
    }
    let weights = base
        .weights()
        .iter()
        .map(|w| {
            let mut w = w.clone();
            w.weight_id = WeightId::new(renamed(w.weight_id.as_str()))?;
            w.value_id = v(&w.value_id)?;
            w.tensor.element_type = ElementType::F32;
            Ok(w)
        })
        .collect::<std::result::Result<_, VNextError>>()?;
    let inputs = base
        .inputs()
        .iter()
        .map(v)
        .collect::<std::result::Result<_, _>>()?;
    let outputs = base
        .outputs()
        .iter()
        .map(v)
        .collect::<std::result::Result<_, _>>()?;
    let input_specs = base
        .input_tensor_specs()
        .keys()
        .map(|id| {
            let id = v(id)?;
            Ok((id.clone(), specs[&id].clone()))
        })
        .collect::<std::result::Result<_, VNextError>>()?;
    Ok((
        ModelProgram::new(
            ModelFamilyId::new(Qwen3VLVisionComfyFamily::FAMILY_ID)?,
            inputs,
            blocks,
            vec![],
            weights,
            outputs,
        )?
        .with_input_tensor_specs(input_specs)?,
        specs,
    ))
}
