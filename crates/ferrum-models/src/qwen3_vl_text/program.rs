use super::{weights, Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as Family};
use ferrum_runtime::*;
use std::collections::BTreeMap;

fn value(layer: u64, suffix: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen3_vl.text.layer{layer}.{suffix}"))
}
fn node(
    layer: u64,
    suffix: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    output: ProgramValueId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.qwen3_vl.text.layer{layer}.{suffix}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
fn dense(
    layer: u64,
    suffix: &str,
    input: ProgramValueId,
    weight_suffix: &str,
    output: ProgramValueId,
    input_width: u64,
    output_width: u64,
) -> Result<ProgramNode, VNextError> {
    node(
        layer,
        suffix,
        DENSE_LINEAR_BF16_OPERATION_ID,
        vec![
            input,
            weights::value_id(&weights::external_name(layer, weight_suffix))?,
        ],
        output,
        BTreeMap::from([
            (
                AttributeId::new("in_features")?,
                SemanticValue::Unsigned(input_width),
            ),
            (
                AttributeId::new("out_features")?,
                SemanticValue::Unsigned(output_width),
            ),
        ]),
    )
}
pub(super) fn build(
    config: &Qwen3VLTextConfig,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    config.validate()?;
    let a = config.architecture();
    let h = a.hidden_size();
    let q = a.query_columns()?;
    let kv = a.key_value_columns()?;
    let m = a.intermediate_size();
    let embeddings = ProgramValueId::new(Family::EMBEDDINGS_ID)?;
    let cosine = ProgramValueId::new(Family::COSINE_ID)?;
    let sine = ProgramValueId::new(Family::SINE_ID)?;
    let weights = config
        .expected_weights()?
        .into_iter()
        .map(|w| {
            Ok(WeightReference {
                weight_id: w.weight_id()?,
                value_id: w.value_id()?,
                tensor: ProgramTensorSpec {
                    dimensions: w.dimensions().to_vec(),
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let norm_attrs = |d| -> Result<_, VNextError> {
        Ok(BTreeMap::from([
            (AttributeId::new("head_dim")?, SemanticValue::Unsigned(d)),
            (
                AttributeId::new("epsilon")?,
                SemanticValue::Rational(a.epsilon()),
            ),
        ]))
    };
    let rotary_attrs = BTreeMap::from([(
        AttributeId::new("head_dim")?,
        SemanticValue::Unsigned(a.head_dim()),
    )]);
    let attention_attrs = config.attention_attributes()?;
    let mut input = embeddings.clone();
    let mut blocks = Vec::new();
    for layer in 0..a.layers() {
        let weight = |suffix| weights::value_id(&weights::external_name(layer, suffix));
        let mut nodes = Vec::new();
        let norm = value(layer, "input_norm")?;
        nodes.push(node(
            layer,
            "input_norm",
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            vec![input.clone(), weight("input_layernorm.weight")?],
            norm.clone(),
            norm_attrs(h)?,
        )?);
        let query = value(layer, "query")?;
        let key = value(layer, "key")?;
        let val = value(layer, "value")?;
        for (name, w, out, width) in [
            ("query", "self_attn.q_proj.weight", query.clone(), q),
            ("key", "self_attn.k_proj.weight", key.clone(), kv),
            ("value", "self_attn.v_proj.weight", val.clone(), kv),
        ] {
            nodes.push(dense(layer, name, norm.clone(), w, out, h, width)?);
        }
        let qn = value(layer, "query_norm")?;
        let kn = value(layer, "key_norm")?;
        for (name, x, w, y) in [
            ("query_norm", query, "self_attn.q_norm.weight", qn.clone()),
            ("key_norm", key, "self_attn.k_norm.weight", kn.clone()),
        ] {
            nodes.push(node(
                layer,
                name,
                HEAD_RMS_NORM_BF16_OPERATION_ID,
                vec![x, weight(w)?],
                y,
                norm_attrs(a.head_dim())?,
            )?);
        }
        let qr = value(layer, "query_rotated")?;
        let kr = value(layer, "key_rotated")?;
        for (name, x, y) in [
            ("query_rotated", qn, qr.clone()),
            ("key_rotated", kn, kr.clone()),
        ] {
            nodes.push(node(
                layer,
                name,
                HALF_SPLIT_ROPE_BF16_OPERATION_ID,
                vec![x, cosine.clone(), sine.clone()],
                y,
                rotary_attrs.clone(),
            )?);
        }
        let attention = value(layer, "attention")?;
        nodes.push(node(
            layer,
            "attention",
            STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
            vec![qr, kr, val],
            attention.clone(),
            attention_attrs.clone(),
        )?);
        let projected = value(layer, "projected")?;
        nodes.push(dense(
            layer,
            "projected",
            attention,
            "self_attn.o_proj.weight",
            projected.clone(),
            q,
            h,
        )?);
        let residual = value(layer, "attention_residual")?;
        nodes.push(node(
            layer,
            "attention_residual",
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![input, projected],
            residual.clone(),
            BTreeMap::new(),
        )?);
        let post = value(layer, "post_norm")?;
        nodes.push(node(
            layer,
            "post_norm",
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            vec![residual.clone(), weight("post_attention_layernorm.weight")?],
            post.clone(),
            norm_attrs(h)?,
        )?);
        let gate = value(layer, "gate")?;
        let up = value(layer, "up")?;
        nodes.push(dense(
            layer,
            "gate",
            post.clone(),
            "mlp.gate_proj.weight",
            gate.clone(),
            h,
            m,
        )?);
        nodes.push(dense(
            layer,
            "up",
            post,
            "mlp.up_proj.weight",
            up.clone(),
            h,
            m,
        )?);
        let activation = value(layer, "activation")?;
        nodes.push(node(
            layer,
            "activation",
            SILU_MUL_BF16_OPERATION_ID,
            vec![gate, up],
            activation.clone(),
            BTreeMap::new(),
        )?);
        let down = value(layer, "down")?;
        nodes.push(dense(
            layer,
            "down",
            activation,
            "mlp.down_proj.weight",
            down.clone(),
            m,
            h,
        )?);
        let output = if layer + 1 == a.layers() {
            ProgramValueId::new(Family::OUTPUT_ID)?
        } else {
            value(layer, "output")?
        };
        nodes.push(node(
            layer,
            "output",
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![residual, down],
            output.clone(),
            BTreeMap::new(),
        )?);
        input = output;
        blocks.push(ProgramBlock {
            id: format!("block.qwen3_vl.text.layer{layer}"),
            nodes,
        });
    }
    ModelProgram::new(
        family.clone(),
        vec![embeddings, cosine, sine],
        blocks,
        vec![],
        weights,
        vec![ProgramValueId::new(Family::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(config.input_tensor_specs()?)
}
