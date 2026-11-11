use super::*;
/// Emit concrete output geometry from the actual node stream, rather than a
/// second list of node names. Grouped slicing relates two independent column
/// symbols and therefore needs explicit metadata at the compiler boundary.
pub(super) fn output_tensor_specs(
    c: &PH3ImageVaeDecoderBlockConfig,
) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
    let graph = build(
        c,
        &ModelFamilyId::new(PH3ImageVaeDecoderBlockFamily::FAMILY_ID)?,
    )?;
    let mut known = c.input_tensor_specs()?;
    known.extend(
        graph
            .weights()
            .iter()
            .map(|w| (w.value_id.clone(), w.tensor.clone())),
    );
    let mut outputs = BTreeMap::new();
    for node in graph.blocks().iter().flat_map(|b| &b.nodes) {
        let source = known
            .get(&node.inputs[0])
            .ok_or_else(|| invalid("program", "output geometry has no source tensor"))?;
        let read = |name: &str| -> Result<u64, VNextError> {
            match node.attributes.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Unsigned(v)) => Ok(*v),
                _ => Err(invalid(
                    "program",
                    "output geometry needs unsigned attribute",
                )),
            }
        };
        let mut tensor = source.clone();
        match node.operation_id.as_str() {
            COMFY_CONVROT_W8A8_F32_OPERATION_ID => tensor.dimensions[1] = read("out_features")?,
            GROUPED_COLUMN_SLICE_F32_OPERATION_ID => {
                let width = read("group_width")?;
                let count = read("column_count")?;
                tensor.dimensions[1] = source.dimensions[1]
                    .checked_div(width)
                    .and_then(|groups| groups.checked_mul(count))
                    .ok_or_else(|| invalid("program", "grouped-slice output width overflows"))?;
            }
            HEAD_RMS_NORM_F32_OPERATION_ID
            | UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID
            | STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID
            | CHANNEL_BIAS_ADD_F32_OPERATION_ID
            | CHANNEL_SCALED_RESIDUAL_F32_OPERATION_ID
            | NAN_TO_NUM_F32_OPERATION_ID
            | SILU_MUL_F32_OPERATION_ID => {}
            _ => {
                return Err(invalid(
                    "program",
                    "new operation requires explicit output geometry",
                ))
            }
        }
        tensor.validate("VAE block output")?;
        for value in &node.outputs {
            known.insert(value.clone(), tensor.clone());
            outputs.insert(value.clone(), tensor.clone());
        }
    }
    Ok(outputs)
}
pub(super) fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.vae.decoder_block.{name}"))
}
fn node(
    name: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    output: ProgramValueId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.p_h3_image.vae.decoder_block.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
fn w8(
    name: &str,
    module: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    k: u64,
    n: u64,
) -> Result<ProgramNode, VNextError> {
    node(
        name,
        COMFY_CONVROT_W8A8_F32_OPERATION_ID,
        vec![
            input,
            weights::value(&format!("{module}.weight"))?,
            weights::value(&format!("{module}.weight_scale"))?,
        ],
        output,
        BTreeMap::from([
            (AttributeId::new("in_features")?, SemanticValue::Unsigned(k)),
            (
                AttributeId::new("out_features")?,
                SemanticValue::Unsigned(n),
            ),
        ]),
    )
}
fn slice(
    name: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    group_width: u64,
    column_start: u64,
    column_count: u64,
) -> Result<ProgramNode, VNextError> {
    node(
        name,
        GROUPED_COLUMN_SLICE_F32_OPERATION_ID,
        vec![input],
        output,
        BTreeMap::from([
            (
                AttributeId::new("group_width")?,
                SemanticValue::Unsigned(group_width),
            ),
            (
                AttributeId::new("column_start")?,
                SemanticValue::Unsigned(column_start),
            ),
            (
                AttributeId::new("column_count")?,
                SemanticValue::Unsigned(column_count),
            ),
        ]),
    )
}
pub(super) fn build(
    c: &PH3ImageVaeDecoderBlockConfig,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    build_with_attention(c, family, STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID)
}
pub(super) fn build_with_attention(
    c: &PH3ImageVaeDecoderBlockConfig,
    family: &ModelFamilyId,
    attention_operation: &str,
) -> Result<ModelProgram, VNextError> {
    c.validate()?;
    type F = PH3ImageVaeDecoderBlockFamily;
    let a = c.architecture();
    let h = a.hidden_size()?;
    let m = a.intermediate_size()?;
    let d = a.head_dim();
    let input = ProgramValueId::new(F::INPUT_ID)?;
    let cosine = ProgramValueId::new(F::COSINE_ID)?;
    let sine = ProgramValueId::new(F::SINE_ID)?;
    let norm_attrs = |width| -> Result<_, VNextError> {
        Ok(BTreeMap::from([
            (
                AttributeId::new("head_dim")?,
                SemanticValue::Unsigned(width),
            ),
            (
                AttributeId::new("epsilon")?,
                SemanticValue::Rational(a.norm_epsilon()),
            ),
        ]))
    };
    let mut rotary_attrs = norm_attrs(d)?;
    rotary_attrs.insert(
        AttributeId::new("rotary_dim")?,
        SemanticValue::Unsigned(a.rotary_dim()),
    );
    let mut nodes = Vec::new();
    let norm1 = value("norm1")?;
    nodes.push(node(
        "norm1",
        HEAD_RMS_NORM_F32_OPERATION_ID,
        vec![input.clone(), weights::value("norm1.weight")?],
        norm1.clone(),
        norm_attrs(h)?,
    )?);
    let raw_qkv = value("qkv_linear")?;
    nodes.push(w8(
        "qkv_linear",
        "attn.to_qkv",
        norm1,
        raw_qkv.clone(),
        h,
        3 * h,
    )?);
    let qkv = value("qkv_biased")?;
    nodes.push(node(
        "qkv_bias",
        CHANNEL_BIAS_ADD_F32_OPERATION_ID,
        vec![raw_qkv, weights::value("attn.to_qkv.bias")?],
        qkv.clone(),
        BTreeMap::new(),
    )?);
    let q = value("query")?;
    let k = value("key")?;
    let v = value("value")?;
    // Source rows are [head0(QD,KD,VD),head1(QD,KD,VD),...]. A
    // contiguous H-wide slice would silently mix semantic heads/components.
    for (name, start, output) in [
        ("query", 0, q.clone()),
        ("key", d, k.clone()),
        ("value", 2 * d, v.clone()),
    ] {
        nodes.push(slice(name, qkv.clone(), output, 3 * d, start, d)?);
    }
    let qr = value("query_normalized_rotated")?;
    let kr = value("key_normalized_rotated")?;
    for (name, x, y) in [
        ("query_normalized_rotated", q, qr.clone()),
        ("key_normalized_rotated", k, kr.clone()),
    ] {
        nodes.push(node(
            name,
            UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID,
            vec![x, cosine.clone(), sine.clone()],
            y,
            rotary_attrs.clone(),
        )?);
    }
    let attention = value("attention")?;
    nodes.push(node(
        "attention",
        attention_operation,
        vec![qr, kr, v],
        attention.clone(),
        c.attention_attributes()?,
    )?);
    let finite = value("attention_finite")?;
    nodes.push(node(
        "attention_nan_to_num",
        NAN_TO_NUM_F32_OPERATION_ID,
        vec![attention],
        finite.clone(),
        BTreeMap::new(),
    )?);
    let projected = value("attention_linear")?;
    nodes.push(w8(
        "attention_linear",
        "attn.to_out",
        finite,
        projected.clone(),
        h,
        h,
    )?);
    let biased = value("attention_biased")?;
    nodes.push(node(
        "attention_bias",
        CHANNEL_BIAS_ADD_F32_OPERATION_ID,
        vec![projected, weights::value("attn.to_out.bias")?],
        biased.clone(),
        BTreeMap::new(),
    )?);
    let residual = value("attention_residual")?;
    nodes.push(node(
        "attention_residual",
        CHANNEL_SCALED_RESIDUAL_F32_OPERATION_ID,
        vec![input, biased, weights::value("scale1")?],
        residual.clone(),
        BTreeMap::new(),
    )?);
    let norm2 = value("norm2")?;
    nodes.push(node(
        "norm2",
        HEAD_RMS_NORM_F32_OPERATION_ID,
        vec![residual.clone(), weights::value("norm2.weight")?],
        norm2.clone(),
        norm_attrs(h)?,
    )?);
    let w1 = value("w1_linear")?;
    nodes.push(w8("w1_linear", "ff.w1", norm2, w1.clone(), h, 2 * m)?);
    let w1_bias = value("w1_biased")?;
    nodes.push(node(
        "w1_bias",
        CHANNEL_BIAS_ADD_F32_OPERATION_ID,
        vec![w1, weights::value("ff.w1.bias")?],
        w1_bias.clone(),
        BTreeMap::new(),
    )?);
    let gate = value("gate")?;
    let up = value("up")?;
    nodes.push(slice("gate", w1_bias.clone(), gate.clone(), 2 * m, 0, m)?);
    nodes.push(slice("up", w1_bias, up.clone(), 2 * m, m, m)?);
    let activation = value("activation")?;
    nodes.push(node(
        "activation",
        SILU_MUL_F32_OPERATION_ID,
        vec![gate, up],
        activation.clone(),
        BTreeMap::new(),
    )?);
    let w2 = value("w2_linear")?;
    nodes.push(w8("w2_linear", "ff.w2", activation, w2.clone(), m, h)?);
    let w2_bias = value("w2_biased")?;
    nodes.push(node(
        "w2_bias",
        CHANNEL_BIAS_ADD_F32_OPERATION_ID,
        vec![w2, weights::value("ff.w2.bias")?],
        w2_bias.clone(),
        BTreeMap::new(),
    )?);
    let output = ProgramValueId::new(F::OUTPUT_ID)?;
    nodes.push(node(
        "output",
        CHANNEL_SCALED_RESIDUAL_F32_OPERATION_ID,
        vec![residual, w2_bias, weights::value("scale2")?],
        output.clone(),
        BTreeMap::new(),
    )?);
    ModelProgram::new(
        family.clone(),
        vec![ProgramValueId::new(F::INPUT_ID)?, cosine, sine],
        vec![ProgramBlock {
            id: format!("block.p_h3_image.vae.decoder.{}", c.block_index()),
            nodes,
        }],
        vec![],
        weights::references(c)?,
        vec![output],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)
}
