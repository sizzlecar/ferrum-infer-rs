use super::*;

fn value(layer: u64, suffix: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen3_vl.text.comfy.layer{layer}.{suffix}"))
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
        id: NodeId::new(format!("node.qwen3_vl.text.comfy.layer{layer}.{suffix}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
fn projection(
    c: &Qwen3VLTextComfyConfig,
    layer: u64,
    suffix: &str,
    module: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    k: u64,
    n: u64,
) -> Result<ProgramNode, VNextError> {
    let module = super::super::weights::external_name(layer, module);
    let quantization = c.projections()[&module];
    let mut inputs = vec![input];
    for suffix in quantization.components() {
        inputs.push(weights::value(&format!("{module}.{suffix}"))?);
    }
    node(
        layer,
        suffix,
        quantization.operation(),
        inputs,
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
pub(super) fn build(
    c: &Qwen3VLTextComfyConfig,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    c.validate()?;
    let text = c.text();
    let a = text.architecture();
    let h = a.hidden_size();
    let q = a.query_columns()?;
    let kv = a.key_value_columns()?;
    let m = a.intermediate_size();
    let norm_attrs = |head_dim| -> Result<_, VNextError> {
        Ok(BTreeMap::from([
            (
                AttributeId::new("head_dim")?,
                SemanticValue::Unsigned(head_dim),
            ),
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
    let cosine = ProgramValueId::new(Qwen3VLTextComfyFamily::COSINE_ID)?;
    let sine = ProgramValueId::new(Qwen3VLTextComfyFamily::SINE_ID)?;
    let embeddings = ProgramValueId::new(Qwen3VLTextComfyFamily::EMBEDDINGS_ID)?;
    let mut input = embeddings.clone();
    let mut blocks = Vec::new();
    if let Some((replacement, replaced)) = visual::replacement(c)? {
        blocks.push(replacement);
        input = replaced;
    }
    for layer in 0..a.layers() {
        let weight =
            |suffix: &str| weights::value(&super::super::weights::external_name(layer, suffix));
        let mut nodes = Vec::new();
        let norm = value(layer, "input_norm")?;
        nodes.push(node(
            layer,
            "input_norm",
            HEAD_RMS_NORM_F32_OPERATION_ID,
            vec![input.clone(), weight("input_layernorm.weight")?],
            norm.clone(),
            norm_attrs(h)?,
        )?);
        let query = value(layer, "query")?;
        let key = value(layer, "key")?;
        let val = value(layer, "value")?;
        for (suffix, module, output, width) in [
            ("query", "self_attn.q_proj", query.clone(), q),
            ("key", "self_attn.k_proj", key.clone(), kv),
            ("value", "self_attn.v_proj", val.clone(), kv),
        ] {
            nodes.push(projection(
                c,
                layer,
                suffix,
                module,
                norm.clone(),
                output,
                h,
                width,
            )?);
        }
        let qn = value(layer, "query_norm")?;
        let kn = value(layer, "key_norm")?;
        for (suffix, x, gamma, y) in [
            ("query_norm", query, "self_attn.q_norm.weight", qn.clone()),
            ("key_norm", key, "self_attn.k_norm.weight", kn.clone()),
        ] {
            nodes.push(node(
                layer,
                suffix,
                HEAD_RMS_NORM_F32_OPERATION_ID,
                vec![x, weight(gamma)?],
                y,
                norm_attrs(a.head_dim())?,
            )?);
        }
        let qr = value(layer, "query_rotated")?;
        let kr = value(layer, "key_rotated")?;
        for (suffix, x, y) in [
            ("query_rotated", qn, qr.clone()),
            ("key_rotated", kn, kr.clone()),
        ] {
            nodes.push(node(
                layer,
                suffix,
                HALF_SPLIT_ROPE_F32_OPERATION_ID,
                vec![x, cosine.clone(), sine.clone()],
                y,
                rotary_attrs.clone(),
            )?);
        }
        let attention = value(layer, "attention")?;
        nodes.push(node(
            layer,
            "attention",
            STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID,
            vec![qr, kr, val],
            attention.clone(),
            text.attention_attributes()?,
        )?);
        let projected = value(layer, "projected")?;
        nodes.push(projection(
            c,
            layer,
            "projected",
            "self_attn.o_proj",
            attention,
            projected.clone(),
            q,
            h,
        )?);
        let residual = value(layer, "attention_residual")?;
        nodes.push(node(
            layer,
            "attention_residual",
            PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
            vec![input, projected],
            residual.clone(),
            BTreeMap::new(),
        )?);
        let post = value(layer, "post_norm")?;
        nodes.push(node(
            layer,
            "post_norm",
            HEAD_RMS_NORM_F32_OPERATION_ID,
            vec![residual.clone(), weight("post_attention_layernorm.weight")?],
            post.clone(),
            norm_attrs(h)?,
        )?);
        let gate = value(layer, "gate")?;
        let up = value(layer, "up")?;
        nodes.push(projection(
            c,
            layer,
            "gate",
            "mlp.gate_proj",
            post.clone(),
            gate.clone(),
            h,
            m,
        )?);
        nodes.push(projection(
            c,
            layer,
            "up",
            "mlp.up_proj",
            post,
            up.clone(),
            h,
            m,
        )?);
        let activation = value(layer, "activation")?;
        nodes.push(node(
            layer,
            "activation",
            SILU_MUL_F32_OPERATION_ID,
            vec![gate, up],
            activation.clone(),
            BTreeMap::new(),
        )?);
        let down = value(layer, "down")?;
        nodes.push(projection(
            c,
            layer,
            "down",
            "mlp.down_proj",
            activation,
            down.clone(),
            m,
            h,
        )?);
        let inject = c.visual_inputs().is_some()
            && layer < Qwen3VLTextComfyFamily::DEEPSTACK_IDS.len() as u64;
        let selected = layer + 1 < a.layers()
            && text
                .hidden_state_indices()
                .binary_search(&(layer + 1))
                .is_ok();
        let output = if layer + 1 == a.layers() && !inject {
            ProgramValueId::new(Qwen3VLTextComfyFamily::OUTPUT_ID)?
        } else if selected && !inject {
            Qwen3VLTextComfyFamily::hidden_state_output_id(layer + 1)?
        } else {
            value(layer, "output")?
        };
        nodes.push(node(
            layer,
            "output",
            PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
            vec![residual, down],
            output.clone(),
            BTreeMap::new(),
        )?);
        input = output;
        if inject {
            let injected = if layer + 1 == a.layers() {
                ProgramValueId::new(Qwen3VLTextComfyFamily::OUTPUT_ID)?
            } else if selected {
                Qwen3VLTextComfyFamily::hidden_state_output_id(layer + 1)?
            } else {
                value(layer, "deepstack")?
            };
            nodes.push(node(
                layer,
                "deepstack",
                ROW_SCATTER_ADD_F32_OPERATION_ID,
                vec![
                    input,
                    ProgramValueId::new(Qwen3VLTextComfyFamily::DEEPSTACK_IDS[layer as usize])?,
                ],
                injected.clone(),
                c.visual_inputs()
                    .expect("checked visual input mode")
                    .attributes()?,
            )?);
            input = injected;
        }
        blocks.push(ProgramBlock {
            id: format!("block.qwen3_vl.text.comfy.layer{layer}"),
            nodes,
        });
    }
    let mut outputs = vec![ProgramValueId::new(Qwen3VLTextComfyFamily::OUTPUT_ID)?];
    for &index in text.hidden_state_indices() {
        let output = Qwen3VLTextComfyFamily::hidden_state_output_id(index)?;
        if index == a.layers() {
            blocks.push(ProgramBlock {
                id: "block.qwen3_vl.text.comfy.final_norm".into(),
                nodes: vec![ProgramNode {
                    id: NodeId::new("node.qwen3_vl.text.comfy.final_norm")?,
                    operation_id: OperationId::new(HEAD_RMS_NORM_F32_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![
                        input.clone(),
                        weights::value(DenseFamily::FINAL_NORM_EXTERNAL_NAME)?,
                    ],
                    outputs: vec![output.clone()],
                    attributes: norm_attrs(h)?,
                }],
            });
        }
        outputs.push(output);
    }
    let mut inputs = vec![embeddings, cosine, sine];
    if c.visual_inputs().is_some() {
        inputs.push(ProgramValueId::new(
            Qwen3VLTextComfyFamily::VISUAL_EMBEDDINGS_ID,
        )?);
        for id in Qwen3VLTextComfyFamily::DEEPSTACK_IDS {
            inputs.push(ProgramValueId::new(id)?);
        }
    }
    ModelProgram::new(
        family.clone(),
        inputs,
        blocks,
        vec![],
        weights::references(c)?,
        outputs,
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)
}
