use super::*;
pub(super) const CONDITION: &str = "value.p_h3_image.token_refiner.condition";
pub(super) fn stage(layer: u64, name: &str) -> String {
    format!("value.p_h3_image.token_refiner.layer{layer}.{name}")
}
fn id(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(name)
}
fn attr(name: &str) -> Result<AttributeId, VNextError> {
    AttributeId::new(name)
}
struct Graph {
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
}
impl Graph {
    /// Every emitted intermediate receives a concrete checked shape together
    /// with its node. Compile and execution consume the same shape contract.
    fn emit(
        &mut self,
        name: &str,
        operation: &str,
        inputs: Vec<ProgramValueId>,
        output: &str,
        shape: Vec<u64>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<(), VNextError> {
        let output = id(output)?;
        let tensor = config::tensor(shape, ElementType::Bf16);
        tensor.validate(output.as_str())?;
        if tensor.byte_len()? > isize::MAX as u64
            || self.specs.insert(output.clone(), tensor).is_some()
        {
            return Err(invalid(
                "intermediate shape exceeds address space or output is duplicated",
            ));
        }
        self.nodes.push(ProgramNode {
            id: NodeId::new(format!("node.p_h3_image.token_refiner.{name}"))?,
            operation_id: OperationId::new(operation)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output],
            attributes,
        });
        Ok(())
    }
    fn norm(
        &mut self,
        name: &str,
        input: &str,
        weight: &str,
        output: &str,
        rows: u64,
        width: u64,
        epsilon: CanonicalRational,
    ) -> Result<(), VNextError> {
        self.emit(
            name,
            ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID,
            vec![id(input)?, weights::value(weight)?],
            output,
            vec![rows, width],
            BTreeMap::from([(attr("epsilon")?, SemanticValue::Rational(epsilon))]),
        )
    }
    fn reshape(
        &mut self,
        name: &str,
        input: &str,
        output: &str,
        rows: u64,
        columns: u64,
    ) -> Result<(), VNextError> {
        self.emit(
            name,
            RESHAPE_COPY_BF16_OPERATION_ID,
            vec![id(input)?],
            output,
            vec![rows, columns],
            BTreeMap::from([
                (attr("rows")?, SemanticValue::Unsigned(rows)),
                (attr("columns")?, SemanticValue::Unsigned(columns)),
            ]),
        )
    }
    fn w8(
        &mut self,
        name: &str,
        input: &str,
        weight: &str,
        output: &str,
        rows: u64,
        k: u64,
        n: u64,
        fused: bool,
    ) -> Result<(), VNextError> {
        self.emit(
            name,
            if fused {
                COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID
            } else {
                COMFY_CONVROT_W8A8_BF16_OPERATION_ID
            },
            vec![
                id(input)?,
                weights::value(&format!("{weight}.weight"))?,
                weights::value(&format!("{weight}.weight_scale"))?,
            ],
            output,
            vec![rows, n],
            BTreeMap::from([
                (attr("in_features")?, SemanticValue::Unsigned(k)),
                (attr("out_features")?, SemanticValue::Unsigned(n)),
            ]),
        )
    }
    fn add(
        &mut self,
        name: &str,
        input: &str,
        update: &str,
        output: &str,
        rows: u64,
        h: u64,
    ) -> Result<(), VNextError> {
        self.emit(
            name,
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![id(input)?, id(update)?],
            output,
            vec![rows, h],
            BTreeMap::new(),
        )
    }
}
fn graph(c: &PH3ImageTokenRefinerConfig) -> Result<Graph, VNextError> {
    let mut g = Graph {
        nodes: Vec::new(),
        specs: c.input_tensor_specs()?,
    };
    let rows = c.rows()?;
    let h = c.hidden;
    let inner = c.inner();
    g.emit(
        "condition_projection",
        AFFINE_LINEAR_BF16_OPERATION_ID,
        vec![
            id(PH3ImageTokenRefinerFamily::INPUT_ID)?,
            weights::value("condition_proj.weight")?,
            weights::value("condition_proj.bias")?,
        ],
        CONDITION,
        vec![rows, h],
        BTreeMap::from([
            (attr("in_features")?, SemanticValue::Unsigned(c.text_dim)),
            (attr("out_features")?, SemanticValue::Unsigned(h)),
        ]),
    )?;
    let mut previous = CONDITION.to_owned();
    for layer in 0..c.layers {
        let output = |name: &str| stage(layer, name);
        let node = |name: &str| format!("layer{layer}.{name}");
        let weight = |name: &str| format!("token_refiner.blocks.{layer}.{name}");
        g.norm(
            &node("norm1"),
            &previous,
            &weight("norm1.weight"),
            &output("norm1"),
            rows,
            h,
            c.norm_eps,
        )?;
        g.w8(
            &node("qkv"),
            &output("norm1"),
            &weight("attn.qkv_proj"),
            &output("qkv"),
            rows,
            h,
            3 * inner,
            false,
        )?;
        for (part, name) in ["q", "k", "v"].into_iter().enumerate() {
            g.emit(
                &node(&format!("{name}_slice")),
                ROW_GATHER_BF16_OPERATION_ID,
                vec![id(&output("qkv"))?],
                &output(name),
                vec![rows, inner],
                BTreeMap::from([
                    (attr("batch")?, SemanticValue::Unsigned(1)),
                    (
                        attr("column_start")?,
                        SemanticValue::Unsigned(part as u64 * inner),
                    ),
                    (attr("column_count")?, SemanticValue::Unsigned(inner)),
                    (
                        attr("run_lengths")?,
                        SemanticValue::Integers(vec![rows as i64]),
                    ),
                    (attr("source_starts")?, SemanticValue::Integers(vec![0])),
                    (
                        attr("source_batch_strides")?,
                        SemanticValue::Integers(vec![0]),
                    ),
                    (
                        attr("source_row_strides")?,
                        SemanticValue::Integers(vec![1]),
                    ),
                ]),
            )?;
        }
        for name in ["q", "k"] {
            g.reshape(
                &node(&format!("{name}_heads")),
                &output(name),
                &output(&format!("{name}_heads")),
                rows * c.heads,
                c.head_dim,
            )?;
            g.norm(
                &node(&format!("{name}_norm")),
                &output(&format!("{name}_heads")),
                &weight(&format!("attn.{name}_norm.weight")),
                &output(&format!("{name}_norm_heads")),
                rows * c.heads,
                c.head_dim,
                c.qk_eps,
            )?;
            g.reshape(
                &node(&format!("{name}_restore")),
                &output(&format!("{name}_norm_heads")),
                &output(&format!("{name}_norm")),
                rows,
                inner,
            )?;
        }
        g.emit(
            &node("attention"),
            SEGMENTED_ATTENTION_BF16_OPERATION_ID,
            vec![
                id(&output("q_norm"))?,
                id(&output("k_norm"))?,
                id(&output("v"))?,
            ],
            &output("attention"),
            vec![rows, inner],
            BTreeMap::from([
                (attr("head_dim")?, SemanticValue::Unsigned(c.head_dim)),
                (attr("scale")?, SemanticValue::Rational(c.attention_scale)),
                (
                    attr("segment_lengths")?,
                    SemanticValue::Integers(c.requests.iter().map(|&n| n as i64).collect()),
                ),
            ]),
        )?;
        g.w8(
            &node("attention_out"),
            &output("attention"),
            &weight("attn.out_proj"),
            &output("attention_out"),
            rows,
            inner,
            h,
            false,
        )?;
        g.add(
            &node("residual1"),
            &previous,
            &output("attention_out"),
            &output("residual1"),
            rows,
            h,
        )?;
        g.norm(
            &node("norm2"),
            &output("residual1"),
            &weight("norm2.weight"),
            &output("norm2"),
            rows,
            h,
            c.norm_eps,
        )?;
        g.w8(
            &node("fc1"),
            &output("norm2"),
            &weight("mlp.fc1"),
            &output("gate_up"),
            rows,
            h,
            2 * c.ffn,
            false,
        )?;
        g.w8(
            &node("fc2"),
            &output("gate_up"),
            &weight("mlp.fc2"),
            &output("mlp"),
            rows,
            c.ffn,
            h,
            true,
        )?;
        g.add(
            &node("residual2"),
            &output("residual1"),
            &output("mlp"),
            &output("residual2"),
            rows,
            h,
        )?;
        previous = output("residual2");
    }
    g.norm(
        "final_norm",
        &previous,
        "token_refiner.final_norm.weight",
        PH3ImageTokenRefinerFamily::OUTPUT_ID,
        rows,
        h,
        c.final_eps,
    )?;
    if g.nodes.len() != c.plan_nodes()? {
        return Err(invalid("graph node count differs from checked geometry"));
    }
    Ok(g)
}
pub(super) fn compile_options(
    c: &PH3ImageTokenRefinerConfig,
) -> Result<ProgramPlanCompileOptions, VNextError> {
    ProgramPlanCompileOptions::new(graph(c)?.specs)
}
pub(super) fn build(
    f: &PH3ImageTokenRefinerFamily,
    c: &PH3ImageTokenRefinerConfig,
) -> Result<ModelProgram, VNextError> {
    let g = graph(c)?;
    if g.nodes.len() > f.limits.max_plan_nodes {
        return Err(invalid("program exceeds declared node capacity"));
    }
    ModelProgram::new(
        f.id.clone(),
        c.input_tensor_specs()?.keys().cloned().collect(),
        vec![ProgramBlock {
            id: "block.p_h3_image.token_refiner".into(),
            nodes: g.nodes,
        }],
        vec![],
        weights::references(c)?,
        vec![id(PH3ImageTokenRefinerFamily::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)
}
