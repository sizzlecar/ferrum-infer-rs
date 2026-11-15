use super::*;

struct Graph {
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    rows: u64,
}
impl Graph {
    fn emit(
        &mut self,
        name: &str,
        operation: &str,
        inputs: Vec<ProgramValueId>,
        width: u64,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<ProgramValueId, VNextError> {
        let output: ProgramValueId = id(&format!("value.fast_h3.text_refiner.{name}"))?;
        let spec = tensor(vec![self.rows, width]);
        spec.validate(output.as_str())?;
        if spec.byte_len()? > isize::MAX as u64 || self.specs.insert(output.clone(), spec).is_some()
        {
            return Err(invalid("duplicate or oversized intermediate"));
        }
        self.nodes.push(ProgramNode {
            id: id(&format!("node.fast_h3.text_refiner.{name}"))?,
            operation_id: id(operation)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output.clone()],
            attributes,
        });
        Ok(output)
    }
    fn norm(
        &mut self,
        name: &str,
        x: ProgramValueId,
        w: usize,
        width: u64,
        head: u64,
        epsilon: CanonicalRational,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            vec![x, weight_value(w)?],
            width,
            BTreeMap::from([
                (id("head_dim")?, SemanticValue::Unsigned(head)),
                (id("epsilon")?, SemanticValue::Rational(epsilon)),
            ]),
        )
    }
    fn dense(
        &mut self,
        name: &str,
        x: ProgramValueId,
        w: usize,
        k: u64,
        n: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            DENSE_LINEAR_BF16_OPERATION_ID,
            vec![x, weight_value(w)?],
            n,
            BTreeMap::from([
                (id("in_features")?, SemanticValue::Unsigned(k)),
                (id("out_features")?, SemanticValue::Unsigned(n)),
            ]),
        )
    }
    fn slice(
        &mut self,
        name: &str,
        x: ProgramValueId,
        start: u64,
        width: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            ROW_GATHER_BF16_OPERATION_ID,
            vec![x],
            width,
            BTreeMap::from([
                (id("batch")?, SemanticValue::Unsigned(1)),
                (id("column_start")?, SemanticValue::Unsigned(start)),
                (id("column_count")?, SemanticValue::Unsigned(width)),
                (
                    id("run_lengths")?,
                    SemanticValue::Integers(vec![self.rows as i64]),
                ),
                (id("source_starts")?, SemanticValue::Integers(vec![0])),
                (
                    id("source_batch_strides")?,
                    SemanticValue::Integers(vec![0]),
                ),
                (id("source_row_strides")?, SemanticValue::Integers(vec![1])),
            ]),
        )
    }
}
pub(super) fn build(
    c: &FastH3TextRefinerConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    c.check_inventory()?;
    type F = FastH3TextRefinerFamily;
    let rows = c.rows()?;
    let inner = c.inner()?;
    let input: ProgramValueId = id(F::INPUT_ID)?;
    let inputs = BTreeMap::from([(input.clone(), tensor(vec![rows, c.hidden]))]);
    let mut g = Graph {
        nodes: Vec::new(),
        specs: inputs.clone(),
        rows,
    };
    let mut previous = input.clone();
    for layer in 0..c.layers {
        let name = |s: &str| format!("layer{layer}.{s}");
        let w = 10 * layer;
        let n = g.norm(
            &name("norm1"),
            previous.clone(),
            w,
            c.hidden,
            c.hidden,
            c.norm_epsilon,
        )?;
        let q = g.dense(&name("q"), n.clone(), w + 4, c.hidden, inner)?;
        let k = g.dense(&name("k"), n.clone(), w + 5, c.hidden, inner)?;
        let v = g.dense(&name("v"), n, w + 6, c.hidden, inner)?;
        let q = g.norm(&name("q_norm"), q, w + 2, inner, c.head_dim, c.qk_epsilon)?;
        let k = g.norm(&name("k_norm"), k, w + 3, inner, c.head_dim, c.qk_epsilon)?;
        let a = g.emit(
            &name("attention"),
            SEGMENTED_ATTENTION_BF16_OPERATION_ID,
            vec![q, k, v],
            inner,
            BTreeMap::from([
                (id("head_dim")?, SemanticValue::Unsigned(c.head_dim)),
                (
                    id("scale")?,
                    SemanticValue::Rational(rational(1.0 / (c.head_dim as f64).sqrt())?),
                ),
                (
                    id("segment_lengths")?,
                    SemanticValue::Integers(c.requests.iter().map(|&r| r as i64).collect()),
                ),
            ]),
        )?;
        let a = g.dense(&name("attention_out"), a, w + 7, inner, c.hidden)?;
        let residual = g.emit(
            &name("residual1"),
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![previous, a],
            c.hidden,
            BTreeMap::new(),
        )?;
        let n = g.norm(
            &name("norm2"),
            residual.clone(),
            w + 1,
            c.hidden,
            c.hidden,
            c.norm_epsilon,
        )?;
        let packed = g.dense(&name("fc_in"), n, w + 8, c.hidden, 2 * c.ffn)?;
        let value = g.slice(&name("value"), packed.clone(), 0, c.ffn)?;
        let gate = g.slice(&name("gate"), packed, c.ffn, c.ffn)?;
        let ff = g.emit(
            &name("swiglu"),
            SILU_MUL_BF16_OPERATION_ID,
            vec![gate, value],
            c.ffn,
            BTreeMap::new(),
        )?;
        let ff = g.dense(&name("fc_out"), ff, w + 9, c.ffn, c.hidden)?;
        previous = g.emit(
            &name("residual2"),
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![residual, ff],
            c.hidden,
            BTreeMap::new(),
        )?;
    }
    let output = g.norm(
        "refined_text",
        previous,
        10 * c.layers,
        c.hidden,
        c.hidden,
        c.final_epsilon,
    )?;
    let weights = c
        .components
        .iter()
        .enumerate()
        .map(|(i, w)| {
            Ok(WeightReference {
                weight_id: weight_id(i)?,
                value_id: weight_value(i)?,
                tensor: tensor(w.dimensions.clone()),
            })
        })
        .collect::<Result<_, VNextError>>()?;
    Ok((
        ModelProgram::new(
            id(F::FAMILY_ID)?,
            vec![input],
            vec![ProgramBlock {
                id: "block.fast_h3.text_refiner".into(),
                nodes: g.nodes,
            }],
            vec![],
            weights,
            vec![output],
        )?
        .with_input_tensor_specs(inputs)?,
        g.specs,
    ))
}
