use super::*;
pub(super) fn build(
    c: &FastH3PooledAttentionConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    type F = FastH3PooledAttentionFamily;
    let inputs = [F::QUERY_ID, F::KEY_ID, F::VALUE_ID]
        .into_iter()
        .map(|s| Ok((id(s)?, tensor(c.pooled_shape()))))
        .collect::<Result<BTreeMap<ProgramValueId, _>, VNextError>>()?;
    let raw = "value.fast_h3.vsa.pooled_attention.raw_scores";
    let probabilities = "value.fast_h3.vsa.pooled_attention.probabilities";
    let mut specs = inputs.clone();
    for name in [raw, probabilities, F::SCORES_ID] {
        specs.insert(id(name)?, tensor(c.score_shape()));
    }
    specs.insert(id(F::OUTPUT_ID)?, tensor(c.pooled_shape()));
    let node = |name: &str,
                op: BatchedF32Op,
                inputs: Vec<ProgramValueId>,
                output: &str,
                attributes|
     -> Result<ProgramNode, VNextError> {
        Ok(ProgramNode {
            id: id(&format!("node.fast_h3.vsa.pooled_attention.{name}"))?,
            operation_id: id(op.operation_id())?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![id(output)?],
            attributes,
        })
    };
    let nodes = vec![
        node(
            "qk",
            BatchedF32Op::Matmul,
            vec![id(F::QUERY_ID)?, id(F::KEY_ID)?],
            raw,
            BatchedF32Geometry::matmul_attributes(true)?,
        )?,
        node(
            "scale",
            BatchedF32Op::Scale,
            vec![id(raw)?],
            F::SCORES_ID,
            BatchedF32Geometry::scale_attributes(f32::from_bits(c.scale_bits))?,
        )?,
        node(
            "softmax",
            BatchedF32Op::Softmax,
            vec![id(F::SCORES_ID)?],
            probabilities,
            BTreeMap::new(),
        )?,
        node(
            "pv",
            BatchedF32Op::Matmul,
            vec![id(probabilities)?, id(F::VALUE_ID)?],
            F::OUTPUT_ID,
            BatchedF32Geometry::matmul_attributes(false)?,
        )?,
    ];
    let p = ModelProgram::new(
        id(F::FAMILY_ID)?,
        inputs.keys().cloned().collect(),
        vec![ProgramBlock {
            id: "block.fast_h3.vsa.pooled_attention".into(),
            nodes,
        }],
        vec![],
        vec![],
        vec![id(F::SCORES_ID)?, id(F::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(inputs)?;
    Ok((p, specs))
}
