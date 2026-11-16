use super::*;
use crate::row_chunks::RowChunks;
use std::num::NonZeroU64;

fn chunks(c: &FastH3FfnConfig) -> Result<RowChunks, VNextError> {
    let rows = NonZeroU64::new(c.rows)
        .filter(|n| n.get() <= i64::MAX as u64)
        .ok_or_else(|| invalid("FFN requires positive addressable rows"))?;
    Ok(RowChunks::new(rows, c.chunk_rows.unwrap_or(rows)))
}

pub(super) fn node_count(c: &FastH3FfnConfig) -> Result<u64, VNextError> {
    let count = chunks(c)?.count();
    if count == 1 {
        return Ok(5);
    }
    count
        .checked_mul(7)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| invalid("FFN chunk node count overflow"))
}

/// The fragments share one physical weight schema. Explicit transport nodes
/// preserve row order, including a short last chunk, and the ordinary serial
/// plan decides temporary lifetimes. There is no hidden runtime chunk loop.
pub(super) fn build(
    c: &FastH3FfnConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let chunks = chunks(c)?;
    if chunks.count() == 1 {
        return unchunked(c);
    }
    let input = ProgramValueId::new(FastH3FfnFamily::INPUT_ID)?;
    let output = ProgramValueId::new(FastH3FfnFamily::OUTPUT_ID)?;
    let input_specs = BTreeMap::from([(
        input.clone(),
        tensor(vec![c.rows, c.hidden], ElementType::Bf16),
    )]);
    let mut specs = input_specs.clone();
    let mut nodes = Vec::new();
    let mut weights = None;
    // Initialize one complete output with an ordinary copy. Row writes keep
    // all untouched bits defined, and each completed chunk can retire at once.
    // Core retains the same backing through the explicit MustAlias chain.
    let mut assembled = ProgramValueId::new(format!("{}.initial", output.as_str()))?;
    specs.insert(
        assembled.clone(),
        tensor(vec![c.rows, c.hidden], ElementType::Bf16),
    );
    nodes.push(gather_rows(
        "initial",
        input.clone(),
        assembled.clone(),
        0,
        c.rows,
        c.hidden,
    )?);
    for (index, range) in chunks.ranges().enumerate() {
        let rows = range.end - range.start;
        let piece = ProgramValueId::new(format!("{}.chunk{index}.input", input.as_str()))?;
        specs.insert(
            piece.clone(),
            tensor(vec![rows, c.hidden], ElementType::Bf16),
        );
        nodes.push(gather_rows(
            &format!("chunk{index}.input"),
            input.clone(),
            piece.clone(),
            range.start,
            rows,
            c.hidden,
        )?);
        let mut part = c.clone();
        part.rows = rows;
        part.chunk_rows = None;
        let (body, geometry) = unchunked(&part)?;
        let weight_values = body
            .weights()
            .iter()
            .map(|w| &w.value_id)
            .collect::<BTreeSet<_>>();
        let remap = |id: &ProgramValueId| -> Result<ProgramValueId, VNextError> {
            if id == &input {
                return Ok(piece.clone());
            }
            if weight_values.contains(id) {
                return Ok(id.clone());
            }
            ProgramValueId::new(format!("{}.chunk{index}", id.as_str()))
        };
        for (id, shape) in geometry {
            specs.insert(remap(&id)?, shape);
        }
        for original in body.blocks().iter().flat_map(|b| &b.nodes) {
            let mut n = original.clone();
            n.id = NodeId::new(format!("{}.chunk{index}", n.id.as_str()))?;
            n.inputs = n.inputs.iter().map(&remap).collect::<Result<_, _>>()?;
            n.outputs = n.outputs.iter().map(&remap).collect::<Result<_, _>>()?;
            nodes.push(n);
        }
        let next = if range.end == c.rows {
            output.clone()
        } else {
            ProgramValueId::new(format!("{}.partial{index}", output.as_str()))?
        };
        specs.insert(
            next.clone(),
            tensor(vec![c.rows, c.hidden], ElementType::Bf16),
        );
        nodes.push(node(
            &format!("chunk{index}.write"),
            ROW_WRITE_BF16_OPERATION_ID,
            vec![assembled, remap(&output)?],
            vec![next.clone()],
            BTreeMap::from([(
                AttributeId::new("row_start")?,
                SemanticValue::Unsigned(range.start),
            )]),
        )?);
        assembled = next;
        match &weights {
            Some(original) if original != body.weights() => {
                return Err(invalid("row chunks changed physical weights"))
            }
            None => weights = Some(body.weights().to_vec()),
            _ => {}
        }
    }
    let program = ModelProgram::new(
        ModelFamilyId::new(FastH3FfnFamily::FAMILY_ID)?,
        vec![input],
        vec![ProgramBlock {
            id: "block.fast_h3.ffn".into(),
            nodes,
        }],
        vec![],
        weights.ok_or_else(|| invalid("empty row partition"))?,
        vec![output],
    )?
    .with_input_tensor_specs(input_specs)?;
    Ok((program, specs))
}

fn gather_rows(
    name: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    start: u64,
    rows: u64,
    columns: u64,
) -> Result<ProgramNode, VNextError> {
    node(
        name,
        ROW_GATHER_BF16_OPERATION_ID,
        vec![input],
        vec![output],
        BTreeMap::from([
            (AttributeId::new("batch")?, SemanticValue::Unsigned(1)),
            (
                AttributeId::new("column_start")?,
                SemanticValue::Unsigned(0),
            ),
            (
                AttributeId::new("column_count")?,
                SemanticValue::Unsigned(columns),
            ),
            (
                AttributeId::new("run_lengths")?,
                SemanticValue::Integers(vec![rows as i64]),
            ),
            (
                AttributeId::new("source_starts")?,
                SemanticValue::Integers(vec![start as i64]),
            ),
            (
                AttributeId::new("source_batch_strides")?,
                SemanticValue::Integers(vec![0]),
            ),
            (
                AttributeId::new("source_row_strides")?,
                SemanticValue::Integers(vec![1]),
            ),
        ]),
    )
}

fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn node(
    name: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    outputs: Vec<ProgramValueId>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.fast_h3.ffn.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs,
        attributes,
    })
}

fn unchunked(
    c: &FastH3FfnConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let input = ProgramValueId::new(FastH3FfnFamily::INPUT_ID)?;
    let output = ProgramValueId::new(FastH3FfnFamily::OUTPUT_ID)?;
    let input_specs = BTreeMap::from([(
        input.clone(),
        tensor(vec![c.rows, c.hidden], ElementType::Bf16),
    )]);
    let mut specs = input_specs.clone();
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    let schema = c.schema()?;
    let mut current = input.clone();
    for (index, p) in c.projections.iter().enumerate() {
        for suffix in ["packed", "scales", "global_sf", "alpha"] {
            let id = weight(index, suffix)?;
            let w = schema
                .tensors
                .iter()
                .find(|w| w.id == id)
                .ok_or_else(|| invalid("missing FFN operand"))?;
            weights.push(WeightReference {
                weight_id: id,
                value_id: value(index, suffix)?,
                tensor: tensor(w.dimensions.clone(), w.logical_element_type),
            });
        }
        let packed = value(index, "activation.packed")?;
        let scales = value(index, "activation.scales")?;
        let activation = Nvfp4MatrixLayout::new(c.rows, p.columns)?;
        specs.insert(
            packed.clone(),
            tensor(activation.packed_shape().to_vec(), ElementType::U8),
        );
        specs.insert(
            scales.clone(),
            tensor(activation.scale_shape().to_vec(), ElementType::U8),
        );
        nodes.push(node(
            &format!("{index}.quantize"),
            NVFP4_QUANTIZE_BF16_OPERATION_ID,
            vec![current, value(index, "global_sf")?],
            vec![packed.clone(), scales.clone()],
            Nvfp4QuantizeProfile::CudaRcpApproxFtzV1.attributes()?,
        )?);
        let projected = if index == 1 {
            output.clone()
        } else {
            value(index, "projected")?
        };
        nodes.push(node(
            &format!("{index}.matmul"),
            NVFP4_MATMUL_BF16_OPERATION_ID,
            vec![
                packed,
                value(index, "packed")?,
                scales,
                value(index, "scales")?,
                value(index, "alpha")?,
            ],
            vec![projected.clone()],
            BTreeMap::new(),
        )?);
        specs.insert(
            projected.clone(),
            tensor(vec![c.rows, p.rows], ElementType::Bf16),
        );
        if index == 0 {
            // Fixed FastVideo all-fusion profile: packed value/gate inputs,
            // complete sigmoid and products in F32, one final BF16 conversion.
            current = value(0, "activated")?;
            nodes.push(node(
                "swiglu",
                PACKED_SWIGLU_BF16_OPERATION_ID,
                vec![projected],
                vec![current.clone()],
                PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound.attributes()?,
            )?);
            specs.insert(
                current.clone(),
                tensor(vec![c.rows, c.intermediate], ElementType::Bf16),
            );
        } else {
            current = projected;
        }
    }
    let program = ModelProgram::new(
        ModelFamilyId::new(FastH3FfnFamily::FAMILY_ID)?,
        vec![input],
        vec![ProgramBlock {
            id: "block.fast_h3.ffn".into(),
            nodes,
        }],
        vec![],
        weights,
        vec![output],
    )?
    .with_input_tensor_specs(input_specs)?;
    Ok((program, specs))
}
