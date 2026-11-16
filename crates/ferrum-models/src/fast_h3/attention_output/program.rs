use super::*;
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
pub(super) fn build(
    c: &FastH3AttentionOutputConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    type F = FastH3AttentionOutputFamily;
    let width = c.attention_width()?;
    let input_specs = [F::FINE_ID, F::POOLED_ID, F::GATE_ID]
        .into_iter()
        .map(|name| Ok((id(name)?, tensor(vec![c.rows, width], ElementType::Bf16))))
        .collect::<Result<BTreeMap<ProgramValueId, _>, VNextError>>()?;
    let mut specs = input_specs.clone();
    let merged = value("merged")?;
    let packed = value("activation.packed")?;
    let scales = value("activation.scales")?;
    let matrix = Nvfp4MatrixLayout::new(c.rows, width)?;
    specs.extend([
        (
            merged.clone(),
            tensor(vec![c.rows, width], ElementType::Bf16),
        ),
        (
            packed.clone(),
            tensor(matrix.packed_shape().to_vec(), ElementType::U8),
        ),
        (
            scales.clone(),
            tensor(matrix.scale_shape().to_vec(), ElementType::U8),
        ),
        (
            id(F::OUTPUT_ID)?,
            tensor(vec![c.rows, c.hidden], ElementType::Bf16),
        ),
    ]);
    let node = |name: &str,
                operation_id: &str,
                inputs,
                outputs,
                attributes|
     -> Result<ProgramNode, VNextError> {
        Ok(ProgramNode {
            id: id(&format!("node.fast_h3.attention_output.{name}"))?,
            operation_id: id(operation_id)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs,
            attributes,
        })
    };
    let nodes = vec![
        node(
            "merge",
            ADDCMUL_FMA_BF16_OPERATION_ID,
            vec![id(F::FINE_ID)?, id(F::POOLED_ID)?, id(F::GATE_ID)?],
            vec![merged.clone()],
            BTreeMap::from([(id("broadcast")?, SemanticValue::Text("elementwise".into()))]),
        )?,
        node(
            "quantize",
            NVFP4_QUANTIZE_BF16_OPERATION_ID,
            vec![merged, value("unit")?],
            vec![packed.clone(), scales.clone()],
            Nvfp4QuantizeProfile::CudaRcpApproxFtzV1.attributes()?,
        )?,
        node(
            "to_out",
            NVFP4_MATMUL_BF16_OPERATION_ID,
            vec![
                packed,
                value("packed")?,
                scales,
                value("scales")?,
                value("alpha")?,
            ],
            vec![id(F::OUTPUT_ID)?],
            BTreeMap::new(),
        )?,
    ];
    let weights = c
        .schema()?
        .tensors
        .into_iter()
        .map(|w| {
            let suffix =
                w.id.as_str()
                    .strip_prefix("weight.fast_h3.attention_output.")
                    .ok_or_else(|| invalid("weight namespace differs"))?;
            Ok(WeightReference {
                value_id: value(suffix)?,
                tensor: tensor(w.dimensions, w.logical_element_type),
                weight_id: w.id,
            })
        })
        .collect::<Result<_, VNextError>>()?;
    let program = ModelProgram::new(
        id(F::FAMILY_ID)?,
        input_specs.keys().cloned().collect(),
        vec![ProgramBlock {
            id: "block.fast_h3.attention_output".into(),
            nodes,
        }],
        vec![],
        weights,
        vec![id(F::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(input_specs)?;
    Ok((program, specs))
}
