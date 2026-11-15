use super::{family, FastH3PddProjectionConfig, FastH3PddProjectionFamily as F};
use ferrum_runtime::*;
use std::collections::BTreeMap;

fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
pub(super) fn build(
    c: &FastH3PddProjectionConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let input = ProgramValueId::new(F::INPUT_ID)?;
    let output = ProgramValueId::new(F::OUTPUT_ID)?;
    let input_spec = tensor(vec![c.rows, c.hidden]);
    let output_spec = tensor(vec![c.rows, c.output]);
    let input_specs = BTreeMap::from([(input.clone(), input_spec.clone())]);
    let specs = BTreeMap::from([(input.clone(), input_spec), (output.clone(), output_spec)]);
    let operands: Vec<_> = (0..2)
        .map(|i| ProgramValueId::new(format!("value.fast_h3.pdd_projection.parameter.{i}")))
        .collect::<Result<_, _>>()?;
    let weights = c
        .parameters
        .iter()
        .enumerate()
        .map(|(i, p)| {
            Ok(WeightReference {
                weight_id: family::weight(i)?,
                value_id: operands[i].clone(),
                tensor: tensor(p.fused.dimensions.clone()),
            })
        })
        .collect::<Result<_, VNextError>>()?;
    let program = ModelProgram::new(
        ModelFamilyId::new(F::FAMILY_ID)?,
        vec![input.clone()],
        vec![ProgramBlock {
            id: "block.fast_h3.pdd_projection".into(),
            nodes: vec![ProgramNode {
                id: NodeId::new("node.fast_h3.pdd_projection.affine")?,
                operation_id: OperationId::new(AFFINE_LINEAR_F32_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![input, operands[0].clone(), operands[1].clone()],
                outputs: vec![output.clone()],
                attributes: BTreeMap::from([
                    (
                        AttributeId::new("in_features")?,
                        SemanticValue::Unsigned(c.hidden),
                    ),
                    (
                        AttributeId::new("out_features")?,
                        SemanticValue::Unsigned(c.output),
                    ),
                ]),
            }],
        }],
        vec![],
        weights,
        vec![output],
    )?
    .with_input_tensor_specs(input_specs)?;
    Ok((program, specs))
}
