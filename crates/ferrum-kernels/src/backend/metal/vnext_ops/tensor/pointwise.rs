use super::*;
use ferrum_runtime::*;

#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub elements: u64,
    pub width: u64,
    pub bytes: u64,
    pub parameter_bytes: u64,
    pub operations: u64,
    pub row_broadcast: bool,
}

pub(super) fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if !kind.is_pointwise() {
        return Err("operation requires its own non-pointwise geometry".into());
    }
    if values.len() != kind.inputs() as usize + 1 {
        return Err("Metal pointwise port count differs".into());
    }
    let input = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let output = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    let (rows, width, row_broadcast) = if kind.broadcasts() {
        if attributes.len() != 1 {
            return Err("Metal pointwise needs exactly the broadcast attribute".into());
        }
        let mode = match attributes.iter().next() {
            Some((key, SemanticValue::Text(value))) if key.as_str() == "broadcast" => {
                RowBroadcast::parse(value).ok_or("unsupported Metal pointwise broadcast")?
            }
            _ => return Err("Metal pointwise broadcast must be text".into()),
        };
        if kind == MetalTensorOperation::GatedResidual
            && binding(values, ResolvedValueRole::Input, 2)?.tensor() != input
        {
            return Err("Metal gated residual update must match its base".into());
        }
        let parameter = binding(values, ResolvedValueRole::Input, 1)?.tensor();
        let shape = Bf16RowBroadcastGeometry::new(input, parameter, output, mode)
            .map_err(|e| e.to_string())?;
        (shape.rows(), shape.width(), mode == RowBroadcast::Row)
    } else {
        if !attributes.is_empty() {
            return Err("Metal same-shape pointwise accepts no attributes".into());
        }
        if kind.inputs() == 2 && binding(values, ResolvedValueRole::Input, 1)?.tensor() != input {
            return Err("Metal binary pointwise operands must match exactly".into());
        }
        let shape = Bf16SameShapeGeometry::new(input, output).map_err(|e| e.to_string())?;
        (shape.rows(), shape.width(), false)
    };
    let elements = rows
        .checked_mul(width)
        .filter(|&n| n > 0 && n <= u32::MAX as u64)
        .ok_or("Metal pointwise extent exceeds its dispatch index range")?;
    let bytes = elements
        .checked_mul(2)
        .ok_or("Metal pointwise byte extent overflows")?;
    Ok(Shape {
        elements,
        width,
        bytes,
        parameter_bytes: if row_broadcast { width * 2 } else { bytes },
        operations: elements
            .checked_mul(kind.pointwise_operations())
            .ok_or("Metal pointwise work overflows")?,
        row_broadcast,
    })
}
