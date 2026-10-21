//! Shared estimation/encoding checks; no payload access or resource authority.

use ferrum_runtime::{
    Bf16RowBroadcastGeometry, Bf16SameShapeGeometry, ResolvedTensorSpec, RowBroadcast,
};

use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    pub(super) width: u64,
    pub(super) elements: u64,
    pub(super) bytes: u64,
    pub(super) parameter_bytes: u64,
    pub(super) arithmetic_operations: u64,
    pub(super) row_broadcast: bool,
}

impl Shape {
    pub(super) fn new(
        kind: Kind,
        rows: u64,
        width: u64,
        row_broadcast: bool,
    ) -> Result<Self, String> {
        if rows == 0 || width == 0 || (row_broadcast && !kind.broadcasts()) {
            return Err(
                "CUDA pointwise requires positive dimensions and explicit compatible broadcast"
                    .into(),
            );
        }
        let elements = rows
            .checked_mul(width)
            .ok_or("CUDA pointwise element count overflows")?;
        let bytes = elements
            .checked_mul(2)
            .ok_or("CUDA pointwise byte length overflows")?;
        if bytes > isize::MAX as u64 {
            return Err("CUDA pointwise extent exceeds the host address space".into());
        }
        let arithmetic_operations = elements
            .checked_mul(kind.operations_per_element())
            .ok_or("CUDA pointwise logical arithmetic work overflows")?;
        // A positive matrix extent has already bounded width * 2.
        let parameter_bytes = if row_broadcast { width * 2 } else { bytes };
        Ok(Self {
            width,
            elements,
            bytes,
            parameter_bytes,
            arithmetic_operations,
            row_broadcast,
        })
    }
}

pub(super) fn geometry(
    kind: Kind,
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != kind.inputs() as usize + 1 {
        return Err("CUDA pointwise has an incorrect port count".into());
    }
    let mode = if kind.broadcasts() {
        if attributes.len() != 1 {
            return Err(
                "CUDA broadcast pointwise requires only its explicit broadcast attribute".into(),
            );
        }
        let Some(SemanticValue::Text(value)) = attributes
            .iter()
            .find(|(key, _)| key.as_str() == "broadcast")
            .map(|(_, value)| value)
        else {
            return Err("CUDA pointwise broadcast must be text".into());
        };
        Some(RowBroadcast::parse(value).ok_or("unsupported CUDA pointwise broadcast mode")?)
    } else {
        if !attributes.is_empty() {
            return Err("CUDA same-shape pointwise accepts no attributes".into());
        }
        None
    };
    let inputs = (0..kind.inputs())
        .map(|ordinal| {
            binding(bindings, ResolvedValueRole::Input, ordinal).map(|binding| binding.tensor())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?.tensor();
    validate_specs(kind, &inputs, output, mode)
}

pub(super) fn validate_specs(
    kind: Kind,
    inputs: &[&ResolvedTensorSpec],
    output: &ResolvedTensorSpec,
    mode: Option<RowBroadcast>,
) -> Result<Shape, String> {
    if inputs.len() != kind.inputs() as usize {
        return Err("CUDA pointwise has an incorrect input count".into());
    }
    let input = inputs[0];
    if kind.broadcasts() {
        let mode = mode.ok_or("CUDA pointwise requires an explicit broadcast mode")?;
        if kind == Kind::GatedResidual && inputs[2] != input {
            return Err("CUDA gated residual update must exactly match its base tensor".into());
        }
        let geometry = Bf16RowBroadcastGeometry::new(input, inputs[1], output, mode)
            .map_err(|error| error.to_string())?;
        Shape::new(
            kind,
            geometry.rows(),
            geometry.width(),
            mode == RowBroadcast::Row,
        )
    } else {
        if mode.is_some() {
            return Err("CUDA same-shape pointwise cannot broadcast".into());
        }
        if kind == Kind::SiluMul && inputs[1] != input {
            return Err(
                "CUDA SiLU-mul operands must have exactly equal tensor specifications".into(),
            );
        }
        let geometry =
            Bf16SameShapeGeometry::new(input, output).map_err(|error| error.to_string())?;
        Shape::new(kind, geometry.rows(), geometry.width(), false)
    }
}
