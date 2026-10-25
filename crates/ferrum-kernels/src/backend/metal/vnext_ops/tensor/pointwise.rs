use super::*;
use ferrum_runtime::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum MetalTensorOperation {
    Silu,
    SiluMul,
    Tanh,
    ModulationScale,
    GatedResidual,
    GeluTanh,
    PlainResidualAdd,
    DenseLinear,
    BlockLinear,
}

impl MetalTensorOperation {
    pub const POINTWISE: [Self; 7] = [
        Self::Silu,
        Self::SiluMul,
        Self::Tanh,
        Self::ModulationScale,
        Self::GatedResidual,
        Self::GeluTanh,
        Self::PlainResidualAdd,
    ];
    pub const ALL: [Self; 9] = [
        Self::Silu,
        Self::SiluMul,
        Self::Tanh,
        Self::ModulationScale,
        Self::GatedResidual,
        Self::GeluTanh,
        Self::PlainResidualAdd,
        Self::DenseLinear,
        Self::BlockLinear,
    ];
    pub(super) fn is_linear(self) -> bool {
        matches!(self, Self::DenseLinear | Self::BlockLinear)
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Silu => silu_bf16_contract(),
            Self::SiluMul => silu_mul_bf16_contract(),
            Self::Tanh => tanh_bf16_contract(),
            Self::ModulationScale => modulation_scale_bf16_contract(),
            Self::GatedResidual => gated_residual_bf16_contract(),
            Self::GeluTanh => gelu_tanh_bf16_contract(),
            Self::PlainResidualAdd => plain_residual_add_bf16_contract(),
            Self::DenseLinear => dense_linear_bf16_contract(),
            Self::BlockLinear => dense_linear_block_bf16_contract(),
        }
    }
    pub fn profile(self) -> &'static str {
        match self {
            Self::Silu => "metal.precise.silu.bf16.v1",
            Self::SiluMul => "metal.precise.silu_mul.bf16.v1",
            Self::Tanh => "metal.precise.tanh.bf16.v1",
            Self::ModulationScale => "metal.precise.modulation_scale.bf16.v1",
            Self::GatedResidual => "metal.precise.gated_residual.bf16.v1",
            Self::GeluTanh => "metal.precise.gelu_tanh.bf16.v1",
            Self::PlainResidualAdd => "metal.precise.plain_residual_add.bf16.v1",
            Self::DenseLinear => super::linear::DENSE_PROFILE,
            Self::BlockLinear => super::linear::BLOCK_PROFILE,
        }
    }
    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::Silu => SILU_BF16_CAPABILITY_ID,
            Self::SiluMul => SILU_MUL_BF16_CAPABILITY_ID,
            Self::Tanh => TANH_BF16_CAPABILITY_ID,
            Self::ModulationScale => MODULATION_SCALE_BF16_CAPABILITY_ID,
            Self::GatedResidual => GATED_RESIDUAL_BF16_CAPABILITY_ID,
            Self::GeluTanh => GELU_TANH_BF16_CAPABILITY_ID,
            Self::PlainResidualAdd => PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID,
            Self::DenseLinear => DENSE_LINEAR_BF16_CAPABILITY_ID,
            Self::BlockLinear => DENSE_LINEAR_BLOCK_BF16_CAPABILITY_ID,
        }
    }
    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::Silu => SILU_BF16_OPERATION_ID,
            Self::SiluMul => SILU_MUL_BF16_OPERATION_ID,
            Self::Tanh => TANH_BF16_OPERATION_ID,
            Self::ModulationScale => MODULATION_SCALE_BF16_OPERATION_ID,
            Self::GatedResidual => GATED_RESIDUAL_BF16_OPERATION_ID,
            Self::GeluTanh => GELU_TANH_BF16_OPERATION_ID,
            Self::PlainResidualAdd => PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            Self::DenseLinear => DENSE_LINEAR_BF16_OPERATION_ID,
            Self::BlockLinear => DENSE_LINEAR_BLOCK_BF16_OPERATION_ID,
        }
    }
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Silu => "silu",
            Self::SiluMul => "silu_mul",
            Self::Tanh => "tanh",
            Self::ModulationScale => "modulation_scale",
            Self::GatedResidual => "gated_residual",
            Self::GeluTanh => "gelu_tanh",
            Self::PlainResidualAdd => "plain_residual_add",
            Self::DenseLinear => "dense_linear",
            Self::BlockLinear => "native_block_linear",
        }
    }
    pub(super) fn inputs(self) -> u32 {
        match self {
            Self::Silu | Self::Tanh | Self::GeluTanh => 1,
            Self::SiluMul | Self::ModulationScale | Self::PlainResidualAdd => 2,
            Self::GatedResidual => 3,
            Self::DenseLinear | Self::BlockLinear => 2,
        }
    }
    pub(super) fn broadcasts(self) -> bool {
        matches!(self, Self::ModulationScale | Self::GatedResidual)
    }
    fn operations(self) -> u64 {
        match self {
            Self::Silu => 4,
            Self::SiluMul => 5,
            Self::Tanh | Self::PlainResidualAdd => 1,
            Self::ModulationScale | Self::GatedResidual => 2,
            Self::GeluTanh => 9,
            Self::DenseLinear | Self::BlockLinear => {
                unreachable!("matrix work is computed by its own geometry")
            }
        }
    }
}

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
    if kind.is_linear() {
        return Err("matrix operations require the linear provider".into());
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
            .checked_mul(kind.operations())
            .ok_or("Metal pointwise work overflows")?,
        row_broadcast,
    })
}
