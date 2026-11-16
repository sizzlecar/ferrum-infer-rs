//! Explicit two-axis batching for dense F32 products and row normalization.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const BATCHED_MATMUL_F32_OPERATION_ID: &str = "operation.batched_matmul.f32";
pub const BATCHED_MATMUL_F32_CAPABILITY_ID: &str = "capability.operation.batched_matmul.f32";
pub const SCALE_RANK4_F32_OPERATION_ID: &str = "operation.scale_rank4.f32";
pub const SCALE_RANK4_F32_CAPABILITY_ID: &str = "capability.operation.scale_rank4.f32";
pub const ROW_SOFTMAX_RANK4_F32_OPERATION_ID: &str = "operation.row_softmax_rank4.f32";
pub const ROW_SOFTMAX_RANK4_F32_CAPABILITY_ID: &str = "capability.operation.row_softmax_rank4.f32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchedF32Op {
    Matmul,
    Scale,
    Softmax,
}
impl BatchedF32Op {
    pub const ALL: [Self; 3] = [Self::Matmul, Self::Scale, Self::Softmax];
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Matmul => BATCHED_MATMUL_F32_OPERATION_ID,
            Self::Scale => SCALE_RANK4_F32_OPERATION_ID,
            Self::Softmax => ROW_SOFTMAX_RANK4_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Matmul => BATCHED_MATMUL_F32_CAPABILITY_ID,
            Self::Scale => SCALE_RANK4_F32_CAPABILITY_ID,
            Self::Softmax => ROW_SOFTMAX_RANK4_F32_CAPABILITY_ID,
        }
    }
    pub const fn input_count(self) -> u32 {
        if matches!(self, Self::Matmul) {
            2
        } else {
            1
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let left = vec![symbol("batch"), symbol("group"), symbol("m"), symbol("k")];
        let mut inputs = vec![contiguous_tensor(
            left.clone(),
            [ElementType::F32],
            TensorAccess::Read,
        )?];
        let (output, attrs) = match self {
            Self::Matmul => {
                inputs.push(contiguous_tensor(
                    vec![
                        symbol("batch"),
                        symbol("group"),
                        symbol("rhs_rows"),
                        symbol("rhs_columns"),
                    ],
                    [ElementType::F32],
                    TensorAccess::Read,
                )?);
                (
                    vec![symbol("batch"), symbol("group"), symbol("m"), symbol("n")],
                    BTreeMap::from([(
                        AttributeId::new("transpose_rhs")?,
                        AttributeSpec {
                            value_kind: AttributeValueKind::Bool,
                            required: true,
                            constraint: AttributeConstraint::None,
                        },
                    )]),
                )
            }
            Self::Scale => (
                left,
                BTreeMap::from([(
                    AttributeId::new("scale_bits")?,
                    AttributeSpec {
                        value_kind: AttributeValueKind::Unsigned,
                        required: true,
                        constraint: AttributeConstraint::UnsignedRange {
                            minimum: 0,
                            maximum: u32::MAX as u64,
                        },
                    },
                )]),
            ),
            Self::Softmax => (left, BTreeMap::new()),
        };
        let descriptor = OperationDescriptor {
            id: OperationId::new(self.operation_id())?,
            version: ContractVersion::new(1, 0),
            inputs,
            outputs: vec![contiguous_tensor(
                output,
                [ElementType::F32],
                TensorAccess::Write,
            )?],
            attributes: AttributeSchema::new(attrs)?,
            resources: no_auxiliary_resources(),
            oracle: OracleSpec::OperationDefined,
            provider: provider_requirement(self.capability_id(), ContractVersion::new(1, 0))?,
            profile_phase: ProfilePhase::Forward,
        };
        descriptor.validate()?;
        Ok(StandardOperationContract { descriptor })
    }
}

/// A[B,G,M,K] times B[B,G,K,N], or transpose of B[B,G,N,K]. Both
/// batch axes must agree exactly; there is no broadcasting, alpha, bias, TF32
/// permission or implicit dtype conversion. The provider declares reduction
/// order. The output is a separately materialized F32 boundary.
pub fn batched_matmul_f32_contract() -> Result<StandardOperationContract, VNextError> {
    BatchedF32Op::Matmul.contract()
}
/// Elementwise RN32(x * finite_scalar), including gradual underflow and IEEE
/// signed zero. The exact F32 scalar encoding is an explicit attribute.
pub fn scale_rank4_f32_contract() -> Result<StandardOperationContract, VNextError> {
    BatchedF32Op::Scale.contract()
}
/// Last-axis stable softmax. Providers declare exp and reduction order. NaN,
/// positive infinity or an all-negative-infinity row produces all NaNs; other
/// negative infinity entries yield positive zero. No BF16 probability boundary.
pub fn row_softmax_rank4_f32_contract() -> Result<StandardOperationContract, VNextError> {
    BatchedF32Op::Softmax.contract()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedF32Geometry {
    op: BatchedF32Op,
    batch: u64,
    group: u64,
    m: u64,
    n: u64,
    k: u64,
    transpose_rhs: bool,
    scale_bits: u32,
}
impl BatchedF32Geometry {
    pub fn matmul_attributes(
        transpose_rhs: bool,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([(
            AttributeId::new("transpose_rhs")?,
            SemanticValue::Bool(transpose_rhs),
        )]))
    }
    pub fn scale_attributes(
        scale: f32,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        if !scale.is_finite() {
            return Err(invalid("scale must be finite"));
        }
        Ok(BTreeMap::from([(
            AttributeId::new("scale_bits")?,
            SemanticValue::Unsigned(u64::from(scale.to_bits())),
        )]))
    }
    pub fn new(
        op: BatchedF32Op,
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        op.contract()?.descriptor.validate_attributes(attrs)?;
        if inputs.len() != op.input_count() as usize {
            return Err(invalid("wrong input count"));
        }
        for t in inputs.iter().copied().chain(std::iter::once(output)) {
            if t.dimensions().len() != 4
                || t.dimensions().contains(&0)
                || t.element_type() != ElementType::F32
                || t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid(
                    "requires positive contiguous addressable rank-four F32 tensors",
                ));
            }
        }
        let x = inputs[0].dimensions();
        let mut g = Self {
            op,
            batch: x[0],
            group: x[1],
            m: x[2],
            n: x[3],
            k: x[3],
            transpose_rhs: false,
            scale_bits: 1f32.to_bits(),
        };
        match op {
            BatchedF32Op::Matmul => {
                let Some(SemanticValue::Bool(transpose)) =
                    attrs.get(&AttributeId::new("transpose_rhs")?)
                else {
                    return Err(invalid("missing transpose_rhs"));
                };
                g.transpose_rhs = *transpose;
                let rhs = inputs[1].dimensions();
                let (k, n) = if *transpose {
                    (rhs[3], rhs[2])
                } else {
                    (rhs[2], rhs[3])
                };
                g.n = n;
                if rhs[..2] != x[..2]
                    || k != g.k
                    || output.dimensions() != [g.batch, g.group, g.m, n]
                {
                    return Err(invalid(
                        "matmul axes differ; batch broadcasting is not supported",
                    ));
                }
            }
            BatchedF32Op::Scale => {
                let Some(SemanticValue::Unsigned(bits)) =
                    attrs.get(&AttributeId::new("scale_bits")?)
                else {
                    return Err(invalid("missing scale_bits"));
                };
                g.scale_bits =
                    u32::try_from(*bits).map_err(|_| invalid("scale encoding exceeds F32"))?;
                if !g.scale().is_finite() {
                    return Err(invalid("scale must be finite"));
                }
                if output != inputs[0] {
                    return Err(invalid("scale output shape differs"));
                }
            }
            BatchedF32Op::Softmax => {
                if output != inputs[0] {
                    return Err(invalid("softmax output shape differs"));
                }
            }
        }
        g.arithmetic_operations()?;
        Ok(g)
    }
    pub const fn op(self) -> BatchedF32Op {
        self.op
    }
    pub const fn batch(self) -> u64 {
        self.batch
    }
    pub const fn group(self) -> u64 {
        self.group
    }
    pub const fn m(self) -> u64 {
        self.m
    }
    pub const fn n(self) -> u64 {
        self.n
    }
    pub const fn k(self) -> u64 {
        self.k
    }
    pub const fn transpose_rhs(self) -> bool {
        self.transpose_rhs
    }
    pub fn scale(self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
    pub fn rows(self) -> u64 {
        self.batch * self.group * self.m
    }
    pub fn output_elements(self) -> u64 {
        self.rows() * self.n
    }
    pub fn output_bytes(self) -> u64 {
        self.output_elements() * 4
    }
    pub fn input_bytes(self, ordinal: usize) -> u64 {
        if self.op != BatchedF32Op::Matmul {
            self.output_bytes()
        } else if ordinal == 0 {
            self.batch * self.group * self.m * self.k * 4
        } else {
            self.batch * self.group * self.n * self.k * 4
        }
    }
    pub fn arithmetic_operations(self) -> Result<u64, VNextError> {
        let factor = match self.op {
            BatchedF32Op::Matmul => self.k.checked_mul(2),
            BatchedF32Op::Scale => Some(1),
            BatchedF32Op::Softmax => Some(5),
        }
        .ok_or_else(|| invalid("work count overflows"))?;
        self.output_elements()
            .checked_mul(factor)
            .ok_or_else(|| invalid("work count overflows"))
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("batched F32: {reason}"),
    }
}

#[cfg(test)]
mod tests;
