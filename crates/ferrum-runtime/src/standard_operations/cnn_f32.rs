//! Independent all-F32 single-frame CNN contracts. Only convolution index
//! geometry is shared with the existing BF16 contract; no value crosses a
//! half-precision boundary and no old provider gains a new port dtype.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
pub const CONV2D_F32_OPERATION_ID: &str = "operation.conv2d.f32";
pub const CONV2D_F32_CAPABILITY_ID: &str = "capability.operation.conv2d.f32";
pub const GROUP_NORM_SILU_NCHW_F32_OPERATION_ID: &str = "operation.group_norm_silu.nchw.f32";
pub const GROUP_NORM_SILU_NCHW_F32_CAPABILITY_ID: &str =
    "capability.operation.group_norm_silu.nchw.f32";
pub const PLAIN_RESIDUAL_ADD_NCHW_F32_OPERATION_ID: &str = "operation.plain_residual_add.nchw.f32";
pub const PLAIN_RESIDUAL_ADD_NCHW_F32_CAPABILITY_ID: &str =
    "capability.operation.plain_residual_add.nchw.f32";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32CnnOp {
    Conv2d,
    GroupNormSilu,
    ResidualAdd,
}
impl F32CnnOp {
    pub const ALL: [Self; 3] = [Self::Conv2d, Self::GroupNormSilu, Self::ResidualAdd];
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Conv2d => CONV2D_F32_OPERATION_ID,
            Self::GroupNormSilu => GROUP_NORM_SILU_NCHW_F32_OPERATION_ID,
            Self::ResidualAdd => PLAIN_RESIDUAL_ADD_NCHW_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Conv2d => CONV2D_F32_CAPABILITY_ID,
            Self::GroupNormSilu => GROUP_NORM_SILU_NCHW_F32_CAPABILITY_ID,
            Self::ResidualAdd => PLAIN_RESIDUAL_ADD_NCHW_F32_CAPABILITY_ID,
        }
    }
    pub const fn input_count(self) -> usize {
        match self {
            Self::Conv2d | Self::GroupNormSilu => 3,
            Self::ResidualAdd => 2,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Conv2d => conv2d_f32_contract(),
            Self::GroupNormSilu => group_norm_silu_nchw_f32_contract(),
            Self::ResidualAdd => plain_residual_add_nchw_f32_contract(),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv2dF32PaddingMode {
    ExplicitZero,
    Reflect,
}
impl Conv2dF32PaddingMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitZero => "explicit_zero",
            Self::Reflect => "reflect",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "explicit_zero" => Some(Self::ExplicitZero),
            "reflect" => Some(Self::Reflect),
            _ => None,
        }
    }
}
/// F32 NCHW/OIHW cross-correlation, then a separate RN32 bias addition.
/// Padding is explicitly materialized (positive zero or non-edge-repeating
/// reflection). Stride/dilation/groups follow ordinary Conv2D index rules.
/// Native providers declare dot reduction/math policy and actual workspace;
/// this does not assert Torch or other backend bit equivalence.
pub fn conv2d_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut d = conv2d_bf16_contract()?.descriptor;
    d.id = OperationId::new(CONV2D_F32_OPERATION_ID)?;
    d.provider = provider_requirement(CONV2D_F32_CAPABILITY_ID, ContractVersion::new(1, 0))?;
    for t in d.inputs.iter_mut().chain(&mut d.outputs) {
        *t = contiguous_tensor(t.dimensions().to_vec(), [ElementType::F32], t.access())?;
    }
    d.attributes = AttributeSchema::new(BTreeMap::from([
        integer_list("padding", 4)?,
        integer_list("stride", 2)?,
        integer_list("dilation", 2)?,
        (
            AttributeId::new("groups")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Unsigned,
                required: true,
                constraint: AttributeConstraint::UnsignedRange {
                    minimum: 1,
                    maximum: u64::MAX,
                },
            },
        ),
        (
            AttributeId::new("padding_mode")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Text,
                required: true,
                constraint: AttributeConstraint::TextChoices {
                    values: BTreeSet::from(["explicit_zero".into(), "reflect".into()]),
                },
            },
        ),
    ]))?;
    d.resources.minimum_value_alignment_bytes = 16;
    d.validate()?;
    Ok(StandardOperationContract { descriptor: d })
}
fn integer_list(name: &str, length: u32) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Integers,
            required: true,
            constraint: AttributeConstraint::IntegerListLength {
                minimum: length,
                maximum: length,
            },
        },
    ))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dF32Geometry {
    index: Conv2dBf16Geometry,
    mode: Conv2dF32PaddingMode,
}
impl Conv2dF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        w: &ResolvedTensorSpec,
        b: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        conv2d_f32_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        for t in [x, w, b, y] {
            check_f32(t)?;
        }
        let Some(SemanticValue::Text(mode)) = attrs.get(&AttributeId::new("padding_mode")?) else {
            return Err(invalid("missing padding mode"));
        };
        let mode =
            Conv2dF32PaddingMode::parse(mode).ok_or_else(|| invalid("unknown padding mode"))?;
        let mut index_attrs = attrs.clone();
        index_attrs.insert(
            AttributeId::new("padding_mode")?,
            SemanticValue::Text(Conv2dPaddingMode::ExplicitZero.as_str().into()),
        );
        // Metadata-only dimension/index sharing, with independent F32 physical
        // checks above. There is no allocation, tensor cast or BF16 execution.
        let index_spec = |t: &ResolvedTensorSpec| {
            ResolvedTensorSpec::new(
                t.dimensions().to_vec(),
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
        };
        let index = Conv2dBf16Geometry::new(
            &index_spec(x)?,
            &index_spec(w)?,
            &index_spec(b)?,
            &index_spec(y)?,
            &index_attrs,
        )?;
        let pad = index.padding();
        if mode == Conv2dF32PaddingMode::Reflect
            && (pad[0] >= index.input_height()
                || pad[1] >= index.input_height()
                || pad[2] >= index.input_width()
                || pad[3] >= index.input_width())
        {
            return Err(invalid(
                "reflect padding must be smaller than its original spatial extent",
            ));
        }
        index
            .padded_input_bytes()
            .checked_mul(2)
            .filter(|n| *n <= isize::MAX as u64)
            .ok_or_else(|| invalid("F32 padded extent exceeds address space"))?;
        Ok(Self { index, mode })
    }
    pub const fn batch(self) -> u64 {
        self.index.batch()
    }
    pub const fn input_channels(self) -> u64 {
        self.index.input_channels()
    }
    pub const fn input_height(self) -> u64 {
        self.index.input_height()
    }
    pub const fn input_width(self) -> u64 {
        self.index.input_width()
    }
    pub const fn output_channels(self) -> u64 {
        self.index.output_channels()
    }
    pub const fn output_height(self) -> u64 {
        self.index.output_height()
    }
    pub const fn output_width(self) -> u64 {
        self.index.output_width()
    }
    pub const fn kernel_height(self) -> u64 {
        self.index.kernel_height()
    }
    pub const fn kernel_width(self) -> u64 {
        self.index.kernel_width()
    }
    pub const fn groups(self) -> u64 {
        self.index.groups()
    }
    pub const fn padding(self) -> [u64; 4] {
        self.index.padding()
    }
    pub const fn stride(self) -> [u64; 2] {
        self.index.stride()
    }
    pub const fn dilation(self) -> [u64; 2] {
        self.index.dilation()
    }
    pub const fn padding_mode(self) -> Conv2dF32PaddingMode {
        self.mode
    }
    pub const fn elements(self) -> u64 {
        self.index.element_count()
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.index.logical_arithmetic_operations()
    }
    /// Map an original-space coordinate after explicit padding. Reflection
    /// does not repeat the edge; invalid zero-pad coordinates return None.
    pub fn source_spatial(self, y: i128, x: i128) -> Option<(u64, u64)> {
        let one = |p: i128, n: u64| {
            if (0..i128::from(n)).contains(&p) {
                Some(p as u64)
            } else if self.mode == Conv2dF32PaddingMode::Reflect {
                Some(if p < 0 { -p } else { 2 * i128::from(n) - 2 - p } as u64)
            } else {
                None
            }
        };
        Some((one(y, self.input_height())?, one(x, self.input_width())?))
    }
}
/// Per-request group statistics across channels and spatial coordinates,
/// followed by per-channel affine and ordinary F32 SiLU. Every center,
/// square, mean/variance, inverse-root, affine and SiLU boundary is F32;
/// providers declare reduction/exp details. No residual or channel L2 norm.
pub fn group_norm_silu_nchw_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut d = group_template(
        GROUP_NORM_SILU_NCHW_F32_OPERATION_ID,
        GROUP_NORM_SILU_NCHW_F32_CAPABILITY_ID,
        3,
    )?;
    d.inputs[1] = contiguous_tensor(
        vec![symbol("channels")],
        [ElementType::F32],
        TensorAccess::Read,
    )?;
    d.inputs[2] = d.inputs[1].clone();
    d.attributes = AttributeSchema::new(BTreeMap::from([
        (
            AttributeId::new("groups")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Unsigned,
                required: true,
                constraint: AttributeConstraint::UnsignedRange {
                    minimum: 1,
                    maximum: u64::MAX,
                },
            },
        ),
        positive_epsilon_attribute("epsilon")?,
    ]))?;
    d.validate()?;
    Ok(StandardOperationContract { descriptor: d })
}
pub fn plain_residual_add_nchw_f32_contract() -> Result<StandardOperationContract, VNextError> {
    Ok(StandardOperationContract {
        descriptor: group_template(
            PLAIN_RESIDUAL_ADD_NCHW_F32_OPERATION_ID,
            PLAIN_RESIDUAL_ADD_NCHW_F32_CAPABILITY_ID,
            2,
        )?,
    })
}
fn group_template(op: &str, cap: &str, count: usize) -> Result<OperationDescriptor, VNextError> {
    let dims = vec![
        symbol("batch"),
        symbol("channels"),
        symbol("height"),
        symbol("width"),
    ];
    let d = OperationDescriptor {
        id: OperationId::new(op)?,
        version: ContractVersion::new(1, 0),
        inputs: (0..count)
            .map(|_| contiguous_tensor(dims.clone(), [ElementType::F32], TensorAccess::Read))
            .collect::<Result<_, _>>()?,
        outputs: vec![contiguous_tensor(
            dims,
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(cap, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    d.validate()?;
    Ok(d)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32NchwGeometry {
    shape: [u64; 4],
    elements: u64,
}
impl F32NchwGeometry {
    pub fn new(x: &ResolvedTensorSpec, y: &ResolvedTensorSpec) -> Result<Self, VNextError> {
        check_f32(x)?;
        check_f32(y)?;
        let shape: [u64; 4] = x
            .dimensions()
            .try_into()
            .map_err(|_| invalid("requires NCHW rank four"))?;
        if x != y || shape.contains(&0) {
            return Err(invalid("requires identical positive F32 NCHW shape"));
        }
        let elements = x.minimum_storage_bytes()? / 4;
        Ok(Self { shape, elements })
    }
    pub const fn shape(self) -> [u64; 4] {
        self.shape
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupNormSiluNchwF32Geometry {
    shape: [u64; 4],
    groups: u64,
    epsilon: CanonicalRational,
    elements: u64,
    group_elements: u64,
    arithmetic: u64,
}
impl GroupNormSiluNchwF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        beta: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        groups: u64,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let g = F32NchwGeometry::new(x, y)?;
        for t in [gamma, beta] {
            check_f32(t)?;
            if t.dimensions() != [g.shape[1]] {
                return Err(invalid("requires one gamma/beta per original channel"));
            }
        }
        if groups == 0
            || !g.shape[1].is_multiple_of(groups)
            || epsilon.numerator() <= 0
            || epsilon.numerator() as u64 > epsilon.denominator()
            || epsilon.numerator() as i128 * 1_000_000_000_000 < i128::from(epsilon.denominator())
        {
            return Err(invalid(
                "groups must divide channels and epsilon must be in [1e-12,1]",
            ));
        }
        let group_elements = g.shape[1] / groups * g.shape[2] * g.shape[3];
        let arithmetic = g
            .elements
            .checked_mul(12)
            .and_then(|n| n.checked_add(g.shape[0].checked_mul(groups)?.checked_mul(4)?))
            .ok_or_else(|| invalid("normalization logical arithmetic overflows"))?;
        Ok(Self {
            shape: g.shape,
            groups,
            epsilon,
            elements: g.elements,
            group_elements,
            arithmetic,
        })
    }
    pub const fn shape(self) -> [u64; 4] {
        self.shape
    }
    pub const fn groups(self) -> u64 {
        self.groups
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn group_elements(self) -> u64 {
        self.group_elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
    pub fn effective_epsilon(self) -> f32 {
        (self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64) as f32
    }
}
fn check_f32(t: &ResolvedTensorSpec) -> Result<(), VNextError> {
    if t.element_type() != ElementType::F32
        || t.layout() != &ResolvedTensorLayout::Contiguous
        || t.minimum_storage_bytes()? > isize::MAX as u64
    {
        return Err(invalid(
            "requires contiguous F32 storage within address space",
        ));
    }
    Ok(())
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 CNN: {reason}"),
    }
}
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32CnnGeometry {
    Conv2d(Conv2dF32Geometry),
    GroupNormSilu(GroupNormSiluNchwF32Geometry),
    ResidualAdd(F32NchwGeometry),
}
impl F32CnnGeometry {
    pub fn new(
        kind: F32CnnOp,
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        kind.contract()?.descriptor.validate_attributes(attrs)?;
        if inputs.len() != kind.input_count() {
            return Err(invalid("wrong CNN operand count"));
        }
        Ok(match kind {
            F32CnnOp::Conv2d => Self::Conv2d(Conv2dF32Geometry::new(
                inputs[0], inputs[1], inputs[2], output, attrs,
            )?),
            F32CnnOp::GroupNormSilu => {
                let Some(SemanticValue::Unsigned(groups)) = attrs.get(&AttributeId::new("groups")?)
                else {
                    return Err(invalid("missing groups"));
                };
                let Some(SemanticValue::Rational(epsilon)) =
                    attrs.get(&AttributeId::new("epsilon")?)
                else {
                    return Err(invalid("missing epsilon"));
                };
                Self::GroupNormSilu(GroupNormSiluNchwF32Geometry::new(
                    inputs[0], inputs[1], inputs[2], output, *groups, *epsilon,
                )?)
            }
            F32CnnOp::ResidualAdd => {
                if inputs[0] != inputs[1] {
                    return Err(invalid(
                        "residual operands must have identical original shape",
                    ));
                }
                Self::ResidualAdd(F32NchwGeometry::new(inputs[0], output)?)
            }
        })
    }
    pub const fn kind(self) -> F32CnnOp {
        match self {
            Self::Conv2d(_) => F32CnnOp::Conv2d,
            Self::GroupNormSilu(_) => F32CnnOp::GroupNormSilu,
            Self::ResidualAdd(_) => F32CnnOp::ResidualAdd,
        }
    }
    pub const fn elements(self) -> u64 {
        match self {
            Self::Conv2d(g) => g.elements(),
            Self::GroupNormSilu(g) => g.elements(),
            Self::ResidualAdd(g) => g.elements(),
        }
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        match self {
            Self::Conv2d(g) => g.logical_arithmetic_operations(),
            Self::GroupNormSilu(g) => g.logical_arithmetic_operations(),
            Self::ResidualAdd(g) => g.elements(),
        }
    }
    pub fn input_elements(self, index: usize) -> Option<u64> {
        match self {
            Self::Conv2d(g) => match index {
                0 => Some(g.batch() * g.input_channels() * g.input_height() * g.input_width()),
                1 => Some(
                    g.output_channels()
                        * (g.input_channels() / g.groups())
                        * g.kernel_height()
                        * g.kernel_width(),
                ),
                2 => Some(g.output_channels()),
                _ => None,
            },
            Self::GroupNormSilu(g) => match index {
                0 => Some(g.elements()),
                1 | 2 => Some(g.shape()[1]),
                _ => None,
            },
            Self::ResidualAdd(g) => (index < 2).then_some(g.elements()),
        }
    }
}
