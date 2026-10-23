//! NCHW/OIHW cross-correlation with explicit geometry and a separate BF16 bias add.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const CONV2D_BF16_OPERATION_ID: &str = "operation.conv2d.bf16";
pub const CONV2D_BF16_CAPABILITY_ID: &str = "capability.operation.conv2d.bf16";

/// These select distinct provider algorithms even when real arithmetic agrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv2dPaddingMode {
    ExplicitZero,
    ImplicitZero,
}
impl Conv2dPaddingMode {
    pub const ALL: [Self; 2] = [Self::ExplicitZero, Self::ImplicitZero];
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitZero => "explicit_zero",
            Self::ImplicitZero => "implicit_zero",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == s)
    }
}

/// X[N,I,H,W], weights[O,I/groups,KH,KW], bias[O], Y[N,O,OH,OW].
/// Cross-correlation (no kernel reversal), zero padding, explicit strides and
/// dilations. Output is B(RN32(B(dot32)+bias)); bias cannot be fused before
/// the convolution's BF16 boundary. Provider profiles specify dot reduction.
/// ExplicitZero requires the backend to preserve the declared external-zero-pad
/// algorithm; a backend needing materialized padding must admit its scratch.
/// OperationDefined does not promise different padding/reduction algorithms
/// have the same bits. NoAlias output; all actual extents are checked again.
pub fn conv2d_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut attrs = BTreeMap::new();
    for (name, n) in [("padding", 4), ("stride", 2), ("dilation", 2)] {
        attrs.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: n,
                    maximum: n,
                },
            },
        );
    }
    attrs.insert(
        AttributeId::new("groups")?,
        AttributeSpec {
            value_kind: AttributeValueKind::Unsigned,
            required: true,
            constraint: AttributeConstraint::UnsignedRange {
                minimum: 1,
                maximum: u64::MAX,
            },
        },
    );
    attrs.insert(
        AttributeId::new("padding_mode")?,
        AttributeSpec {
            value_kind: AttributeValueKind::Text,
            required: true,
            constraint: AttributeConstraint::TextChoices {
                values: Conv2dPaddingMode::ALL
                    .into_iter()
                    .map(|m| m.as_str().into())
                    .collect(),
            },
        },
    );
    let descriptor = OperationDescriptor {
        id: OperationId::new(CONV2D_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![
                    symbol("batch"),
                    symbol("input_channels"),
                    symbol("input_height"),
                    symbol("input_width"),
                ],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    symbol("output_channels"),
                    symbol("channels_per_group"),
                    symbol("kernel_height"),
                    symbol("kernel_width"),
                ],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("output_channels")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![
                symbol("batch"),
                symbol("output_channels"),
                symbol("output_height"),
                symbol("output_width"),
            ],
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attrs)?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(CONV2D_BF16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dBf16Geometry {
    input: [u64; 4],
    weight: [u64; 4],
    output: [u64; 4],
    padding: [u64; 4],
    stride: [u64; 2],
    dilation: [u64; 2],
    groups: u64,
    padding_mode: Conv2dPaddingMode,
    padded_input_bytes: u64,
    elements: u64,
    operations: u64,
}
impl Conv2dBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        weight: &ResolvedTensorSpec,
        bias: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        conv2d_bf16_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        for t in [input, weight, bias, output] {
            if t.element_type() != ElementType::Bf16
                || t.layout() != &ResolvedTensorLayout::Contiguous
            {
                return Err(invalid("all operands must be contiguous BF16"));
            }
            t.minimum_storage_bytes()?;
        }
        let x: [u64; 4] = input
            .dimensions()
            .try_into()
            .map_err(|_| invalid("input must be NCHW rank four"))?;
        let w: [u64; 4] = weight
            .dimensions()
            .try_into()
            .map_err(|_| invalid("weight must be OIHW rank four"))?;
        let y: [u64; 4] = output
            .dimensions()
            .try_into()
            .map_err(|_| invalid("output must be NCHW rank four"))?;
        let padding = list::<4>(attrs, "padding", false)?;
        let stride = list::<2>(attrs, "stride", true)?;
        let dilation = list::<2>(attrs, "dilation", true)?;
        let Some(SemanticValue::Unsigned(groups)) = attr(attrs, "groups") else {
            return Err(invalid("missing groups"));
        };
        let Some(SemanticValue::Text(mode)) = attr(attrs, "padding_mode") else {
            return Err(invalid("missing padding mode"));
        };
        let padding_mode =
            Conv2dPaddingMode::parse(mode).ok_or_else(|| invalid("unknown padding mode"))?;
        if x.contains(&0)
            || w.contains(&0)
            || y.contains(&0)
            || *groups == 0
            || x[1] % groups != 0
            || w[0] % groups != 0
            || w[1] != x[1] / groups
            || bias.dimensions() != [w[0]]
        {
            return Err(invalid(
                "requires positive shapes, exact O bias, and group-divisible I/O channels",
            ));
        }
        let ph = x[2]
            .checked_add(padding[0])
            .and_then(|v| v.checked_add(padding[1]))
            .ok_or_else(|| invalid("padded height overflows"))?;
        let pw = x[3]
            .checked_add(padding[2])
            .and_then(|v| v.checked_add(padding[3]))
            .ok_or_else(|| invalid("padded width overflows"))?;
        let extent = |k: u64, d: u64, p: u64, s: u64| -> Result<u64, VNextError> {
            let effective = (k - 1)
                .checked_mul(d)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| invalid("effective kernel overflows"))?;
            p.checked_sub(effective)
                .map(|v| v / s + 1)
                .ok_or_else(|| invalid("effective kernel exceeds padded input"))
        };
        if y != [
            x[0],
            w[0],
            extent(w[2], dilation[0], ph, stride[0])?,
            extent(w[3], dilation[1], pw, stride[1])?,
        ] {
            return Err(invalid("output differs from exact convolution geometry"));
        }
        let padded_input_bytes = product([2, x[0], x[1], ph, pw])?;
        let elements = product(y)?;
        let reduction = product([w[1], w[2], w[3]])?;
        // Nominal full dot: K products + K-1 additions, then bias addition.
        let operations = elements
            .checked_mul(reduction)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        Ok(Self {
            input: x,
            weight: w,
            output: y,
            padding,
            stride,
            dilation,
            groups: *groups,
            padding_mode,
            padded_input_bytes,
            elements,
            operations,
        })
    }
    pub const fn batch(self) -> u64 {
        self.input[0]
    }
    pub const fn input_channels(self) -> u64 {
        self.input[1]
    }
    pub const fn input_height(self) -> u64 {
        self.input[2]
    }
    pub const fn input_width(self) -> u64 {
        self.input[3]
    }
    pub const fn output_channels(self) -> u64 {
        self.output[1]
    }
    pub const fn output_height(self) -> u64 {
        self.output[2]
    }
    pub const fn output_width(self) -> u64 {
        self.output[3]
    }
    pub const fn kernel_height(self) -> u64 {
        self.weight[2]
    }
    pub const fn kernel_width(self) -> u64 {
        self.weight[3]
    }
    pub const fn groups(self) -> u64 {
        self.groups
    }
    pub const fn padding(self) -> [u64; 4] {
        self.padding
    }
    pub const fn stride(self) -> [u64; 2] {
        self.stride
    }
    pub const fn dilation(self) -> [u64; 2] {
        self.dilation
    }
    pub const fn padding_mode(self) -> Conv2dPaddingMode {
        self.padding_mode
    }
    pub const fn padded_input_bytes(self) -> u64 {
        self.padded_input_bytes
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.operations
    }
}
fn attr<'a>(a: &'a BTreeMap<AttributeId, SemanticValue>, n: &str) -> Option<&'a SemanticValue> {
    a.iter().find(|(k, _)| k.as_str() == n).map(|(_, v)| v)
}
fn list<const N: usize>(
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
    positive: bool,
) -> Result<[u64; N], VNextError> {
    let Some(SemanticValue::Integers(v)) = attr(attrs, name) else {
        return Err(invalid("missing convolution geometry list"));
    };
    if v.len() != N || v.iter().any(|x| *x < if positive { 1 } else { 0 }) {
        return Err(invalid("invalid convolution geometry list"));
    }
    Ok(std::array::from_fn(|i| v[i] as u64))
}
fn product<const N: usize>(a: [u64; N]) -> Result<u64, VNextError> {
    a.into_iter()
        .try_fold(1_u64, |p, x| p.checked_mul(x))
        .ok_or_else(|| invalid("tensor extent overflows"))
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 convolution: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
