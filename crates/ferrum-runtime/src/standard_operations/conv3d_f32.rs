//! Contiguous F32 NCDHW/OIDHW cross-correlation. Each axis declares zero or
//! non-edge-repeating reflect padding independently. Temporal causality is a
//! model choice of left padding, not a different allocator or convolution.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const CONV3D_F32_OPERATION_ID: &str = "operation.conv3d.f32";
pub const CONV3D_F32_CAPABILITY_ID: &str = "capability.operation.conv3d.f32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv3dF32PaddingMode {
    Zero,
    Reflect,
}
impl Conv3dF32PaddingMode {
    pub const fn attribute_value(self) -> i64 {
        match self {
            Self::Zero => 0,
            Self::Reflect => 1,
        }
    }
}

/// Separate RN32 bias addition follows the convolution. Providers declare
/// their dot reduction/math policy; this contract does not imply equivalence
/// between sequential scalar reductions and cuDNN FMA reductions.
pub fn conv3d_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut attrs = BTreeMap::new();
    for (name, length) in [
        ("padding", 6),
        ("stride", 3),
        ("dilation", 3),
        ("padding_modes", 3),
    ] {
        attrs.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: length,
                    maximum: length,
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
    let tensor = |names: &[&str], access| {
        contiguous_tensor(
            names.iter().map(|n| symbol(n)).collect(),
            [ElementType::F32],
            access,
        )
    };
    let descriptor = OperationDescriptor {
        id: OperationId::new(CONV3D_F32_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            tensor(
                &["batch", "input_channels", "depth", "height", "width"],
                TensorAccess::Read,
            )?,
            tensor(
                &[
                    "output_channels",
                    "channels_per_group",
                    "kernel_depth",
                    "kernel_height",
                    "kernel_width",
                ],
                TensorAccess::Read,
            )?,
            tensor(&["output_channels"], TensorAccess::Read)?,
        ],
        outputs: vec![tensor(
            &[
                "batch",
                "output_channels",
                "output_depth",
                "output_height",
                "output_width",
            ],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attrs)?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(CONV3D_F32_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3dF32Geometry {
    input: [u64; 5],
    weight: [u64; 5],
    output: [u64; 5],
    padding: [u64; 6],
    stride: [u64; 3],
    dilation: [u64; 3],
    modes: [Conv3dF32PaddingMode; 3],
    padded: [u64; 3],
    groups: u64,
    input_elements: [u64; 3],
    output_elements: u64,
    padded_bytes: u64,
    arithmetic: u64,
}
impl Conv3dF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        w: &ResolvedTensorSpec,
        b: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        conv3d_f32_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        for t in [x, w, b, y] {
            if t.element_type() != ElementType::F32
                || t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
                || t.dimensions().contains(&0)
            {
                return Err(invalid(
                    "requires positive addressable contiguous F32 storage",
                ));
            }
        }
        let shape = |t: &ResolvedTensorSpec| {
            t.dimensions()
                .try_into()
                .map_err(|_| invalid("requires rank-five input, filter and output"))
        };
        let input: [u64; 5] = shape(x)?;
        let weight: [u64; 5] = shape(w)?;
        let output: [u64; 5] = shape(y)?;
        let padding = integers::<6>(attrs, "padding", 0)?;
        let stride = integers::<3>(attrs, "stride", 1)?;
        let dilation = integers::<3>(attrs, "dilation", 1)?;
        let modes = integers::<3>(attrs, "padding_modes", 0)?;
        if modes.iter().any(|&m| m > 1) {
            return Err(invalid("padding modes must be zero=0 or reflect=1"));
        }
        let modes = modes.map(|m| {
            if m == 0 {
                Conv3dF32PaddingMode::Zero
            } else {
                Conv3dF32PaddingMode::Reflect
            }
        });
        let Some(SemanticValue::Unsigned(groups)) = attrs.get(&AttributeId::new("groups")?) else {
            return Err(invalid("missing groups"));
        };
        let groups = *groups;
        if groups == 0
            || input[0] != output[0]
            || output[1] != weight[0]
            || input[1] % groups != 0
            || output[1] % groups != 0
            || weight[1].checked_mul(groups) != Some(input[1])
            || b.dimensions() != [output[1]]
        {
            return Err(invalid("batch, grouped channels or bias differ"));
        }
        let mut padded = [0; 3];
        for axis in 0..3 {
            let extent = input[axis + 2];
            let [before, after] = [padding[2 * axis], padding[2 * axis + 1]];
            if modes[axis] == Conv3dF32PaddingMode::Reflect && (before >= extent || after >= extent)
            {
                return Err(invalid(
                    "reflect padding must be smaller than its original axis",
                ));
            }
            padded[axis] = extent
                .checked_add(before)
                .and_then(|v| v.checked_add(after))
                .ok_or_else(|| invalid("padded extent overflows"))?;
            let kernel = (weight[axis + 2] - 1)
                .checked_mul(dilation[axis])
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| invalid("effective kernel overflows"))?;
            let actual = padded[axis]
                .checked_sub(kernel)
                .and_then(|v| (v / stride[axis]).checked_add(1))
                .ok_or_else(|| invalid("kernel exceeds padded input"))?;
            if output[axis + 2] != actual {
                return Err(invalid("output axis differs from stride/dilation/padding"));
            }
        }
        let padded_bytes =
            checked_product(&[input[0], input[1], padded[0], padded[1], padded[2], 4])?;
        if padded_bytes > isize::MAX as u64 {
            return Err(invalid("padded input exceeds address space"));
        }
        let output_elements = y.minimum_storage_bytes()? / 4;
        let dot = checked_product(&weight[1..])?;
        let arithmetic = dot
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_mul(output_elements))
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        Ok(Self {
            input,
            weight,
            output,
            padding,
            stride,
            dilation,
            modes,
            padded,
            groups,
            input_elements: [
                x.minimum_storage_bytes()? / 4,
                w.minimum_storage_bytes()? / 4,
                b.minimum_storage_bytes()? / 4,
            ],
            output_elements,
            padded_bytes,
            arithmetic,
        })
    }
    pub const fn input(self) -> [u64; 5] {
        self.input
    }
    pub const fn weight(self) -> [u64; 5] {
        self.weight
    }
    pub const fn output(self) -> [u64; 5] {
        self.output
    }
    pub const fn padding(self) -> [u64; 6] {
        self.padding
    }
    pub const fn stride(self) -> [u64; 3] {
        self.stride
    }
    pub const fn dilation(self) -> [u64; 3] {
        self.dilation
    }
    pub const fn padding_modes(self) -> [Conv3dF32PaddingMode; 3] {
        self.modes
    }
    pub const fn padded_spatial(self) -> [u64; 3] {
        self.padded
    }
    pub const fn groups(self) -> u64 {
        self.groups
    }
    pub const fn input_elements(self) -> [u64; 3] {
        self.input_elements
    }
    pub const fn elements(self) -> u64 {
        self.output_elements
    }
    pub const fn padded_input_bytes(self) -> u64 {
        self.padded_bytes
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }

    /// Original-space coordinates after removing the explicit before padding.
    /// A zero-padded coordinate makes the whole input element positive zero.
    pub fn source_spatial(self, coordinates: [i128; 3]) -> Option<[u64; 3]> {
        let mut source = [0; 3];
        for axis in 0..3 {
            let p = coordinates[axis];
            let size = i128::from(self.input[axis + 2]);
            let mapped = if (0..size).contains(&p) {
                p
            } else if self.modes[axis] == Conv3dF32PaddingMode::Reflect {
                if p < 0 {
                    -p
                } else {
                    2 * size - 2 - p
                }
            } else {
                return None;
            };
            if !(0..size).contains(&mapped) {
                return None;
            }
            source[axis] = mapped as u64;
        }
        Some(source)
    }
}
fn integers<const N: usize>(
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
    minimum: i64,
) -> Result<[u64; N], VNextError> {
    let Some(SemanticValue::Integers(v)) = attrs.get(&AttributeId::new(name)?) else {
        return Err(invalid("missing convolution integer attribute"));
    };
    let v: [i64; N] = v
        .as_slice()
        .try_into()
        .map_err(|_| invalid("wrong attribute axis count"))?;
    if v.iter().any(|&n| n < minimum) {
        return Err(invalid("invalid convolution axis extent"));
    }
    Ok(v.map(|n| n as u64))
}
fn checked_product(v: &[u64]) -> Result<u64, VNextError> {
    v.iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| invalid("extent product overflows"))
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 Conv3D: {reason}"),
    }
}

#[cfg(test)]
mod tests;
