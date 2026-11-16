//! F32 NCHW transposed convolution, IOHW weights and a separate RN32 bias add.
//! This is the adjoint of zero-padded cross-correlation. Output padding selects
//! the trailing output extent; it does not add a second padded-input buffer.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const CONV_TRANSPOSE2D_F32_OPERATION_ID: &str = "operation.conv_transpose2d.f32";
pub const CONV_TRANSPOSE2D_F32_CAPABILITY_ID: &str = "capability.operation.conv_transpose2d.f32";

pub fn conv_transpose2d_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = conv2d_f32_contract()?.descriptor;
    descriptor.id = OperationId::new(CONV_TRANSPOSE2D_F32_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        CONV_TRANSPOSE2D_F32_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    descriptor.inputs[1] = contiguous_tensor(
        vec![
            symbol("input_channels"),
            symbol("output_channels_per_group"),
            symbol("kernel_height"),
            symbol("kernel_width"),
        ],
        [ElementType::F32],
        TensorAccess::Read,
    )?;
    let mut attrs = BTreeMap::new();
    for name in ["padding", "stride", "dilation", "output_padding"] {
        attrs.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: 2,
                    maximum: 2,
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
    descriptor.attributes = AttributeSchema::new(attrs)?;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose2dF32Geometry {
    input: [u64; 4],
    weight: [u64; 4],
    output: [u64; 4],
    padding: [u64; 2],
    stride: [u64; 2],
    dilation: [u64; 2],
    output_padding: [u64; 2],
    groups: u64,
    input_elements: [u64; 3],
    elements: u64,
    work: u64,
}
impl ConvTranspose2dF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        w: &ResolvedTensorSpec,
        b: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        conv_transpose2d_f32_contract()?
            .descriptor()
            .validate_attributes(attrs)?;
        for tensor in [x, w, b, y] {
            if tensor.element_type() != ElementType::F32
                || tensor.layout() != &ResolvedTensorLayout::Contiguous
                || tensor.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid("requires contiguous addressable F32 storage"));
            }
        }
        let input: [u64; 4] = x
            .dimensions()
            .try_into()
            .map_err(|_| invalid("input requires NCHW"))?;
        let weight: [u64; 4] = w
            .dimensions()
            .try_into()
            .map_err(|_| invalid("weight requires IOHW"))?;
        let output: [u64; 4] = y
            .dimensions()
            .try_into()
            .map_err(|_| invalid("output requires NCHW"))?;
        let pair = |name: &str, positive: bool| -> Result<[u64; 2], VNextError> {
            let Some(SemanticValue::Integers(values)) = attrs.get(&AttributeId::new(name)?) else {
                return Err(invalid("missing spatial attribute"));
            };
            let values: [i64; 2] = values
                .as_slice()
                .try_into()
                .map_err(|_| invalid("spatial attribute requires two axes"))?;
            if values.iter().any(|&v| v < i64::from(positive)) {
                return Err(invalid("invalid spatial attribute extent"));
            }
            Ok(values.map(|v| v as u64))
        };
        let padding = pair("padding", false)?;
        let stride = pair("stride", true)?;
        let dilation = pair("dilation", true)?;
        let output_padding = pair("output_padding", false)?;
        let Some(SemanticValue::Unsigned(groups)) = attrs.get(&AttributeId::new("groups")?) else {
            return Err(invalid("missing convolution groups"));
        };
        let groups = *groups;
        if groups == 0
            || input.contains(&0)
            || weight.contains(&0)
            || output.contains(&0)
            || input[0] != output[0]
            || input[1] != weight[0]
            || input[1] % groups != 0
            || weight[1].checked_mul(groups) != Some(output[1])
            || b.dimensions() != [output[1]]
        {
            return Err(invalid(
                "batch, grouped IOHW channels or bias dimensions differ",
            ));
        }
        for axis in 0..2 {
            // This declared profile uses the standard stride ambiguity range.
            // Dilation-only extensions with output_padding >= stride are excluded.
            if output_padding[axis] >= stride[axis] {
                return Err(invalid("output_padding must be smaller than stride"));
            }
            let extent = (input[axis + 2] - 1)
                .checked_mul(stride[axis])
                .and_then(|v| {
                    (weight[axis + 2] - 1)
                        .checked_mul(dilation[axis])
                        .and_then(|k| v.checked_add(k))
                })
                .and_then(|v| v.checked_add(output_padding[axis]))
                .and_then(|v| v.checked_add(1))
                .and_then(|v| padding[axis].checked_mul(2).and_then(|p| v.checked_sub(p)));
            if extent != Some(output[axis + 2]) {
                return Err(invalid(
                    "output extent differs from transposed convolution geometry",
                ));
            }
        }
        let input_elements = [x, w, b].map(|t| t.minimum_storage_bytes().map(|v| v / 4));
        let [a, w_elements, bias] = input_elements;
        let input_elements = [a?, w_elements?, bias?];
        let elements = y.minimum_storage_bytes()? / 4;
        let work = input_elements[0]
            .checked_mul(weight[1])
            .and_then(|v| v.checked_mul(weight[2]))
            .and_then(|v| v.checked_mul(weight[3]))
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(elements))
            .ok_or_else(|| invalid("logical work extent overflow"))?;
        Ok(Self {
            input,
            weight,
            output,
            padding,
            stride,
            dilation,
            output_padding,
            groups,
            input_elements,
            elements,
            work,
        })
    }
    pub const fn input(&self) -> [u64; 4] {
        self.input
    }
    pub const fn weight(&self) -> [u64; 4] {
        self.weight
    }
    pub const fn output(&self) -> [u64; 4] {
        self.output
    }
    pub const fn padding(&self) -> [u64; 2] {
        self.padding
    }
    pub const fn stride(&self) -> [u64; 2] {
        self.stride
    }
    pub const fn dilation(&self) -> [u64; 2] {
        self.dilation
    }
    pub const fn output_padding(&self) -> [u64; 2] {
        self.output_padding
    }
    pub const fn groups(&self) -> u64 {
        self.groups
    }
    pub fn input_elements(&self, index: usize) -> Option<u64> {
        self.input_elements.get(index).copied()
    }
    pub const fn elements(&self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(&self) -> u64 {
        self.work
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 transposed convolution: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec(shape: &[u64]) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            shape.to_vec(),
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    fn attrs(
        stride: [i64; 2],
        pad: [i64; 2],
        dilation: [i64; 2],
        output_pad: [i64; 2],
        groups: u64,
    ) -> BTreeMap<AttributeId, SemanticValue> {
        let mut result = BTreeMap::new();
        for (name, values) in [
            ("stride", stride),
            ("padding", pad),
            ("dilation", dilation),
            ("output_padding", output_pad),
        ] {
            result.insert(
                AttributeId::new(name).unwrap(),
                SemanticValue::Integers(values.to_vec()),
            );
        }
        result.insert(
            AttributeId::new("groups").unwrap(),
            SemanticValue::Unsigned(groups),
        );
        result
    }
    #[test]
    fn transpose_iohw_geometry_preserves_grouping_and_trailing_extent() {
        let x = spec(&[2, 4, 1, 5]);
        let w = spec(&[4, 3, 1, 4]);
        let b = spec(&[6]);
        let a = attrs([1, 2], [0, 1], [1, 1], [0, 0], 2);
        let g = ConvTranspose2dF32Geometry::new(&x, &w, &b, &spec(&[2, 6, 1, 10]), &a).unwrap();
        assert_eq!(g.groups(), 2);
        assert_eq!(g.input_elements(1), Some(48));
        assert!(ConvTranspose2dF32Geometry::new(&x, &w, &b, &spec(&[2, 6, 1, 9]), &a).is_err());
        assert!(ConvTranspose2dF32Geometry::new(
            &x,
            &spec(&[6, 2, 1, 4]),
            &b,
            &spec(&[2, 6, 1, 10]),
            &a
        )
        .is_err());
        // Full 2D with dilation and output padding: ((3-1)*2-2+2*(3-1)+1+1, (4-1)*3-2+2+1+2).
        let a = attrs([2, 3], [1, 1], [2, 1], [1, 2], 1);
        ConvTranspose2dF32Geometry::new(
            &spec(&[1, 2, 3, 4]),
            &spec(&[2, 3, 3, 3]),
            &spec(&[3]),
            &spec(&[1, 3, 8, 12]),
            &a,
        )
        .unwrap();
        let invalid_padding = attrs([2, 3], [1, 1], [2, 1], [2, 2], 1);
        assert!(ConvTranspose2dF32Geometry::new(
            &spec(&[1, 2, 3, 4]),
            &spec(&[2, 3, 3, 3]),
            &spec(&[3]),
            &spec(&[1, 3, 9, 12]),
            &invalid_padding
        )
        .is_err());
        // The actual stride-five upsampler uses IOHW, not OIHW, and gives exactly 5L.
        let a = attrs([1, 5], [0, 2], [1, 1], [0, 0], 1);
        ConvTranspose2dF32Geometry::new(
            &spec(&[2, 8, 1, 7]),
            &spec(&[8, 4, 1, 9]),
            &spec(&[4]),
            &spec(&[2, 4, 1, 35]),
            &a,
        )
        .unwrap();
    }
}
