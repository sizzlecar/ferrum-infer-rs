//! Single-frame spatial/channel rearrangement followed by a group mean.
use super::spatial_attention_bf16::checked_nchw;
use super::*;
use crate::{ResolvedTensorSpec, SemanticValue};

pub const SPATIAL_GROUP_MEAN_BF16_OPERATION_ID: &str = "operation.spatial_group_mean.bf16";
pub const SPATIAL_GROUP_MEAN_BF16_CAPABILITY_ID: &str =
    "capability.operation.spatial_group_mean.bf16";

/// X[N,C,H,W] -> Y[N,O,H/sh,W/sw]. The conceptual rearranged channel
/// order is [C, frame_factor, sh, sw]. The single real frame is last; all
/// preceding frame positions are positive zero. Consecutive groups of
/// C*frame_factor*sh*sw/O values are averaged, with one final BF16 boundary.
/// No full padded/reshuffled tensor is required. NoAlias, no auxiliary storage.
/// Provider-specific FP32 reduction order is explicit numerical evidence;
/// OperationDefined does not promise Torch reduction bitwise identity.
pub fn spatial_group_mean_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let attributes = ["scale_h", "scale_w", "frame_factor"]
        .into_iter()
        .map(|name| {
            Ok((
                AttributeId::new(name)?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Unsigned,
                    required: true,
                    constraint: AttributeConstraint::UnsignedRange {
                        minimum: 1,
                        maximum: 2,
                    },
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
    let descriptor = OperationDescriptor {
        id: OperationId::new(SPATIAL_GROUP_MEAN_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            vec![
                symbol("batch"),
                symbol("input_channels"),
                symbol("input_height"),
                symbol("input_width"),
            ],
            [ElementType::Bf16],
            TensorAccess::Read,
        )?],
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
        attributes: AttributeSchema::new(attributes)?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            SPATIAL_GROUP_MEAN_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialGroupMeanBf16Geometry {
    input: [u64; 4],
    output: [u64; 4],
    scale: [u64; 2],
    frames: u64,
    group: u64,
    elements: u64,
    operations: u64,
}
impl SpatialGroupMeanBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        spatial_group_mean_bf16_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        let x = checked_nchw(input)?;
        let y = checked_nchw(output)?;
        let unsigned = |name: &str| -> Result<u64, VNextError> {
            match attrs.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Unsigned(n)) if (1..=2).contains(n) => Ok(*n),
                _ => Err(invalid("scale and frame factors must be 1 or 2")),
            }
        };
        let sh = unsigned("scale_h")?;
        let sw = unsigned("scale_w")?;
        let frames = unsigned("frame_factor")?;
        let expanded = x[1]
            .checked_mul(frames)
            .and_then(|n| n.checked_mul(sh))
            .and_then(|n| n.checked_mul(sw))
            .ok_or_else(|| invalid("expanded channels overflow"))?;
        if x[2] % sh != 0
            || x[3] % sw != 0
            || x[0] != y[0]
            || y[2] != x[2] / sh
            || y[3] != x[3] / sw
            || expanded % y[1] != 0
        {
            return Err(invalid(
                "output geometry must exactly divide spatial and expanded channel extents",
            ));
        }
        let group = expanded / y[1];
        if group == 0 || group > 1 << 24 {
            return Err(invalid(
                "group size must be 1..=2^24 for an exact FP32 divisor",
            ));
        }
        let elements = y
            .iter()
            .try_fold(1_u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("output elements overflow"))?;
        // G-1 additions and one division, including conceptual zero entries.
        let operations = elements
            .checked_mul(group)
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            input: x,
            output: y,
            scale: [sh, sw],
            frames,
            group,
            elements,
            operations,
        })
    }
    pub const fn batch(&self) -> u64 {
        self.input[0]
    }
    pub const fn input_channels(&self) -> u64 {
        self.input[1]
    }
    pub const fn input_height(&self) -> u64 {
        self.input[2]
    }
    pub const fn input_width(&self) -> u64 {
        self.input[3]
    }
    pub const fn output_channels(&self) -> u64 {
        self.output[1]
    }
    pub const fn output_height(&self) -> u64 {
        self.output[2]
    }
    pub const fn output_width(&self) -> u64 {
        self.output[3]
    }
    pub const fn scale_h(&self) -> u64 {
        self.scale[0]
    }
    pub const fn scale_w(&self) -> u64 {
        self.scale[1]
    }
    pub const fn frame_factor(&self) -> u64 {
        self.frames
    }
    pub const fn group_size(&self) -> u64 {
        self.group
    }
    pub const fn element_count(&self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(&self) -> u64 {
        self.operations
    }
}
fn invalid(message: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 spatial group mean: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: &[u64]) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            d.to_vec(),
            ElementType::Bf16,
            crate::ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
        ["scale_h", "scale_w", "frame_factor"]
            .into_iter()
            .map(|n| (AttributeId::new(n).unwrap(), SemanticValue::Unsigned(2)))
            .collect()
    }
    #[test]
    fn group_mean_has_explicit_single_frame_geometry_and_checked_work() {
        let g = SpatialGroupMeanBf16Geometry::new(&t(&[2, 3, 4, 6]), &t(&[2, 6, 2, 3]), &attrs())
            .unwrap();
        assert_eq!(
            (
                g.group_size(),
                g.element_count(),
                g.logical_arithmetic_operations()
            ),
            (4, 72, 288)
        );
        assert_eq!(g.frame_factor(), 2);
        for bad in [[2, 5, 2, 3], [2, 6, 3, 2], [1, 6, 2, 3]] {
            assert!(
                SpatialGroupMeanBf16Geometry::new(&t(&[2, 3, 4, 6]), &t(&bad), &attrs()).is_err()
            );
        }
    }
    #[test]
    fn group_mean_rejects_foreign_attrs_dtype_and_nondivisible_axes() {
        let mut a = attrs();
        a.insert(
            AttributeId::new("unknown").unwrap(),
            SemanticValue::Unsigned(1),
        );
        assert!(
            SpatialGroupMeanBf16Geometry::new(&t(&[1, 1, 2, 2]), &t(&[1, 2, 1, 1]), &a).is_err()
        );
        assert!(
            SpatialGroupMeanBf16Geometry::new(&t(&[1, 1, 3, 2]), &t(&[1, 2, 1, 1]), &attrs())
                .is_err()
        );
        let f32_tensor = ResolvedTensorSpec::new(
            vec![1, 1, 2, 2],
            ElementType::F32,
            crate::ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        assert!(
            SpatialGroupMeanBf16Geometry::new(&f32_tensor, &t(&[1, 2, 1, 1]), &attrs()).is_err()
        );
    }
}
