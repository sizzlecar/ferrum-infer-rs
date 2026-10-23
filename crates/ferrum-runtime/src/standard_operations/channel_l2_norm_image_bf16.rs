//! Explicit image gamma ABI for the existing channel-L2 arithmetic.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID: &str = "operation.channel_l2_norm_image.bf16";
pub const CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID: &str =
    "capability.operation.channel_l2_norm_image.bf16";

/// Same staged arithmetic as channel_l2_norm.bf16, with actual gamma[C,1,1].
/// X/Y remain NCHW; a rank-four gamma is rejected at this public boundary.
pub fn channel_l2_norm_image_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut c = channel_l2_norm_bf16_contract()?;
    c.descriptor.id = OperationId::new(CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID)?;
    c.descriptor.inputs[1] = contiguous_tensor(
        vec![
            symbol("channels"),
            DimensionConstraint::Exact(1),
            DimensionConstraint::Exact(1),
        ],
        [ElementType::Bf16],
        TensorAccess::Read,
    )?;
    c.descriptor.provider = provider_requirement(
        CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    c.descriptor.validate()?;
    Ok(c)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelL2NormImageBf16Geometry(ChannelL2NormBf16Geometry);
impl ChannelL2NormImageBf16Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        channel_l2_norm_image_bf16_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        let [_, channels, _, _] = x.dimensions() else {
            return Err(invalid("X must be NCHW"));
        };
        if gamma.dimensions() != [*channels, 1, 1]
            || gamma.element_type() != ElementType::Bf16
            || gamma.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid("actual gamma must be contiguous BF16 [C,1,1]"));
        }
        let gamma_bytes = gamma.minimum_storage_bytes()?;
        if gamma_bytes > isize::MAX as u64 {
            return Err(invalid("gamma exceeds address space"));
        }
        // This is only a descriptor for sharing the arithmetic geometry. The
        // real operand retains its checked rank-three schema and storage.
        let arithmetic_gamma = ResolvedTensorSpec::new(
            vec![*channels, 1, 1, 1],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let geometry = ChannelL2NormBf16Geometry::new(x, &arithmetic_gamma, y, attrs)?;
        if x.minimum_storage_bytes()? > isize::MAX as u64 {
            return Err(invalid("X exceeds address space"));
        }
        Ok(Self(geometry))
    }
    pub const fn arithmetic_geometry(self) -> ChannelL2NormBf16Geometry {
        self.0
    }
    pub const fn batch(self) -> u64 {
        self.0.batch()
    }
    pub const fn channels(self) -> u64 {
        self.0.channels()
    }
    pub const fn height(self) -> u64 {
        self.0.height()
    }
    pub const fn width(self) -> u64 {
        self.0.width()
    }
    pub const fn epsilon(self) -> CanonicalRational {
        self.0.epsilon()
    }
    pub const fn scale(self) -> CanonicalRational {
        self.0.scale()
    }
    pub fn effective_epsilon(self) -> f32 {
        self.0.effective_epsilon()
    }
    pub fn effective_scale(self) -> f32 {
        self.0.effective_scale()
    }
    pub const fn element_count(self) -> u64 {
        self.0.element_count()
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.0.logical_arithmetic_operations()
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 image channel L2: {}", reason.into()),
    }
}
