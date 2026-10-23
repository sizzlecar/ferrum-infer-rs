//! Bounded spatial tile/channel gather, preserving every BF16 bit.
use super::spatial_attention_bf16::checked_nchw;
use super::*;
use crate::{ResolvedTensorSpec, SemanticValue};
pub const SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID: &str =
    "operation.spatial_upsample_gather.bf16";
pub const SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID: &str =
    "capability.operation.spatial_upsample_gather.bf16";
pub const SPATIAL_UPSAMPLE_GATHER_BF16_MAX_TILE: u64 = 64;
/// Y[n,o,h*sh+dy,w*sw+dx] = X[n,o*channel_stride+offset[dy*sw+dx],h,w].
/// Exact opaque-bit copy (including NaN payload and signed zero), no arithmetic.
/// Positive sh/sw/stride, at most 64 tile offsets, no source-index wrapping.
/// Output dimensions are explicit and revalidated against attrs; NoAlias.
pub fn spatial_upsample_gather_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut attrs = BTreeMap::from([
        unsigned_attribute("scale_h")?,
        unsigned_attribute("scale_w")?,
        unsigned_attribute("channel_stride")?,
    ]);
    attrs.insert(
        AttributeId::new("channel_offsets")?,
        AttributeSpec {
            value_kind: AttributeValueKind::Integers,
            required: true,
            constraint: AttributeConstraint::IntegerListLength {
                minimum: 1,
                maximum: 64,
            },
        },
    );
    let descriptor = OperationDescriptor {
        id: OperationId::new(SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID)?,
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
        attributes: AttributeSchema::new(attrs)?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialUpsampleGatherBf16Geometry {
    input: [u64; 4],
    output: [u64; 4],
    scale: [u64; 2],
    stride: u64,
    offsets: Vec<u64>,
    elements: u64,
}
impl SpatialUpsampleGatherBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        spatial_upsample_gather_bf16_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        let x = checked_nchw(input)?;
        let y = checked_nchw(output)?;
        let unsigned = |name: &str| -> Result<u64, VNextError> {
            match attrs.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Unsigned(n)) if *n > 0 => Ok(*n),
                _ => Err(invalid("scale/stride must be positive unsigned")),
            }
        };
        let sh = unsigned("scale_h")?;
        let sw = unsigned("scale_w")?;
        let stride = unsigned("channel_stride")?;
        let tiles = sh
            .checked_mul(sw)
            .filter(|n| *n <= SPATIAL_UPSAMPLE_GATHER_BF16_MAX_TILE)
            .ok_or_else(|| invalid("tile extent exceeds 64"))?;
        let Some(SemanticValue::Integers(offsets)) =
            attrs.get(&AttributeId::new("channel_offsets")?)
        else {
            return Err(invalid("offsets must be integers"));
        };
        if offsets.len() != tiles as usize || offsets.iter().any(|v| *v < 0) {
            return Err(invalid("offsets must be nonnegative and match sh*sw"));
        }
        let offsets: Vec<u64> = offsets.iter().map(|v| *v as u64).collect();
        let max_offset = *offsets
            .iter()
            .max()
            .ok_or_else(|| invalid("offsets empty"))?;
        let last_channel = (y[1] - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(max_offset))
            .ok_or_else(|| invalid("source channel overflows"))?;
        if last_channel >= x[1]
            || y[0] != x[0]
            || Some(y[2]) != x[2].checked_mul(sh)
            || Some(y[3]) != x[3].checked_mul(sw)
        {
            return Err(invalid("output geometry or source channel bound differs"));
        }
        let elements = y
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("output elements overflow"))?;
        Ok(Self {
            input: x,
            output: y,
            scale: [sh, sw],
            stride,
            offsets,
            elements,
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
    pub const fn channel_stride(&self) -> u64 {
        self.stride
    }
    pub fn channel_offsets(&self) -> &[u64] {
        &self.offsets
    }
    pub const fn element_count(&self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(&self) -> u64 {
        0
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 spatial upsample gather: {}", reason.into()),
    }
}
