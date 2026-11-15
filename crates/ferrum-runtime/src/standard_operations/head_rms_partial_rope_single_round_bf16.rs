//! Per-head RMS and partial RoPE with F32 normalization/rotation intermediates.
use super::*;
use crate::{ResolvedTensorSpec, SemanticValue};

pub const HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID: &str =
    "operation.head_rms_partial_rope.single_round_bf16";
pub const HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_CAPABILITY_ID: &str =
    "capability.operation.head_rms_partial_rope.single_round_bf16";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadRmsPartialRopeSingleRoundBf16Profile {
    Block128Warp4Fma,
}
impl HeadRmsPartialRopeSingleRoundBf16Profile {
    pub const fn as_str(self) -> &'static str {
        "block128_warp4_fma_v1"
    }
    pub fn attributes(self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([(
            AttributeId::new("profile")?,
            SemanticValue::Text(self.as_str().into()),
        )]))
    }
}

/// BF16 X[R,heads*D], gamma[D], cosine/sine[R,rot] -> BF16 Y[R,heads*D].
/// There is no BF16 normalization or individual-product boundary. With
/// r=rsqrt(RN32(sum/D)+epsilon), n=RN32(RN32(r*x)*gamma). For each rotated
/// channel let p=RN32(r*partner_x). The first half uses FMA32(-p,partner_gamma,+0),
/// the second RN32(p*partner_gamma); call this s. Y=B(FMA32(n,cos,RN32(s*sin))).
/// The unrotated tail is B(n), never the original input. B denotes BF16 RNE.
///
/// The required fixed profile maps a head to 128 lanes; inactive lanes are zero.
/// Each lane starts with RN32(x*x). The first warp butterfly at XOR16 instead
/// computes FMA32(x,x,peer_square); XOR8,4,2,1 then use RN32 additions. Four warp
/// totals reduce with XOR2,1. CUDA uses div.full.f32 / rsqrt.approx.ftz.f32;
/// other arithmetic has gradual underflow. Reference declares its independent
/// sqrt/div approximation. Provider identity distinguishes approximations;
/// this does not claim arbitrary-shape Triton or Torch bitwise equivalence.
pub fn head_rms_partial_rope_single_round_bf16_contract(
) -> Result<StandardOperationContract, VNextError> {
    // Existing signature/shape rules are reusable, its BF16 arithmetic is not.
    let mut descriptor = fused_head_rms_partial_rope_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    let mut attrs = descriptor.attributes.entries().clone();
    let profile = text_choices_attribute(
        "profile",
        [HeadRmsPartialRopeSingleRoundBf16Profile::Block128Warp4Fma.as_str()],
    )?;
    attrs.insert(profile.0, profile.1);
    descriptor.attributes = AttributeSchema::new(attrs)?;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked geometry only: this type grants no allocation/execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadRmsPartialRopeSingleRoundBf16Geometry {
    shape: FusedHeadRmsPartialRopeBf16Geometry,
}
impl HeadRmsPartialRopeSingleRoundBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        cosine: &ResolvedTensorSpec,
        sine: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        head_rms_partial_rope_single_round_bf16_contract()?
            .descriptor
            .validate_attributes(attributes)?;
        let unsigned = |name: &str| match attributes.get(&AttributeId::new(name)?) {
            Some(SemanticValue::Unsigned(v)) => Ok(*v),
            _ => Err(invalid("requires explicit unsigned head/rotary dimensions")),
        };
        let epsilon = match attributes.get(&AttributeId::new("epsilon")?) {
            Some(SemanticValue::Rational(v)) => *v,
            _ => return Err(invalid("requires rational epsilon")),
        };
        let head = unsigned("head_dim")?;
        if head > 128 {
            return Err(invalid("fixed block128 profile requires head_dim <=128"));
        }
        Ok(Self {
            shape: FusedHeadRmsPartialRopeBf16Geometry::new(
                input,
                gamma,
                cosine,
                sine,
                output,
                head,
                unsigned("rotary_dim")?,
                epsilon,
            )?,
        })
    }
    pub const fn rows(self) -> u64 {
        self.shape.rows()
    }
    pub const fn columns(self) -> u64 {
        self.shape.columns()
    }
    pub const fn head_dim(self) -> u64 {
        self.shape.head_dim()
    }
    pub const fn rotary_dim(self) -> u64 {
        self.shape.rotary_dim()
    }
    pub fn effective_epsilon(self) -> f32 {
        self.shape.effective_epsilon()
    }
    pub fn logical_arithmetic_operations(self) -> Result<u64, VNextError> {
        self.shape.logical_arithmetic_operations()
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("single-round head RMS/RoPE: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResolvedTensorLayout;
    fn spec(shape: Vec<u64>) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn explicit_profile_checks_real_head_tail_and_shared_row_tables() {
        for (head, rot, heads) in [(128, 96, 56), (2, 2, 2), (127, 96, 3)] {
            let x = spec(vec![3, head * heads]);
            let gamma = spec(vec![head]);
            let table = spec(vec![3, rot]);
            let mut attrs = HeadRmsPartialRopeSingleRoundBf16Profile::Block128Warp4Fma
                .attributes()
                .unwrap();
            for (name, v) in [
                ("head_dim", SemanticValue::Unsigned(head)),
                ("rotary_dim", SemanticValue::Unsigned(rot)),
                (
                    "epsilon",
                    SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
                ),
            ] {
                attrs.insert(AttributeId::new(name).unwrap(), v);
            }
            let g = HeadRmsPartialRopeSingleRoundBf16Geometry::new(
                &x, &gamma, &table, &table, &x, &attrs,
            )
            .unwrap();
            assert_eq!(
                (g.rows(), g.columns(), g.head_dim(), g.rotary_dim()),
                (3, head * heads, head, rot)
            );
            assert!(HeadRmsPartialRopeSingleRoundBf16Geometry::new(
                &x,
                &gamma,
                &spec(vec![2, rot]),
                &table,
                &x,
                &attrs
            )
            .is_err());
            attrs.insert(
                AttributeId::new("profile").unwrap(),
                SemanticValue::Text("bf16-normalization".into()),
            );
            assert!(HeadRmsPartialRopeSingleRoundBf16Geometry::new(
                &x, &gamma, &table, &table, &x, &attrs
            )
            .is_err());
        }
    }
}
