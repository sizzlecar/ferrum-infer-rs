//! Head RMS normalization followed by leading-channel half-split rotation.
//! This operation has different BF16 boundaries from composing the existing
//! head RMS and half-split RoPE operations; their contracts remain unchanged.
use super::*;
use crate::ResolvedTensorSpec;

pub const FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID: &str =
    "operation.fused_head_rms_partial_rope.bf16";
pub const FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_CAPABILITY_ID: &str =
    "capability.operation.fused_head_rms_partial_rope.bf16";

/// Inputs are BF16 X[R,C], gamma[D], cos[R,rot], sin[R,rot], output BF16 Y[R,C].
/// RMS reduces the full D-element head in F32. With r=rsqrt(mean(X^2)+eps),
/// n=B(RN32(RN32(X*r)*gamma)): there is NO BF16 boundary before gamma.
/// The leading rot channels use half-split pairs; each pair combines F32
/// products before its final B, without rounding individual products to BF16.
/// Tail channels return n, not the original input bits. Cos/sin vary per row
/// and are shared across heads; equality of the two frequency halves is not
/// required. B is BF16 RNE. Reduction/rsqrt/FMA/underflow profiles belong to
/// providers; this contract does not imply Torch binary or model parity.
pub fn fused_head_rms_partial_rope_bf16_contract() -> Result<StandardOperationContract, VNextError>
{
    let matrix = vec![symbol("rows"), symbol("columns")];
    let frequency = vec![symbol("rows"), symbol("rotary_dim")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(matrix.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("head_dim")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(frequency.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(frequency, [ElementType::Bf16], TensorAccess::Read)?,
        ],
        outputs: vec![contiguous_tensor(
            matrix,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("head_dim")?,
            unsigned_attribute("rotary_dim")?,
            positive_epsilon_attribute("epsilon")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked metadata only, without ownership or allocation authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedHeadRmsPartialRopeBf16Geometry {
    norm: HeadRmsNormBf16Geometry,
    rotary_dim: u64,
}
impl FusedHeadRmsPartialRopeBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        cosine: &ResolvedTensorSpec,
        sine: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        head_dim: u64,
        rotary_dim: u64,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let norm = HeadRmsNormBf16Geometry::new(input, gamma, output, head_dim, epsilon)?;
        PartialHalfSplitRopeBf16Geometry::new(input, cosine, sine, output, head_dim, rotary_dim)?;
        let geometry = Self { norm, rotary_dim };
        geometry.logical_arithmetic_operations()?;
        Ok(geometry)
    }
    pub const fn rows(self) -> u64 {
        self.norm.rows()
    }
    pub const fn columns(self) -> u64 {
        self.norm.columns()
    }
    pub const fn head_dim(self) -> u64 {
        self.norm.head_dim()
    }
    pub const fn rotary_dim(self) -> u64 {
        self.rotary_dim
    }
    pub const fn epsilon(self) -> CanonicalRational {
        self.norm.epsilon()
    }
    pub fn effective_epsilon(self) -> f32 {
        self.norm.effective_epsilon()
    }
    pub fn logical_arithmetic_operations(self) -> Result<u64, VNextError> {
        self.rows()
            .checked_mul(self.columns() / self.head_dim())
            .and_then(|heads| {
                self.head_dim()
                    .checked_mul(4)?
                    .checked_add(3)?
                    .checked_add((self.rotary_dim / 2).checked_mul(7)?)?
                    .checked_mul(heads)
            })
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: "fused head RMS/RoPE logical work overflows".into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResolvedTensorLayout;
    fn spec(dimensions: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(dimensions.to_vec(), dtype, ResolvedTensorLayout::Contiguous)
            .unwrap()
    }
    #[test]
    fn accepts_partial_and_full_rotations_and_counts_normalized_tails() {
        for (head, rot) in [(128, 96), (8, 6), (7, 4), (8, 8)] {
            let x = spec(&[3, 2 * head], ElementType::Bf16);
            let g = spec(&[head], ElementType::Bf16);
            let c = spec(&[3, rot], ElementType::Bf16);
            let checked = FusedHeadRmsPartialRopeBf16Geometry::new(
                &x,
                &g,
                &c,
                &c,
                &x,
                head,
                rot,
                CanonicalRational::new(1, 100000).unwrap(),
            )
            .unwrap();
            assert_eq!(
                checked.logical_arithmetic_operations().unwrap(),
                6 * (4 * head + 3 + 7 * (rot / 2))
            );
            assert_eq!(checked.effective_epsilon(), 1e-5_f32);
        }
    }
    #[test]
    fn rejects_bad_gamma_precision_frequencies_partition_and_epsilon() {
        let x = spec(&[3, 256], ElementType::Bf16);
        let g = spec(&[128], ElementType::Bf16);
        let c = spec(&[3, 96], ElementType::Bf16);
        let eps = CanonicalRational::new(1, 100000).unwrap();
        let check = |g: &ResolvedTensorSpec, c: &ResolvedTensorSpec, h, r, e| {
            FusedHeadRmsPartialRopeBf16Geometry::new(&x, g, c, c, &x, h, r, e)
        };
        assert!(check(&spec(&[128], ElementType::F32), &c, 128, 96, eps).is_err());
        assert!(check(&spec(&[127], ElementType::Bf16), &c, 128, 96, eps).is_err());
        assert!(check(&g, &spec(&[2, 96], ElementType::Bf16), 128, 96, eps).is_err());
        for (h, r) in [(128, 0), (128, 95), (128, 130), (127, 96)] {
            assert!(check(&g, &c, h, r, eps).is_err());
        }
        assert!(check(&g, &c, 128, 96, CanonicalRational::new(0, 1).unwrap()).is_err());
        assert!(check(&g, &c, 128, 96, CanonicalRational::new(2, 1).unwrap()).is_err());
        let descriptor = fused_head_rms_partial_rope_bf16_contract()
            .unwrap()
            .descriptor;
        assert_eq!(descriptor.inputs.len(), 4);
        assert_eq!(
            descriptor.id.as_str(),
            FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID
        );
    }
}
