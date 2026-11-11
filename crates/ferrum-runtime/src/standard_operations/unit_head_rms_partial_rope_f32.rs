//! Unit-gain F32 head RMS followed by rotation of a leading channel prefix.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID: &str =
    "operation.unit_head_rms_partial_rope.f32";
pub const UNIT_HEAD_RMS_PARTIAL_ROPE_F32_CAPABILITY_ID: &str =
    "capability.operation.unit_head_rms_partial_rope.f32";

/// X/Y are F32[R,C]; cos/sin are F32[R,rotary_dim], shared across heads.
/// Each full head is normalized with unit gain: r=RN32(1/sqrt(mean(x*x)+eps)),
/// n=RN32(x*r).
/// The leading even prefix uses half-split rotation, with two separately
/// rounded products and a rounded sum. The tail is normalized but unrotated.
/// Providers declare their reduction and inverse-root profile. There is no
/// gamma tensor, hidden unit-weight allocation, FMA or half-precision boundary.
pub fn unit_head_rms_partial_rope_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = fused_head_rms_partial_rope_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        UNIT_HEAD_RMS_PARTIAL_ROPE_F32_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    let matrix = vec![symbol("rows"), symbol("columns")];
    let frequency = vec![symbol("rows"), symbol("rotary_dim")];
    descriptor.inputs = vec![
        contiguous_tensor(matrix.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(frequency.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(frequency, [ElementType::F32], TensorAccess::Read)?,
    ];
    descriptor.outputs = vec![contiguous_tensor(
        matrix,
        [ElementType::F32],
        TensorAccess::Write,
    )?];
    descriptor.resources = ResourceRequirements {
        minimum_value_alignment_bytes: 16,
        ..no_auxiliary_resources()
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitHeadRmsPartialRopeF32Geometry {
    norm: HeadRmsNormF32Geometry,
    rotary_dim: u64,
    arithmetic: u64,
}
impl UnitHeadRmsPartialRopeF32Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        cosine: &ResolvedTensorSpec,
        sine: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        head_dim: u64,
        rotary_dim: u64,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        // Reuse shape/epsilon validation only. This is metadata, never a
        // weight source, runtime operand or allocation for synthetic gamma.
        let shape = ResolvedTensorSpec::new(
            vec![head_dim],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let norm = HeadRmsNormF32Geometry::new(input, &shape, output, head_dim, epsilon)?;
        if rotary_dim == 0
            || !rotary_dim.is_multiple_of(2)
            || rotary_dim > head_dim
            || cosine != sine
            || cosine.element_type() != ElementType::F32
            || cosine.layout() != &ResolvedTensorLayout::Contiguous
            || cosine.dimensions() != [norm.rows(), rotary_dim]
            || cosine.minimum_storage_bytes()? > isize::MAX as u64
        {
            return Err(invalid(
                "requires F32 cos/sin[rows,rotary_dim] and even 0<rotary_dim<=head_dim",
            ));
        }
        let arithmetic = head_dim
            .checked_mul(3)
            .and_then(|n| n.checked_add(3))
            .and_then(|n| n.checked_add((rotary_dim / 2).checked_mul(7)?))
            .and_then(|n| n.checked_mul(norm.elements() / head_dim))
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        Ok(Self {
            norm,
            rotary_dim,
            arithmetic,
        })
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
    pub const fn elements(self) -> u64 {
        self.norm.elements()
    }
    pub const fn epsilon(self) -> CanonicalRational {
        self.norm.epsilon()
    }
    pub fn effective_epsilon(self) -> f32 {
        self.norm.effective_epsilon()
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("unit F32 head RMS/partial RoPE: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn prefix_geometry_retains_full_head_normalization_and_has_no_gamma_operand() {
        let x = t(&[3, 256], ElementType::F32);
        let c = t(&[3, 48], ElementType::F32);
        let eps = CanonicalRational::new(1, 1000000).unwrap();
        let g = UnitHeadRmsPartialRopeF32Geometry::new(&x, &c, &c, &x, 64, 48, eps).unwrap();
        assert_eq!((g.elements(), g.head_dim(), g.rotary_dim()), (768, 64, 48));
        for rot in [0, 47, 66] {
            // TensorSpec itself rejects an empty dimension. Keep a valid
            // physical operand so the zero attribute reaches this geometry.
            let f = t(&[3, rot.max(1)], ElementType::F32);
            assert!(UnitHeadRmsPartialRopeF32Geometry::new(&x, &f, &f, &x, 64, rot, eps).is_err());
        }
        assert!(UnitHeadRmsPartialRopeF32Geometry::new(
            &x,
            &t(&[3, 48], ElementType::Bf16),
            &c,
            &x,
            64,
            48,
            eps
        )
        .is_err());
        assert!(UnitHeadRmsPartialRopeF32Geometry::new(
            &x,
            &t(&[1, 48], ElementType::F32),
            &c,
            &x,
            64,
            48,
            eps
        )
        .is_err());
        let contract = unit_head_rms_partial_rope_f32_contract().unwrap();
        assert_eq!(contract.descriptor().inputs.len(), 3);
        assert_eq!(
            contract.descriptor().resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
