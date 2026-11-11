//! Full-F32 patch projection and bit-preserving matrix regrouping.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const PATCH_PROJECTION_F32_OPERATION_ID: &str = "operation.patch_projection.f32";
pub const PATCH_PROJECTION_F32_CAPABILITY_ID: &str = "capability.operation.patch_projection.f32";
pub const RESHAPE_COPY_F32_OPERATION_ID: &str = "operation.reshape_copy.f32";
pub const RESHAPE_COPY_F32_CAPABILITY_ID: &str = "capability.operation.reshape_copy.f32";

/// F32 X[R,C*T*Ph*Pw], original contiguous W[O,C,T,Ph,Pw], bias[O]
/// -> Y[R,O]. The patch order is C,T,Ph,Pw. Complete the F32 dot result,
/// then one F32 bias add, with the existing F32 affine error domain/profile.
/// Weight rank and physical shape remain five; no BF16 dot boundary occurs.
pub fn patch_projection_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(
        TensorPrimitiveBf16::PatchProjection,
        PATCH_PROJECTION_F32_OPERATION_ID,
        PATCH_PROJECTION_F32_CAPABILITY_ID,
    )
}
/// Exact physical bit copy, F32[R,H] -> F32[rows,columns], with equal element
/// count. No transpose, arithmetic, inferred broadcasting or storage alias.
pub fn reshape_copy_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(
        TensorPrimitiveBf16::ReshapeCopy,
        RESHAPE_COPY_F32_OPERATION_ID,
        RESHAPE_COPY_F32_CAPABILITY_ID,
    )
}
fn contract(
    base: TensorPrimitiveBf16,
    operation: &str,
    capability: &str,
) -> Result<StandardOperationContract, VNextError> {
    let mut d = base.contract()?.descriptor;
    d.id = OperationId::new(operation)?;
    d.provider = provider_requirement(capability, ContractVersion::new(1, 0))?;
    for port in &mut d.inputs {
        *port = contiguous_tensor(
            port.dimensions().to_vec(),
            [ElementType::F32],
            TensorAccess::Read,
        )?;
    }
    for port in &mut d.outputs {
        *port = contiguous_tensor(
            port.dimensions().to_vec(),
            [ElementType::F32],
            TensorAccess::Write,
        )?;
    }
    d.validate()?;
    Ok(StandardOperationContract { descriptor: d })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchProjectionF32Geometry(AffineLinearF32Geometry);
impl PatchProjectionF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        w: &ResolvedTensorSpec,
        b: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [out, channels, time, height, width] = w.dimensions() else {
            return Err(invalid(
                "patch weight must retain original O,C,T,Ph,Pw rank five",
            ));
        };
        if w.element_type() != ElementType::F32
            || w.layout() != &ResolvedTensorLayout::Contiguous
            || [*out, *channels, *time, *height, *width].contains(&0)
        {
            return Err(invalid("patch weight must be nonempty contiguous F32"));
        }
        let inner = channels
            .checked_mul(*time)
            .and_then(|n| n.checked_mul(*height))
            .and_then(|n| n.checked_mul(*width))
            .ok_or_else(|| invalid("patch flattened extent overflows"))?;
        // This derived BLAS geometry is metadata only. The public contract,
        // WeightReference and original physical component retain their rank.
        let flattened = ResolvedTensorSpec::new(
            vec![*out, inner],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        if flattened.minimum_storage_bytes()? != w.minimum_storage_bytes()? {
            return Err(invalid(
                "flattened patch extent differs from original weight",
            ));
        }
        Ok(Self(AffineLinearF32Geometry::new(
            x, &flattened, b, y, inner, *out,
        )?))
    }
    pub const fn affine(self) -> AffineLinearF32Geometry {
        self.0
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReshapeCopyF32Geometry {
    elements: u64,
}
impl ReshapeCopyF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        rows: u64,
        columns: u64,
    ) -> Result<Self, VNextError> {
        if [x, y].iter().any(|t| {
            t.dimensions().len() != 2
                || t.dimensions().contains(&0)
                || t.element_type() != ElementType::F32
                || t.layout() != &ResolvedTensorLayout::Contiguous
        }) || y.dimensions() != [rows, columns]
            || x.minimum_storage_bytes()? != y.minimum_storage_bytes()?
            || x.minimum_storage_bytes()? > isize::MAX as u64
        {
            return Err(invalid("reshape requires equal original F32 matrix extents and exact declared output shape"));
        }
        Ok(Self {
            elements: x.minimum_storage_bytes()? / 4,
        })
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn bytes(self) -> u64 {
        self.elements * 4
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 vision layout: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: &[u64], e: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(d.to_vec(), e, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn patch_keeps_real_rank_five_and_checked_flattened_axis_order() {
        let x = t(&[3, 24], ElementType::F32);
        let w = t(&[8, 3, 2, 2, 2], ElementType::F32);
        let b = t(&[8], ElementType::F32);
        let y = t(&[3, 8], ElementType::F32);
        let g = PatchProjectionF32Geometry::new(&x, &w, &b, &y)
            .unwrap()
            .affine();
        assert_eq!((g.rows(), g.inputs(), g.outputs()), (3, 24, 8));
        for bad in [
            t(&[8, 24], ElementType::F32),
            t(&[8, 3, 2, 2, 2], ElementType::Bf16),
            t(&[8, 3, 2, 2, 3], ElementType::F32),
        ] {
            assert!(PatchProjectionF32Geometry::new(&x, &bad, &b, &y).is_err());
        }
        assert!(
            PatchProjectionF32Geometry::new(&x, &w, &t(&[1, 8], ElementType::F32), &y).is_err()
        );
    }
    #[test]
    fn reshape_preserves_exact_scalar_extent_and_rejects_half_and_wrong_declarations() {
        let x = t(&[8, 3], ElementType::F32);
        let y = t(&[2, 12], ElementType::F32);
        assert_eq!(
            ReshapeCopyF32Geometry::new(&x, &y, 2, 12).unwrap().bytes(),
            96
        );
        assert!(ReshapeCopyF32Geometry::new(&x, &y, 8, 3).is_err());
        assert!(ReshapeCopyF32Geometry::new(&x, &t(&[2, 11], ElementType::F32), 2, 11).is_err());
        assert!(ReshapeCopyF32Geometry::new(&x, &t(&[2, 12], ElementType::Bf16), 2, 12).is_err());
    }
}
