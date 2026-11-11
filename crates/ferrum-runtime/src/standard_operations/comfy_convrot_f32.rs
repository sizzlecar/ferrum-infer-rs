//! Canonical W4/W8 with independent full-F32 activation/output arithmetic.
use super::*;
pub const COMFY_CONVROT_W4A8_F32_OPERATION_ID: &str = "operation.comfy_convrot_codebook_w4a8.f32";
pub const COMFY_CONVROT_W4A8_F32_CAPABILITY_ID: &str =
    "capability.operation.comfy_convrot_codebook_w4a8.f32";
pub const COMFY_CONVROT_W8A8_F32_OPERATION_ID: &str = "operation.comfy_convrot_w8a8.f32";
pub const COMFY_CONVROT_W8A8_F32_CAPABILITY_ID: &str =
    "capability.operation.comfy_convrot_w8a8.f32";

/// F32 X[M,K] plus canonical W4 packed I8[N,K/2], U8 E4M3[N,K/16],
/// F32 channel[N], F32 codebook[16] -> F32 Y[M,N]. Regular H4^4 stages
/// keep F32 rounding. Scale=max(min(amax,F32_MAX)*RN32(1/127),1e-30).
/// Codes=RNE(RN32(rotated/scale)) clamped to [-128,127], with no BF16/F16
/// conversion of the numerator, denominator, quotient or output. Decode W4
/// to I8 RNE[-127,127], complete nonoverflowing I32 dot, then
/// RN32(RN32(F32(acc)*row_scale)*channel_scale). No bias or correction.
/// Providers declare division, contraction and FTZ behavior; this independently
/// versioned arithmetic does not certify an upstream fast-math binary.
/// Rotation, group sizes and charged scratch match the canonical BF16 storage
/// ABI; the activation/output arithmetic is independent of it.
pub fn comfy_convrot_w4a8_f32_contract() -> Result<StandardOperationContract, VNextError> {
    f32_contract(
        comfy_convrot_w4a8_bf16_contract()?,
        COMFY_CONVROT_W4A8_F32_OPERATION_ID,
        COMFY_CONVROT_W4A8_F32_CAPABILITY_ID,
    )
}
/// The same full-F32 A8 activation and epilogue with direct signed I8 W[N,K]
/// (including -128), exact F32 scales [N,1], and no decoded-weight scratch.
pub fn comfy_convrot_w8a8_f32_contract() -> Result<StandardOperationContract, VNextError> {
    f32_contract(
        comfy_convrot_w8a8_bf16_contract()?,
        COMFY_CONVROT_W8A8_F32_OPERATION_ID,
        COMFY_CONVROT_W8A8_F32_CAPABILITY_ID,
    )
}
fn f32_contract(
    contract: StandardOperationContract,
    operation: &str,
    capability: &str,
) -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = contract.descriptor;
    descriptor.id = OperationId::new(operation)?;
    descriptor.provider = provider_requirement(capability, ContractVersion::new(1, 0))?;
    let shape = |last: &str| {
        vec![
            DimensionConstraint::Symbol("rows".into()),
            DimensionConstraint::Symbol(last.into()),
        ]
    };
    descriptor.inputs[0] =
        contiguous_tensor(shape("in_features"), [ElementType::F32], TensorAccess::Read)?;
    descriptor.outputs[0] = contiguous_tensor(
        shape("out_features"),
        [ElementType::F32],
        TensorAccess::Write,
    )?;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn f32_geometry_rejects_half_ports_and_preserves_per_invocation_scratch() {
        let x = tensor(&[3, 256], ElementType::F32);
        let y = tensor(&[3, 4], ElementType::F32);
        let bx = tensor(&[3, 256], ElementType::Bf16);
        let by = tensor(&[3, 4], ElementType::Bf16);
        let w = tensor(&[4, 128], ElementType::I8);
        let rel = tensor(&[4, 16], ElementType::U8);
        let scale = tensor(&[4], ElementType::F32);
        let cb = tensor(&[16], ElementType::F32);
        let ports = [&x, &w, &rel, &scale, &cb];
        let g = ComfyConvRotW4A8Geometry::new_f32(&ports, &y, 256, 4).unwrap();
        let b = ComfyConvRotW4A8Geometry::new(&[&bx, &w, &rel, &scale, &cb], &by, 256, 4).unwrap();
        assert_eq!(g.scratch(), b.scratch());
        assert!(ComfyConvRotW4A8Geometry::new(&ports, &by, 256, 4).is_err());
        assert!(ComfyConvRotW4A8Geometry::new_f32(&ports, &by, 256, 4).is_err());
        let w = tensor(&[4, 256], ElementType::I8);
        let scale = tensor(&[4, 1], ElementType::F32);
        let ports = [&x, &w, &scale];
        let g = ComfyConvRotW8A8Geometry::new_f32(&ports, &y, 256, 4).unwrap();
        let b = ComfyConvRotW8A8Geometry::new(&[&bx, &w, &scale], &by, 256, 4).unwrap();
        assert_eq!(g.scratch(), b.scratch());
        assert!(ComfyConvRotW8A8Geometry::new_f32(&ports, &by, 256, 4).is_err());
        assert!(ComfyConvRotW8A8Geometry::new_f32(
            &[&x, &w, &tensor(&[4], ElementType::F32)],
            &y,
            256,
            4
        )
        .is_err());
        for contract in [
            comfy_convrot_w4a8_f32_contract().unwrap(),
            comfy_convrot_w8a8_f32_contract().unwrap(),
        ] {
            assert_eq!(
                contract.descriptor().inputs[0].element_types(),
                &BTreeSet::from([ElementType::F32])
            );
            assert_eq!(
                contract.descriptor().outputs[0].element_types(),
                &BTreeSet::from([ElementType::F32])
            );
            assert_eq!(
                contract.descriptor().resources.scratch,
                ResourcePresenceRequirement::Required
            );
        }
    }
}
