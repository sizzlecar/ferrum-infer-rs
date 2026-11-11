//! F32 SwiGLU activation retained through the ConvRot H256 stages.
use super::*;

pub const COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID: &str =
    "operation.comfy_convrot_codebook_w4a8.swiglu_f32.bf16";
pub const COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_CAPABILITY_ID: &str =
    "capability.operation.comfy_convrot_codebook_w4a8.swiglu_f32.bf16";
pub const COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID: &str =
    "operation.comfy_convrot_w8a8.swiglu_f32.bf16";
pub const COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_CAPABILITY_ID: &str =
    "capability.operation.comfy_convrot_w8a8.swiglu_f32.bf16";

/// BF16 X[M,2K] contains contiguous gate|up halves in each row. Compute
/// z=RN32(RN32(gate/RN32(1+exp_f32(-gate)))*up), retain z in F32, then use
/// regular H256 and the existing BF16/A8, INT32-dot, BF16-store boundaries.
/// In particular there is no BF16 store of SiLU or its product before H256.
/// The versioned provider profile identifies its exponential implementation;
/// division/add/multiply are RN32, with FMA and FTZ disabled. No row-count
/// threshold may change this activation arithmetic. This is an independent
/// profile, not a claim of equivalence to an unqualified fast-math binary.
/// Canonical W4 ports, rotation/scale groups and invocation scratch otherwise
/// follow `comfy_convrot_w4a8_bf16_contract`; K is the activated width.
pub fn comfy_convrot_swiglu_f32_w4a8_bf16_contract() -> Result<StandardOperationContract, VNextError>
{
    swiglu_contract(
        comfy_convrot_w4a8_bf16_contract()?,
        COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID,
        COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_CAPABILITY_ID,
    )
}

/// The same F32 SwiGLU activation, with canonical signed I8 W[N,K] and exact
/// F32 scales [N,1]. No decoded-weight scratch is allocated for W8.
pub fn comfy_convrot_swiglu_f32_w8a8_bf16_contract() -> Result<StandardOperationContract, VNextError>
{
    swiglu_contract(
        comfy_convrot_w8a8_bf16_contract()?,
        COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID,
        COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_CAPABILITY_ID,
    )
}

fn swiglu_contract(
    contract: StandardOperationContract,
    operation: &str,
    capability: &str,
) -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = contract.descriptor;
    descriptor.id = OperationId::new(operation)?;
    descriptor.provider = provider_requirement(capability, ContractVersion::new(1, 0))?;
    descriptor.inputs[0] = contiguous_tensor(
        vec![
            DimensionConstraint::Symbol("rows".into()),
            DimensionConstraint::Symbol("gate_up_columns".into()),
        ],
        [ElementType::Bf16],
        TensorAccess::Read,
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
    fn fused_gate_up_geometry_preserves_scratch_and_rejects_raw_width() {
        let raw = tensor(&[3, 256], ElementType::Bf16);
        let fused = tensor(&[3, 512], ElementType::Bf16);
        let w4 = tensor(&[4, 128], ElementType::I8);
        let relative = tensor(&[4, 16], ElementType::U8);
        let channel4 = tensor(&[4], ElementType::F32);
        let codebook = tensor(&[16], ElementType::F32);
        let w8 = tensor(&[4, 256], ElementType::I8);
        let channel8 = tensor(&[4, 1], ElementType::F32);
        let output = tensor(&[3, 4], ElementType::Bf16);
        let a = [&raw, &w4, &relative, &channel4, &codebook];
        let b = [&fused, &w4, &relative, &channel4, &codebook];
        let raw4 = ComfyConvRotW4A8Geometry::new(&a, &output, 256, 4).unwrap();
        let fused4 = ComfyConvRotW4A8Geometry::new_swiglu_f32(&b, &output, 256, 4).unwrap();
        assert_eq!(raw4.scratch(), fused4.scratch());
        assert_eq!(
            fused4.logical_arithmetic_operations(),
            raw4.logical_arithmetic_operations() + 5 * 3 * 256
        );
        assert!(ComfyConvRotW4A8Geometry::new(&b, &output, 256, 4).is_err());
        assert!(ComfyConvRotW4A8Geometry::new_swiglu_f32(&a, &output, 256, 4).is_err());
        let a = [&raw, &w8, &channel8];
        let b = [&fused, &w8, &channel8];
        let raw8 = ComfyConvRotW8A8Geometry::new(&a, &output, 256, 4).unwrap();
        let fused8 = ComfyConvRotW8A8Geometry::new_swiglu_f32(&b, &output, 256, 4).unwrap();
        assert_eq!(raw8.scratch(), fused8.scratch());
        assert!(ComfyConvRotW8A8Geometry::new(&b, &output, 256, 4).is_err());
        assert!(ComfyConvRotW8A8Geometry::new_swiglu_f32(&a, &output, 256, 4).is_err());
        assert!(fused8.scratch().total_bytes < fused4.scratch().total_bytes);
        assert!(ComfyConvRotW8A8Geometry::new_swiglu_f32(&b, &output, u64::MAX, 4).is_err());
    }

    #[test]
    fn fused_contracts_are_distinct_and_retain_canonical_component_types() {
        for (raw, fused) in [
            (
                comfy_convrot_w4a8_bf16_contract().unwrap(),
                comfy_convrot_swiglu_f32_w4a8_bf16_contract().unwrap(),
            ),
            (
                comfy_convrot_w8a8_bf16_contract().unwrap(),
                comfy_convrot_swiglu_f32_w8a8_bf16_contract().unwrap(),
            ),
        ] {
            assert_ne!(raw.descriptor().id, fused.descriptor().id);
            assert_eq!(raw.descriptor().inputs[1..], fused.descriptor().inputs[1..]);
            assert_eq!(
                fused.descriptor().resources.scratch,
                ResourcePresenceRequirement::Required
            );
        }
    }
}
