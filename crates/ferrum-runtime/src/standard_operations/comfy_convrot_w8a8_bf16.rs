//! Rotated signed INT8 weights and independent per-output F32 scales.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const COMFY_CONVROT_W8A8_BF16_OPERATION_ID: &str = "operation.comfy_convrot_w8a8.bf16";
pub const COMFY_CONVROT_W8A8_BF16_CAPABILITY_ID: &str =
    "capability.operation.comfy_convrot_w8a8.bf16";

/// BF16 X[M,K], already-rotated I8 W[N,K], F32 channel scales [N,1] -> BF16
/// Y[M,N]. Scale shape is preserved from the canonical checkpoint; scalar,
/// [1], [N] and implicit broadcasting belong to other, unsupported contracts.
/// Regular H256, BF16/A8 row quantization, INT32 GEMM and epilogue follow the
/// separately versioned staged numerical declaration. No bias/correction.
pub fn comfy_convrot_w8a8_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = dense_linear_contract_with_type(
        COMFY_CONVROT_W8A8_BF16_OPERATION_ID,
        ContractVersion::new(1, 0),
        COMFY_CONVROT_W8A8_BF16_CAPABILITY_ID,
        ElementType::Bf16,
        OracleSpec::OperationDefined,
    )?
    .descriptor;
    let symbol = |name: &str| DimensionConstraint::Symbol(name.to_owned());
    descriptor.inputs[1] = contiguous_tensor(
        vec![symbol("out_features"), symbol("in_features")],
        [ElementType::I8],
        TensorAccess::Read,
    )?;
    descriptor.inputs.push(contiguous_tensor(
        vec![symbol("out_features"), DimensionConstraint::Exact(1)],
        [ElementType::F32],
        TensorAccess::Read,
    )?);
    descriptor.resources.scratch = ResourcePresenceRequirement::Required;
    descriptor.resources.minimum_value_alignment_bytes = 16;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComfyConvRotW8Scratch {
    pub rotated_activation: u64,
    pub quantized_activation: u64,
    pub row_scales: u64,
    pub accumulator: u64,
    pub total_bytes: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComfyConvRotW8A8Geometry {
    rows: u64,
    inputs: u64,
    outputs: u64,
    scratch: ComfyConvRotW8Scratch,
    arithmetic: u64,
}
impl ComfyConvRotW8A8Geometry {
    pub fn new(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
    ) -> Result<Self, VNextError> {
        Self::new_for(inputs, output, k, n, false, ElementType::Bf16)
    }

    /// Gate|up [M,2K] is consumed without a BF16 activation materialization;
    /// rotated/quantized scratch retains K columns, as in the raw contract.
    pub fn new_swiglu_f32(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
    ) -> Result<Self, VNextError> {
        Self::new_for(inputs, output, k, n, true, ElementType::Bf16)
    }

    /// F32 activation/output with the same direct-I8 canonical storage and
    /// scratch geometry. This does not select the BF16 quantization boundary.
    pub fn new_f32(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
    ) -> Result<Self, VNextError> {
        Self::new_for(inputs, output, k, n, false, ElementType::F32)
    }

    fn new_for(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
        swiglu: bool,
        activation_type: ElementType,
    ) -> Result<Self, VNextError> {
        if inputs.len() != 3 {
            return Err(invalid("requires three input ports"));
        }
        let [m, width] = inputs[0].dimensions() else {
            return Err(invalid("activation must be rank two"));
        };
        let activation_width = k
            .checked_mul(if swiglu { 2 } else { 1 })
            .ok_or_else(|| invalid("activation width overflows"))?;
        if *m == 0
            || k == 0
            || n == 0
            || *width != activation_width
            || !k.is_multiple_of(256)
            || !n.is_multiple_of(4)
            || [*m, k, n].iter().any(|value| *value > i32::MAX as u64)
            || k.checked_mul(128 * 128)
                .is_none_or(|value| value > i32::MAX as u64)
        {
            return Err(invalid("positive i32 dimensions require K%256=0, N%4=0 and nonoverflowing signed INT8 dot products"));
        }
        for ((tensor, shape), dtype) in inputs
            .iter()
            .zip([vec![*m, activation_width], vec![n, k], vec![n, 1]])
            .zip([activation_type, ElementType::I8, ElementType::F32])
        {
            if tensor.dimensions() != shape
                || tensor.element_type() != dtype
                || tensor.layout() != &ResolvedTensorLayout::Contiguous
            {
                return Err(invalid(
                    "canonical INT8 weight/per-output scale dtype, shape or layout differs",
                ));
            }
        }
        if output.dimensions() != [*m, n]
            || output.element_type() != activation_type
            || output.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid(
                "output must be contiguous [M,N] with the selected activation dtype",
            ));
        }
        for tensor in inputs.iter().copied().chain(std::iter::once(output)) {
            if tensor.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let bytes = |a: u64, b: u64, size: u64| {
            a.checked_mul(b)
                .and_then(|value| value.checked_mul(size))
                .ok_or_else(|| invalid("scratch extent overflows"))
        };
        let mut end = 0_u64;
        let mut reserve = |length: u64| -> Result<u64, VNextError> {
            let offset = end
                .checked_add(255)
                .map(|value| value & !255)
                .ok_or_else(|| invalid("alignment overflows"))?;
            end = offset
                .checked_add(length)
                .filter(|value| *value <= isize::MAX as u64)
                .ok_or_else(|| invalid("scratch exceeds address space"))?;
            Ok(offset)
        };
        let rotated_activation = reserve(bytes(*m, k, 4)?)?;
        let quantized_activation = reserve(bytes(*m, k, 1)?)?;
        let row_scales = reserve(bytes(*m, 1, 4)?)?;
        let accumulator = reserve(bytes(*m, n, 4)?)?;
        let scratch = ComfyConvRotW8Scratch {
            rotated_activation,
            quantized_activation,
            row_scales,
            accumulator,
            total_bytes: end,
        };
        let mut arithmetic = m
            .checked_mul(k)
            .and_then(|value| value.checked_mul(n))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| invalid("logical work overflows"))?;
        if swiglu {
            arithmetic = m
                .checked_mul(k)
                .and_then(|value| value.checked_mul(5))
                .and_then(|value| arithmetic.checked_add(value))
                .ok_or_else(|| invalid("SwiGLU logical work overflows"))?;
        }
        Ok(Self {
            rows: *m,
            inputs: k,
            outputs: n,
            scratch,
            arithmetic,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn inputs(self) -> u64 {
        self.inputs
    }
    pub const fn outputs(self) -> u64 {
        self.outputs
    }
    pub const fn scratch(self) -> ComfyConvRotW8Scratch {
        self.scratch
    }
    /// Prefix used when GEMM stores the final scaled output directly. Every
    /// rotation/quantization/decoded-weight span remains unchanged; the trailing
    /// I32 accumulator is omitted. This does not assert native availability.
    pub const fn scratch_without_accumulator_bytes(self) -> u64 {
        self.scratch.accumulator
    }
    pub const fn output_elements(self) -> u64 {
        self.rows * self.outputs
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("Comfy ConvRot W8A8: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    fn geometry(k: u64, scale_shape: &[u64]) -> Result<ComfyConvRotW8A8Geometry, VNextError> {
        ComfyConvRotW8A8Geometry::new(
            &[
                &tensor(&[3, k], ElementType::Bf16),
                &tensor(&[4, k], ElementType::I8),
                &tensor(scale_shape, ElementType::F32),
            ],
            &tensor(&[3, 4], ElementType::Bf16),
            k,
            4,
        )
    }
    #[test]
    fn w8_scale_scope_and_integer_overflow_boundary_are_explicit() {
        let shape = geometry(256, &[4, 1]).unwrap();
        let scratch = shape.scratch();
        assert_eq!(scratch.total_bytes, scratch.accumulator + 3 * 4 * 4);
        assert_eq!(
            shape.scratch_without_accumulator_bytes() + 3 * 4 * 4,
            scratch.total_bytes
        );
        assert!(scratch.accumulator >= scratch.row_scales + 3 * 4);
        assert!(geometry(130816, &[4, 1]).is_ok());
        assert!(geometry(131072, &[4, 1]).is_err());
        for shape in [&[][..], &[1][..], &[4][..], &[1, 1][..]] {
            assert!(geometry(256, shape).is_err());
        }
        let w4 = ComfyConvRotW4A8Geometry::new(
            &[
                &tensor(&[3, 256], ElementType::Bf16),
                &tensor(&[4, 128], ElementType::I8),
                &tensor(&[4, 16], ElementType::U8),
                &tensor(&[4], ElementType::F32),
                &tensor(&[16], ElementType::F32),
            ],
            &tensor(&[3, 4], ElementType::Bf16),
            256,
            4,
        )
        .unwrap();
        assert!(scratch.total_bytes < w4.scratch().total_bytes);
    }
}
