//! Full-row F32 matrix projection with an independent F32 bias boundary.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const AFFINE_LINEAR_F32_OPERATION_ID: &str = "operation.affine_linear.f32";
pub const AFFINE_LINEAR_F32_CAPABILITY_ID: &str = "capability.operation.affine_linear.f32";

/// F32 X[M,K], OI weight[N,K], bias[N] -> F32 Y[M,N]. First complete
/// the F32 dense result, then add the F32 bias with round-to-nearest-even.
/// This boundary excludes TF32, F16/BF16 operands and a bias downcast. It does
/// not promise bitwise equality with a different GEMM/bias implementation.
///
/// The dense oracle's finite, no-overflow/underflow domain and error bound
/// apply to the dot. The final add introduces at most one further F32 rounding:
/// |error| <= (1+u)*gamma(2K,u)*sum|x*w| + u*|exact_dot+bias|, u=2^-24.
/// This is a numerical domain, not a runtime scan of input values.
pub fn affine_linear_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = dense_linear_contract_with_type(
        AFFINE_LINEAR_F32_OPERATION_ID,
        ContractVersion::new(1, 0),
        AFFINE_LINEAR_F32_CAPABILITY_ID,
        ElementType::F32,
        OracleSpec::OperationDefined,
    )?
    .descriptor;
    descriptor.inputs.push(contiguous_tensor(
        vec![DimensionConstraint::Symbol("out_features".into())],
        [ElementType::F32],
        TensorAccess::Read,
    )?);
    descriptor.resources = no_auxiliary_resources();
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Allocation-free geometry proof. Native BLAS uses positive i32 dimensions;
/// every tensor and the complete logical work must fit the declared ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineLinearF32Geometry {
    rows: u64,
    inputs: u64,
    outputs: u64,
    arithmetic: u64,
}
impl AffineLinearF32Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        weight: &ResolvedTensorSpec,
        bias: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
    ) -> Result<Self, VNextError> {
        let [m, width] = input.dimensions() else {
            return Err(invalid("input must be rank two"));
        };
        if *width != k
            || [*m, k, n]
                .iter()
                .any(|dimension| *dimension == 0 || *dimension > i32::MAX as u64)
            || weight.dimensions() != [n, k]
            || bias.dimensions() != [n]
            || output.dimensions() != [*m, n]
            || [input, weight, bias, output].iter().any(|tensor| {
                tensor.element_type() != ElementType::F32
                    || tensor.layout() != &ResolvedTensorLayout::Contiguous
            })
        {
            return Err(invalid("requires contiguous F32 X[M,K], W[N,K], bias[N], Y[M,N] and positive i32 dimensions"));
        }
        for tensor in [input, weight, bias, output] {
            if tensor.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let arithmetic = m
            .checked_mul(n)
            .and_then(|count| {
                k.checked_mul(2)
                    .and_then(|dot| dot.checked_add(1))
                    .and_then(|per_output| count.checked_mul(per_output))
            })
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            rows: *m,
            inputs: k,
            outputs: n,
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
    pub const fn output_elements(self) -> u64 {
        self.rows * self.outputs
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 affine: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn f32_affine_preserves_exact_bias_shape_precision_and_capacity_boundaries() {
        let input = tensor(&[3, 16], ElementType::F32);
        let weight = tensor(&[4, 16], ElementType::F32);
        let bias = tensor(&[4], ElementType::F32);
        let output = tensor(&[3, 4], ElementType::F32);
        let geometry =
            AffineLinearF32Geometry::new(&input, &weight, &bias, &output, 16, 4).unwrap();
        assert_eq!(geometry.output_elements(), 12);
        assert_eq!(geometry.logical_arithmetic_operations(), 12 * 33);
        for invalid_bias in [
            tensor(&[1, 4], ElementType::F32),
            tensor(&[4], ElementType::Bf16),
            tensor(&[1], ElementType::F32),
        ] {
            assert!(
                AffineLinearF32Geometry::new(&input, &weight, &invalid_bias, &output, 16, 4)
                    .is_err()
            );
        }
        assert!(AffineLinearF32Geometry::new(
            &input,
            &tensor(&[4, 16], ElementType::Bf16),
            &bias,
            &output,
            16,
            4
        )
        .is_err());
        let huge = i32::MAX as u64 + 1;
        assert!(AffineLinearF32Geometry::new(
            &tensor(&[1, huge], ElementType::F32),
            &tensor(&[4, huge], ElementType::F32),
            &bias,
            &tensor(&[1, 4], ElementType::F32),
            huge,
            4
        )
        .is_err());
        let side = i32::MAX as u64;
        assert!(AffineLinearF32Geometry::new(
            &tensor(&[side, side], ElementType::F32),
            &tensor(&[side, side], ElementType::F32),
            &tensor(&[side], ElementType::F32),
            &tensor(&[side, side], ElementType::F32),
            side,
            side
        )
        .is_err());
        let contract = affine_linear_f32_contract().unwrap();
        assert_eq!(contract.descriptor().inputs.len(), 3);
        assert_eq!(
            contract.descriptor().resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
