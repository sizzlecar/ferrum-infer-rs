//! Independent quantized dense operation; it does not weaken BF16 dense's
//! multiplication/accumulation error contract.

use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const DENSE_LINEAR_FP8_BF16_OPERATION_ID: &str = "operation.dense_linear_fp8.bf16";
pub const DENSE_LINEAR_FP8_BF16_CAPABILITY_ID: &str = "capability.operation.dense_linear_fp8.bf16";

/// Logical X[M,K], W[N,K], Y[M,N] are BF16. The selected physical weight ABI
/// must be explicit; the first provider consumes exact OI E4M3FN bytes and one
/// multiplicative F32 scale (WholeTensor, not inverse or per-channel scales).
/// A provider declares activation quantization, scale rounding, reduction and
/// epilogue rounding. Quantization error is additional to FP32 reduction error:
/// this operation does not inherit `dense_linear_bf16`'s error bound or promise
/// equivalence to a particular deployed implementation. Qualification is separate.
/// There is no bias. Output cannot alias inputs. Quantized activation, row scales
/// and native workspace are required, core-admitted invocation scratch; binding
/// and persistent storage are forbidden. No hidden device allocation is allowed.
pub fn dense_linear_fp8_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = dense_linear_contract_with_type(
        DENSE_LINEAR_FP8_BF16_OPERATION_ID,
        ContractVersion::new(1, 0),
        DENSE_LINEAR_FP8_BF16_CAPABILITY_ID,
        ElementType::Bf16,
        OracleSpec::OperationDefined,
    )?
    .descriptor;
    descriptor.resources.scratch = ResourcePresenceRequirement::Required;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Logical geometry only; neither resource authority nor physical weight proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearFp8Bf16Geometry {
    rows: u64,
    inputs: u64,
    outputs: u64,
    arithmetic: u64,
}

impl DenseLinearFp8Bf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        weight: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        inputs: u64,
        outputs: u64,
    ) -> Result<Self, VNextError> {
        let [rows, actual_inputs] = input.dimensions() else {
            return Err(invalid("input must be [M,K]"));
        };
        if *rows == 0
            || inputs == 0
            || outputs == 0
            || *actual_inputs != inputs
            || weight.dimensions() != [outputs, inputs]
            || output.dimensions() != [*rows, outputs]
            || [input, weight, output].iter().any(|tensor| {
                tensor.element_type() != ElementType::Bf16
                    || tensor.layout() != &ResolvedTensorLayout::Contiguous
            })
        {
            return Err(invalid(
                "requires contiguous BF16 X[M,K], W[N,K], Y[M,N] matching positive attributes",
            ));
        }
        for tensor in [input, weight, output] {
            let bytes = tensor.minimum_storage_bytes()?;
            if bytes > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let arithmetic = rows
            .checked_mul(inputs)
            .and_then(|n| n.checked_mul(outputs))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            rows: *rows,
            inputs,
            outputs,
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
    /// Dense mathematical 2MNK work; quantization is not counted as GEMM FLOPs.
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FP8/BF16 dense geometry: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn contract_is_independent_and_requires_managed_scratch() {
        let new = dense_linear_fp8_bf16_contract().unwrap();
        let old = dense_linear_bf16_contract().unwrap();
        assert_ne!(new.descriptor().id, old.descriptor().id);
        assert_ne!(
            new.descriptor().fingerprint().unwrap(),
            old.descriptor().fingerprint().unwrap()
        );
        assert_eq!(
            new.descriptor().resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            new.descriptor().resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            new.descriptor().resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(new.descriptor().outputs[0].alias(), &AliasPolicy::NoAlias);
    }
    #[test]
    fn rectangular_geometry_and_logical_work_are_checked() {
        let x = tensor(&[3, 16], ElementType::Bf16);
        let w = tensor(&[32, 16], ElementType::Bf16);
        let y = tensor(&[3, 32], ElementType::Bf16);
        let g = DenseLinearFp8Bf16Geometry::new(&x, &w, &y, 16, 32).unwrap();
        assert_eq!(
            (g.rows(), g.inputs(), g.outputs(), g.output_elements()),
            (3, 16, 32, 96)
        );
        assert_eq!(g.logical_arithmetic_operations(), 3072);
        assert!(DenseLinearFp8Bf16Geometry::new(&x, &w, &y, 32, 16).is_err());
        assert!(DenseLinearFp8Bf16Geometry::new(
            &x,
            &w,
            &tensor(&[3, 32], ElementType::F16),
            16,
            32
        )
        .is_err());
        assert!(DenseLinearFp8Bf16Geometry::new(
            &x,
            &tensor(&[16, 32], ElementType::Bf16),
            &y,
            16,
            32
        )
        .is_err());
    }
}
