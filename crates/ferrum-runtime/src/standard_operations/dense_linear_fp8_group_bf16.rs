//! Independent projections sharing one activation quantization boundary.

use super::*;
use crate::ResolvedTensorSpec;

pub const DENSE_LINEAR_FP8_PAIR_BF16_OPERATION_ID: &str = "operation.dense_linear_fp8_pair.bf16";
pub const DENSE_LINEAR_FP8_PAIR_BF16_CAPABILITY_ID: &str =
    "capability.operation.dense_linear_fp8_pair.bf16";
pub const DENSE_LINEAR_FP8_TRIPLE_BF16_OPERATION_ID: &str =
    "operation.dense_linear_fp8_triple.bf16";
pub const DENSE_LINEAR_FP8_TRIPLE_BF16_CAPABILITY_ID: &str =
    "capability.operation.dense_linear_fp8_triple.bf16";

/// Bounded number of independent projections with identical logical geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseLinearFp8Group {
    Pair,
    Triple,
}

impl DenseLinearFp8Group {
    pub const fn count(self) -> usize {
        match self {
            Self::Pair => 2,
            Self::Triple => 3,
        }
    }

    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Pair => DENSE_LINEAR_FP8_PAIR_BF16_OPERATION_ID,
            Self::Triple => DENSE_LINEAR_FP8_TRIPLE_BF16_OPERATION_ID,
        }
    }

    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Pair => DENSE_LINEAR_FP8_PAIR_BF16_CAPABILITY_ID,
            Self::Triple => DENSE_LINEAR_FP8_TRIPLE_BF16_CAPABILITY_ID,
        }
    }
}

/// Input ordinal 0 is X[M,K]; ordinals 1..=count are independent W[N,K].
/// Output ordinal i is Y_i[M,N]. All logical ports are contiguous BF16, and
/// each weight retains its own declared physical encoding and scale. Providers
/// may quantize X once and reuse those bytes for all branches; outputs preserve
/// each branch's BF16 rounding boundary and cannot alias any input or output.
/// Invocation scratch is core-admitted, with no binding/persistent workspace.
pub fn dense_linear_fp8_group_bf16_contract(
    group: DenseLinearFp8Group,
) -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = dense_linear_fp8_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(group.operation_id())?;
    descriptor.provider = provider_requirement(group.capability_id(), ContractVersion::new(1, 0))?;
    let weight = descriptor.inputs[1].clone();
    descriptor.inputs.truncate(1);
    descriptor.inputs.extend(vec![weight; group.count()]);
    descriptor.outputs = vec![descriptor.outputs[0].clone(); group.count()];
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn dense_linear_fp8_pair_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    dense_linear_fp8_group_bf16_contract(DenseLinearFp8Group::Pair)
}

pub fn dense_linear_fp8_triple_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    dense_linear_fp8_group_bf16_contract(DenseLinearFp8Group::Triple)
}

/// Checked logical geometry and work totals; physical storage remains a
/// provider/materializer responsibility for every independent weight port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearFp8GroupBf16Geometry {
    group: DenseLinearFp8Group,
    branch: DenseLinearFp8Bf16Geometry,
    output_elements: u64,
    arithmetic: u64,
}

impl DenseLinearFp8GroupBf16Geometry {
    pub fn new(
        group: DenseLinearFp8Group,
        input: &ResolvedTensorSpec,
        weights: &[&ResolvedTensorSpec],
        outputs: &[&ResolvedTensorSpec],
        in_features: u64,
        out_features: u64,
    ) -> Result<Self, VNextError> {
        if weights.len() != group.count() || outputs.len() != group.count() {
            return Err(invalid(
                "weight/output counts differ from the projection group",
            ));
        }
        let branch = DenseLinearFp8Bf16Geometry::new(
            input,
            weights[0],
            outputs[0],
            in_features,
            out_features,
        )?;
        for (weight, output) in weights.iter().zip(outputs).skip(1) {
            DenseLinearFp8Bf16Geometry::new(input, weight, output, in_features, out_features)?;
        }
        let count = group.count() as u64;
        let output_elements = branch
            .output_elements()
            .checked_mul(count)
            .ok_or_else(|| invalid("total output elements overflow"))?;
        let arithmetic = branch
            .logical_arithmetic_operations()
            .checked_mul(count)
            .ok_or_else(|| invalid("total logical arithmetic overflows"))?;
        Ok(Self {
            group,
            branch,
            output_elements,
            arithmetic,
        })
    }

    pub const fn group(self) -> DenseLinearFp8Group {
        self.group
    }

    pub const fn branch(self) -> DenseLinearFp8Bf16Geometry {
        self.branch
    }

    pub const fn rows(self) -> u64 {
        self.branch.rows()
    }

    pub const fn inputs(self) -> u64 {
        self.branch.inputs()
    }

    pub const fn outputs(self) -> u64 {
        self.branch.outputs()
    }

    pub const fn output_elements(self) -> u64 {
        self.output_elements
    }

    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("grouped FP8/BF16 dense geometry: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResolvedTensorLayout;

    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }

    #[test]
    fn groups_preserve_independent_bf16_ports_and_managed_scratch() {
        let scalar = dense_linear_fp8_bf16_contract().unwrap();
        for group in [DenseLinearFp8Group::Pair, DenseLinearFp8Group::Triple] {
            let contract = dense_linear_fp8_group_bf16_contract(group).unwrap();
            let descriptor = contract.descriptor();
            assert_eq!(descriptor.inputs.len(), group.count() + 1);
            assert_eq!(descriptor.outputs.len(), group.count());
            assert_eq!(descriptor.attributes, scalar.descriptor().attributes);
            assert_eq!(descriptor.inputs[0], scalar.descriptor().inputs[0]);
            for weight in &descriptor.inputs[1..] {
                assert_eq!(weight, &scalar.descriptor().inputs[1]);
            }
            for output in &descriptor.outputs {
                assert_eq!(output, &scalar.descriptor().outputs[0]);
                assert_eq!(output.alias(), &AliasPolicy::NoAlias);
            }
            assert_eq!(
                descriptor.resources.scratch,
                ResourcePresenceRequirement::Required
            );
            assert_eq!(
                descriptor.resources.binding,
                ResourcePresenceRequirement::Forbidden
            );
            assert_eq!(
                descriptor.resources.persistent,
                ResourcePresenceRequirement::Forbidden
            );
            assert_ne!(descriptor.id, scalar.descriptor().id);
        }
    }

    #[test]
    fn geometry_checks_every_branch_and_exact_group_cardinality() {
        let x = tensor(&[3, 16], ElementType::Bf16);
        let w = tensor(&[32, 16], ElementType::Bf16);
        let y = tensor(&[3, 32], ElementType::Bf16);
        for group in [DenseLinearFp8Group::Pair, DenseLinearFp8Group::Triple] {
            let weights = vec![&w; group.count()];
            let mut outputs = vec![&y; group.count()];
            let g = DenseLinearFp8GroupBf16Geometry::new(group, &x, &weights, &outputs, 16, 32)
                .unwrap();
            assert_eq!((g.rows(), g.inputs(), g.outputs()), (3, 16, 32));
            assert_eq!(g.output_elements(), 96 * group.count() as u64);
            assert_eq!(
                g.logical_arithmetic_operations(),
                3072 * group.count() as u64
            );
            assert!(
                DenseLinearFp8GroupBf16Geometry::new(group, &x, &weights, &outputs, 32, 16)
                    .is_err()
            );
            assert!(DenseLinearFp8GroupBf16Geometry::new(
                group,
                &x,
                &weights[..1],
                &outputs,
                16,
                32
            )
            .is_err());
            let wrong_dtype = tensor(&[3, 32], ElementType::F16);
            outputs[group.count() - 1] = &wrong_dtype;
            assert!(
                DenseLinearFp8GroupBf16Geometry::new(group, &x, &weights, &outputs, 16, 32)
                    .is_err()
            );
            let wrong_shape = tensor(&[2, 32], ElementType::Bf16);
            outputs[group.count() - 1] = &wrong_shape;
            assert!(
                DenseLinearFp8GroupBf16Geometry::new(group, &x, &weights, &outputs, 16, 32)
                    .is_err()
            );
        }
    }

    #[test]
    fn group_work_overflow_is_rejected_even_when_each_branch_fits() {
        let n = isize::MAX as u64 / 2;
        let x = tensor(&[1, 1], ElementType::Bf16);
        let w = tensor(&[n, 1], ElementType::Bf16);
        let y = tensor(&[1, n], ElementType::Bf16);
        assert!(DenseLinearFp8Bf16Geometry::new(&x, &w, &y, 1, n).is_ok());
        assert!(DenseLinearFp8GroupBf16Geometry::new(
            DenseLinearFp8Group::Triple,
            &x,
            &[&w; 3],
            &[&y; 3],
            1,
            n,
        )
        .is_err());
    }
}
