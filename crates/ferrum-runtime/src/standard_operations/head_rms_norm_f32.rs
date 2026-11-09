//! Full-F32 head-local RMS normalization with an explicit F32 gamma.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const HEAD_RMS_NORM_F32_OPERATION_ID: &str = "operation.head_rms_norm.f32";
pub const HEAD_RMS_NORM_F32_CAPABILITY_ID: &str = "capability.operation.head_rms_norm.f32";
pub const HEAD_RMS_NORM_F32_MAX_HEAD_DIM: u64 = 1 << 24;
/// F32 X/Y[rows,columns] and shared F32 gamma[head_dim]. For each independent
/// head, mean=sum(x*x)/head_dim; inverse=1/sqrt(mean+epsilon); output=(x*inverse)*gamma.
/// Every boundary remains F32, with no intermediate BF16/F16 or gamma downcast.
/// Epsilon is the canonical rational divided in F64 then converted to F32.
/// Providers declare their F32 reduction and inverse-root implementation;
/// this contract does not assert Torch fused-RMS or rsqrt binary equivalence.
/// There is no centering, bias, gamma offset, rotary operation or hidden view.
pub fn head_rms_norm_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = head_rms_norm_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(HEAD_RMS_NORM_F32_OPERATION_ID)?;
    descriptor.provider =
        provider_requirement(HEAD_RMS_NORM_F32_CAPABILITY_ID, ContractVersion::new(1, 0))?;
    let matrix = vec![symbol("rows"), symbol("columns")];
    descriptor.inputs = vec![
        contiguous_tensor(matrix.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(
            vec![symbol("head_dim")],
            [ElementType::F32],
            TensorAccess::Read,
        )?,
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
pub struct HeadRmsNormF32Geometry {
    rows: u64,
    columns: u64,
    head_dim: u64,
    epsilon: CanonicalRational,
    elements: u64,
    arithmetic: u64,
}
impl HeadRmsNormF32Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        head_dim: u64,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let [rows, columns] = input.dimensions() else {
            return Err(invalid("input must have rank two"));
        };
        if input != output
            || input.element_type() != ElementType::F32
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || *columns == 0
            || !(1..=HEAD_RMS_NORM_F32_MAX_HEAD_DIM).contains(&head_dim)
            || !columns.is_multiple_of(head_dim)
            || gamma.dimensions() != [head_dim]
            || gamma.element_type() != ElementType::F32
            || gamma.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid("requires contiguous F32 x/y[rows,columns], gamma[head_dim], positive head_dim<=2^24 dividing columns"));
        }
        let numerator = i128::from(epsilon.numerator());
        let denominator = i128::from(epsilon.denominator());
        if numerator * 1_000_000_000_000 < denominator || numerator > denominator {
            return Err(invalid("epsilon must be in [1e-12,1]"));
        }
        for t in [input, gamma, output] {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        let arithmetic = elements
            .checked_div(head_dim)
            .and_then(|heads| head_dim.checked_mul(4)?.checked_add(3)?.checked_mul(heads))
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            head_dim,
            epsilon,
            elements,
            arithmetic,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn head_dim(self) -> u64 {
        self.head_dim
    }
    pub const fn heads(self) -> u64 {
        self.columns / self.head_dim
    }
    pub const fn epsilon(self) -> CanonicalRational {
        self.epsilon
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
    pub fn effective_epsilon(self) -> f32 {
        (self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64) as f32
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 head RMSNorm: {reason}"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn head_norm_requires_f32_gamma_and_supports_non_power_of_two_hidden_width() {
        let x = t(&[2, 5120], ElementType::F32);
        let gamma = t(&[2560], ElementType::F32);
        let eps = CanonicalRational::new(1, 4).unwrap();
        let g = HeadRmsNormF32Geometry::new(&x, &gamma, &x, 2560, eps).unwrap();
        assert_eq!(g.elements(), 10240);
        assert!(
            HeadRmsNormF32Geometry::new(&x, &t(&[2560], ElementType::Bf16), &x, 2560, eps).is_err()
        );
        assert!(HeadRmsNormF32Geometry::new(&x, &gamma, &x, 128, eps).is_err());
        assert!(HeadRmsNormF32Geometry::new(
            &x,
            &gamma,
            &x,
            2560,
            CanonicalRational::new(0, 1).unwrap()
        )
        .is_err());
        assert_eq!(
            head_rms_norm_f32_contract()
                .unwrap()
                .descriptor()
                .resources
                .scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
