//! Row-wise weighted RMS normalization with one final BF16 boundary.
use super::*;
use crate::ResolvedTensorSpec;
pub const ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID: &str =
    "operation.row_rms_norm_single_round.bf16";
pub const ROW_RMS_NORM_SINGLE_ROUND_BF16_CAPABILITY_ID: &str =
    "capability.operation.row_rms_norm_single_round.bf16";
/// X/Y are BF16 [rows,width], gamma BF16 [width]. The full row reduces in
/// F32; r=rsqrt(mean(X^2)+epsilon), Y=B(RN32(RN32(X*r)*gamma)). There is no
/// BF16 normalized intermediate before gamma, no head partition or rotary
/// input, and no power-of-two width requirement. Providers declare reduction,
/// sqrt/reciprocal and underflow profiles; no Torch binary parity is implied.
pub fn row_rms_norm_single_round_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let matrix = vec![symbol("rows"), symbol("width")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(matrix.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("width")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            matrix,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([positive_epsilon_attribute("epsilon")?]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            ROW_RMS_NORM_SINGLE_ROUND_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowRmsNormSingleRoundBf16Geometry {
    norm: HeadRmsNormBf16Geometry,
}
impl RowRmsNormSingleRoundBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let [_, width] = input.dimensions() else {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "row RMS normalization requires rank-two input".into(),
            });
        };
        let result = Self {
            norm: HeadRmsNormBf16Geometry::new(input, gamma, output, *width, epsilon)?,
        };
        result.logical_arithmetic_operations()?;
        Ok(result)
    }
    pub const fn rows(self) -> u64 {
        self.norm.rows()
    }
    pub const fn width(self) -> u64 {
        self.norm.columns()
    }
    pub const fn epsilon(self) -> CanonicalRational {
        self.norm.epsilon()
    }
    pub fn effective_epsilon(self) -> f32 {
        self.norm.effective_epsilon()
    }
    pub fn logical_arithmetic_operations(self) -> Result<u64, VNextError> {
        self.width()
            .checked_mul(4)
            .and_then(|v| v.checked_add(3))
            .and_then(|v| v.checked_mul(self.rows()))
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: "row RMS arithmetic count overflows".into(),
            })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResolvedTensorLayout;
    fn spec(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn accepts_5376_and_odd_width_without_a_rotary_or_head_attribute() {
        for width in [7, 128, 5376] {
            let x = spec(&[2, width], ElementType::Bf16);
            let g = spec(&[width], ElementType::Bf16);
            let a = RowRmsNormSingleRoundBf16Geometry::new(
                &x,
                &g,
                &x,
                CanonicalRational::new(1, 100000).unwrap(),
            )
            .unwrap();
            assert_eq!(
                a.logical_arithmetic_operations().unwrap(),
                2 * (4 * width + 3)
            );
        }
        assert_eq!(
            row_rms_norm_single_round_bf16_contract()
                .unwrap()
                .descriptor
                .inputs
                .len(),
            2
        );
    }
    #[test]
    fn rejects_wrong_gamma_precision_shape_and_nonpositive_epsilon() {
        let x = spec(&[2, 7], ElementType::Bf16);
        for g in [
            spec(&[7], ElementType::F32),
            spec(&[1, 7], ElementType::Bf16),
            spec(&[8], ElementType::Bf16),
        ] {
            assert!(RowRmsNormSingleRoundBf16Geometry::new(
                &x,
                &g,
                &x,
                CanonicalRational::new(1, 100000).unwrap()
            )
            .is_err());
        }
        assert!(RowRmsNormSingleRoundBf16Geometry::new(
            &x,
            &spec(&[7], ElementType::Bf16),
            &x,
            CanonicalRational::new(0, 1).unwrap()
        )
        .is_err());
    }
}
