//! Explicit projection arithmetic. All other SwiGLU work remains strict.
use super::*;
use crate::vnext::{
    ActivationScaleRule, AffineMinCorrection, CompositeNumericalArithmetic,
    FloatingPointContraction, FloatingReductionOrder, FloatingStorageRounding,
    IntegerQuantizationRounding, NonFiniteQuantizationPolicy, NumericalArithmeticStage,
    ProjectionArithmeticOverride, ProjectionBlockFormat, ProjectionRole,
    ProjectionShapeEligibility, QuantizedProjectionLeafContract, StagedNumericalArithmetic,
    StrictNumericalOperation, StrictProjectionFallback, ZeroQuantizationPolicy,
    COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION, NUMERICAL_ARITHMETIC_SCHEMA_VERSION,
};

pub const DENSE_SWIGLU_IQ4XS_Q8ACT_G32_OPERATION_ID: &str =
    "operation.dense_swiglu.iq4xs-q8act-g32";
pub const DENSE_SWIGLU_IQ4XS_Q8ACT_G32_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.iq4xs-q8act-g32";
pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_Q8ACT_G32_OPERATION_ID: &str =
    "operation.dense_swiglu.q4k-q5k-iq4xs-q8act-g32";
pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_Q8ACT_G32_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.q4k-q5k-iq4xs-q8act-g32";

/// Independent numerical identities, not a mutable set of supported kernels.
/// Adding a provider export must not broaden an existing profile's leaf policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8ActSwiGluProfile {
    Iq4Xs,
    Q4KQ5KIq4Xs,
}

impl Q8ActSwiGluProfile {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Iq4Xs => DENSE_SWIGLU_IQ4XS_Q8ACT_G32_OPERATION_ID,
            Self::Q4KQ5KIq4Xs => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_Q8ACT_G32_OPERATION_ID,
        }
    }

    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Iq4Xs => DENSE_SWIGLU_IQ4XS_Q8ACT_G32_CAPABILITY_ID,
            Self::Q4KQ5KIq4Xs => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_Q8ACT_G32_CAPABILITY_ID,
        }
    }

    pub fn arithmetic(self) -> CompositeNumericalArithmetic {
        match self {
            Self::Iq4Xs => dense_swiglu_iq4xs_q8act_g32_arithmetic(),
            Self::Q4KQ5KIq4Xs => dense_swiglu_q4k_q5k_iq4xs_q8act_g32_arithmetic(),
        }
    }

    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let mut contract = dense_swiglu_contract()?;
        contract.descriptor.id = OperationId::new(self.operation_id())?;
        contract.descriptor.provider =
            provider_requirement(self.capability_id(), ContractVersion::new(1, 0))?;
        contract.descriptor.validate()?;
        Ok(contract)
    }
}

/// G32 is a local I32 dot over 32 values, followed by noncontracted F32
/// rescaling, lane accumulation and warp-tree reduction. It is not dot4 or
/// MMA. No batch-dependent change to this arithmetic is permitted.
pub fn dense_swiglu_iq4xs_q8act_g32_arithmetic() -> CompositeNumericalArithmetic {
    let arithmetic = StagedNumericalArithmetic {
        schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION,
        stages: vec![
            NumericalArithmeticStage::ActivationQuantization {
                input_type: ElementType::F16,
                code_type: ElementType::I8,
                scale_type: ElementType::F32,
                group_values: 32,
                max_code: 127,
                scale_rule: ActivationScaleRule::AbsMaxOverMaxCode,
                rounding: IntegerQuantizationRounding::NearestTiesAwayFromZero,
                zero: ZeroQuantizationPolicy::PositiveZeroScaleAndZeroCodes,
                non_finite: NonFiniteQuantizationPolicy::NanScaleAndZeroCodes,
            },
            NumericalArithmeticStage::IntegerDot {
                activation_type: ElementType::I8,
                weight_type: ElementType::I8,
                accumulation_type: ElementType::I32,
                values_per_partial: 32,
            },
            NumericalArithmeticStage::Rescale {
                integer_input_type: ElementType::I32,
                activation_scale_type: ElementType::F32,
                weight_coefficient_type: ElementType::F32,
                arithmetic_type: ElementType::F32,
                min_correction: AffineMinCorrection::None {},
                contraction: FloatingPointContraction::Disallowed,
            },
            NumericalArithmeticStage::FloatingReduction {
                input_type: ElementType::F32,
                accumulation_type: ElementType::F32,
                order: FloatingReductionOrder::OperationDefined,
            },
            NumericalArithmeticStage::OutputRounding {
                input_type: ElementType::F32,
                output_type: ElementType::F16,
                rounding: FloatingStorageRounding::NearestTiesToEven,
            },
        ],
    };
    CompositeNumericalArithmetic {
        schema_version: COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION,
        strict_base: StrictNumericalOperation {
            operation_id: OperationId::new(DENSE_SWIGLU_OPERATION_ID).expect("fixed operation ID"),
            version: ContractVersion::new(1, 0),
        },
        projections: [
            (ProjectionRole::SwiGluGateUp, 1),
            (ProjectionRole::SwiGluDown, 2),
        ]
        .into_iter()
        .map(
            |(role, weight_input_ordinal)| ProjectionArithmeticOverride {
                role,
                weight_input_ordinal,
                activation_input_type: ElementType::F16,
                activation_output_type: ElementType::F16,
                leaves: vec![QuantizedProjectionLeafContract {
                    format: ProjectionBlockFormat::Iq4Xs,
                    shape: ProjectionShapeEligibility {
                        minimum_input_features: 256,
                        minimum_output_features: 1,
                        input_features_multiple: 256,
                    },
                    arithmetic: arithmetic.clone(),
                }],
                fallback: StrictProjectionFallback::RetainBaseArithmetic {},
            },
        )
        .collect(),
    }
}

pub fn dense_swiglu_iq4xs_q8act_g32_contract() -> Result<StandardOperationContract, VNextError> {
    Q8ActSwiGluProfile::Iq4Xs.contract()
}

/// The affine formats subtract their minimum using the same K32 activation
/// codes as the integer dot. The old IQ4-only declaration remains unchanged.
pub fn dense_swiglu_q4k_q5k_iq4xs_q8act_g32_arithmetic() -> CompositeNumericalArithmetic {
    let mut composite = dense_swiglu_iq4xs_q8act_g32_arithmetic();
    for projection in &mut composite.projections {
        let iq4 = projection.leaves[0].clone();
        projection.leaves = [ProjectionBlockFormat::Q4K, ProjectionBlockFormat::Q5K]
            .into_iter()
            .map(|format| {
                let mut leaf = iq4.clone();
                leaf.format = format;
                let NumericalArithmeticStage::Rescale { min_correction, .. } =
                    &mut leaf.arithmetic.stages[2]
                else {
                    unreachable!("fixed G32 stage declaration")
                };
                *min_correction = AffineMinCorrection::QuantizedPartialSum {
                    accumulation_type: ElementType::I32,
                };
                leaf
            })
            .chain(std::iter::once(iq4.clone()))
            .collect();
    }
    composite
}

pub fn dense_swiglu_q4k_q5k_iq4xs_q8act_g32_contract(
) -> Result<StandardOperationContract, VNextError> {
    Q8ActSwiGluProfile::Q4KQ5KIq4Xs.contract()
}
