//! Versioned mixed arithmetic, separate from storage and provider selection.

use serde::{Deserialize, Serialize};

use super::{floating, ElementType};

/// The first staged wire schema describes a quantized projection pipeline.
/// Missing stages in an old operation remain the old floating-point contract;
/// they are never inferred from an operation/profile/model name. New schemas
/// require an explicit migration instead of ignoring unknown stages or fields.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION: u32 = 1;
/// Closed upstream projection policies. Schema 1 retains its original stages.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM: u32 = 2;
/// Exact G32 for local rows 1..32, MarkerV2 MMQ for rows 33..2048.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ: u32 = 3;
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA: u32 = 4;
/// Extra-format MMQ prefill; version 5 is reserved for the separate k8 hybrid.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL: u32 = 6;
/// Explicit physical-leaf geometry selection; legacy route schemas stay closed.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY: u32 = 8;
/// Independent Q6_K F16 input/output ABI, distinct from the F32 head schema.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16: u32 = 9;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StagedNumericalArithmetic {
    pub schema_version: u32,
    /// Execution order is semantic and is never sorted during fingerprinting.
    /// Schema 1 is quantize -> local integer dot -> rescale -> reduce -> store.
    pub stages: Vec<NumericalArithmeticStage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationScaleRule {
    /// Divide max(abs(x)) by `max_code` in F32, nearest ties-to-even. Form
    /// x/scale with the same F32 division before the declared integer rounding.
    AbsMaxOverMaxCode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegerQuantizationRounding {
    NearestTiesAwayFromZero,
    NearestTiesToEven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZeroQuantizationPolicy {
    PositiveZeroScaleAndZeroCodes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonFiniteQuantizationPolicy {
    /// Any nonfinite input marks the complete group; rescaling propagates NaN.
    NanScaleAndZeroCodes,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatingPointContraction {
    /// Multiplication and addition/subtraction have separate rounding points.
    Disallowed,
    /// A different numerical contract: a provider may contract eligible pairs.
    Permitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum AffineMinCorrection {
    // An empty struct variant deliberately rejects extra wire fields. Serde's
    // internally tagged unit-variant visitor would otherwise discard them.
    None {},
    /// Sum exactly the same quantized activation codes consumed by one local
    /// integer dot. Convert the sum to the rescale arithmetic type before the
    /// weight-minimum/activation-scale multiplication and subtraction.
    QuantizedPartialSum {
        accumulation_type: ElementType,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatingReductionOrder {
    /// The versioned operation specifies permitted serial/tree associations
    /// and its oracle. This is not a bitwise promise across batches/providers.
    OperationDefined,
    Sequential,
    Tree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatingStorageRounding {
    /// IEEE conversion preserves nonfinite values and overflows to infinity.
    NearestTiesToEven,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericalArithmeticStage {
    Q6MmqF32Projection {
        policy: super::Q6MmqF32Policy,
    },
    /// A complete, separately versioned pipeline, including its final F16
    /// conversion. It is never mixed with the schema-1 five-stage sequence.
    G32MmqPrefillProjection {
        policy: super::G32MmqPrefillPolicy,
    },
    UpstreamProjection {
        policy: super::UpstreamProjectionPolicy,
    },
    ActivationQuantization {
        input_type: ElementType,
        code_type: ElementType,
        scale_type: ElementType,
        group_values: u32,
        /// Symmetric signed clipping interval [-max_code, max_code].
        max_code: u8,
        scale_rule: ActivationScaleRule,
        rounding: IntegerQuantizationRounding,
        zero: ZeroQuantizationPolicy,
        non_finite: NonFiniteQuantizationPolicy,
    },
    IntegerDot {
        activation_type: ElementType,
        weight_type: ElementType,
        accumulation_type: ElementType,
        /// The actual I32 accumulation extent before conversion/rescaling.
        /// DP4A partials use 4 even when activation scales cover 32 values.
        values_per_partial: u32,
    },
    Rescale {
        integer_input_type: ElementType,
        activation_scale_type: ElementType,
        /// Both the weight scale and optional affine minimum coefficient.
        weight_coefficient_type: ElementType,
        arithmetic_type: ElementType,
        min_correction: AffineMinCorrection,
        contraction: FloatingPointContraction,
    },
    FloatingReduction {
        input_type: ElementType,
        accumulation_type: ElementType,
        order: FloatingReductionOrder,
    },
    OutputRounding {
        input_type: ElementType,
        output_type: ElementType,
        rounding: FloatingStorageRounding,
    },
}

impl StagedNumericalArithmetic {
    pub(super) fn output_type(&self) -> Option<ElementType> {
        match self.stages.last() {
            Some(NumericalArithmeticStage::Q6MmqF32Projection { .. }) => Some(ElementType::F32),
            Some(
                NumericalArithmeticStage::UpstreamProjection { .. }
                | NumericalArithmeticStage::G32MmqPrefillProjection { .. },
            ) => Some(ElementType::F16),
            Some(NumericalArithmeticStage::OutputRounding { output_type, .. }) => {
                Some(*output_type)
            }
            _ => None,
        }
    }

    /// Structural numerical validation is independent of backend availability.
    /// Family registration still has to reproduce this declaration; operation
    /// and provider versions must implement it before any plan can execute it.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16 {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::UpstreamProjection { policy }]
                    if policy.format == super::ProjectionBlockFormat::Q6K =>
                {
                    policy.validate()
                }
                _ => Err("Q6 F16 schema requires its independent upstream policy".into()),
            };
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::UpstreamProjection { policy }]
                    if matches!(
                        policy.geometry_selection,
                        Some(super::UpstreamProjectionGeometrySelection::M8Q4Q5Mmq { .. })
                    ) =>
                {
                    policy.validate()
                }
                _ => Err(
                    "geometry schema requires one explicit geometry-selected upstream policy"
                        .into(),
                ),
            };
        }
        if self.schema_version == super::NUMERICAL_ARITHMETIC_SCHEMA_VERSION_Q6_MMQ_F32 {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::Q6MmqF32Projection { policy }] => policy.validate(),
                _ => Err("Q6 F32 schema requires exactly one independent Q6 D4/MMQ policy".into()),
            };
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::G32MmqPrefillProjection { policy }] => policy.validate(),
                _ => Err("hybrid schema requires exactly one closed G32/MMQ policy".into()),
            };
        }
        if matches!(
            self.schema_version,
            NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM
                | NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA
                | NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
        ) {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::UpstreamProjection { policy }] => {
                    if policy.format == super::ProjectionBlockFormat::Q6K {
                        return Err("Q6 F16 requires its independent staged schema".into());
                    }
                    if policy.geometry_selection.is_some() {
                        return Err("legacy upstream schema cannot carry geometry selection".into());
                    }
                    if policy.format.is_extra()
                        != matches!(
                            self.schema_version,
                            NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA
                                | NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
                        )
                    {
                        return Err("upstream format belongs to a different staged schema".into());
                    }
                    let extra_prefill = policy.format.is_extra()
                        && policy
                            .routes
                            .iter()
                            .any(|route| route.prefill_rows.is_some());
                    if extra_prefill
                        != (self.schema_version
                            == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL)
                    {
                        return Err("extra prefill requires its separate staged schema".into());
                    }
                    policy.validate()
                }
                _ => Err("upstream schema requires exactly one closed projection policy".into()),
            };
        }
        if self.schema_version != NUMERICAL_ARITHMETIC_SCHEMA_VERSION {
            return Err(
                "unsupported staged arithmetic schema version; explicit migration required".into(),
            );
        }
        let [NumericalArithmeticStage::ActivationQuantization {
            input_type,
            code_type,
            scale_type,
            group_values,
            max_code,
            ..
        }, NumericalArithmeticStage::IntegerDot {
            activation_type,
            weight_type,
            accumulation_type: integer_accumulation,
            values_per_partial,
        }, NumericalArithmeticStage::Rescale {
            integer_input_type,
            activation_scale_type,
            weight_coefficient_type,
            arithmetic_type,
            min_correction,
            ..
        }, NumericalArithmeticStage::FloatingReduction {
            input_type: reduction_input,
            accumulation_type: reduction_accumulation,
            ..
        }, NumericalArithmeticStage::OutputRounding {
            input_type: store_input,
            output_type,
            ..
        }] = self.stages.as_slice()
        else {
            return Err("staged projection must declare quantization, local integer dot, rescale, floating reduction and output rounding in execution order".into());
        };
        // Schema 1 follows the implemented F16 input domain. Its smallest
        // nonzero magnitude divided by max_code remains a normal F32 scale.
        // F32/BF16 require a separately versioned finite-scale-underflow rule.
        if *input_type != ElementType::F16
            || *code_type != ElementType::I8
            || *scale_type != ElementType::F32
            || *group_values == 0
            || !(1..=127).contains(max_code)
        {
            return Err("schema 1 activation quantization requires F16 input, symmetric I8 codes, F32 scales and nonempty groups".into());
        }
        if activation_type != code_type
            || *weight_type != ElementType::I8
            || *integer_accumulation != ElementType::I32
            || *values_per_partial == 0
            || *values_per_partial > *group_values
            || *group_values % *values_per_partial != 0
        {
            return Err("integer dot must consume the I8 codes and accumulate I32 partials that divide the activation scale group".into());
        }
        // Signed weight codes may include -128. This bound also covers the
        // affine code sum. Widen before multiplication over the entire u32 ABI.
        let maximum_partial = u64::from(*values_per_partial) * u64::from(*max_code) * 128;
        if maximum_partial > i32::MAX as u64 {
            return Err("declared local integer dot can overflow its I32 accumulator".into());
        }
        if integer_input_type != integer_accumulation
            || activation_scale_type != scale_type
            || *weight_coefficient_type != ElementType::F32
            || *arithmetic_type != ElementType::F32
            || matches!(min_correction, AffineMinCorrection::QuantizedPartialSum { accumulation_type } if *accumulation_type != ElementType::I32)
        {
            return Err("rescale/min correction must convert I32 partials using the declared F32 scales, coefficients and arithmetic".into());
        }
        if reduction_input != arithmetic_type
            || *reduction_accumulation != ElementType::F32
            || store_input != reduction_accumulation
            || !floating(*output_type)
        {
            return Err("F32 rescaled partials must reduce in F32 before an explicit floating output boundary".into());
        }
        Ok(())
    }
}
