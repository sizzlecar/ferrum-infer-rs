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
/// Independent Q6_K F16 boundary; the F32 head remains a separate arithmetic.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16: u32 = 9;
/// Canonical codebook W4, regular H256 BF16/A8, full INT32 dot, BF16 store.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W4A8: u32 = 10;
/// Already-rotated signed INT8 weights with per-output [N,1] F32 scales.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W8A8: u32 = 11;
/// Gate|up BF16 -> F32 SwiGLU -> H256 -> BF16/A8, canonical W4 weights.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_SWIGLU_F32_W4A8: u32 = 12;
/// The same F32 SwiGLU activation with canonical rotated signed INT8 weights.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_SWIGLU_F32_W8A8: u32 = 13;
/// Full-F32 activation and output around canonical W4/A8/INT32 projection.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W4A8_F32: u32 = 14;
/// Full-F32 activation and output around canonical signed W8/A8 projection.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W8A8_F32: u32 = 15;
/// Selected I8 rows, F32 scale/H256, explicit source dtype boundary -> F32.
pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_INT8_EMBEDDING_F32: u32 = 16;

/// The layout's source compute dtype, independent of serialized I8 codes and
/// the F32 result consumed by the transformer. There is no implicit default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyEmbeddingDequantizedDtype {
    F32,
    Bf16ThenF32,
}
impl ComfyEmbeddingDequantizedDtype {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16ThenF32 => "bf16_then_f32",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "f32" => Some(Self::F32),
            "bf16_then_f32" => Some(Self::Bf16ThenF32),
            _ => None,
        }
    }
}

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
    /// Closed semantics of `operation.comfy_convrot_codebook_w4a8.bf16` v1:
    /// regular H4^4 with FP32 1/2-scaled stages; scale=max(min(amax,BF16_MAX)
    /// *RN32(1/127),1e-30); RNE(B(B(x)/B(scale))) clipped to [-128,127];
    /// canonical codebook*E4M3 scales decoded RNE to [-127,127]; complete
    /// nonoverflowing INT32 dot; RN32(acc*row_scale), RN32(*channel_scale),
    /// BF16 RNE. No bias, affine correction or stochastic rounding.
    ComfyConvRotW4A8Projection {},
    /// The same regular H256 BF16/A8 activation and epilogue boundaries as
    /// W4A8, but consumes canonical I8[N,K] directly, including -128, with
    /// F32 scales [N,1]. No codebook, decoding, bias or broadcasting.
    ComfyConvRotW8A8Projection {},
    /// Gate|up BF16[M,2K] -> RN32(gate / RN32(1+exp_f32(-gate))) ->
    /// RN32(*up), with no intermediate BF16 conversion. Then the closed W4
    /// H256/A8/INT32/epilogue boundaries above. Exponential implementation is
    /// provider-profiled; add/divide/multiply are RN32, no FMA and no FTZ.
    /// Independent of M; this does not select upstream fast-math implicitly.
    ComfyConvRotSwiGluF32W4A8Projection {},
    /// The same activation arithmetic and row-count independence with W8
    /// storage/scale semantics from ComfyConvRotW8A8Projection.
    ComfyConvRotSwiGluF32W8A8Projection {},
    /// F32 input -> regular H4^4 RN32 stages -> s=max(min(amax,F32_MAX)
    /// *RN32(1/127),1e-30) -> RNE(RN32(rotated/s)) clamp[-128,127].
    /// No BF16 conversions. W4 decode/integer dot match the canonical W4
    /// policy; epilogue RN32(acc*s), RN32(*channel) stores F32. No FMA/FTZ.
    /// Row count does not select different arithmetic; no bias/correction.
    ComfyConvRotW4A8F32Projection {},
    /// The same F32 activation/output policy, direct signed I8[N,K] weights
    /// and exact F32[N,1] channel scales, including the -128 weight code.
    ComfyConvRotW8A8F32Projection {},
    /// I8 selected rows -> RN32(code*scale) -> regular H4^4 F32 stages with
    /// 1/2 per stage -> explicit source dtype conversion -> F32. No FMA/FTZ,
    /// whole-table decode or A8 requantization. Invalid IDs produce zero rows.
    ComfyConvRotInt8EmbeddingF32 {
        dequantized_dtype: ComfyEmbeddingDequantizedDtype,
    },
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
    pub fn comfy_convrot_int8_embedding_f32(
        dequantized_dtype: ComfyEmbeddingDequantizedDtype,
    ) -> Self {
        Self {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_INT8_EMBEDDING_F32,
            stages: vec![NumericalArithmeticStage::ComfyConvRotInt8EmbeddingF32 {
                dequantized_dtype,
            }],
        }
    }
    pub fn comfy_convrot_w4a8_f32() -> Self {
        Self {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W4A8_F32,
            stages: vec![NumericalArithmeticStage::ComfyConvRotW4A8F32Projection {}],
        }
    }
    pub fn comfy_convrot_w8a8_f32() -> Self {
        Self {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W8A8_F32,
            stages: vec![NumericalArithmeticStage::ComfyConvRotW8A8F32Projection {}],
        }
    }
    pub fn comfy_convrot_swiglu_f32_w4a8_bf16() -> Self {
        Self {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_SWIGLU_F32_W4A8,
            stages: vec![NumericalArithmeticStage::ComfyConvRotSwiGluF32W4A8Projection {}],
        }
    }

    pub fn comfy_convrot_swiglu_f32_w8a8_bf16() -> Self {
        Self {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_SWIGLU_F32_W8A8,
            stages: vec![NumericalArithmeticStage::ComfyConvRotSwiGluF32W8A8Projection {}],
        }
    }

    pub fn comfy_convrot_w8a8_bf16() -> Self {
        Self {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W8A8,
            stages: vec![NumericalArithmeticStage::ComfyConvRotW8A8Projection {}],
        }
    }

    pub fn comfy_convrot_w4a8_bf16() -> Self {
        Self {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W4A8,
            stages: vec![NumericalArithmeticStage::ComfyConvRotW4A8Projection {}],
        }
    }

    pub(super) fn output_type(&self) -> Option<ElementType> {
        match self.stages.last() {
            Some(NumericalArithmeticStage::ComfyConvRotInt8EmbeddingF32 { .. }) => {
                Some(ElementType::F32)
            }
            Some(
                NumericalArithmeticStage::ComfyConvRotW4A8F32Projection {}
                | NumericalArithmeticStage::ComfyConvRotW8A8F32Projection {},
            ) => Some(ElementType::F32),
            Some(
                NumericalArithmeticStage::ComfyConvRotW4A8Projection {}
                | NumericalArithmeticStage::ComfyConvRotW8A8Projection {}
                | NumericalArithmeticStage::ComfyConvRotSwiGluF32W4A8Projection {}
                | NumericalArithmeticStage::ComfyConvRotSwiGluF32W8A8Projection {},
            ) => Some(ElementType::Bf16),
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
        if self.schema_version
            == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_INT8_EMBEDDING_F32
        {
            return match self.stages.as_slice(){[NumericalArithmeticStage::ComfyConvRotInt8EmbeddingF32{..}]=>Ok(()),_=>Err("INT8 ConvRot embedding schema requires exactly its closed source-dtype/gather policy".into())};
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W4A8_F32 {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::ComfyConvRotW4A8F32Projection {}] => Ok(()),
                _ => Err(
                    "F32 ConvRot W4A8 schema requires exactly its closed full-F32 policy".into(),
                ),
            };
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W8A8_F32 {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::ComfyConvRotW8A8F32Projection {}] => Ok(()),
                _ => Err(
                    "F32 ConvRot W8A8 schema requires exactly its closed full-F32 policy".into(),
                ),
            };
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_SWIGLU_F32_W4A8
        {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::ComfyConvRotSwiGluF32W4A8Projection {}] => Ok(()),
                _ => Err("F32 SwiGLU ConvRot W4A8 schema requires exactly its closed activation/projection policy".into()),
            };
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_SWIGLU_F32_W8A8
        {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::ComfyConvRotSwiGluF32W8A8Projection {}] => Ok(()),
                _ => Err("F32 SwiGLU ConvRot W8A8 schema requires exactly its closed activation/projection policy".into()),
            };
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W8A8 {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::ComfyConvRotW8A8Projection {}] => Ok(()),
                _ => Err("Comfy ConvRot W8A8 schema requires exactly its closed signed INT8 projection policy".into()),
            };
        }
        if self.schema_version == NUMERICAL_ARITHMETIC_SCHEMA_VERSION_COMFY_CONVROT_W4A8 {
            return match self.stages.as_slice() {
                [NumericalArithmeticStage::ComfyConvRotW4A8Projection {}] => Ok(()),
                _ => Err("Comfy ConvRot W4A8 schema requires exactly its closed BF16/A8 projection policy".into()),
            };
        }
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
                    if policy.format == super::ProjectionBlockFormat::Q6K
                        || policy.geometry_selection.is_some()
                    {
                        return Err("Q6 F16 geometry requires its independent staged schema".into());
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

#[cfg(test)]
mod comfy_tests {
    use super::*;
    #[test]
    fn comfy_convrot_schema_is_closed_and_cannot_replace_legacy_stages() {
        for policy in [
            StagedNumericalArithmetic::comfy_convrot_w4a8_bf16(),
            StagedNumericalArithmetic::comfy_convrot_w8a8_bf16(),
            StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w4a8_bf16(),
            StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w8a8_bf16(),
        ] {
            policy.validate().unwrap();
            assert_eq!(policy.output_type(), Some(ElementType::Bf16));
            assert_eq!(
                serde_json::from_slice::<StagedNumericalArithmetic>(
                    &serde_json::to_vec(&policy).unwrap()
                )
                .unwrap(),
                policy
            );
            let mut wrong = policy.clone();
            wrong.schema_version = NUMERICAL_ARITHMETIC_SCHEMA_VERSION;
            assert!(wrong.validate().is_err());
            wrong = policy.clone();
            wrong.stages.clear();
            assert!(wrong.validate().is_err());
            wrong = policy.clone();
            wrong.stages.extend(policy.stages.clone());
            assert!(wrong.validate().is_err());
            let mut unexpected = serde_json::to_value(&policy).unwrap();
            unexpected["stages"][0]["bias"] = serde_json::json!(true);
            assert!(serde_json::from_value::<StagedNumericalArithmetic>(unexpected).is_err());
        }
        let mut mixed = StagedNumericalArithmetic::comfy_convrot_w8a8_bf16();
        mixed.stages = StagedNumericalArithmetic::comfy_convrot_w4a8_bf16().stages;
        assert!(mixed.validate().is_err());
        let mut fused = StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w4a8_bf16();
        fused.stages = StagedNumericalArithmetic::comfy_convrot_w4a8_bf16().stages;
        assert!(fused.validate().is_err());
    }
    #[test]
    fn f32_quant_schema_is_distinct_closed_and_declares_f32_output() {
        for (policy, wrong) in [
            (
                StagedNumericalArithmetic::comfy_convrot_w4a8_f32(),
                StagedNumericalArithmetic::comfy_convrot_w4a8_bf16(),
            ),
            (
                StagedNumericalArithmetic::comfy_convrot_w8a8_f32(),
                StagedNumericalArithmetic::comfy_convrot_w8a8_bf16(),
            ),
        ] {
            policy.validate().unwrap();
            assert_eq!(policy.output_type(), Some(ElementType::F32));
            assert_eq!(
                serde_json::from_value::<StagedNumericalArithmetic>(
                    serde_json::to_value(&policy).unwrap()
                )
                .unwrap(),
                policy
            );
            let mut crossed = policy.clone();
            crossed.stages = wrong.stages;
            assert!(crossed.validate().is_err());
            let mut extra = serde_json::to_value(&policy).unwrap();
            extra["stages"][0]["round_to_bf16"] = serde_json::Value::Bool(true);
            assert!(serde_json::from_value::<StagedNumericalArithmetic>(extra).is_err());
        }
    }
    #[test]
    fn embedding_source_dtype_is_explicit_closed_and_preserves_f32_output() {
        for dtype in [
            ComfyEmbeddingDequantizedDtype::F32,
            ComfyEmbeddingDequantizedDtype::Bf16ThenF32,
        ] {
            assert_eq!(
                ComfyEmbeddingDequantizedDtype::parse(dtype.as_str()),
                Some(dtype)
            );
            let policy = StagedNumericalArithmetic::comfy_convrot_int8_embedding_f32(dtype);
            policy.validate().unwrap();
            assert_eq!(policy.output_type(), Some(ElementType::F32));
            let encoded = serde_json::to_value(&policy).unwrap();
            assert_eq!(
                serde_json::from_value::<StagedNumericalArithmetic>(encoded.clone()).unwrap(),
                policy
            );
            let mut unknown = encoded;
            unknown["stages"][0]["dequantized_dtype"] = serde_json::json!("f16");
            assert!(serde_json::from_value::<StagedNumericalArithmetic>(unknown).is_err());
            let mut wrong = policy;
            wrong.stages = StagedNumericalArithmetic::comfy_convrot_w8a8_f32().stages;
            assert!(wrong.validate().is_err());
        }
        assert!(ComfyEmbeddingDequantizedDtype::parse("bf16").is_none());
    }
}
