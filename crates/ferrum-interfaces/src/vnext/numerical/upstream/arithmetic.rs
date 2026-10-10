use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::super::{
    NumericalArithmeticStage, ProjectionBlockFormat, StagedNumericalArithmetic,
    StrictProjectionFallback, NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM,
    NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA,
    NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL,
    NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY,
    NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16,
};
use super::UpstreamProjectionGeometrySelection;

/// Numerical versions follow the pinned upstream projection equations,
/// not the adapter's function names or a performance threshold. Changing any
/// pack, coefficient, min correction or reduction rule requires a new variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamProjectionArithmetic {
    /// IQ4_XS: reciprocal F32 pack, F32 scales, F32 weight coefficients.
    MmqD4V1,
    /// Q4_K/Q5_K: reciprocal pack; half scale/original sum and half-rounded
    /// group weight scale/minimum coefficients. Min consumes original sum.
    MmqDs4V1,
    /// F32 amax/127 and x/d pack; half scale and stored original sum. Dot does
    /// NOT consume original sum: affine correction uses actual d*sum(q), with
    /// integer group coefficients inside the integer partial before rescale.
    MmvqQ8_1V1,
    /// Device marker rules are a new numerical version, not aliases of V1.
    MmqD4MarkerV2,
    MmqDs4MarkerV2,
    MmvqQ8_1MarkerV2,
    /// Q3_K/IQ3_S/IQ4_NL: D4 F32 activation scales and F32 consumed weight
    /// coefficients; no affine minimum or original-sum correction.
    MmqD4ExtraMarkerV2,
    /// Separate no-min Q8_1 dot ABI. The stored half original sum is unused.
    MmvqQ8_1ExtraMarkerV2,
    /// Independent Q6_K F16 boundary ABI: D4 pack, Q6 signed coefficients,
    /// F32 MMQ/fixup and F16 RN storage with the canonical marker protocol.
    MmqD4Q6F16MarkerV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamProjectionLayout {
    /// M columns of one physical weight matrix.
    Columns,
    /// M channels, each with one activation column and the same retained matrix.
    /// This is a separate dispatch/layout identity, not inferred from client C.
    Channels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamActivationPack {
    MmqTransposedD4_144BytesPer128,
    MmqTransposedDs4_144BytesPer128,
    MmvqRowMajorQ8_1_36BytesPer32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamScaleExpression {
    Reciprocal127OverAbsMaxThenReciprocal,
    AbsMaxOver127ThenInputDivision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamCoefficientRule {
    HalfBaseTimesIntegerInF32,
    HalfRoundedGroupScaleAndNegativeMinimum,
    IntegerGroupCoefficientBeforeF32Rescale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamMinSource {
    None,
    HalfRoundedOriginalActivationSum,
    ActualQuantizedI8SumTimesStoredScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamReduction {
    /// Local exact I32 products, upstream MMA/stream-K F32 accumulation/fixup.
    MmqMmaStreamKF32,
    /// Exact local integer dot/coefficient, upstream warp/block F32 reduction.
    MmvqWarpBlockF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamDynamicDomain {
    /// Finite F16 inputs, nonzero amax in each *logical* K32 group, finite
    /// consumed coefficients/metadata. Padding groups are upstream-generated.
    /// Zero/nonfinite groups are not assigned the CPU oracle's reporting marker.
    MmqFiniteNonzeroLogicalGroups,
    /// Finite F16 inputs and finite consumed coefficients. Zero has the actual
    /// upstream q=0 branch. Unused half original-sum overflow is permitted.
    MmvqFiniteInputs,
    /// Accept every F16 input bit pattern with device poison propagation.
    /// This is not rejection-before-dot and requires the V2 pack/cast kernels.
    CanonicalNanMarkerV2,
}

/// Normative V2 behavior, derived from the version instead of editable knobs.
/// A row flag applies to this packed input row; a retained weight flag applies
/// to every output of this physical leaf, not to adjacent packed components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UpstreamMarkerSemantics {
    pub canonical_f16_nan_bits: u16,
    pub canonical_f32_nan_bits: u32,
    pub zero_logical_and_padding_groups_use_positive_zero_metadata_and_codes: bool,
    pub nonfinite_input_or_consumed_metadata_poison_entire_input_row: bool,
    pub nonfinite_consumed_weight_coefficient_poisons_entire_physical_leaf: bool,
    pub nonfinite_f32_result_or_f16_overflow_poisons_output_element: bool,
    pub unused_original_sum_overflow_is_ignored: bool,
    pub half_scale_underflow_to_zero_is_allowed: bool,
}

/// Expanded evidence derived from the immutable version, not editable wire
/// knobs. All routes accept F16 and store F32 dot output before F16 RN cast.
/// Fast math permits contraction, FTZ and approximate reciprocal/division;
/// this is neither G32's nonfused arithmetic nor a cross-route bitwise promise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpstreamArithmeticSemantics {
    pub pack: UpstreamActivationPack,
    pub scale_expression: UpstreamScaleExpression,
    pub activation_scale_type: crate::vnext::ElementType,
    pub coefficient_rule: UpstreamCoefficientRule,
    pub min_source: UpstreamMinSource,
    pub reduction: UpstreamReduction,
    pub dynamic_domain: UpstreamDynamicDomain,
    pub fast_math_contraction_ftz_approximate_division: bool,
    pub half_storage_preserves_underflow_to_zero: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marker: Option<UpstreamMarkerSemantics>,
}

impl UpstreamProjectionArithmetic {
    pub const fn is_extra(self) -> bool {
        matches!(self, Self::MmqD4ExtraMarkerV2 | Self::MmvqQ8_1ExtraMarkerV2)
    }
    pub const fn is_marker_v2(self) -> bool {
        matches!(
            self,
            Self::MmqD4MarkerV2
                | Self::MmqDs4MarkerV2
                | Self::MmvqQ8_1MarkerV2
                | Self::MmqD4ExtraMarkerV2
                | Self::MmvqQ8_1ExtraMarkerV2
                | Self::MmqD4Q6F16MarkerV1
        )
    }

    pub(super) const fn finite_expression(self) -> Self {
        match self {
            Self::MmqD4MarkerV2 | Self::MmqD4ExtraMarkerV2 | Self::MmqD4Q6F16MarkerV1 => {
                Self::MmqD4V1
            }
            Self::MmqDs4MarkerV2 => Self::MmqDs4V1,
            Self::MmvqQ8_1MarkerV2 | Self::MmvqQ8_1ExtraMarkerV2 => Self::MmvqQ8_1V1,
            other => other,
        }
    }
    pub fn semantics(
        self,
        format: ProjectionBlockFormat,
    ) -> Result<UpstreamArithmeticSemantics, String> {
        use crate::vnext::ElementType;
        use ProjectionBlockFormat::{Iq3S, Iq4Nl, Iq4Xs, Q3K, Q4K, Q5K, Q6K};
        if self.is_extra() != format.is_extra()
            || (self == Self::MmqD4Q6F16MarkerV1) != (format == Q6K)
        {
            return Err(
                "upstream arithmetic and weight format belong to different native families".into(),
            );
        }
        let (
            pack,
            scale_expression,
            activation_scale_type,
            coefficient_rule,
            min_source,
            reduction,
            dynamic_domain,
        ) = match (self.finite_expression(), format) {
            (Self::MmqD4V1, Iq4Xs | Q3K | Iq3S | Iq4Nl | Q6K) => (
                UpstreamActivationPack::MmqTransposedD4_144BytesPer128,
                UpstreamScaleExpression::Reciprocal127OverAbsMaxThenReciprocal,
                ElementType::F32,
                UpstreamCoefficientRule::HalfBaseTimesIntegerInF32,
                UpstreamMinSource::None,
                UpstreamReduction::MmqMmaStreamKF32,
                UpstreamDynamicDomain::MmqFiniteNonzeroLogicalGroups,
            ),
            (Self::MmqDs4V1, Q4K | Q5K) => (
                UpstreamActivationPack::MmqTransposedDs4_144BytesPer128,
                UpstreamScaleExpression::Reciprocal127OverAbsMaxThenReciprocal,
                ElementType::F16,
                UpstreamCoefficientRule::HalfRoundedGroupScaleAndNegativeMinimum,
                UpstreamMinSource::HalfRoundedOriginalActivationSum,
                UpstreamReduction::MmqMmaStreamKF32,
                UpstreamDynamicDomain::MmqFiniteNonzeroLogicalGroups,
            ),
            (Self::MmvqQ8_1V1, Iq4Xs | Q4K | Q5K | Q3K | Iq3S | Iq4Nl) => (
                UpstreamActivationPack::MmvqRowMajorQ8_1_36BytesPer32,
                UpstreamScaleExpression::AbsMaxOver127ThenInputDivision,
                ElementType::F16,
                UpstreamCoefficientRule::IntegerGroupCoefficientBeforeF32Rescale,
                if matches!(format, Q4K | Q5K) {
                    UpstreamMinSource::ActualQuantizedI8SumTimesStoredScale
                } else {
                    UpstreamMinSource::None
                },
                UpstreamReduction::MmvqWarpBlockF32,
                UpstreamDynamicDomain::MmvqFiniteInputs,
            ),
            _ => {
                return Err(
                    "upstream arithmetic does not implement the declared weight format".into(),
                )
            }
        };
        Ok(UpstreamArithmeticSemantics {
            pack,
            scale_expression,
            activation_scale_type,
            coefficient_rule,
            min_source,
            reduction,
            dynamic_domain: if self.is_marker_v2() {
                UpstreamDynamicDomain::CanonicalNanMarkerV2
            } else {
                dynamic_domain
            },
            fast_math_contraction_ftz_approximate_division: true,
            half_storage_preserves_underflow_to_zero: true,
            marker: self.is_marker_v2().then_some(UpstreamMarkerSemantics {
                canonical_f16_nan_bits: 0x7e00,
                canonical_f32_nan_bits: 0x7fc0_0000,
                zero_logical_and_padding_groups_use_positive_zero_metadata_and_codes: true,
                nonfinite_input_or_consumed_metadata_poison_entire_input_row: true,
                nonfinite_consumed_weight_coefficient_poisons_entire_physical_leaf: true,
                nonfinite_f32_result_or_f16_overflow_poisons_output_element: true,
                unused_original_sum_overflow_is_ignored: true,
                half_scale_underflow_to_zero_is_allowed: true,
            }),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamProjectionRouteDeclaration {
    pub arithmetic: UpstreamProjectionArithmetic,
    pub layout: UpstreamProjectionLayout,
    /// Actual local activation rows. No implicit interval or offered-concurrency
    /// mapping; intermediate widths require an explicit, qualified declaration.
    #[serde(deserialize_with = "unique_rows")]
    pub local_rows: BTreeSet<u32>,
    /// An explicitly qualified prefill interval. Omitted on legacy declarations
    /// so their wire representation and numerical fingerprints stay unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_rows: Option<UpstreamPrefillRows>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamPrefillRows {
    pub first: u32,
    pub last: u32,
}

impl UpstreamProjectionRouteDeclaration {
    pub fn contains_rows(&self, rows: u32) -> bool {
        self.local_rows.contains(&rows)
            || self
                .prefill_rows
                .is_some_and(|r| (r.first..=r.last).contains(&rows))
    }
}

fn unique_rows<'de, D: serde::Deserializer<'de>>(d: D) -> Result<BTreeSet<u32>, D::Error> {
    let rows = Vec::<u32>::deserialize(d)?;
    let unique: BTreeSet<_> = rows.iter().copied().collect();
    if unique.len() != rows.len() {
        return Err(serde::de::Error::custom("duplicate local row declaration"));
    }
    Ok(unique)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamProjectionPolicy {
    pub format: ProjectionBlockFormat,
    pub routes: Vec<UpstreamProjectionRouteDeclaration>,
    pub fallback: StrictProjectionFallback,
    /// Omitted on every legacy declaration to preserve its exact wire identity.
    /// A present selection requires the separate geometry arithmetic schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_selection: Option<UpstreamProjectionGeometrySelection>,
}

impl UpstreamProjectionPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if self.routes.is_empty() {
            return Err("upstream policy needs an explicit route".into());
        }
        let mut domains = BTreeSet::new();
        for route in &self.routes {
            route.arithmetic.semantics(self.format)?;
            if route.local_rows.is_empty() && route.prefill_rows.is_none() {
                return Err("route needs explicit local rows".into());
            }
            if let Some(range) = route.prefill_rows {
                if range.first < 33
                    || range.last > 2048
                    || range.first > range.last
                    || route.layout != UpstreamProjectionLayout::Columns
                    || !matches!(
                        route.arithmetic,
                        UpstreamProjectionArithmetic::MmqD4MarkerV2
                            | UpstreamProjectionArithmetic::MmqDs4MarkerV2
                            | UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
                    )
                {
                    return Err(
                        "prefill interval requires MarkerV2 Columns MMQ within33..2048".into(),
                    );
                }
            }
            for &rows in &route.local_rows {
                if rows == 0 || rows > 32 || !domains.insert((route.layout, rows)) {
                    return Err("empty, ambiguous or unsupported upstream row domain".into());
                }
                match (route.arithmetic.finite_expression(), route.layout) {
                    (UpstreamProjectionArithmetic::MmvqQ8_1V1, UpstreamProjectionLayout::Columns) if !matches!(rows, 1 | 4 | 8) => return Err("MMVQ Columns adapter only declares actual exports 1/4/8; no intermediate-width extrapolation".into()),
                    (UpstreamProjectionArithmetic::MmqD4V1 | UpstreamProjectionArithmetic::MmqDs4V1, UpstreamProjectionLayout::Channels) => return Err("MMQ channel routing is outside the declared dense ABI".into()),
                    _ => {}
                }
            }
        }
        // Compare intervals directly, without expanding thousands of rows into
        // either the contract or the planner's cold cache.
        for (index, route) in self.routes.iter().enumerate() {
            if let Some(a) = route.prefill_rows {
                for other in &self.routes[index + 1..] {
                    if route.layout == other.layout
                        && other
                            .prefill_rows
                            .is_some_and(|b| a.first <= b.last && b.first <= a.last)
                    {
                        return Err("overlapping upstream prefill intervals".into());
                    }
                }
            }
        }
        if let Some(selection) = self.geometry_selection {
            selection.validate(self)?;
        }
        Ok(())
    }

    pub fn staged(self) -> Result<StagedNumericalArithmetic, String> {
        self.validate()?;
        Ok(StagedNumericalArithmetic {
            schema_version: if self.format == ProjectionBlockFormat::Q6K {
                NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16
            } else if self.geometry_selection.is_some() {
                NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY
            } else if self.format.is_extra()
                && self.routes.iter().any(|route| route.prefill_rows.is_some())
            {
                NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
            } else if self.format.is_extra() {
                NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA
            } else {
                NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM
            },
            stages: vec![NumericalArithmeticStage::UpstreamProjection { policy: self }],
        })
    }
}
