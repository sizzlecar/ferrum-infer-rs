//! Independent F32 Q6 projection arithmetic. This does not extend the F16
//! upstream projection schemas or their closed physical-format sets.
use serde::{Deserialize, Serialize};

use super::{NumericalArithmeticStage, StagedNumericalArithmetic, StrictNumericalOperation};
use crate::vnext::{BlockQuantizationSpec, ContractVersion, ElementType, OperationId};

pub const NUMERICAL_ARITHMETIC_SCHEMA_VERSION_Q6_MMQ_F32: u32 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Q6MmqF32Arithmetic {
    /// F32 copy/pad, fast reciprocal D4 pack, Q6 integer MMA/F32 stream-K,
    /// and F32 marker publication. No F16 activation or output conversion.
    D4MarkerV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q6MmqF32Route {
    Mmq,
    Strict,
}

/// The route domain is part of the numerical version, not artifact discovery.
/// An eligible MMQ route with unavailable code or an execution error must fail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q6MmqF32Policy {
    pub arithmetic: Q6MmqF32Arithmetic,
    pub minimum_physical_rows: u32,
    pub maximum_physical_rows: u32,
    pub ineligible_strict: StrictNumericalOperation,
}

impl Q6MmqF32Policy {
    pub fn new() -> Self {
        Self {
            arithmetic: Q6MmqF32Arithmetic::D4MarkerV1,
            minimum_physical_rows: 1,
            maximum_physical_rows: 32,
            ineligible_strict: StrictNumericalOperation {
                operation_id: OperationId::new(
                    crate::vnext::LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
                )
                .expect("standard operation ID"),
                version: ContractVersion::new(1, 0),
            },
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self != &Self::new() {
            return Err(
                "Q6 F32 D4/MMQ v1 requires its exact rows and strict fallback contract".into(),
            );
        }
        Ok(())
    }

    /// `physical_rows` counts selected last-token rows in this actual dispatch,
    /// not prompt tokens, offered concurrency or a maximum reservation.
    /// `eligible_format_layout_transform` means the validated physical leaf is
    /// plain Q6_K, complete K256, aligned contiguous Columns without a transform.
    /// It must never depend on artifact availability, pointer liveness or a
    /// failed native launch: those require independent errors/authorization.
    pub fn route(
        &self,
        physical_rows: u32,
        eligible_format_layout_transform: bool,
    ) -> Result<Q6MmqF32Route, String> {
        self.validate()?;
        if physical_rows == 0 {
            return Err("last-token projection requires at least one physical row".into());
        }
        Ok(if eligible_format_layout_transform && physical_rows <= 32 {
            Q6MmqF32Route::Mmq
        } else {
            Q6MmqF32Route::Strict
        })
    }

    /// Static encoding/shape eligibility only. A provider must additionally
    /// validate actual matrix layout, alignment, transforms and live extents.
    pub fn eligible_weight(block: Option<&BlockQuantizationSpec>, k: u64, n: u64) -> bool {
        k != 0
            && k % 256 == 0
            && n != 0
            && block.is_some_and(|b| {
                b.format_id.as_str() == "quantization.gguf.q6-k"
                    && b.logical_values_per_block == 256
                    && b.bytes_per_block == 210
            })
    }

    pub fn staged(self) -> StagedNumericalArithmetic {
        StagedNumericalArithmetic {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_Q6_MMQ_F32,
            stages: vec![NumericalArithmeticStage::Q6MmqF32Projection { policy: self }],
        }
    }

    pub const fn semantics(&self) -> Q6MmqF32Semantics {
        Q6MmqF32Semantics {
            input_type: ElementType::F32,
            output_type: ElementType::F32,
            weight_values_per_block: 256,
            weight_bytes_per_block: 210,
            activation_scale_group_values: 32,
            packed_bytes_per_128_values: 144,
            canonical_nan_bits: 0x7fc00000,
            retained_weight_flag_bytes_per_leaf: 4,
        }
    }
}

impl Default for Q6MmqF32Policy {
    fn default() -> Self {
        Self::new()
    }
}

/// Normative equations of D4MarkerV1: amax and 127/amax then reciprocal use
/// F32 fast math (approximate division/reciprocal, FTZ and contraction allowed).
/// Codes are roundf(x * reciprocal), signed I8; scale is stored F32. Zero groups
/// and K padding use positive-zero scale/codes. Nonfinite inputs or consumed
/// scale/reciprocal/code metadata poison the whole row (NaN scale, zero codes).
/// Finite scale underflow to zero is allowed. In particular a finite input is
/// not a promise that approximate division stays in its valid dynamic domain.
/// Q6 signed codes [-32,31] and signed I8 scales per16 values enter exact local
/// integer dots; half base scale is converted to F32. There is no affine min,
/// stored original sum, or extra half rounding of group coefficients.
/// Upstream MMA/stream-K reductions produce F32. A nonfinite weight coefficient
/// poisons the physical leaf; row/leaf markers or a nonfinite raw output publish
/// canonical F32 NaN. Every finite raw F32 value, including MAX, is preserved.
///
/// The operation-defined finite oracle decodes the actual packed I8 codes and
/// F32 scales, then accumulates their products with independently decoded Q6
/// weights in F64. Implementation error is assessed with an input-dependent
/// bound based on expanded absolute products, separately from quantization
/// error against the original F32 activation. Output-only relative tolerance
/// against the strict head does not define this arithmetic's correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Q6MmqF32Semantics {
    pub input_type: ElementType,
    pub output_type: ElementType,
    pub weight_values_per_block: u32,
    pub weight_bytes_per_block: u32,
    pub activation_scale_group_values: u32,
    pub packed_bytes_per_128_values: u32,
    pub canonical_nan_bits: u32,
    pub retained_weight_flag_bytes_per_leaf: u32,
}

#[cfg(test)]
mod tests;
