//! Local projection arithmetic inside an otherwise unchanged strict operation.
//!
//! These are declarations, not provider registrations or evidence that a leaf
//! was executed. Schema 1 binds only the projection ports of the two standard
//! operations below. Nonlinearities, normalization, recurrent state, residuals
//! and conversions outside those ports retain the referenced base semantics.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{AffineMinCorrection, NumericalArithmeticStage, StagedNumericalArithmetic};
use crate::vnext::{
    dense_swiglu_contract, gated_delta_recurrent_attention_f32_master_contract,
    BlockQuantizationSpec, ContractVersion, ElementType, OperationContract, OperationId,
    StandardOperationContract, DENSE_SWIGLU_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
};

pub const COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrictNumericalOperation {
    pub operation_id: OperationId,
    pub version: ContractVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionRole {
    SwiGluGateUp,
    SwiGluDown,
    GatedDeltaInput,
    GatedDeltaOutput,
}

/// Logical K/N eligibility is independent of token count, concurrency, device
/// and provider. Numerical policy cannot silently switch with batch size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionShapeEligibility {
    pub minimum_input_features: u64,
    pub minimum_output_features: u64,
    pub input_features_multiple: u32,
}

/// Exact opaque block ABIs, not a container name or a generic INT4 label.
/// Schema 1 only declares untransformed weights in these formats. Any other
/// valid format or transformed weight retains the strict base arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionBlockFormat {
    Q4K,
    Q5K,
    Iq4Xs,
}

impl ProjectionBlockFormat {
    fn abi(self) -> (&'static str, u32, u32) {
        match self {
            Self::Q4K => ("quantization.gguf.q4-k", 256, 144),
            Self::Q5K => ("quantization.gguf.q5-k", 256, 176),
            Self::Iq4Xs => ("quantization.gguf.iq4-xs", 256, 136),
        }
    }

    fn matches(self, block: &BlockQuantizationSpec) -> bool {
        let (id, values, bytes) = self.abi();
        block.format_id.as_str() == id
            && block.logical_values_per_block == values
            && block.bytes_per_block == bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizedProjectionLeafContract {
    pub format: ProjectionBlockFormat,
    pub shape: ProjectionShapeEligibility,
    pub arithmetic: StagedNumericalArithmetic,
}

/// An explicit fallback is required even when all currently observed weights
/// qualify. It is part of the fingerprint, not a provider's private heuristic.
/// Only declared ineligibility permits this fallback. Missing support for an
/// eligible staged leaf is a qualification failure, not permission to change
/// its arithmetic. The strict base itself still needs a qualified provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StrictProjectionFallback {
    RetainBaseArithmetic {},
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionArithmeticOverride {
    pub role: ProjectionRole,
    /// Ordinal in the referenced standard operation's input signature.
    pub weight_input_ordinal: u32,
    /// Internal projection ports, not necessarily the fused operation's ports.
    pub activation_input_type: ElementType,
    pub activation_output_type: ElementType,
    pub leaves: Vec<QuantizedProjectionLeafContract>,
    pub fallback: StrictProjectionFallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeNumericalArithmetic {
    pub schema_version: u32,
    pub strict_base: StrictNumericalOperation,
    /// Roles omitted here keep their base arithmetic, as does all non-projection
    /// work. A packed weight's physical matrix parts are evaluated separately.
    pub projections: Vec<ProjectionArithmeticOverride>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrictProjectionReason {
    UnmodifiedProjection,
    TransformedWeight,
    FormatNotDeclared,
    ShapeNotDeclared,
}

/// The contract's answer for supplied static facts. This does not resolve a
/// provider, inspect real weights, or attest that an execution used these stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredProjectionArithmetic<'a> {
    Staged(&'a StagedNumericalArithmetic),
    StrictBase(StrictProjectionReason),
}

impl CompositeNumericalArithmetic {
    pub(super) fn base_contract(&self) -> Result<StandardOperationContract, String> {
        let contract = match self.strict_base.operation_id.as_str() {
            DENSE_SWIGLU_OPERATION_ID => dense_swiglu_contract(),
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID => {
                gated_delta_recurrent_attention_f32_master_contract()
            }
            _ => return Err("composite schema 1 requires a supported standard strict base".into()),
        }
        .map_err(|error| error.to_string())?;
        if contract.descriptor().version != self.strict_base.version {
            return Err(
                "composite strict base must match the exact standard operation version".into(),
            );
        }
        Ok(contract)
    }

    fn role_port(&self, role: ProjectionRole) -> Result<(u32, usize), String> {
        match (self.strict_base.operation_id.as_str(), role) {
            (DENSE_SWIGLU_OPERATION_ID, ProjectionRole::SwiGluGateUp) => Ok((1, 3)),
            (DENSE_SWIGLU_OPERATION_ID, ProjectionRole::SwiGluDown) => Ok((2, 2)),
            (
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
                ProjectionRole::GatedDeltaInput,
            ) => Ok((2, 2)),
            (
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
                ProjectionRole::GatedDeltaOutput,
            ) => Ok((7, 2)),
            _ => Err("projection role does not belong to the declared strict base".into()),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION {
            return Err(
                "unsupported composite arithmetic schema version; explicit migration required"
                    .into(),
            );
        }
        let base = self.base_contract()?;
        if self.projections.is_empty() {
            return Err(
                "composite arithmetic must declare at least one projection override".into(),
            );
        }
        let mut roles = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for projection in &self.projections {
            let (slot, rank) = self.role_port(projection.role)?;
            if !roles.insert(projection.role) || !slots.insert(projection.weight_input_ordinal) {
                return Err("projection roles and weight input ordinals must be unique".into());
            }
            let weight = base
                .descriptor()
                .inputs
                .get(slot as usize)
                .ok_or("standard base is missing the declared projection weight port")?;
            if projection.weight_input_ordinal != slot
                || weight.dimensions().len() != rank
                || weight.element_types() != &BTreeSet::from([ElementType::F16])
            {
                return Err(
                    "projection weight input ordinal disagrees with the standard base signature"
                        .into(),
                );
            }
            // These two versioned bases round both internal linear boundaries
            // to F16. In particular GDN's surrounding hidden/state path is F32.
            if projection.activation_input_type != ElementType::F16
                || projection.activation_output_type != ElementType::F16
                || projection.leaves.is_empty()
            {
                return Err(
                    "schema 1 projection overrides require F16 internal ports and nonempty leaves"
                        .into(),
                );
            }
            let mut formats = BTreeSet::new();
            for leaf in &projection.leaves {
                if !formats.insert(leaf.format) {
                    return Err(
                        "a projection cannot declare competing leaves for the same block format"
                            .into(),
                    );
                }
                leaf.arithmetic.validate()?;
                let NumericalArithmeticStage::IntegerDot {
                    values_per_partial, ..
                } = &leaf.arithmetic.stages[1]
                else {
                    unreachable!("validated staged schema")
                };
                // All supported ABIs change scale (and Q4_K/Q5_K minimum) every
                // 32 values. One integer partial cannot cross coefficients
                // before its single rescale/min-correction stage.
                if *values_per_partial > 32 || 32 % *values_per_partial != 0 {
                    return Err("projection integer partials must divide each 32-value weight coefficient group".into());
                }
                let NumericalArithmeticStage::ActivationQuantization {
                    input_type,
                    group_values,
                    ..
                } = &leaf.arithmetic.stages[0]
                else {
                    unreachable!("validated staged schema")
                };
                if *input_type != projection.activation_input_type
                    || leaf.arithmetic.output_type() != Some(projection.activation_output_type)
                {
                    return Err(
                        "leaf quantization/store dtype differs from its internal projection ports"
                            .into(),
                    );
                }
                let multiple = leaf.shape.input_features_multiple;
                if leaf.shape.minimum_input_features == 0
                    || leaf.shape.minimum_output_features == 0
                    || multiple == 0
                    || multiple % leaf.format.abi().1 != 0
                    || multiple % *group_values != 0
                {
                    return Err("projection shape must declare positive extents and complete weight/activation groups".into());
                }
                let NumericalArithmeticStage::Rescale { min_correction, .. } =
                    &leaf.arithmetic.stages[2]
                else {
                    unreachable!("validated staged schema")
                };
                let valid_min = matches!(
                    (leaf.format, min_correction),
                    (
                        ProjectionBlockFormat::Q4K | ProjectionBlockFormat::Q5K,
                        AffineMinCorrection::QuantizedPartialSum { .. }
                    ) | (ProjectionBlockFormat::Iq4Xs, AffineMinCorrection::None {})
                );
                if !valid_min {
                    return Err(
                        "leaf affine minimum policy disagrees with its declared block format"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Evaluate only declared eligibility. `input_features`/`output_features`
    /// describe one physical matrix part, including parts of packed weights.
    /// No token-count heuristic or implicit dense/dequantized conversion exists.
    pub fn declared_projection_arithmetic(
        &self,
        role: ProjectionRole,
        block: Option<&BlockQuantizationSpec>,
        input_features: u64,
        output_features: u64,
        has_weight_transform: bool,
    ) -> Result<DeclaredProjectionArithmetic<'_>, String> {
        self.validate()?;
        self.role_port(role)?;
        if input_features == 0 || output_features == 0 {
            return Err("projection facts must describe a nonempty matrix".into());
        }
        if let Some(block) = block {
            block.validate().map_err(|error| error.to_string())?;
        }
        let strict = DeclaredProjectionArithmetic::StrictBase;
        let Some(projection) = self.projections.iter().find(|p| p.role == role) else {
            return Ok(strict(StrictProjectionReason::UnmodifiedProjection));
        };
        if has_weight_transform {
            return Ok(strict(StrictProjectionReason::TransformedWeight));
        }
        let Some(leaf) = projection
            .leaves
            .iter()
            .find(|leaf| block.is_some_and(|b| leaf.format.matches(b)))
        else {
            return Ok(strict(StrictProjectionReason::FormatNotDeclared));
        };
        if input_features < leaf.shape.minimum_input_features
            || output_features < leaf.shape.minimum_output_features
            || input_features % u64::from(leaf.shape.input_features_multiple) != 0
        {
            return Ok(strict(StrictProjectionReason::ShapeNotDeclared));
        }
        Ok(DeclaredProjectionArithmetic::Staged(&leaf.arithmetic))
    }
}
