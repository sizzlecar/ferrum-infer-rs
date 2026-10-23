//! BF16 scale modulation and gated residuals with explicit row broadcasting.
//! Their internal BF16 boundaries are semantic, even when a provider keeps
//! the intermediate scalars in registers rather than materializing tensors.

use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const MODULATION_SCALE_BF16_OPERATION_ID: &str = "operation.modulation_scale.bf16";
pub const MODULATION_SCALE_BF16_CAPABILITY_ID: &str = "capability.operation.modulation_scale.bf16";
pub const GATED_RESIDUAL_BF16_OPERATION_ID: &str = "operation.gated_residual.bf16";
pub const GATED_RESIDUAL_BF16_CAPABILITY_ID: &str = "capability.operation.gated_residual.bf16";

/// Explicit mapping from a contiguous parameter matrix to activation rows.
/// No mode is inferred from a tensor's element count or its leading extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowBroadcast {
    /// Parameter shape is exactly the activation shape `[rows, width]`.
    Elementwise,
    /// Parameter shape is `[1, width]`; every activation row reads row zero.
    Row,
}

impl RowBroadcast {
    pub const ALL: [Self; 2] = [Self::Elementwise, Self::Row];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Row => "row",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.as_str() == value)
    }
}

/// Scale-only modulation: `y = B(x * B(1 + s))`.
///
/// Input ordinals are `x`, then `s`. Each add/multiply uses FP32, and each
/// `B` is BF16 round-to-nearest-even. The inner rounding must survive fusion;
/// there is no shift term. `broadcast` selects elementwise or single-row
/// parameter access without materializing a broadcast parameter tensor.
pub fn modulation_scale_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    pointwise_bf16_contract(
        MODULATION_SCALE_BF16_OPERATION_ID,
        MODULATION_SCALE_BF16_CAPABILITY_ID,
        false,
    )
}

/// Gated residual: `y = B(x + B(g * u))`.
///
/// Input ordinals are residual `x`, supplied gate `g`, and update `u`. Each
/// add/multiply uses FP32, and each `B` is BF16 round-to-nearest-even. This
/// does not apply an activation to `g`, constrain its values, or replace the
/// inner BF16 boundary with a fused FP32 multiply-add. Only `g` broadcasts.
pub fn gated_residual_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    pointwise_bf16_contract(
        GATED_RESIDUAL_BF16_OPERATION_ID,
        GATED_RESIDUAL_BF16_CAPABILITY_ID,
        true,
    )
}

fn pointwise_bf16_contract(
    operation_id: &str,
    capability_id: &str,
    has_update: bool,
) -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("width")];
    let mut inputs = vec![
        contiguous_tensor(dimensions.clone(), [ElementType::Bf16], TensorAccess::Read)?,
        contiguous_tensor(
            vec![symbol("parameter_rows"), symbol("width")],
            [ElementType::Bf16],
            TensorAccess::Read,
        )?,
    ];
    if has_update {
        inputs.push(contiguous_tensor(
            dimensions.clone(),
            [ElementType::Bf16],
            TensorAccess::Read,
        )?);
    }
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([(
            AttributeId::new("broadcast")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Text,
                required: true,
                constraint: AttributeConstraint::TextChoices {
                    values: RowBroadcast::ALL
                        .into_iter()
                        .map(|mode| mode.as_str().to_owned())
                        .collect(),
                },
            },
        )]))?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(capability_id, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked semantic geometry, not an allocation or invocation authority.
///
/// Providers call this during resource estimation and encoding: independent
/// shape symbols do not by themselves relate `parameter_rows` to `broadcast`.
/// A gated-residual provider must additionally check that its update tensor
/// equals the base tensor's specification. This helper neither inspects
/// payload values nor authorizes physical aliasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16RowBroadcastGeometry {
    rows: u64,
    width: u64,
    parameter_rows: u64,
    mode: RowBroadcast,
}

impl Bf16RowBroadcastGeometry {
    pub fn new(
        base: &ResolvedTensorSpec,
        parameter: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        mode: RowBroadcast,
    ) -> Result<Self, VNextError> {
        let [rows, width] = base.dimensions() else {
            return Err(invalid_geometry("base tensor must have rank two"));
        };
        let [parameter_rows, parameter_width] = parameter.dimensions() else {
            return Err(invalid_geometry("parameter tensor must have rank two"));
        };
        if base != output
            || base.element_type() != ElementType::Bf16
            || parameter.element_type() != ElementType::Bf16
            || base.layout() != &ResolvedTensorLayout::Contiguous
            || parameter.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || *width == 0
            || *parameter_rows == 0
            || width != parameter_width
        {
            return Err(invalid_geometry(
                "base/output must be identical contiguous BF16 [rows, width] tensors and the BF16 parameter must have the same width",
            ));
        }
        let expected_parameter_rows = match mode {
            RowBroadcast::Elementwise => *rows,
            RowBroadcast::Row => 1,
        };
        if *parameter_rows != expected_parameter_rows {
            return Err(invalid_geometry(format!(
                "{} broadcast requires {expected_parameter_rows} parameter rows, found {parameter_rows}",
                mode.as_str(),
            )));
        }
        // ResolvedTensorSpec construction already checked each tensor's byte
        // extent. Pointwise arithmetic needs no width-to-FP32 conversion, so
        // there is no LayerNorm-style 2^24 width restriction here.
        Ok(Self {
            rows: *rows,
            width: *width,
            parameter_rows: *parameter_rows,
            mode,
        })
    }

    pub const fn rows(self) -> u64 {
        self.rows
    }

    pub const fn width(self) -> u64 {
        self.width
    }

    pub const fn parameter_rows(self) -> u64 {
        self.parameter_rows
    }

    pub const fn mode(self) -> RowBroadcast {
        self.mode
    }
}

fn invalid_geometry(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 row broadcast geometry: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
