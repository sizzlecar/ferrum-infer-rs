//! Compact table routing with fused F32 RMS normalization and modulation.
use super::*;
use crate::{ResolvedTensorSpec, SemanticValue};

pub const ROW_SELECTED_RMS_MODULATION_BF16_OPERATION_ID: &str =
    "operation.row_selected_rms_modulation.bf16";
pub const ROW_SELECTED_RMS_MODULATION_BF16_CAPABILITY_ID: &str =
    "capability.operation.row_selected_rms_modulation.bf16";
pub const ROW_SELECTED_RESIDUAL_RMS_MODULATION_BF16_OPERATION_ID: &str =
    "operation.row_selected_residual_rms_modulation.bf16";
pub const ROW_SELECTED_RESIDUAL_RMS_MODULATION_BF16_CAPABILITY_ID: &str =
    "capability.operation.row_selected_residual_rms_modulation.bf16";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSelectedRmsModulationKind {
    Normalize,
    Residual,
}
impl RowSelectedRmsModulationKind {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Normalize => ROW_SELECTED_RMS_MODULATION_BF16_OPERATION_ID,
            Self::Residual => ROW_SELECTED_RESIDUAL_RMS_MODULATION_BF16_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Normalize => ROW_SELECTED_RMS_MODULATION_BF16_CAPABILITY_ID,
            Self::Residual => ROW_SELECTED_RESIDUAL_RMS_MODULATION_BF16_CAPABILITY_ID,
        }
    }
    /// X, compact table, gamma, and (only for Residual) branch.
    pub const fn input_count(self) -> usize {
        match self {
            Self::Normalize => 3,
            Self::Residual => 4,
        }
    }
    /// Normalize returns Y. Residual returns rounded hidden followed by Y.
    pub const fn output_count(self) -> usize {
        match self {
            Self::Normalize => 1,
            Self::Residual => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSelectedRmsModulationProfile {
    Block8192Vec8Warp16Fma,
}
impl RowSelectedRmsModulationProfile {
    pub const fn as_str(self) -> &'static str {
        "block8192_vec8_warp16_fma_v1"
    }
    pub fn attributes(self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([(
            AttributeId::new("profile")?,
            SemanticValue::Text(self.as_str().into()),
        )]))
    }
}

/// BF16 X[M,H], table[P,W], gamma[H] -> BF16 Y[M,H]. Table row runs and
/// shift/scale column windows use the existing checked compact-routing syntax.
/// Decode to F32. Reduce squares in the explicit 8192-column profile below,
/// divide by H, add epsilon, and compute rsqrt. Let u=RN32(RN32(X*r)*gamma).
/// Y=BF16_RNE(FMA32(u, RN32(1+scale), shift)). No intermediate BF16 norm or
/// BF16 factor is permitted. CUDA uses div.full.f32 and rsqrt.approx.ftz.f32;
/// Reference declares its own RN division/sqrt approximation. Other operations
/// have gradual underflow. Provider fingerprints identify these approximations.
///
/// Reduction maps each of 512 lanes to columns lane*8+[0..8), then +4096.
/// Invalid columns are zero. A lane starts with RN32(x1*x1), then FMA(x0,x0,s).
/// Normalize adds each even square by FMA, each odd square by separate RN32
/// multiply/add. Residual adds all remaining squares by FMA. Warp XOR offsets
/// are 16,8,4,2,1, followed by the 16 warp totals at XOR offsets 8,4,2,1.
/// This is a fixed reduction profile, not an implicit shape-dependent recipe.
pub fn row_selected_rms_modulation_bf16_contract() -> Result<StandardOperationContract, VNextError>
{
    contract(RowSelectedRmsModulationKind::Normalize)
}
/// Inputs add BF16 branch[M,H]. Compute h=FMA32(gate,branch,X), store BF16(h)
/// as output0, and use the ORIGINAL F32 h for the square reduction and output1
/// modulation. Reading output0 back into normalization changes this contract.
/// Both outputs must be disjoint from all inputs and from each other.
pub fn row_selected_residual_rms_modulation_bf16_contract(
) -> Result<StandardOperationContract, VNextError> {
    contract(RowSelectedRmsModulationKind::Residual)
}
fn contract(kind: RowSelectedRmsModulationKind) -> Result<StandardOperationContract, VNextError> {
    // Reuse only the existing routing/column schema and tensor ports; the
    // numerical operation and capability are new and independently selected.
    let mut descriptor = row_selected_modulation_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(kind.operation_id())?;
    descriptor.provider = provider_requirement(kind.capability_id(), ContractVersion::new(1, 0))?;
    descriptor.inputs.push(contiguous_tensor(
        vec![symbol("columns")],
        [ElementType::Bf16],
        TensorAccess::Read,
    )?);
    let mut attributes = descriptor.attributes.entries().clone();
    let epsilon = positive_epsilon_attribute("epsilon")?;
    attributes.insert(epsilon.0, epsilon.1);
    let profile = text_choices_attribute(
        "profile",
        [RowSelectedRmsModulationProfile::Block8192Vec8Warp16Fma.as_str()],
    )?;
    attributes.insert(profile.0, profile.1);
    if kind == RowSelectedRmsModulationKind::Residual {
        descriptor.inputs.push(descriptor.inputs[0].clone());
        descriptor.outputs.push(descriptor.outputs[0].clone());
        let gate = nonnegative_unsigned_attribute("gate_column")?;
        attributes.insert(gate.0, gate.1);
    }
    descriptor.attributes = AttributeSchema::new(attributes)?;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSelectedRmsModulationGeometry {
    kind: RowSelectedRmsModulationKind,
    routing: RowSelectedBf16Geometry,
    epsilon: CanonicalRational,
    gate_column: u64,
    arithmetic: u64,
}
impl RowSelectedRmsModulationGeometry {
    pub fn new(
        kind: RowSelectedRmsModulationKind,
        inputs: &[&ResolvedTensorSpec],
        outputs: &[&ResolvedTensorSpec],
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        contract(kind)?.descriptor.validate_attributes(attrs)?;
        if inputs.len() != kind.input_count() || outputs.len() != kind.output_count() {
            return Err(invalid("port count differs"));
        }
        let routing_attrs = attrs
            .iter()
            .filter(|(key, _)| {
                matches!(
                    key.as_str(),
                    "run_lengths" | "parameter_rows" | "shift_column" | "scale_column"
                )
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let routing = RowSelectedBf16Geometry::new(
            RowSelectedBf16Kind::Modulation,
            &inputs[..2],
            outputs[outputs.len() - 1],
            &routing_attrs,
        )?;
        let epsilon = match attrs.get(&AttributeId::new("epsilon")?) {
            Some(SemanticValue::Rational(v)) => *v,
            _ => return Err(invalid("missing rational epsilon")),
        };
        RowRmsNormSingleRoundBf16Geometry::new(
            inputs[0],
            inputs[2],
            outputs[outputs.len() - 1],
            epsilon,
        )?;
        if routing.columns() > 8192
            || (kind == RowSelectedRmsModulationKind::Residual
                && (inputs[3] != inputs[0] || outputs[0] != inputs[0]))
        {
            return Err(invalid(
                "profile requires width 1..8192 and identical residual/branch/outputs",
            ));
        }
        let gate_column = if kind == RowSelectedRmsModulationKind::Residual {
            match attrs.get(&AttributeId::new("gate_column")?) {
                Some(SemanticValue::Unsigned(v))
                    if v.checked_add(routing.columns())
                        .is_some_and(|n| n <= routing.parameter_columns()) =>
                {
                    *v
                }
                _ => return Err(invalid("gate column exceeds compact table")),
            }
        } else {
            0
        };
        let arithmetic = routing
            .columns()
            .checked_mul(if kind == RowSelectedRmsModulationKind::Residual {
                9
            } else {
                7
            })
            .and_then(|n| n.checked_add(3))
            .and_then(|n| n.checked_mul(routing.rows()))
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            kind,
            routing,
            epsilon,
            gate_column,
            arithmetic,
        })
    }
    pub const fn kind(&self) -> RowSelectedRmsModulationKind {
        self.kind
    }
    pub fn routing(&self) -> &RowSelectedBf16Geometry {
        &self.routing
    }
    pub const fn gate_column(&self) -> u64 {
        self.gate_column
    }
    pub fn effective_epsilon(&self) -> f32 {
        (self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64) as f32
    }
    pub const fn logical_arithmetic_operations(&self) -> u64 {
        self.arithmetic
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("fused row RMS modulation: {reason}"),
    }
}

#[cfg(test)]
mod tests;
