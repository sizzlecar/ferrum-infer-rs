//! Exact F32 column transport and explicitly broadcast channel arithmetic.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
pub const GROUPED_COLUMN_SLICE_F32_OPERATION_ID: &str = "operation.grouped_column_slice.f32";
pub const GROUPED_COLUMN_SLICE_F32_CAPABILITY_ID: &str =
    "capability.operation.grouped_column_slice.f32";
pub const CHANNEL_BIAS_ADD_F32_OPERATION_ID: &str = "operation.channel_bias_add.f32";
pub const CHANNEL_BIAS_ADD_F32_CAPABILITY_ID: &str = "capability.operation.channel_bias_add.f32";
pub const CHANNEL_SCALED_RESIDUAL_F32_OPERATION_ID: &str = "operation.channel_scaled_residual.f32";
pub const CHANNEL_SCALED_RESIDUAL_F32_CAPABILITY_ID: &str =
    "capability.operation.channel_scaled_residual.f32";
pub const NAN_TO_NUM_F32_OPERATION_ID: &str = "operation.nan_to_num.f32";
pub const NAN_TO_NUM_F32_CAPABILITY_ID: &str = "capability.operation.nan_to_num.f32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32ChannelwiseOp {
    GroupedColumnSlice,
    BiasAdd,
    ScaledResidual,
    NanToNum,
}
impl F32ChannelwiseOp {
    pub const ALL: [Self; 4] = [
        Self::GroupedColumnSlice,
        Self::BiasAdd,
        Self::ScaledResidual,
        Self::NanToNum,
    ];
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::GroupedColumnSlice => GROUPED_COLUMN_SLICE_F32_OPERATION_ID,
            Self::BiasAdd => CHANNEL_BIAS_ADD_F32_OPERATION_ID,
            Self::ScaledResidual => CHANNEL_SCALED_RESIDUAL_F32_OPERATION_ID,
            Self::NanToNum => NAN_TO_NUM_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::GroupedColumnSlice => GROUPED_COLUMN_SLICE_F32_CAPABILITY_ID,
            Self::BiasAdd => CHANNEL_BIAS_ADD_F32_CAPABILITY_ID,
            Self::ScaledResidual => CHANNEL_SCALED_RESIDUAL_F32_CAPABILITY_ID,
            Self::NanToNum => NAN_TO_NUM_F32_CAPABILITY_ID,
        }
    }
    pub const fn input_count(self) -> usize {
        match self {
            Self::GroupedColumnSlice | Self::NanToNum => 1,
            Self::BiasAdd => 2,
            Self::ScaledResidual => 3,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        contract(self)
    }
}
/// Exact bit copy: F32 X[R,G*W] -> Y[R,G*C]. For every group, select
/// `[column_start,column_start+column_count)` within its `group_width`.
/// Row/group order, signed zeros, subnormals and all NaN payload bits survive.
pub fn grouped_column_slice_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(F32ChannelwiseOp::GroupedColumnSlice)
}
/// F32 X/Y[R,C], bias[C]. Y[r,c]=RN32(X[r,c]+bias[c]); no scalar, row,
/// leading-axis or rank-two vector broadcast is inferred.
pub fn channel_bias_add_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(F32ChannelwiseOp::BiasAdd)
}
/// F32 base/branch/Y[R,C], scale[C]. Y=fma(branch,scale,base), with ONE
/// final F32 rounding, including cancellation of overflowing separate products.
/// This is not separate RN32(branch*scale)+base. IEEE NaN/Inf propagation and
/// gradual underflow apply; NaN payload identity is not an arithmetic promise.
pub fn channel_scaled_residual_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(F32ChannelwiseOp::ScaledResidual)
}
/// F32 X/Y[R,C]. NaN -> positive zero, +Inf -> F32_MAX, -Inf -> -F32_MAX.
/// All finite input bits, including signed zero and subnormals, survive exactly.
pub fn nan_to_num_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(F32ChannelwiseOp::NanToNum)
}
fn contract(kind: F32ChannelwiseOp) -> Result<StandardOperationContract, VNextError> {
    let x = vec![
        symbol("rows"),
        symbol(if kind == F32ChannelwiseOp::GroupedColumnSlice {
            "source_columns"
        } else {
            "columns"
        }),
    ];
    let y = vec![symbol("rows"), symbol("columns")];
    let mut inputs = vec![contiguous_tensor(
        x,
        [ElementType::F32],
        TensorAccess::Read,
    )?];
    if kind == F32ChannelwiseOp::ScaledResidual {
        inputs.push(contiguous_tensor(
            y.clone(),
            [ElementType::F32],
            TensorAccess::Read,
        )?);
    }
    if matches!(
        kind,
        F32ChannelwiseOp::BiasAdd | F32ChannelwiseOp::ScaledResidual
    ) {
        inputs.push(contiguous_tensor(
            vec![symbol("columns")],
            [ElementType::F32],
            TensorAccess::Read,
        )?);
    }
    let attributes = if kind == F32ChannelwiseOp::GroupedColumnSlice {
        AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("group_width")?,
            unsigned_attribute("column_count")?,
            (
                AttributeId::new("column_start")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Unsigned,
                    required: true,
                    constraint: AttributeConstraint::UnsignedRange {
                        minimum: 0,
                        maximum: u64::MAX,
                    },
                },
            ),
        ]))?
    } else {
        AttributeSchema::empty()
    };
    let descriptor = OperationDescriptor {
        id: OperationId::new(kind.operation_id())?,
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![contiguous_tensor(
            y,
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(kind.capability_id(), ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked scalar/index metadata only. Physical ranges and owners are admitted
/// independently by each provider/runtime; matching element counts are not a
/// substitute for the exact matrix/vector shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32ChannelwiseGeometry {
    kind: F32ChannelwiseOp,
    rows: u64,
    source_columns: u64,
    columns: u64,
    group_width: u64,
    column_start: u64,
    column_count: u64,
    elements: u64,
    arithmetic: u64,
}
impl F32ChannelwiseGeometry {
    pub fn new(
        kind: F32ChannelwiseOp,
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        kind.contract()?.descriptor.validate_attributes(attrs)?;
        if inputs.len() != kind.input_count() {
            return Err(invalid("input count differs from selected operation"));
        }
        for t in inputs.iter().copied().chain(std::iter::once(output)) {
            if t.element_type() != ElementType::F32
                || t.layout() != &ResolvedTensorLayout::Contiguous
                || t.dimensions().contains(&0)
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid(
                    "requires nonempty contiguous F32 tensors within address space",
                ));
            }
        }
        let [rows, source_columns] = inputs[0].dimensions() else {
            return Err(invalid("first input must be rank two"));
        };
        let [out_rows, columns] = output.dimensions() else {
            return Err(invalid("output must be rank two"));
        };
        if rows != out_rows {
            return Err(invalid("output row count differs"));
        }
        let (group_width, column_start, column_count) =
            if kind == F32ChannelwiseOp::GroupedColumnSlice {
                let read = |name: &str| match attrs.get(&AttributeId::new(name)?) {
                    Some(SemanticValue::Unsigned(v)) => Ok(*v),
                    _ => Err(invalid("slice attribute must be unsigned")),
                };
                let width = read("group_width")?;
                let start = read("column_start")?;
                let count = read("column_count")?;
                if width == 0
                    || count == 0
                    || source_columns % width != 0
                    || start.checked_add(count).is_none_or(|n| n > width)
                    || (source_columns / width).checked_mul(count) != Some(*columns)
                {
                    return Err(invalid(
                        "slice window/group count must exactly fit original and output columns",
                    ));
                }
                (width, start, count)
            } else {
                if inputs[0] != output {
                    return Err(invalid("output must exactly match the first matrix"));
                }
                match kind {
                    F32ChannelwiseOp::BiasAdd => {
                        if inputs[1].dimensions() != [*columns] {
                            return Err(invalid("bias must be a rank-one vector matching columns"));
                        }
                    }
                    F32ChannelwiseOp::ScaledResidual => {
                        if inputs[1] != inputs[0] || inputs[2].dimensions() != [*columns] {
                            return Err(invalid(
                                "requires an identical branch matrix and rank-one channel scale",
                            ));
                        }
                    }
                    _ => {}
                }
                (*columns, 0, *columns)
            };
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("output element count overflows"))?;
        let arithmetic = elements
            .checked_mul(match kind {
                F32ChannelwiseOp::BiasAdd => 1,
                F32ChannelwiseOp::ScaledResidual => 2,
                _ => 0,
            })
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        Ok(Self {
            kind,
            rows: *rows,
            source_columns: *source_columns,
            columns: *columns,
            group_width,
            column_start,
            column_count,
            elements,
            arithmetic,
        })
    }
    pub const fn kind(self) -> F32ChannelwiseOp {
        self.kind
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn source_columns(self) -> u64 {
        self.source_columns
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn group_width(self) -> u64 {
        self.group_width
    }
    pub const fn column_start(self) -> u64 {
        self.column_start
    }
    pub const fn column_count(self) -> u64 {
        self.column_count
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
    pub fn input_elements(self, index: usize) -> Option<u64> {
        match (self.kind, index) {
            (_, 0) => self.rows.checked_mul(self.source_columns),
            (F32ChannelwiseOp::BiasAdd, 1) | (F32ChannelwiseOp::ScaledResidual, 2) => {
                Some(self.columns)
            }
            (F32ChannelwiseOp::ScaledResidual, 1) => Some(self.elements),
            _ => None,
        }
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 channelwise: {reason}"),
    }
}
#[cfg(test)]
mod tests;
