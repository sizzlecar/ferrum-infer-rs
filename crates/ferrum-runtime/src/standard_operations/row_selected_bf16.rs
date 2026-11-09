//! Compact BF16 parameter-table routing with explicit rounding boundaries.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const ROW_SELECTED_MODULATION_BF16_OPERATION_ID: &str =
    "operation.row_selected_modulation.bf16";
pub const ROW_SELECTED_MODULATION_BF16_CAPABILITY_ID: &str =
    "capability.operation.row_selected_modulation.bf16";
pub const ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID: &str = "operation.row_selected_addcmul.bf16";
pub const ROW_SELECTED_ADDCMUL_BF16_CAPABILITY_ID: &str =
    "capability.operation.row_selected_addcmul.bf16";
pub const ROW_SELECTED_BF16_MAX_RUNS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSelectedBf16Kind {
    Modulation,
    Addcmul,
}
impl RowSelectedBf16Kind {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Modulation => ROW_SELECTED_MODULATION_BF16_OPERATION_ID,
            Self::Addcmul => ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Modulation => ROW_SELECTED_MODULATION_BF16_CAPABILITY_ID,
            Self::Addcmul => ROW_SELECTED_ADDCMUL_BF16_CAPABILITY_ID,
        }
    }
    pub const fn input_count(self) -> usize {
        match self {
            Self::Modulation => 2,
            Self::Addcmul => 3,
        }
    }
}

/// X BF16[M,H], table BF16[P,W] -> Y BF16[M,H]. Explicit RLE selects
/// original table rows; column offsets select H consecutive shift/scale values.
/// factor=B(RN32(1+scale)); product=B(RN32(X*factor));
/// Y=B(RN32(product+shift)), where B is BF16 RNE.
/// Thus both the factor and product BF16 boundaries survive fusion. No FMA,
/// expanded parameter matrix, implicit scratch or mathematical reassociation.
pub fn row_selected_modulation_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    contract(RowSelectedBf16Kind::Modulation)
}
/// X/table/U -> Y, with X/U/Y BF16[M,H]. Y=B(RN32(X+RN32(gate*U))).
/// The product stays F32 until the separate F32 add; only the result rounds B.
pub fn row_selected_addcmul_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    contract(RowSelectedBf16Kind::Addcmul)
}
fn contract(kind: RowSelectedBf16Kind) -> Result<StandardOperationContract, VNextError> {
    let matrix = vec![symbol("rows"), symbol("columns")];
    let mut inputs = vec![
        contiguous_tensor(matrix.clone(), [ElementType::Bf16], TensorAccess::Read)?,
        contiguous_tensor(
            vec![symbol("parameter_count"), symbol("parameter_columns")],
            [ElementType::Bf16],
            TensorAccess::Read,
        )?,
    ];
    if kind == RowSelectedBf16Kind::Addcmul {
        inputs.push(contiguous_tensor(
            matrix.clone(),
            [ElementType::Bf16],
            TensorAccess::Read,
        )?);
    }
    let mut attributes = BTreeMap::new();
    for name in ["run_lengths", "parameter_rows"] {
        attributes.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: 1,
                    maximum: ROW_SELECTED_BF16_MAX_RUNS as u32,
                },
            },
        );
    }
    for name in match kind {
        RowSelectedBf16Kind::Modulation => &["shift_column", "scale_column"][..],
        RowSelectedBf16Kind::Addcmul => &["gate_column"][..],
    } {
        let (id, spec) = nonnegative_unsigned_attribute(name)?;
        attributes.insert(id, spec);
    }
    let descriptor = OperationDescriptor {
        id: OperationId::new(kind.operation_id())?,
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![contiguous_tensor(
            matrix,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attributes)?,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSelectedBf16Geometry {
    kind: RowSelectedBf16Kind,
    rows: u64,
    columns: u64,
    parameter_count: u64,
    parameter_columns: u64,
    elements: u64,
    arithmetic: u64,
    shift_column: u64,
    scale_column: u64,
    gate_column: u64,
    run_ends: Vec<u64>,
    parameter_rows: Vec<u64>,
}
impl RowSelectedBf16Geometry {
    pub fn new(
        kind: RowSelectedBf16Kind,
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        contract(kind)?.descriptor.validate_attributes(attrs)?;
        if inputs.len() != kind.input_count() {
            return Err(invalid("input port count differs from selected operation"));
        }
        let [rows, columns] = inputs[0].dimensions() else {
            return Err(invalid("activation must be rank two"));
        };
        let [parameter_count, parameter_columns] = inputs[1].dimensions() else {
            return Err(invalid("parameter table must be rank two"));
        };
        if inputs
            .iter()
            .copied()
            .chain(std::iter::once(output))
            .any(|t| {
                t.element_type() != ElementType::Bf16
                    || t.layout() != &ResolvedTensorLayout::Contiguous
                    || t.dimensions().contains(&0)
            })
            || output != inputs[0]
            || (kind == RowSelectedBf16Kind::Addcmul && inputs[2] != inputs[0])
        {
            return Err(invalid(
                "requires contiguous BF16 X/U/Y[M,H] and table[P,W]",
            ));
        }
        for t in inputs.iter().copied().chain(std::iter::once(output)) {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let column = |name: &str| -> Result<u64, VNextError> {
            match attrs.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Unsigned(value))
                    if value
                        .checked_add(*columns)
                        .is_some_and(|end| end <= *parameter_columns) =>
                {
                    Ok(*value)
                }
                _ => Err(invalid("parameter column window exceeds original table")),
            }
        };
        let (shift_column, scale_column, gate_column) = match kind {
            RowSelectedBf16Kind::Modulation => {
                (column("shift_column")?, column("scale_column")?, 0)
            }
            RowSelectedBf16Kind::Addcmul => (0, 0, column("gate_column")?),
        };
        let list = |name: &str| -> Result<&[i64], VNextError> {
            match attrs.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Integers(v)) => Ok(v),
                _ => Err(invalid("routing fields must be integer lists")),
            }
        };
        let lengths = list("run_lengths")?;
        let indices = list("parameter_rows")?;
        if lengths.len() != indices.len()
            || !(1..=ROW_SELECTED_BF16_MAX_RUNS).contains(&lengths.len())
        {
            return Err(invalid("routing lists need matching lengths in 1..32"));
        }
        let mut end = 0u64;
        let mut run_ends = Vec::with_capacity(lengths.len());
        let mut parameter_rows = Vec::with_capacity(lengths.len());
        for (&length, &index) in lengths.iter().zip(indices) {
            let length = u64::try_from(length)
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("run lengths must be positive"))?;
            let index = u64::try_from(index)
                .ok()
                .filter(|i| *i < *parameter_count)
                .ok_or_else(|| invalid("parameter row is outside original table"))?;
            end = end
                .checked_add(length)
                .ok_or_else(|| invalid("routing extent overflows"))?;
            run_ends.push(end);
            parameter_rows.push(index);
        }
        if end != *rows {
            return Err(invalid("routing must cover exactly the activation rows"));
        }
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        let arithmetic = elements
            .checked_mul(if kind == RowSelectedBf16Kind::Modulation {
                3
            } else {
                2
            })
            .ok_or_else(|| invalid("arithmetic count overflows"))?;
        Ok(Self {
            kind,
            rows: *rows,
            columns: *columns,
            parameter_count: *parameter_count,
            parameter_columns: *parameter_columns,
            elements,
            arithmetic,
            shift_column,
            scale_column,
            gate_column,
            run_ends,
            parameter_rows,
        })
    }
    pub const fn kind(&self) -> RowSelectedBf16Kind {
        self.kind
    }
    pub const fn rows(&self) -> u64 {
        self.rows
    }
    pub const fn columns(&self) -> u64 {
        self.columns
    }
    pub const fn parameter_count(&self) -> u64 {
        self.parameter_count
    }
    pub const fn parameter_columns(&self) -> u64 {
        self.parameter_columns
    }
    pub const fn elements(&self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(&self) -> u64 {
        self.arithmetic
    }
    pub const fn shift_column(&self) -> u64 {
        self.shift_column
    }
    pub const fn scale_column(&self) -> u64 {
        self.scale_column
    }
    pub const fn gate_column(&self) -> u64 {
        self.gate_column
    }
    pub fn run_ends(&self) -> &[u64] {
        &self.run_ends
    }
    pub fn parameter_rows(&self) -> &[u64] {
        &self.parameter_rows
    }
    pub fn parameter_row(&self, row: u64) -> Option<u64> {
        if row >= self.rows {
            None
        } else {
            Some(self.parameter_rows[self.run_ends.partition_point(|end| *end <= row)])
        }
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("row-selected BF16: {reason}"),
    }
}

#[cfg(test)]
mod tests;
