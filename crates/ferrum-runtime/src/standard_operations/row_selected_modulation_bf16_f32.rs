//! BF16 activation promotion with row-selected, full-F32 scale and shift.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
pub const ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID: &str =
    "operation.row_selected_modulation.bf16_f32";
pub const ROW_SELECTED_MODULATION_BF16_F32_CAPABILITY_ID: &str =
    "capability.operation.row_selected_modulation.bf16_f32";
pub const ROW_SELECTED_MODULATION_MAX_RUNS: usize = 32;
/// X is BF16[R,C], parameters are F32[M,2C] with [shift | scale] per row;
/// output is F32[R,C]. Positive run_lengths cover exactly R rows and the
/// corresponding parameter_rows select each run's row in the original matrix.
/// Y=RN32(RN32(F32(X)*RN32(1+scale))+shift). Every operation is separate:
/// there is no FMA, BF16 parameter conversion, intermediate BF16 product,
/// sigmoid, normalization, gate or implicit selection by modality/model name.
/// IEEE overflow/nonfinite propagation applies; all inputs are original dense
/// storage and no expanded row-routing/parameter matrix is allocated.
pub fn row_selected_modulation_bf16_f32_contract() -> Result<StandardOperationContract, VNextError>
{
    let mut attributes = BTreeMap::new();
    for name in ["run_lengths", "parameter_rows"] {
        attributes.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: 1,
                    maximum: ROW_SELECTED_MODULATION_MAX_RUNS as u32,
                },
            },
        );
    }
    let matrix = vec![symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(matrix.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("parameter_count"), symbol("parameter_columns")],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            matrix,
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attributes)?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            ROW_SELECTED_MODULATION_BF16_F32_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSelectedModulationBf16F32Geometry {
    rows: u64,
    columns: u64,
    parameter_count: u64,
    elements: u64,
    arithmetic: u64,
    run_ends: Vec<u64>,
    parameter_rows: Vec<u64>,
}
impl RowSelectedModulationBf16F32Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        parameters: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        row_selected_modulation_bf16_f32_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        let [rows, columns] = input.dimensions() else {
            return Err(invalid("input must be rank two"));
        };
        let [parameter_count, parameter_columns] = parameters.dimensions() else {
            return Err(invalid("parameters must be rank two"));
        };
        if input.element_type() != ElementType::Bf16
            || parameters.element_type() != ElementType::F32
            || output.element_type() != ElementType::F32
            || [input, parameters, output].iter().any(|t| {
                t.layout() != &ResolvedTensorLayout::Contiguous || t.dimensions().contains(&0)
            })
            || output.dimensions() != input.dimensions()
            || columns.checked_mul(2) != Some(*parameter_columns)
        {
            return Err(invalid(
                "requires BF16 X[R,C], F32 parameters[M,2C], and F32 Y[R,C]",
            ));
        }
        for t in [input, parameters, output] {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let read = |name: &str| -> Result<&[i64], VNextError> {
            match attrs.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Integers(v)) => Ok(v),
                _ => Err(invalid("routing attributes must be integer lists")),
            }
        };
        let lengths = read("run_lengths")?;
        let indices = read("parameter_rows")?;
        if lengths.len() != indices.len()
            || !(1..=ROW_SELECTED_MODULATION_MAX_RUNS).contains(&lengths.len())
        {
            return Err(invalid("routing lists need equal lengths in 1..32"));
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
                .ok_or_else(|| invalid("parameter row is outside original matrix"))?;
            end = end
                .checked_add(length)
                .ok_or_else(|| invalid("routing row extent overflows"))?;
            run_ends.push(end);
            parameter_rows.push(index);
        }
        if end != *rows {
            return Err(invalid(
                "routing lengths must cover exactly the activation rows",
            ));
        }
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        let arithmetic = elements
            .checked_mul(3)
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            parameter_count: *parameter_count,
            elements,
            arithmetic,
            run_ends,
            parameter_rows,
        })
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
    pub const fn elements(&self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(&self) -> u64 {
        self.arithmetic
    }
    pub fn run_ends(&self) -> &[u64] {
        &self.run_ends
    }
    pub fn parameter_rows(&self) -> &[u64] {
        &self.parameter_rows
    }
    pub fn parameter_row(&self, row: u64) -> Option<u64> {
        if row >= self.rows {
            return None;
        }
        let run = self.run_ends.partition_point(|end| *end <= row);
        Some(self.parameter_rows[run])
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("row-selected BF16/F32 modulation: {reason}"),
    }
}
#[cfg(test)]
mod tests;
