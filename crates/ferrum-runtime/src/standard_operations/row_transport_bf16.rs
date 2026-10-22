//! Exact BF16 bit transport over explicitly described, bounded row mappings.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const ROW_GATHER_BF16_OPERATION_ID: &str = "operation.row_gather.bf16";
pub const ROW_GATHER_BF16_CAPABILITY_ID: &str = "capability.operation.row_gather.bf16";
pub const BATCHED_ROW_CONCAT_BF16_OPERATION_ID: &str = "operation.batched_row_concat.bf16";
pub const BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID: &str =
    "capability.operation.batched_row_concat.bf16";
pub const ROW_GATHER_BF16_MAX_RUNS: usize = 32;

/// Copies a column window from explicit row runs, without numerical conversion.
/// Run k at batch b reads source_start[k]+b*batch_stride[k]+j*row_stride[k]
/// for j in 0..run_length[k], concatenating runs within each output batch.
/// Row strides are 0 (repeat) or 1 (consecutive). No model roles are inferred.
pub fn row_gather_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut attributes = BTreeMap::from([
        unsigned_attribute("batch")?,
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
        unsigned_attribute("column_count")?,
    ]);
    for name in [
        "run_lengths",
        "source_starts",
        "source_batch_strides",
        "source_row_strides",
    ] {
        attributes.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: 1,
                    maximum: ROW_GATHER_BF16_MAX_RUNS as u32,
                },
            },
        );
    }
    transport_contract(
        ROW_GATHER_BF16_OPERATION_ID,
        ROW_GATHER_BF16_CAPABILITY_ID,
        vec![vec![symbol("source_rows"), symbol("source_columns")]],
        vec![symbol("output_rows"), symbol("output_columns")],
        attributes,
    )
}

/// Concatenates left then right rows separately for each batch. This differs
/// from concatenating the two flattened matrices when batch is greater than one.
pub fn batched_row_concat_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    transport_contract(
        BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
        BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID,
        vec![
            vec![symbol("left_rows"), symbol("columns")],
            vec![symbol("right_rows"), symbol("columns")],
        ],
        vec![symbol("output_rows"), symbol("columns")],
        BTreeMap::from([unsigned_attribute("batch")?]),
    )
}

fn transport_contract(
    operation: &str,
    capability: &str,
    inputs: Vec<Vec<DimensionConstraint>>,
    output: Vec<DimensionConstraint>,
    attributes: BTreeMap<AttributeId, AttributeSpec>,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation)?,
        version: ContractVersion::new(1, 0),
        inputs: inputs
            .into_iter()
            .map(|dims| contiguous_tensor(dims, [ElementType::Bf16], TensorAccess::Read))
            .collect::<Result<_, _>>()?,
        outputs: vec![contiguous_tensor(
            output,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attributes)?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(capability, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Validated metadata only; it confers no allocation or submission authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowGatherBf16Geometry {
    batch: u64,
    source_rows: u64,
    source_width: u64,
    rows_per_batch: u64,
    column_start: u64,
    column_count: u64,
    elements: u64,
    run_lengths: Vec<u64>,
    source_starts: Vec<u64>,
    source_batch_strides: Vec<u64>,
    source_row_strides: Vec<u64>,
}

impl RowGatherBf16Geometry {
    pub fn new(
        source: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        row_gather_bf16_contract()?
            .descriptor
            .validate_attributes(attributes)?;
        let (source_rows, source_width) = matrix(source)?;
        let (output_rows, output_columns) = matrix(output)?;
        let batch = unsigned(attributes, "batch")?;
        let column_start = unsigned(attributes, "column_start")?;
        let column_count = unsigned(attributes, "column_count")?;
        if batch == 0
            || column_count == 0
            || column_start
                .checked_add(column_count)
                .is_none_or(|end| end > source_width)
        {
            return Err(invalid(
                "batch/column_count must be positive and the column window must fit source width",
            ));
        }
        let run_lengths = integers(attributes, "run_lengths")?;
        let source_starts = integers(attributes, "source_starts")?;
        let source_batch_strides = integers(attributes, "source_batch_strides")?;
        let source_row_strides = integers(attributes, "source_row_strides")?;
        let count = run_lengths.len();
        if !(1..=ROW_GATHER_BF16_MAX_RUNS).contains(&count)
            || [
                source_starts.len(),
                source_batch_strides.len(),
                source_row_strides.len(),
            ]
            .into_iter()
            .any(|n| n != count)
        {
            return Err(invalid(
                "all four run arrays must have the same length in 1..=32",
            ));
        }
        let mut rows_per_batch = 0_u64;
        for index in 0..count {
            let length = run_lengths[index];
            if length == 0 || source_row_strides[index] > 1 {
                return Err(invalid(
                    "run lengths must be positive and source row stride must be 0 or 1",
                ));
            }
            rows_per_batch = rows_per_batch
                .checked_add(length)
                .ok_or_else(|| invalid("sum of run lengths overflows"))?;
            let last = (batch - 1)
                .checked_mul(source_batch_strides[index])
                .and_then(|n| n.checked_add(source_starts[index]))
                .and_then(|n| {
                    (length - 1)
                        .checked_mul(source_row_strides[index])
                        .and_then(|delta| n.checked_add(delta))
                })
                .ok_or_else(|| invalid("source row index overflows"))?;
            if last >= source_rows {
                return Err(invalid("source row run exceeds the actual source tensor"));
            }
        }
        let rows = batch
            .checked_mul(rows_per_batch)
            .ok_or_else(|| invalid("output row count overflows"))?;
        if (output_rows, output_columns) != (rows, column_count) {
            return Err(invalid(
                "output shape does not equal [batch*sum(run_lengths),column_count]",
            ));
        }
        let elements = rows
            .checked_mul(column_count)
            .ok_or_else(|| invalid("output element count overflows"))?;
        Ok(Self {
            batch,
            source_rows,
            source_width,
            rows_per_batch,
            column_start,
            column_count,
            elements,
            run_lengths,
            source_starts,
            source_batch_strides,
            source_row_strides,
        })
    }
    pub const fn batch(&self) -> u64 {
        self.batch
    }
    pub const fn source_rows(&self) -> u64 {
        self.source_rows
    }
    pub const fn source_width(&self) -> u64 {
        self.source_width
    }
    pub const fn rows_per_batch(&self) -> u64 {
        self.rows_per_batch
    }
    pub const fn column_start(&self) -> u64 {
        self.column_start
    }
    pub const fn column_count(&self) -> u64 {
        self.column_count
    }
    pub const fn element_count(&self) -> u64 {
        self.elements
    }
    pub fn run_lengths(&self) -> &[u64] {
        &self.run_lengths
    }
    pub fn source_starts(&self) -> &[u64] {
        &self.source_starts
    }
    pub fn source_batch_strides(&self) -> &[u64] {
        &self.source_batch_strides
    }
    pub fn source_row_strides(&self) -> &[u64] {
        &self.source_row_strides
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedRowConcatBf16Geometry {
    batch: u64,
    left_rows_per_batch: u64,
    right_rows_per_batch: u64,
    columns: u64,
    elements: u64,
}
impl BatchedRowConcatBf16Geometry {
    pub fn new(
        left: &ResolvedTensorSpec,
        right: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        batch: u64,
    ) -> Result<Self, VNextError> {
        let (left_rows, columns) = matrix(left)?;
        let (right_rows, right_columns) = matrix(right)?;
        let (output_rows, output_columns) = matrix(output)?;
        if batch == 0
            || left_rows % batch != 0
            || right_rows % batch != 0
            || right_columns != columns
            || output_columns != columns
        {
            return Err(invalid(
                "concat requires positive batch, batch-divisible row counts and matching columns",
            ));
        }
        let expected = left_rows
            .checked_add(right_rows)
            .ok_or_else(|| invalid("concat output rows overflow"))?;
        if output_rows != expected {
            return Err(invalid("concat output row count differs from left+right"));
        }
        let elements = expected
            .checked_mul(columns)
            .ok_or_else(|| invalid("concat output element count overflows"))?;
        Ok(Self {
            batch,
            left_rows_per_batch: left_rows / batch,
            right_rows_per_batch: right_rows / batch,
            columns,
            elements,
        })
    }
    pub const fn batch(self) -> u64 {
        self.batch
    }
    pub const fn left_rows_per_batch(self) -> u64 {
        self.left_rows_per_batch
    }
    pub const fn right_rows_per_batch(self) -> u64 {
        self.right_rows_per_batch
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
}

fn matrix(tensor: &ResolvedTensorSpec) -> Result<(u64, u64), VNextError> {
    let [rows, columns] = tensor.dimensions() else {
        return Err(invalid("transport requires rank-two tensors"));
    };
    if *rows == 0
        || *columns == 0
        || tensor.element_type() != ElementType::Bf16
        || tensor.layout() != &ResolvedTensorLayout::Contiguous
    {
        return Err(invalid(
            "transport requires positive contiguous BF16 matrices",
        ));
    }
    tensor.minimum_storage_bytes()?;
    Ok((*rows, *columns))
}
fn unsigned(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, VNextError> {
    match attributes
        .iter()
        .find(|(id, _)| id.as_str() == name)
        .map(|(_, v)| v)
    {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(invalid(format!("{name} must be unsigned"))),
    }
}
fn integers(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<Vec<u64>, VNextError> {
    match attributes
        .iter()
        .find(|(id, _)| id.as_str() == name)
        .map(|(_, v)| v)
    {
        Some(SemanticValue::Integers(values)) => values
            .iter()
            .map(|v| {
                u64::try_from(*v).map_err(|_| invalid(format!("{name} contains a negative value")))
            })
            .collect(),
        _ => Err(invalid(format!("{name} must be an integer list"))),
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 row transport: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
