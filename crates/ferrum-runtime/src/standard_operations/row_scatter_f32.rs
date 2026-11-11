//! Selected-row F32 replacement/addition with one shared, checked row map.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
pub const ROW_SCATTER_REPLACE_F32_OPERATION_ID: &str = "operation.row_scatter_replace.f32";
pub const ROW_SCATTER_REPLACE_F32_CAPABILITY_ID: &str =
    "capability.operation.row_scatter_replace.f32";
pub const ROW_SCATTER_ADD_F32_OPERATION_ID: &str = "operation.row_scatter_add.f32";
pub const ROW_SCATTER_ADD_F32_CAPABILITY_ID: &str = "capability.operation.row_scatter_add.f32";
/// Native launch metadata bound; 32 triples of u64 plus shape fit a compact
/// kernel parameter packet. Emit more than one node for a larger row map.
pub const ROW_SCATTER_F32_MAX_RUNS: usize = 32;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowScatterF32Op {
    Replace,
    Add,
}
impl RowScatterF32Op {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Replace => ROW_SCATTER_REPLACE_F32_OPERATION_ID,
            Self::Add => ROW_SCATTER_ADD_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Replace => ROW_SCATTER_REPLACE_F32_CAPABILITY_ID,
            Self::Add => ROW_SCATTER_ADD_F32_CAPABILITY_ID,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        contract(self)
    }
}
/// F32 X/Y[R,H], updates[U,H]. Concatenate ordered runs into U update rows.
/// Selected rows copy update bits; all other rows copy X bits. The replaced X
/// rows are not read or evaluated, including their nonfinite payloads.
pub fn row_scatter_replace_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(RowScatterF32Op::Replace)
}
/// Same map as replacement, but selected rows compute RN32(X+update). No
/// BF16/F16 conversion, FMA or multiplication. Unselected bits remain exact.
pub fn row_scatter_add_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(RowScatterF32Op::Add)
}
fn contract(kind: RowScatterF32Op) -> Result<StandardOperationContract, VNextError> {
    let mut attributes = BTreeMap::new();
    for name in ["destination_starts", "run_lengths"] {
        attributes.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: 1,
                    maximum: ROW_SCATTER_F32_MAX_RUNS as u32,
                },
            },
        );
    }
    let matrix = vec![symbol("rows"), symbol("columns")];
    let d = OperationDescriptor {
        id: OperationId::new(kind.operation_id())?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(matrix.clone(), [ElementType::F32], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("updates"), symbol("columns")],
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
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(kind.capability_id(), ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    d.validate()?;
    Ok(StandardOperationContract { descriptor: d })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowScatterF32Run {
    pub destination_start: u64,
    pub source_start: u64,
    pub length: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowScatterF32Geometry {
    rows: u64,
    columns: u64,
    updates: u64,
    elements: u64,
    runs: Vec<RowScatterF32Run>,
}
impl RowScatterF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        updates: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        row_scatter_add_f32_contract()?
            .descriptor()
            .validate_attributes(attributes)?;
        let [rows, columns] = x.dimensions() else {
            return Err(invalid("base must be rank two"));
        };
        if x != y
            || [x, updates].iter().any(|t| {
                t.dimensions().len() != 2
                    || t.dimensions().contains(&0)
                    || t.element_type() != ElementType::F32
                    || t.layout() != &ResolvedTensorLayout::Contiguous
            })
            || updates.dimensions()[1] != *columns
        {
            return Err(invalid(
                "requires exact contiguous F32 base/output and same-width updates",
            ));
        }
        for t in [x, updates, y] {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds Host address range"));
            }
        }
        let list = |name: &str| -> Result<&[i64], VNextError> {
            match attributes.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Integers(v)) => Ok(v),
                _ => Err(invalid("missing typed row-map attribute")),
            }
        };
        let starts = list("destination_starts")?;
        let lengths = list("run_lengths")?;
        if starts.len() != lengths.len() {
            return Err(invalid("run list extents differ"));
        }
        let mut runs = Vec::with_capacity(starts.len());
        let mut previous = 0;
        let mut source = 0u64;
        for (&start, &length) in starts.iter().zip(lengths) {
            let start = u64::try_from(start).map_err(|_| invalid("negative row start"))?;
            let length = u64::try_from(length).map_err(|_| invalid("negative run length"))?;
            let end = start
                .checked_add(length)
                .filter(|&n| length > 0 && n <= *rows && start >= previous)
                .ok_or_else(|| invalid("runs must be nonempty, ordered, disjoint and in bounds"))?;
            runs.push(RowScatterF32Run {
                destination_start: start,
                source_start: source,
                length,
            });
            source = source
                .checked_add(length)
                .ok_or_else(|| invalid("update extent overflows"))?;
            previous = end;
        }
        if source != updates.dimensions()[0] {
            return Err(invalid("runs must consume every update row exactly once"));
        }
        Ok(Self {
            rows: *rows,
            columns: *columns,
            updates: source,
            elements: rows
                .checked_mul(*columns)
                .ok_or_else(|| invalid("element count overflows"))?,
            runs,
        })
    }
    pub const fn rows(&self) -> u64 {
        self.rows
    }
    pub const fn columns(&self) -> u64 {
        self.columns
    }
    pub const fn update_rows(&self) -> u64 {
        self.updates
    }
    pub const fn elements(&self) -> u64 {
        self.elements
    }
    pub fn runs(&self) -> &[RowScatterF32Run] {
        &self.runs
    }
    pub fn update_row(&self, row: u64) -> Option<u64> {
        if row >= self.rows {
            return None;
        }
        let i = self
            .runs
            .partition_point(|r| r.destination_start <= row)
            .checked_sub(1)?;
        let r = self.runs[i];
        (row < r.destination_start + r.length).then(|| r.source_start + row - r.destination_start)
    }
    pub fn logical_operations(&self, kind: RowScatterF32Op) -> u64 {
        match kind {
            RowScatterF32Op::Replace => 0,
            RowScatterF32Op::Add => self.updates * self.columns,
        }
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 row scatter: {reason}"),
    }
}

#[cfg(test)]
mod tests;
