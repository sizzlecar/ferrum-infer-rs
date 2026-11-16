//! Exact replacement of one contiguous row range in an initialized tensor.
//! The output is the same full backing as the base input. Untouched rows keep
//! their original bits; updates must have disjoint storage. This is ordinary
//! explicit mutation, not a partial-initialization or allocation escape hatch.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const ROW_WRITE_BF16_OPERATION_ID: &str = "operation.row_write.bf16";
pub const ROW_WRITE_BF16_CAPABILITY_ID: &str = "capability.operation.row_write.bf16";

pub fn row_write_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let matrix = vec![symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROW_WRITE_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(matrix.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("update_rows"), symbol("columns")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor_with_alias(
            matrix,
            [ElementType::Bf16],
            TensorAccess::Write,
            AliasPolicy::MustAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([(
            AttributeId::new("row_start")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Unsigned,
                required: true,
                constraint: AttributeConstraint::UnsignedRange {
                    minimum: 0,
                    maximum: u64::MAX,
                },
            },
        )]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(ROW_WRITE_BF16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowWriteBf16Geometry {
    base_elements: u64,
    update_elements: u64,
    offset_elements: u64,
}
impl RowWriteBf16Geometry {
    pub fn new(
        base: &ResolvedTensorSpec,
        updates: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        row_start: u64,
    ) -> Result<Self, VNextError> {
        let invalid = || {
            VNextError::InvalidExecutionPlan {
            reason: "BF16 row write requires equal base/output, contiguous matrices, and an in-bounds row range".into(),
        }
        };
        if base != output
            || [base, updates].iter().any(|t| {
                t.element_type() != ElementType::Bf16
                    || t.layout() != &ResolvedTensorLayout::Contiguous
                    || t.dimensions().len() != 2
                    || t.dimensions().contains(&0)
            })
        {
            return Err(invalid());
        }
        let [rows, columns] = base.dimensions() else {
            unreachable!()
        };
        let update_rows = updates.dimensions()[0];
        if updates.dimensions()[1] != *columns
            || row_start
                .checked_add(update_rows)
                .is_none_or(|end| end > *rows)
        {
            return Err(invalid());
        }
        let base_elements = rows.checked_mul(*columns).ok_or_else(invalid)?;
        if base_elements > isize::MAX as u64 / 2 {
            return Err(invalid());
        }
        Ok(Self {
            base_elements,
            update_elements: update_rows * columns,
            offset_elements: row_start * columns,
        })
    }
    pub fn base_elements(self) -> u64 {
        self.base_elements
    }
    pub fn update_elements(self) -> u64 {
        self.update_elements
    }
    pub fn offset_elements(self) -> u64 {
        self.offset_elements
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SemanticValue;
    fn tensor(rows: u64, columns: u64) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            vec![rows, columns],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    #[test]
    fn tail_range_is_checked_without_relaxing_full_backing_alias() {
        let base = tensor(9, 5);
        let tail = tensor(1, 5);
        let g = RowWriteBf16Geometry::new(&base, &tail, &base, 8).unwrap();
        assert_eq!(
            (g.base_elements(), g.update_elements(), g.offset_elements()),
            (45, 5, 40)
        );
        for start in [9, u64::MAX] {
            assert!(RowWriteBf16Geometry::new(&base, &tail, &base, start).is_err());
        }
        assert!(RowWriteBf16Geometry::new(&base, &tensor(1, 6), &base, 0).is_err());
        let contract = row_write_bf16_contract().unwrap();
        for row_start in [0, 8] {
            contract
                .descriptor()
                .validate_attributes(&BTreeMap::from([(
                    AttributeId::new("row_start").unwrap(),
                    SemanticValue::Unsigned(row_start),
                )]))
                .unwrap();
            assert!(RowWriteBf16Geometry::new(&base, &tail, &base, row_start).is_ok());
        }
        assert_eq!(
            contract.descriptor().outputs[0].alias(),
            &AliasPolicy::MustAlias { tensor_index: 0 }
        );
    }
}
