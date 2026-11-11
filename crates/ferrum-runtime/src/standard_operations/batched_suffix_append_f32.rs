//! Exact F32 row transport with shared suffix rows and explicit positive zero.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const BATCHED_SUFFIX_APPEND_F32_OPERATION_ID: &str = "operation.batched_suffix_append.f32";
pub const BATCHED_SUFFIX_APPEND_F32_CAPABILITY_ID: &str =
    "capability.operation.batched_suffix_append.f32";
/// x[B*P,C], suffix[S,C] -> y[B*(P+S+Z),C]. For each batch the output is
/// [its P input rows | the same S suffix rows | Z positive-zero rows]. Copying
/// preserves every F32 bit, including signed zero, subnormals and NaN payloads.
/// No expanded suffix or scratch exists, and output cannot alias either input.
pub fn batched_suffix_append_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(BATCHED_SUFFIX_APPEND_F32_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![symbol("input_rows"), symbol("columns")],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("suffix_rows"), symbol("columns")],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![symbol("output_rows"), symbol("columns")],
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            (
                AttributeId::new("batch")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Unsigned,
                    required: true,
                    constraint: AttributeConstraint::UnsignedRange {
                        minimum: 1,
                        maximum: u64::MAX,
                    },
                },
            ),
            (
                AttributeId::new("zero_rows")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Unsigned,
                    required: true,
                    constraint: AttributeConstraint::UnsignedRange {
                        minimum: 0,
                        maximum: u64::MAX,
                    },
                },
            ),
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            BATCHED_SUFFIX_APPEND_F32_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedSuffixAppendF32Geometry {
    batch: u64,
    patch_rows: u64,
    suffix_rows: u64,
    zero_rows: u64,
    columns: u64,
    elements: u64,
}
impl BatchedSuffixAppendF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        suffix: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        batch: u64,
        zero_rows: u64,
    ) -> Result<Self, VNextError> {
        let [rows, columns] = x.dimensions() else {
            return Err(invalid("input must be rank two"));
        };
        let [suffix_rows, suffix_columns] = suffix.dimensions() else {
            return Err(invalid("suffix must be rank two"));
        };
        if batch == 0
            || *rows == 0
            || *columns == 0
            || *suffix_rows == 0
            || !rows.is_multiple_of(batch)
            || columns != suffix_columns
            || [x, suffix, y].iter().any(|t| {
                t.element_type() != ElementType::F32
                    || t.layout() != &ResolvedTensorLayout::Contiguous
            })
        {
            return Err(invalid(
                "requires positive F32 x[B*P,C] and suffix[S,C] with batch dividing input rows",
            ));
        }
        let patch_rows = rows / batch;
        let output_rows = patch_rows
            .checked_add(*suffix_rows)
            .and_then(|n| n.checked_add(zero_rows))
            .and_then(|n| n.checked_mul(batch))
            .ok_or_else(|| invalid("output rows overflow"))?;
        if y.dimensions() != [output_rows, *columns] {
            return Err(invalid(
                "output shape differs from exact per-batch concatenation",
            ));
        }
        for t in [x, suffix, y] {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let elements = output_rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("output element count overflows"))?;
        Ok(Self {
            batch,
            patch_rows,
            suffix_rows: *suffix_rows,
            zero_rows,
            columns: *columns,
            elements,
        })
    }
    pub const fn batch(self) -> u64 {
        self.batch
    }
    pub const fn patch_rows(self) -> u64 {
        self.patch_rows
    }
    pub const fn suffix_rows(self) -> u64 {
        self.suffix_rows
    }
    pub const fn zero_rows(self) -> u64 {
        self.zero_rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn rows_per_batch(self) -> u64 {
        self.patch_rows + self.suffix_rows + self.zero_rows
    }
    pub const fn input_elements(self) -> u64 {
        self.batch * self.patch_rows * self.columns
    }
    pub const fn suffix_elements(self) -> u64 {
        self.suffix_rows * self.columns
    }
}
fn invalid(s: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 batched suffix append: {s}"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: &[u64]) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            d.to_vec(),
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    #[test]
    fn suffix_broadcast_checks_exact_rows_dtype_and_overflow_without_requiring_padding() {
        let x = t(&[6, 5]);
        let s = t(&[2, 5]);
        let y = t(&[12, 5]);
        let g = BatchedSuffixAppendF32Geometry::new(&x, &s, &y, 2, 1).unwrap();
        assert_eq!(
            (g.patch_rows(), g.rows_per_batch(), g.elements()),
            (3, 6, 60)
        );
        assert!(BatchedSuffixAppendF32Geometry::new(&x, &s, &t(&[10, 5]), 2, 0).is_ok());
        assert!(BatchedSuffixAppendF32Geometry::new(&x, &s, &t(&[60, 1]), 2, 1).is_err());
        assert!(BatchedSuffixAppendF32Geometry::new(&x, &s, &y, 4, 1).is_err());
        assert!(BatchedSuffixAppendF32Geometry::new(&x, &s, &y, 0, 1).is_err());
        assert!(BatchedSuffixAppendF32Geometry::new(&x, &s, &y, 2, u64::MAX).is_err());
        let half = ResolvedTensorSpec::new(
            vec![2, 5],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        assert!(BatchedSuffixAppendF32Geometry::new(&x, &half, &y, 2, 1).is_err());
        assert_eq!(
            batched_suffix_append_f32_contract()
                .unwrap()
                .descriptor()
                .resources
                .scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
