//! Explicit, non-aliasing rank bridges between per-row matrix operations and
//! batched sequence operations. No permutation, arithmetic, or implicit view.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const MERGE_LEADING_BATCH_BF16_OPERATION_ID: &str = "operation.merge_leading_batch.bf16";
pub const MERGE_LEADING_BATCH_BF16_CAPABILITY_ID: &str =
    "capability.operation.merge_leading_batch.bf16";
pub const SPLIT_LEADING_BATCH_BF16_OPERATION_ID: &str = "operation.split_leading_batch.bf16";
pub const SPLIT_LEADING_BATCH_BF16_CAPABILITY_ID: &str =
    "capability.operation.split_leading_batch.bf16";

pub const MERGE_LEADING_BATCH_BF16_VIEW_OPERATION_ID: &str =
    "operation.view.merge_leading_batch.bf16";
pub const SPLIT_LEADING_BATCH_BF16_VIEW_OPERATION_ID: &str =
    "operation.view.split_leading_batch.bf16";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeadingBatchBf16Kind {
    Merge,
    Split,
}
impl LeadingBatchBf16Kind {
    pub fn view_operation_id(self) -> &'static str {
        match self {
            Self::Merge => MERGE_LEADING_BATCH_BF16_VIEW_OPERATION_ID,
            Self::Split => SPLIT_LEADING_BATCH_BF16_VIEW_OPERATION_ID,
        }
    }
    pub fn view_capability_id(self) -> &'static str {
        match self {
            Self::Merge => "capability.operation.view.merge_leading_batch.bf16",
            Self::Split => "capability.operation.view.split_leading_batch.bf16",
        }
    }
    /// Explicit opt-in to a read-only view. The copy contract remains unchanged.
    pub fn view_contract(self) -> Result<StandardOperationContract, VNextError> {
        let mut contract = self.contract()?;
        contract.descriptor.id = OperationId::new(self.view_operation_id())?;
        contract.descriptor.provider =
            provider_requirement(self.view_capability_id(), ContractVersion::new(1, 0))?;
        let output = &contract.descriptor.outputs[0];
        contract.descriptor.outputs[0] = TensorContract::new(
            output.dimensions().to_vec(),
            output.element_types().clone(),
            output.layouts().to_vec(),
            TensorAccess::Read,
            AliasPolicy::ReadOnlyView { tensor_index: 0 },
        )?;
        contract.descriptor.validate()?;
        Ok(contract)
    }
    pub fn operation_id(self) -> &'static str {
        match self {
            Self::Merge => MERGE_LEADING_BATCH_BF16_OPERATION_ID,
            Self::Split => SPLIT_LEADING_BATCH_BF16_OPERATION_ID,
        }
    }
    pub fn capability_id(self) -> &'static str {
        match self {
            Self::Merge => MERGE_LEADING_BATCH_BF16_CAPABILITY_ID,
            Self::Split => SPLIT_LEADING_BATCH_BF16_CAPABILITY_ID,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let matrix = vec![symbol("flat_rows"), symbol("columns")];
        let batched = vec![symbol("batch"), symbol("rows"), symbol("columns")];
        let (input, output) = match self {
            Self::Merge => (batched, matrix),
            Self::Split => (matrix, batched),
        };
        let descriptor = OperationDescriptor {
            id: OperationId::new(self.operation_id())?,
            version: ContractVersion::new(1, 0),
            inputs: vec![contiguous_tensor(
                input,
                [ElementType::Bf16],
                TensorAccess::Read,
            )?],
            outputs: vec![contiguous_tensor(
                output,
                [ElementType::Bf16],
                TensorAccess::Write,
            )?],
            attributes: AttributeSchema::new(BTreeMap::new())?,
            resources: no_auxiliary_resources(),
            oracle: OracleSpec::OperationDefined,
            provider: provider_requirement(self.capability_id(), ContractVersion::new(1, 0))?,
            profile_phase: ProfilePhase::Forward,
        };
        descriptor.validate()?;
        Ok(StandardOperationContract { descriptor })
    }
}

/// Merge [B,S,C] -> [B*S,C], or its explicit inverse. Shape equality alone
/// cannot turn this into a view: the output has NoAlias and its bytes are owned
/// by the ordinary output lifetime. All 65536 BF16 encodings are transported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeadingBatchBf16Geometry {
    batches: u64,
    rows: u64,
    columns: u64,
}
impl LeadingBatchBf16Geometry {
    pub fn new(
        kind: LeadingBatchBf16Kind,
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let (batched, matrix) = match kind {
            LeadingBatchBf16Kind::Merge => (input, output),
            LeadingBatchBf16Kind::Split => (output, input),
        };
        let [b, s, c] = batched.dimensions() else {
            return Err(invalid("batched side requires rank three"));
        };
        let flat = b
            .checked_mul(*s)
            .ok_or_else(|| invalid("batch row product overflows"))?;
        if [*b, *s, *c].contains(&0) || matrix.dimensions() != [flat, *c] {
            return Err(invalid("requires matching positive [B,S,C] and [B*S,C]"));
        }
        for t in [input, output] {
            if t.element_type() != ElementType::Bf16
                || t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid("requires contiguous addressable BF16 extents"));
            }
        }
        Ok(Self {
            batches: *b,
            rows: *s,
            columns: *c,
        })
    }
    pub fn flat_rows(self) -> u64 {
        self.batches * self.rows
    }
    pub fn columns(self) -> u64 {
        self.columns
    }
    pub fn elements(self) -> u64 {
        self.flat_rows() * self.columns
    }
    pub fn bytes(self) -> u64 {
        self.elements() * 2
    }
}
fn invalid(s: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 leading batch: {s}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: Vec<u64>) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(d, ElementType::Bf16, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn leading_batch_view_is_explicit_and_preserves_the_copy_contract() {
        for kind in [LeadingBatchBf16Kind::Merge, LeadingBatchBf16Kind::Split] {
            let copy = kind.contract().unwrap();
            let view = kind.view_contract().unwrap();
            assert_ne!(copy.descriptor.id, view.descriptor.id);
            assert_eq!(copy.descriptor.outputs[0].alias(), &AliasPolicy::NoAlias);
            assert_eq!(view.descriptor.outputs[0].access(), TensorAccess::Read);
            assert_eq!(
                view.descriptor.outputs[0].alias(),
                &AliasPolicy::ReadOnlyView { tensor_index: 0 }
            );
            let input = crate::ResolvedValueBinding::new(
                crate::ProgramValueId::new("value/source").unwrap(),
                crate::ResolvedValueRole::Input,
                0,
                if kind == LeadingBatchBf16Kind::Merge {
                    t(vec![2, 3, 5])
                } else {
                    t(vec![6, 5])
                },
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                crate::BufferUsage::Activations,
                None,
                crate::ResolvedValueStorage::single(
                    crate::ResourceId::new("resource/shared").unwrap(),
                    0,
                    60,
                    ElementType::Bf16,
                )
                .unwrap(),
            )
            .unwrap();
            let output = crate::ResolvedValueBinding::new(
                crate::ProgramValueId::new("value/view").unwrap(),
                crate::ResolvedValueRole::Output,
                0,
                if kind == LeadingBatchBf16Kind::Merge {
                    t(vec![6, 5])
                } else {
                    t(vec![2, 3, 5])
                },
                TensorAccess::Read,
                AliasPolicy::ReadOnlyView { tensor_index: 0 },
                crate::BufferUsage::Activations,
                None,
                input.storage().clone(),
            )
            .unwrap();
            view.descriptor
                .validate_resolved_bindings(&[input.clone(), output.clone()])
                .unwrap();
            // Same declared axes but fewer bytes must not create a truncated view.
            let mut dimensions = output.tensor().dimensions().to_vec();
            dimensions[0] -= 1;
            let shortened = crate::ResolvedValueBinding::new(
                output.value_id().clone(),
                output.role(),
                0,
                t(dimensions),
                output.access(),
                output.alias().clone(),
                output.usage(),
                None,
                output.storage().clone(),
            )
            .unwrap();
            assert!(view
                .descriptor
                .validate_resolved_bindings(&[input, shortened])
                .is_err());
        }
    }
    #[test]
    fn leading_batch_checks_axis_order_product_and_real_addressability() {
        let b = t(vec![2, 3, 5]);
        let m = t(vec![6, 5]);
        for kind in [LeadingBatchBf16Kind::Merge, LeadingBatchBf16Kind::Split] {
            let (x, y) = if kind == LeadingBatchBf16Kind::Merge {
                (&b, &m)
            } else {
                (&m, &b)
            };
            assert_eq!(
                LeadingBatchBf16Geometry::new(kind, x, y).unwrap().bytes(),
                60
            );
            assert!(LeadingBatchBf16Geometry::new(kind, y, x).is_err());
            assert_eq!(
                kind.contract().unwrap().descriptor().outputs[0].alias(),
                &AliasPolicy::NoAlias
            );
        }
        assert!(
            LeadingBatchBf16Geometry::new(LeadingBatchBf16Kind::Merge, &b, &t(vec![10, 3]))
                .is_err()
        );
        assert!(LeadingBatchBf16Geometry::new(
            LeadingBatchBf16Kind::Merge,
            &t(vec![2, 3, 1 << 60]),
            &t(vec![6, 1 << 60])
        )
        .is_err());
    }
}
