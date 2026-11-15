//! Exact bit transport with explicit, total out-of-range semantics.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID: &str =
    "operation.row_index_gather_or_zero.bf16";
pub const ROW_INDEX_GATHER_OR_ZERO_BF16_CAPABILITY_ID: &str =
    "capability.operation.row_index_gather_or_zero.bf16";

/// BF16 source [B,S,C], I32 indices [R] -> BF16 [B,R,C]. The same row map
/// applies independently to each batch. Indices outside [0,S) produce positive
/// zero; valid indices preserve all source bits, including NaN payloads. This
/// total transport contract does not certify a model's permutation or padding.
/// That proof belongs to the checked model layout producing the index tensor.
pub fn row_index_gather_or_zero_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![symbol("b"), symbol("s"), symbol("c")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(vec![symbol("r")], [ElementType::I32], TensorAccess::Read)?,
        ],
        outputs: vec![contiguous_tensor(
            vec![symbol("b"), symbol("r"), symbol("c")],
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::new())?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            ROW_INDEX_GATHER_OR_ZERO_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowIndexGatherBf16Geometry {
    batches: u64,
    source_rows: u64,
    output_rows: u64,
    columns: u64,
}
impl RowIndexGatherBf16Geometry {
    pub fn new(
        source: &ResolvedTensorSpec,
        indices: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [b, s, c] = source.dimensions() else {
            return Err(invalid("source must be rank three"));
        };
        let [r] = indices.dimensions() else {
            return Err(invalid("indices must be rank one"));
        };
        if [b, s, c, r].iter().any(|&&v| v == 0)
            || *s > i32::MAX as u64
            || source.element_type() != ElementType::Bf16
            || indices.element_type() != ElementType::I32
            || output.element_type() != ElementType::Bf16
            || output.dimensions() != [*b, *r, *c]
        {
            return Err(invalid(
                "requires positive BF16 [B,S,C], I32 [R], BF16 [B,R,C] and I32-addressable S",
            ));
        }
        for t in [source, indices, output] {
            if t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid("requires contiguous addressable extents"));
            }
        }
        Ok(Self {
            batches: *b,
            source_rows: *s,
            output_rows: *r,
            columns: *c,
        })
    }
    pub const fn batches(self) -> u64 {
        self.batches
    }
    pub const fn source_rows(self) -> u64 {
        self.source_rows
    }
    pub const fn output_rows(self) -> u64 {
        self.output_rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub fn source_bytes(self) -> u64 {
        self.batches * self.source_rows * self.columns * 2
    }
    pub fn index_bytes(self) -> u64 {
        self.output_rows * 4
    }
    pub fn output_bytes(self) -> u64 {
        self.element_count() * 2
    }
    pub fn element_count(self) -> u64 {
        self.batches * self.output_rows * self.columns
    }
}
fn invalid(s: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("indexed BF16 row gather: {s}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], ty: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), ty, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn indexed_rows_require_matching_batch_columns_integer_map_and_addressable_storage() {
        let x = t(&[2, 5, 3], ElementType::Bf16);
        let i = t(&[7], ElementType::I32);
        let y = t(&[2, 7, 3], ElementType::Bf16);
        let g = RowIndexGatherBf16Geometry::new(&x, &i, &y).unwrap();
        assert_eq!(
            (g.source_bytes(), g.index_bytes(), g.output_bytes()),
            (60, 28, 84)
        );
        for (x, i, y) in [
            (x.clone(), t(&[7], ElementType::F32), y.clone()),
            (x.clone(), i.clone(), t(&[7, 2, 3], ElementType::Bf16)),
            (
                t(&[2, i32::MAX as u64 + 1, 3], ElementType::Bf16),
                i.clone(),
                y.clone(),
            ),
            (
                t(&[2, 5, 1 << 59], ElementType::Bf16),
                i.clone(),
                t(&[2, 7, 1 << 59], ElementType::Bf16),
            ),
        ] {
            assert!(RowIndexGatherBf16Geometry::new(&x, &i, &y).is_err());
        }
        let contract = row_index_gather_or_zero_bf16_contract().unwrap();
        assert_eq!(
            contract.descriptor().outputs[0].alias(),
            &AliasPolicy::NoAlias
        );
    }
}
