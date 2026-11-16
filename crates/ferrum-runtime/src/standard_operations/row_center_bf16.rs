//! Column-wise sequence centering with an explicit BF16 mean boundary.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
pub const ROW_CENTER_BF16_OPERATION_ID: &str = "operation.row_center.bf16";
pub const ROW_CENTER_BF16_CAPABILITY_ID: &str = "capability.operation.row_center.bf16";
pub const ROW_CENTER_BF16_PROFILE: &str = "f32_stride8_tree_mean_bf16_subtract_rn_v1";

/// BF16 X[B,S,C] -> centered Y[B,S,C], mean M[B,1,C]. Every source row,
/// including padding, participates; batches and columns are independent.
/// Eight F32 accumulators start at +0 and visit r=lane, lane+8,... in order;
/// combine lanes by descending strides 4,2,1. Multiply the F32 sum by
/// F32(B*C)/F32(B*S*C), then RNE-convert to BF16. Y is the RNE BF16 result
/// of F32(X)-F32(M). There is no reassociation across either BF16 boundary.
/// All F32 operations use RN and gradual underflow. NaNs propagate, payloads
/// are unspecified. This explicit reduction profile does not claim Torch
/// reduction-tree identity. The two outputs are normal admitted tensors;
/// there is no hidden scratch allocator or host readback of the mean.
pub fn row_center_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let shape = || vec![symbol("batch"), symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROW_CENTER_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            shape(),
            [ElementType::Bf16],
            TensorAccess::Read,
        )?],
        outputs: vec![
            contiguous_tensor(shape(), [ElementType::Bf16], TensorAccess::Write)?,
            contiguous_tensor(
                vec![
                    symbol("batch"),
                    DimensionConstraint::Exact(1),
                    symbol("columns"),
                ],
                [ElementType::Bf16],
                TensorAccess::Write,
            )?,
        ],
        attributes: AttributeSchema::new(BTreeMap::from([(
            AttributeId::new("profile")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Text,
                required: true,
                constraint: AttributeConstraint::TextChoices {
                    values: BTreeSet::from([ROW_CENTER_BF16_PROFILE.to_owned()]),
                },
            },
        )]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(ROW_CENTER_BF16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowCenterBf16Geometry {
    batches: u64,
    rows: u64,
    columns: u64,
    arithmetic: u64,
}
impl RowCenterBf16Geometry {
    pub fn attributes() -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([(
            AttributeId::new("profile")?,
            SemanticValue::Text(ROW_CENTER_BF16_PROFILE.into()),
        )]))
    }
    pub fn new(
        x: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        mean: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let [b, s, c] = x.dimensions() else {
            return Err(invalid("input requires rank three"));
        };
        if [*b, *s, *c].contains(&0)
            || y != x
            || mean.dimensions() != [*b, 1, *c]
            || attrs != &Self::attributes()?
        {
            return Err(invalid(
                "requires positive equal X/Y[B,S,C], mean[B,1,C] and declared arithmetic",
            ));
        }
        for t in [x, y, mean] {
            if t.element_type() != ElementType::Bf16
                || t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid("requires contiguous addressable BF16 extents"));
            }
        }
        let elements = x.minimum_storage_bytes()? / 2;
        let means = mean.minimum_storage_bytes()? / 2;
        let arithmetic = elements
            .checked_mul(2)
            .and_then(|n| means.checked_mul(8).and_then(|m| n.checked_add(m)))
            .ok_or_else(|| invalid("arithmetic extent overflows"))?;
        Ok(Self {
            batches: *b,
            rows: *s,
            columns: *c,
            arithmetic,
        })
    }
    pub const fn batches(self) -> u64 {
        self.batches
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub fn elements(self) -> u64 {
        self.batches * self.rows * self.columns
    }
    pub fn mean_elements(self) -> u64 {
        self.batches * self.columns
    }
    pub fn data_bytes(self) -> u64 {
        self.elements() * 2
    }
    pub fn mean_bytes(self) -> u64 {
        self.mean_elements() * 2
    }
    pub fn factor(self) -> f32 {
        self.mean_elements() as f32 / self.elements() as f32
    }
    pub const fn arithmetic(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(s: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 row centering: {s}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: Vec<u64>, ty: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(d, ty, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn centering_rejects_hidden_axis_reinterpretation_dtype_alias_and_capacity() {
        let x = t(vec![2, 7, 3], ElementType::Bf16);
        let m = t(vec![2, 1, 3], ElementType::Bf16);
        let attrs = RowCenterBf16Geometry::attributes().unwrap();
        let g = RowCenterBf16Geometry::new(&x, &x, &m, &attrs).unwrap();
        assert_eq!(
            (g.data_bytes(), g.mean_bytes(), g.arithmetic()),
            (84, 12, 132)
        );
        assert_eq!(g.factor().to_bits(), (6f32 / 42.).to_bits());
        assert!(
            RowCenterBf16Geometry::new(&x, &t(vec![2, 3, 7], ElementType::Bf16), &m, &attrs)
                .is_err()
        );
        assert!(
            RowCenterBf16Geometry::new(&x, &x, &t(vec![2, 1, 3], ElementType::F32), &attrs)
                .is_err()
        );
        assert!(RowCenterBf16Geometry::new(&x, &x, &m, &BTreeMap::new()).is_err());
        let huge = t(vec![2, 3, 1 << 60], ElementType::Bf16);
        assert!(RowCenterBf16Geometry::new(
            &huge,
            &huge,
            &t(vec![2, 1, 1 << 60], ElementType::Bf16),
            &attrs
        )
        .is_err());
        for output in row_center_bf16_contract()
            .unwrap()
            .descriptor()
            .outputs
            .iter()
        {
            assert_eq!(output.alias(), &AliasPolicy::NoAlias);
        }
    }
}
