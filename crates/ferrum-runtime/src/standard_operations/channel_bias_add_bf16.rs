//! Channel bias addition with explicit BF16 matrix and vector ports.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const CHANNEL_BIAS_ADD_BF16_OPERATION_ID: &str = "operation.channel_bias_add.bf16";
pub const CHANNEL_BIAS_ADD_BF16_CAPABILITY_ID: &str = "capability.operation.channel_bias_add.bf16";

/// X/Y BF16[R,C], bias BF16[C]. Y[r,c] = B(RN32(X[r,c]+bias[c])),
/// where B is BF16 round-to-nearest-even. Gradual underflow and IEEE
/// nonfinite propagation apply; NaN payloads are unspecified. No scalar,
/// row-vector, leading-axis, or element-count broadcast is inferred.
pub fn channel_bias_add_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let shape = vec![symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(CHANNEL_BIAS_ADD_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(shape.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("columns")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            shape,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            CHANNEL_BIAS_ADD_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelBiasAddBf16Geometry {
    rows: u64,
    columns: u64,
    elements: u64,
}
impl ChannelBiasAddBf16Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        bias: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [rows, columns] = x.dimensions() else {
            return Err(invalid("input must be rank two"));
        };
        if x != y || bias.dimensions() != [*columns] {
            return Err(invalid("requires exact X/Y[R,C] and bias[C]"));
        }
        for t in [x, bias, y] {
            if t.element_type() != ElementType::Bf16
                || t.layout() != &ResolvedTensorLayout::Contiguous
                || t.dimensions().contains(&0)
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid(
                    "requires nonempty contiguous addressable BF16 tensors",
                ));
            }
        }
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            elements,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub fn matrix_bytes(self) -> u64 {
        self.elements * 2
    }
    pub fn bias_bytes(self) -> u64 {
        self.columns * 2
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 channel bias: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: Vec<u64>, ty: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(d, ty, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn channel_bias_bf16_requires_explicit_vector_matrix_and_addressable_extents() {
        let x = t(vec![3, 5], ElementType::Bf16);
        let b = t(vec![5], ElementType::Bf16);
        let g = ChannelBiasAddBf16Geometry::new(&x, &b, &x).unwrap();
        assert_eq!(
            (
                g.rows(),
                g.columns(),
                g.elements(),
                g.matrix_bytes(),
                g.bias_bytes()
            ),
            (3, 5, 15, 30, 10)
        );
        for wrong in [
            t(vec![1, 5], ElementType::Bf16),
            t(vec![15], ElementType::Bf16),
            t(vec![1], ElementType::Bf16),
            t(vec![5], ElementType::F32),
        ] {
            assert!(ChannelBiasAddBf16Geometry::new(&x, &wrong, &x).is_err());
        }
        assert!(
            ChannelBiasAddBf16Geometry::new(&x, &b, &t(vec![5, 3], ElementType::Bf16)).is_err()
        );
        let huge = t(vec![3, 1 << 61], ElementType::Bf16);
        assert!(ChannelBiasAddBf16Geometry::new(
            &huge,
            &t(vec![1 << 61], ElementType::Bf16),
            &huge
        )
        .is_err());
        let c = channel_bias_add_bf16_contract().unwrap();
        assert_eq!(c.descriptor().outputs[0].alias(), &AliasPolicy::NoAlias);
        assert!(c
            .descriptor()
            .validate_attributes(&BTreeMap::from([(
                AttributeId::new("broadcast").unwrap(),
                crate::SemanticValue::Text("scalar".into())
            )]))
            .is_err());
    }
}
