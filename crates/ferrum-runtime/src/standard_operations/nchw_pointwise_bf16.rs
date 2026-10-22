//! Explicit NCHW ports for existing element-local BF16 arithmetic.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const SILU_NCHW_BF16_OPERATION_ID: &str = "operation.silu.nchw.bf16";
pub const SILU_NCHW_BF16_CAPABILITY_ID: &str = "capability.operation.silu.nchw.bf16";
pub const PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID: &str =
    "operation.plain_residual_add.nchw.bf16";
pub const PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID: &str =
    "capability.operation.plain_residual_add.nchw.bf16";

/// The scalar stages/profile obligations of silu.bf16, over exact NCHW inputs.
/// It adds no view/reshape node and no intermediate BF16 boundary.
pub fn silu_nchw_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    nchw(
        silu_bf16_contract()?,
        SILU_NCHW_BF16_OPERATION_ID,
        SILU_NCHW_BF16_CAPABILITY_ID,
        1,
    )
}
/// B(RN32(x+update)) at corresponding NCHW coordinates; no broadcasting.
pub fn plain_residual_add_nchw_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    nchw(
        plain_residual_add_bf16_contract()?,
        PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
        PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID,
        2,
    )
}
fn nchw(
    mut contract: StandardOperationContract,
    id: &str,
    capability: &str,
    count: usize,
) -> Result<StandardOperationContract, VNextError> {
    let dims = vec![
        symbol("batch"),
        symbol("channels"),
        symbol("height"),
        symbol("width"),
    ];
    contract.descriptor.id = OperationId::new(id)?;
    contract.descriptor.inputs = (0..count)
        .map(|_| contiguous_tensor(dims.clone(), [ElementType::Bf16], TensorAccess::Read))
        .collect::<Result<_, _>>()?;
    contract.descriptor.outputs = vec![contiguous_tensor(
        dims,
        [ElementType::Bf16],
        TensorAccess::Write,
    )?];
    contract.descriptor.provider = provider_requirement(capability, ContractVersion::new(1, 0))?;
    contract.descriptor.validate()?;
    Ok(contract)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16NchwGeometry {
    shape: [u64; 4],
    elements: u64,
}
impl Bf16NchwGeometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let shape: [u64; 4] = input
            .dimensions()
            .try_into()
            .map_err(|_| invalid("requires rank-four NCHW"))?;
        if input != output
            || shape.contains(&0)
            || input.element_type() != ElementType::Bf16
            || input.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid(
                "requires exactly matching contiguous BF16 NCHW operands",
            ));
        }
        input.minimum_storage_bytes()?;
        let elements = shape
            .into_iter()
            .try_fold(1_u64, |n, d| n.checked_mul(d))
            .ok_or_else(|| invalid("element count overflows"))?;
        Ok(Self { shape, elements })
    }
    pub const fn batch(self) -> u64 {
        self.shape[0]
    }
    pub const fn channels(self) -> u64 {
        self.shape[1]
    }
    pub const fn height(self) -> u64 {
        self.shape[2]
    }
    pub const fn width(self) -> u64 {
        self.shape[3]
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 NCHW pointwise: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: &[u64], e: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(d.to_vec(), e, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn real_nchw_abi_retains_scalar_contract_and_rejects_fake_views() {
        let x = t(&[2, 3, 4, 5], ElementType::Bf16);
        assert_eq!(Bf16NchwGeometry::new(&x, &x).unwrap().element_count(), 120);
        for y in [
            t(&[2, 3, 5, 4], ElementType::Bf16),
            t(&[24, 5], ElementType::Bf16),
            t(&[2, 3, 4, 5], ElementType::F16),
        ] {
            assert!(Bf16NchwGeometry::new(&x, &y).is_err());
        }
        for (c, count) in [
            (silu_nchw_bf16_contract().unwrap(), 1),
            (plain_residual_add_nchw_bf16_contract().unwrap(), 2),
        ] {
            assert_eq!(c.descriptor.inputs.len(), count);
            assert!(c
                .descriptor
                .inputs
                .iter()
                .chain(&c.descriptor.outputs)
                .all(|t| t.dimensions().len() == 4 && t.alias() == &AliasPolicy::NoAlias));
            assert_eq!(c.descriptor.attributes, AttributeSchema::empty());
            assert_eq!(c.descriptor.oracle, OracleSpec::OperationDefined);
        }
    }
}
