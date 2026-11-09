//! Independent BF16 addcmul, retaining F32 arithmetic until the final BF16.
use super::*;
use crate::ResolvedTensorSpec;
pub const ADDCMUL_BF16_OPERATION_ID: &str = "operation.addcmul.bf16";
pub const ADDCMUL_BF16_CAPABILITY_ID: &str = "capability.operation.addcmul.bf16";
/// Inputs are residual X, multiplier G, update U; Y=B(RN32(X+RN32(G*U))).
/// Only the final conversion rounds to BF16. This is fixed-alpha-one addcmul,
/// not the existing gated residual's B(X+B(G*U)); no gate activation is applied.
/// X/U/Y share [rows,width], G has the explicitly selected RowBroadcast shape.
pub fn addcmul_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = gated_residual_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(ADDCMUL_BF16_OPERATION_ID)?;
    descriptor.provider =
        provider_requirement(ADDCMUL_BF16_CAPABILITY_ID, ContractVersion::new(1, 0))?;
    descriptor.resources = no_auxiliary_resources();
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddcmulBf16Geometry {
    broadcast: Bf16RowBroadcastGeometry,
}
impl AddcmulBf16Geometry {
    pub fn new(
        base: &ResolvedTensorSpec,
        multiplier: &ResolvedTensorSpec,
        update: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        mode: RowBroadcast,
    ) -> Result<Self, VNextError> {
        if update != base {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "addcmul update and residual specifications differ".into(),
            });
        }
        let result = Self {
            broadcast: Bf16RowBroadcastGeometry::new(base, multiplier, output, mode)?,
        };
        result.logical_arithmetic_operations()?;
        Ok(result)
    }
    pub const fn rows(self) -> u64 {
        self.broadcast.rows()
    }
    pub const fn width(self) -> u64 {
        self.broadcast.width()
    }
    pub const fn mode(self) -> RowBroadcast {
        self.broadcast.mode()
    }
    pub fn logical_arithmetic_operations(self) -> Result<u64, VNextError> {
        self.rows()
            .checked_mul(self.width())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: "addcmul arithmetic count overflows".into(),
            })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, ResolvedTensorLayout};
    fn spec(shape: &[u64]) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            shape.to_vec(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    #[test]
    fn declared_broadcast_covers_rows_without_implicit_extent_inference() {
        let x = spec(&[2, 3]);
        for (g, mode) in [
            (spec(&[1, 3]), RowBroadcast::Row),
            (x.clone(), RowBroadcast::Elementwise),
        ] {
            let a = AddcmulBf16Geometry::new(&x, &g, &x, &x, mode).unwrap();
            assert_eq!(a.logical_arithmetic_operations().unwrap(), 12);
        }
        assert!(
            AddcmulBf16Geometry::new(&x, &spec(&[1, 3]), &x, &x, RowBroadcast::Elementwise)
                .is_err()
        );
        assert!(
            AddcmulBf16Geometry::new(&x, &x, &spec(&[1, 3]), &x, RowBroadcast::Elementwise)
                .is_err()
        );
    }
    #[test]
    fn product_bf16_rounding_would_change_the_contract() {
        let gate = 129.0_f32 / 128.0;
        let update = 132.0_f32 / 128.0;
        // B(gate*update)=133/128; adding one lands exactly at the
        // BF16 midpoint 2.0390625, whose even neighbor is 2.03125.
        // The unrounded product lies above it and rounds to 2.046875.
        assert_eq!(gate * update, 1.039306640625);
        assert!(1.0 + gate * update > 2.0390625);
        assert_eq!(1.0 + 133.0 / 128.0, 2.0390625);
        assert_ne!(
            addcmul_bf16_contract().unwrap().descriptor.id,
            gated_residual_bf16_contract().unwrap().descriptor.id
        );
    }
}
