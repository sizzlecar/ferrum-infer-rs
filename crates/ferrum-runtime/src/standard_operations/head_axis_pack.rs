//! Explicit head-major to row-major transport, optionally with one narrowing.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const PACK_HEAD_AXIS_BF16_OPERATION_ID: &str = "operation.pack_head_axis.bf16";
pub const PACK_HEAD_AXIS_BF16_CAPABILITY_ID: &str = "capability.operation.pack_head_axis.bf16";
pub const PACK_HEAD_AXIS_F32_TO_BF16_OPERATION_ID: &str = "operation.pack_head_axis.f32_to_bf16";
pub const PACK_HEAD_AXIS_F32_TO_BF16_CAPABILITY_ID: &str =
    "capability.operation.pack_head_axis.f32_to_bf16";
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadAxisPackKind {
    Bf16,
    F32ToBf16,
}
impl HeadAxisPackKind {
    pub const ALL: [Self; 2] = [Self::Bf16, Self::F32ToBf16];
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Bf16 => PACK_HEAD_AXIS_BF16_OPERATION_ID,
            Self::F32ToBf16 => PACK_HEAD_AXIS_F32_TO_BF16_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Bf16 => PACK_HEAD_AXIS_BF16_CAPABILITY_ID,
            Self::F32ToBf16 => PACK_HEAD_AXIS_F32_TO_BF16_CAPABILITY_ID,
        }
    }
    pub const fn input_type(self) -> ElementType {
        match self {
            Self::Bf16 => ElementType::Bf16,
            Self::F32ToBf16 => ElementType::F32,
        }
    }
    pub const fn input_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32ToBf16 => 4,
        }
    }
    pub const fn output_type(self) -> ElementType {
        ElementType::Bf16
    }
    pub const fn output_bytes(self) -> u64 {
        2
    }
    /// X[B,H,R,D] -> Y[B,R,H*D], contiguous with no alias or temporary.
    /// Bf16 preserves every source encoding, including NaN payload/signaling
    /// bits and signed zero. F32ToBf16 applies IEEE RNE exactly once at the
    /// destination; gradual underflow, infinity and zero sign are preserved,
    /// and NaNs remain NaNs (payload unspecified). No F16 intermediate.
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let descriptor = OperationDescriptor {
            id: OperationId::new(self.operation_id())?,
            version: ContractVersion::new(1, 0),
            inputs: vec![contiguous_tensor(
                vec![
                    symbol("batch"),
                    symbol("heads"),
                    symbol("rows"),
                    symbol("head_dim"),
                ],
                [self.input_type()],
                TensorAccess::Read,
            )?],
            outputs: vec![contiguous_tensor(
                vec![symbol("batch"), symbol("rows"), symbol("columns")],
                [ElementType::Bf16],
                TensorAccess::Write,
            )?],
            attributes: AttributeSchema::empty(),
            resources: no_auxiliary_resources(),
            oracle: OracleSpec::OperationDefined,
            provider: provider_requirement(self.capability_id(), ContractVersion::new(1, 0))?,
            profile_phase: ProfilePhase::Forward,
        };
        descriptor.validate()?;
        Ok(StandardOperationContract { descriptor })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadAxisPackGeometry {
    kind: HeadAxisPackKind,
    batches: u64,
    heads: u64,
    rows: u64,
    head_dim: u64,
    elements: u64,
}
impl HeadAxisPackGeometry {
    pub fn new(
        kind: HeadAxisPackKind,
        x: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [b, h, r, d] = x.dimensions() else {
            return Err(invalid("input must have rank four"));
        };
        let columns = h
            .checked_mul(*d)
            .ok_or_else(|| invalid("head width overflows"))?;
        if [b, h, r, d].iter().any(|&&v| v == 0)
            || x.element_type() != kind.input_type()
            || y.element_type() != ElementType::Bf16
            || y.dimensions() != [*b, *r, columns]
        {
            return Err(invalid(
                "requires positive X[B,H,R,D] and BF16 Y[B,R,H*D] of selected input dtype",
            ));
        }
        for t in [x, y] {
            if t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid("requires contiguous addressable extents"));
            }
        }
        let elements = y.minimum_storage_bytes()? / 2;
        Ok(Self {
            kind,
            batches: *b,
            heads: *h,
            rows: *r,
            head_dim: *d,
            elements,
        })
    }
    pub const fn kind(self) -> HeadAxisPackKind {
        self.kind
    }
    pub const fn batches(self) -> u64 {
        self.batches
    }
    pub const fn heads(self) -> u64 {
        self.heads
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn head_dim(self) -> u64 {
        self.head_dim
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub fn input_bytes(self) -> u64 {
        self.elements * self.kind.input_bytes()
    }
    pub fn output_bytes(self) -> u64 {
        self.elements * 2
    }
    pub fn arithmetic_operations(self) -> u64 {
        if self.kind == HeadAxisPackKind::F32ToBf16 {
            self.elements
        } else {
            0
        }
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("head-axis pack: {reason}"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn head_axis_permutation_keeps_batch_row_and_channel_geometry_explicit() {
        for kind in HeadAxisPackKind::ALL {
            let x = t(&[2, 3, 5, 7], kind.input_type());
            let y = t(&[2, 5, 21], ElementType::Bf16);
            let g = HeadAxisPackGeometry::new(kind, &x, &y).unwrap();
            assert_eq!(g.elements(), 210);
            assert_eq!(g.input_bytes(), 210 * kind.input_bytes());
            for bad in [
                t(&[2, 3, 35], ElementType::Bf16),
                t(&[1, 10, 21], ElementType::Bf16),
                t(&[2, 5, 21], ElementType::F32),
            ] {
                assert!(HeadAxisPackGeometry::new(kind, &x, &bad).is_err());
            }
            assert_eq!(
                kind.contract().unwrap().descriptor().outputs[0].alias(),
                &AliasPolicy::NoAlias
            );
        }
    }
    #[test]
    fn head_axis_pack_checks_both_physical_sizes_and_selected_dtype() {
        let n = isize::MAX as u64 / 4 + 1;
        assert!(HeadAxisPackGeometry::new(
            HeadAxisPackKind::F32ToBf16,
            &t(&[1, 1, 1, n], ElementType::F32),
            &t(&[1, 1, n], ElementType::Bf16)
        )
        .is_err());
        assert!(HeadAxisPackGeometry::new(
            HeadAxisPackKind::Bf16,
            &t(&[1, 1, 1, 1], ElementType::F32),
            &t(&[1, 1, 1], ElementType::Bf16)
        )
        .is_err());
    }
}
