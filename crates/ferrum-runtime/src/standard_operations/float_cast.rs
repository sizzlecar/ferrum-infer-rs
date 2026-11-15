//! Two explicit cast contracts share geometry and physical execution plumbing.
//! Narrowing keeps its existing RNE contract; widening is exact bit expansion.
use super::*;
use crate::ResolvedTensorSpec;

pub const CAST_BF16_TO_F32_OPERATION_ID: &str = "operation.cast.bf16_to_f32";
pub const CAST_BF16_TO_F32_CAPABILITY_ID: &str = "capability.operation.cast.bf16_to_f32";

/// Expand each BF16 encoding into the upper 16 bits of F32, with low bits zero.
/// This is exact for every finite value including subnormals and signed zero.
/// Infinity, NaN sign/payload and signaling/quiet bits are preserved. There is
/// no floating arithmetic, FTZ, rounding or intermediate F16 representation.
/// Shapes are identical contiguous matrices and the output must not alias X.
pub fn cast_bf16_to_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = cast_f32_to_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(CAST_BF16_TO_F32_OPERATION_ID)?;
    descriptor.inputs = vec![contiguous_tensor(
        vec![symbol("rows"), symbol("width")],
        [ElementType::Bf16],
        TensorAccess::Read,
    )?];
    descriptor.outputs = vec![contiguous_tensor(
        vec![symbol("rows"), symbol("width")],
        [ElementType::F32],
        TensorAccess::Write,
    )?];
    descriptor.provider =
        provider_requirement(CAST_BF16_TO_F32_CAPABILITY_ID, ContractVersion::new(1, 0))?;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32Bf16CastKind {
    Narrow,
    Widen,
}
impl F32Bf16CastKind {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Narrow => CAST_F32_TO_BF16_OPERATION_ID,
            Self::Widen => CAST_BF16_TO_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Narrow => CAST_F32_TO_BF16_CAPABILITY_ID,
            Self::Widen => CAST_BF16_TO_F32_CAPABILITY_ID,
        }
    }
    pub const fn input_type(self) -> ElementType {
        match self {
            Self::Narrow => ElementType::F32,
            Self::Widen => ElementType::Bf16,
        }
    }
    pub const fn output_type(self) -> ElementType {
        match self {
            Self::Narrow => ElementType::Bf16,
            Self::Widen => ElementType::F32,
        }
    }
    pub const fn input_bytes(self) -> u64 {
        match self {
            Self::Narrow => 4,
            Self::Widen => 2,
        }
    }
    pub const fn output_bytes(self) -> u64 {
        match self {
            Self::Narrow => 2,
            Self::Widen => 4,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Narrow => cast_f32_to_bf16_contract(),
            Self::Widen => cast_bf16_to_f32_contract(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32Bf16CastGeometry {
    kind: F32Bf16CastKind,
    extent: CastF32ToBf16Geometry,
}
impl F32Bf16CastGeometry {
    pub fn new(
        kind: F32Bf16CastKind,
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let extent = match kind {
            F32Bf16CastKind::Narrow => CastF32ToBf16Geometry::new(input, output)?,
            F32Bf16CastKind::Widen => CastF32ToBf16Geometry::new(output, input)?,
        };
        Ok(Self { kind, extent })
    }
    pub const fn kind(self) -> F32Bf16CastKind {
        self.kind
    }
    pub const fn rows(self) -> u64 {
        self.extent.rows()
    }
    pub const fn width(self) -> u64 {
        self.extent.width()
    }
    pub const fn elements(self) -> u64 {
        self.extent.elements()
    }
}
impl From<CastF32ToBf16Geometry> for F32Bf16CastGeometry {
    fn from(extent: CastF32ToBf16Geometry) -> Self {
        Self {
            kind: F32Bf16CastKind::Narrow,
            extent,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastBf16ToF32Geometry(F32Bf16CastGeometry);
impl CastBf16ToF32Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        F32Bf16CastGeometry::new(F32Bf16CastKind::Widen, input, output).map(Self)
    }
    pub const fn rows(self) -> u64 {
        self.0.rows()
    }
    pub const fn width(self) -> u64 {
        self.0.width()
    }
    pub const fn elements(self) -> u64 {
        self.0.elements()
    }
}
impl From<CastBf16ToF32Geometry> for F32Bf16CastGeometry {
    fn from(value: CastBf16ToF32Geometry) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResolvedTensorLayout;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn directions_keep_distinct_dtypes_and_check_both_byte_extents() {
        let x = tensor(&[2, 17], ElementType::Bf16);
        let y = tensor(&[2, 17], ElementType::F32);
        assert_eq!(CastBf16ToF32Geometry::new(&x, &y).unwrap().elements(), 34);
        assert!(CastF32ToBf16Geometry::new(&x, &y).is_err());
        assert!(CastBf16ToF32Geometry::new(&y, &x).is_err());
        for bad in [
            tensor(&[1, 34], ElementType::F32),
            tensor(&[2, 17], ElementType::F16),
            tensor(&[34], ElementType::F32),
        ] {
            assert!(CastBf16ToF32Geometry::new(&x, &bad).is_err());
        }
        let huge = isize::MAX as u64 / 4 + 1;
        assert!(CastBf16ToF32Geometry::new(
            &tensor(&[1, huge], ElementType::Bf16),
            &tensor(&[1, huge], ElementType::F32)
        )
        .is_err());
    }
}
