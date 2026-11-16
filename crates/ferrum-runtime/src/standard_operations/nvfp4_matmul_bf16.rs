//! Packed NVFP4 multiplication, independently usable after activation quantization.
use super::*;
use crate::{
    QuantizationFormatId, QuantizationGrouping, QuantizationPacking, QuantizationSpec,
    ResolvedTensorLayout, ResolvedTensorSpec, WeightComponentRole, WeightEncoding,
};

pub const NVFP4_MATMUL_BF16_OPERATION_ID: &str = "operation.nvfp4_matmul.bf16";
pub const NVFP4_MATMUL_BF16_CAPABILITY_ID: &str = "capability.operation.nvfp4_matmul.bf16";
pub const NVFP4_E2M1_E4M3_128X4_FORMAT_ID: &str =
    "quantization.nvfp4-e2m1-e4m3-group16-128x4-lsb-even.v1";

impl QuantizationSpec {
    /// Serialized/container-independent E2M1 bytes with raw E4M3 block scales.
    pub fn nvfp4_e2m1_e4m3_128x4() -> Self {
        Self {
            format_id: QuantizationFormatId::new(NVFP4_E2M1_E4M3_128X4_FORMAT_ID)
                .expect("fixed NVFP4 format ID"),
            bits_per_weight: 4,
            grouping: QuantizationGrouping::fixed(16),
            packing: QuantizationPacking::Interleaved,
            scale_type: ElementType::U8,
            zero_point_type: None,
        }
    }
}

/// A[M,K/2], B[N,K/2] contain low-nibble-first E2M1. Each group of 16
/// elements uses one nonnegative finite E4M3FN scale in the 128x4 layout below.
/// Inputs are A, B, scale_A, scale_B, alpha[1] (F32), in that order. Y[M,N]
/// is BF16(alpha * dot(decode(A)*scale_A, decode(B)*scale_B)). Alpha is finite
/// and positive. Providers declare accumulation and final rounding; no promise
/// of bitwise agreement with another provider or activation quantizer is made.
/// No weight-global-scale recomputation, activation quantization, bias or input
/// permutation occurs here. Those are explicit producer/consumer operations.
/// All inputs must be ready on the invocation stream before submission, and
/// remain live through completion. Output never aliases inputs. Workspace is
/// optional because a prepared native kernel may require zero bytes; any
/// nonzero workspace must be admitted invocation scratch, never hidden memory.
pub fn nvfp4_matmul_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(NVFP4_MATMUL_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![symbol("m"), symbol("packed_k")],
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("n"), symbol("packed_k")],
                [ElementType::U8],
                TensorAccess::Read,
            )?
            .with_physical_weight_component(
                WeightComponentRole::PackedValues,
                WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4()),
            )?,
            contiguous_tensor(
                vec![symbol("scale_m"), symbol("scale_k")],
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("scale_n"), symbol("scale_k")],
                [ElementType::U8],
                TensorAccess::Read,
            )?
            .with_physical_weight_component(
                WeightComponentRole::Scales,
                WeightEncoding::Dense {
                    element_type: ElementType::U8,
                },
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Exact(1)],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![symbol("m"), symbol("n")],
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::new())?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            NVFP4_MATMUL_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Canonical storage geometry shared by checkpoint adapters and operations.
/// Padding describes storage only; it never adds mathematical rows or groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4MatrixLayout {
    rows: u64,
    columns: u64,
    scale_rows: u64,
    scale_columns: u64,
    elements: u64,
    scale_bytes: u64,
}
impl Nvfp4MatrixLayout {
    pub fn new(rows: u64, columns: u64) -> Result<Self, VNextError> {
        if rows == 0 || columns == 0 || columns % 16 != 0 {
            return Err(invalid("positive dimensions and group-16 width required"));
        }
        let elements = rows
            .checked_mul(columns)
            .filter(|v| *v <= isize::MAX as u64)
            .ok_or_else(|| invalid("matrix extent exceeds address space"))?;
        let scale_rows = rows
            .checked_add(127)
            .map(|n| n / 128 * 128)
            .ok_or_else(|| invalid("scale row padding overflow"))?;
        let scale_columns = (columns / 16)
            .checked_add(3)
            .map(|n| n / 4 * 4)
            .ok_or_else(|| invalid("scale column padding overflow"))?;
        let scale_bytes = scale_rows
            .checked_mul(scale_columns)
            .filter(|v| *v <= isize::MAX as u64)
            .ok_or_else(|| invalid("scale extent exceeds address space"))?;
        Ok(Self {
            rows,
            columns,
            scale_rows,
            scale_columns,
            elements,
            scale_bytes,
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
    pub const fn packed_bytes(self) -> u64 {
        self.elements / 2
    }
    pub const fn scale_bytes(self) -> u64 {
        self.scale_bytes
    }
    pub const fn packed_shape(self) -> [u64; 2] {
        [self.rows, self.columns / 2]
    }
    pub const fn scale_shape(self) -> [u64; 2] {
        [self.scale_rows, self.scale_columns]
    }
    /// Index in [row_tile, group_tile, row%32, row/32%4, group%4].
    pub fn scale_index(self, row: u64, group: u64) -> Result<u64, VNextError> {
        if row >= self.rows || group >= self.columns / 16 {
            return Err(invalid("scale coordinate out of bounds"));
        }
        Ok(
            ((((row / 128) * (self.scale_columns / 4) + group / 4) * 32 + row % 32) * 4
                + (row / 32) % 4)
                * 4
                + group % 4,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4MatmulBf16Geometry {
    a: Nvfp4MatrixLayout,
    b: Nvfp4MatrixLayout,
    arithmetic: u64,
}
impl Nvfp4MatmulBf16Geometry {
    pub fn new(
        inputs: [&ResolvedTensorSpec; 5],
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [a, b, sa, sb, alpha] = inputs;
        let [m, packed_k] = a.dimensions() else {
            return Err(invalid("A must be rank two"));
        };
        let [n, other_k] = b.dimensions() else {
            return Err(invalid("B must be rank two"));
        };
        if packed_k != other_k {
            return Err(invalid("packed widths differ"));
        }
        let k = packed_k
            .checked_mul(2)
            .ok_or_else(|| invalid("logical width overflow"))?;
        let a_layout = Nvfp4MatrixLayout::new(*m, k)?;
        let b_layout = Nvfp4MatrixLayout::new(*n, k)?;
        if sa.dimensions() != a_layout.scale_shape()
            || sb.dimensions() != b_layout.scale_shape()
            || alpha.dimensions() != [1]
            || output.dimensions() != [*m, *n]
            || [a, b, sa, sb]
                .iter()
                .any(|t| t.element_type() != ElementType::U8)
            || alpha.element_type() != ElementType::F32
            || output.element_type() != ElementType::Bf16
            || inputs
                .into_iter()
                .chain([output])
                .any(|t| t.layout() != &ResolvedTensorLayout::Contiguous)
        {
            return Err(invalid(
                "requires exact contiguous packed/scaled operands, F32 alpha[1] and BF16 Y[M,N]",
            ));
        }
        for t in inputs.into_iter().chain([output]) {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let arithmetic = m
            .checked_mul(*n)
            .and_then(|v| v.checked_mul(k))
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("logical arithmetic overflow"))?;
        Ok(Self {
            a: a_layout,
            b: b_layout,
            arithmetic,
        })
    }
    pub const fn a(self) -> Nvfp4MatrixLayout {
        self.a
    }
    pub const fn b(self) -> Nvfp4MatrixLayout {
        self.b
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("NVFP4 matmul: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn packed_geometry_preserves_padding_and_dtype_boundaries() {
        let a = tensor(&[3, 32], ElementType::U8);
        let b = tensor(&[17, 32], ElementType::U8);
        let scales = tensor(&[128, 4], ElementType::U8);
        let alpha = tensor(&[1], ElementType::F32);
        let y = tensor(&[3, 17], ElementType::Bf16);
        let g = Nvfp4MatmulBf16Geometry::new([&a, &b, &scales, &scales, &alpha], &y).unwrap();
        assert_eq!(g.logical_arithmetic_operations(), 6528);
        assert!(Nvfp4MatmulBf16Geometry::new(
            [&a, &b, &tensor(&[3, 4], ElementType::U8), &scales, &alpha],
            &y
        )
        .is_err());
        assert!(Nvfp4MatmulBf16Geometry::new(
            [&a, &b, &scales, &scales, &tensor(&[1], ElementType::Bf16)],
            &y
        )
        .is_err());
        let d = nvfp4_matmul_bf16_contract().unwrap();
        assert_eq!(d.descriptor().outputs[0].alias(), &AliasPolicy::NoAlias);
        assert_eq!(
            d.descriptor().resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            d.descriptor().resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
    }
    #[test]
    fn scale_layout_spans_multiple_tiles() {
        let layout = Nvfp4MatrixLayout::new(129, 80).unwrap();
        assert_eq!(layout.scale_shape(), [256, 8]);
        for (row, group, index) in [
            (0, 0, 0),
            (1, 0, 16),
            (32, 0, 4),
            (127, 3, 511),
            (0, 4, 512),
            (128, 0, 1024),
        ] {
            assert_eq!(layout.scale_index(row, group).unwrap(), index);
        }
        assert!(layout.scale_index(129, 0).is_err());
        assert!(Nvfp4MatrixLayout::new(u64::MAX, 16).is_err());
        assert!(Nvfp4MatrixLayout::new(1, 32).is_ok());
        assert!(Nvfp4MatrixLayout::new(1, 31).is_err());
    }
}
