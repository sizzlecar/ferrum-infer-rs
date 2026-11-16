//! Explicit activation quantization, independent of a projection's weights.
use super::*;
use crate::{
    ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue, WeightComponentRole, WeightEncoding,
};

pub const NVFP4_QUANTIZE_BF16_OPERATION_ID: &str = "operation.nvfp4_quantize.bf16";
pub const NVFP4_QUANTIZE_BF16_CAPABILITY_ID: &str = "capability.operation.nvfp4_quantize.bf16";

/// Arithmetic recipes are explicit graph attributes. Approximate reciprocal
/// and FTZ are not inferred from an architecture, model name or environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4QuantizeProfile {
    CudaRcpApproxFtzV1,
}
impl Nvfp4QuantizeProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CudaRcpApproxFtzV1 => "cuda_rcp_approx_ftz_v1",
        }
    }
    pub fn attributes(self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([(
            AttributeId::new("profile")?,
            SemanticValue::Text(self.as_str().into()),
        )]))
    }
}

/// BF16 X[M,K], F32 global_sf[1] -> E2M1 packed[M,K/2], E4M3 scales in
/// Nvfp4MatrixLayout's 128x4 byte layout. K is divisible by 16. Padding scales
/// are positive zero; padding never adds activation rows. All outputs are
/// overwritten and do not alias inputs or one another.
///
/// The source/producer supplies finite X and a positive finite global_sf.
/// For each group, amax is the maximum absolute decoded BF16 value. Profile
/// CudaRcpApproxFtzV1 uses CUDA PTX R=rcp.approx.ftz.f32 and separately rounded
/// F=mul.rn.ftz.f32: sf=E4M3_RNE_satfinite(F(global,F(amax,R(6)))); the encode
/// multiplier is Z(amax) ? +0 : R(F(decode(sf),R(global))), where
/// Z uses setp.eq.ftz.f32 against zero (including F32-subnormal amax). Each packed code
/// is E2M1_RNE_satfinite(F(decode(x), multiplier)), even column in low nibble.
/// In particular, scale underflow to zero is not replaced by an epsilon or a
/// different branch. Qualification of the PTX nonfinite intermediate cases
/// belongs to this numerical profile; a CPU reciprocal is not its oracle.
/// No weight alpha, global amax reduction, weight conversion or SiLU occurs.
pub fn nvfp4_quantize_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(NVFP4_QUANTIZE_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![symbol("m"), symbol("k")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Exact(1)],
                [ElementType::F32],
                TensorAccess::Read,
            )?
            .with_numerical_or_physical_weight_component(
                WeightComponentRole::Metadata,
                WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
            )?,
        ],
        outputs: vec![
            contiguous_tensor(
                vec![symbol("m"), symbol("packed_k")],
                [ElementType::U8],
                TensorAccess::Write,
            )?,
            contiguous_tensor(
                vec![symbol("scale_m"), symbol("scale_k")],
                [ElementType::U8],
                TensorAccess::Write,
            )?,
        ],
        attributes: AttributeSchema::new(BTreeMap::from([(
            AttributeId::new("profile")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Text,
                required: true,
                constraint: AttributeConstraint::TextChoices {
                    values: BTreeSet::from([Nvfp4QuantizeProfile::CudaRcpApproxFtzV1
                        .as_str()
                        .into()]),
                },
            },
        )]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            NVFP4_QUANTIZE_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4QuantizeBf16Geometry {
    matrix: Nvfp4MatrixLayout,
}
impl Nvfp4QuantizeBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        global: &ResolvedTensorSpec,
        packed: &ResolvedTensorSpec,
        scales: &ResolvedTensorSpec,
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        nvfp4_quantize_bf16_contract()?
            .descriptor
            .validate_attributes(attributes)?;
        let [m, k] = input.dimensions() else {
            return Err(invalid("input must be rank two"));
        };
        let matrix = Nvfp4MatrixLayout::new(*m, *k)?;
        if input.element_type() != ElementType::Bf16
            || global.element_type() != ElementType::F32
            || global.dimensions() != [1]
            || packed.element_type() != ElementType::U8
            || scales.element_type() != ElementType::U8
            || packed.dimensions() != matrix.packed_shape()
            || scales.dimensions() != matrix.scale_shape()
            || [input, global, packed, scales]
                .into_iter()
                .any(|t| t.layout() != &ResolvedTensorLayout::Contiguous)
        {
            return Err(invalid(
                "requires exact contiguous BF16 X, F32 global[1], U8 packed and padded scales",
            ));
        }
        for tensor in [input, global, packed, scales] {
            if tensor.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        Ok(Self { matrix })
    }
    pub const fn matrix(self) -> Nvfp4MatrixLayout {
        self.matrix
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("NVFP4 quantize: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn quantization_keeps_logical_rows_and_requires_explicit_recipe() {
        let x = t(&[129, 80], ElementType::Bf16);
        let g = t(&[1], ElementType::F32);
        let q = t(&[129, 40], ElementType::U8);
        let s = t(&[256, 8], ElementType::U8);
        let attrs = Nvfp4QuantizeProfile::CudaRcpApproxFtzV1
            .attributes()
            .unwrap();
        let geometry = Nvfp4QuantizeBf16Geometry::new(&x, &g, &q, &s, &attrs).unwrap();
        assert_eq!(geometry.matrix().columns(), 80);
        assert!(Nvfp4QuantizeBf16Geometry::new(&x, &g, &q, &s, &BTreeMap::new()).is_err());
        assert!(Nvfp4QuantizeBf16Geometry::new(
            &x,
            &g,
            &t(&[256, 40], ElementType::U8),
            &s,
            &attrs
        )
        .is_err());
        assert!(
            Nvfp4QuantizeBf16Geometry::new(&x, &t(&[], ElementType::F32), &q, &s, &attrs).is_err()
        );
        assert!(
            Nvfp4QuantizeBf16Geometry::new(&x, &g, &q, &t(&[129, 5], ElementType::U8), &attrs)
                .is_err()
        );
    }
}
