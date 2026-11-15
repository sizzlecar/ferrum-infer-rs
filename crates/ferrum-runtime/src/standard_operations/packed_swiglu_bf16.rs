//! Value-first packed BF16 SwiGLU with an explicit rounding profile.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const PACKED_SWIGLU_BF16_OPERATION_ID: &str = "operation.packed_swiglu.bf16";
pub const PACKED_SWIGLU_BF16_CAPABILITY_ID: &str = "capability.operation.packed_swiglu.bf16";

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedSwiGluBf16Profile {
    Exp2SigmoidF32SingleRound,
    SiluBf16ThenMultiply,
}
impl PackedSwiGluBf16Profile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exp2SigmoidF32SingleRound => "exp2_div_sigmoid_f32_single_bf16_v1",
            Self::SiluBf16ThenMultiply => "silu_bf16_then_multiply_bf16_v1",
        }
    }
    pub fn attributes(self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([(
            AttributeId::new("profile")?,
            SemanticValue::Text(self.as_str().into()),
        )]))
    }
    pub fn from_attributes(
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        for profile in [Self::Exp2SigmoidF32SingleRound, Self::SiluBf16ThenMultiply] {
            if *attrs == profile.attributes()? {
                return Ok(profile);
            }
        }
        Err(invalid("requires exactly one declared rounding profile"))
    }
    pub const fn arithmetic_operations_per_element(self) -> u64 {
        match self {
            Self::Exp2SigmoidF32SingleRound => 7,
            Self::SiluBf16ThenMultiply => 5,
        }
    }
}

/// X[M,2N] contains value then gate halves. With `Exp2SigmoidF32SingleRound`,
/// decode both to F32, evaluate
/// z=RN32(RN32(0-gate)*F32::from_bits(0x3fb8aa3b)), e=exp2_profile(z),
/// d=RN32(1+e), s=div_profile(1,d). Multiply RN32(gate*s), then
/// RN32(value*that), and round only the final result to BF16 RNE.
/// An intermediate BF16 SiLU, gate/div rewrite,
/// stable sigmoid rewrite, FMA, or fused downstream quantization is different.
/// Arithmetic outside exp2/div uses gradual underflow. The CUDA profile uses
/// ex2.approx.f32 and div.full.f32, both without FTZ, as emitted by the pinned
/// Triton3.7 baseline. Reference exp2/div approximations are identified in
/// provider fingerprints; this contract does not assert bitwise equality of CPU and GPU transcendental
/// implementations. Inputs/outputs are contiguous and must not alias.
/// `SiluBf16ThenMultiply` instead evaluates the existing BF16 SiLU contract:
/// RN32(gate / RN32(1 + exp_profile(-gate))), round to BF16, then multiply
/// that rounded value by `value` in F32 and round to BF16 again. Packing only
/// removes transport; it must not erase this intermediate rounding boundary.
pub fn packed_swiglu_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(PACKED_SWIGLU_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            vec![symbol("m"), symbol("packed_n")],
            [ElementType::Bf16],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor(
            vec![symbol("m"), symbol("n")],
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([(
            AttributeId::new("profile")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Text,
                required: true,
                constraint: AttributeConstraint::TextChoices {
                    values: [
                        PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound,
                        PackedSwiGluBf16Profile::SiluBf16ThenMultiply,
                    ]
                    .into_iter()
                    .map(|p| p.as_str().into())
                    .collect(),
                },
            },
        )]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            PACKED_SWIGLU_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedSwiGluBf16Geometry {
    rows: u64,
    width: u64,
    elements: u64,
}
impl PackedSwiGluBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [rows, packed] = input.dimensions() else {
            return Err(invalid("requires rank2 packed input"));
        };
        if *rows == 0
            || *packed == 0
            || packed % 2 != 0
            || output.dimensions() != [*rows, *packed / 2]
            || [input, output].iter().any(|s| {
                s.element_type() != ElementType::Bf16
                    || s.layout() != &ResolvedTensorLayout::Contiguous
            })
        {
            return Err(invalid("requires contiguous BF16 [M,2N] -> [M,N]"));
        }
        for s in [input, output] {
            if s.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("packed SwiGLU exceeds address space"));
            }
        }
        let elements = rows
            .checked_mul(*packed / 2)
            .ok_or_else(|| invalid("packed SwiGLU extent overflows"))?;
        elements
            .checked_mul(7)
            .ok_or_else(|| invalid("packed SwiGLU logical work overflows"))?;
        Ok(Self {
            rows: *rows,
            width: *packed / 2,
            elements,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn width(self) -> u64 {
        self.width
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn input_bytes(self) -> u64 {
        self.elements * 4
    }
    pub const fn output_bytes(self) -> u64 {
        self.elements * 2
    }
    pub const fn arithmetic_operations(self, profile: PackedSwiGluBf16Profile) -> u64 {
        self.elements * profile.arithmetic_operations_per_element()
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("packed SwiGLU: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec(shape: Vec<u64>, dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape, dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn packed_geometry_and_explicit_profile_are_required() {
        let x = spec(vec![3, 256], ElementType::Bf16);
        let y = spec(vec![3, 128], ElementType::Bf16);
        let g = PackedSwiGluBf16Geometry::new(&x, &y).unwrap();
        assert_eq!(g.input_bytes(), 1536);
        assert_eq!(g.output_bytes(), 768);
        for bad in [
            spec(vec![3, 127], ElementType::Bf16),
            spec(vec![3, 256], ElementType::Bf16),
            spec(vec![3, 128], ElementType::F32),
        ] {
            assert!(PackedSwiGluBf16Geometry::new(&x, &bad).is_err());
        }
        assert!(PackedSwiGluBf16Geometry::new(&spec(vec![3, 255], ElementType::Bf16), &y).is_err());
        let c = packed_swiglu_bf16_contract().unwrap();
        assert!(c
            .descriptor()
            .validate_attributes(&BTreeMap::new())
            .is_err());
        for profile in [
            PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound,
            PackedSwiGluBf16Profile::SiluBf16ThenMultiply,
        ] {
            let attrs = profile.attributes().unwrap();
            c.descriptor().validate_attributes(&attrs).unwrap();
            assert_eq!(
                PackedSwiGluBf16Profile::from_attributes(&attrs).unwrap(),
                profile
            );
        }
        assert!(PackedSwiGluBf16Profile::from_attributes(&BTreeMap::new()).is_err());
        assert_eq!(
            g.arithmetic_operations(PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound),
            3 * 128 * 7
        );
        assert_eq!(
            g.arithmetic_operations(PackedSwiGluBf16Profile::SiluBf16ThenMultiply),
            3 * 128 * 5
        );
    }
}
