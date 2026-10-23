//! Stateless normal draws from an explicit Philox seed and subsequence offset.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const PHILOX_NORMAL_BF16_OPERATION_ID: &str = "operation.philox_normal.bf16";
pub const PHILOX_NORMAL_BF16_CAPABILITY_ID: &str = "capability.operation.philox_normal.bf16";

/// U32[4] is [seed low, seed high, offset low, offset high]. It is a real
/// read-only input, never hidden global RNG state. Output is contiguous BF16
/// [1,C,H,W]. Every u64 offset has the provider's documented cuRAND semantics;
/// this operation does not advance or mutate its input state.
///
/// The provider identifies the device-dependent launch mapping and normal
/// implementation. This contract does not promise CPU/CUDA, device-to-device,
/// or Torch bitwise equivalence. The first implementation supports output
/// byte extents <= i32::MAX to avoid TensorIterator's large-index splitting.
pub fn philox_normal_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(PHILOX_NORMAL_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            vec![DimensionConstraint::Exact(4)],
            [ElementType::U32],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor(
            vec![
                DimensionConstraint::Exact(1),
                symbol("channels"),
                symbol("height"),
                symbol("width"),
            ],
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::empty(),
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            PHILOX_NORMAL_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Explicit state serialization; fresh seed is the initial first-image path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhiloxSeedOffset {
    seed: u64,
    offset: u64,
}
impl PhiloxSeedOffset {
    pub const fn new(seed: u64, offset: u64) -> Self {
        Self { seed, offset }
    }
    pub const fn fresh(seed: u64) -> Self {
        Self::new(seed, 0)
    }
    pub const fn seed(self) -> u64 {
        self.seed
    }
    pub const fn offset(self) -> u64 {
        self.offset
    }
    pub const fn words(self) -> [u32; 4] {
        [
            self.seed as u32,
            (self.seed >> 32) as u32,
            self.offset as u32,
            (self.offset >> 32) as u32,
        ]
    }
    pub fn little_endian_bytes(self) -> [u8; 16] {
        let mut bytes = [0; 16];
        for (dst, word) in bytes.chunks_exact_mut(4).zip(self.words()) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhiloxNormalBf16Geometry {
    shape: [u64; 4],
    elements: u64,
}
impl PhiloxNormalBf16Geometry {
    pub fn new(
        state: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        if state.element_type() != ElementType::U32
            || state.dimensions() != [4]
            || state.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid("state must be contiguous U32[4]"));
        }
        let shape: [u64; 4] = output
            .dimensions()
            .try_into()
            .map_err(|_| invalid("output must be NCHW rank four"))?;
        if shape[0] != 1
            || shape.contains(&0)
            || output.element_type() != ElementType::Bf16
            || output.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid("output must be positive contiguous BF16 [1,C,H,W]"));
        }
        let bytes = output.minimum_storage_bytes()?;
        if bytes > i32::MAX as u64 || bytes > isize::MAX as u64 {
            return Err(invalid("output exceeds supported 32-bit byte indexing"));
        }
        Ok(Self {
            shape,
            elements: bytes / 2,
        })
    }
    pub const fn shape(self) -> [u64; 4] {
        self.shape
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
    pub const fn output_bytes(self) -> u64 {
        self.elements * 2
    }
    /// One logical normal draw per output, not a hardware FLOP count. The
    /// provider may draw unused tail values in its four-wide generator.
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.elements
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("Philox BF16 normal: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(dims: &[u64], ty: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(dims.to_vec(), ty, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn explicit_state_preserves_all_seed_and_offset_bits() {
        let state = PhiloxSeedOffset::new(0x0123456789abcdef, 0xfedcba9876543210);
        assert_eq!(
            state.words(),
            [0x89abcdef, 0x01234567, 0x76543210, 0xfedcba98]
        );
        assert_eq!(
            state.little_endian_bytes(),
            [
                0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba,
                0xdc, 0xfe
            ]
        );
        assert_eq!(
            PhiloxSeedOffset::fresh(u64::MAX).words(),
            [u32::MAX, u32::MAX, 0, 0]
        );
    }
    #[test]
    fn exact_ports_nchw_and_32_bit_bound_are_checked_without_payloads() {
        let state = tensor(&[4], ElementType::U32);
        let output = tensor(&[1, 64, 64, 64], ElementType::Bf16);
        let g = PhiloxNormalBf16Geometry::new(&state, &output).unwrap();
        assert_eq!(g.element_count(), 262144);
        assert_eq!(g.output_bytes(), 524288);
        for wrong in [
            tensor(&[2, 2], ElementType::U32),
            tensor(&[4], ElementType::F32),
            tensor(&[8], ElementType::U32),
        ] {
            assert!(PhiloxNormalBf16Geometry::new(&wrong, &output).is_err());
        }
        for wrong in [
            tensor(&[2, 32, 64, 64], ElementType::Bf16),
            tensor(&[1, 64, 64, 64], ElementType::F16),
            tensor(&[1, 1, 1, 1 << 30], ElementType::Bf16),
        ] {
            assert!(PhiloxNormalBf16Geometry::new(&state, &wrong).is_err());
        }
        assert!(PhiloxNormalBf16Geometry::new(
            &state,
            &tensor(&[1, 1, 1, (1 << 30) - 1], ElementType::Bf16)
        )
        .is_ok());
        let c = philox_normal_bf16_contract().unwrap();
        assert_eq!(c.descriptor().outputs[0].alias(), &AliasPolicy::NoAlias);
        assert!(c
            .descriptor()
            .validate_attributes(&BTreeMap::from([(
                AttributeId::new("seed").unwrap(),
                crate::SemanticValue::Unsigned(1)
            )]))
            .is_err());
    }
}
