//! Direct F32 normal draws from explicit Philox seed/offset state.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const PHILOX_NORMAL_F32_OPERATION_ID: &str = "operation.philox_normal.f32";
pub const PHILOX_NORMAL_F32_CAPABILITY_ID: &str = "capability.operation.philox_normal.f32";

/// Read-only U32[4] is [seed low, seed high, offset low, offset high], as for
/// [`PhiloxSeedOffset`]. This independent precision contract writes the native
/// F32 normal, without an intermediate BF16 conversion. It does not advance
/// state, require a global RNG allocation, or promise Torch CPU/CUDA parity.
/// The selected provider declares the native generator and launch mapping.
pub fn philox_normal_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(PHILOX_NORMAL_F32_OPERATION_ID)?,
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
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::empty(),
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            PHILOX_NORMAL_F32_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhiloxNormalF32Geometry {
    shape: [u64; 4],
    elements: u64,
}
impl PhiloxNormalF32Geometry {
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
            || output.element_type() != ElementType::F32
            || output.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid("output must be positive contiguous F32 [1,C,H,W]"));
        }
        let bytes = output.minimum_storage_bytes()?;
        if bytes > i32::MAX as u64 || bytes > isize::MAX as u64 {
            return Err(invalid("output exceeds supported 32-bit byte indexing"));
        }
        Ok(Self {
            shape,
            elements: bytes / 4,
        })
    }
    pub const fn shape(self) -> [u64; 4] {
        self.shape
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
    pub const fn output_bytes(self) -> u64 {
        self.elements * 4
    }
    /// Logical draws, including no charge for unused four-wide tail values;
    /// this is not a hardware FLOP count.
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.elements
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("Philox F32 normal: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SemanticValue;
    fn tensor(dims: &[u64], ty: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(dims.to_vec(), ty, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn f32_precision_ports_and_real_byte_capacity_are_independent_of_bf16() {
        let state = tensor(&[4], ElementType::U32);
        let output = tensor(&[1, 3, 1, 257], ElementType::F32);
        let g = PhiloxNormalF32Geometry::new(&state, &output).unwrap();
        assert_eq!(
            (g.shape(), g.element_count(), g.output_bytes()),
            ([1, 3, 1, 257], 771, 3084)
        );
        for wrong in [
            tensor(&[4], ElementType::F32),
            tensor(&[2, 2], ElementType::U32),
        ] {
            assert!(PhiloxNormalF32Geometry::new(&wrong, &output).is_err());
        }
        for wrong in [
            tensor(&[2, 1, 1, 257], ElementType::F32),
            tensor(&[1, 3, 1, 257], ElementType::Bf16),
            tensor(&[1, 1, 1, 1 << 29], ElementType::F32),
        ] {
            assert!(PhiloxNormalF32Geometry::new(&state, &wrong).is_err());
        }
        // Empty geometry is rejected before an operation can receive a spec.
        assert!(ResolvedTensorSpec::new(
            vec![1, 1, 0, 257],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .is_err());
        assert!(PhiloxNormalF32Geometry::new(
            &state,
            &tensor(&[1, 1, 1, (1 << 29) - 1], ElementType::F32)
        )
        .is_ok());
        let c = philox_normal_f32_contract().unwrap();
        assert_eq!(c.descriptor().outputs[0].alias(), &AliasPolicy::NoAlias);
        assert!(c
            .descriptor()
            .validate_attributes(&BTreeMap::from([(
                AttributeId::new("seed").unwrap(),
                SemanticValue::Unsigned(42)
            )]))
            .is_err());
        assert_ne!(
            c.descriptor().fingerprint().unwrap(),
            philox_normal_bf16_contract()
                .unwrap()
                .descriptor()
                .fingerprint()
                .unwrap()
        );
    }
}
