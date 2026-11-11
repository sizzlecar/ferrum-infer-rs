//! Canonical codebook W4 storage, regular ConvRot activation INT8, integer GEMM.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const COMFY_CONVROT_W4A8_BF16_OPERATION_ID: &str = "operation.comfy_convrot_codebook_w4a8.bf16";
pub const COMFY_CONVROT_W4A8_BF16_CAPABILITY_ID: &str =
    "capability.operation.comfy_convrot_codebook_w4a8.bf16";

/// Canonical storage is explicit in five input ports: BF16 X[M,K], packed
/// I8 W[N,K/2], U8 E4M3FN relative scales [N,K/16], F32 channel scales [N],
/// and F32 codebook [16]. Output is BF16 Y[M,N]. No bias or correction.
/// This ABI requires scale groups of 16 and regular Hadamard groups of 256.
/// It does not accept resident-only MMA-interleaved weights or other quants.
///
/// Each provider declares the rotation, BF16/division boundaries, INT8
/// activation rounding and epilogue profile. Activation/weight quantization
/// error is part of that operation-defined oracle, separate from dense BF16.
/// Scratch includes decoded INT8 weights for this invocation only; canonical
/// weights remain W4. No implicit matrix, library workspace or allocation.
pub fn comfy_convrot_w4a8_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let symbol = |name: &str| DimensionConstraint::Symbol(name.to_owned());
    let inputs = [
        (
            vec![symbol("rows"), symbol("in_features")],
            ElementType::Bf16,
        ),
        (
            vec![symbol("out_features"), symbol("packed_columns")],
            ElementType::I8,
        ),
        (
            vec![symbol("out_features"), symbol("scale_groups")],
            ElementType::U8,
        ),
        (vec![symbol("out_features")], ElementType::F32),
        (vec![DimensionConstraint::Exact(16)], ElementType::F32),
    ]
    .into_iter()
    .map(|(shape, dtype)| contiguous_tensor(shape, [dtype], TensorAccess::Read))
    .collect::<Result<Vec<_>, _>>()?;
    let descriptor = OperationDescriptor {
        id: OperationId::new(COMFY_CONVROT_W4A8_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![contiguous_tensor(
            vec![symbol("rows"), symbol("out_features")],
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("in_features")?,
            unsigned_attribute("out_features")?,
        ]))?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Required,
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            COMFY_CONVROT_W4A8_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// All offsets address one invocation-owned U8 scratch allocation. Alignment
/// is explicit; no independently acquired resources hide behind these spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComfyConvRotScratch {
    pub rotated_activation: u64,
    pub quantized_activation: u64,
    pub row_scales: u64,
    pub decoded_weight: u64,
    pub accumulator: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComfyConvRotW4A8Geometry {
    rows: u64,
    inputs: u64,
    outputs: u64,
    scratch: ComfyConvRotScratch,
    arithmetic: u64,
}
impl ComfyConvRotW4A8Geometry {
    pub fn new(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
    ) -> Result<Self, VNextError> {
        Self::new_for(inputs, output, k, n, false, ElementType::Bf16)
    }

    /// Gate|up input is [M,2K]; intermediate activation and all scratch spans
    /// retain width K. The raw BF16 constructor keeps its original contract.
    pub fn new_swiglu_f32(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
    ) -> Result<Self, VNextError> {
        Self::new_for(inputs, output, k, n, true, ElementType::Bf16)
    }

    /// Independent F32 activation/output contract, sharing canonical storage
    /// and scratch geometry; no BF16 activation or output boundary is applied.
    pub fn new_f32(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
    ) -> Result<Self, VNextError> {
        Self::new_for(inputs, output, k, n, false, ElementType::F32)
    }

    fn new_for(
        inputs: &[&ResolvedTensorSpec],
        output: &ResolvedTensorSpec,
        k: u64,
        n: u64,
        swiglu: bool,
        activation_type: ElementType,
    ) -> Result<Self, VNextError> {
        if inputs.len() != 5 {
            return Err(invalid("requires exactly five input ports"));
        }
        let [m, width] = inputs[0].dimensions() else {
            return Err(invalid("activation must have rank two"));
        };
        let activation_width = k
            .checked_mul(if swiglu { 2 } else { 1 })
            .ok_or_else(|| invalid("activation width overflows"))?;
        if *m == 0
            || k == 0
            || n == 0
            || *width != activation_width
            || !k.is_multiple_of(256)
            || !n.is_multiple_of(4)
            || [*m, k, n].iter().any(|value| *value > i32::MAX as u64)
            || k.checked_mul(128 * 127)
                .is_none_or(|maximum| maximum > i32::MAX as u64)
        {
            return Err(invalid("positive i32 dimensions require K%256=0, N%4=0 and a nonoverflowing worst-case INT32 dot product"));
        }
        let shapes = [
            vec![*m, activation_width],
            vec![n, k / 2],
            vec![n, k / 16],
            vec![n],
            vec![16],
        ];
        let dtypes = [
            activation_type,
            ElementType::I8,
            ElementType::U8,
            ElementType::F32,
            ElementType::F32,
        ];
        for ((tensor, shape), dtype) in inputs.iter().zip(shapes).zip(dtypes) {
            if tensor.dimensions() != shape
                || tensor.element_type() != dtype
                || tensor.layout() != &ResolvedTensorLayout::Contiguous
            {
                return Err(invalid(
                    "canonical component shape, dtype or layout differs",
                ));
            }
        }
        if output.dimensions() != [*m, n]
            || output.element_type() != activation_type
            || output.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid(
                "output must be contiguous [M,N] with the selected activation dtype",
            ));
        }
        for tensor in inputs.iter().copied().chain(std::iter::once(output)) {
            if tensor.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let bytes = |a: u64, b: u64, size: u64| {
            a.checked_mul(b)
                .and_then(|value| value.checked_mul(size))
                .ok_or_else(|| invalid("scratch extent overflows"))
        };
        let mut end = 0_u64;
        let mut reserve = |length: u64| -> Result<u64, VNextError> {
            let offset = end
                .checked_add(255)
                .map(|value| value & !255)
                .ok_or_else(|| invalid("scratch alignment overflows"))?;
            end = offset
                .checked_add(length)
                .filter(|value| *value <= isize::MAX as u64)
                .ok_or_else(|| invalid("scratch exceeds address space"))?;
            Ok(offset)
        };
        let rotated_activation = reserve(bytes(*m, k, 4)?)?;
        let quantized_activation = reserve(bytes(*m, k, 1)?)?;
        let row_scales = reserve(bytes(*m, 1, 4)?)?;
        let decoded_weight = reserve(bytes(n, k, 1)?)?;
        let accumulator = reserve(bytes(*m, n, 4)?)?;
        let scratch = ComfyConvRotScratch {
            rotated_activation,
            quantized_activation,
            row_scales,
            decoded_weight,
            accumulator,
            total_bytes: end,
        };
        let mut arithmetic = m
            .checked_mul(k)
            .and_then(|value| value.checked_mul(n))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| invalid("logical work overflows"))?;
        if swiglu {
            // Negation, exponential, add, divide and multiply per element.
            arithmetic = m
                .checked_mul(k)
                .and_then(|value| value.checked_mul(5))
                .and_then(|value| arithmetic.checked_add(value))
                .ok_or_else(|| invalid("SwiGLU logical work overflows"))?;
        }
        Ok(Self {
            rows: *m,
            inputs: k,
            outputs: n,
            scratch,
            arithmetic,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn inputs(self) -> u64 {
        self.inputs
    }
    pub const fn outputs(self) -> u64 {
        self.outputs
    }
    pub const fn scratch(self) -> ComfyConvRotScratch {
        self.scratch
    }
    /// Prefix used when GEMM stores the final scaled output directly. Every
    /// rotation/quantization/decoded-weight span remains unchanged; the trailing
    /// I32 accumulator is omitted. This does not assert native availability.
    pub const fn scratch_without_accumulator_bytes(self) -> u64 {
        self.scratch.accumulator
    }
    pub const fn output_elements(self) -> u64 {
        self.rows * self.outputs
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("Comfy ConvRot W4A8 geometry: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    fn geometry(k: u64, n: u64) -> Result<ComfyConvRotW4A8Geometry, VNextError> {
        ComfyConvRotW4A8Geometry::new(
            &[
                &tensor(&[3, k], ElementType::Bf16),
                &tensor(&[n, k / 2], ElementType::I8),
                &tensor(&[n, k / 16], ElementType::U8),
                &tensor(&[n], ElementType::F32),
                &tensor(&[16], ElementType::F32),
            ],
            &tensor(&[3, n], ElementType::Bf16),
            k,
            n,
        )
    }
    #[test]
    fn scratch_covers_every_live_matrix_and_integer_capacity_boundary() {
        let shape = geometry(256, 4).unwrap();
        let scratch = shape.scratch();
        assert_eq!(scratch.rotated_activation, 0);
        assert!(scratch.quantized_activation >= 3 * 256 * 4);
        assert!(scratch.row_scales >= scratch.quantized_activation + 3 * 256);
        assert!(scratch.decoded_weight >= scratch.row_scales + 3 * 4);
        assert!(scratch.accumulator >= scratch.decoded_weight + 4 * 256);
        assert_eq!(scratch.total_bytes, scratch.accumulator + 3 * 4 * 4);
        assert_eq!(
            shape.scratch_without_accumulator_bytes() + 3 * 4 * 4,
            scratch.total_bytes
        );
        assert_eq!(shape.logical_arithmetic_operations(), 3 * 4 * 256 * 2);
        assert!(geometry(132096, 4).is_ok());
        assert!(geometry(132352, 4).is_err());
        assert!(geometry(272, 4).is_err());
        assert!(geometry(256, 5).is_err());
    }
    #[test]
    fn contract_preserves_typed_canonical_components_and_managed_scratch() {
        let contract = comfy_convrot_w4a8_bf16_contract().unwrap();
        assert_eq!(
            contract.descriptor().resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            contract.descriptor().resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
        for (port, dtype) in contract.descriptor().inputs.iter().zip([
            ElementType::Bf16,
            ElementType::I8,
            ElementType::U8,
            ElementType::F32,
            ElementType::F32,
        ]) {
            assert_eq!(port.element_types(), &BTreeSet::from([dtype]));
        }
        assert_ne!(
            contract.descriptor().id,
            dense_linear_bf16_contract().unwrap().descriptor().id
        );
    }
}
