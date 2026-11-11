//! Lookup selected canonical signed INT8 rows without expanding the table.
use super::*;
use crate::{ComfyEmbeddingDequantizedDtype, ResolvedTensorLayout, ResolvedTensorSpec};
pub const COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID: &str =
    "operation.token_embedding.comfy_convrot_i8.f32";
pub const COMFY_CONVROT_INT8_EMBEDDING_F32_CAPABILITY_ID: &str =
    "capability.operation.token_embedding.comfy_convrot_i8.f32";

/// U32 IDs[T], canonical rotated I8 table[V,K], F32 scales[V,1] -> F32[T,K].
/// Each selected code first multiplies its row scale with RN32; then the
/// inverse regular H256 uses four F32 H4 stages with a 1/2 factor at each
/// stage. No A8 quantization or GEMM is performed. Finally the explicit
/// dequantized_dtype selects full F32 or BF16 RNE followed by exact F32
/// promotion, matching the source layout's orig_dtype boundary. This cannot
/// be inferred from an I8 checkpoint's storage dtype. No FTZ/FMA is permitted.
/// Invalid U32 IDs produce a zero row without reading the table, matching the
/// ordinary dense token provider's safety rule. Product token validation must
/// reject invalid IDs; this differs from PyTorch's out-of-range exception.
/// This independent staged profile is not a bitwise promise for a different
/// F32 matrix-reduction implementation. There is no whole-table expansion,
/// invocation scratch or implicit allocation; output cannot alias any input.
pub fn comfy_convrot_int8_embedding_f32_contract() -> Result<StandardOperationContract, VNextError>
{
    let s = |name: &str| DimensionConstraint::Symbol(name.into());
    let descriptor = OperationDescriptor {
        id: OperationId::new(COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(vec![s("tokens")], [ElementType::U32], TensorAccess::Read)?,
            contiguous_tensor(
                vec![s("vocab_size"), s("hidden_size")],
                [ElementType::I8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![s("vocab_size"), DimensionConstraint::Exact(1)],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![s("tokens"), s("hidden_size")],
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("vocab_size")?,
            (
                AttributeId::new("dequantized_dtype")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Text,
                    required: true,
                    constraint: AttributeConstraint::TextChoices {
                        values: BTreeSet::from(["f32".into(), "bf16_then_f32".into()]),
                    },
                },
            ),
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            COMFY_CONVROT_INT8_EMBEDDING_F32_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComfyConvRotInt8EmbeddingF32Geometry {
    tokens: u64,
    width: u64,
    vocabulary: u64,
    elements: u64,
    arithmetic: u64,
    dequantized_dtype: ComfyEmbeddingDequantizedDtype,
}
impl ComfyConvRotInt8EmbeddingF32Geometry {
    pub fn new(
        ids: &ResolvedTensorSpec,
        table: &ResolvedTensorSpec,
        scales: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        width: u64,
        vocabulary: u64,
        dequantized_dtype: ComfyEmbeddingDequantizedDtype,
    ) -> Result<Self, VNextError> {
        let [tokens] = ids.dimensions() else {
            return Err(invalid("token IDs must have rank one"));
        };
        if *tokens == 0
            || width == 0
            || !width.is_multiple_of(256)
            || vocabulary == 0
            || vocabulary > u32::MAX as u64
            || table.dimensions() != [vocabulary, width]
            || scales.dimensions() != [vocabulary, 1]
            || output.dimensions() != [*tokens, width]
            || ids.element_type() != ElementType::U32
            || table.element_type() != ElementType::I8
            || scales.element_type() != ElementType::F32
            || output.element_type() != ElementType::F32
            || [ids, table, scales, output]
                .iter()
                .any(|t| t.layout() != &ResolvedTensorLayout::Contiguous)
        {
            return Err(invalid("requires contiguous U32[T], I8[V,K], F32[V,1], F32[T,K], positive T/V, u32 vocabulary and K%256=0"));
        }
        for tensor in [ids, table, scales, output] {
            if tensor.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let elements = tokens
            .checked_mul(width)
            .ok_or_else(|| invalid("output count overflows"))?;
        // One scale multiply plus four radix-4 stages of 3 adds + 1 multiply.
        let arithmetic = elements
            .checked_mul(17)
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            tokens: *tokens,
            width,
            vocabulary,
            elements,
            arithmetic,
            dequantized_dtype,
        })
    }
    pub const fn tokens(self) -> u64 {
        self.tokens
    }
    pub const fn width(self) -> u64 {
        self.width
    }
    pub const fn vocabulary(self) -> u64 {
        self.vocabulary
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
    pub const fn dequantized_dtype(self) -> ComfyEmbeddingDequantizedDtype {
        self.dequantized_dtype
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("Comfy ConvRot INT8 embedding: {reason}"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn embedding_geometry_preserves_token_dtype_scale_scope_and_zero_scratch() {
        let ids = t(&[6], ElementType::U32);
        let table = t(&[4, 512], ElementType::I8);
        let scales = t(&[4, 1], ElementType::F32);
        let output = t(&[6, 512], ElementType::F32);
        let g = ComfyConvRotInt8EmbeddingF32Geometry::new(
            &ids,
            &table,
            &scales,
            &output,
            512,
            4,
            ComfyEmbeddingDequantizedDtype::F32,
        )
        .unwrap();
        assert_eq!(g.elements(), 3072);
        for bad in [
            t(&[4], ElementType::F32),
            t(&[1, 1], ElementType::F32),
            t(&[4, 1], ElementType::Bf16),
        ] {
            assert!(ComfyConvRotInt8EmbeddingF32Geometry::new(
                &ids,
                &table,
                &bad,
                &output,
                512,
                4,
                ComfyEmbeddingDequantizedDtype::F32
            )
            .is_err());
        }
        assert!(ComfyConvRotInt8EmbeddingF32Geometry::new(
            &t(&[6], ElementType::I32),
            &table,
            &scales,
            &output,
            512,
            4,
            ComfyEmbeddingDequantizedDtype::F32
        )
        .is_err());
        assert!(ComfyConvRotInt8EmbeddingF32Geometry::new(
            &ids,
            &table,
            &scales,
            &output,
            256,
            4,
            ComfyEmbeddingDequantizedDtype::F32
        )
        .is_err());
        let large = ((isize::MAX as u64) / 24 / 256 + 1) * 256;
        assert!(ComfyConvRotInt8EmbeddingF32Geometry::new(
            &ids,
            &t(&[4, large], ElementType::I8),
            &scales,
            &t(&[6, large], ElementType::F32),
            large,
            4,
            ComfyEmbeddingDequantizedDtype::F32
        )
        .is_err());
        let c = comfy_convrot_int8_embedding_f32_contract().unwrap();
        assert_eq!(
            c.descriptor().resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
