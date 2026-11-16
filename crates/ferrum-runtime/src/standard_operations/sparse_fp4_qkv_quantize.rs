//! Explicit attention FP4 preparation, independent of smoothing and selection.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID: &str =
    "operation.sparse_fp4_qkv_quantize.bf16";
pub const SPARSE_FP4_QKV_QUANTIZE_BF16_CAPABILITY_ID: &str =
    "capability.operation.sparse_fp4_qkv_quantize.bf16";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseFp4QkvQuantizeProfile {
    SageGroup16FastMathV1,
}
impl SparseFp4QkvQuantizeProfile {
    pub const fn as_str(self) -> &'static str {
        "sage_group16_fast_math_v1"
    }
    pub fn attributes(
        self,
        heads: u64,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let attributes = BTreeMap::from([
            (AttributeId::new("heads")?, SemanticValue::Unsigned(heads)),
            (
                AttributeId::new("profile")?,
                SemanticValue::Text(self.as_str().into()),
            ),
        ]);
        sparse_fp4_qkv_quantize_bf16_contract()?
            .descriptor
            .validate_attributes(&attributes)?;
        Ok(attributes)
    }
}

/// BF16 Q[B,Q,H*128], K/V[B,K,H*128] -> six physical U8 tensors accepted by
/// sparse_fp4_attention: packed Q/K[B,H,L,64], V[B,H,128,K/2], SFQ/SFK[B,H,L,8]
/// and SFV[B,H,128,K/16]. Q/K lengths are positive multiples of 128. Query rows
/// are exactly those supplied: this operation neither finds nor removes prefix
/// queries. K centering is likewise an explicit earlier graph operation; pooled
/// QK scores must retain the original pre-centering values.
///
/// Q/K quantize consecutive 16-channel groups, with K's original within-32-row
/// permutation; V quantizes consecutive 16-sequence-row groups and transposes.
/// Scale bytes use the Sage 64-row/4-column swizzle, not a GEMM scale layout.
/// Group amax/6 is rounded to E4M3, decoded to F32, inverted (zero scale -> zero),
/// then multiplied into decoded BF16 before E2M1 RN satfinite packing. The
/// declared provider fixes the source compiler's fast-math lowering. There is
/// no global scale, alpha, host readback, hidden padding or temporary tensor.
/// Each source tensor's element count must fit the native signed 32-bit
/// indexing domain; this is checked before any device memory is admitted.
pub fn sparse_fp4_qkv_quantize_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let input = |rows| {
        contiguous_tensor(
            vec![symbol("b"), symbol(rows), symbol("c")],
            [ElementType::Bf16],
            TensorAccess::Read,
        )
    };
    let output = |dimensions| contiguous_tensor(dimensions, [ElementType::U8], TensorAccess::Write);
    let descriptor = OperationDescriptor {
        id: OperationId::new(SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![input("q")?, input("k")?, input("k")?],
        outputs: vec![
            output(vec![
                symbol("b"),
                symbol("h"),
                symbol("q"),
                DimensionConstraint::Exact(64),
            ])?,
            output(vec![
                symbol("b"),
                symbol("h"),
                symbol("k"),
                DimensionConstraint::Exact(64),
            ])?,
            output(vec![
                symbol("b"),
                symbol("h"),
                DimensionConstraint::Exact(128),
                symbol("packed_k"),
            ])?,
            output(vec![
                symbol("b"),
                symbol("h"),
                symbol("q"),
                DimensionConstraint::Exact(8),
            ])?,
            output(vec![
                symbol("b"),
                symbol("h"),
                symbol("k"),
                DimensionConstraint::Exact(8),
            ])?,
            output(vec![
                symbol("b"),
                symbol("h"),
                DimensionConstraint::Exact(128),
                symbol("scale_k"),
            ])?,
        ],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("heads")?,
            text_choices_attribute(
                "profile",
                [SparseFp4QkvQuantizeProfile::SageGroup16FastMathV1.as_str()],
            )?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            SPARSE_FP4_QKV_QUANTIZE_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseFp4QkvQuantizeGeometry {
    batches: u32,
    heads: u32,
    query_rows: u32,
    key_rows: u32,
}
impl SparseFp4QkvQuantizeGeometry {
    pub fn new(
        inputs: [&ResolvedTensorSpec; 3],
        outputs: [&ResolvedTensorSpec; 6],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        sparse_fp4_qkv_quantize_bf16_contract()?
            .descriptor
            .validate_attributes(attributes)?;
        let Some(SemanticValue::Unsigned(h)) = attributes.get(&AttributeId::new("heads")?) else {
            return Err(invalid(
                "heads must be an explicit positive unsigned attribute",
            ));
        };
        let [b, q, c] = inputs[0].dimensions() else {
            return Err(invalid("Q must be BF16[B,Q,H*128]"));
        };
        let [kb, k, kc] = inputs[1].dimensions() else {
            return Err(invalid("K must be BF16[B,K,H*128]"));
        };
        if [*b, *h, *q, *k]
            .iter()
            .any(|&v| v == 0 || v > u32::MAX as u64)
            || b != kb
            || c != kc
            || h.checked_mul(128) != Some(*c)
            || q % 128 != 0
            || k % 128 != 0
        {
            return Err(invalid(
                "matching B/H*128, positive heads and tile128 Q/K are required",
            ));
        }
        let input_shapes = [vec![*b, *q, *c], vec![*b, *k, *c], vec![*b, *k, *c]];
        let output_shapes = [
            vec![*b, *h, *q, 64],
            vec![*b, *h, *k, 64],
            vec![*b, *h, 128, k / 2],
            vec![*b, *h, *q, 8],
            vec![*b, *h, *k, 8],
            vec![*b, *h, 128, k / 16],
        ];
        for (t, shape) in inputs.into_iter().zip(input_shapes) {
            if t.dimensions() != shape || t.element_type() != ElementType::Bf16 {
                return Err(invalid("exact sequence-major BF16 Q/K/V extents required"));
            }
            if t.minimum_storage_bytes()? / 2 > i32::MAX as u64 {
                return Err(invalid(
                    "source elements exceed native signed indexing capacity",
                ));
            }
        }
        for (t, shape) in outputs.into_iter().zip(output_shapes) {
            if t.dimensions() != shape || t.element_type() != ElementType::U8 {
                return Err(invalid(
                    "exact packed Q/K/V and scale byte extents required",
                ));
            }
        }
        for t in inputs.into_iter().chain(outputs) {
            if t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid(
                    "contiguous addressable quantization spans required",
                ));
            }
        }
        Ok(Self {
            batches: *b as u32,
            heads: *h as u32,
            query_rows: *q as u32,
            key_rows: *k as u32,
        })
    }
    pub const fn batches(self) -> u32 {
        self.batches
    }
    pub const fn heads(self) -> u32 {
        self.heads
    }
    pub const fn query_rows(self) -> u32 {
        self.query_rows
    }
    pub const fn key_rows(self) -> u32 {
        self.key_rows
    }
    pub fn input_elements(self) -> Result<u64, VNextError> {
        u64::from(self.key_rows)
            .checked_mul(2)
            .and_then(|v| v.checked_add(u64::from(self.query_rows)))
            .and_then(|v| v.checked_mul(u64::from(self.batches)))
            .and_then(|v| v.checked_mul(u64::from(self.heads)))
            .and_then(|v| v.checked_mul(128))
            .ok_or_else(|| invalid("quantization work extent overflows"))
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("sparse FP4 QKV quantization: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: Vec<u64>, ty: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape, ty, ResolvedTensorLayout::Contiguous).unwrap()
    }
    fn ports(b: u64, h: u64, q: u64, k: u64) -> ([ResolvedTensorSpec; 3], [ResolvedTensorSpec; 6]) {
        (
            [
                tensor(vec![b, q, h * 128], ElementType::Bf16),
                tensor(vec![b, k, h * 128], ElementType::Bf16),
                tensor(vec![b, k, h * 128], ElementType::Bf16),
            ],
            [
                vec![b, h, q, 64],
                vec![b, h, k, 64],
                vec![b, h, 128, k / 2],
                vec![b, h, q, 8],
                vec![b, h, k, 8],
                vec![b, h, 128, k / 16],
            ]
            .map(|s| tensor(s, ElementType::U8)),
        )
    }
    #[test]
    fn sparse_fp4_quantize_declares_independent_sequence_inputs_and_original_packed_outputs() {
        for (b, h, q, k) in [(2, 3, 256, 384), (1, 2, 256, 128)] {
            let (i, o) = ports(b, h, q, k);
            let attrs = SparseFp4QkvQuantizeProfile::SageGroup16FastMathV1
                .attributes(h)
                .unwrap();
            let g = SparseFp4QkvQuantizeGeometry::new(i.each_ref(), o.each_ref(), &attrs).unwrap();
            assert_eq!(
                (g.batches(), g.heads(), g.query_rows(), g.key_rows()),
                (b as u32, h as u32, q as u32, k as u32)
            );
            assert_eq!(g.input_elements().unwrap(), b * h * (q + 2 * k) * 128);
            let bytes = o
                .iter()
                .map(|s| s.minimum_storage_bytes().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                bytes,
                vec![
                    b * h * q * 64,
                    b * h * k * 64,
                    b * h * k * 64,
                    b * h * q * 8,
                    b * h * k * 8,
                    b * h * k * 8
                ]
            );
        }
        let contract = sparse_fp4_qkv_quantize_bf16_contract().unwrap();
        for output in &contract.descriptor().outputs {
            assert_eq!(output.alias(), &AliasPolicy::NoAlias);
        }
    }
    #[test]
    fn sparse_fp4_quantize_rejects_axis_reinterpretation_wrong_layouts_profiles_and_native_capacity(
    ) {
        let (i, o) = ports(2, 3, 256, 384);
        let attrs = SparseFp4QkvQuantizeProfile::SageGroup16FastMathV1
            .attributes(3)
            .unwrap();
        for ordinal in 0..3 {
            let mut inputs = i.clone();
            let d = inputs[ordinal].dimensions();
            inputs[ordinal] = tensor(vec![d[0], 3, d[1], 128], ElementType::Bf16);
            assert!(
                SparseFp4QkvQuantizeGeometry::new(inputs.each_ref(), o.each_ref(), &attrs).is_err()
            );
        }
        for ordinal in 0..6 {
            let mut outputs = o.clone();
            let mut shape = outputs[ordinal].dimensions().to_vec();
            shape[0] *= shape[1];
            shape[1] = 1; // same byte count, wrong B/H semantics
            outputs[ordinal] = tensor(shape, ElementType::U8);
            assert!(
                SparseFp4QkvQuantizeGeometry::new(i.each_ref(), outputs.each_ref(), &attrs)
                    .is_err()
            );
        }
        let mut wrong = attrs.clone();
        wrong.insert(
            AttributeId::new("profile").unwrap(),
            SemanticValue::Text("nvfp4_gemm".into()),
        );
        assert!(SparseFp4QkvQuantizeGeometry::new(i.each_ref(), o.each_ref(), &wrong).is_err());
        assert!(SparseFp4QkvQuantizeProfile::SageGroup16FastMathV1
            .attributes(0)
            .is_err());
        let (i, o) = ports(1, 1, 128, 129);
        assert!(SparseFp4QkvQuantizeGeometry::new(
            i.each_ref(),
            o.each_ref(),
            &SparseFp4QkvQuantizeProfile::SageGroup16FastMathV1
                .attributes(1)
                .unwrap()
        )
        .is_err());
        // 2^31 elements fit a u64 tensor extent but overflow the native index.
        let (i, o) = ports(1, 1, 128, 1 << 24);
        assert!(SparseFp4QkvQuantizeGeometry::new(
            i.each_ref(),
            o.each_ref(),
            &SparseFp4QkvQuantizeProfile::SageGroup16FastMathV1
                .attributes(1)
                .unwrap()
        )
        .is_err());
    }
}
