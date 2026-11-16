//! Fixed-tensor sparse attention with an explicitly bound immutable Host pattern.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const SPARSE_FP4_ATTENTION_CHECKED_HOST_OPERATION_ID: &str =
    "operation.sparse_fp4_attention.checked_host.bf16";
pub const SPARSE_FP4_ATTENTION_CHECKED_HOST_CAPABILITY_ID: &str =
    "capability.operation.sparse_fp4_attention.checked_host.bf16";
pub const SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID: &str =
    "operation.sparse_fp4_attention.region_topk.bf16";
pub const SPARSE_FP4_ATTENTION_REGION_TOPK_CAPABILITY_ID: &str =
    "capability.operation.sparse_fp4_attention.region_topk.bf16";

/// Six U8 inputs are Q[B,H,Q,64], K[B,H,K,64], V[B,H,128,K/2], and
/// their three scale arrays [B,H,Q,8], [B,H,K,8], [B,H,128,K/16].
/// Physical layouts are the declared sparse-FP4 provider's Q/K/V layouts,
/// not the NVFP4 matrix operation's layout. Output is BF16[B,H,Q,128].
/// Q/K are multiples of 128; D is fixed to 128. This profile has no LSE
/// output: the fixed source does not write it on the normal noncausal path.
///
/// The SHA256 attribute refers to an explicitly supplied checked immutable
/// Host payload. A hash alone supplies no data or authority: the selected
/// estimator must bind the matching charged owner through prepared retention.
/// This profile does not accept arbitrary device I32 lists and does not imply
/// a dynamic GPU mask producer. Provider numerical profiles declare QKV/P
/// quantization, online-softmax accumulation and final output rounding.
pub fn sparse_fp4_attention_checked_host_contract() -> Result<StandardOperationContract, VNextError>
{
    sparse_fp4_contract(false)
}

/// The six packed operands are identical to the checked Host operation. A
/// seventh F32[B,H,Kblocks,Kblocks] operand supplies original pooled scores.
/// Query blocks are the nonempty suffix after a nonempty dense prefix.
/// `pattern_sha256` binds immutable checked region ranges/budgets/valid counts,
/// not dynamic lists. The declared producer generates those lists in private
/// admitted scratch and consumes them on the same stream. Scores are unchanged
/// and can also feed compression. The provider declares its top-k tie and
/// nonfinite profile; finite qualification does not prove all inputs finite.
pub fn sparse_fp4_attention_region_topk_contract() -> Result<StandardOperationContract, VNextError>
{
    sparse_fp4_contract(true)
}
fn sparse_fp4_contract(region_topk: bool) -> Result<StandardOperationContract, VNextError> {
    let (operation, capability) = if region_topk {
        (
            SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID,
            SPARSE_FP4_ATTENTION_REGION_TOPK_CAPABILITY_ID,
        )
    } else {
        (
            SPARSE_FP4_ATTENTION_CHECKED_HOST_OPERATION_ID,
            SPARSE_FP4_ATTENTION_CHECKED_HOST_CAPABILITY_ID,
        )
    };
    let shape = |last: DimensionConstraint, row: DimensionConstraint| {
        vec![symbol("b"), symbol("h"), row, last]
    };
    let attributes = BTreeMap::from([
        positive_rational_attribute("scale")?,
        (
            AttributeId::new("pattern_sha256")?,
            AttributeSpec {
                value_kind: AttributeValueKind::Text,
                required: true,
                constraint: AttributeConstraint::None,
            },
        ),
    ]);
    let mut descriptor = OperationDescriptor {
        id: OperationId::new(operation)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                shape(DimensionConstraint::Exact(64), symbol("q")),
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                shape(DimensionConstraint::Exact(64), symbol("k")),
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                shape(symbol("packed_k"), DimensionConstraint::Exact(128)),
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                shape(DimensionConstraint::Exact(8), symbol("q")),
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                shape(DimensionConstraint::Exact(8), symbol("k")),
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                shape(symbol("scale_k"), DimensionConstraint::Exact(128)),
                [ElementType::U8],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            shape(DimensionConstraint::Exact(128), symbol("q")),
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attributes)?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Required,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(capability, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    if region_topk {
        descriptor.inputs.push(contiguous_tensor(
            vec![symbol("b"), symbol("h"), symbol("tiles"), symbol("tiles")],
            [ElementType::F32],
            TensorAccess::Read,
        )?);
    }
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseFp4AttentionGeometry {
    batches: u32,
    heads: u32,
    query_rows: u32,
    key_rows: u32,
    scale: CanonicalRational,
    pattern_sha256: String,
}
impl SparseFp4AttentionGeometry {
    pub fn with_region_scores(
        inputs: [&ResolvedTensorSpec; 7],
        outputs: [&ResolvedTensorSpec; 1],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let [q, k, v, sq, sk, sv, scores] = inputs;
        let geometry = Self::new([q, k, v, sq, sk, sv], outputs, attributes)?;
        let tiles = u64::from(geometry.key_blocks());
        if geometry.query_blocks() >= geometry.key_blocks()
            || scores.dimensions()
                != [
                    u64::from(geometry.batches),
                    u64::from(geometry.heads),
                    tiles,
                    tiles,
                ]
            || scores.element_type() != ElementType::F32
            || scores.layout() != &ResolvedTensorLayout::Contiguous
            || scores.minimum_storage_bytes()? > isize::MAX as u64
        {
            return Err(invalid(
                "region scores require full F32 B/H/Ktiles/Ktiles and nonempty dense prefix",
            ));
        }
        Ok(geometry)
    }
    pub fn new(
        inputs: [&ResolvedTensorSpec; 6],
        outputs: [&ResolvedTensorSpec; 1],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let [b, h, q, packed_d] = inputs[0].dimensions() else {
            return Err(invalid("Q must be rank four"));
        };
        let [kb, kh, k, kd] = inputs[1].dimensions() else {
            return Err(invalid("K must be rank four"));
        };
        if [b, h, q, k].iter().any(|&&v| v == 0 || v > i32::MAX as u64)
            || b != kb
            || h != kh
            || *packed_d != 64
            || *kd != 64
            || q % 128 != 0
            || k % 128 != 0
            || q > k
        {
            return Err(invalid(
                "positive matching batch/heads, tile128 Q/K and D128 are required",
            ));
        }
        let input_shapes = [
            vec![*b, *h, *q, 64],
            vec![*b, *h, *k, 64],
            vec![*b, *h, 128, k / 2],
            vec![*b, *h, *q, 8],
            vec![*b, *h, *k, 8],
            vec![*b, *h, 128, k / 16],
        ];
        let output_shapes = [vec![*b, *h, *q, 128]];
        if inputs
            .iter()
            .zip(&input_shapes)
            .any(|(t, s)| t.dimensions() != s || t.element_type() != ElementType::U8)
            || outputs
                .iter()
                .zip(&output_shapes)
                .zip([ElementType::Bf16])
                .any(|((t, s), ty)| t.dimensions() != s || t.element_type() != ty)
            || inputs
                .into_iter()
                .chain(outputs)
                .any(|t| t.layout() != &ResolvedTensorLayout::Contiguous)
        {
            return Err(invalid(
                "exact contiguous packed/scaled input and output extents required",
            ));
        }
        for t in inputs.into_iter().chain(outputs) {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let scale_id = AttributeId::new("scale")?;
        let pattern_id = AttributeId::new("pattern_sha256")?;
        let Some(SemanticValue::Rational(scale)) = attributes.get(&scale_id) else {
            return Err(invalid("scale must be rational"));
        };
        let Some(SemanticValue::Text(pattern)) = attributes.get(&pattern_id) else {
            return Err(invalid("checked pattern digest is absent"));
        };
        let effective = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
        if attributes.len() != 2
            || scale.numerator() <= 0
            || !effective.is_finite()
            || effective <= 0.0
            || !(effective * std::f32::consts::LOG2_E).is_finite()
            || pattern.len() != 64
            || !pattern
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(invalid(
                "positive finite scale and canonical pattern SHA256 required",
            ));
        }
        Ok(Self {
            batches: *b as u32,
            heads: *h as u32,
            query_rows: *q as u32,
            key_rows: *k as u32,
            scale: scale.clone(),
            pattern_sha256: pattern.clone(),
        })
    }
    pub const fn batches(&self) -> u32 {
        self.batches
    }
    pub const fn heads(&self) -> u32 {
        self.heads
    }
    pub const fn query_blocks(&self) -> u32 {
        self.query_rows / 128
    }
    pub const fn key_blocks(&self) -> u32 {
        self.key_rows / 128
    }
    pub const fn query_rows(&self) -> u32 {
        self.query_rows
    }
    pub const fn key_rows(&self) -> u32 {
        self.key_rows
    }
    pub fn softmax_scale(&self) -> f32 {
        (self.scale.numerator() as f64 / self.scale.denominator() as f64) as f32
    }
    pub fn pattern_sha256(&self) -> &str {
        &self.pattern_sha256
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("sparse FP4 checked Host attention: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    fn inputs() -> [ResolvedTensorSpec; 6] {
        [
            [2, 3, 256, 64],
            [2, 3, 512, 64],
            [2, 3, 128, 256],
            [2, 3, 256, 8],
            [2, 3, 512, 8],
            [2, 3, 128, 32],
        ]
        .map(|shape| tensor(&shape, ElementType::U8))
    }
    fn outputs() -> [ResolvedTensorSpec; 1] {
        [tensor(&[2, 3, 256, 128], ElementType::Bf16)]
    }
    fn attributes() -> BTreeMap<AttributeId, SemanticValue> {
        BTreeMap::from([
            (
                AttributeId::new("scale").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
            ),
            (
                AttributeId::new("pattern_sha256").unwrap(),
                SemanticValue::Text("ab".repeat(32)),
            ),
        ])
    }
    #[test]
    fn sparse_fp4_region_scores_require_full_domain_and_distinct_contract() {
        let packed = inputs();
        let outputs = outputs();
        let [q, k, v, sq, sk, sv] = packed.each_ref();
        let scores = tensor(&[2, 3, 4, 4], ElementType::F32);
        assert!(SparseFp4AttentionGeometry::with_region_scores(
            [q, k, v, sq, sk, sv, &scores],
            outputs.each_ref(),
            &attributes()
        )
        .is_ok());
        for scores in [
            tensor(&[2, 3, 2, 4], ElementType::F32),
            tensor(&[2, 3, 4, 4], ElementType::Bf16),
            tensor(&[2, 2, 4, 4], ElementType::F32),
        ] {
            assert!(SparseFp4AttentionGeometry::with_region_scores(
                [q, k, v, sq, sk, sv, &scores],
                outputs.each_ref(),
                &attributes()
            )
            .is_err());
        }
        let c = sparse_fp4_attention_region_topk_contract().unwrap();
        assert_eq!(c.descriptor().inputs.len(), 7);
        assert_ne!(
            c.descriptor().fingerprint().unwrap(),
            sparse_fp4_attention_checked_host_contract()
                .unwrap()
                .descriptor()
                .fingerprint()
                .unwrap()
        );
    }
    #[test]
    fn sparse_fp4_geometry_preserves_rectangular_extents_and_only_initialized_output() {
        let g = SparseFp4AttentionGeometry::new(
            inputs().each_ref(),
            outputs().each_ref(),
            &attributes(),
        )
        .unwrap();
        assert_eq!(
            (g.batches(), g.heads(), g.query_blocks(), g.key_blocks()),
            (2, 3, 2, 4)
        );
        assert_eq!(g.softmax_scale(), 0.125);
        let contract = sparse_fp4_attention_checked_host_contract().unwrap();
        for output in &contract.descriptor().outputs {
            assert_eq!(output.alias(), &AliasPolicy::NoAlias);
        }
        assert_eq!(
            contract.descriptor().resources.scratch,
            ResourcePresenceRequirement::Required
        );
        for (index, bad) in [
            (0, tensor(&[2, 3, 255, 64], ElementType::U8)),
            (2, tensor(&[2, 3, 512, 64], ElementType::U8)), // same byte count, wrong transpose
            (3, tensor(&[2, 3, 256, 8], ElementType::F32)),
            (5, tensor(&[2, 3, 128, 31], ElementType::U8)),
        ] {
            let mut input = inputs();
            input[index] = bad;
            assert!(SparseFp4AttentionGeometry::new(
                input.each_ref(),
                outputs().each_ref(),
                &attributes()
            )
            .is_err());
        }
        let mut output = outputs();
        output[0] = tensor(&[2, 3, 256, 128], ElementType::F32);
        assert!(SparseFp4AttentionGeometry::new(
            inputs().each_ref(),
            output.each_ref(),
            &attributes()
        )
        .is_err());
    }
    #[test]
    fn sparse_fp4_semantic_identity_rejects_missing_noncanonical_or_extra_attributes() {
        for digest in ["a".repeat(63), "AB".repeat(32), "z".repeat(64)] {
            let mut attrs = attributes();
            attrs.insert(
                AttributeId::new("pattern_sha256").unwrap(),
                SemanticValue::Text(digest),
            );
            assert!(SparseFp4AttentionGeometry::new(
                inputs().each_ref(),
                outputs().each_ref(),
                &attrs
            )
            .is_err());
        }
        let mut attrs = attributes();
        attrs.remove(&AttributeId::new("pattern_sha256").unwrap());
        assert!(
            SparseFp4AttentionGeometry::new(inputs().each_ref(), outputs().each_ref(), &attrs)
                .is_err()
        );
        attrs = attributes();
        attrs.insert(
            AttributeId::new("skip_validation").unwrap(),
            SemanticValue::Text("true".into()),
        );
        assert!(
            SparseFp4AttentionGeometry::new(inputs().each_ref(), outputs().each_ref(), &attrs)
                .is_err()
        );
        attrs = attributes();
        attrs.insert(
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(0, 1).unwrap()),
        );
        assert!(
            SparseFp4AttentionGeometry::new(inputs().each_ref(), outputs().each_ref(), &attrs)
                .is_err()
        );
    }
}
