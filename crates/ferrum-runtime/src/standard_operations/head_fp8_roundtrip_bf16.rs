//! Per-token, per-head E4M3FN roundtrip with an explicit zero-scale rule.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID: &str = "operation.head_fp8_roundtrip.bf16";
pub const HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID: &str =
    "capability.operation.head_fp8_roundtrip.bf16";
pub const HEAD_FP8_ROUNDTRIP_BF16_PROFILE: &str = "amax_div448_zero_inverse_e4m3fn_rne_bf16_v1";
pub const INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID: &str =
    "operation.indexed_head_fp8_roundtrip.bf16";
pub const INDEXED_HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID: &str =
    "capability.operation.indexed_head_fp8_roundtrip.bf16";

/// Apply the same independent head arithmetic only to selected source rows.
/// BF16[B,S,C], I32[R] -> BF16[B,R,C]. A repeated index repeats the complete
/// source head; invalid indices produce positive zero without reading source.
/// This is equivalent to the ordinary roundtrip followed by gather-or-zero,
/// including nonfinite and signed-zero rules, with no intermediate allocation.
pub fn indexed_head_fp8_roundtrip_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = head_fp8_roundtrip_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID)?;
    descriptor.inputs.push(contiguous_tensor(
        vec![symbol("selected_rows")],
        [ElementType::I32],
        TensorAccess::Read,
    )?);
    descriptor.outputs = vec![contiguous_tensor(
        vec![symbol("batch"), symbol("selected_rows"), symbol("columns")],
        [ElementType::Bf16],
        TensorAccess::Write,
    )?];
    descriptor.provider = provider_requirement(
        INDEXED_HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// X/Y are contiguous BF16[B,S,H*D]; each contiguous D-element head is independent.
/// Decode to F32; a=max(abs(x)) with NaN propagation; scale=div_profile(a,448),
/// inverse=scale==0 ? +0 : div_profile(1,scale). Multiply x*inverse in F32,
/// clamp to [-448,448] preserving NaNs, round to E4M3FN RNE/saturate-finite,
/// decode to F32, multiply by scale and RNE-convert once to BF16. No epsilon,
/// zero-scale replacement with one, or elimination of either division.
/// CUDA uses div.full.f32 without FTZ, as in the pinned Inductor implementation;
/// Reference uses RN F32 division. Provider fingerprints distinguish these
/// numerical approximations: this contract does not assert CPU/GPU bit parity.
/// Other F32 arithmetic preserves subnormals. Tiny nonzero scales may overflow
/// the inverse, so even finite input can yield NaN for a zero lane (0*infinity).
/// NaN payloads are unspecified; signed zero is preserved. No input/output alias.
pub fn head_fp8_roundtrip_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let shape = || vec![symbol("batch"), symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            shape(),
            [ElementType::Bf16],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor(
            shape(),
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            (
                AttributeId::new("head_dim")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Unsigned,
                    required: true,
                    constraint: AttributeConstraint::UnsignedRange {
                        minimum: 1,
                        maximum: u64::MAX,
                    },
                },
            ),
            (
                AttributeId::new("profile")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Text,
                    required: true,
                    constraint: AttributeConstraint::TextChoices {
                        values: BTreeSet::from([HEAD_FP8_ROUNDTRIP_BF16_PROFILE.into()]),
                    },
                },
            ),
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadFp8RoundtripBf16Geometry {
    elements: u64,
    head_dim: u64,
    arithmetic: u64,
    gather: Option<crate::RowIndexGatherBf16Geometry>,
}
impl HeadFp8RoundtripBf16Geometry {
    pub fn attributes(head_dim: u64) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        if head_dim == 0 {
            return Err(invalid("head dimension must be positive"));
        }
        Ok(BTreeMap::from([
            (
                AttributeId::new("head_dim")?,
                SemanticValue::Unsigned(head_dim),
            ),
            (
                AttributeId::new("profile")?,
                SemanticValue::Text(HEAD_FP8_ROUNDTRIP_BF16_PROFILE.into()),
            ),
        ]))
    }
    pub fn new(
        x: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let [b, s, c] = x.dimensions() else {
            return Err(invalid("requires rank three"));
        };
        let Some(SemanticValue::Unsigned(head_dim)) = attrs.get(&AttributeId::new("head_dim")?)
        else {
            return Err(invalid("missing head dimension"));
        };
        if [*b, *s, *c, *head_dim].contains(&0)
            || c % head_dim != 0
            || x != y
            || attrs != &Self::attributes(*head_dim)?
            || x.element_type() != ElementType::Bf16
            || x.layout() != &ResolvedTensorLayout::Contiguous
            || x.minimum_storage_bytes()? > isize::MAX as u64
        {
            return Err(invalid(
                "requires positive equal contiguous BF16[B,S,H*D] and explicit profile",
            ));
        }
        let elements = x.minimum_storage_bytes()? / 2;
        let arithmetic = elements
            .checked_mul(7)
            .and_then(|n| {
                (elements / head_dim)
                    .checked_mul(2)
                    .and_then(|v| n.checked_add(v))
            })
            .ok_or_else(|| invalid("arithmetic work overflows"))?;
        Ok(Self {
            elements,
            head_dim: *head_dim,
            arithmetic,
            gather: None,
        })
    }
    pub fn indexed(
        source: &ResolvedTensorSpec,
        indices: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let gather = crate::RowIndexGatherBf16Geometry::new(source, indices, output)?;
        let mut geometry = Self::new(output, output, attrs)?;
        geometry.gather = Some(gather);
        Ok(geometry)
    }
    pub const fn gather(self) -> Option<crate::RowIndexGatherBf16Geometry> {
        self.gather
    }
    pub fn source_bytes(self) -> u64 {
        self.gather.map_or(self.data_bytes(), |g| g.source_bytes())
    }
    pub fn operation_id(self) -> &'static str {
        if self.gather.is_some() {
            INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID
        } else {
            HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID
        }
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn head_dim(self) -> u64 {
        self.head_dim
    }
    pub fn groups(self) -> u64 {
        self.elements / self.head_dim
    }
    pub fn data_bytes(self) -> u64 {
        self.elements * 2
    }
    pub const fn arithmetic(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(s: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("head FP8 roundtrip: {s}"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexed_heads_keep_source_and_selected_extents_distinct() {
        let t = |shape, ty| {
            ResolvedTensorSpec::new(shape, ty, ResolvedTensorLayout::Contiguous).unwrap()
        };
        let source = t(vec![2, 17, 21], ElementType::Bf16);
        let indices = t(vec![5], ElementType::I32);
        let output = t(vec![2, 5, 21], ElementType::Bf16);
        let attrs = HeadFp8RoundtripBf16Geometry::attributes(7).unwrap();
        let g = HeadFp8RoundtripBf16Geometry::indexed(&source, &indices, &output, &attrs).unwrap();
        assert_eq!(
            (g.source_bytes(), g.data_bytes(), g.groups()),
            (1428, 420, 30)
        );
        for (indices, output) in [
            (t(vec![5], ElementType::F32), output.clone()),
            (indices.clone(), t(vec![2, 6, 21], ElementType::Bf16)),
            (indices.clone(), t(vec![2, 5, 14], ElementType::Bf16)),
        ] {
            assert!(
                HeadFp8RoundtripBf16Geometry::indexed(&source, &indices, &output, &attrs).is_err()
            );
        }
        assert!(HeadFp8RoundtripBf16Geometry::indexed(
            &source,
            &indices,
            &output,
            &HeadFp8RoundtripBf16Geometry::attributes(8).unwrap()
        )
        .is_err());
        assert_eq!(
            indexed_head_fp8_roundtrip_bf16_contract()
                .unwrap()
                .descriptor()
                .outputs[0]
                .alias(),
            &AliasPolicy::NoAlias
        );
    }
    #[test]
    fn fp8_heads_require_explicit_geometry_profile_and_bounded_work() {
        let spec =
            |d, ty| ResolvedTensorSpec::new(d, ty, ResolvedTensorLayout::Contiguous).unwrap();
        let x = spec(vec![2, 5, 21], ElementType::Bf16);
        let a = HeadFp8RoundtripBf16Geometry::attributes(7).unwrap();
        let g = HeadFp8RoundtripBf16Geometry::new(&x, &x, &a).unwrap();
        assert_eq!((g.groups(), g.data_bytes()), (30, 420));
        for bad in [
            BTreeMap::new(),
            HeadFp8RoundtripBf16Geometry::attributes(8).unwrap(),
        ] {
            assert!(HeadFp8RoundtripBf16Geometry::new(&x, &x, &bad).is_err());
        }
        assert!(
            HeadFp8RoundtripBf16Geometry::new(&x, &spec(vec![2, 5, 21], ElementType::F32), &a)
                .is_err()
        );
        let huge = spec(vec![1, 1, 1 << 61], ElementType::Bf16);
        assert!(HeadFp8RoundtripBf16Geometry::new(
            &huge,
            &huge,
            &HeadFp8RoundtripBf16Geometry::attributes(1).unwrap()
        )
        .is_err());
        let c = head_fp8_roundtrip_bf16_contract().unwrap();
        c.descriptor().validate_attributes(&a).unwrap();
        assert!(c
            .descriptor()
            .validate_attributes(&BTreeMap::new())
            .is_err());
        assert_eq!(c.descriptor().outputs[0].alias(), &AliasPolicy::NoAlias);
    }
}
