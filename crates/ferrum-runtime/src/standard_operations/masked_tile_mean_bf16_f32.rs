//! Masked tile reduction with explicit prefix validity and output head order.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const MASKED_TILE_MEAN_BF16_F32_OPERATION_ID: &str = "operation.masked_tile_mean.bf16_f32";
pub const MASKED_TILE_MEAN_BF16_F32_CAPABILITY_ID: &str =
    "capability.operation.masked_tile_mean.bf16_f32";
pub const MASKED_TILE_MEAN_BF16_F32_PROFILE: &str = "f32_ascending_masked_rn_v1";

/// Continuous BF16 X[B,T*R,H*D] and I32 counts[T] produce F32 Y[B,H,T,D].
/// Only `heads=H` is supplied; T is the count extent, R and D are checked
/// integer quotients. Counts are shared across batches and heads.
///
/// For 1 <= count <= R, start at positive zero and visit r=0..R in order.
/// Add the exactly promoted BF16 element when r<count, otherwise positive
/// zero without reading X. Each addition rounds F32; divide by F32(count)
/// with IEEE round-to-nearest-even and gradual underflow. There is no BF16
/// output boundary and no assumption that padding contains finite values.
/// For count<=0 or count>R, every output in that tile is canonical positive
/// quiet NaN (0x7fc00000), and no source element is read. Counts never index
/// storage. This total invalid-count behavior is not a layout-validity proof;
/// model sources must validate their original positive prefix lengths.
pub fn masked_tile_mean_bf16_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(MASKED_TILE_MEAN_BF16_F32_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![symbol("batch"), symbol("source_rows"), symbol("columns")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("tiles")],
                [ElementType::I32],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![
                symbol("batch"),
                symbol("heads"),
                symbol("tiles"),
                symbol("head_dim"),
            ],
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            (
                AttributeId::new("heads")?,
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
                        values: BTreeSet::from([MASKED_TILE_MEAN_BF16_F32_PROFILE.to_owned()]),
                    },
                },
            ),
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            MASKED_TILE_MEAN_BF16_F32_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskedTileMeanBf16F32Geometry {
    batch: u64,
    source_rows: u64,
    columns: u64,
    heads: u64,
    tiles: u64,
    tile_rows: u64,
    head_dim: u64,
}
impl MaskedTileMeanBf16F32Geometry {
    pub fn attributes(heads: u64) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([
            (AttributeId::new("heads")?, SemanticValue::Unsigned(heads)),
            (
                AttributeId::new("profile")?,
                SemanticValue::Text(MASKED_TILE_MEAN_BF16_F32_PROFILE.into()),
            ),
        ]))
    }
    pub fn new(
        x: &ResolvedTensorSpec,
        counts: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let [batch, source_rows, columns] = x.dimensions() else {
            return Err(invalid("source must have rank three"));
        };
        let [tiles] = counts.dimensions() else {
            return Err(invalid("counts must have rank one"));
        };
        let Some(SemanticValue::Unsigned(heads)) = attrs.get(&AttributeId::new("heads")?) else {
            return Err(invalid("missing heads"));
        };
        if attrs != &Self::attributes(*heads)?
            || *heads == 0
            || [batch, source_rows, columns, tiles]
                .iter()
                .any(|&&n| n == 0)
            || source_rows % tiles != 0
            || columns % heads != 0
            || source_rows / tiles > (1 << 24)
            || x.element_type() != ElementType::Bf16
            || counts.element_type() != ElementType::I32
            || y.element_type() != ElementType::F32
            || y.dimensions() != [*batch, *heads, *tiles, columns / heads]
        {
            return Err(invalid("requires BF16 [B,T*R,H*D], I32 [T], F32 [B,H,T,D], positive exact quotients and R<=2^24"));
        }
        for t in [x, counts, y] {
            if t.layout() != &ResolvedTensorLayout::Contiguous
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid("requires contiguous addressable extents"));
            }
        }
        let g = Self {
            batch: *batch,
            source_rows: *source_rows,
            columns: *columns,
            heads: *heads,
            tiles: *tiles,
            tile_rows: source_rows / tiles,
            head_dim: columns / heads,
        };
        g.output_elements()
            .checked_mul(g.tile_rows + 1)
            .ok_or_else(|| invalid("arithmetic count overflows"))?;
        Ok(g)
    }
    pub const fn batch(self) -> u64 {
        self.batch
    }
    pub const fn source_rows(self) -> u64 {
        self.source_rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn heads(self) -> u64 {
        self.heads
    }
    pub const fn tiles(self) -> u64 {
        self.tiles
    }
    pub const fn tile_rows(self) -> u64 {
        self.tile_rows
    }
    pub const fn head_dim(self) -> u64 {
        self.head_dim
    }
    pub fn input_bytes(self) -> u64 {
        self.batch * self.source_rows * self.columns * 2
    }
    pub fn count_bytes(self) -> u64 {
        self.tiles * 4
    }
    pub fn output_elements(self) -> u64 {
        self.batch * self.heads * self.tiles * self.head_dim
    }
    pub fn output_bytes(self) -> u64 {
        self.output_elements() * 4
    }
    /// Nominal semantic arithmetic for valid counts: every masked tile row
    /// contributes one F32 addition, followed by one F32 division.
    pub fn arithmetic_operations(self) -> u64 {
        self.output_elements() * (self.tile_rows + 1)
    }
}
fn invalid(s: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("masked tile mean: {s}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn masked_tile_mean_checks_axes_profile_exact_quotients_and_divisor_domain() {
        let x = t(&[2, 21, 15], ElementType::Bf16);
        let c = t(&[3], ElementType::I32);
        let y = t(&[2, 3, 3, 5], ElementType::F32);
        let a = MaskedTileMeanBf16F32Geometry::attributes(3).unwrap();
        let g = MaskedTileMeanBf16F32Geometry::new(&x, &c, &y, &a).unwrap();
        assert_eq!(
            (
                g.tile_rows(),
                g.head_dim(),
                g.input_bytes(),
                g.count_bytes(),
                g.output_bytes()
            ),
            (7, 5, 1260, 12, 360)
        );
        for (x, c, y) in [
            (x.clone(), t(&[3], ElementType::F32), y.clone()),
            (t(&[2, 22, 15], ElementType::Bf16), c.clone(), y.clone()),
            (t(&[2, 21, 14], ElementType::Bf16), c.clone(), y.clone()),
            (x.clone(), c.clone(), t(&[2, 3, 5, 3], ElementType::F32)),
        ] {
            assert!(MaskedTileMeanBf16F32Geometry::new(&x, &c, &y, &a).is_err());
        }
        assert!(
            MaskedTileMeanBf16F32Geometry::new(
                &t(&[1, (1 << 24) + 1, 1], ElementType::Bf16),
                &t(&[1], ElementType::I32),
                &t(&[1, 1, 1, 1], ElementType::F32),
                &MaskedTileMeanBf16F32Geometry::attributes(1).unwrap(),
            )
            .is_err(),
            "count divisor must be exactly representable in F32"
        );
        let mut bad = a.clone();
        bad.remove(&AttributeId::new("profile").unwrap());
        assert!(MaskedTileMeanBf16F32Geometry::new(&x, &c, &y, &bad).is_err());
        assert_eq!(
            masked_tile_mean_bf16_f32_contract()
                .unwrap()
                .descriptor()
                .outputs[0]
                .alias(),
            &AliasPolicy::NoAlias
        );
    }
}
