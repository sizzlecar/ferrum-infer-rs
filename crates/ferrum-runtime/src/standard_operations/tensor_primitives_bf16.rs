//! Model-independent BF16 affine, layout and segmented-attention primitives.
//! All shapes and row maps are explicit; no model names or inferred image roles.
use super::*;
use crate::{
    ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueBinding, ResolvedValueRole,
    SemanticValue,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorPrimitiveBf16 {
    AffineLinear,
    PatchProjection,
    AffineLayerNorm,
    GeluErf,
    HalfRotary,
    RowScatterAdd,
    ReshapeCopy,
    SegmentedAttention,
}

macro_rules! operations {
    ($(($variant:ident, $name:literal, $op:ident, $cap:ident, $function:ident)),+ $(,)?) => {
        $(pub const $op: &str = concat!("operation.", $name, ".bf16");
        pub const $cap: &str = concat!("capability.operation.", $name, ".bf16");
        pub fn $function() -> Result<StandardOperationContract, VNextError> {
            TensorPrimitiveBf16::$variant.contract()
        })+
        impl TensorPrimitiveBf16 {
            pub const ALL: [Self; 8] = [$(Self::$variant),+];
            pub const fn name(self) -> &'static str { match self { $(Self::$variant => $name),+ } }
            pub const fn operation_id(self) -> &'static str { match self { $(Self::$variant => $op),+ } }
            pub const fn capability_id(self) -> &'static str { match self { $(Self::$variant => $cap),+ } }
        }
    }
}
operations! {
    (AffineLinear, "affine_linear", AFFINE_LINEAR_BF16_OPERATION_ID, AFFINE_LINEAR_BF16_CAPABILITY_ID, affine_linear_bf16_contract),
    (PatchProjection, "patch_projection", PATCH_PROJECTION_BF16_OPERATION_ID, PATCH_PROJECTION_BF16_CAPABILITY_ID, patch_projection_bf16_contract),
    (AffineLayerNorm, "affine_layer_norm", AFFINE_LAYER_NORM_BF16_OPERATION_ID, AFFINE_LAYER_NORM_BF16_CAPABILITY_ID, affine_layer_norm_bf16_contract),
    (GeluErf, "gelu_erf", GELU_ERF_BF16_OPERATION_ID, GELU_ERF_BF16_CAPABILITY_ID, gelu_erf_bf16_contract),
    (HalfRotary, "half_rotary", HALF_ROTARY_BF16_OPERATION_ID, HALF_ROTARY_BF16_CAPABILITY_ID, half_rotary_bf16_contract),
    (RowScatterAdd, "row_scatter_add", ROW_SCATTER_ADD_BF16_OPERATION_ID, ROW_SCATTER_ADD_BF16_CAPABILITY_ID, row_scatter_add_bf16_contract),
    (ReshapeCopy, "reshape_copy", RESHAPE_COPY_BF16_OPERATION_ID, RESHAPE_COPY_BF16_CAPABILITY_ID, reshape_copy_bf16_contract),
    (SegmentedAttention, "segmented_attention", SEGMENTED_ATTENTION_BF16_OPERATION_ID, SEGMENTED_ATTENTION_BF16_CAPABILITY_ID, segmented_attention_bf16_contract),
}

impl TensorPrimitiveBf16 {
    pub const fn input_count(self) -> u32 {
        match self {
            Self::GeluErf | Self::ReshapeCopy => 1,
            Self::RowScatterAdd => 2,
            _ => 3,
        }
    }

    /// AffineLinear accumulates dot product and bias in FP32 before BF16.
    /// PatchProjection retains the Conv3d storage boundary: BF16(dot), then
    /// FP32 addition of BF16 bias, then BF16. Reduction order is provider-specific.
    /// Affine norm computes population variance in FP32, then scale and bias,
    /// with one final BF16 conversion. HalfRotary multiplies FP32 cos/sin and
    /// half-split BF16 values in FP32 before one BF16 conversion. Providers
    /// declare reduction/transcendental profiles; bitwise Torch parity is not
    /// implied. Segments in attention are mutually isolated and noncausal.
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let matrix = || vec![symbol("rows"), symbol("width")];
        let mut attrs = BTreeMap::new();
        let (inputs, output) = match self {
            Self::AffineLinear => {
                attrs.extend([
                    unsigned_attribute("in_features")?,
                    unsigned_attribute("out_features")?,
                ]);
                (
                    vec![
                        vec![symbol("rows"), symbol("inner")],
                        vec![symbol("out"), symbol("inner")],
                        vec![symbol("out")],
                    ],
                    vec![symbol("rows"), symbol("out")],
                )
            }
            Self::PatchProjection => (
                vec![
                    vec![symbol("rows"), symbol("flattened")],
                    vec![
                        symbol("out"),
                        symbol("channels"),
                        symbol("time"),
                        symbol("patch_h"),
                        symbol("patch_w"),
                    ],
                    vec![symbol("out")],
                ],
                vec![symbol("rows"), symbol("out")],
            ),
            Self::AffineLayerNorm => {
                attrs.extend([unsigned_attribute("hidden_size")?, rational("epsilon")?]);
                (
                    vec![matrix(), vec![symbol("width")], vec![symbol("width")]],
                    matrix(),
                )
            }
            Self::GeluErf => (vec![matrix()], matrix()),
            Self::HalfRotary => {
                attrs.extend([unsigned_attribute("head_dim")?]);
                (
                    vec![
                        matrix(),
                        vec![symbol("rows"), symbol("head_dim")],
                        vec![symbol("rows"), symbol("head_dim")],
                    ],
                    matrix(),
                )
            }
            Self::RowScatterAdd => {
                attrs.extend([integers("destination_starts")?, integers("run_lengths")?]);
                (
                    vec![matrix(), vec![symbol("updates"), symbol("width")]],
                    matrix(),
                )
            }
            Self::ReshapeCopy => {
                attrs.extend([unsigned_attribute("rows")?, unsigned_attribute("columns")?]);
                (
                    vec![matrix()],
                    vec![symbol("output_rows"), symbol("output_columns")],
                )
            }
            Self::SegmentedAttention => {
                attrs.extend([
                    unsigned_attribute("head_dim")?,
                    rational("scale")?,
                    integers("segment_lengths")?,
                ]);
                (vec![matrix(), matrix(), matrix()], matrix())
            }
        };
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(index, dims)| {
                let dtype = if self == Self::HalfRotary && index > 0 {
                    ElementType::F32
                } else {
                    ElementType::Bf16
                };
                contiguous_tensor(dims, [dtype], TensorAccess::Read)
            })
            .collect::<Result<_, _>>()?;
        let descriptor = OperationDescriptor {
            id: OperationId::new(self.operation_id())?,
            version: ContractVersion::new(1, 0),
            inputs,
            outputs: vec![contiguous_tensor(
                output,
                [ElementType::Bf16],
                TensorAccess::Write,
            )?],
            attributes: AttributeSchema::new(attrs)?,
            resources: ResourceRequirements {
                scratch: ResourcePresenceRequirement::Optional,
                ..no_auxiliary_resources()
            },
            oracle: OracleSpec::OperationDefined,
            provider: provider_requirement(self.capability_id(), ContractVersion::new(1, 0))?,
            profile_phase: ProfilePhase::Forward,
        };
        descriptor.validate()?;
        Ok(StandardOperationContract { descriptor })
    }
}

fn rational(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Rational,
            required: true,
            constraint: AttributeConstraint::RationalRange {
                minimum: CanonicalRational::new(1, 1_000_000_000_000)?,
                maximum: CanonicalRational::new(1, 1)?,
            },
        },
    ))
}
fn integers(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Integers,
            required: true,
            constraint: AttributeConstraint::IntegerListLength {
                minimum: 1,
                maximum: 32,
            },
        },
    ))
}

/// Checked semantic metadata only; providers still validate managed bindings.
#[derive(Clone, Debug)]
pub struct TensorPrimitiveBf16Geometry {
    pub kind: TensorPrimitiveBf16,
    pub rows: u64,
    pub columns: u64,
    pub inner: u64,
    pub head_dim: u64,
    pub scalar: f32,
    pub starts: Vec<u64>,
    pub lengths: Vec<u64>,
    pub input_bytes: Vec<u64>,
    pub output_bytes: u64,
}

impl TensorPrimitiveBf16Geometry {
    pub fn new(
        kind: TensorPrimitiveBf16,
        values: &[ResolvedValueBinding],
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        kind.contract()?.descriptor.validate_attributes(attrs)?;
        if values.len() != kind.input_count() as usize + 1 {
            return Err(invalid("incorrect arity"));
        }
        let binding = |role, ordinal| {
            values
                .iter()
                .find(|v| v.role() == role && v.ordinal() == ordinal)
                .map(|v| v.tensor())
                .ok_or_else(|| invalid("missing or duplicated port"))
        };
        let input: Vec<_> = (0..kind.input_count())
            .map(|i| binding(ResolvedValueRole::Input, i))
            .collect::<Result<_, _>>()?;
        let output = binding(ResolvedValueRole::Output, 0)?;
        for (index, tensor) in input.iter().enumerate() {
            check_tensor(
                tensor,
                if kind == TensorPrimitiveBf16::HalfRotary && index > 0 {
                    ElementType::F32
                } else {
                    ElementType::Bf16
                },
            )?;
        }
        check_tensor(output, ElementType::Bf16)?;
        let [rows, columns] = output.dimensions() else {
            return Err(invalid("output must be a matrix"));
        };
        let [source_rows, source_columns] = input[0].dimensions() else {
            return Err(invalid("first input must be a matrix"));
        };
        let mut g = Self {
            kind,
            rows: *rows,
            columns: *columns,
            inner: 0,
            head_dim: 0,
            scalar: 0.0,
            starts: Vec::new(),
            lengths: Vec::new(),
            input_bytes: input
                .iter()
                .map(|t| t.minimum_storage_bytes())
                .collect::<Result<_, _>>()?,
            output_bytes: output.minimum_storage_bytes()?,
        };
        use TensorPrimitiveBf16::*;
        match kind {
            AffineLinear | PatchProjection => {
                g.inner = *source_columns;
                let weight = input[1].dimensions();
                if source_rows != rows
                    || input[2].dimensions() != [*columns]
                    || weight.first() != Some(columns)
                {
                    return Err(invalid("projection row or bias dimensions differ"));
                }
                if kind == AffineLinear {
                    if weight != [*columns, *source_columns]
                        || uint(attrs, "in_features")? != *source_columns
                        || uint(attrs, "out_features")? != *columns
                    {
                        return Err(invalid("affine projection geometry differs"));
                    }
                } else if weight.len() != 5 || product(&weight[1..])? != *source_columns {
                    return Err(invalid("patch projection requires original rank-five O,C,T,H,W weight and matching flattened patches"));
                }
            }
            AffineLayerNorm => {
                if input[0] != output
                    || input[1].dimensions() != [*columns]
                    || input[2].dimensions() != [*columns]
                    || uint(attrs, "hidden_size")? != *columns
                    || *columns > (1 << 24)
                {
                    return Err(invalid("affine normalization geometry differs"));
                }
                g.scalar = ratio(attrs, "epsilon")?;
            }
            GeluErf => {
                if input[0] != output {
                    return Err(invalid("GELU shape differs"));
                }
            }
            HalfRotary => {
                g.head_dim = uint(attrs, "head_dim")?;
                if input[0] != output
                    || g.head_dim < 2
                    || g.head_dim > 256
                    || g.head_dim % 2 != 0
                    || columns % g.head_dim != 0
                    || input[1].dimensions() != [*rows, g.head_dim]
                    || input[2] != input[1]
                {
                    return Err(invalid("half rotary geometry differs"));
                }
            }
            RowScatterAdd => {
                if input[0] != output
                    || input[1].dimensions().len() != 2
                    || input[1].dimensions()[1] != *columns
                {
                    return Err(invalid("row scatter dimensions differ"));
                }
                g.starts = list(attrs, "destination_starts")?;
                g.lengths = list(attrs, "run_lengths")?;
                if g.starts.len() != g.lengths.len() {
                    return Err(invalid("scatter run count differs"));
                }
                let mut previous_end = 0;
                let mut updates = 0_u64;
                for (&start, &length) in g.starts.iter().zip(&g.lengths) {
                    let end = start
                        .checked_add(length)
                        .ok_or_else(|| invalid("scatter extent overflows"))?;
                    if length == 0 || start < previous_end || end > *rows {
                        return Err(invalid(
                            "scatter runs must be ordered, nonempty, disjoint and in bounds",
                        ));
                    }
                    previous_end = end;
                    updates = updates
                        .checked_add(length)
                        .ok_or_else(|| invalid("scatter updates overflow"))?;
                }
                if updates != input[1].dimensions()[0] {
                    return Err(invalid("scatter must consume all updates exactly once"));
                }
            }
            ReshapeCopy => {
                if uint(attrs, "rows")? != *rows
                    || uint(attrs, "columns")? != *columns
                    || product(input[0].dimensions())? != product(output.dimensions())?
                {
                    return Err(invalid("reshape must preserve element count"));
                }
            }
            SegmentedAttention => {
                g.head_dim = uint(attrs, "head_dim")?;
                g.scalar = ratio(attrs, "scale")?;
                if input.iter().any(|t| *t != output)
                    || g.head_dim < 2
                    || g.head_dim > 256
                    || g.head_dim % 2 != 0
                    || columns % g.head_dim != 0
                {
                    return Err(invalid(
                        "segmented attention requires equal matrices and even head_dim in 2..=256",
                    ));
                }
                g.lengths = list(attrs, "segment_lengths")?;
                let mut end = 0_u64;
                for &length in &g.lengths {
                    if length == 0 {
                        return Err(invalid("attention segments must be nonempty"));
                    }
                    g.starts.push(end);
                    end = end
                        .checked_add(length)
                        .ok_or_else(|| invalid("attention extent overflows"))?;
                }
                if end != *rows {
                    return Err(invalid("attention segments must partition every row"));
                }
            }
        }
        Ok(g)
    }
    pub fn elements(&self) -> u64 {
        self.output_bytes / 2
    }
}

fn check_tensor(t: &ResolvedTensorSpec, dtype: ElementType) -> Result<(), VNextError> {
    if t.element_type() != dtype
        || t.layout() != &ResolvedTensorLayout::Contiguous
        || t.dimensions().contains(&0)
        || t.minimum_storage_bytes()? > isize::MAX as u64
    {
        return Err(invalid(
            "requires positive contiguous tensor with bounded storage",
        ));
    }
    Ok(())
}
fn uint(attrs: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> Result<u64, VNextError> {
    match attrs.get(&AttributeId::new(name)?) {
        Some(SemanticValue::Unsigned(v)) => Ok(*v),
        _ => Err(invalid("missing unsigned attribute")),
    }
}
fn ratio(attrs: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> Result<f32, VNextError> {
    match attrs.get(&AttributeId::new(name)?) {
        Some(SemanticValue::Rational(v)) => {
            Ok((v.numerator() as f64 / v.denominator() as f64) as f32)
        }
        _ => Err(invalid("missing rational attribute")),
    }
}
fn list(attrs: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> Result<Vec<u64>, VNextError> {
    match attrs.get(&AttributeId::new(name)?) {
        Some(SemanticValue::Integers(v)) => v
            .iter()
            .map(|&n| u64::try_from(n).map_err(|_| invalid("negative row index or extent")))
            .collect(),
        _ => Err(invalid("missing integer list")),
    }
}
fn product(values: &[u64]) -> Result<u64, VNextError> {
    values.iter().try_fold(1_u64, |p, &n| {
        p.checked_mul(n)
            .ok_or_else(|| invalid("tensor extent overflows"))
    })
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 tensor primitive: {reason}"),
    }
}
