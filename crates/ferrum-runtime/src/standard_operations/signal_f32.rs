//! F32 signal arithmetic on explicitly declared contiguous NCHW tensors.
//! FIR operates along W with H=1 and preserves independent batch/channel axes.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const FIR_RESAMPLE1D_F32_OPERATION_ID: &str = "operation.fir_resample1d.f32";
pub const FIR_RESAMPLE1D_F32_CAPABILITY_ID: &str = "capability.operation.fir_resample1d.f32";
pub const SNAKE_BETA_LOG_NCHW_F32_OPERATION_ID: &str = "operation.snake_beta_log.nchw.f32";
pub const SNAKE_BETA_LOG_NCHW_F32_CAPABILITY_ID: &str =
    "capability.operation.snake_beta_log.nchw.f32";
pub const SNAKE_ALPHA_NCHW_F32_OPERATION_ID: &str = "operation.snake_alpha.nchw.f32";
pub const SNAKE_ALPHA_NCHW_F32_CAPABILITY_ID: &str = "capability.operation.snake_alpha.nchw.f32";
pub const CLAMP_RANK4_F32_OPERATION_ID: &str = "operation.clamp_rank4.f32";
pub const CLAMP_RANK4_F32_CAPABILITY_ID: &str = "capability.operation.clamp_rank4.f32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalF32Op {
    FirResample,
    SnakeBetaLog,
    SnakeAlpha,
    Clamp,
}
impl SignalF32Op {
    pub const ALL: [Self; 4] = [
        Self::FirResample,
        Self::SnakeBetaLog,
        Self::SnakeAlpha,
        Self::Clamp,
    ];
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::FirResample => FIR_RESAMPLE1D_F32_OPERATION_ID,
            Self::SnakeBetaLog => SNAKE_BETA_LOG_NCHW_F32_OPERATION_ID,
            Self::SnakeAlpha => SNAKE_ALPHA_NCHW_F32_OPERATION_ID,
            Self::Clamp => CLAMP_RANK4_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::FirResample => FIR_RESAMPLE1D_F32_CAPABILITY_ID,
            Self::SnakeBetaLog => SNAKE_BETA_LOG_NCHW_F32_CAPABILITY_ID,
            Self::SnakeAlpha => SNAKE_ALPHA_NCHW_F32_CAPABILITY_ID,
            Self::Clamp => CLAMP_RANK4_F32_CAPABILITY_ID,
        }
    }
    pub const fn input_count(self) -> usize {
        match self {
            Self::FirResample | Self::SnakeAlpha => 2,
            Self::SnakeBetaLog => 3,
            Self::Clamp => 1,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let x = vec![
            symbol("batch"),
            symbol("channels"),
            symbol("height"),
            symbol("width"),
        ];
        let mut inputs = vec![contiguous_tensor(
            x.clone(),
            [ElementType::F32],
            TensorAccess::Read,
        )?];
        let mut attributes = BTreeMap::new();
        let y = match self {
            Self::FirResample => {
                inputs.push(contiguous_tensor(
                    vec![symbol("taps")],
                    [ElementType::F32],
                    TensorAccess::Read,
                )?);
                attributes.insert(
                    AttributeId::new("transpose")?,
                    AttributeSpec {
                        value_kind: AttributeValueKind::Bool,
                        required: true,
                        constraint: AttributeConstraint::None,
                    },
                );
                attributes.insert(AttributeId::new("stride")?, unsigned(1, i32::MAX as u64));
                for name in ["padding", "crop"] {
                    attributes.insert(
                        AttributeId::new(name)?,
                        AttributeSpec {
                            value_kind: AttributeValueKind::Integers,
                            required: true,
                            constraint: AttributeConstraint::IntegerListLength {
                                minimum: 2,
                                maximum: 2,
                            },
                        },
                    );
                }
                vec![
                    symbol("batch"),
                    symbol("channels"),
                    symbol("height"),
                    symbol("output_width"),
                ]
            }
            Self::SnakeBetaLog | Self::SnakeAlpha => {
                for _ in 1..self.input_count() {
                    inputs.push(contiguous_tensor(
                        vec![symbol("channels")],
                        [ElementType::F32],
                        TensorAccess::Read,
                    )?);
                }
                x
            }
            Self::Clamp => {
                for name in ["minimum_bits", "maximum_bits"] {
                    attributes.insert(AttributeId::new(name)?, unsigned(0, u32::MAX as u64));
                }
                x
            }
        };
        let descriptor = OperationDescriptor {
            id: OperationId::new(self.operation_id())?,
            version: ContractVersion::new(1, 0),
            inputs,
            outputs: vec![contiguous_tensor(
                y,
                [ElementType::F32],
                TensorAccess::Write,
            )?],
            attributes: AttributeSchema::new(attributes)?,
            resources: ResourceRequirements {
                minimum_value_alignment_bytes: 16,
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
/// Shared filter[K] on X[B,C,1,L]. Replicate-pad both ends, then ordinary
/// cross-correlation or transposed convolution with positive stride, then crop.
/// Each output accumulates valid taps in increasing tap order using F32 FMA,
/// starting from positive zero. Transpose inserts exact zero gaps without
/// multiplying the omitted positions. No filter replication allocation, scaling,
/// channel mixing, temporal history or implicit padding is part of this op.
pub fn fir_resample1d_f32_contract() -> Result<StandardOperationContract, VNextError> {
    SignalF32Op::FirResample.contract()
}
/// alpha=exp_profile(log_alpha[C]); beta=exp_profile(log_beta[C]);
/// z=RN32(alpha*x); s=sin_profile(z); q=RN32(s*s);
/// r=RN32(1/RN32(beta+F32(1e-9))); y=RN32(x+RN32(r*q)).
/// All boundaries are F32, separate from FIR filtering. Providers declare the
/// transcendental profile; this contract does not claim Torch compiled parity.
pub fn snake_beta_log_nchw_f32_contract() -> Result<StandardOperationContract, VNextError> {
    SignalF32Op::SnakeBetaLog.contract()
}
/// Direct F32 alpha[C], including zero and negative values: z=RN32(alpha*x),
/// s=sin_profile(z), q=RN32(s*s), r=RN32(1/RN32(alpha+F32(1e-9))),
/// y=RN32(x+RN32(r*q)). No logarithm, positivity transformation, extra beta,
/// or alias-free resampling. Providers declare their sine/division profiles.
pub fn snake_alpha_nchw_f32_contract() -> Result<StandardOperationContract, VNextError> {
    SignalF32Op::SnakeAlpha.contract()
}
/// Clamp by ordered comparisons; preserve in-range bits, including signed zero
/// and NaN payloads. Bounds must be finite and minimum <= maximum.
pub fn clamp_rank4_f32_contract() -> Result<StandardOperationContract, VNextError> {
    SignalF32Op::Clamp.contract()
}
fn unsigned(minimum: u64, maximum: u64) -> AttributeSpec {
    AttributeSpec {
        value_kind: AttributeValueKind::Unsigned,
        required: true,
        constraint: AttributeConstraint::UnsignedRange { minimum, maximum },
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalF32Geometry {
    pub kind: SignalF32Op,
    pub batch: u64,
    pub channels: u64,
    pub height: u64,
    pub input_width: u64,
    pub output_width: u64,
    pub taps: u64,
    pub stride: u64,
    pub transpose: bool,
    pub padding: [u64; 2],
    pub crop: [u64; 2],
    pub minimum: f32,
    pub maximum: f32,
    operations: u64,
}
impl SignalF32Geometry {
    pub fn fir_attributes(
        transpose: bool,
        stride: u64,
        padding: [u64; 2],
        crop: [u64; 2],
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let ints = |values: [u64; 2]| -> Result<SemanticValue, VNextError> {
            Ok(SemanticValue::Integers(
                values
                    .into_iter()
                    .map(|x| i64::try_from(x).map_err(|_| invalid("FIR index exceeds i64")))
                    .collect::<Result<_, _>>()?,
            ))
        };
        Ok(BTreeMap::from([
            (
                AttributeId::new("transpose")?,
                SemanticValue::Bool(transpose),
            ),
            (AttributeId::new("stride")?, SemanticValue::Unsigned(stride)),
            (AttributeId::new("padding")?, ints(padding)?),
            (AttributeId::new("crop")?, ints(crop)?),
        ]))
    }
    pub fn clamp_attributes(
        minimum: f32,
        maximum: f32,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(invalid("invalid clamp bounds"));
        }
        Ok(BTreeMap::from([
            (
                AttributeId::new("minimum_bits")?,
                SemanticValue::Unsigned(minimum.to_bits().into()),
            ),
            (
                AttributeId::new("maximum_bits")?,
                SemanticValue::Unsigned(maximum.to_bits().into()),
            ),
        ]))
    }
    pub fn new(
        kind: SignalF32Op,
        inputs: &[&ResolvedTensorSpec],
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        kind.contract()?.descriptor.validate_attributes(attrs)?;
        if inputs.len() != kind.input_count() {
            return Err(invalid("wrong input count"));
        }
        for t in inputs.iter().copied().chain(std::iter::once(y)) {
            if t.element_type() != ElementType::F32
                || t.layout() != &ResolvedTensorLayout::Contiguous
                || t.dimensions().contains(&0)
                || t.minimum_storage_bytes()? > isize::MAX as u64
            {
                return Err(invalid(
                    "requires nonempty addressable contiguous F32 tensors",
                ));
            }
        }
        let [batch, channels, height, width] = inputs[0].dimensions() else {
            return Err(invalid("input requires NCHW"));
        };
        let mut g = Self {
            kind,
            batch: *batch,
            channels: *channels,
            height: *height,
            input_width: *width,
            output_width: *width,
            taps: 0,
            stride: 1,
            transpose: false,
            padding: [0; 2],
            crop: [0; 2],
            minimum: 0.,
            maximum: 0.,
            operations: 0,
        };
        let get = |s: &str| attrs.get(&AttributeId::new(s).expect("attribute"));
        match kind {
            SignalF32Op::FirResample => {
                if *height != 1 {
                    return Err(invalid("FIR operates only on N,C,1,L"));
                }
                let [taps] = inputs[1].dimensions() else {
                    return Err(invalid("FIR requires one shared rank-one filter"));
                };
                g.taps = *taps;
                let Some(SemanticValue::Bool(transpose)) = get("transpose") else {
                    return Err(invalid("missing transpose"));
                };
                g.transpose = *transpose;
                let Some(SemanticValue::Unsigned(stride)) = get("stride") else {
                    return Err(invalid("missing stride"));
                };
                g.stride = *stride;
                let pair = |name: &str| -> Result<[u64; 2], VNextError> {
                    let Some(SemanticValue::Integers(v)) = get(name) else {
                        return Err(invalid("missing index pair"));
                    };
                    if v.len() != 2 {
                        return Err(invalid("index pair length"));
                    }
                    Ok([
                        u64::try_from(v[0]).map_err(|_| invalid("negative index"))?,
                        u64::try_from(v[1]).map_err(|_| invalid("negative index"))?,
                    ])
                };
                g.padding = pair("padding")?;
                g.crop = pair("crop")?;
                let padded = width
                    .checked_add(g.padding[0])
                    .and_then(|n| n.checked_add(g.padding[1]))
                    .filter(|&n| n <= i64::MAX as u64)
                    .ok_or_else(|| invalid("padded index overflow"))?;
                let raw = if g.transpose {
                    padded
                        .checked_sub(1)
                        .and_then(|n| n.checked_mul(g.stride))
                        .and_then(|n| n.checked_add(g.taps))
                } else {
                    padded.checked_sub(g.taps).map(|n| n / g.stride + 1)
                }
                .filter(|&n| n <= i64::MAX as u64)
                .ok_or_else(|| invalid("FIR output extent invalid"))?;
                g.output_width = raw
                    .checked_sub(g.crop[0])
                    .and_then(|n| n.checked_sub(g.crop[1]))
                    .filter(|&n| n > 0)
                    .ok_or_else(|| invalid("FIR crop removes output"))?;
                let contributions = if !g.transpose {
                    g.output_width.checked_mul(g.taps)
                } else {
                    // Constant-time count; metadata validation must not iterate
                    // across an attacker-controlled filter or signal length.
                    let before = |n: u64| -> u128 {
                        let full = if n >= g.taps {
                            ((n - g.taps) / g.stride + 1).min(padded)
                        } else {
                            0
                        };
                        let positive = n.div_ceil(g.stride).min(padded);
                        let partial = positive - full;
                        u128::from(full) * u128::from(g.taps) + u128::from(partial) * u128::from(n)
                            - u128::from(g.stride)
                                * u128::from(partial)
                                * (u128::from(full) + u128::from(positive.saturating_sub(1)))
                                / 2
                    };
                    u64::try_from(before(g.crop[0] + g.output_width) - before(g.crop[0])).ok()
                }
                .ok_or_else(|| invalid("FIR work overflow"))?;
                g.operations = contributions
                    .checked_mul(*batch)
                    .and_then(|n| n.checked_mul(*channels))
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(|| invalid("FIR work overflow"))?;
            }
            SignalF32Op::SnakeBetaLog | SignalF32Op::SnakeAlpha => {
                if inputs[1..]
                    .iter()
                    .any(|input| input.dimensions() != [*channels])
                {
                    return Err(invalid(
                        "Snake parameters must be exact separate channel vectors C",
                    ));
                }
                g.operations = inputs[0]
                    .minimum_storage_bytes()?
                    .checked_div(4)
                    .and_then(|n| {
                        n.checked_mul(if kind == SignalF32Op::SnakeAlpha {
                            7
                        } else {
                            10
                        })
                    })
                    .ok_or_else(|| invalid("Snake work overflow"))?;
            }
            SignalF32Op::Clamp => {
                let value = |name: &str| -> Result<f32, VNextError> {
                    let Some(SemanticValue::Unsigned(bits)) = get(name) else {
                        return Err(invalid("missing clamp bound"));
                    };
                    Ok(f32::from_bits(
                        u32::try_from(*bits).map_err(|_| invalid("clamp encoding exceeds F32"))?,
                    ))
                };
                g.minimum = value("minimum_bits")?;
                g.maximum = value("maximum_bits")?;
                if !g.minimum.is_finite() || !g.maximum.is_finite() || g.minimum > g.maximum {
                    return Err(invalid("invalid clamp bounds"));
                }
            }
        }
        if y.dimensions() != [g.batch, g.channels, g.height, g.output_width] {
            return Err(invalid("declared output shape differs"));
        }
        Ok(g)
    }
    pub fn input_elements(self, index: usize) -> Option<u64> {
        match index {
            0 => Some(self.batch * self.channels * self.height * self.input_width),
            1 if self.kind == SignalF32Op::FirResample => Some(self.taps),
            1 | 2 if self.kind == SignalF32Op::SnakeBetaLog => Some(self.channels),
            1 if self.kind == SignalF32Op::SnakeAlpha => Some(self.channels),
            _ => None,
        }
    }
    pub fn output_elements(self) -> u64 {
        self.batch * self.channels * self.height * self.output_width
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.operations
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 signal: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(d: &[u64]) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            d.to_vec(),
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    #[test]
    fn signal_geometry_checks_replicate_crop_and_physical_axes_without_length_dependent_planning() {
        for (length, taps, stride, padding, crop) in [
            (1, 12, 2, [5, 5], [15, 15]),
            (7, 12, 2, [5, 5], [15, 15]),
            (3, 3, 2, [2, 1], [2, 3]),
            (11, 7, 5, [1, 2], [4, 7]),
        ] {
            let padded = length + padding[0] + padding[1];
            let width = (padded - 1) * stride + taps - crop[0] - crop[1];
            let g = SignalF32Geometry::new(
                SignalF32Op::FirResample,
                &[&t(&[2, 3, 1, length]), &t(&[taps])],
                &t(&[2, 3, 1, width]),
                &SignalF32Geometry::fir_attributes(true, stride, padding, crop).unwrap(),
            )
            .unwrap();
            let contributions = (0..padded)
                .flat_map(|j| (0..taps).map(move |k| j * stride + k))
                .filter(|&i| i >= crop[0] && i < crop[0] + width)
                .count() as u64;
            assert_eq!(g.logical_arithmetic_operations(), contributions * 2 * 3 * 2);
            assert_eq!(g.output_width, width);
        }
        let x = t(&[2, 3, 1, 4]);
        let filter = t(&[12]);
        let attrs = SignalF32Geometry::fir_attributes(false, 2, [5, 6], [0, 0]).unwrap();
        assert!(SignalF32Geometry::new(
            SignalF32Op::FirResample,
            &[&x, &filter],
            &t(&[2, 3, 1, 2]),
            &attrs
        )
        .is_ok());
        assert!(SignalF32Geometry::new(
            SignalF32Op::FirResample,
            &[&t(&[2, 3, 2, 4]), &filter],
            &t(&[2, 3, 2, 2]),
            &attrs
        )
        .is_err());
        assert!(SignalF32Geometry::new(
            SignalF32Op::FirResample,
            &[&x, &t(&[1, 1, 12])],
            &t(&[2, 3, 1, 2]),
            &attrs
        )
        .is_err());
        assert!(SignalF32Geometry::clamp_attributes(f32::NAN, 1.).is_err());
        assert!(SignalF32Geometry::clamp_attributes(1., -1.).is_err());
        assert!(SignalF32Geometry::new(
            SignalF32Op::SnakeBetaLog,
            &[&x, &t(&[1, 3]), &t(&[3])],
            &x,
            &BTreeMap::new()
        )
        .is_err());
    }
}
