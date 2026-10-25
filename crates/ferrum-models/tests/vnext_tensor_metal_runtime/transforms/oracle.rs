//! FP64 semantic reference with propagated FP32/BF16 rounding intervals.
//! Bounds depend on the input and tree depth, not an empirical tolerance.
use super::*;

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn point(value: f32) -> Self {
        Self {
            lo: value as f64,
            hi: value as f64,
        }
    }
    fn outward(lo: f64, hi: f64) -> Self {
        Self {
            lo: lo.next_down(),
            hi: hi.next_up(),
        }
    }
    fn add(self, rhs: Self) -> Self {
        Self::outward(self.lo + rhs.lo, self.hi + rhs.hi)
    }
    fn sub(self, rhs: Self) -> Self {
        Self::outward(self.lo - rhs.hi, self.hi - rhs.lo)
    }
    fn mul(self, rhs: Self) -> Self {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        Self::outward(
            products.into_iter().fold(f64::INFINITY, f64::min),
            products.into_iter().fold(f64::NEG_INFINITY, f64::max),
        )
    }
    fn square(self) -> Self {
        let lo = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            (self.lo * self.lo).min(self.hi * self.hi)
        };
        Self {
            lo: lo.next_down().max(0.0),
            hi: (self.lo * self.lo).max(self.hi * self.hi).next_up(),
        }
    }
    fn fp32(self) -> Self {
        // One outward FP32 neighbour also covers independent reference rounding.
        Self {
            lo: (self.lo as f32).next_down() as f64,
            hi: (self.hi as f32).next_up() as f64,
        }
    }
    fn bf16(self) -> Self {
        Self {
            lo: rounds(self.lo as f32) as f64,
            hi: rounds(self.hi as f32) as f64,
        }
    }
    fn divide(self, value: f64) -> Self {
        Self::outward(self.lo / value, self.hi / value).fp32()
    }
    fn sqrt(self) -> Self {
        Self::outward(self.lo.max(0.0).sqrt(), self.hi.sqrt()).fp32()
    }
    fn inverse(self) -> Self {
        assert!(self.lo > 0.0);
        Self::outward(1.0 / self.hi, 1.0 / self.lo).fp32()
    }
}
fn tree_sum(values: &[Interval]) -> Interval {
    let height = values.len().next_power_of_two().ilog2() as f64;
    let factor = height * 2_f64.powi(-24);
    let gamma = factor / (1.0 - factor);
    let absolute = values
        .iter()
        .map(|v| v.lo.abs().max(v.hi.abs()))
        .sum::<f64>();
    // FP64 reduction uncertainty of this independent reference.
    let reference =
        (values.len() as f64 * f64::EPSILON) / (1.0 - values.len() as f64 * f64::EPSILON);
    let error = (gamma + reference) * absolute;
    Interval::outward(
        values.iter().map(|v| v.lo).sum::<f64>() - error,
        values.iter().map(|v| v.hi).sum::<f64>() + error,
    )
}

pub(crate) fn norm(
    kind: MetalTensorOperation,
    input: &[f32],
    gamma: &[f32],
    width: usize,
    epsilon: f32,
    actual: &[f32],
) {
    use MetalTensorOperation::*;
    assert_eq!(input.len(), actual.len());
    for (group, (x, y)) in input
        .chunks_exact(width)
        .zip(actual.chunks_exact(width))
        .enumerate()
    {
        let has_nan = x.iter().any(|v| v.is_nan());
        let has_inf = x.iter().any(|v| v.is_infinite());
        if has_nan || has_inf {
            for (&x, &y) in x.iter().zip(y) {
                if has_nan || kind == LayerNorm || x.is_infinite() {
                    assert!(y.is_nan());
                } else {
                    assert_eq!(y, 0.0);
                }
            }
            continue;
        }
        if kind == LayerNorm && x.iter().all(|v| *v == x[0]) && matches!(x[0], 0.0 | 2.0) {
            // These exactly representable constants have an exact zero variance.
            assert!(y.iter().all(|v| *v == 0.0));
            continue;
        }
        let source: Vec<_> = x.iter().copied().map(Interval::point).collect();
        let mean = if kind == LayerNorm {
            tree_sum(&source).divide(width as f64)
        } else {
            Interval::point(0.0)
        };
        let centered: Vec<_> = source.iter().map(|v| v.sub(mean).fp32()).collect();
        let squares: Vec<_> = centered.iter().map(|v| v.square().fp32()).collect();
        let variance = tree_sum(&squares).divide(width as f64);
        let inverse = variance
            .add(Interval::point(epsilon))
            .fp32()
            .sqrt()
            .inverse();
        for (column, (&x, &y)) in centered.iter().zip(y).enumerate() {
            let mut result = x.mul(inverse).fp32();
            if kind == HeadRmsNorm {
                result = result.bf16().mul(Interval::point(gamma[column])).fp32();
            }
            if kind == ZeroCenteredRmsNorm {
                result = result
                    .mul(
                        Interval::point(gamma[column])
                            .add(Interval::point(1.0))
                            .fp32(),
                    )
                    .fp32();
            }
            let expected = result.bf16();
            assert!(
                y.is_finite() && y as f64 >= expected.lo && y as f64 <= expected.hi,
                "{kind:?} group={group} column={column}: {y} outside [{}, {}]",
                expected.lo,
                expected.hi
            );
        }
    }
}

pub(crate) fn rope(
    kind: MetalTensorOperation,
    input: &[f32],
    cos: &[f32],
    sin: &[f32],
    columns: usize,
    head: usize,
    sequence: usize,
    actual: &[f32],
) {
    assert_eq!(actual.len(), input.len());
    for row in 0..input.len() / columns {
        for h in 0..columns / head {
            for channel in 0..head {
                let index = row * columns + h * head + channel;
                let expected = if kind == MetalTensorOperation::AdjacentRope {
                    let pair = channel / 2;
                    let offset = row * columns + h * head + pair * 2;
                    let frequency = (row % sequence) * head + 2 * pair;
                    let (a, b, c, s) = (
                        input[offset],
                        input[offset + 1],
                        cos[frequency],
                        cos[frequency + 1],
                    );
                    if channel % 2 == 0 {
                        libm::fmaf(a, c, -(b * s))
                    } else {
                        libm::fmaf(b, c, a * s)
                    }
                } else {
                    let rotated = if channel < head / 2 {
                        -input[index + head / 2]
                    } else {
                        input[index - head / 2]
                    };
                    rounds(input[index] * cos[row * head + channel])
                        + rounds(rotated * sin[row * head + channel])
                };
                let wanted = bf16::from_f32(expected);
                let got = bf16::from_f32(actual[index]);
                if wanted.is_nan() {
                    assert!(got.is_nan());
                } else {
                    assert_eq!(
                        got.to_bits(),
                        wanted.to_bits(),
                        "{kind:?} row={row} head={h} channel={channel}"
                    );
                }
            }
        }
    }
}
