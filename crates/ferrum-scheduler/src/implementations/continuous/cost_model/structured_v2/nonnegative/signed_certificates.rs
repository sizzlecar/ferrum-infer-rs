//! Optional signed witnesses for the same nonnegative empirical coefficient
//! set as the original positive certificates. Floating arithmetic only suggests
//! weights; every returned bound is checked again against original integer rows.
use super::super::fit::FitGeometry;
use super::*;

#[cfg(test)]
mod diagnostic;
#[cfg(test)]
mod tests;

const SIGNED_REVISION: &[u8] = b"ferrum.signed-basis-envelope.v1\0";
const WEIGHT_DENOMINATOR: i128 = 1 << 20;

/// Frozen numerical advice, not an authority to predict. Anchors are indices
/// into the original Fit population. Replay reconstructs their rows and all
/// coefficient caps from that population; it does not rerun the floating solve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedBasisCertificateV1 {
    pub anchor_indices: Vec<usize>,
    /// Row-major rank x dimensions map, applied to Fit-normalized query axes.
    pub normalized_query_to_anchor_bits: Vec<u64>,
}

impl SignedBasisCertificateV1 {
    pub(super) fn retained_heap_bytes(&self) -> Option<usize> {
        self.anchor_indices
            .capacity()
            .checked_mul(std::mem::size_of::<usize>())?
            .checked_add(
                self.normalized_query_to_anchor_bits
                    .capacity()
                    .checked_mul(std::mem::size_of::<u64>())?,
            )
    }

    pub(super) fn bind_parameters(&self, digest: &mut sha2::Sha256) {
        use sha2::Digest;
        digest.update(SIGNED_REVISION);
        digest.update((self.anchor_indices.len() as u64).to_le_bytes());
        for &index in &self.anchor_indices {
            digest.update((index as u64).to_le_bytes());
        }
        digest.update((self.normalized_query_to_anchor_bits.len() as u64).to_le_bytes());
        for bits in &self.normalized_query_to_anchor_bits {
            digest.update(bits.to_le_bytes());
        }
    }
}

#[derive(Debug)]
struct Anchor {
    axes: Vec<u64>,
    wall_ns: u64,
}

#[derive(Debug, Clone, Copy)]
struct AxisCap {
    numerator: u128,
    // Zero denotes an axis that is zero in every original Fit row.
    denominator: u64,
}

#[derive(Debug)]
pub(super) struct SignedBasis {
    anchors: Vec<Anchor>,
    caps: Vec<AxisCap>,
}

fn vector<T>(capacity: usize) -> Result<Vec<T>> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(StructuredUnknown::Capacity)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| StructuredUnknown::Capacity)?;
    if values.capacity() != capacity {
        return Err(StructuredUnknown::Capacity);
    }
    Ok(values)
}

/// Bound the added cold input work along with the existing fixed 64 NNLS
/// sweeps. This does not create a second/reset allowance. One charged visit is
/// one examined scalar, as in the existing envelope settings. The bound covers
/// triangular products/solve, map and anchor copies, caps and validation scans.
pub(super) fn check_work(samples: usize, dims: usize, rank: usize, maximum: u64) -> Result<()> {
    let original = samples
        .checked_mul(dims)
        .and_then(|v| v.checked_mul(SWEEPS));
    let extra = rank
        .checked_mul(rank)
        .and_then(|v| v.checked_mul(dims))
        .and_then(|v| v.checked_mul(3))
        .and_then(|v| v.checked_add(samples.checked_mul(dims)?.checked_mul(2)?))
        .and_then(|v| v.checked_add(rank.checked_mul(dims)?.checked_mul(4)?))
        .and_then(|v| v.checked_add(rank.checked_mul(rank)?));
    let visits = original
        .and_then(|v| v.checked_add(extra?))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or(StructuredUnknown::Capacity)?;
    if visits > maximum {
        return Err(StructuredUnknown::Capacity);
    }
    Ok(())
}

/// Build a bounded input-only left inverse on the actual pivot anchors.
/// The triangular arithmetic is merely advice: even a poor finite map cannot
/// authorize an underestimate because query evaluation checks its integer dual.
pub(super) fn propose(geometry: &FitGeometry) -> Result<SignedBasisCertificateV1> {
    let rank = geometry.basis.len();
    let dims = geometry.scale.len();
    if rank == 0
        || dims == 0
        || geometry.pivot_indices.len() != rank
        || geometry.basis.iter().any(|r| r.len() != dims)
    {
        return Err(StructuredUnknown::InvalidInput);
    }
    let cells = rank.checked_mul(dims).ok_or(StructuredUnknown::Capacity)?;
    let mapping = match propose_mapping(geometry) {
        Ok(mapping) => mapping,
        Err(reason @ (StructuredUnknown::Numerical | StructuredUnknown::IllConditioned)) => {
            // Optional numerical advice cannot withdraw a valid positive
            // certificate. A zero hint is still checked by the same dual.
            tracing::warn!(
                target: "ferrum::slo_transaction",
                ?reason,
                rank,
                axes = dims,
                "SLO signed basis suggestion unavailable; retaining positive certificates"
            );
            let mut mapping = vector(cells)?;
            mapping.resize(cells, 0.0_f64.to_bits());
            mapping
        }
        Err(error) => return Err(error),
    };
    let mut anchor_indices = vector(rank)?;
    anchor_indices.extend_from_slice(&geometry.pivot_indices);
    Ok(SignedBasisCertificateV1 {
        anchor_indices,
        normalized_query_to_anchor_bits: mapping,
    })
}

fn propose_mapping(geometry: &FitGeometry) -> Result<Vec<u64>> {
    let rank = geometry.basis.len();
    let dims = geometry.scale.len();
    let cells = rank.checked_mul(dims).ok_or(StructuredUnknown::Capacity)?;
    let square = rank.checked_mul(rank).ok_or(StructuredUnknown::Capacity)?;
    let mut triangular = vector::<f64>(square)?;
    triangular.resize(square, 0.0);
    for (i, &index) in geometry.pivot_indices.iter().enumerate() {
        let row = geometry
            .rows
            .get(index)
            .ok_or(StructuredUnknown::InvalidInput)?;
        if row.len() != dims {
            return Err(StructuredUnknown::InvalidInput);
        }
        for j in 0..=i {
            let value = row
                .iter()
                .zip(&geometry.basis[j])
                .map(|(x, y)| x * y)
                .sum::<f64>();
            if !value.is_finite() {
                return Err(StructuredUnknown::Numerical);
            }
            triangular[i * rank + j] = value;
        }
        if triangular[i * rank + i] <= 0.0 {
            return Err(StructuredUnknown::IllConditioned);
        }
    }
    let mut mapping = vector::<u64>(cells)?;
    mapping.resize(cells, 0.0_f64.to_bits());
    for axis in 0..dims {
        for i in (0..rank).rev() {
            let rest = (i + 1..rank)
                .map(|j| triangular[j * rank + i] * f64::from_bits(mapping[j * dims + axis]))
                .sum::<f64>();
            let value = (geometry.basis[i][axis] - rest) / triangular[i * rank + i];
            if !value.is_finite() {
                return Err(StructuredUnknown::Numerical);
            }
            mapping[i * dims + axis] = value.to_bits();
        }
    }
    Ok(mapping)
}

impl SignedBasis {
    pub(super) fn replay(
        samples: &[FitSample<'_>],
        maxima: &[u64],
        rank: usize,
        epsilon_ns: u64,
        frozen: &SignedBasisCertificateV1,
    ) -> Result<Self> {
        let dims = maxima.len();
        if rank == 0
            || rank > samples.len()
            || rank > dims
            || frozen.anchor_indices.len() != rank
            || frozen.normalized_query_to_anchor_bits.len()
                != rank.checked_mul(dims).ok_or(StructuredUnknown::Capacity)?
            || frozen
                .normalized_query_to_anchor_bits
                .iter()
                .any(|&v| !f64::from_bits(v).is_finite())
            || frozen.anchor_indices.iter().enumerate().any(|(i, &index)| {
                index >= samples.len() || frozen.anchor_indices[..i].contains(&index)
            })
        {
            return Err(StructuredUnknown::WrongSource);
        }
        let mut caps = vector::<AxisCap>(dims)?;
        caps.resize(
            dims,
            AxisCap {
                numerator: 0,
                denominator: 0,
            },
        );
        for sample in samples {
            if sample.axes.len() != dims {
                return Err(StructuredUnknown::WrongSource);
            }
            let numerator = u128::from(sample.wall_ns)
                .checked_add(u128::from(epsilon_ns))
                .ok_or(StructuredUnknown::Numerical)?;
            for (cap, &value) in caps.iter_mut().zip(sample.axes) {
                if value == 0 {
                    continue;
                }
                // Keep exact ratios. Rounding a coefficient cap down is unsafe.
                if cap.denominator == 0
                    || numerator
                        .checked_mul(u128::from(cap.denominator))
                        .ok_or(StructuredUnknown::Numerical)?
                        < cap
                            .numerator
                            .checked_mul(u128::from(value))
                            .ok_or(StructuredUnknown::Numerical)?
                {
                    *cap = AxisCap {
                        numerator,
                        denominator: value,
                    };
                }
            }
        }
        let mut anchors = vector::<Anchor>(rank)?;
        for &index in &frozen.anchor_indices {
            let sample = &samples[index];
            let mut axes = vector(dims)?;
            axes.extend_from_slice(sample.axes);
            anchors.push(Anchor {
                axes,
                wall_ns: sample.wall_ns,
            });
        }
        Ok(Self { anchors, caps })
    }

    pub(super) fn retained_heap_bytes(&self) -> Option<usize> {
        let mut bytes = self
            .anchors
            .capacity()
            .checked_mul(std::mem::size_of::<Anchor>())?
            .checked_add(
                self.caps
                    .capacity()
                    .checked_mul(std::mem::size_of::<AxisCap>())?,
            )?;
        for anchor in &self.anchors {
            bytes = bytes.checked_add(
                anchor
                    .axes
                    .capacity()
                    .checked_mul(std::mem::size_of::<u64>())?,
            )?;
        }
        Some(bytes)
    }

    pub(super) fn upper_bound(
        &self,
        input: &[u64],
        maxima: &[u64],
        epsilon_ns: u64,
        frozen: &SignedBasisCertificateV1,
    ) -> Option<u64> {
        let dims = input.len();
        if dims == 0
            || maxima.len() != dims
            || self.caps.len() != dims
            || frozen.normalized_query_to_anchor_bits.len()
                != self.anchors.len().checked_mul(dims)?
        {
            return None;
        }
        // O(rank) scratch, never an original-sample scan or a query matrix.
        let mut weights = vector::<i64>(self.anchors.len()).ok()?;
        for mapping in frozen.normalized_query_to_anchor_bits.chunks_exact(dims) {
            let mut weight = 0.0;
            for ((&bits, &x), &scale) in mapping.iter().zip(input).zip(maxima) {
                weight += f64::from_bits(bits) * (x as f64 / scale.max(1) as f64);
            }
            let fixed = (weight * WEIGHT_DENOMINATOR as f64).round();
            // Explicit half-open conversion range; Rust's saturating float cast
            // must never silently supply a weight. Any rejected hint is optional.
            let endpoint = (1_u64 << 63) as f64;
            if !fixed.is_finite() || fixed < -endpoint || fixed >= endpoint {
                return None;
            }
            weights.push(fixed as i64);
        }
        self.bound_for_weights(input, epsilon_ns, &weights)
    }

    /// For every beta >= 0 satisfying all original |X beta - y| <= epsilon:
    /// a=Sum(w X), a.beta <= b, and D*q.beta <= b + Sum((D*q-a)+ * cap).
    /// All operations below are exact integers, with outward rational ceilings.
    /// Overflow makes only this witness unavailable; no saturation is allowed.
    fn bound_for_weights(&self, input: &[u64], epsilon_ns: u64, weights: &[i64]) -> Option<u64> {
        if weights.len() != self.anchors.len() || input.len() != self.caps.len() {
            return None;
        }
        let mut bound = 0_i128;
        for (anchor, &weight) in self.anchors.iter().zip(weights) {
            let wall = i128::from(anchor.wall_ns);
            let epsilon = i128::from(epsilon_ns);
            let endpoint = if weight >= 0 {
                wall.checked_add(epsilon)?
            } else {
                wall.checked_sub(epsilon)?
            };
            bound = bound.checked_add(i128::from(weight).checked_mul(endpoint)?)?;
        }
        for (axis, (&query, cap)) in input.iter().zip(&self.caps).enumerate() {
            let mut actual = 0_i128;
            for (anchor, &weight) in self.anchors.iter().zip(weights) {
                actual = actual.checked_add(
                    i128::from(weight).checked_mul(i128::from(*anchor.axes.get(axis)?))?,
                )?;
            }
            let remainder = i128::from(query)
                .checked_mul(WEIGHT_DENOMINATOR)?
                .checked_sub(actual)?;
            if remainder > 0 {
                if cap.denominator == 0 {
                    return None;
                }
                let product = u128::try_from(remainder).ok()?.checked_mul(cap.numerator)?;
                let correction =
                    certificates::ceil_div(product, u128::from(cap.denominator)).ok()?;
                bound = bound.checked_add(i128::try_from(correction).ok()?)?;
            }
        }
        // A negative intermediate b is legitimate. Convert only the completed
        // dual bound; it cannot be negative for a nonempty checked feasible set.
        let numerator = u128::try_from(bound).ok()?;
        u64::try_from(certificates::ceil_div(numerator, WEIGHT_DENOMINATOR as u128).ok()?).ok()
    }
}
