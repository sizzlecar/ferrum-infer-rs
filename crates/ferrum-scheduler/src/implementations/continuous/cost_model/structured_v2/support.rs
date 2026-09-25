use super::*;

pub(super) struct JointSupport {
    minimum: Vec<u64>,
    points: Vec<Vec<u64>>,
}
impl JointSupport {
    pub(super) fn bind_parameters(&self, digest: &mut sha2::Sha256) {
        use sha2::Digest;
        digest.update(b"joint-support-v1\0");
        digest.update((self.points.len() as u64).to_le_bytes());
        for values in std::iter::once(&self.minimum).chain(self.points.iter()) {
            digest.update((values.len() as u64).to_le_bytes());
            for value in values {
                digest.update(value.to_le_bytes());
            }
        }
    }
    pub(super) fn new<'a>(points: impl Iterator<Item = &'a [u64]>) -> Result<Self> {
        let mut stored = Vec::new();
        let mut minimum: Vec<u64> = Vec::new();
        for point in points {
            if stored.len() == 4096 || point.is_empty() || point.len() > 4096 {
                return Err(StructuredUnknown::Capacity);
            }
            if stored.is_empty() {
                minimum = point.to_vec();
            } else if point.len() != minimum.len() {
                return Err(StructuredUnknown::InvalidInput);
            }
            for (low, value) in minimum.iter_mut().zip(point) {
                *low = (*low).min(*value);
            }
            stored.push(point.to_vec());
        }
        if stored.is_empty() {
            return Err(StructuredUnknown::InsufficientSamples);
        }
        Ok(Self {
            minimum,
            points: stored,
        })
    }
    pub(super) fn contains(&self, query: &[u64]) -> bool {
        self.contains_envelope(query, query)
    }
    /// One original complete point must dominate the whole envelope; maxima
    /// from separate observations cannot be assembled into fictitious support.
    pub(super) fn contains_envelope(&self, lower: &[u64], upper: &[u64]) -> bool {
        lower.len() == self.minimum.len()
            && upper.len() == self.minimum.len()
            && lower.iter().zip(upper).all(|(a, b)| a <= b)
            && lower.iter().zip(&self.minimum).all(|(q, low)| q >= low)
            && self
                .points
                .iter()
                .any(|point| upper.iter().zip(point).all(|(q, high)| q <= high))
    }
}
