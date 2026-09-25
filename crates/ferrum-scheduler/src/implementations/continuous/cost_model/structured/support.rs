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
        let points: Vec<Vec<u64>> = points.map(<[u64]>::to_vec).collect();
        let Some(first) = points.first() else {
            return Err(StructuredUnknown::InsufficientSamples);
        };
        if points.len() > 4096 || first.is_empty() || first.len() > 4096 {
            return Err(StructuredUnknown::Capacity);
        }
        let mut minimum = first.clone();
        for point in &points {
            if point.len() != minimum.len() {
                return Err(StructuredUnknown::InvalidInput);
            }
            for (low, value) in minimum.iter_mut().zip(point) {
                *low = (*low).min(*value);
            }
        }
        Ok(Self { minimum, points })
    }
    pub(super) fn contains(&self, query: &[u64]) -> bool {
        query.len() == self.minimum.len()
            && query.iter().zip(&self.minimum).all(|(q, low)| q >= low)
            && self
                .points
                .iter()
                .any(|point| query.iter().zip(point).all(|(q, upper)| q <= upper))
    }
}
