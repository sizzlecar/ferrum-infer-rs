//! CPU-only facts for one actual wave. Only the evidence consumer resolves them.
use super::*;

/// Conservative retained and expansion budgets, including shared payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingWaveBounds {
    pub retained_bytes: usize,
    pub maximum_resolved_bytes: usize,
    pub retained_rows: usize,
}

/// Implementations contain immutable CPU metadata and original-wave values.
/// They must not own execution resources or consult a runtime/sequence later.
/// There is deliberately no closure blanket implementation or lazy accessor.
pub trait PendingActualWaveProjection: Send + Sync {
    fn rows(&self) -> &[ActualWaveRow];
    fn bounds(&self) -> PendingWaveBounds;
    fn project(&self) -> Result<ActualWaveShape, ActualWaveEvidenceUnknown>;
}

#[derive(Clone)]
pub struct PendingActualWave {
    projection: Arc<dyn PendingActualWaveProjection>,
    bounds: PendingWaveBounds,
}
impl std::fmt::Debug for PendingActualWave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Formatting must never call the projector or enumerate static recipes.
        f.debug_struct("PendingActualWave")
            .field("bounds", &self.bounds)
            .finish()
    }
}
impl PendingActualWave {
    pub fn new(
        projection: Arc<dyn PendingActualWaveProjection>,
    ) -> Result<Self, ActualWaveEvidenceUnknown> {
        let bounds = projection.bounds();
        if projection.rows().is_empty()
            || projection.rows().len() > 1024
            || bounds.retained_rows < projection.rows().len()
            || bounds.retained_bytes == 0
            || bounds.maximum_resolved_bytes == 0
        {
            return Err(ActualWaveEvidenceUnknown::Capacity);
        }
        Ok(Self { projection, bounds })
    }
    pub fn rows(&self) -> &[ActualWaveRow] {
        self.projection.rows()
    }
    pub fn bounds(&self) -> PendingWaveBounds {
        self.bounds
    }
    /// Explicit consumer operation. None of the metadata accessors resolves.
    pub fn resolve(self) -> Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
        let shape = self.projection.project()?;
        if resolved_shape_bytes(&shape)
            .is_none_or(|bytes| bytes > self.bounds.maximum_resolved_bytes)
        {
            return Err(ActualWaveEvidenceUnknown::Capacity);
        }
        if shape.rows != self.projection.rows() {
            return Err(ActualWaveEvidenceUnknown::ParticipantCorrelation);
        }
        shape
            .validate(1024)
            .map_err(|_| ActualWaveEvidenceUnknown::ShapeOverflow)?;
        Ok(shape)
    }
}

fn resolved_shape_bytes(shape: &ActualWaveShape) -> Option<usize> {
    std::mem::size_of::<ActualWaveShape>()
        .checked_add(
            shape
                .rows
                .capacity()
                .checked_mul(std::mem::size_of::<ActualWaveRow>())?,
        )?
        .checked_add(match &shape.numeric_features {
            Some(rows) => rows
                .rows
                .capacity()
                .checked_mul(std::mem::size_of::<CostRowNumericFeatures>())?,
            None => 0,
        })?
        .checked_add(match &shape.row_multiset_features {
            Some(rows) => rows
                .rows
                .capacity()
                .checked_mul(std::mem::size_of::<HostRowStaticCostFeaturesV2>())?,
            None => 0,
        })?
        .checked_add(
            match shape
                .statistical_evidence
                .as_ref()
                .and_then(|s| s.structured_capture())
                .and_then(Result::ok)
            {
                Some(recipe) => recipe.retained_bytes().ok()?,
                None => 0,
            },
        )
}
