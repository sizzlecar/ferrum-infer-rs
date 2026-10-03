//! CPU-only facts for one actual wave. Only the evidence consumer resolves them.
use super::*;

/// Conservative retained and expansion budgets, including shared payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingWaveBounds {
    pub retained_bytes: usize,
    pub maximum_resolved_bytes: usize,
    pub retained_rows: usize,
}

/// Two projections of the same original submission. Physical evidence cannot
/// turn a rejected numerical graph route into an eligible cost shape.
pub struct ActualWaveProjection {
    pub shape: Result<ActualWaveShape, ActualWaveEvidenceUnknown>,
    pub physical_evidence: Option<ActualWavePhysicalEvidenceV1>,
}

/// Implementations contain immutable CPU metadata and original-wave values.
/// They must not own execution resources or consult a runtime/sequence later.
/// There is deliberately no closure blanket implementation or lazy accessor.
pub trait PendingActualWaveProjection: Send + Sync {
    fn rows(&self) -> &[ActualWaveRow];
    fn bounds(&self) -> PendingWaveBounds;
    fn project(&self) -> Result<ActualWaveShape, ActualWaveEvidenceUnknown>;
    fn project_with_physical_evidence(&self) -> ActualWaveProjection {
        ActualWaveProjection {
            shape: self.project(),
            physical_evidence: None,
        }
    }
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
        self.resolve_with_physical_evidence().shape
    }

    pub(crate) fn resolve_with_physical_evidence(self) -> ActualWaveProjection {
        let mut result = self.projection.project_with_physical_evidence();
        // An unusable non-numerical sidecar cannot change the original graph
        // rejection or rescue another projection error.
        if result.physical_evidence.is_some_and(|physical| {
            physical.validate_rows(self.projection.rows()).is_err()
                || result
                    .shape
                    .as_ref()
                    .is_ok_and(|shape| physical != ActualWavePhysicalEvidenceV1::from_shape(shape))
        }) {
            result.physical_evidence = None;
        }
        let validated = (|| {
            let physical_bytes = if result.physical_evidence.is_some() {
                std::mem::size_of::<ActualWavePhysicalEvidenceV1>()
            } else {
                0
            };
            let shape_bytes = match &result.shape {
                Ok(shape) => {
                    resolved_shape_bytes(shape).ok_or(ActualWaveEvidenceUnknown::Capacity)?
                }
                Err(ActualWaveEvidenceUnknown::GraphPath) => 0,
                Err(reason) => return Err(*reason),
            };
            if shape_bytes
                .checked_add(physical_bytes)
                .is_none_or(|bytes| bytes > self.bounds.maximum_resolved_bytes)
            {
                return Err(ActualWaveEvidenceUnknown::Capacity);
            }
            // Preserve the original numeric error precedence: expansion
            // capacity, then participant binding, then shape validity.
            if let Ok(shape) = &result.shape {
                if shape.rows != self.projection.rows() {
                    return Err(ActualWaveEvidenceUnknown::ParticipantCorrelation);
                }
                shape
                    .validate(1024)
                    .map_err(|_| ActualWaveEvidenceUnknown::ShapeOverflow)?;
            }
            Ok(())
        })();
        if let Err(reason) = validated {
            result.shape = Err(reason);
            result.physical_evidence = None;
        }
        result
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
