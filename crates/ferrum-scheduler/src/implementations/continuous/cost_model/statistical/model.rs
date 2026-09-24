//! Explicit offline fit -> independent residual -> heldout pipeline. No default
//! predictor or online trainer calls this module. All timings are whole-wave.
use super::super::{CostBoundary, CostModelSettings, ExecutionFingerprint, WaveObservationOutcome};
use super::*;
use std::collections::{BTreeMap, BTreeSet};
mod settings;
pub use settings::WholeWaveSettingsV1;
mod fit;
mod support;
use fit::Affine;
use support::{coordinates, Support};
#[cfg(test)]
pub(crate) mod tests;

pub const MODEL_REVISION: &str = "whole_wave_piecewise_affine_v1";
pub const INDEPENDENT_ATTENTION_MODEL_REVISION: &str =
    "whole_wave_piecewise_affine_independent_attention_v2";
pub const WORK_SUPPORT_MODEL_REVISION: &str =
    "whole_wave_piecewise_affine_independent_attention_work_support_v1";
/// Model semantics are separate from the producer's statistical family schema.
/// WorkSupportV1 still consumes the exact V2 family and original 26-field data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WholeWaveModelRevision {
    OrderedV1,
    IndependentAttentionV2,
    IndependentAttentionWorkSupportV1,
}
impl WholeWaveModelRevision {
    pub fn family(self) -> SelectedStatisticalFamily {
        match self {
            Self::OrderedV1 => SelectedStatisticalFamily::OrderedV1,
            Self::IndependentAttentionV2 | Self::IndependentAttentionWorkSupportV1 => {
                SelectedStatisticalFamily::IndependentAttentionV2
            }
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OrderedV1 => MODEL_REVISION,
            Self::IndependentAttentionV2 => INDEPENDENT_ATTENTION_MODEL_REVISION,
            Self::IndependentAttentionWorkSupportV1 => WORK_SUPPORT_MODEL_REVISION,
        }
    }
    pub fn fit(
        self,
        fingerprint: ExecutionFingerprint,
        settings: WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        samples: &[WholeWaveObservationV1],
        now_ns: u64,
    ) -> Result<FittedWholeWaveModelV1, ModelUnknown> {
        FittedWholeWaveModelV1::fit_for_revision(
            fingerprint,
            settings,
            partition,
            samples,
            now_ns,
            self,
        )
    }
}
impl SelectedStatisticalFamily {
    pub fn model_revision(self) -> &'static str {
        match self {
            Self::OrderedV1 => MODEL_REVISION,
            Self::IndependentAttentionV2 => INDEPENDENT_ATTENTION_MODEL_REVISION,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelUnknown {
    Evidence(Unknown),
    InvalidSettings,
    InvalidSample,
    WrongFingerprint,
    WrongSource,
    PhaseLeakage,
    DuplicateRecord,
    Capacity,
    Clock,
    Stale,
    InsufficientFit,
    InsufficientResidual,
    FamilyMissing,
    JointSupport,
    Numerical,
    /// Engine-only explicit feedback gate; never inferred from offline samples.
    RuntimeValidity,
}
impl From<Unknown> for ModelUnknown {
    fn from(value: Unknown) -> Self {
        Self::Evidence(value)
    }
}

/// A digest identifies one real capture stream, not a phase or a prediction.
/// Profile6 separately pins the immutable source file SHA after publication.
/// Accepted ordinals are FIFO positions; call IDs remain a separate identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationPartitionV1 {
    pub source_sha256: [u8; 32],
    pub protocol_sha256: [u8; 32],
    pub fit_through_ordinal: u64,
    pub residual_through_ordinal: u64,
}
impl CalibrationPartitionV1 {
    fn validate(self) -> Result<(), ModelUnknown> {
        if self.source_sha256 == [0; 32]
            || self.protocol_sha256 == [0; 32]
            || self.fit_through_ordinal == 0
            || self.residual_through_ordinal <= self.fit_through_ordinal
        {
            return Err(ModelUnknown::PhaseLeakage);
        }
        Ok(())
    }
}

/// Public input is checked rather than trusted. Device-only/legacy commit,
/// failed calls and unbound/missing selected evidence are rejected on ingest.
#[derive(Debug, Clone)]
pub struct WholeWaveObservationV1 {
    pub source_sha256: [u8; 32],
    pub accepted_ordinal: u64,
    pub call_id: u64,
    pub fingerprint: ExecutionFingerprint,
    pub exact: CanonicalWaveCostShape,
    pub selected: StatisticalWaveEvidenceV1,
    pub boundary: CostBoundary,
    pub outcome: WaveObservationOutcome,
    pub observed_at_ns: u64,
    pub wall_ns: u64,
}
impl WholeWaveObservationV1 {
    fn input(
        &self,
        fingerprint: &ExecutionFingerprint,
        settings: &WholeWaveSettingsV1,
        source: [u8; 32],
        now: u64,
    ) -> Result<StatisticalModelInputV1, ModelUnknown> {
        if &self.fingerprint != fingerprint {
            return Err(ModelUnknown::WrongFingerprint);
        }
        if self.source_sha256 != source {
            return Err(ModelUnknown::WrongSource);
        }
        if self.accepted_ordinal == 0
            || self.call_id == 0
            || self.wall_ns == 0
            || self.wall_ns > settings.max_wave_ns.get()
            || self.boundary != CostBoundary::PreparationToHostSettledV1
            || self.outcome != WaveObservationOutcome::Completed
        {
            return Err(ModelUnknown::InvalidSample);
        }
        let age = now
            .checked_sub(self.observed_at_ns)
            .ok_or(ModelUnknown::Clock)?;
        if age > settings.max_sample_age_ns.get() {
            return Err(ModelUnknown::Stale);
        }
        let input = StatisticalModelInputV1::from_future(&self.exact, &self.selected)?;
        validate_input(&input, settings)?;
        Ok(input)
    }
}
fn settings_valid(settings: &WholeWaveSettingsV1) -> Result<(), ModelUnknown> {
    settings.validate()
}
fn validate_input(
    input: &StatisticalModelInputV1,
    settings: &WholeWaveSettingsV1,
) -> Result<(), ModelUnknown> {
    let h = input.host_and_sequence();
    if h.rows > settings.shape_limits.max_rows.get() as u64
        || h.kv_tokens_max > u64::from(settings.shape_limits.max_context_tokens.get())
        || h.prompt_tokens_max > u64::from(settings.shape_limits.max_context_tokens.get())
        || h.prefill_tokens > settings.shape_limits.max_prefill_tokens_per_wave.get()
        || h.recurrent_bytes > settings.shape_limits.max_state_bytes.get()
    {
        return Err(ModelUnknown::Capacity);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct FittedSegment {
    affine: Affine,
    support: Support,
    expires_at_ns: u64,
    samples: usize,
}
#[derive(Debug, Clone)]
pub struct FittedWholeWaveModelV1 {
    family: SelectedStatisticalFamily,
    revision: WholeWaveModelRevision,
    fingerprint: ExecutionFingerprint,
    settings: WholeWaveSettingsV1,
    partition: CalibrationPartitionV1,
    fit_at_ns: u64,
    last_fit_observed_at_ns: u64,
    retained_samples: usize,
    retained_rows: usize,
    fit_calls: BTreeSet<u64>,
    unavailable: BTreeMap<[u8; 32], ModelUnknown>,
    segments: BTreeMap<[u8; 32], FittedSegment>,
}
#[derive(Debug, Clone)]
struct CalibratedSegment {
    fit: FittedSegment,
    support: Support,
    q99_ns: u64,
    expires_at_ns: u64,
    residual_samples: usize,
}
#[derive(Debug, Clone)]
pub struct WholeWaveModelV1 {
    family: SelectedStatisticalFamily,
    revision: WholeWaveModelRevision,
    fingerprint: ExecutionFingerprint,
    settings: WholeWaveSettingsV1,
    partition: CalibrationPartitionV1,
    frozen_at_ns: u64,
    calibration_calls: BTreeSet<u64>,
    unavailable: BTreeMap<[u8; 32], ModelUnknown>,
    segments: BTreeMap<[u8; 32], CalibratedSegment>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WholeWavePredictionV1 {
    pub fitted_ns: u64,
    pub residual_ns: u64,
    pub static_margin_ns: u64,
    pub planning_ns: u64,
    pub valid_until_ns: u64,
    pub fit_samples: usize,
    pub residual_samples: usize,
}
/// Identity selected by this exact read-only lookup, never reconstructed from training rows.
/// Serialized only in diagnostic results; profile/source V1--V7 wires are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct SelectedQueryIdentityV1 {
    pub schema_version: u32,
    pub model_revision: &'static str,
    pub family_schema_version: u32,
    pub family_signature: [u8; 32],
}
impl SelectedQueryIdentityV1 {
    fn new(revision: WholeWaveModelRevision, signature: [u8; 32]) -> Self {
        let family = revision.family();
        Self {
            schema_version: 1,
            model_revision: revision.as_str(),
            family_schema_version: match family {
                SelectedStatisticalFamily::OrderedV1 => 1,
                SelectedStatisticalFamily::IndependentAttentionV2 => 2,
            },
            family_signature: signature,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentifiedPredictionV1 {
    pub query_identity: Option<SelectedQueryIdentityV1>,
    pub prediction: Result<WholeWavePredictionV1, ModelUnknown>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldoutEvaluationV1 {
    pub query_identity: Option<SelectedQueryIdentityV1>,
    pub prediction: Result<WholeWavePredictionV1, ModelUnknown>,
    pub actual_ns: u64,
    pub underestimate_ns: Option<u64>,
}

impl FittedWholeWaveModelV1 {
    /// Seal the real residual FIFO boundary after fitting. This only tightens a
    /// predeclared upper bound; it cannot change fit rows or fitted parameters.
    pub fn seal_residual_cut(mut self, accepted_ordinal: u64) -> Result<Self, ModelUnknown> {
        if accepted_ordinal <= self.partition.fit_through_ordinal
            || accepted_ordinal > self.partition.residual_through_ordinal
        {
            return Err(ModelUnknown::PhaseLeakage);
        }
        self.partition.residual_through_ordinal = accepted_ordinal;
        Ok(self)
    }

    /// Stable digest of actual normalized affine parameters, excluding clocks,
    /// future residual data and source-file publication timing. Rebuilding a
    /// deployment profile must reproduce this pre-residual freeze exactly.
    pub fn parameter_signature(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"ferrum.whole-wave.frozen-fit.v1\0");
        hash.update(self.revision.as_str().as_bytes());
        for (family, fitted) in &self.segments {
            hash.update(family);
            hash.update((fitted.samples as u64).to_le_bytes());
            for bits in fitted.affine.parameter_bits() {
                hash.update(bits.to_le_bytes());
            }
        }
        hash.finalize().into()
    }

    pub fn fit(
        fingerprint: ExecutionFingerprint,
        settings: WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        samples: &[WholeWaveObservationV1],
        now_ns: u64,
    ) -> Result<Self, ModelUnknown> {
        Self::fit_for_revision(
            fingerprint,
            settings,
            partition,
            samples,
            now_ns,
            WholeWaveModelRevision::OrderedV1,
        )
    }
    /// New capture only: every sample must carry the independently produced V2
    /// digest. This does not reinterpret profile6 or infer grouped evidence.
    pub fn fit_independent_attention_v2(
        fingerprint: ExecutionFingerprint,
        settings: WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        samples: &[WholeWaveObservationV1],
        now_ns: u64,
    ) -> Result<Self, ModelUnknown> {
        Self::fit_for_revision(
            fingerprint,
            settings,
            partition,
            samples,
            now_ns,
            WholeWaveModelRevision::IndependentAttentionV2,
        )
    }
    pub fn fit_work_support_v1(
        fingerprint: ExecutionFingerprint,
        settings: WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        samples: &[WholeWaveObservationV1],
        now_ns: u64,
    ) -> Result<Self, ModelUnknown> {
        WholeWaveModelRevision::IndependentAttentionWorkSupportV1.fit(
            fingerprint,
            settings,
            partition,
            samples,
            now_ns,
        )
    }
    fn fit_for_revision(
        fingerprint: ExecutionFingerprint,
        settings: WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        samples: &[WholeWaveObservationV1],
        now_ns: u64,
        revision: WholeWaveModelRevision,
    ) -> Result<Self, ModelUnknown> {
        let family = revision.family();
        settings_valid(&settings)?;
        partition.validate()?;
        let groups = group_samples(
            samples,
            &fingerprint,
            &settings,
            partition,
            now_ns,
            false,
            family,
        )?;
        let mut segments = BTreeMap::new();
        let mut unavailable = BTreeMap::new();
        for (family, points) in groups {
            if points.len() < settings.min_samples.get() {
                unavailable.insert(family, ModelUnknown::InsufficientFit);
                continue;
            }
            let affine = Affine::fit(&points)?;
            let support = Support::new(
                points
                    .iter()
                    .map(|(input, _, _)| coordinates(input, revision)),
            )?;
            let expires_at_ns = expires(&points, &settings)?;
            segments.insert(
                family,
                FittedSegment {
                    affine,
                    support,
                    expires_at_ns,
                    samples: points.len(),
                },
            );
        }
        if segments.is_empty() {
            return Err(ModelUnknown::InsufficientFit);
        }
        Ok(Self {
            family,
            revision,
            fingerprint,
            settings,
            partition,
            fit_at_ns: now_ns,
            last_fit_observed_at_ns: samples
                .last()
                .ok_or(ModelUnknown::InvalidSample)?
                .observed_at_ns,
            retained_samples: samples.len(),
            retained_rows: samples.iter().map(|s| s.exact.rows.len()).sum(),
            fit_calls: samples.iter().map(|s| s.call_id).collect(),
            unavailable,
            segments,
        })
    }
    /// Consumes the fitted state: residual observations can never refit it.
    pub fn calibrate(
        self,
        residual: &[WholeWaveObservationV1],
        now_ns: u64,
    ) -> Result<WholeWaveModelV1, ModelUnknown> {
        if now_ns < self.fit_at_ns {
            return Err(ModelUnknown::Clock);
        }
        if self
            .retained_samples
            .checked_add(residual.len())
            .is_none_or(|n| n > self.settings.max_retained_samples.get())
        {
            return Err(ModelUnknown::Capacity);
        }
        let rows = residual
            .iter()
            .try_fold(self.retained_rows, |n, s| n.checked_add(s.exact.rows.len()))
            .ok_or(ModelUnknown::Capacity)?;
        if rows > self.settings.max_retained_shape_rows.get() {
            return Err(ModelUnknown::Capacity);
        }
        let groups = group_samples(
            residual,
            &self.fingerprint,
            &self.settings,
            self.partition,
            now_ns,
            true,
            self.family,
        )?;
        if residual.iter().any(|s| self.fit_calls.contains(&s.call_id)) {
            return Err(ModelUnknown::DuplicateRecord);
        }
        if residual
            .first()
            .is_some_and(|s| s.observed_at_ns < self.last_fit_observed_at_ns)
        {
            return Err(ModelUnknown::Clock);
        }
        let mut segments = BTreeMap::new();
        let mut unavailable = self.unavailable.clone();
        for family in self.segments.keys() {
            unavailable.insert(*family, ModelUnknown::InsufficientResidual);
        }
        for (family, points) in groups {
            let Some(fit) = self.segments.get(&family) else {
                continue;
            };
            if fit
                .samples
                .checked_add(points.len())
                .is_none_or(|n| n > self.settings.max_samples_per_bucket.get())
            {
                return Err(ModelUnknown::Capacity);
            }
            if now_ns > fit.expires_at_ns {
                unavailable.insert(family, ModelUnknown::Stale);
                continue;
            }
            if points.len() < self.settings.min_samples.get() {
                continue;
            }
            let mut errors = Vec::with_capacity(points.len());
            if points
                .iter()
                .any(|(input, _, _)| !fit.support.contains(&coordinates(input, self.revision)))
            {
                unavailable.insert(family, ModelUnknown::JointSupport);
                continue;
            }
            for (input, wall, _) in &points {
                errors.push(wall.saturating_sub(fit.affine.predict(input)?));
            }
            errors.sort_unstable();
            let index = ((errors.len() as f64 * self.settings.residual_quantile).ceil() as usize)
                .saturating_sub(1)
                .min(errors.len() - 1);
            let support = Support::new(
                points
                    .iter()
                    .map(|(input, _, _)| coordinates(input, self.revision)),
            )?;
            let expiry = expires(&points, &self.settings)?.min(fit.expires_at_ns);
            unavailable.remove(&family);
            segments.insert(
                family,
                CalibratedSegment {
                    fit: fit.clone(),
                    support,
                    q99_ns: errors[index],
                    expires_at_ns: expiry,
                    residual_samples: points.len(),
                },
            );
        }
        if segments.is_empty() {
            return Err(ModelUnknown::InsufficientResidual);
        }
        let mut calibration_calls = self.fit_calls;
        calibration_calls.extend(residual.iter().map(|s| s.call_id));
        Ok(WholeWaveModelV1 {
            family: self.family,
            revision: self.revision,
            fingerprint: self.fingerprint,
            settings: self.settings,
            partition: self.partition,
            frozen_at_ns: now_ns,
            calibration_calls,
            unavailable,
            segments,
        })
    }
}
impl WholeWaveModelV1 {
    pub fn revision(&self) -> WholeWaveModelRevision {
        self.revision
    }
    pub fn selected_family(&self) -> SelectedStatisticalFamily {
        self.family
    }
    pub fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        exact: &CanonicalWaveCostShape,
        evidence: &StatisticalWaveEvidenceV1,
        now_ns: u64,
    ) -> Result<WholeWavePredictionV1, ModelUnknown> {
        if fingerprint != &self.fingerprint {
            return Err(ModelUnknown::WrongFingerprint);
        }
        if now_ns < self.frozen_at_ns {
            return Err(ModelUnknown::Clock);
        }
        let input = StatisticalModelInputV1::from_future(exact, evidence)?;
        self.predict_input(fingerprint, &input, now_ns)
    }
    /// Same lookup and error precedence as predict. The identity is captured
    /// only after the validated input selects the model's statistical family.
    pub fn predict_identified(
        &self,
        fingerprint: &ExecutionFingerprint,
        exact: &CanonicalWaveCostShape,
        evidence: &StatisticalWaveEvidenceV1,
        now_ns: u64,
    ) -> IdentifiedPredictionV1 {
        let mut query_identity = None;
        let prediction = (|| {
            if fingerprint != &self.fingerprint {
                return Err(ModelUnknown::WrongFingerprint);
            }
            if now_ns < self.frozen_at_ns {
                return Err(ModelUnknown::Clock);
            }
            let input = StatisticalModelInputV1::from_future(exact, evidence)?;
            self.predict_input_using(fingerprint, &input, now_ns, |identity| {
                query_identity = Some(identity);
            })
        })();
        IdentifiedPredictionV1 {
            query_identity,
            prediction,
        }
    }
    /// For an already validated, immutable execution edge. The planner binds
    /// this input to its exact alternative before construction; this is numeric
    /// cost evidence and never an executable authorization.
    pub fn predict_input(
        &self,
        fingerprint: &ExecutionFingerprint,
        input: &StatisticalModelInputV1,
        now_ns: u64,
    ) -> Result<WholeWavePredictionV1, ModelUnknown> {
        self.predict_input_using(fingerprint, input, now_ns, |_| {})
    }
    fn predict_input_using(
        &self,
        fingerprint: &ExecutionFingerprint,
        input: &StatisticalModelInputV1,
        now_ns: u64,
        identify: impl FnOnce(SelectedQueryIdentityV1),
    ) -> Result<WholeWavePredictionV1, ModelUnknown> {
        if fingerprint != &self.fingerprint {
            return Err(ModelUnknown::WrongFingerprint);
        }
        if now_ns < self.frozen_at_ns {
            return Err(ModelUnknown::Clock);
        }

        validate_input(input, &self.settings)?;
        let family = input.family_signature_for(self.family)?;
        identify(SelectedQueryIdentityV1::new(self.revision, *family));
        let segment = self.segments.get(family).ok_or_else(|| {
            self.unavailable
                .get(family)
                .copied()
                .unwrap_or(ModelUnknown::FamilyMissing)
        })?;
        if now_ns > segment.expires_at_ns {
            return Err(ModelUnknown::Stale);
        }
        let values = coordinates(input, self.revision);
        if !segment.fit.support.contains(&values) || !segment.support.contains(&values) {
            return Err(ModelUnknown::JointSupport);
        }
        let fitted_ns = segment.fit.affine.predict(input)?;
        let planning_ns = fitted_ns
            .checked_add(segment.q99_ns)
            .and_then(|n| n.checked_add(self.settings.drift_margin_ns))
            .ok_or(ModelUnknown::Numerical)?;
        Ok(WholeWavePredictionV1 {
            fitted_ns,
            residual_ns: segment.q99_ns,
            static_margin_ns: self.settings.drift_margin_ns,
            planning_ns,
            valid_until_ns: segment.expires_at_ns,
            fit_samples: segment.fit.samples,
            residual_samples: segment.residual_samples,
        })
    }
    /// Heldout never updates coefficients, residuals, support, clocks or margins.
    pub fn evaluate_heldout(
        &self,
        sample: &WholeWaveObservationV1,
        now_ns: u64,
    ) -> Result<HeldoutEvaluationV1, ModelUnknown> {
        if sample.accepted_ordinal <= self.partition.residual_through_ordinal {
            return Err(ModelUnknown::PhaseLeakage);
        }
        if self.calibration_calls.contains(&sample.call_id) {
            return Err(ModelUnknown::DuplicateRecord);
        }
        sample.input(
            &self.fingerprint,
            &self.settings,
            self.partition.source_sha256,
            now_ns,
        )?;
        let identified =
            self.predict_identified(&sample.fingerprint, &sample.exact, &sample.selected, now_ns);
        let prediction = identified.prediction;
        let underestimate_ns = prediction
            .as_ref()
            .ok()
            .map(|p| sample.wall_ns.saturating_sub(p.planning_ns));
        Ok(HeldoutEvaluationV1 {
            query_identity: identified.query_identity,
            prediction,
            actual_ns: sample.wall_ns,
            underestimate_ns,
        })
    }
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}
type Point = (StatisticalModelInputV1, u64, u64);
fn group_samples(
    samples: &[WholeWaveObservationV1],
    fingerprint: &ExecutionFingerprint,
    settings: &WholeWaveSettingsV1,
    partition: CalibrationPartitionV1,
    now: u64,
    residual: bool,
    family: SelectedStatisticalFamily,
) -> Result<BTreeMap<[u8; 32], Vec<Point>>, ModelUnknown> {
    if samples.len() > settings.max_retained_samples.get() {
        return Err(ModelUnknown::Capacity);
    }
    let mut result: BTreeMap<[u8; 32], Vec<Point>> = BTreeMap::new();
    let mut ordinals = BTreeSet::new();
    let mut calls = BTreeSet::new();
    let mut retained_rows = 0usize;
    let mut previous = None;
    for sample in samples {
        let ordinal = sample.accepted_ordinal;
        if ordinal == 0
            || (!residual && ordinal > partition.fit_through_ordinal)
            || (residual
                && (ordinal <= partition.fit_through_ordinal
                    || ordinal > partition.residual_through_ordinal))
        {
            return Err(ModelUnknown::PhaseLeakage);
        }
        if !ordinals.insert(ordinal) || !calls.insert(sample.call_id) {
            return Err(ModelUnknown::DuplicateRecord);
        }
        if previous.is_some_and(|(old_ordinal, old_time)| {
            ordinal <= old_ordinal || sample.observed_at_ns < old_time
        }) {
            return Err(ModelUnknown::Clock);
        }
        previous = Some((ordinal, sample.observed_at_ns));
        let input = sample.input(fingerprint, settings, partition.source_sha256, now)?;
        retained_rows = retained_rows
            .checked_add(sample.exact.rows.len())
            .ok_or(ModelUnknown::Capacity)?;
        if retained_rows > settings.max_retained_shape_rows.get() {
            return Err(ModelUnknown::Capacity);
        }
        let key = *input.family_signature_for(family)?;
        if !result.contains_key(&key) && result.len() == settings.max_buckets.get() {
            return Err(ModelUnknown::Capacity);
        }
        let bucket = result.entry(key).or_default();
        if bucket.len() == settings.max_samples_per_bucket.get() {
            return Err(ModelUnknown::Capacity);
        }
        bucket.push((input, sample.wall_ns, sample.observed_at_ns));
    }
    Ok(result)
}
fn expires(points: &[Point], settings: &WholeWaveSettingsV1) -> Result<u64, ModelUnknown> {
    points
        .iter()
        .try_fold(None, |earliest: Option<u64>, (_, _, at)| {
            let expiry = at
                .checked_add(settings.max_sample_age_ns.get())
                .ok_or(ModelUnknown::Clock)?;
            Ok::<Option<u64>, ModelUnknown>(
                earliest.map_or(Some(expiry), |old| Some(old.min(expiry))),
            )
        })?
        .ok_or(ModelUnknown::InvalidSample)
}
