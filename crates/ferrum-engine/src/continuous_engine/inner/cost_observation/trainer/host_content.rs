//! Worker-side conversion of already completed private producer receipts.
//! This never turns a legacy token-commit interval into host-settled timing.
use super::super::audit::{HostContentEvaluation, HostContentRejection};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model as model;
pub(super) mod statistical;

pub(super) struct EvaluatedHostContent {
    pub evaluation: HostContentEvaluation,
    pub sample: Option<model::WaveCostObservation>,
}

fn sample(
    stages: Option<&HostStageEvidenceV1>,
    legacy: Option<CostCallRejection>,
    row_multiset: bool,
) -> Result<model::WaveCostObservation, HostContentRejection> {
    use HostContentRejection as R;
    let stages = stages.ok_or(R::MissingStages)?;
    if legacy.is_some_and(|reason| reason != CostCallRejection::Composite) {
        return Err(R::LegacyFailure);
    }
    if stages.schema_version != 1
        || stages.completeness != HostStageCompleteness::CompleteSingleWave
        || stages.rows.is_empty()
        || stages
            .rows
            .iter()
            .any(|row| row.completeness != HostStageCompleteness::CompleteSingleWave)
    {
        return Err(R::Incomplete);
    }
    let fingerprint = stages.fingerprint.as_ref().ok_or(R::IdentityMissing)?;
    let shape = stages.actual_shape.as_ref().ok_or(R::IdentityMissing)?;
    if !row_multiset
        && shape
            .host_content_features
            .as_ref()
            .is_none_or(|features| features.validate().is_err())
    {
        return Err(R::DomainMissing);
    }
    if row_multiset {
        let features = shape
            .row_multiset_features
            .as_ref()
            .ok_or(R::DomainMissing)?;
        features
            .validate(stages.rows.len())
            .map_err(|_| R::DomainMissing)?;
        shape
            .numeric_features
            .as_ref()
            .ok_or(R::DomainMissing)?
            .validate(stages.rows.len())
            .map_err(|_| R::DomainMissing)?;
        // These are physical rows, not host processing order or sorted model
        // tuples. The model alone may canonicalize a supported permutation.
        let mut decodes = shape.decode_kv_tokens.iter();
        let mut prefills = shape.prefill_chunks.iter();
        for (feature, row) in features.rows.iter().zip(&stages.rows) {
            let matches = match (feature.role, row.actual_work) {
                (HostRowRoleV2::Decode, HostStageWork::Decode { kv_tokens }) => {
                    decodes.next() == Some(&kv_tokens)
                }
                (
                    HostRowRoleV2::Prefill,
                    HostStageWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    },
                ) => prefills.next().is_some_and(|shape| {
                    shape.offset == offset
                        && shape.count.get() == count
                        && shape.total_prompt_tokens.get() == total_prompt_tokens
                }),
                _ => false,
            };
            if !matches {
                return Err(R::IdentityMissing);
            }
        }
        if decodes.next().is_some() || prefills.next().is_some() {
            return Err(R::IdentityMissing);
        }
    }
    if shape
        .decode_kv_tokens
        .len()
        .checked_add(shape.prefill_chunks.len())
        != Some(stages.rows.len())
    {
        return Err(R::IdentityMissing);
    }
    let prepare = stages.prepare_started_at_ns.ok_or(R::InvalidClock)?;
    let returned = stages.executor_returned_at_ns.ok_or(R::InvalidClock)?;
    let observed_at_ns = stages.finalized_at_ns.ok_or(R::InvalidClock)?;
    let mut latest = returned;
    if returned < prepare || observed_at_ns < returned {
        return Err(R::InvalidClock);
    }
    for row in &stages.rows {
        let start = row.host_started_at_ns.ok_or(R::InvalidClock)?;
        let committed = row.token_committed_at_ns.ok_or(R::InvalidClock)?;
        let end = row.settled_at_ns.ok_or(R::InvalidClock)?;
        if start < returned
            || committed < start
            || end < committed
            || end > observed_at_ns
            || row
                .output_published_at_ns
                .is_some_and(|published| published < committed || published > end)
        {
            return Err(R::InvalidClock);
        }
        latest = latest.max(end);
    }
    let wall = latest
        .checked_sub(prepare)
        .filter(|wall| *wall > 0)
        .ok_or(R::InvalidClock)?;
    if stages.full_wall_ns != Some(wall) {
        return Err(R::InvalidClock);
    }
    Ok(model::WaveCostObservation {
        fingerprint: fingerprint.clone(),
        actual_shape: shape.clone(),
        boundary: model::CostBoundary::PreparationToHostSettledV1,
        outcome: model::WaveObservationOutcome::Completed,
        timing: model::WaveTiming {
            wall_total_ns: wall,
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns,
    })
}

pub(super) fn observe(
    stages: Option<&HostStageEvidenceV1>,
    legacy: Option<CostCallRejection>,
    row_multiset: bool,
    trainer: Option<&mut CostTrainer>,
    previous: Option<&EngineCostSnapshot>,
) -> EvaluatedHostContent {
    let sample = match sample(stages, legacy, row_multiset) {
        Ok(sample) => sample,
        Err(reason) => {
            return EvaluatedHostContent {
                evaluation: HostContentEvaluation {
                    rejection: Some(reason),
                    training: TrainingDisposition::Unavailable,
                    pre_update_prediction: None,
                },
                sample: None,
            }
        }
    };
    let prediction = PreUpdatePrediction::query(previous, &sample);
    let disposition =
        TrainingDisposition::from_result(trainer.map(|trainer| trainer.observe(sample.clone())));
    let prediction = prediction.with_comparison(Some(sample.timing.wall_total_ns), disposition);
    EvaluatedHostContent {
        evaluation: HostContentEvaluation {
            rejection: None,
            training: disposition,
            pre_update_prediction: Some(prediction),
        },
        sample: Some(sample),
    }
}
