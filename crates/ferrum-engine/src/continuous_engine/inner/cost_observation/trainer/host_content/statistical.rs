//! Conversion at a real numbered FIFO entry. No legacy observation upgrades,
//! default worker/model switch, raw rejoining by a separately counted ordinal,
//! or execution authority is introduced here.
use super::*;
use model::statistical::model::{ModelUnknown, WholeWaveObservationV1};

pub(in crate::continuous_engine::inner::cost_observation) fn whole_wave_observation(
    entry: &CostEvidenceEntry,
    accepted_ordinal: u64,
    source_sha256: [u8; 32],
) -> Result<WholeWaveObservationV1, ModelUnknown> {
    if accepted_ordinal == 0 || source_sha256 == [0; 32] {
        return Err(ModelUnknown::WrongSource);
    }
    let actual = complete_observation(entry)?;
    Ok(WholeWaveObservationV1 {
        source_sha256,
        accepted_ordinal,
        call_id: actual.call_id,
        fingerprint: actual.fingerprint,
        exact: actual.exact,
        selected: actual.selected,
        boundary: actual.boundary,
        outcome: actual.outcome,
        observed_at_ns: actual.observed_at_ns,
        wall_ns: actual.wall_ns,
    })
}

/// Actual successful receipt without an offline capture identity. Ordinary
/// serving audit and numbered calibration use the same boundary validation.
pub(in crate::continuous_engine::inner::cost_observation) struct CompleteSelectedObservation {
    pub call_id: u64,
    pub fingerprint: model::ExecutionFingerprint,
    pub exact: CanonicalWaveCostShape,
    pub selected: ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
    pub boundary: model::CostBoundary,
    pub outcome: model::WaveObservationOutcome,
    pub observed_at_ns: u64,
    pub wall_ns: u64,
}
pub(in crate::continuous_engine::inner::cost_observation) fn complete_observation(
    entry: &CostEvidenceEntry,
) -> Result<CompleteSelectedObservation, ModelUnknown> {
    complete_observation_with_graph(entry, false)
}

/// V2 alone supports complete warm graph work. This still consumes the actual
/// private settlement and its original attached recipe; numerical import never
/// calls this receipt adapter. Legacy selected/V1 callers retain Disabled-only.
pub(in crate::continuous_engine::inner::cost_observation) fn complete_structured_observation_v2(
    entry: &CostEvidenceEntry,
) -> Result<CompleteSelectedObservation, ModelUnknown> {
    complete_observation_with_graph(entry, true)
}

fn complete_observation_with_graph(
    entry: &CostEvidenceEntry,
    structured_v2: bool,
) -> Result<CompleteSelectedObservation, ModelUnknown> {
    let (stages, legacy) = match entry {
        CostEvidenceEntry::Training { sample, stages } => {
            let stages = stages.as_deref().ok_or(ModelUnknown::Evidence(
                StatisticalEvidenceUnknown::MissingProducer,
            ))?;
            if stages.fingerprint.as_ref() != Some(&sample.fingerprint)
                || stages.actual_shape.as_ref() != Some(&sample.actual_shape)
            {
                return Err(ModelUnknown::InvalidSample);
            }
            (stages, None)
        }
        CostEvidenceEntry::StagesOnly {
            stages,
            legacy_rejection,
        } => (stages.as_ref(), Some(*legacy_rejection)),
    };
    let observed = sample(Some(stages), legacy, true).map_err(|_| ModelUnknown::InvalidSample)?;
    let shape = &observed.actual_shape;
    if shape.path != model::WaveExecutionPath::PlanRuntime
        || !(shape.graph_state == model::WaveGraphState::Disabled
            || (structured_v2
                && matches!(
                    shape.graph_state,
                    model::WaveGraphState::Warm | model::WaveGraphState::ConfiguredEager
                )))
        || shape.order != model::BatchOrderSemantics::Ordered
        || shape.restore_bytes != 0
        || shape.maintenance_bytes != 0
        || shape.maintenance_units != 0
    {
        return Err(ModelUnknown::Evidence(
            StatisticalEvidenceUnknown::UnsupportedWave,
        ));
    }
    let kind = match shape.kind {
        model::WaveKind::Decode => ActualWaveKind::Decode,
        model::WaveKind::Prefill => ActualWaveKind::Prefill,
        model::WaveKind::Mixed => ActualWaveKind::Mixed,
        _ => return Err(ModelUnknown::InvalidSample),
    };
    let rows = stages
        .rows
        .iter()
        .map(|r| match r.actual_work {
            HostStageWork::Decode { kv_tokens } => Ok(ActualRowWork::Decode { kv_tokens }),
            HostStageWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => Ok(ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            }),
            _ => Err(ModelUnknown::InvalidSample),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exact = CanonicalWaveCostShape {
        kind,
        path: ActualWavePath::PlanRuntime,
        graph: match shape.graph_state {
            model::WaveGraphState::Warm => ActualWaveGraphState::Warm,
            model::WaveGraphState::ConfiguredEager => ActualWaveGraphState::ConfiguredEager,
            model::WaveGraphState::Disabled => ActualWaveGraphState::Disabled,
            model::WaveGraphState::Cold => return Err(ModelUnknown::InvalidSample),
        },
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: shape.provider_signature,
        output_policy_signature: shape.output_policy_signature,
        numeric_features: shape.numeric_features.clone(),
        host_content_features: shape.host_content_features,
        row_multiset_features: shape.row_multiset_features.clone(),
        rows,
        recurrent_state_bytes: shape.recurrent_state_bytes,
    };
    let selected = stages
        .statistical_evidence
        .clone()
        .ok_or(ModelUnknown::Evidence(
            StatisticalEvidenceUnknown::MissingProducer,
        ))?;
    selected.validate_exact(&exact)?;
    if structured_v2 {
        let qualified = stages
            .structured_evidence
            .as_ref()
            .and_then(|value| value.as_ref().ok())
            .ok_or(ModelUnknown::InvalidSample)?;
        qualified
            .validate_host_stages(stages)
            .map_err(|_| ModelUnknown::InvalidSample)?;
        let recipe = selected
            .structured_capture()
            .ok_or(ModelUnknown::Evidence(
                StatisticalEvidenceUnknown::MissingProducer,
            ))??;
        recipe.validate_exact(&exact)?;
        recipe.algorithm_work()?.validate_structure(recipe)?;
        if !std::ptr::eq(qualified.recipe(), recipe.as_ref())
            || qualified.full_wall_ns() != observed.timing.wall_total_ns
        {
            return Err(ModelUnknown::InvalidSample);
        }
    }
    if stages.call_id == 0 {
        return Err(ModelUnknown::InvalidSample);
    }
    Ok(CompleteSelectedObservation {
        call_id: stages.call_id,
        fingerprint: observed.fingerprint,
        exact,
        selected,
        boundary: observed.boundary,
        outcome: observed.outcome,
        observed_at_ns: observed.observed_at_ns,
        wall_ns: observed.timing.wall_total_ns,
    })
}
