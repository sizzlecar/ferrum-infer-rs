//! Capture-only qualification from this call's real consumed settlement receipts.
//! This is not a fit input, predictor, source/profile import, or execution permit.
use super::*;
use ferrum_interfaces::execution_cost::{
    HostTerminalExpectationV1, UnsettledStructuredWaveEvidenceV1,
};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredSettlementUnknown {
    MissingProducer,
    MissingIdentity,
    ExactBindingMismatch,
    Unsettled(HostStageCompleteness),
    HostIdentityMismatch,
    InvalidClock,
}

/// No public constructor or deserializer. The private binder below can only
/// read the original recorder and host progress after PendingHostRow::settle
/// consumed the real owner and its completion/cancellation/cache receipts.
/// The containing HostStageEvidenceV1 retains the real serial host row order;
/// the recipe retains physical order. Neither order is inferred from the other.
#[derive(Debug, Clone, Serialize)]
pub struct QualifiedStructuredWaveEvidenceV1 {
    protocol: &'static str,
    call_id: u64,
    #[serde(serialize_with = "serialize_recipe")]
    recipe: Arc<UnsettledStructuredWaveEvidenceV1>,
    stage_binding: [u8; 32],
    executor_envelope_ns: u64,
    host_settled_after_executor_ns: u64,
    full_wall_ns: u64,
}
fn serialize_recipe<S: serde::Serializer>(
    recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    recipe.as_ref().serialize(serializer)
}
impl QualifiedStructuredWaveEvidenceV1 {
    pub fn recipe(&self) -> &UnsettledStructuredWaveEvidenceV1 {
        &self.recipe
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn stage_binding(&self) -> [u8; 32] {
        self.stage_binding
    }
    pub fn full_wall_ns(&self) -> u64 {
        self.full_wall_ns
    }
    pub(super) fn retained_rows(&self) -> usize {
        self.recipe.retained_rows()
    }

    /// Detect mutation of the enclosing public diagnostic fields. This is not
    /// a loader: the qualified value itself has no public constructor/import.
    pub fn validate_host_stages(
        &self,
        stages: &HostStageEvidenceV1,
    ) -> Result<(), StructuredSettlementUnknown> {
        if self.call_id != stages.call_id
            || stages.completeness != HostStageCompleteness::CompleteSingleWave
            || stages.full_wall_ns != Some(self.full_wall_ns)
            || self.stage_binding != stage_binding(stages)?
        {
            return Err(StructuredSettlementUnknown::HostIdentityMismatch);
        }
        Ok(())
    }
}

// Streaming serialization into a small digest state; no extra row/JSON Vec.
// Explicitly omit the qualified sidecar itself (no recursive serialization).
struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn stage_binding(stages: &HostStageEvidenceV1) -> Result<[u8; 32], StructuredSettlementUnknown> {
    let mut writer = HashWriter(Sha256::new());
    writer.0.update(b"ferrum.structured-host-settlement.v1");
    #[derive(Serialize)]
    struct ShapeBinding<'a> {
        #[serde(serialize_with = "wire::serialize_shape")]
        actual_shape: &'a Option<WaveExecutionShape>,
        statistics: &'a Option<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>,
        independent_attention:
            Option<&'a ferrum_interfaces::execution_cost::IndependentAttentionWaveEvidenceV2>,
    }
    let shape = ShapeBinding {
        actual_shape: &stages.actual_shape,
        statistics: &stages.statistical_evidence,
        independent_attention: stages
            .statistical_evidence
            .as_ref()
            .and_then(|v| v.independent_attention_v2()),
    };
    serde_json::to_writer(
        &mut writer,
        &(
            stages.call_id,
            stages.fingerprint.as_ref().map(|value| {
                (
                    value.model_weights,
                    value.numerical_policy,
                    value.device_runtime,
                    value.execution_config,
                )
            }),
            &stages.rows,
            shape,
            stages.prepare_started_at_ns,
            stages.executor_returned_at_ns,
            stages.finalized_at_ns,
            stages.full_wall_ns,
            stages.completeness,
        ),
    )
    .map_err(|_| StructuredSettlementUnknown::HostIdentityMismatch)?;
    Ok(writer.0.finalize().into())
}

pub(super) fn qualify(
    call: &EngineCostCall,
    actual: &ActualWaveShape,
    stages: &HostStageEvidenceV1,
) -> Result<QualifiedStructuredWaveEvidenceV1, StructuredSettlementUnknown> {
    use StructuredSettlementUnknown as U;
    if stages.completeness != HostStageCompleteness::CompleteSingleWave {
        return Err(U::Unsettled(stages.completeness));
    }
    if stages.fingerprint.is_none() || stages.actual_shape.is_none() {
        return Err(U::MissingIdentity);
    }
    let statistics = actual
        .statistical_evidence
        .as_ref()
        .ok_or(U::MissingProducer)?;
    statistics
        .validate_actual(actual)
        .map_err(|_| U::ExactBindingMismatch)?;
    let recipe = statistics
        .structured_capture()
        .ok_or(U::MissingProducer)?
        .map_err(|_| U::MissingProducer)?;
    recipe
        .validate_actual(actual)
        .map_err(|_| U::ExactBindingMismatch)?;
    if call.call_id.get() != stages.call_id
        || stages.rows.len() != actual.rows.len()
        || recipe.physical_host_rows().len() != actual.rows.len()
    {
        return Err(U::HostIdentityMismatch);
    }
    // The original participants, recorder and receipt-derived rows must agree.
    // make_host_stages already checked unique host ordinals and serial clocks.
    // Recheck these identity edges here rather than trusting public stage fields.
    for (position, ((physical, observed), declared)) in actual
        .rows
        .iter()
        .zip(&stages.rows)
        .zip(recipe.physical_host_rows())
        .enumerate()
    {
        let participant = call
            .participants
            .iter()
            .find(|p| {
                p.request_id == physical.request_id
                    && p.owner_incarnation == physical.owner_incarnation
                    && p.work_generation == physical.work_generation
                    && p.input_index == physical.input_index
            })
            .ok_or(U::HostIdentityMismatch)?;
        if declared.physical_position as usize != position
            || observed.request_id != participant.request_id
            || observed.owner_incarnation != participant.owner_incarnation
            || observed.work_generation != participant.work_generation
            || observed.input_index != participant.input_index
            || observed.completeness != HostStageCompleteness::CompleteSingleWave
        {
            return Err(U::HostIdentityMismatch);
        }
        let expected_work = match physical.work {
            ActualRowWork::Decode { kv_tokens } => HostStageWork::Decode { kv_tokens },
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => HostStageWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            },
            _ => return Err(U::HostIdentityMismatch),
        };
        if observed.actual_work != expected_work {
            return Err(U::HostIdentityMismatch);
        }
        match declared.terminal_expectation {
            HostTerminalExpectationV1::NoTokenProduced if observed.terminal.is_some() => {
                return Err(U::HostIdentityMismatch)
            }
            // At capacity, require the actual terminal receipt, not Length as a
            // guessed reason: EOS/stop may win at the last token.
            HostTerminalExpectationV1::LengthBoundary if observed.terminal.is_none() => {
                return Err(U::HostIdentityMismatch)
            }
            _ => {}
        }
        if let Some(terminal) = &observed.terminal {
            if !matches!(
                terminal.finish_reason,
                FinishReason::Length | FinishReason::EOS | FinishReason::Stop
            ) {
                return Err(U::Unsettled(HostStageCompleteness::Failed));
            }
            let host = participant.host_features.ok_or(U::HostIdentityMismatch)?;
            if host.state.generated_tokens_before.checked_add(1) != Some(terminal.generated_tokens)
            {
                return Err(U::HostIdentityMismatch);
            }
            // Receipt classification remains the product's real no-extra-work
            // condition. Never reinterpret reclaimed bytes or epoch movement.
            if terminal.output_failed
                || terminal.physical_failed
                || terminal.scheduler_failed
                || !terminal.terminal_handoff_succeeded
                || !terminal.request_slot_closed
                || !terminal.owner_matched
                || terminal.pending_restore_removed
                || terminal.other_physical_resources
                || terminal.admission_cancellation_work != ExecutorCompletionWork::NoAdditionalWork
                || terminal.cache_completion_work != ExecutorCompletionWork::NoAdditionalWork
            {
                return Err(U::Unsettled(HostStageCompleteness::AdditionalOrUnknownWork));
            }
        }
    }
    let prepare = stages.prepare_started_at_ns.ok_or(U::InvalidClock)?;
    let returned = stages.executor_returned_at_ns.ok_or(U::InvalidClock)?;
    let end = stages
        .rows
        .iter()
        .try_fold(returned, |end, row| {
            row.settled_at_ns.map(|time| end.max(time))
        })
        .ok_or(U::InvalidClock)?;
    let executor_envelope_ns = returned.checked_sub(prepare).ok_or(U::InvalidClock)?;
    let host_settled_after_executor_ns = end.checked_sub(returned).ok_or(U::InvalidClock)?;
    let full_wall_ns = end
        .checked_sub(prepare)
        .filter(|v| *v > 0)
        .ok_or(U::InvalidClock)?;
    if stages.full_wall_ns != Some(full_wall_ns) {
        return Err(U::InvalidClock);
    }
    Ok(QualifiedStructuredWaveEvidenceV1 {
        protocol: "ferrum.structured-host-settled-capture.v1",
        call_id: stages.call_id,
        recipe: Arc::clone(recipe),
        stage_binding: stage_binding(stages)?,
        executor_envelope_ns,
        host_settled_after_executor_ns,
        full_wall_ns,
    })
}
