//! Actual physical-wave shape from canonical participants and device evidence.
//! No model-name/config guesses, JSON, input-token hashing, or device timing.

use super::super::*;
use ferrum_interfaces::execution_cost::*;
mod route;
pub(in crate::executor::vnext_executor) fn actual_shape<R: DeviceRuntime>(
    executor: &VNextModelExecutor<R>,
    context: &PlanRuntimeCostObservationContext<'_>,
    participants: &[VNextExecutionParticipant<'_, R>],
    kind: VNextExecutionWaveKind,
    output_mode: VNextProductOutputMode,
    token_masks: &[VNextProductTokenMaskSubmissionPlan],
    attribution: Option<&BoundDeviceSubmissionAttribution>,
    retries: u32,
    core_readback_route: CoreReadbackRoute,
) -> std::result::Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
    actual_shape_from_device_with_capture(
        executor,
        participants,
        kind,
        output_mode,
        token_masks,
        attribution.map(BoundDeviceSubmissionAttribution::device),
        retries,
        core_readback_route,
        context.structured_capture_enabled(),
        |request_id| context.participant(request_id),
    )
}

/// One projection for completed observation and the final pre-submit check.
/// The device attribution is borrowed from actual encoding; work, output and
/// mask roles come from actual prepared participants, never expected hashes.
/// The callback supplies only correlated host inputs/lifecycle identity. Its
/// evidence must come from the caller's current validated request snapshot.
pub(in crate::executor::vnext_executor) fn actual_shape_from_device<'h, R: DeviceRuntime>(
    executor: &VNextModelExecutor<R>,
    participants: &[VNextExecutionParticipant<'_, R>],
    kind: VNextExecutionWaveKind,
    output_mode: VNextProductOutputMode,
    token_masks: &[VNextProductTokenMaskSubmissionPlan],
    attribution: Option<&DeviceSubmissionAttribution>,
    retries: u32,
    core_readback_route: CoreReadbackRoute,
    correlate: impl FnMut(&ferrum_types::RequestId) -> Option<&'h CostObservationParticipant>,
) -> std::result::Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
    actual_shape_from_device_with_capture(
        executor,
        participants,
        kind,
        output_mode,
        token_masks,
        attribution,
        retries,
        core_readback_route,
        false,
        correlate,
    )
}

pub(in crate::executor::vnext_executor) fn actual_shape_from_device_with_capture<
    'h,
    R: DeviceRuntime,
>(
    executor: &VNextModelExecutor<R>,
    participants: &[VNextExecutionParticipant<'_, R>],
    kind: VNextExecutionWaveKind,
    output_mode: VNextProductOutputMode,
    token_masks: &[VNextProductTokenMaskSubmissionPlan],
    attribution: Option<&DeviceSubmissionAttribution>,
    retries: u32,
    core_readback_route: CoreReadbackRoute,
    structured_capture: bool,
    mut correlate: impl FnMut(&ferrum_types::RequestId) -> Option<&'h CostObservationParticipant>,
) -> std::result::Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
    if participants.is_empty()
        || participants.len() > 1024
        || token_masks.len() != participants.len()
    {
        return Err(ActualWaveEvidenceUnknown::ShapeOverflow);
    }
    if executor.checkpoint_capture.is_some() || executor.diagnostic_fault.is_some() {
        return Err(ActualWaveEvidenceUnknown::OutputPolicy);
    }
    let route::ObservedRoute {
        mut canonical,
        graph,
    } = route::actual_route_with_capture(
        attribution,
        |index| {
            executor
                .providers
                .providers()
                .get(index as usize)
                .map(|provider| {
                    let descriptor = provider.descriptor();
                    CostProviderIdentity {
                        provider_id: descriptor.provider_id().as_str(),
                        implementation_fingerprint: descriptor
                            .provider_implementation_fingerprint(),
                        operation_fingerprint: descriptor.operation_fingerprint(),
                    }
                })
        },
        executor.runtime.cost_graph_capture_capability(),
        match output_mode {
            VNextProductOutputMode::FullLogits => CostProductOutput::FullLogits,
            VNextProductOutputMode::GreedyToken => CostProductOutput::GreedyToken,
        },
        retries,
        structured_capture,
    )?;
    canonical
        .core_readback_route(core_readback_route)
        .map_err(|_| ActualWaveEvidenceUnknown::OutputPolicy)?;
    // Fixed state comes from the actual compiled numerical program. A general
    // non-KV token-scaled state still needs an actual touched-state contract.
    if executor
        .sequence_state_memory
        .other_token_scaled_bytes_per_token
        != 0
    {
        return Err(ActualWaveEvidenceUnknown::RecurrentState);
    }
    let recurrent_state_bytes = executor
        .sequence_state_memory
        .fixed_bytes_per_sequence
        .checked_mul(participants.len() as u64)
        .ok_or(ActualWaveEvidenceUnknown::ShapeOverflow)?;
    let mut rows = Vec::with_capacity(participants.len());
    for (participant, mask) in participants.iter().zip(token_masks) {
        let correlation = correlate(participant.sequence.request_id())
            .filter(|correlation| &correlation.request_id == participant.sequence.request_id())
            .ok_or(ActualWaveEvidenceUnknown::ParticipantCorrelation)?;
        let host_policy_signature = correlation
            .output_policy_signature
            .ok_or(ActualWaveEvidenceUnknown::OutputPolicy)?;
        let range = participant.span.immediate_token_range();
        let offset =
            u32::try_from(range.start).map_err(|_| ActualWaveEvidenceUnknown::ShapeOverflow)?;
        let count = u32::try_from(
            range
                .end
                .checked_sub(range.start)
                .ok_or(ActualWaveEvidenceUnknown::ShapeOverflow)?,
        )
        .map_err(|_| ActualWaveEvidenceUnknown::ShapeOverflow)?;
        let (work, output) = match participant.output_role {
            VNextParticipantOutputRole::Decode(policy) => {
                let repetition = product_repetition_input(Some(policy), output_mode);
                (
                    ActualRowWork::Decode { kv_tokens: offset },
                    CostRowOutput::Decode {
                        requires_full_logits: policy.requires_full_logits(),
                        repetition_tokens: repetition.token_ids.len() as u64,
                        repetition_penalty_bits: repetition.penalty.to_bits(),
                    },
                )
            }
            role => (
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens: u32::try_from(participant.span.full_input_tokens())
                        .map_err(|_| ActualWaveEvidenceUnknown::ShapeOverflow)?,
                },
                CostRowOutput::Prefill {
                    final_logits: matches!(role, VNextParticipantOutputRole::FinalPrefill),
                },
            ),
        };
        canonical
            .row(CanonicalCostRow {
                work,
                host_policy_signature,
                host_features: correlation.host_features,
                mask_upload_required: mask.upload_required,
                output,
            })
            .map_err(|_| ActualWaveEvidenceUnknown::OutputPolicy)?;
        rows.push(ActualWaveRow {
            request_id: correlation.request_id.clone(),
            owner_incarnation: correlation.owner_incarnation,
            work_generation: correlation.work_generation,
            input_index: correlation.input_index,
            work,
        });
    }
    let kind = match kind {
        VNextExecutionWaveKind::Decode => ActualWaveKind::Decode,
        VNextExecutionWaveKind::Prefill => ActualWaveKind::Prefill,
        VNextExecutionWaveKind::Mixed => ActualWaveKind::Mixed,
    };
    let path = if retries > 0 {
        ActualWavePath::UnsupportedFallback
    } else {
        ActualWavePath::PlanRuntime
    };
    let canonical = canonical
        .finish_with_captured_structure(
            kind,
            path,
            graph,
            ActualWaveRowOrder::Ordered,
            recurrent_state_bytes,
        )
        .map_err(|_| ActualWaveEvidenceUnknown::ProviderPath)?;
    let statistical_evidence = canonical.statistical.ok();
    let canonical = canonical.exact;
    let shape = ActualWaveShape {
        statistical_evidence,
        kind,
        path,
        graph,
        row_order: canonical.row_order,
        provider_signature: canonical.provider_signature,
        output_policy_signature: canonical.output_policy_signature,
        numeric_features: canonical.numeric_features,
        host_content_features: canonical.host_content_features,
        row_multiset_features: canonical.row_multiset_features,
        rows,
        recurrent_state_bytes,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    };
    shape
        .validate(1024)
        .map_err(|_| ActualWaveEvidenceUnknown::ShapeOverflow)?;
    Ok(shape)
}

/// Preserve the existing compile-time no-timing specialization when only cost
/// attribution is requested. No host or device profiling clock is enabled.
pub(in crate::executor::vnext_executor) struct NoTiming;
impl DeviceSubmissionTimingSink for NoTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        unreachable!("disabled timing sink");
    }
}
impl SubmissionWaveDispatchTimingSink for NoTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        unreachable!("disabled timing sink");
    }
}

fn graph_state(
    capability: DeviceCostGraphCaptureCapability,
    replayed: bool,
    evidence: Option<ferrum_interfaces::vnext::DeviceSubmissionGraphEvidence>,
) -> std::result::Result<ActualWaveGraphState, ActualWaveEvidenceUnknown> {
    match (capability, replayed) {
        (DeviceCostGraphCaptureCapability::Unsupported, false)
            if evidence.is_none_or(|proof| proof.proves_unconfigured_eager()) =>
        {
            Ok(ActualWaveGraphState::Disabled)
        }
        (DeviceCostGraphCaptureCapability::Supported, false)
            if evidence.is_some_and(|proof| proof.proves_unconfigured_eager()) =>
        {
            Ok(ActualWaveGraphState::Disabled)
        }
        (DeviceCostGraphCaptureCapability::Supported, true)
            if evidence.is_some_and(|proof| proof.proves_warm_direct_replay()) =>
        {
            Ok(ActualWaveGraphState::Warm)
        }
        (DeviceCostGraphCaptureCapability::Supported, false)
            if evidence.is_some_and(|proof| proof.proves_configured_eager_observation()) =>
        {
            Ok(ActualWaveGraphState::ConfiguredEager)
        }
        // Replayed attribution on a runtime declaring no graph support is a
        // contradiction. Eager work on other runtimes may also capture graphs;
        // neither case is silently promoted to a known Cold/Warm cost shape.
        _ => Err(ActualWaveEvidenceUnknown::GraphPath),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_eager_requires_same_wave_proof_and_rejects_adaptive_work() {
        use ferrum_interfaces::vnext::{
            DeviceCostGraphConfiguration as C, DeviceCostGraphStreamState as S,
            DeviceSubmissionGraphEvidence as E,
        };
        let ready = S::new(C::OnDemand, 3, 2, 0).unwrap();
        let changed = S::new(C::OnDemand, 4, 2, 0).unwrap();
        let exact = E::new(ready, ready, false, 0, 0, 0, 0, 0).unwrap();
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                false,
                Some(exact)
            ),
            Ok(ActualWaveGraphState::ConfiguredEager)
        );
        for capability in [
            DeviceCostGraphCaptureCapability::Unsupported,
            DeviceCostGraphCaptureCapability::Unknown,
        ] {
            assert_eq!(
                graph_state(capability, false, Some(exact)),
                Err(ActualWaveEvidenceUnknown::GraphPath)
            );
        }
        for proof in [
            None,
            Some(E::new(ready, changed, false, 0, 0, 0, 0, 0).unwrap()),
            Some(E::new(ready, ready, true, 0, 0, 0, 0, 0).unwrap()),
            Some(E::new(ready, ready, false, 1, 0, 0, 0, 0).unwrap()),
            Some(E::new(ready, ready, false, 1, 1, 0, 1, 0).unwrap()),
            Some(E::new(ready, ready, false, 1, 0, 1, 0, 0).unwrap()),
            Some(E::new(ready, ready, false, 0, 0, 0, 0, 1).unwrap()),
        ] {
            assert_eq!(
                graph_state(DeviceCostGraphCaptureCapability::Supported, false, proof),
                Err(ActualWaveEvidenceUnknown::GraphPath)
            );
        }
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                true,
                Some(exact)
            ),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
    }

    #[test]
    fn only_actual_sealed_direct_replay_maps_to_warm() {
        use ferrum_interfaces::vnext::{
            DeviceCostGraphConfiguration as C, DeviceCostGraphStreamState as S,
            DeviceSubmissionGraphEvidence as E,
        };
        let state = S::new(C::OnDemand, 1, 1, 0).unwrap();
        let proof = E::new(state, state, false, 0, 0, 0, 0, 1).unwrap();
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                true,
                Some(proof)
            ),
            Ok(ActualWaveGraphState::Warm)
        );
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Unsupported,
                true,
                Some(proof)
            ),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                false,
                Some(proof)
            ),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
        let prepared = E::new(state, state, true, 1, 1, 0, 1, 1).unwrap();
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                true,
                Some(prepared)
            ),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
    }

    #[test]
    fn eager_path_alone_does_not_prove_graph_capture_absent() {
        assert_eq!(
            graph_state(DeviceCostGraphCaptureCapability::Unknown, false, None),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
        assert_eq!(
            graph_state(DeviceCostGraphCaptureCapability::Unknown, true, None),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
    }

    #[test]
    fn declared_no_capture_requires_consistent_actual_eager_path() {
        assert_eq!(
            graph_state(DeviceCostGraphCaptureCapability::Unsupported, false, None),
            Ok(ActualWaveGraphState::Disabled)
        );
        assert_eq!(
            graph_state(DeviceCostGraphCaptureCapability::Unsupported, true, None),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
    }

    #[test]
    fn graph_capable_runtime_needs_actual_same_wave_unconfigured_evidence() {
        use ferrum_interfaces::vnext::{
            DeviceCostGraphConfiguration as C, DeviceCostGraphStreamState as S,
            DeviceSubmissionGraphEvidence as E,
        };
        let empty = S::new(C::Unconfigured, 0, 0, 0).unwrap();
        let eager = E::new(empty, empty, false, 5, 0, 0, 0, 0).unwrap();
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                false,
                Some(eager)
            ),
            Ok(ActualWaveGraphState::Disabled)
        );
        assert_eq!(
            graph_state(DeviceCostGraphCaptureCapability::Supported, false, None),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                true,
                Some(eager)
            ),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
        for configured in [C::StartupPreparing, C::StartupReady, C::OnDemand] {
            let changed = E::new(
                empty,
                S::new(configured, 0, 0, 0).unwrap(),
                false,
                5,
                0,
                0,
                0,
                0,
            )
            .unwrap();
            assert_eq!(
                graph_state(
                    DeviceCostGraphCaptureCapability::Supported,
                    false,
                    Some(changed)
                ),
                Err(ActualWaveEvidenceUnknown::GraphPath)
            );
        }
        let captured = E::new(empty, empty, false, 5, 1, 0, 1, 0).unwrap();
        assert_eq!(
            graph_state(
                DeviceCostGraphCaptureCapability::Supported,
                false,
                Some(captured)
            ),
            Err(ActualWaveEvidenceUnknown::GraphPath)
        );
    }
}
