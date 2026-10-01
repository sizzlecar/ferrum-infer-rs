//! Frozen current-wave facts. This type cannot reach a device or live request.
use super::*;

pub(in crate::executor::vnext_executor) struct ProviderIdentityTable {
    rows: Box<[OwnedProviderIdentity]>,
    retained_bytes: usize,
}
struct OwnedProviderIdentity {
    provider_id: String,
    implementation_fingerprint: String,
    operation_fingerprint: String,
}
impl ProviderIdentityTable {
    fn capture<R: DeviceRuntime>(executor: &VNextModelExecutor<R>) -> Self {
        let rows: Box<[_]> = executor
            .providers
            .providers()
            .iter()
            .map(|provider| {
                let d = provider.descriptor();
                OwnedProviderIdentity {
                    provider_id: d.provider_id().as_str().to_owned(),
                    implementation_fingerprint: d.provider_implementation_fingerprint().to_owned(),
                    operation_fingerprint: d.operation_fingerprint().to_owned(),
                }
            })
            .collect();
        let retained_bytes = rows
            .len()
            .checked_mul(std::mem::size_of::<OwnedProviderIdentity>())
            .and_then(|initial| {
                rows.iter().try_fold(initial, |total, row| {
                    total
                        .checked_add(row.provider_id.capacity())?
                        .checked_add(row.implementation_fingerprint.capacity())?
                        .checked_add(row.operation_fingerprint.capacity())
                })
            })
            .unwrap_or(usize::MAX);
        Self {
            rows,
            retained_bytes,
        }
    }
    fn get(&self, index: u32) -> Option<CostProviderIdentity<'_>> {
        self.rows
            .get(index as usize)
            .map(|row| CostProviderIdentity {
                provider_id: &row.provider_id,
                implementation_fingerprint: &row.implementation_fingerprint,
                operation_fingerprint: &row.operation_fingerprint,
            })
    }
}

struct FrozenActualProjection {
    providers: Arc<ProviderIdentityTable>,
    attribution: DeviceSubmissionAttribution,
    rows: Vec<ActualWaveRow>,
    canonical_rows: Vec<CanonicalCostRow>,
    kind: ActualWaveKind,
    product: CostProductOutput,
    graph_capability: DeviceCostGraphCaptureCapability,
    readback: CoreReadbackRoute,
    retries: u32,
    recurrent_state_bytes: u64,
    structured_capture: bool,
    numeric_observation: DeviceCostObservationDemand,
    bounds: PendingWaveBounds,
}
impl PendingActualWaveProjection for FrozenActualProjection {
    fn rows(&self) -> &[ActualWaveRow] {
        &self.rows
    }
    fn bounds(&self) -> PendingWaveBounds {
        self.bounds
    }
    fn project(&self) -> std::result::Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
        // Resolution cannot erase the original exact execution ledger. A bad
        // passive sidecar remains unavailable and never reuses capture values.
        let resolved = self
            .numeric_observation
            .is_required()
            .then(|| self.attribution.clone().resolve_observation());
        let statistics = matches!(resolved, Some(Ok(_)));
        let attribution = resolved
            .as_ref()
            .and_then(|result| result.as_ref().ok())
            .unwrap_or(&self.attribution);
        let route::ObservedRoute {
            mut canonical,
            graph,
        } = route::actual_route_projection(
            Some(attribution),
            |index| self.providers.get(index),
            self.graph_capability,
            self.product,
            self.retries,
            self.structured_capture && statistics,
            statistics,
        )?;
        canonical
            .core_readback_route(self.readback)
            .map_err(|_| ActualWaveEvidenceUnknown::OutputPolicy)?;
        for row in &self.canonical_rows {
            canonical
                .row(*row)
                .map_err(|_| ActualWaveEvidenceUnknown::OutputPolicy)?;
        }
        let path = if self.retries > 0 {
            ActualWavePath::UnsupportedFallback
        } else {
            ActualWavePath::PlanRuntime
        };
        let (exact, statistical_evidence) = if statistics {
            let value = canonical
                .finish_with_captured_structure(
                    self.kind,
                    path,
                    graph,
                    ActualWaveRowOrder::Ordered,
                    self.recurrent_state_bytes,
                )
                .map_err(|_| ActualWaveEvidenceUnknown::ProviderPath)?;
            (value.exact, value.statistical.ok())
        } else {
            (
                canonical
                    .finish(
                        self.kind,
                        path,
                        graph,
                        ActualWaveRowOrder::Ordered,
                        self.recurrent_state_bytes,
                    )
                    .map_err(|_| ActualWaveEvidenceUnknown::ProviderPath)?,
                None,
            )
        };
        Ok(ActualWaveShape {
            kind: self.kind,
            path,
            graph,
            row_order: exact.row_order,
            provider_signature: exact.provider_signature,
            output_policy_signature: exact.output_policy_signature,
            numeric_features: exact.numeric_features,
            host_content_features: exact.host_content_features,
            row_multiset_features: exact.row_multiset_features,
            statistical_evidence,
            rows: self.rows.clone(),
            recurrent_state_bytes: self.recurrent_state_bytes,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        })
    }
}

pub(in crate::executor::vnext_executor) fn pending_actual_shape<R: DeviceRuntime>(
    executor: &VNextModelExecutor<R>,
    context: &PlanRuntimeCostObservationContext<'_>,
    participants: &[VNextExecutionParticipant<'_, R>],
    kind: VNextExecutionWaveKind,
    output_mode: VNextProductOutputMode,
    token_masks: &[VNextProductTokenMaskSubmissionPlan],
    attribution: Option<&BoundDeviceSubmissionAttribution>,
    retries: u32,
    core_readback_route: CoreReadbackRoute,
    numeric_observation: DeviceCostObservationDemand,
) -> std::result::Result<PendingActualWave, ActualWaveEvidenceUnknown> {
    if participants.is_empty()
        || participants.len() > 1024
        || token_masks.len() != participants.len()
    {
        return Err(ActualWaveEvidenceUnknown::ShapeOverflow);
    }
    if executor.checkpoint_capture.is_some() || executor.diagnostic_fault.is_some() {
        return Err(ActualWaveEvidenceUnknown::OutputPolicy);
    }
    if executor
        .sequence_state_memory
        .other_token_scaled_bytes_per_token
        != 0
    {
        return Err(ActualWaveEvidenceUnknown::RecurrentState);
    }
    let attribution = attribution
        .ok_or(ActualWaveEvidenceUnknown::GraphPath)?
        .device()
        .clone();
    let providers = executor
        .cost_observation_providers
        .get_or_init(|| Arc::new(ProviderIdentityTable::capture(executor)))
        .clone();
    let recurrent_state_bytes = executor
        .sequence_state_memory
        .fixed_bytes_per_sequence
        .checked_mul(participants.len() as u64)
        .ok_or(ActualWaveEvidenceUnknown::ShapeOverflow)?;
    let mut rows = Vec::new();
    let mut canonical_rows = Vec::new();
    rows.try_reserve_exact(participants.len())
        .map_err(|_| ActualWaveEvidenceUnknown::Capacity)?;
    canonical_rows
        .try_reserve_exact(participants.len())
        .map_err(|_| ActualWaveEvidenceUnknown::Capacity)?;
    for (participant, mask) in participants.iter().zip(token_masks) {
        let host = context
            .participant(participant.sequence.request_id())
            .ok_or(ActualWaveEvidenceUnknown::ParticipantCorrelation)?;
        let work = prepared_participant_work(participant)?;
        let output = match participant.output_role {
            VNextParticipantOutputRole::Decode(policy) => {
                let repetition = product_repetition_input(Some(policy), output_mode);
                CostRowOutput::Decode {
                    requires_full_logits: policy.requires_full_logits(),
                    repetition_tokens: repetition.token_ids.len() as u64,
                    repetition_penalty_bits: repetition.penalty.to_bits(),
                }
            }
            role => CostRowOutput::Prefill {
                final_logits: matches!(role, VNextParticipantOutputRole::FinalPrefill),
            },
        };
        canonical_rows.push(CanonicalCostRow {
            work,
            host_policy_signature: host
                .output_policy_signature
                .ok_or(ActualWaveEvidenceUnknown::OutputPolicy)?,
            host_features: host.host_features,
            mask_upload_required: mask.upload_required,
            output,
        });
        rows.push(ActualWaveRow {
            request_id: host.request_id.clone(),
            owner_incarnation: host.owner_incarnation,
            work_generation: host.work_generation,
            input_index: host.input_index,
            work,
        });
    }
    let local_bytes = std::mem::size_of::<FrozenActualProjection>()
        .checked_add(providers.retained_bytes)
        .and_then(|n| {
            n.checked_add(
                rows.capacity()
                    .checked_mul(std::mem::size_of::<ActualWaveRow>())?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                canonical_rows
                    .capacity()
                    .checked_mul(std::mem::size_of::<CanonicalCostRow>())?,
            )
        })
        .ok_or(ActualWaveEvidenceUnknown::Capacity)?;
    let retained_bytes = local_bytes
        .checked_add(
            attribution
                .call_owned_payload_bytes()
                .ok_or(ActualWaveEvidenceUnknown::Capacity)?,
        )
        .ok_or(ActualWaveEvidenceUnknown::Capacity)?;
    // The device bound includes simultaneous raw + projected command storage.
    // Canonical host/numeric projections have a bounded number of rows per input.
    let row_projection = rows
        .len()
        .checked_mul(
            std::mem::size_of::<ActualWaveRow>()
                + std::mem::size_of::<CanonicalCostRow>()
                + std::mem::size_of::<CostRowNumericFeatures>()
                + std::mem::size_of::<HostRowStaticCostFeaturesV2>()
                + std::mem::size_of::<StructuredHostRowV1>(),
        )
        .ok_or(ActualWaveEvidenceUnknown::Capacity)?;
    let maximum_resolved_bytes = local_bytes
        .checked_add(row_projection)
        .and_then(|n| n.checked_add(std::mem::size_of::<ActualWaveShape>()))
        .and_then(|n| n.checked_add(std::mem::size_of::<UnsettledStructuredWaveEvidenceV1>()))
        .and_then(|n| n.checked_add(attribution.maximum_working_bytes()?))
        .ok_or(ActualWaveEvidenceUnknown::Capacity)?;
    let bounds = PendingWaveBounds {
        retained_bytes,
        maximum_resolved_bytes,
        retained_rows: rows.capacity(),
    };
    PendingActualWave::new(Arc::new(FrozenActualProjection {
        providers,
        attribution,
        rows,
        canonical_rows,
        kind: match kind {
            VNextExecutionWaveKind::Decode => ActualWaveKind::Decode,
            VNextExecutionWaveKind::Prefill => ActualWaveKind::Prefill,
            VNextExecutionWaveKind::Mixed => ActualWaveKind::Mixed,
        },
        product: match output_mode {
            VNextProductOutputMode::FullLogits => CostProductOutput::FullLogits,
            VNextProductOutputMode::GreedyToken => CostProductOutput::GreedyToken,
        },
        graph_capability: executor.runtime.cost_graph_capture_capability(),
        readback: core_readback_route,
        retries,
        recurrent_state_bytes,
        structured_capture: context.structured_capture_enabled(),
        numeric_observation,
        bounds,
    }))
}
