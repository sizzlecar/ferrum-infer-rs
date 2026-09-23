//! Production future-route bridge. Reads use the same registry/session fences
//! as resource planning; queries never construct a Step or encode commands.
use super::*;
use ferrum_interfaces::execution_cost::*;
use ferrum_interfaces::model_executor::ExecutorResourcePlanningRequest;
use std::num::NonZeroU64;

mod masks;
mod uploads;
pub(super) use uploads::{
    product_readback_binding, product_token_upload_layout, product_upload_layouts,
};
use ExecutionCostRouteUnknown as U;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn capture_future_cost_route(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        limits: ResourcePlanningLimits,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<ExecutionCostRouteView> {
        if let Err(reason) = self.future_cost_policy() {
            return ExecutionCostRouteAvailability::Unknown(reason);
        }
        let captured = resource_planning::with_registry_sequences(
            &self.sequences,
            requests,
            limits,
            budget,
            |sequences, sessions, budget| {
                let mut frontiers = Vec::new();
                if frontiers.try_reserve_exact(sequences.len()).is_err() {
                    return ExecutionCostRouteAvailability::Unknown(U::Capacity);
                }
                for (sequence, request) in sequences.iter().zip(requests) {
                    if !budget.has_budget() {
                        return ExecutionCostRouteAvailability::Unknown(U::BudgetExhausted);
                    }
                    let frontier = if request.cache_id.is_some() {
                        let Some(tokens) = sequence.tokens.try_lock() else {
                            return ExecutionCostRouteAvailability::Unknown(U::Resource(
                                ResourcePlanningUnknown::BusyOrUnavailable,
                            ));
                        };
                        tokens.len()
                    } else {
                        sequence.prefill_tokens_processed.load(Ordering::Acquire)
                    };
                    let Ok(frontier) = u64::try_from(frontier) else {
                        return ExecutionCostRouteAvailability::Unknown(U::InvalidInput);
                    };
                    frontiers.push(frontier);
                }
                // Keep the product ledger locked across the lane/arena capture;
                // neither a successful upload nor failure invalidation can
                // straddle this numerical snapshot. Every nested read is try-only.
                let Some(ledger) = self.product_token_mask_residency.try_lock() else {
                    return ExecutionCostRouteAvailability::Unknown(U::Resource(
                        ResourcePlanningUnknown::BusyOrUnavailable,
                    ));
                };
                let token_masks =
                    match masks::capture(&ledger, self.io.token_mask_residency_eligible, budget) {
                        Ok(snapshot) => snapshot,
                        Err(reason) => return ExecutionCostRouteAvailability::Unknown(reason),
                    };
                let view = self
                    .plan_resources
                    .execution_cost_route_view(sessions, &frontiers, &self.lane, limits, budget);
                drop(ledger);
                match view {
                    ExecutionCostRouteAvailability::Known(view) => {
                        if let Err(reason) = self.future_cost_graph_policy(&view) {
                            return ExecutionCostRouteAvailability::Unknown(reason);
                        }
                        ExecutionCostRouteAvailability::Known(
                            view.with_token_mask_residency(token_masks),
                        )
                    }
                    ExecutionCostRouteAvailability::Unknown(reason) => {
                        ExecutionCostRouteAvailability::Unknown(reason)
                    }
                }
            },
        );
        captured
            .unwrap_or_else(|reason| ExecutionCostRouteAvailability::Unknown(U::Resource(reason)))
    }

    pub(super) fn future_cost_policy(&self) -> std::result::Result<(), U> {
        if self.checkpoint_capture.is_some()
            || self.diagnostic_fault.is_some()
            || self.device_timing_mode() != DeviceTimingMode::Off
            || self
                .resolved_plan
                .execution_plan()
                .payload()
                .memory()
                .reusable_execution()
                .and_then(|plan| plan.program_policy())
                .is_some()
            || self.runtime.cost_graph_capture_capability()
                == DeviceCostGraphCaptureCapability::Unknown
        {
            return Err(U::ExecutionPolicy);
        }
        if self
            .sequence_state_memory
            .other_token_scaled_bytes_per_token
            != 0
        {
            return Err(U::Unsupported);
        }
        if self.runtime.cost_core_execution_capabilities().is_none() {
            return Err(U::Unsupported);
        }
        Ok(())
    }

    fn future_cost_graph_policy(
        &self,
        view: &ExecutionCostRouteView,
    ) -> std::result::Result<(), U> {
        match self.runtime.cost_graph_capture_capability() {
            DeviceCostGraphCaptureCapability::Unsupported => Ok(()),
            DeviceCostGraphCaptureCapability::Supported
                if view
                    .graph_stream_state()
                    .is_some_and(|state| state.is_unconfigured_empty()) =>
            {
                Ok(())
            }
            _ => Err(U::ExecutionPolicy),
        }
    }

    pub(super) fn project_future_cost_route(
        &self,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        query: &FutureWaveCostQuery<'_>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<ExecutionCostRouteProjection> {
        match self.project_future_cost_route_inner(view, state, query, budget) {
            Ok(projection) => ExecutionCostRouteAvailability::Known(projection),
            Err(reason) => ExecutionCostRouteAvailability::Unknown(reason),
        }
    }

    fn project_future_cost_route_inner(
        &self,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        query: &FutureWaveCostQuery<'_>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> std::result::Result<ExecutionCostRouteProjection, U> {
        self.future_cost_policy()?;
        self.future_cost_graph_policy(view)?;
        if view.lane_id() != self.lane.id() {
            return Err(U::StaleView);
        }
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        if query.rows.is_empty() || query.rows.len() > MAX_COST_ROWS {
            return Err(U::Capacity);
        }
        let kind = match query.kind {
            ActualWaveKind::Prefill => VNextExecutionWaveKind::Prefill,
            ActualWaveKind::Decode => VNextExecutionWaveKind::Decode,
            ActualWaveKind::Mixed => VNextExecutionWaveKind::Mixed,
            _ => return Err(U::InvalidInput),
        };
        let mut roles = Vec::new();
        let mut work = Vec::new();
        let mut indices = Vec::new();
        roles
            .try_reserve_exact(query.rows.len())
            .map_err(|_| U::Capacity)?;
        work.try_reserve_exact(query.rows.len())
            .map_err(|_| U::Capacity)?;
        indices
            .try_reserve_exact(query.rows.len())
            .map_err(|_| U::Capacity)?;
        for row in query.rows {
            if !budget.has_budget() {
                return Err(U::BudgetExhausted);
            }
            let (offset, count, full_input_tokens, role) = match (row.work, row.output) {
                (
                    ActualRowWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    },
                    FutureCostOutput::Prefill { final_logits },
                ) if count > 0
                    && offset.checked_add(count).is_some_and(|end| {
                        end <= total_prompt_tokens && final_logits == (end == total_prompt_tokens)
                    }) =>
                {
                    (
                        u64::from(offset),
                        u64::from(count),
                        u64::from(total_prompt_tokens),
                        if final_logits {
                            VNextParticipantOutputRole::FinalPrefill
                        } else {
                            VNextParticipantOutputRole::IntermediatePrefill
                        },
                    )
                }
                (ActualRowWork::Decode { kv_tokens }, FutureCostOutput::Decode { policy }) => {
                    match policy {
                        LogitsReturnPolicy::GreedyArgmax {
                            repetition_penalty: Some(_),
                            ..
                        } => return Err(U::OutputBranch),
                        _ => {}
                    }
                    let end = u64::from(kv_tokens).checked_add(1).ok_or(U::InvalidInput)?;
                    (
                        u64::from(kv_tokens),
                        1,
                        end,
                        VNextParticipantOutputRole::Decode(policy.clone()),
                    )
                }
                _ => return Err(U::InvalidInput),
            };
            roles.push(role);
            work.push(OperationCostWorkRow {
                offset,
                count: NonZeroU64::new(count).ok_or(U::InvalidInput)?,
                full_input_tokens: NonZeroU64::new(full_input_tokens).ok_or(U::InvalidInput)?,
            });
            indices.push(row.participant_index);
        }
        let output_mode = product_output_mode_for_roles(kind, &roles);
        let mut mask_contents = Vec::new();
        mask_contents
            .try_reserve_exact(roles.len())
            .map_err(|_| U::Capacity)?;
        for role in &roles {
            if !budget.has_budget() {
                return Err(U::BudgetExhausted);
            }
            mask_contents.push(masks::requested(
                role,
                output_mode,
                self.io.output_elements,
            )?);
        }
        let product = match output_mode {
            VNextProductOutputMode::FullLogits => CostProductOutput::FullLogits,
            VNextProductOutputMode::GreedyToken => CostProductOutput::GreedyToken,
        };
        let mut canonical = CanonicalWaveCostBuilder::new(0, product);
        let uploads = uploads::input_uploads(&self.io, &work)?;
        let readbacks = uploads::readbacks(&self.io, work.len(), output_mode)?;
        let total_tokens = work
            .iter()
            .try_fold(0_u64, |sum, row| sum.checked_add(row.count.get()))
            .ok_or(U::Capacity)?;
        let core = EagerCoreWaveCostQuery {
            rows: &work,
            participant_indices: &indices,
            uploads: &uploads,
            readbacks: &readbacks,
            attempt_staged_readbacks: output_mode == VNextProductOutputMode::GreedyToken,
            reusable_bucket: self.reusable_bucket_for_shape(
                kind,
                work.len() as u32,
                total_tokens,
                0,
            ),
            token_mask_input: Some(EagerCoreTokenMaskInput {
                node_id: &self.io.token_mask_input_node_id,
                input_ordinal: self.io.token_mask_input_ordinal,
                vocabulary_size: u64::try_from(self.io.output_elements).map_err(|_| U::Capacity)?,
                contents: &mask_contents,
            }),
        };
        let next = append_complete_eager_cost_route(
            self.runtime.as_ref(),
            &self.providers,
            &self.resolved_plan,
            &self.plan_resources,
            view,
            state,
            &core,
            &mut canonical,
            budget,
        )?;
        let mask_uploads = next.last_token_mask_uploads().ok_or(U::OutputBranch)?;
        if mask_uploads.len() != query.rows.len() {
            return Err(U::InvalidInput);
        }
        for ((row, role), &mask_upload_required) in query.rows.iter().zip(&roles).zip(mask_uploads)
        {
            if !budget.has_budget() {
                return Err(U::BudgetExhausted);
            }
            let output = match role {
                VNextParticipantOutputRole::IntermediatePrefill => CostRowOutput::Prefill {
                    final_logits: false,
                },
                VNextParticipantOutputRole::FinalPrefill => {
                    CostRowOutput::Prefill { final_logits: true }
                }
                VNextParticipantOutputRole::Decode(policy) => {
                    let repetition = product_repetition_input(Some(policy), output_mode);
                    CostRowOutput::Decode {
                        requires_full_logits: policy.requires_full_logits(),
                        repetition_tokens: repetition.token_ids.len() as u64,
                        repetition_penalty_bits: repetition.penalty.to_bits(),
                    }
                }
            };
            canonical
                .row(CanonicalCostRow {
                    work: row.work,
                    host_policy_signature: row.host_policy_signature,
                    host_features: row.host_features,
                    mask_upload_required,
                    output,
                })
                .map_err(|_| U::OutputBranch)?;
        }
        let recurrent = self
            .sequence_state_memory
            .fixed_bytes_per_sequence
            .checked_mul(query.rows.len() as u64)
            .ok_or(U::Capacity)?;
        let shape = canonical
            .finish(
                query.kind,
                ActualWavePath::PlanRuntime,
                ActualWaveGraphState::Disabled,
                ActualWaveRowOrder::Ordered,
                recurrent,
            )
            .map_err(|_| U::InvalidInput)?;
        Ok(ExecutionCostRouteProjection { shape, state: next })
    }
}
