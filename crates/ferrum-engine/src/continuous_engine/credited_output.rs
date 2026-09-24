//! Output admission and the engine's side of the transport ownership boundary.
//!
//! Physical request/KV authority stays in the existing executor path. A batch
//! may run only after every token-producing credited participant owns a grant.
//! No transport wait is performed by this module.

use super::*;
use ferrum_interfaces::output_credit::{OutputAccountLimits, OutputCreditPool, OutputPoolLimits};
use ferrum_interfaces::output_flow::{
    BoundedOutputError, CreditedOutputSession, OutputProjectionContract, RequestOutputBudget,
    RequestOutputPlan,
};
use ferrum_scheduler::implementations::continuous::ExecutionReadinessWake;
use std::num::NonZeroUsize;

mod completion;
mod decode;
pub(super) use decode::CreditedDecodePolicy;

use super::output_flow_runtime::{
    spawn_output_flow_runtime, OutputDelta, OutputFlowPort, OutputFlowRuntimeOptions,
    OutputReadiness, OutputReadinessState, ReadyOutputGrant,
};

pub(super) struct CreditedSequenceOutput {
    pub(super) port: OutputFlowPort,
    pub(super) decoder: CreditedDecodePolicy,
    pub(super) grant: Option<ReadyOutputGrant>,
    pub(super) tokens_before_grant: usize,
    pub(super) accepted_ordinal: u64,
    pub(super) deferred: Option<ExecutionReadinessWake>,
    pub(super) failure: Option<BoundedOutputError>,
}

impl std::fmt::Debug for CreditedSequenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreditedSequenceOutput")
            .field("grant", &self.grant.is_some())
            .field("tokens_before_grant", &self.tokens_before_grant)
            .field("accepted_ordinal", &self.accepted_ordinal)
            .field("deferred", &self.deferred.is_some())
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl Drop for CreditedSequenceOutput {
    fn drop(&mut self) {
        // An unresolved grant is abandoned, never relabelled NotSubmitted.
        // Its command is released before the port's projection lifetime.
        drop(self.grant.take());
        // A ticket only enables a reprobe; cancelling it cannot manufacture
        // executor capacity. This owner cannot leave a waiter behind on exit.
        if let Some(wake) = self.deferred.take() {
            wake.cancel();
        }
    }
}

impl ContinuousBatchEngine {
    pub(super) async fn submit_credited_stream(
        &self,
        mut request: InferenceRequest,
        context: ferrum_interfaces::InferenceRequestContext,
        contract: Arc<OutputProjectionContract>,
    ) -> Result<CreditedOutputSession> {
        if self.inner.shutdown_started.load(Ordering::Acquire) {
            return Err(FerrumError::cancelled("inference engine is shutting down"));
        }
        if self.inner.model_executor.execution_resource_authority()
            != ExecutionResourceAuthority::PlanRuntime
            || self.inner.spec_config.is_some()
        {
            return Err(FerrumError::unsupported(
                "credited output currently requires a non-speculative PlanRuntime executor",
            ));
        }
        let slo = context.resolve_slo(&self.inner.config.scheduler.slo)?;
        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        clamp_default_max_tokens_to_context(
            &mut request,
            input_tokens.len(),
            &self.inner.config,
            &self.inner.runtime_config,
            self.inner.model_executor.kv_capacity(),
        );
        validate_request_context_budget(
            &request,
            input_tokens.len(),
            &self.inner.config,
            &self.inner.runtime_config,
            self.inner.model_executor.kv_capacity(),
        )?;
        let plan = RequestOutputPlan::derive(
            contract,
            self.inner.tokenizer.as_ref(),
            &request,
            input_tokens.len(),
        )
        .map_err(|error| FerrumError::invalid_request(error.to_string()))?;
        let limits = OutputAccountLimits::from_slo(
            &self.inner.config.scheduler.slo.output,
            NonZeroUsize::new(plan.terminal_credit().events)
                .expect("a valid codec has terminal events"),
        )
        .map_err(|error| FerrumError::config(error.to_string()))?;
        tracing::debug!(
            request_id = %request.id,
            effective_max_tokens = plan.effective_max_tokens(),
            max_token_bytes = plan.max_token_bytes(),
            max_decoded_bytes = plan.max_decoded_bytes(),
            required_wire_bytes = plan.lifetime_wire_bytes(),
            required_projection_bytes = plan.retained_projection_bytes(),
            required_terminal_bytes = plan.terminal_credit().bytes,
            minimum_events = plan.minimum_event_capacity(),
            configured_bytes = limits.maximum.bytes,
            configured_projection_bytes = limits.maximum.projection_bytes,
            configured_terminal_bytes = limits.terminal.bytes,
            configured_events = limits.maximum.events,
            "derived credited output capacity before admission"
        );
        let decoder = CreditedDecodePolicy::from_plan(&plan);
        let budget = RequestOutputBudget::open(self.inner.output_credit_pool()?, limits, plan)
            .map_err(|error| FerrumError::resource_exhausted(error.to_string()))?;
        request.metadata.insert(
            PROMPT_TOKENS_METADATA_KEY.to_string(),
            serde_json::Value::from(input_tokens.len() as u64),
        );
        let request_id = request.id.clone();
        let completion_plan = budget.plan().completion_plan();
        let mut sequence = SequenceState::try_new_with_completion_plan(
            request.clone(),
            input_tokens,
            Some(self.inner.tokenizer.clone()),
            Some(self.inner.model_executor.info().vocab_size),
            None,
            Some(&completion_plan),
        )?;
        sequence.bind_credited_execution_evidence(budget.plan().evidence_plan())?;
        // Avoid geometric growth beyond the declared token-history envelope.
        sequence.generated_tokens = Vec::with_capacity(request.sampling_params.max_tokens);
        let (port, session) = spawn_output_flow_runtime(
            budget,
            self.inner.work_notify.clone(),
            OutputFlowRuntimeOptions::from_config(&self.inner.config.scheduler.slo.output),
        );
        sequence.slo = slo;
        sequence.credited_output = Some(CreditedSequenceOutput {
            port,
            decoder,
            grant: None,
            tokens_before_grant: 0,
            accepted_ordinal: 0,
            deferred: None,
            failure: None,
        });
        sequence.request_slot = Some(RequestSlotLease::open(&self.inner, request_id.clone()));
        self.inner.initialize_sequence_cost(&mut sequence);
        {
            let _iteration = self.inner.iteration_lock.lock().await;
            if self.inner.shutdown_started.load(Ordering::Acquire) {
                if let Some(slot) = sequence.request_slot.take() {
                    slot.reject(&self.inner, "inference engine is shutting down".to_owned());
                }
                return Err(FerrumError::cancelled("inference engine is shutting down"));
            }
            {
                let mut sequences = self.inner.sequences.write();
                if sequences.contains_key(&request_id) {
                    let error = FerrumError::already_exists(format!(
                        "request {request_id} is already active"
                    ));
                    if let Some(slot) = sequence.request_slot.take() {
                        slot.reject(&self.inner, error.to_string());
                    }
                    return Err(error);
                }
                sequences.insert(request_id.clone(), sequence);
            }
            if let Err(error) = self.inner.scheduler.submit(request).await {
                let mut sequence = self
                    .inner
                    .sequences
                    .write()
                    .remove(&request_id)
                    .expect("published request remains present until submit finishes");
                if let Some(slot) = sequence.request_slot.take() {
                    slot.reject(&self.inner, error.to_string());
                }
                return Err(error);
            }
            let mut sequences = self.inner.sequences.write();
            let sequence = sequences
                .get_mut(&request_id)
                .expect("submitted sequence remains published");
            self.inner.initialize_sequence_prefill_reference(sequence);
            self.inner.initialize_sequence_time_admission(sequence);
            sequence
                .request_slot
                .as_mut()
                .expect("submitted request retains its slot")
                .admit(&self.inner);
        }
        self.ensure_bg_loop();
        self.inner.work_notify.notify_one();
        Ok(session)
    }
}

impl EngineInner {
    fn output_credit_pool(&self) -> Result<&OutputCreditPool> {
        self.output_credit_pool
            .get_or_init(|| {
                let slots = NonZeroUsize::new(self.config.scheduler.max_running_requests)
                    .ok_or_else(|| {
                        "output admission needs a positive request capacity".to_owned()
                    })?;
                let limits = OutputPoolLimits::from_slo(&self.config.scheduler.slo.output, slots)
                    .map_err(|error| error.to_string())?;
                OutputCreditPool::new(limits).map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| FerrumError::config(error.clone()))
    }

    /// Called before asking the scheduler for another plan. Actor notifications
    /// retain a work-notify permit; a state change before this check is not lost.
    pub(super) fn refresh_credited_output_readiness(&self) {
        for sequence in self.sequences.write().values_mut() {
            let Some(output) = sequence.credited_output.as_mut() else {
                continue;
            };
            if output.deferred.is_some()
                && matches!(output.port.readiness(), OutputReadinessState::Ready)
            {
                if let Some(wake) = output.deferred.take() {
                    wake.mark_ready();
                }
            }
        }
    }

    /// The selected physical cohort is never silently narrowed. On output
    /// pressure return all grants and let the scheduler build a fresh cohort.
    pub(super) fn reserve_batch_output(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> Result<bool> {
        let mut blocked = Vec::new();
        let mut sequences = self.sequences.write();
        // Validate before acquiring anything, so a broken caller cannot leave
        // an earlier participant with a new grant after this error.
        if batch.requests.iter().any(|scheduled| {
            sequences
                .get(&scheduled.request.id)
                .and_then(|sequence| sequence.credited_output.as_ref())
                .is_some_and(|output| output.grant.is_some())
        }) {
            return Err(FerrumError::internal(
                "credited request already owns a wave grant",
            ));
        }
        for scheduled in &batch.requests {
            let Some(sequence) = sequences.get_mut(&scheduled.request.id) else {
                continue;
            };
            let final_prefill = sequence.prefill_complete
                || scheduled
                    .tokens_to_process
                    .unwrap_or(sequence.prefill_context_len())
                    >= sequence
                        .prefill_context_len()
                        .saturating_sub(scheduled.tokens_processed);
            if !final_prefill {
                continue;
            }
            let Some(output) = sequence.credited_output.as_mut() else {
                continue;
            };
            match output.port.try_take() {
                OutputReadiness::Ready(grant) => {
                    output.tokens_before_grant = sequence.generated_tokens.len();
                    output.grant = Some(grant);
                }
                OutputReadiness::ProjectionBusy => {
                    blocked.push(scheduled.request.id.clone());
                }
                OutputReadiness::OutputBlocked(reason) => {
                    tracing::trace!(request_id = %scheduled.request.id, ?reason, "request awaits output credit");
                    blocked.push(scheduled.request.id.clone());
                }
                OutputReadiness::Closing(reason) => {
                    output.failure = Some(BoundedOutputError::new(reason.message()));
                    blocked.push(scheduled.request.id.clone());
                }
            }
        }
        if blocked.is_empty() {
            return Ok(true);
        }
        for scheduled in &batch.requests {
            if let Some(output) = sequences
                .get_mut(&scheduled.request.id)
                .and_then(|sequence| sequence.credited_output.as_mut())
            {
                if let Some(grant) = output.grant.take() {
                    grant.return_unsubmitted();
                }
            }
        }
        // Each request has its own readiness ticket: one recovered consumer
        // must not wake the other blocked frontiers as if they had credit.
        for request_id in blocked {
            let output = sequences
                .get_mut(&request_id)
                .unwrap()
                .credited_output
                .as_mut()
                .unwrap();
            match self
                .scheduler
                .defer_for_execution_readiness(std::slice::from_ref(&request_id))
            {
                Ok(receipt) => output.deferred = Some(receipt.into_wake()),
                Err(error) => {
                    for scheduled in &batch.requests {
                        if let Some(output) = sequences
                            .get_mut(&scheduled.request.id)
                            .and_then(|sequence| sequence.credited_output.as_mut())
                        {
                            if let Some(wake) = output.deferred.take() {
                                wake.cancel();
                            }
                        }
                    }
                    return Err(error);
                }
            }
        }
        Ok(false)
    }

    /// Non-final/narrowed prefill and typed NotSubmitted need no output. Give
    /// back the original reservoir, never reserve a replacement after work.
    pub(super) fn finish_batch_output(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
        completed: bool,
    ) {
        for scheduled in &batch.requests {
            let mut sequences = self.sequences.write();
            let Some(sequence) = sequences.get_mut(&scheduled.request.id) else {
                continue;
            };
            let Some(output) = sequence.credited_output.as_mut() else {
                continue;
            };
            let Some(grant) = output.grant.take() else {
                continue;
            };
            if completed && sequence.generated_tokens.len() == output.tokens_before_grant {
                grant.return_unsubmitted();
            } else if completed
                && output.tokens_before_grant.checked_add(1)
                    == Some(sequence.generated_tokens.len())
            {
                output.accepted_ordinal = grant.committed(OutputDelta {
                    text: String::new(),
                    token: sequence.generated_tokens.last().copied(),
                    generated_tokens: sequence.generated_tokens.len(),
                    created: chrono::Utc::now().timestamp().max(0) as u64,
                });
            } else {
                drop(grant);
                output.failure = Some(BoundedOutputError::new(
                    "credited output lost a committed wave frontier",
                ));
                output.port.cancel();
            }
        }
    }
}
