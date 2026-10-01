//! The transport future owns the receiver. Pending requests may reserve real
//! physical resources, but cannot execute a model wave or enter completion
//! recovery until a fresh finite witness crosses this acceptance boundary.
use super::*;
use ferrum_types::errors::SloTimeAdmissionRejection;
use tokio::sync::oneshot;

#[derive(Debug)]
pub(super) struct StrictAcceptance {
    sender: Option<oneshot::Sender<Result<()>>>,
    expires_at: Instant,
    rejection: Option<FerrumError>,
}

#[cfg(test)]
mod tests;

impl EngineInner {
    pub(in crate::continuous_engine::inner::slo_controller) fn strict_candidate(
        &self,
        sequences: &HashMap<RequestId, SequenceState>,
        queue: &PlanningQueueSnapshot,
        model_version: u64,
    ) -> Option<RequestId> {
        if !self.requires_strict_acceptance() {
            return None;
        }
        let obligations = sequences
            .values()
            .filter(|sequence| {
                !sequence
                    .time_admission
                    .as_ref()
                    .is_some_and(SequenceTimeAdmission::before_acceptance)
            })
            .count()
            + 1;
        sequences
            .iter()
            .filter_map(|(id, sequence)| {
                let state = sequence.time_admission.as_ref()?;
                let pending = state.acceptance.as_ref()?;
                // These are the same current projection prerequisites checked
                // by capture. A prompt without them stays honestly unaccepted
                // and expiry-bounded; it cannot monopolize other candidates.
                if sampling::capability(sequence).is_err()
                    || sequence
                        .prefill_reference
                        .as_ref()
                        .is_none_or(|reference| reference.known().is_err())
                {
                    return None;
                }
                (state.deferred.as_ref().is_none_or(|wait| {
                    !wait.pending_for(queue, obligations, model_version, slo_clock_now())
                }) && pending.rejection.is_none()
                    && pending
                        .sender
                        .as_ref()
                        .is_some_and(|sender| !sender.is_closed()))
                .then_some((id, state.last_assessment.map(|last| last.at), state.ingress))
            })
            .min_by_key(|(_, last, ingress)| (*last, *ingress))
            .map(|(id, _, _)| id.clone())
    }

    pub(in crate::continuous_engine) fn requires_strict_acceptance(&self) -> bool {
        self.config.scheduler.slo.mode == ferrum_types::SloMode::Enforce
            && self.config.scheduler.slo.admission.time_policy
                == ferrum_types::SloTimeAdmissionPolicy::RequireSlo
    }

    pub(in crate::continuous_engine) fn install_strict_acceptance(
        &self,
        sequence: &mut SequenceState,
    ) -> Result<Option<oneshot::Receiver<Result<()>>>> {
        if !self.requires_strict_acceptance() {
            return Ok(None);
        }
        let state = sequence.time_admission.as_mut().ok_or_else(|| {
            FerrumError::invalid_request(
                "strict admission requires trusted original ingress timing",
            )
        })?;
        let expires_at = state
            .ingress
            .checked_add(self.config.scheduler.slo.admission.max_wait())
            .ok_or_else(|| FerrumError::invalid_request("strict admission expiry overflow"))?;
        let (sender, receiver) = oneshot::channel();
        state.acceptance = Some(StrictAcceptance {
            sender: Some(sender),
            expires_at,
            rejection: None,
        });
        Ok(Some(receiver))
    }

    /// Counts the actual unaccepted engine owners, including those physically
    /// prepared in Prefill. Their queue transition cannot release wait capacity.
    /// The common ingress/iteration mutex serializes this check with publication.
    pub(in crate::continuous_engine) fn check_strict_waiting_capacity(
        &self,
        incoming: &SequenceState,
    ) -> Result<()> {
        if !self.requires_strict_acceptance() {
            return Ok(());
        }
        let mut requests = 1usize;
        let mut tokens = incoming.input_tokens.len();
        let mut bytes = incoming.original_request.prompt.len();
        let overflow =
            || FerrumError::resource_exhausted("strict waiting prompt accounting overflow");
        let mut availability = Vec::new();
        let epochs = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)?
            .ok_or_else(|| {
                FerrumError::resource_exhausted("strict waiting capacity snapshot unavailable")
            })?;
        let queue = self
            .scheduler
            .planning_state(
                NonZeroUsize::new(4096).unwrap(),
                AdmissionWakeSnapshot::new(
                    AdmissionWakeEpochs::new(
                        epochs.coordinator_id,
                        epochs.release_epoch,
                        epochs.capacity_epoch,
                        0,
                    ),
                    &availability,
                ),
            )
            .map_err(|_| {
                FerrumError::resource_exhausted("strict waiting queue snapshot unavailable")
            })?;
        let waiting: std::collections::HashSet<_> = queue
            .requests()
            .iter()
            .filter(|row| row.queue == PlanningQueueKind::Waiting)
            .map(|row| &row.key.request_id)
            .collect();
        for (_, sequence) in self.sequences.read().iter().filter(|(id, sequence)| {
            waiting.contains(id)
                || sequence
                    .time_admission
                    .as_ref()
                    .is_some_and(SequenceTimeAdmission::before_acceptance)
        }) {
            requests = requests.checked_add(1).ok_or_else(overflow)?;
            tokens = tokens
                .checked_add(sequence.input_tokens.len())
                .ok_or_else(overflow)?;
            bytes = bytes
                .checked_add(sequence.original_request.prompt.len())
                .ok_or_else(overflow)?;
        }
        let policy = &self.config.scheduler.slo.admission;
        for (name, usage, limit) in [
            (
                "requests",
                requests,
                policy
                    .max_waiting_requests
                    .get()
                    .min(self.config.scheduler.max_waiting_requests),
            ),
            (
                "prompt tokens",
                tokens,
                policy.max_waiting_prompt_tokens.get(),
            ),
            (
                "prompt UTF-8 bytes",
                bytes,
                policy.max_waiting_prompt_bytes.get(),
            ),
        ] {
            if usage > limit {
                return Err(FerrumError::resource_exhausted(format!(
                    "strict waiting {name} capacity exhausted: proposed {usage}, limit {limit}"
                )));
            }
        }
        Ok(())
    }

    /// Runs under the iteration mutex. Only a successfully installed guarded
    /// first wave may call this. Acceptance is durable even if that wave later
    /// returns NotSubmitted: the accepted owner then retains completion service.
    pub(in crate::continuous_engine::inner::slo_controller) fn accept_strict_witness(
        &self,
        pending: &PendingTimeWitness,
        valid_until: Instant,
    ) -> bool {
        let mut sequences = self.sequences.write();
        let Some(sequence) = sequences.get_mut(&pending.request_id) else {
            return false;
        };
        let matches = Arc::ptr_eq(&sequence.stream_projection_identity, &pending.owner)
            && sequence
                .cost_frontier
                .is_some_and(|frontier| frontier.work_generation.get() == pending.work_generation)
            && sequence.generated_tokens.is_empty()
            && sequence.input_tokens.len() == pending.original_input_tokens
            && sequence.sampling_params.max_tokens == pending.maximum_output_tokens;
        let Some(state) = sequence.time_admission.as_mut() else {
            return false;
        };
        let Some(acceptance) = state.acceptance.as_mut() else {
            return true;
        };
        let now = slo_clock_now();
        if !matches
            || self.shutdown_started.load(Ordering::Acquire)
            || state.ingress != pending.ingress
            || now >= acceptance.expires_at
            || now > valid_until
            || acceptance.rejection.is_some()
            || !self.controller_model_current(pending.evidence.model_version)
        {
            return false;
        }
        let Some(sender) = acceptance.sender.take() else {
            return false;
        };
        if sender.send(Ok(())).is_err() {
            return false;
        }
        state.acceptance = None;
        sequence
            .request_slot
            .as_mut()
            .expect("pending owner retains its request slot")
            .admit(self);
        true
    }

    pub(super) fn record_strict_rejection(
        &self,
        target: &RequestWorkKey,
        reason: TimeAdmissionRejectReason,
    ) {
        let mut sequences = self.sequences.write();
        let Some(sequence) = sequences.get_mut(&target.request_id) else {
            return;
        };
        let Some(pending) = sequence
            .time_admission
            .as_mut()
            .and_then(|state| state.acceptance.as_mut())
        else {
            return;
        };
        pending.rejection = Some(FerrumError::SloTimeAdmissionRejected {
            reason: match reason {
                TimeAdmissionRejectReason::TargetTimeImpossible(_) => {
                    SloTimeAdmissionRejection::TargetTimeImpossible
                }
                TimeAdmissionRejectReason::StrictWaitExpired { .. } => {
                    SloTimeAdmissionRejection::WaitExpired
                }
            },
        });
        self.work_notify.notify_one();
    }

    /// Clears resources and scheduler ownership before returning an error to
    /// HTTP/run. An accepted request cannot enter this branch.
    pub(in crate::continuous_engine) async fn reject_strict_pending(
        &self,
        request_id: &RequestId,
        error: FerrumError,
    ) -> Result<bool> {
        let sender = {
            let mut sequences = self.sequences.write();
            let Some(sequence) = sequences.get_mut(request_id) else {
                return Ok(false);
            };
            let Some(pending) = sequence
                .time_admission
                .as_mut()
                .and_then(|state| state.acceptance.as_mut())
            else {
                return Ok(false);
            };
            let sender = pending.sender.take();
            if let Some(slot) = sequence.request_slot.take() {
                slot.reject(self, error.to_string());
            }
            sender
        };
        let result = self.cancel_abandoned_request(request_id).await;
        if let Some(sender) = sender {
            let _ = sender.send(Err(error));
        }
        result.map(|()| true)
    }

    pub(in crate::continuous_engine) async fn expire_strict_admissions(&self) -> Result<()> {
        let now = slo_clock_now();
        let rejected: Vec<_> = self
            .sequences
            .read()
            .iter()
            .filter_map(|(id, sequence)| {
                let pending = sequence.time_admission.as_ref()?.acceptance.as_ref()?;
                let reason = pending
                    .rejection
                    .clone()
                    .or_else(|| {
                        (now >= pending.expires_at).then_some(
                            FerrumError::SloTimeAdmissionRejected {
                                reason: SloTimeAdmissionRejection::WaitExpired,
                            },
                        )
                    })
                    .or_else(|| {
                        pending
                            .sender
                            .as_ref()
                            .is_none_or(oneshot::Sender::is_closed)
                            .then(|| FerrumError::cancelled("strict admission receiver closed"))
                    })?;
                Some((id.clone(), reason))
            })
            .collect();
        for (id, error) in rejected {
            self.reject_strict_pending(&id, error).await?;
        }
        Ok(())
    }

    pub(in crate::continuous_engine::inner::slo_controller) fn strict_review_at(
        sequence: &SequenceState,
    ) -> Option<Instant> {
        sequence
            .time_admission
            .as_ref()?
            .acceptance
            .as_ref()
            .map(|pending| pending.expires_at)
    }
}
