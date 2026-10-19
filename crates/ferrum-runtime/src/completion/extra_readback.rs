use super::*;
use crate::{ExecutionPlan, PlanHash, ResolvedValueRole};

/// One additional read-only range of a formal program output. Construction
/// binds the immutable plan's retention contract; terminal wait revalidates the
/// original submission, participant, current backing, and translated range.
///
/// The count is exactly one, and `max_bytes` bounds its logical transfer bytes
/// before any readback allocation. It is not a total allocator/RSS bound. No
/// extra output is retained by the plan and no execution policy is changed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "an extra readback must accompany the original complete primary batch"]
pub struct CompletionExtraReadbackRequest {
    plan_hash: PlanHash,
    request: CompletionReadbackRequest,
}

impl CompletionExtraReadbackRequest {
    pub fn new(
        plan: &ExecutionPlan,
        request: CompletionReadbackRequest,
        max_bytes: u64,
    ) -> Result<Self, VNextError> {
        let bytes = request.output_layout().byte_len()?;
        if bytes == 0
            || bytes > max_bytes
            || usize::try_from(bytes).is_err()
            || bytes > isize::MAX as u64
        {
            return Err(invalid_completion(
                "extra readback exceeds its nonzero host byte budget",
            ));
        }
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == request.node_id())
            .ok_or_else(|| invalid_completion("extra readback node is absent from its plan"))?;
        if !plan
            .payload()
            .terminal_output_resources()
            .contains(request.resource_id())
            || !node.values().iter().any(|value| {
                value.role() == ResolvedValueRole::Output
                    && value.usage() == request.expected_usage()
                    && value.storage().components().iter().any(|component| {
                        component.resource_id() == request.resource_id()
                            && component.element_type() == request.output_layout().element_type()
                    })
            })
        {
            return Err(invalid_completion(
                "extra readback requires a formal retained program output of its original node",
            ));
        }
        Ok(Self {
            plan_hash: plan.plan_hash().clone(),
            request,
        })
    }

    pub fn request(&self) -> &CompletionReadbackRequest {
        &self.request
    }

    fn validate_for<R: DeviceRuntime>(
        &self,
        resources: &CompletionResourceLease<R>,
        identity: &BatchOperationIdentity,
    ) -> Result<(), VNextError> {
        let request = &self.request;
        let node = identity.node_index(request.node_id()).ok_or_else(|| {
            invalid_completion("extra readback node is absent from its submission")
        })?;
        if identity.plan_hash() != &self.plan_hash
            || request.participant_index() as usize
                >= identity.node_participant_count(node).ok_or_else(|| {
                    invalid_completion("extra readback node has no participant count")
                })?
        {
            return Err(invalid_completion(
                "extra readback differs from its original plan or participant",
            ));
        }
        let bytes = request.output_layout().byte_len()?;
        let end = request
            .logical_offset_bytes()
            .checked_add(bytes)
            .ok_or_else(|| invalid_completion("extra readback range overflows"))?;
        let range = resources.readback_range(
            request.node_id(),
            request.participant_index(),
            request.resource_id(),
            request.logical_offset_bytes()..end,
        )?;
        let backing = resources.backing_view(
            request.node_id(),
            request.participant_index(),
            request.resource_id(),
        )?;
        let element_bytes = request.output_layout().element_type().size_bytes();
        if backing.usage() != request.expected_usage()
            || backing.element_type() != request.output_layout().element_type()
            || range.end > backing.size_bytes()
            || range.end.checked_sub(range.start) != Some(bytes)
            || range.start % element_bytes != 0
        {
            return Err(invalid_completion(
                "extra readback is outside its current typed participant output",
            ));
        }
        Ok(())
    }
}

/// Primary and extra evidence remain separate. A successful original execution
/// alone does not imply that either readback succeeded: consumers must inspect
/// both sets of typed dispositions. Both receipts refer to the same terminal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[must_use = "inspect both primary and extra dispositions, including failures"]
pub struct CompletionReadbackWithExtraReceipt {
    primary: CompletionReadbackBatchReceipt,
    extra: CompletionReadbackReceipt,
}

impl CompletionReadbackWithExtraReceipt {
    pub fn primary(&self) -> &CompletionReadbackBatchReceipt {
        &self.primary
    }
    pub fn extra(&self) -> &CompletionReadbackReceipt {
        &self.extra
    }
    pub fn into_parts(self) -> (CompletionReadbackBatchReceipt, CompletionReadbackReceipt) {
        (self.primary, self.extra)
    }
}

#[derive(Debug)]
#[must_use = "nonterminal observations retain original fence and resource custody"]
pub enum CompletionReadbackWithExtraObservation {
    Pending,
    Terminal(CompletionReadbackWithExtraReceipt),
    Indeterminate(Vec<IdentifiedFailure>),
    SubmissionIndeterminate,
    ObservationPanicked,
    Quarantined(CompletionQuarantineReceipt),
}

impl<R: DeviceRuntime> CompletionHandle<R> {
    /// Consume the original complete primary batch and one bounded extra output
    /// in a single terminal observation. An invalid request returns an error
    /// before waiting and leaves the original slot intact. Cancellation/drop of
    /// this weak handle never releases the reaper's original fence or resources.
    pub fn wait_with_readbacks_and_extra(
        &self,
        primary: CompletionReadbackBatchRequest,
        extra: CompletionExtraReadbackRequest,
    ) -> Result<CompletionReadbackWithExtraObservation, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .wait_bound_with_readbacks_and_extra(self.slot_id(), primary, extra)
    }
}

impl<R: DeviceRuntime> CompletionReaper<R> {
    pub(crate) fn wait_bound_with_readbacks_and_extra(
        &self,
        slot_id: CompletionSlotId,
        primary: CompletionReadbackBatchRequest,
        extra: CompletionExtraReadbackRequest,
    ) -> Result<CompletionReadbackWithExtraObservation, VNextError> {
        // Preserve exact prepared-packet equality and the complete-node cohort
        // check. The extra row is never appended to or substituted for primary.
        self.validate_bound_readback_batch(slot_id, &primary)?;
        {
            let record = self.lookup(slot_id)?;
            let guard = record
                .lock()
                .map_err(|_| invalid_completion("completion slot mutex is poisoned"))?;
            match &*guard {
                CompletionRecord::InFlight {
                    resources,
                    batch_identity,
                    ..
                }
                | CompletionRecord::SubmissionIndeterminate {
                    resources,
                    batch_identity,
                    ..
                } => {
                    extra.validate_for(resources, batch_identity)?;
                }
                CompletionRecord::Quarantined { .. } => {}
                _ => {
                    return Err(invalid_completion(
                        "extra readback requires its original live submission",
                    ))
                }
            }
        }
        let observation = self.observe_bound_with(
            slot_id,
            true,
            |resources, lane, identity, disposition, timing_mode| {
                let primary = attempt_completion_readbacks(
                    resources,
                    lane,
                    identity,
                    disposition,
                    timing_mode,
                    primary,
                );
                let request = extra.request;
                let extra = if primary
                    .iter()
                    .all(|item| matches!(item, CompletionReadbackDisposition::Succeeded(_)))
                {
                    // Also catch error-description/validation panics: the original
                    // terminal is already proven, and diagnostics cannot unwind
                    // past its final resource accounting or pretend success.
                    catch_unwind(AssertUnwindSafe(|| {
                        attempt_completion_readback(
                            resources,
                            lane,
                            identity,
                            disposition,
                            timing_mode,
                            request.clone(),
                        )
                    }))
                    .unwrap_or_else(|_| {
                        lane.fail_closed();
                        CompletionReadbackDisposition::ContractFailedButQuiescent {
                            request,
                            failure: QuiescentCompletionContractFailure::new(
                                "extra completion readback panicked",
                            ),
                        }
                    })
                } else {
                    CompletionReadbackDisposition::NotAttempted(request)
                };
                (primary, extra)
            },
        )?;
        Ok(match observation {
            BoundCompletionObservation::Pending => CompletionReadbackWithExtraObservation::Pending,
            BoundCompletionObservation::Terminal {
                completion,
                terminal: (primary, extra),
            } => CompletionReadbackWithExtraObservation::Terminal(
                CompletionReadbackWithExtraReceipt {
                    extra: CompletionReadbackReceipt::new(completion.clone(), extra),
                    primary: CompletionReadbackBatchReceipt::new(completion, primary),
                },
            ),
            BoundCompletionObservation::Indeterminate(failures) => {
                CompletionReadbackWithExtraObservation::Indeterminate(failures)
            }
            BoundCompletionObservation::SubmissionIndeterminate => {
                CompletionReadbackWithExtraObservation::SubmissionIndeterminate
            }
            BoundCompletionObservation::ObservationPanicked => {
                CompletionReadbackWithExtraObservation::ObservationPanicked
            }
            BoundCompletionObservation::Quarantined(receipt) => {
                CompletionReadbackWithExtraObservation::Quarantined(receipt)
            }
        })
    }
}
