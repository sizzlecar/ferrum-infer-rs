//! Audit only the original typed prefix result. No prediction or permission is minted.
use super::*;

type Labels = (&'static str, &'static str);

impl ControllerBudget {
    pub fn record_prefix_comparison(&self, result: &PrefixRendezvousDecision) -> Labels {
        match result {
            PrefixRendezvousDecision::Compared { comparison, search } => self.prefix_audit(
                *search,
                None,
                if comparison.should_hold() {
                    ("prefix_hold_recommended", "captured_obligation_scope")
                } else {
                    ("prefix_direct_preferred", "direct_not_slower")
                },
            ),
            PrefixRendezvousDecision::Unknown { reason, search } => {
                self.prefix_unknown(*reason, *search)
            }
        }
    }

    pub fn record_prefix_continuation(&self, result: &PrefixContinuationDecision) -> Labels {
        match result {
            PrefixContinuationDecision::Ready {
                continuation,
                search,
            } => self.prefix_replay(
                *search,
                continuation.remaining().steps(),
                matches!(continuation.action(), PrefixContinuationAction::Wave(_)),
            ),
            PrefixContinuationDecision::Unknown { reason, search } => {
                self.prefix_unknown(*reason, *search)
            }
        }
    }

    pub fn record_ready_prefix(&self, result: &ReadyPrefixDecision) -> Labels {
        match result {
            ReadyPrefixDecision::Ready { evidence, search } => self.prefix_replay(
                *search,
                evidence.remaining().steps(),
                matches!(evidence.action(), ReadyPrefixAction::Wave(_)),
            ),
            ReadyPrefixDecision::PreferDirect { search, .. } => self.prefix_audit(
                *search,
                None,
                ("prefix_direct_preferred", "direct_not_slower"),
            ),
            ReadyPrefixDecision::Unknown { reason, search } => {
                self.prefix_unknown(*reason, *search)
            }
        }
    }

    pub fn record_prefix_cache_capture(&self, result: &PrefixCacheCaptureDecision) -> Labels {
        match result {
            PrefixCacheCaptureDecision::Known { evidence, search } => self.prefix_replay(
                *search,
                evidence.steps(),
                matches!(evidence.action(), PrefixCacheCaptureAction::Wave(_)),
            ),
            PrefixCacheCaptureDecision::Unknown { reason, search } => {
                self.prefix_unknown(*reason, *search)
            }
        }
    }

    fn prefix_replay(
        &self,
        search: PlanningSearchStats,
        steps: &[PrefixPathStep],
        selected_model_wave: bool,
    ) -> Labels {
        // The finite replay can contain Capture/Restore edges. They retain their
        // resource and time proof but are not model waves in witness metrics.
        // A maintenance-first decision does not report a selected model wave.
        let witness = selected_model_wave
            .then(|| {
                let waves = u64::try_from(
                    steps
                        .iter()
                        .filter(|step| matches!(step, PrefixPathStep::Wave(_)))
                        .count(),
                )
                .ok()?;
                Some(ControllerWitnessAudit {
                    waves,
                    tail_waves: waves.checked_sub(1)?,
                })
            })
            .flatten();
        self.prefix_audit(
            search,
            witness,
            if selected_model_wave {
                ("prefix_wave_replayed", "captured_obligation_scope")
            } else {
                ("prefix_maintenance_replayed", "captured_obligation_scope")
            },
        )
    }

    fn prefix_unknown(&self, reason: PlanningUnknownReason, search: PlanningSearchStats) -> Labels {
        let labels = self.prefix_audit(search, None, ("unknown", unknown_label(reason)));
        if reason == PlanningUnknownReason::ComputeBudgetExhausted {
            // Same reporting as ordinary record_search: a reserved-phase stop
            // is not proof that the original hard transaction deadline elapsed.
            self.planner_exhausted.store(true, Ordering::Release);
            let _ = self.poll();
        }
        labels
    }

    fn prefix_audit(
        &self,
        search: PlanningSearchStats,
        witness: Option<ControllerWitnessAudit>,
        labels: Labels,
    ) -> Labels {
        *self.search.lock() = search;
        *self.witness.lock() = witness;
        *self.decision.lock() = labels;
        // backend_submitted and host_reconciled remain solely owned by the
        // original actual dispatch and reconciliation boundaries.
        labels
    }
}

#[cfg(test)]
mod tests;
