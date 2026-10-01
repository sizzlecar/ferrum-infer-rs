//! Independently progressing numerical families over one original offer ledger.
//! Producers retain the existing private tickets. Accepted records enter the
//! worker-owned collector incrementally; only a completely retired block can
//! freeze a numerical phase or publish independently qualified children.
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{
        OwnerBlockScheduleV1, OwnerInputReadinessV1, OwnerOpeningFrontierPolicyV1,
    },
    cost_profile::StructuredServiceAuditV7,
};

mod budget;
pub(super) mod close_task;
pub(super) mod rolling;
mod session;
#[cfg(test)]
mod tests;
pub(super) use session::BlockSession;

pub(super) fn schedule(
    settings: &SloAutomaticCalibrationSettingsV1,
    numerical: &StructuredSettingsV2,
) -> Result<OwnerBlockScheduleV1, FerrumError> {
    use ferrum_types::{
        SloAutomaticCalibrationInputReadinessV1,
        SloAutomaticCalibrationNumericalStrategyV1 as NumericalStrategy,
    };
    let block = settings.discovery_offered_waves.get();
    let offered = settings.phase_offered_waves.map(|n| n.get());
    let minimum = [numerical.min_phase_samples; 3];
    match settings.input_readiness {
        SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {} => {
            OwnerBlockScheduleV1::new(block, offered, minimum)
        }
        SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV1 {
            maximum_phase_blocks,
            maximum_geometry_visits,
        } => OwnerInputReadinessV1::new(
            maximum_phase_blocks.map(|blocks| blocks.get()),
            maximum_geometry_visits.get(),
        )
        .and_then(|policy| {
            OwnerBlockScheduleV1::new_with_input_readiness(block, offered, minimum, policy)
        }),
        SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV2 {
            maximum_phase_blocks,
            maximum_geometry_visits,
        } => OwnerInputReadinessV1::new_cached_residual_v2(
            maximum_phase_blocks.map(|blocks| blocks.get()),
            maximum_geometry_visits.get(),
        )
        .and_then(|policy| {
            OwnerBlockScheduleV1::new_with_input_readiness(block, offered, minimum, policy)
        }),
        SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 {
            maximum_phase_blocks,
            maximum_geometry_visits,
        } => (if settings.numerical_strategy == NumericalStrategy::IdentifiedFitGlobalResidualV1 {
            OwnerInputReadinessV1::new_fit_target_v4(
                maximum_phase_blocks.map(|blocks| blocks.get()),
                maximum_geometry_visits.get(),
            )
        } else {
            OwnerInputReadinessV1::new_zero_column_v3(
                maximum_phase_blocks.map(|blocks| blocks.get()),
                maximum_geometry_visits.get(),
            )
        })
        .and_then(|policy| {
            OwnerBlockScheduleV1::new_with_input_readiness(block, offered, minimum, policy)
        }),
    }
    .map(|mut schedule| {
        schedule.opening_frontier = Some(OwnerOpeningFrontierPolicyV1::FirstOfferFifoV1);
        schedule.prediction_validity = super::super::super::automatic_prediction_validity(settings);
        schedule
    })
    .map_err(|e| error(format!("owner block schedule: {e:?}")))
}

/// Preserve bounded numerical failure details before the original collector is
/// released. Memory-only operation must explain why no model was published.
fn failed_population_reason(audit: &StructuredServiceAuditV7) -> String {
    use std::fmt::Write as _;
    let mut reason = format!(
        "all independently collected families failed (block={}, offered={}, families={})",
        audit.block,
        audit.offered,
        audit.owners.len()
    );
    for owner in audit.owners.iter().take(8) {
        let _ = write!(
            reason,
            "; attempt={} role={:?} rows={} eligible={}/{}: ",
            owner.owner_attempt_id,
            owner.owner.role,
            owner.owner.rows,
            owner.eligible,
            owner.owner_offered
        );
        reason.extend(
            owner
                .failure
                .as_deref()
                .unwrap_or("failure detail unavailable")
                .chars()
                .take(128),
        );
    }
    if audit.owners.len() > 8 {
        let _ = write!(
            reason,
            "; {} further failed families",
            audit.owners.len() - 8
        );
    }
    reason
}

impl Controller {
    pub(in super::super) fn incremental_owner_records(&self) -> bool {
        if self.uses_rolling_owner_blocks() {
            return self
                .state
                .lock()
                .rolling
                .as_ref()
                .is_some_and(|rolling| rolling.has_sources());
        }
        self.uses_owner_blocks() && self.state.lock().blocks.is_some()
    }

    /// Run after the original FIFO feedback turn and outside its locks.
    pub(in super::super) fn ingest_owner_record(&self, live: &LiveCalibration, cutoff: u64) {
        if !self.uses_owner_blocks() {
            return;
        }
        let mut state = self.state.lock();
        if self.uses_rolling_owner_blocks() {
            if let Some(rolling) = &mut state.rolling {
                rolling.ingest(live, cutoff);
            }
            return;
        }
        if state.failure.is_some() {
            return;
        }
        if let Some(blocks) = state.blocks.as_mut() {
            if let Err(reason) = blocks.ingest_one(live, cutoff) {
                state.failure = Some(reason.to_string());
                live.active.read().fail();
            }
        }
    }

    pub(in super::super) fn has_pending_owner_records(&self, live: &LiveCalibration) -> bool {
        if !self.uses_owner_blocks() {
            return false;
        }
        let state = self.state.lock();
        if self.uses_rolling_owner_blocks() {
            return state
                .rolling
                .as_ref()
                .is_some_and(|rolling| rolling.has_pending_records(live));
        }
        state.failure.is_none()
            && state
                .blocks
                .as_ref()
                .is_some_and(|blocks| blocks.has_pending_records(live))
    }

    pub(super) fn advance_owner_blocks(
        &self,
        live: &LiveCalibration,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        stopping: bool,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        if self.uses_rolling_owner_blocks() {
            return self.advance_rolling_owner_blocks(live, clock, cutoff, stopping);
        }
        if stopping {
            self.stop();
        }
        let mut state = self.state.lock();
        if !self.requested.load(Ordering::Acquire) || state.install_pending {
            return Ok(None);
        }
        let stopped = self.stopped.load(Ordering::Acquire);
        let mut window = live.active.read().clone();
        if state.generation == 0 || state.finished {
            if stopped || window.audit().issued != window.audit().retired {
                return Ok(None);
            }
            let generation = state.generation.checked_add(1).ok_or_else(|| {
                self.stop();
                error("generation counter exhausted")
            })?;
            let opening = ExportClockReading::opening(clock).map_err(error)?;
            let blocks = BlockSession::open(
                live,
                self,
                generation,
                opening,
                cutoff,
                clock,
                state.cold_algorithm_seed.as_ref(),
                None,
            )?;
            state.generation = generation;
            state.opening_ns = opening.monotonic_ns;
            state.declaration_sha256 = Some(blocks.declaration_sha256);
            state.blocks = Some(blocks);
            state.phase_populations = std::array::from_fn(|_| None);
            state.failure = None;
            state.finished = false;
            open_window(
                live,
                generation,
                1,
                self.settings.discovery_offered_waves.get(),
                opening.monotonic_ns,
            )?;
            window = live.active.read().clone();
        }
        let next_pending = state.blocks.as_ref().is_some_and(|s| s.next_pending);
        if state.failure.is_some() {
            return self.fail_owner_blocks(live, &mut state, &window, clock, cutoff);
        }
        if next_pending {
            // The last block remains closed during catalog installation. No
            // producer can enter a new block before its assignment is frozen.
            let audit = state.blocks.as_ref().unwrap().audit();
            let all_terminal =
                !audit.owners.is_empty() && audit.owners.iter().all(|owner| owner.phase.is_none());
            if all_terminal && !audit.owners.iter().any(|owner| owner.qualified) {
                state.failure = Some(failed_population_reason(&audit));
                return self.fail_owner_blocks(live, &mut state, &window, clock, cutoff);
            }
            if stopped || all_terminal {
                let finished = state.blocks.as_mut().unwrap().finish(live, clock);
                match finished {
                    Ok(Some(source)) => {
                        state.last_diagnostic_publication = Some(DiagnosticPublication {
                            generation: state.generation,
                            source,
                        });
                    }
                    Ok(None) => {}
                    Err(reason) => {
                        state.failure = Some(reason.to_string());
                        return self.fail_owner_blocks(live, &mut state, &window, clock, cutoff);
                    }
                }
                state.blocks = None;
                state.finished = true;
                if stopped {
                    return Ok(None);
                }
                // Complete an epoch and open its fresh successor at the same
                // accepted FIFO barrier. The recursive call has no offers,
                // hence cannot close another block or recurse again.
                drop(state);
                return self.advance_owner_blocks(live, clock, cutoff, false);
            }
            let opened_at = clock
                .now_ns()
                .ok_or_else(|| error("block clock unavailable"))?;
            let block = match state
                .blocks
                .as_mut()
                .unwrap()
                .open_next(live, opened_at, cutoff)
            {
                Ok(block) => block,
                Err(reason) => {
                    state.failure = Some(reason.to_string());
                    return self.fail_owner_blocks(live, &mut state, &window, clock, cutoff);
                }
            };
            open_window(
                live,
                state.generation,
                block,
                self.settings.discovery_offered_waves.get(),
                state.opening_ns,
            )?;
            window = live.active.read().clone();
        }
        let now = clock.now_ns();
        let complete = now.is_some_and(|at| window.complete(at));
        if now.is_none() {
            window.fail();
            state
                .failure
                .get_or_insert_with(|| "clock unavailable".into());
        }
        // A completed final block may still close and qualify on shutdown.
        // An incomplete block can never become an independent population.
        if stopped && !complete {
            window.fail();
            state
                .failure
                .get_or_insert_with(|| "shutdown before complete owner block".into());
        }
        if window.audit().failed {
            state.failure.get_or_insert_with(|| {
                live.collected.lock().failure.map_or_else(
                    || "original owner block failed or expired".into(),
                    |reason| format!("original owner block failed: {reason}"),
                )
            });
        }
        if state.failure.is_some() {
            return self.fail_owner_blocks(live, &mut state, &window, clock, cutoff);
        }
        if !complete
            || !state
                .blocks
                .as_ref()
                .is_some_and(|blocks| blocks.records_complete())
        {
            return Ok(None);
        }
        let result = state
            .blocks
            .as_mut()
            .ok_or_else(|| error("owner block session missing"))?
            .complete(live, &window, clock, cutoff, &self.import);
        match result {
            Ok(Some(publication)) => {
                state.install_pending = true;
                let generation = state.generation;
                self.record_history(
                    &mut state,
                    GenerationAudit {
                        generation,
                        discovery_origin: None,
                        failed: false,
                        installed: false,
                        reason: None,
                        first_settlement_failure: None,
                        population: window.audit(),
                        phase_populations: std::array::from_fn(|_| None),
                    },
                );
                Ok(Some(publication))
            }
            Ok(None) => Ok(None),
            Err(reason) => {
                state.failure = Some(reason.to_string());
                self.fail_owner_blocks(live, &mut state, &window, clock, cutoff)
            }
        }
    }

    fn fail_owner_blocks(
        &self,
        live: &LiveCalibration,
        state: &mut State,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        window.fail();
        let population = window.audit();
        if population.issued != population.retired {
            return Ok(None);
        }
        let reason = state
            .failure
            .clone()
            .unwrap_or_else(|| "owner block failed".into());
        if let Some(mut blocks) = state.blocks.take() {
            blocks.fail(live, window, clock, cutoff, &reason);
        }
        state.last_failure = Some(RetainedFailure {
            window: window.clone(),
            collected: std::mem::take(&mut *live.collected.lock()),
        });
        state.finished = true;
        state.install_pending = false;
        state.pending_origins = None;
        state.pending_payload_bytes = None;
        self.record_history(
            state,
            GenerationAudit {
                generation: state.generation,
                discovery_origin: None,
                failed: true,
                installed: false,
                reason: Some(reason.chars().take(4096).collect()),
                first_settlement_failure: state
                    .last_failure
                    .as_ref()
                    .and_then(|f| f.collected.first_settlement_failure),
                population,
                phase_populations: std::array::from_fn(|_| None),
            },
        );
        let _ = live
            .failed_generations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1));
        live.remember_failed_source(state.generation, None, false, false, &reason);
        Err(error(reason))
    }
}
