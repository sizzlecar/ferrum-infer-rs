//! Bounded execution for private calibration cohorts. The caller freezes the
//! source population before entering this driver. Reports and byte counters
//! are diagnostics; the original session collectors own numerical membership.
use super::*;
use ferrum_interfaces::output_flow::OutputCompletion;
use ferrum_types::InferenceRequest;
use futures::StreamExt;
use std::num::{NonZeroU32, NonZeroUsize};
use std::time::Duration;
use tokio::{task::JoinSet, time::Instant};

pub(super) struct ProbeRequest {
    pub request: InferenceRequest,
    pub contract: Arc<OutputProjectionContract>,
}

/// An input-declared prefix preparation path. Every chosen wave still needs
/// the original executor/resource proof; this is never execution permission.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ProbePrefillPlan {
    Joint,
    PreparedSequentialV1,
}

impl ProbePrefillPlan {
    pub(super) fn rows_per_wave(self, width: usize) -> usize {
        match self {
            Self::Joint => width,
            Self::PreparedSequentialV1 => 1,
        }
    }

    pub(super) fn waves(self, chunks_per_row: usize, width: usize) -> Option<usize> {
        match self {
            Self::Joint => Some(chunks_per_row),
            Self::PreparedSequentialV1 => chunks_per_row.checked_mul(width),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ProbeCohortSettings {
    pub prefill_plan: ProbePrefillPlan,
    pub prefill_chunk: NonZeroU32,
    pub decode_route: CalibrationDecodeRoute,
    pub reset_token_policy: bool,
}

/// Reported completed preflight admissions and nonrefundable reservations.
/// An interrupted capture/driver may have admitted fewer owners than reserved;
/// reported admissions are not a complete failure-path creation counter.
/// None of these fields is an offered numerical sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) struct ProbePreflightCharge {
    pub admitted_requests: usize,
    pub planning_admitted_requests: usize,
    pub readiness_admitted_requests: usize,
    pub planning_reserved_requests: usize,
    pub readiness_reserved_requests: usize,
    pub projection_attempts: usize,
}

/// The remaining cost allowance while the independent reference probe runs.
/// Ownership is consumed on resume; collection cannot recreate the original
/// request allowance or renew the duration already spent on cost preflight.
pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeExecutionBudget {
    remaining: Duration,
    requests_remaining: usize,
    input_projection_requests_remaining: usize,
    attempts_remaining: usize,
    preflight: ProbePreflightCharge,
}

impl PreparedProbeExecutionBudget {
    /// Only the static preparation-to-reference handoff. Readiness executes
    /// after resume; this constructor must not recreate its spent wave credit.
    pub fn new(
        remaining: Duration,
        maximum_requests: NonZeroUsize,
        maximum_wave_attempts: NonZeroUsize,
        maximum_input_projection_requests: NonZeroUsize,
        preflight: ProbePreflightCharge,
    ) -> Result<Self> {
        if remaining.is_zero() {
            return Err(FerrumError::resource_exhausted(
                "automatic cost preflight duration budget exhausted",
            ));
        }
        if preflight.readiness_reserved_requests != 0 {
            return Err(invalid(
                "readiness cannot precede the static budget handoff",
            ));
        }
        let requests_remaining = maximum_requests
            .get()
            .checked_sub(preflight.readiness_reserved_requests)
            .ok_or_else(|| {
                FerrumError::resource_exhausted("automatic cost preflight request budget exhausted")
            })?;
        let input_projection_requests_remaining = maximum_input_projection_requests
            .get()
            .checked_sub(preflight.planning_reserved_requests)
            .ok_or_else(|| {
                FerrumError::resource_exhausted(
                    "automatic input projection request budget exhausted",
                )
            })?;
        if preflight.planning_admitted_requests > preflight.planning_reserved_requests
            || preflight.readiness_admitted_requests > preflight.readiness_reserved_requests
            || preflight
                .planning_admitted_requests
                .checked_add(preflight.readiness_admitted_requests)
                != Some(preflight.admitted_requests)
        {
            return Err(invalid("preflight admission accounting differs"));
        }
        Ok(Self {
            remaining,
            requests_remaining,
            input_projection_requests_remaining,
            attempts_remaining: maximum_wave_attempts.get(),
            preflight,
        })
    }

    pub fn resume(self) -> Result<ProbeExecutionBudget> {
        let deadline = Instant::now()
            .checked_add(self.remaining)
            .ok_or_else(|| FerrumError::config("automatic cost probe deadline overflow"))?;
        Ok(ProbeExecutionBudget {
            deadline,
            requests_remaining: self.requests_remaining,
            input_projection_requests_remaining: self.input_projection_requests_remaining,
            attempts_remaining: self.attempts_remaining,
            preflight: self.preflight,
            selection_requests_remaining: self.requests_remaining,
            selection_attempts_remaining: self.attempts_remaining,
            input_geometry_work: None,
        })
    }
}

/// One budget spans every cohort and independent phase. Neither a failed
/// sample nor a new cohort renews its deadline or replenishes work.
pub(super) struct ProbeExecutionBudget {
    deadline: Instant,
    requests_remaining: usize,
    input_projection_requests_remaining: usize,
    attempts_remaining: usize,
    preflight: ProbePreflightCharge,
    // Input-only reservations never refund EOS savings into a later source.
    // Actual requests/attempts still pass the independent original driver gate.
    selection_requests_remaining: usize,
    selection_attempts_remaining: usize,
    // One allowance across every population, progressive input unit and phase.
    input_geometry_work: Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputGeometryWorkV1>,
}

impl ProbeExecutionBudget {
    pub fn new(
        deadline: Instant,
        maximum_requests: NonZeroUsize,
        maximum_wave_attempts: NonZeroUsize,
    ) -> Self {
        Self::new_with_input_projection_limit(
            deadline,
            maximum_requests,
            maximum_wave_attempts,
            maximum_requests,
        )
    }

    pub fn new_with_input_projection_limit(
        deadline: Instant,
        maximum_requests: NonZeroUsize,
        maximum_wave_attempts: NonZeroUsize,
        maximum_input_projection_requests: NonZeroUsize,
    ) -> Self {
        Self {
            deadline,
            requests_remaining: maximum_requests.get(),
            input_projection_requests_remaining: maximum_input_projection_requests.get(),
            attempts_remaining: maximum_wave_attempts.get(),
            preflight: ProbePreflightCharge::default(),
            selection_requests_remaining: maximum_requests.get(),
            selection_attempts_remaining: maximum_wave_attempts.get(),
            input_geometry_work: None,
        }
    }

    pub fn input_geometry_work(
        &mut self,
        maximum_visits: Option<std::num::NonZeroU64>,
    ) -> Result<Option<&mut ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputGeometryWorkV1>>{
        self.require_time()?;
        let Some(maximum_visits) = maximum_visits else {
            return Ok(None);
        };
        if let Some(work) = &self.input_geometry_work {
            if work.maximum_visits() != maximum_visits.get() {
                return Err(invalid(
                    "cold input geometry allowance changed within original budget",
                ));
            }
        } else {
            self.input_geometry_work = Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputGeometryWorkV1::new(maximum_visits));
        }
        Ok(self.input_geometry_work.as_mut())
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    pub fn requests_remaining(&self) -> usize {
        self.requests_remaining
    }

    pub fn input_projection_requests_remaining(&self) -> usize {
        self.input_projection_requests_remaining
    }

    pub fn attempts_remaining(&self) -> usize {
        self.attempts_remaining
    }

    pub fn selection_requests_remaining(&self) -> usize {
        self.selection_requests_remaining
    }

    pub fn selection_attempts_remaining(&self) -> usize {
        self.selection_attempts_remaining
    }

    pub fn require_selection_time(&self) -> Result<()> {
        self.require_time()
    }

    pub fn reserve_selected_source(&mut self, requests: usize, waves: usize) -> Result<()> {
        self.require_time()?;
        if requests == 0 || waves == 0 {
            return Err(invalid("empty declared source work"));
        }
        let requests = self
            .selection_requests_remaining
            .checked_sub(requests)
            .ok_or_else(|| invalid("declared source request budget exhausted"))?;
        let waves = self
            .selection_attempts_remaining
            .checked_sub(waves)
            .ok_or_else(|| invalid("declared source wave budget exhausted"))?;
        self.selection_requests_remaining = requests;
        self.selection_attempts_remaining = waves;
        Ok(())
    }

    pub fn reserve_readiness_waves(&mut self, waves: usize) -> Result<()> {
        self.require_time()?;
        self.selection_attempts_remaining = self
            .selection_attempts_remaining
            .checked_sub(waves)
            .ok_or_else(|| invalid("declared readiness wave budget exhausted"))?;
        Ok(())
    }

    /// Before a source collector opens, fixed original members may require
    /// more work on the cold path. Keep every setup and sibling reservation;
    /// only unused selection credit can cover the additional collection work.
    /// Refusal does not mutate either ledger, and actual execution pays later.
    pub fn try_reserve_additional_source_actions(&mut self, actions: usize) -> Result<bool> {
        self.require_time()?;
        let Some(remaining) = self.selection_attempts_remaining.checked_sub(actions) else {
            return Ok(false);
        };
        self.selection_attempts_remaining = remaining;
        Ok(true)
    }

    pub fn preflight_charge(&self) -> ProbePreflightCharge {
        self.preflight
    }

    /// Reserve before creating any planning owner. A cancelled or failed group
    /// keeps this charge; later groups and recaptures cannot renew its quota.
    pub fn claim_input_projection_requests(&mut self, count: usize) -> Result<()> {
        if count == 0 || Instant::now() >= self.deadline {
            return Err(invalid(
                "automatic input projection has no request/time budget",
            ));
        }
        let remaining = self
            .input_projection_requests_remaining
            .checked_sub(count)
            .ok_or_else(|| invalid("automatic input projection request budget exhausted"))?;
        let reserved = self
            .preflight
            .planning_reserved_requests
            .checked_add(count)
            .ok_or_else(|| invalid("planning reservation overflow"))?;
        self.input_projection_requests_remaining = remaining;
        self.preflight.planning_reserved_requests = reserved;
        Ok(())
    }

    /// Report completed pure capture against an already reserved planning
    /// allowance. Execution requests and real wave attempts are untouched.
    pub fn record_geometry(&mut self, charge: ProbePreflightCharge) -> Result<()> {
        if charge.admitted_requests != charge.planning_admitted_requests
            || charge.readiness_admitted_requests != 0
            || charge.readiness_reserved_requests != 0
            || charge.planning_reserved_requests != 0
        {
            return Err(invalid(
                "geometry charge contains execution or reservation counts",
            ));
        }
        let planning = self
            .preflight
            .planning_admitted_requests
            .checked_add(charge.planning_admitted_requests)
            .filter(|n| *n <= self.preflight.planning_reserved_requests)
            .ok_or_else(|| invalid("geometry exceeded its planning reservation"))?;
        let total = self
            .preflight
            .admitted_requests
            .checked_add(charge.admitted_requests)
            .ok_or_else(|| invalid("preflight admission counter overflow"))?;
        let projections = self
            .preflight
            .projection_attempts
            .checked_add(charge.projection_attempts)
            .ok_or_else(|| invalid("geometry projection counter overflow"))?;
        self.preflight.planning_admitted_requests = planning;
        self.preflight.admitted_requests = total;
        self.preflight.projection_attempts = projections;
        Ok(())
    }

    /// Readiness's ordinary driver already charged the requests and real wave
    /// attempts. Record the completed preparation without charging twice.
    pub fn record_readiness_admissions(&mut self, count: usize) -> Result<()> {
        let readiness = self
            .preflight
            .readiness_admitted_requests
            .checked_add(count)
            .filter(|n| *n <= self.preflight.readiness_reserved_requests)
            .ok_or_else(|| invalid("readiness exceeded its execution reservation"))?;
        let total = self
            .preflight
            .admitted_requests
            .checked_add(count)
            .ok_or_else(|| invalid("preflight request counter overflow"))?;
        self.preflight.readiness_admitted_requests = readiness;
        self.preflight.admitted_requests = total;
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn claim_readiness_requests(
        &mut self,
        count: usize,
    ) -> Result<()> {
        let reserved = self
            .preflight
            .readiness_reserved_requests
            .checked_add(count)
            .ok_or_else(|| invalid("readiness reservation overflow"))?;
        let selection_remaining = self
            .selection_requests_remaining
            .checked_sub(count)
            .ok_or_else(|| invalid("declared readiness request budget exhausted"))?;
        self.claim_requests(count)?;
        self.preflight.readiness_reserved_requests = reserved;
        self.selection_requests_remaining = selection_remaining;
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn claim_checkpoint_owner(
        &mut self,
    ) -> Result<()> {
        let remaining = self
            .selection_requests_remaining
            .checked_sub(1)
            .ok_or_else(|| invalid("checkpoint probe declared owner budget exhausted"))?;
        self.claim_requests(1)?;
        self.selection_requests_remaining = remaining;
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn claim_checkpoint_action(
        &mut self,
    ) -> Result<()> {
        let remaining = self
            .selection_attempts_remaining
            .checked_sub(1)
            .ok_or_else(|| invalid("checkpoint probe declared action budget exhausted"))?;
        self.claim_attempt()?;
        self.selection_attempts_remaining = remaining;
        Ok(())
    }

    /// Source selection already reserved these actions. The remaining
    /// difference is consumed exactly once by the real native submission.
    pub(in crate::continuous_engine::inner::calibration) fn claim_reserved_checkpoint_owner(
        &mut self,
    ) -> Result<()> {
        self.requests_remaining
            .checked_sub(self.selection_requests_remaining)
            .filter(|remaining| *remaining > 0)
            .ok_or_else(|| invalid("native owner has no original selected reservation"))?;
        self.claim_requests(1)
    }
    pub(in crate::continuous_engine::inner::calibration) fn claim_reserved_checkpoint_action(
        &mut self,
    ) -> Result<()> {
        self.attempts_remaining
            .checked_sub(self.selection_attempts_remaining)
            .filter(|remaining| *remaining > 0)
            .ok_or_else(|| invalid("native action has no original selected reservation"))?;
        self.claim_attempt()
    }

    fn claim_requests(&mut self, count: usize) -> Result<()> {
        if count == 0 || Instant::now() >= self.deadline {
            return Err(invalid("automatic cost probe has no request/time budget"));
        }
        self.requests_remaining = self
            .requests_remaining
            .checked_sub(count)
            .ok_or_else(|| invalid("automatic cost probe request budget exhausted"))?;
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn require_time(&self) -> Result<()> {
        if Instant::now() >= self.deadline {
            return Err(invalid(
                "automatic cost probe duration budget expired before new work",
            ));
        }
        Ok(())
    }

    fn claim_attempt(&mut self) -> Result<()> {
        self.require_time()?;
        self.attempts_remaining = self
            .attempts_remaining
            .checked_sub(1)
            .ok_or_else(|| invalid("automatic cost probe wave budget exhausted"))?;
        Ok(())
    }
}

#[derive(Default, Debug)]
pub(super) struct ProbeCohortSummary {
    pub completed_requests: usize,
    pub completed_output_tokens: u64,
    pub wave_attempts: usize,
    pub reconciled_waves: usize,
    pub released_prefix_rows: usize,
    /// Native maintenance actions are never offered inference samples.
    pub native_restore_attempts: usize,
    pub acknowledged_prefix_restores: usize,
    pub prefix_restore_fallbacks: usize,
}

fn invalid(message: &str) -> FerrumError {
    FerrumError::invalid_request(message)
}

#[derive(Clone, Copy)]
enum ProbeCohortPurpose {
    Numerical,
    Readiness,
}

impl CalibrationSession {
    /// Runs only in the session's existing exclusive driver boundary. The
    /// deadline prevents new work; an already started session transaction keeps
    /// its output consumers until original physical and host settlement returns.
    /// Callers must not wrap this owned driver in a deadline that drops it.
    /// Unrelated external cancellation still requires the session's normal drain.
    pub(super) async fn run_probe_cohort(
        &mut self,
        requests: Vec<ProbeRequest>,
        settings: ProbeCohortSettings,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<ProbeCohortSummary> {
        Box::pin(self.run_probe_cohort_with_purpose(
            requests,
            settings,
            budget,
            ProbeCohortPurpose::Numerical,
            None,
        ))
        .await
    }

    pub(super) async fn run_readiness_probe_cohort(
        &mut self,
        requests: Vec<ProbeRequest>,
        settings: ProbeCohortSettings,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<ProbeCohortSummary> {
        Box::pin(self.run_probe_cohort_with_purpose(
            requests,
            settings,
            budget,
            ProbeCohortPurpose::Readiness,
            None,
        ))
        .await
    }

    /// API proof for prepared Decode acquisition. The source declaration and
    /// original F/R/Q collector remain unchanged. Ordinary ActualPrefill
    /// populations cannot be silently turned into warm-prefix observations.
    pub(in crate::continuous_engine::inner::calibration) async fn run_acquired_prefix_probe_cohort(
        &mut self,
        requests: Vec<ProbeRequest>,
        settings: ProbeCohortSettings,
        budget: &mut ProbeExecutionBudget,
        acquired: &super::startup::AcquiredProbePrefix,
    ) -> Result<ProbeCohortSummary> {
        if !self.prepared_owner_source_active()
            || self.prepared_owner_prefix_release_generated()?.is_none()
            || acquired.prefill_chunk() != settings.prefill_chunk
        {
            return Err(invalid(
                "acquired probe prefix requires original prepared Decode membership",
            ));
        }
        Box::pin(self.run_probe_cohort_with_purpose(
            requests,
            settings,
            budget,
            ProbeCohortPurpose::Numerical,
            Some(acquired),
        ))
        .await
    }

    async fn run_probe_cohort_with_purpose(
        &mut self,
        requests: Vec<ProbeRequest>,
        settings: ProbeCohortSettings,
        budget: &mut ProbeExecutionBudget,
        purpose: ProbeCohortPurpose,
        acquired: Option<&super::startup::AcquiredProbePrefix>,
    ) -> Result<ProbeCohortSummary> {
        self.completed_owner_boundary()?;
        if !self.engine.inner.manual_calibration_driver
            || self.engine.inner.bg_loop_spawned.load(Ordering::Acquire)
            || self.engine.inner.is_running.load(Ordering::Acquire)
            || self.engine.inner.shutdown_started.load(Ordering::Acquire)
            || requests.len() > self.limits.maximum_requests().get()
        {
            return Err(invalid(
                "automatic cost probe requires an unused private cohort",
            ));
        }
        if matches!(
            settings.prefill_plan,
            ProbePrefillPlan::PreparedSequentialV1
        ) && (!self.prepared_owner_source_active()
            || self.prepared_owner_prefix_release_generated()?.is_none())
        {
            return Err(invalid(
                "sequential probe prefill requires an original source8 prefix cohort",
            ));
        }
        // Reject duplicate owners before any request is admitted.
        for (i, current) in requests.iter().enumerate() {
            if requests[..i]
                .iter()
                .any(|previous| previous.request.id == current.request.id)
            {
                return Err(invalid("automatic cost probe declares duplicate requests"));
            }
        }
        let requested = requests.len();
        match purpose {
            ProbeCohortPurpose::Numerical => budget.claim_requests(requested)?,
            ProbeCohortPurpose::Readiness => budget.claim_readiness_requests(requested)?,
        }
        let summary =
            Box::pin(self.drive_probe_cohort(requests, settings, budget, acquired)).await?;
        if matches!(purpose, ProbeCohortPurpose::Readiness) {
            budget.record_readiness_admissions(requested)?;
        }
        Ok(summary)
    }

    async fn drive_probe_cohort(
        &mut self,
        requests: Vec<ProbeRequest>,
        settings: ProbeCohortSettings,
        budget: &mut ProbeExecutionBudget,
        acquired: Option<&super::startup::AcquiredProbePrefix>,
    ) -> Result<ProbeCohortSummary> {
        budget.require_time()?;
        if settings.reset_token_policy {
            use ferrum_interfaces::model_executor::TokenPolicyResidencyInvalidation;
            // No wave or output owner exists at this cancellable boundary.
            if !matches!(
                tokio::time::timeout_at(budget.deadline, self.invalidate_token_policy_residency())
                    .await
                    .map_err(|_| invalid("automatic cost probe residency deadline expired"))?,
                TokenPolicyResidencyInvalidation::Cleared { .. }
            ) {
                return Err(invalid("declared cost probe residency reset unavailable"));
            }
        }
        let request_count = requests.len();
        let mut owners = Vec::with_capacity(request_count);
        let mut consumers = JoinSet::new();
        let result = async {
        for declared in requests {
            budget.require_time()?;
            let id = declared.request.id.clone();
            let maximum = declared.request.sampling_params.max_tokens;
            let output = self
                .add_request(
                    declared.request,
                    InferenceRequestContext::capture(),
                    declared.contract,
                )
                .await?;
            owners.push(id);
            consumers.spawn(consume_probe(output, maximum));
        }
        let mut summary = ProbeCohortSummary::default();
        let mut acquisition_restored = false;
        while summary.completed_requests < request_count {
            self.check_prepared_owner_capture()?;
            while let Some(joined) = consumers.try_join_next() {
                let tokens =
                    joined.map_err(|_| invalid("automatic cost probe output task failed"))??;
                summary.completed_output_tokens = summary
                    .completed_output_tokens
                    .checked_add(tokens)
                    .ok_or_else(|| invalid("automatic cost probe output counter overflow"))?;
                summary.completed_requests += 1;
            }
            if summary.completed_requests == request_count {
                break;
            }
            // Each action is an owned transaction. Once started, retain these
            // consumers until it settles; check the unchanged deadline before
            // starting another action, never by cancelling an in-flight one.
            budget.require_time()?;
            match self.step(CalibrationAction::Maintenance).await? {
                CalibrationTurn::MaintenanceReconciled => continue,
                CalibrationTurn::Blocked(_) => {}
                _ => return Err(invalid("unexpected cost probe maintenance state")),
            }
            budget.require_time()?;
            let admission = self.step(CalibrationAction::AdmitOne).await?;
            if !matches!(admission, CalibrationTurn::Blocked(_)) {
                tokio::task::yield_now().await;
                continue;
            }
            if !acquisition_restored {
                if let Some(acquired) = acquired {
                    if self.frontiers()?.len() != request_count {
                        return Err(invalid("prefix acquisition requires the complete admitted cohort"));
                    }
                    for id in &owners {
                        tokio::time::timeout_at(budget.deadline(), self.startup_output_ready(id))
                            .await.map_err(|_| invalid("prefix restore output readiness expired"))??;
                        let before = budget.attempts_remaining();
                        let restored = self.restore_acquired_probe_prefix(id, acquired, budget).await?;
                        let attempts = before.checked_sub(budget.attempts_remaining())
                            .ok_or_else(|| invalid("prefix restore budget increased"))?;
                        summary.native_restore_attempts = summary.native_restore_attempts.checked_add(attempts)
                            .ok_or_else(|| invalid("prefix restore attempt counter overflow"))?;
                        match restored {
                            super::startup::ProbePrefixRestore::Acknowledged { .. } => {
                                summary.acknowledged_prefix_restores += 1;
                            }
                            super::startup::ProbePrefixRestore::ColdFallback(reason) => {
                                // Some acquisition is a frozen source contract:
                                // cancel and drain instead of changing to cold.
                                return Err(invalid(&format!("declared native prefix restore unavailable: {reason:?}")));
                            }
                        }
                    }
                }
                acquisition_restored = true;
            }
            let prefix = if self.prepared_owner_source_active() {
                self.advance_prepared_owner_prefix_release()?
            } else {
                self.advance_structured_prefix_release_v5()?
            };
            if let PrefixReleaseProgressV5::Released { receipts } = &prefix {
                summary.released_prefix_rows = summary
                    .released_prefix_rows
                    .checked_add(receipts.len())
                    .ok_or_else(|| invalid("automatic cost probe prefix counter overflow"))?;
            }
            if matches!(prefix, PrefixReleaseProgressV5::AwaitingCredit) {
                tokio::task::yield_now().await;
                continue;
            }
            let preparing = matches!(prefix, PrefixReleaseProgressV5::Preparing);
            let release = if preparing {
                if self.prepared_owner_source_active() {
                    self.prepared_owner_prefix_release_generated()?
                } else {
                    self.structured_prefix_release_generated_v5()?
                }
            } else {
                None
            };
            let frontiers = self.frontiers()?;
            if summary.reconciled_waves == 0 && frontiers.len() != request_count {
                return Err(invalid("automatic cost probe requires the entire declared cohort admitted before its first wave"));
            }
            if frontiers
                .iter()
                .any(|frontier| !owners.contains(frontier.request_id()))
            {
                return Err(invalid("unrelated owner entered private cost probe cohort"));
            }
            let mut prefills = Vec::new();
            let mut decodes = Vec::new();
            for id in &owners {
                let Some(frontier) = frontiers.iter().find(|f| f.request_id() == id) else {
                    continue;
                };
                if release.is_some_and(|n| frontier.generated_tokens() >= n) {
                    continue;
                }
                if let Some((offset, total)) = frontier.prefill_progress() {
                    let count = total
                        .checked_sub(offset)
                        .and_then(|n| u32::try_from(n).ok())
                        .and_then(|n| NonZeroU32::new(n.min(settings.prefill_chunk.get())))
                        .ok_or_else(|| invalid("automatic cost probe prefill frontier invalid"))?;
                    prefills.push(frontier.prefill_work(count)?);
                } else {
                    decodes.push(frontier.decode_work_with_route(if preparing {
                        CalibrationDecodeRoute::Actual
                    } else {
                        settings.decode_route
                    })?);
                }
            }
            // The declared probe driver uses split execution. A future mixed
            // policy requires an explicit declaration rather than an implicit
            // fallback when a requested shape cannot execute.
            let work = if prefills.is_empty() {
                decodes
            } else {
                // Source8 freezes this path in its cohort manifest. Reach the
                // declared joint decode frontier through legal single-owner
                // prefills, retaining every original prefix/FIFO receipt. A
                // prepared row is not an ordinary numerical member.
                let plan = if preparing && self.prepared_owner_source_active() {
                    settings.prefill_plan
                } else {
                    ProbePrefillPlan::Joint
                };
                prefills.truncate(plan.rows_per_wave(prefills.len()));
                prefills
            };
            if !work.is_empty() {
                // A scheduling yield is not evidence that an output actor has
                // applied the previous frame and restored credit. Wait on its
                // real readiness before offering the next declared attempt.
                // No wave is in flight at this boundary, so cancelling only
                // this readiness wait cannot invalidate an owned settlement.
                for row in &work {
                    tokio::time::timeout_at(budget.deadline,
                        self.startup_output_ready(row.frontier.request_id()))
                        .await.map_err(|_| invalid("automatic cost probe output readiness deadline expired"))??;
                }
                budget.claim_attempt()?;
                summary.wave_attempts += 1;
                match self.step(CalibrationAction::Wave(work)).await? {
                    CalibrationTurn::Wave(report) | CalibrationTurn::Reaped(report) => {
                        if report.submission == CalibrationSubmissionState::InFlightUnknown {
                            return Err(invalid("automatic cost probe submission indeterminate"));
                        }
                        if let Some(error) = report.error {
                            return Err(error);
                        }
                        if report.submission == CalibrationSubmissionState::HostReconciled {
                            summary.reconciled_waves += 1;
                        }
                    }
                    CalibrationTurn::Blocked(_) => {}
                    _ => return Err(invalid("unexpected automatic cost probe wave state")),
                }
                self.check_prepared_owner_capture()?;
            }
            // Give the real output actors and the single overall deadline a
            // chance to progress even while all work is temporarily blocked.
            tokio::task::yield_now().await;
        }
        self.completed_owner_boundary()?;
        Ok(summary)
        }.await;
        if result.is_err() {
            // No submitted transaction remains inside this future here. Wait
            // for cancellation to drop its real leases before caller cleanup
            // examines the request owners; abort-on-drop alone is asynchronous.
            consumers.shutdown().await;
        }
        result
    }
}

async fn consume_probe(output: CreditedOutputSession, maximum_output: usize) -> Result<u64> {
    let CreditedOutputSession {
        mut frames,
        completion,
    } = output;
    let mut terminal = false;
    while let Some(frame) = frames.next().await {
        if frame.wire().credit().events == 0
            || frame.wire().payload().capacity() > frame.wire().credit().bytes
        {
            return Err(invalid(
                "automatic cost probe output exceeded its original lease",
            ));
        }
        terminal = frame.metadata().terminal;
        drop(frame);
        if terminal {
            break;
        }
    }
    if !terminal {
        return Err(invalid(
            "automatic cost probe output ended without a terminal frame",
        ));
    }
    let completion = completion
        .await
        .map_err(|_| invalid("automatic cost probe lost its completion lease"))?;
    match completion.payload() {
        OutputCompletion::Succeeded { usage, .. } => {
            if usage.completion_tokens > maximum_output {
                return Err(invalid(
                    "automatic cost probe exceeded its declared output length",
                ));
            }
            u64::try_from(usage.completion_tokens)
                .map_err(|_| invalid("automatic cost probe output counter overflow"))
        }
        OutputCompletion::Failed(_) => Err(invalid("automatic cost probe request failed")),
    }
}

#[cfg(test)]
mod budget_tests;

#[cfg(test)]
mod native_reservation_tests {
    use super::*;
    fn budget() -> ProbeExecutionBudget {
        ProbeExecutionBudget::new(
            Instant::now() + Duration::from_secs(30),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        )
    }
    #[test]
    fn selected_native_actions_consume_actual_budget_exactly_once() {
        let mut budget = budget();
        budget.reserve_selected_source(2, 3).unwrap();
        assert_eq!(
            (
                budget.selection_requests_remaining(),
                budget.selection_attempts_remaining()
            ),
            (0, 0)
        );
        for _ in 0..2 {
            budget.claim_reserved_checkpoint_owner().unwrap();
        }
        for _ in 0..3 {
            budget.claim_reserved_checkpoint_action().unwrap();
        }
        assert_eq!(
            (budget.requests_remaining(), budget.attempts_remaining()),
            (0, 0)
        );
        assert_eq!(
            (
                budget.selection_requests_remaining(),
                budget.selection_attempts_remaining()
            ),
            (0, 0)
        );
        assert!(budget.claim_reserved_checkpoint_owner().is_err());
        assert!(budget.claim_reserved_checkpoint_action().is_err());
    }
    #[test]
    fn native_actual_only_api_rejects_unreserved_and_keeps_startup_double_accounting() {
        let mut budget = budget();
        assert!(budget.claim_reserved_checkpoint_owner().is_err());
        assert!(budget.claim_reserved_checkpoint_action().is_err());
        assert_eq!(
            (budget.requests_remaining(), budget.attempts_remaining()),
            (2, 3)
        );
        budget.claim_checkpoint_owner().unwrap();
        budget.claim_checkpoint_action().unwrap();
        assert_eq!(
            (
                budget.requests_remaining(),
                budget.selection_requests_remaining()
            ),
            (1, 1)
        );
        assert_eq!(
            (
                budget.attempts_remaining(),
                budget.selection_attempts_remaining()
            ),
            (2, 2)
        );
        assert!(budget.claim_reserved_checkpoint_owner().is_err());
        assert!(budget.claim_reserved_checkpoint_action().is_err());
    }
}
