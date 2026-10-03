//! A private producer is acquired once and retired before any numerical
//! cohort. Its immutable lease is not a request, logits cache or fit sample.
use super::super::super::cohort_driver::{ProbePrefillPlan, ProbeRequest};
use super::super::super::token_preparation::PrefixFrontierV1;
use super::*;
use ferrum_interfaces::model_executor::{PrefixCaptureLease, PrefixCaptureStatus};
use ferrum_scheduler::implementations::continuous::cost_profile::{
    StructuredNativePrefixAcquisitionCohortV1, StructuredNativePrefixScopeV1,
};

/// Input-declared work. Native maintenance is separate from numerical offers.
/// The driver consumes these same boundary/chunk values for real execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct ProbePrefixAcquisitionPlan {
    prompt_tokens: NonZeroUsize,
    boundary: NonZeroUsize,
    prefill_chunk: NonZeroU32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct ProbePrefixWork {
    pub requests: usize,
    pub inference_waves: usize,
    pub native_captures: usize,
    pub native_restores: usize,
}

impl ProbePrefixWork {
    pub fn actions(self) -> Result<usize> {
        self.inference_waves
            .checked_add(self.native_captures)
            .and_then(|n| n.checked_add(self.native_restores))
            .ok_or_else(|| invalid("prefix acquisition action count overflow"))
    }
}

impl ProbePrefixAcquisitionPlan {
    pub fn new(prompt_tokens: usize, boundary: usize, prefill_chunk: NonZeroU32) -> Result<Self> {
        if boundary == 0 || boundary >= prompt_tokens {
            return Err(invalid(
                "prefix acquisition requires a proper nonempty prefix",
            ));
        }
        Ok(Self {
            prompt_tokens: NonZeroUsize::new(prompt_tokens).unwrap(),
            boundary: NonZeroUsize::new(boundary).unwrap(),
            prefill_chunk,
        })
    }

    pub fn prompt_tokens(self) -> usize {
        self.prompt_tokens.get()
    }
    pub fn prefill_chunk(self) -> NonZeroU32 {
        self.prefill_chunk
    }
    pub fn boundary(self) -> usize {
        self.boundary.get()
    }

    pub fn setup_work(self) -> ProbePrefixWork {
        ProbePrefixWork {
            requests: 1,
            inference_waves: self.boundary().div_ceil(self.prefill_chunk.get() as usize),
            native_captures: 1,
            native_restores: 0,
        }
    }

    /// This is the successful restore path. Selection must independently keep
    /// the cold fallback allowance unless failure cancels the declared source.
    pub fn restored_cohort_work(
        self,
        width: NonZeroUsize,
        maximum_output: NonZeroUsize,
        prefill_plan: ProbePrefillPlan,
    ) -> Result<ProbePrefixWork> {
        let chunks = (self.prompt_tokens.get() - self.boundary())
            .div_ceil(self.prefill_chunk.get() as usize);
        let prefills = prefill_plan
            .waves(chunks, width.get())
            .ok_or_else(|| invalid("restored prefix wave count overflow"))?;
        Ok(ProbePrefixWork {
            requests: width.get(),
            inference_waves: prefills
                .checked_add(maximum_output.get() - 1)
                .ok_or_else(|| invalid("restored prefix suffix wave count overflow"))?,
            native_captures: 0,
            native_restores: width.get(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) enum ProbePrefixFallback {
    Unsupported,
    NoProperBoundary,
    HostCapacity,
    InterestUnavailable,
    CaptureUnavailable,
    LeaseUnavailable,
    PromptMismatch,
    RestoreAdmissionUnavailable,
    RestoreUnavailable,
    RestoreDeferred,
}

/// The immutable native checkpoint remains owned by the executor's existing
/// physical checkpoint ledger. This host payload reports its own retained
/// allocation; it neither duplicates nor hides that native capacity lease.
pub(in crate::continuous_engine::inner::calibration) struct AcquiredProbePrefix {
    plan: ProbePrefixAcquisitionPlan,
    tokens: Arc<[TokenId]>,
    lease: Arc<dyn PrefixCaptureLease>,
    deadline: std::time::Instant,
    acknowledged_capture: bool,
    maximum_guard_owners: usize,
    captured_identity: NativeCheckpointTransferIdentity,
    clock: Arc<dyn ferrum_interfaces::execution_cost::CostObservationClock>,
    captured_at_ns: u64,
    expires_at_ns: u64,
    input_tokens_sha256: [u8; 32],
    charge: ProbePrefixCharge,
}

impl AcquiredProbePrefix {
    pub fn plan(&self) -> ProbePrefixAcquisitionPlan {
        self.plan
    }
    pub fn input_tokens_sha256(&self) -> [u8; 32] {
        self.input_tokens_sha256
    }
    pub fn ready(&self) -> bool {
        self.acknowledged_capture
            && std::time::Instant::now() < self.deadline
            && self.lease.purpose() == PrefixCapturePurpose::PrivateCalibration
            && self.lease.status() == PrefixCaptureStatus::Ready
            && self.lease.boundary() == self.plan.boundary()
    }
    pub fn retained_payload_bytes(&self) -> Result<usize> {
        Self::payload_bytes(self.tokens.len(), self.maximum_guard_owners)?
            .checked_add(self.captured_identity.plan_hash().as_str().len())
            .and_then(|n| n.checked_add(self.captured_identity.layout_fingerprint().len()))
            .and_then(|n| {
                n.checked_add(
                    self.captured_identity
                        .runtime_implementation_fingerprint()
                        .len(),
                )
            })
            .and_then(|n| n.checked_add(self.captured_identity.device_id().as_str().len()))
            .ok_or_else(|| invalid("prefix acquisition native identity payload overflow"))
    }
    pub fn declaration(&self) -> StructuredNativePrefixAcquisitionCohortV1 {
        StructuredNativePrefixAcquisitionCohortV1 {
            prompt_tokens: self.plan.prompt_tokens.get() as u64,
            boundary_tokens: self.plan.boundary() as u64,
            input_tokens_sha256: self.input_tokens_sha256,
            native_scope: native_scope(&self.captured_identity),
        }
    }
    pub fn prefill_chunk(&self) -> NonZeroU32 {
        self.plan.prefill_chunk
    }
    fn payload_bytes(tokens: usize, owners: usize) -> Result<usize> {
        let guards = owners
            .checked_mul(std::mem::size_of::<StartupOwner>())
            .and_then(|n| n.checked_add(std::mem::size_of::<StartupPrefixGuard>()))
            .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>()))
            .ok_or_else(|| invalid("prefix acquisition guard workspace overflow"))?;
        tokens
            .checked_mul(std::mem::size_of::<TokenId>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>()))
            .and_then(|n| n.checked_add(guards))
            .ok_or_else(|| invalid("prefix acquisition host payload overflow"))
    }
}

#[derive(Debug)]
pub(in crate::continuous_engine::inner::calibration) enum ProbePrefixAcquisition {
    Ready(AcquiredProbePrefix),
    ColdFallback(ProbePrefixFallback),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) enum ProbePrefixRestore {
    Acknowledged { restored_tokens: usize },
    ColdFallback(ProbePrefixFallback),
}

#[derive(Debug, Clone, Copy)]
enum ProbePrefixCharge {
    IndependentStartup,
    ReservedSource,
}
impl ProbePrefixCharge {
    fn owner(self, budget: &mut ProbeExecutionBudget) -> Result<()> {
        match self {
            Self::IndependentStartup => budget.claim_checkpoint_owner(),
            Self::ReservedSource => budget.claim_reserved_checkpoint_owner(),
        }
    }
    fn action(self, budget: &mut ProbeExecutionBudget) -> Result<()> {
        match self {
            Self::IndependentStartup => budget.claim_checkpoint_action(),
            Self::ReservedSource => budget.claim_reserved_checkpoint_action(),
        }
    }
}

impl CalibrationSession {
    /// No source collector is active while the private acquisition producer
    /// runs. Dropping its consumer follows original cancellation/settlement;
    /// only the actual immutable lease crosses completed_owner_boundary.
    pub(in crate::continuous_engine::inner::calibration) async fn acquire_probe_prefix(
        &mut self,
        request: ProbeRequest,
        declared: ProbePrefixAcquisitionPlan,
        budget: &mut ProbeExecutionBudget,
        maximum_retained_host_bytes: usize,
    ) -> Result<ProbePrefixAcquisition> {
        self.acquire_probe_prefix_with_charge(
            request,
            declared,
            budget,
            maximum_retained_host_bytes,
            ProbePrefixCharge::IndependentStartup,
        )
        .await
    }

    pub(in crate::continuous_engine::inner::calibration) async fn acquire_reserved_probe_prefix(
        &mut self,
        request: ProbeRequest,
        declared: ProbePrefixAcquisitionPlan,
        budget: &mut ProbeExecutionBudget,
        maximum_retained_host_bytes: usize,
    ) -> Result<ProbePrefixAcquisition> {
        if !self.startup_inventory_active() {
            return Err(invalid(
                "reserved native acquisition requires original startup inventory authority",
            ));
        }
        self.acquire_probe_prefix_with_charge(
            request,
            declared,
            budget,
            maximum_retained_host_bytes,
            ProbePrefixCharge::ReservedSource,
        )
        .await
    }

    async fn acquire_probe_prefix_with_charge(
        &mut self,
        request: ProbeRequest,
        declared: ProbePrefixAcquisitionPlan,
        budget: &mut ProbeExecutionBudget,
        maximum_retained_host_bytes: usize,
        charge: ProbePrefixCharge,
    ) -> Result<ProbePrefixAcquisition> {
        self.completed_owner_boundary()?;
        let inner = &self.engine.inner;
        if !inner.manual_calibration_driver
            || !inner.automatic_reference_bootstrap
            || inner.bg_loop_spawned.load(Ordering::Acquire)
            || inner.is_running.load(Ordering::Acquire)
            || inner.shutdown_started.load(Ordering::Acquire)
            || self.prepared_owner_capture.is_some()
            || self.prefix_source5
            || (self.prefix_source8 && !matches!(charge, ProbePrefixCharge::ReservedSource))
            || (matches!(charge, ProbePrefixCharge::ReservedSource)
                && !self.startup_inventory_active())
        {
            return Err(invalid(
                "prefix acquisition requires an unused automatic session",
            ));
        }
        if !inner
            .model_executor
            .supports_guarded_prefix_maintenance_for(PrefixCapturePurpose::PrivateCalibration)
        {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::Unsupported,
            ));
        }
        charge.owner(budget)?;
        let source = request.request.id.clone();
        let output = self
            .add_request(
                request.request,
                InferenceRequestContext::capture(),
                request.contract,
            )
            .await?;
        self.startup_checkpoint_sampling = true;
        let result = self
            .acquire_admitted_probe_prefix(
                &source,
                declared,
                budget,
                maximum_retained_host_bytes,
                charge,
            )
            .await;
        self.startup_checkpoint_sampling = false;
        drop(output);
        // No timer drops submitted work, output settlement or native leases.
        self.drain_startup_geometry().await?;
        self.completed_owner_boundary()?;
        result
    }

    async fn acquire_admitted_probe_prefix(
        &mut self,
        source: &RequestId,
        declared: ProbePrefixAcquisitionPlan,
        budget: &mut ProbeExecutionBudget,
        maximum_retained_host_bytes: usize,
        charge: ProbePrefixCharge,
    ) -> Result<ProbePrefixAcquisition> {
        self.admit_startup_checkpoint_owners(budget).await?;
        let (tokens, maximum_sequence_tokens) = {
            let sequences = self.engine.inner.sequences.read();
            let sequence = sequences
                .get(source)
                .ok_or_else(|| invalid("prefix acquisition owner disappeared"))?;
            // Check the peak before cloning immutable tokens. Arc conversion
            // temporarily retains both the Vec and Arc payloads.
            let retained = AcquiredProbePrefix::payload_bytes(
                sequence.input_tokens.len(),
                self.limits.maximum_requests().get(),
            )?;
            let peak = sequence
                .input_tokens
                .len()
                .checked_mul(std::mem::size_of::<TokenId>())
                .and_then(|n| n.checked_add(retained))
                .ok_or_else(|| invalid("prefix acquisition host peak overflow"))?;
            if peak > maximum_retained_host_bytes {
                return Ok(ProbePrefixAcquisition::ColdFallback(
                    ProbePrefixFallback::HostCapacity,
                ));
            }
            (
                Arc::<[TokenId]>::from(sequence.input_tokens.as_slice()),
                sequence.model_maximum_sequence_tokens(),
            )
        };
        let Some(common) = tokens.len().checked_sub(1).filter(|n| *n > 0) else {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::NoProperBoundary,
            ));
        };
        let Some(boundary) = self
            .engine
            .inner
            .model_executor
            .plan_prefix_capture_boundary_for(
                PrefixCapturePurpose::PrivateCalibration,
                PrefixCaptureBoundary {
                    processed_tokens: 0,
                    source_prompt_tokens: tokens.len(),
                    common_prefix_tokens: common,
                    follower_prompt_tokens: &[tokens.len()],
                },
            )
        else {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::NoProperBoundary,
            ));
        };
        if boundary.boundary > common {
            return Err(invalid(
                "executor acquisition boundary exceeds the requested proper prefix",
            ));
        }
        if boundary.boundary != declared.boundary() || tokens.len() != declared.prompt_tokens.get()
        {
            return Err(invalid(
                "actual model-declared acquisition differs from the frozen input plan",
            ));
        }
        let last_span = (declared.boundary() - 1) % declared.prefill_chunk.get() as usize + 1;
        if !boundary.span.permits(last_span as u64) {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::NoProperBoundary,
            ));
        }
        let plan = declared;
        let mut blocked_attempts = 0usize;
        let mut last_blocked_reason = None;
        loop {
            budget.require_time()?;
            let frontier = self
                .frontiers()?
                .into_iter()
                .find(|f| f.request_id() == source)
                .ok_or_else(|| invalid("prefix acquisition frontier disappeared"))?;
            let (offset, _) = frontier
                .prefill_progress()
                .ok_or_else(|| invalid("prefix acquisition unexpectedly decoded"))?;
            if offset == plan.boundary() {
                break;
            }
            let count = plan
                .boundary()
                .checked_sub(offset)
                .filter(|n| *n > 0)
                .and_then(|n| u32::try_from(n.min(plan.prefill_chunk.get() as usize)).ok())
                .and_then(NonZeroU32::new)
                .ok_or_else(|| {
                    invalid("prefix acquisition progress crossed its declared boundary")
                })?;
            tokio::time::timeout_at(budget.deadline(), self.startup_output_ready(source))
                .await
                .map_err(|_| invalid("prefix acquisition output readiness expired"))??;
            if let Err(error) = charge.action(budget) {
                tracing::warn!(
                    %error,
                    source_owner = %source,
                    ?plan,
                    offset,
                    blocked_attempts,
                    ?last_blocked_reason,
                    actual_requests_remaining = budget.requests_remaining(),
                    selection_requests_remaining = budget.selection_requests_remaining(),
                    actual_attempts_remaining = budget.attempts_remaining(),
                    selection_attempts_remaining = budget.selection_attempts_remaining(),
                    "Private prefix prefill action charge rejected"
                );
                return Err(error);
            }
            match self
                .step(CalibrationAction::Wave(vec![frontier.prefill_work(count)?]))
                .await?
            {
                CalibrationTurn::Wave(report) | CalibrationTurn::Reaped(report) => {
                    if let Some(error) = report.error {
                        return Err(error);
                    }
                    if report.submission == CalibrationSubmissionState::InFlightUnknown {
                        return Err(invalid("prefix acquisition submission is indeterminate"));
                    }
                }
                CalibrationTurn::Blocked(reason) => {
                    blocked_attempts = blocked_attempts.saturating_add(1);
                    last_blocked_reason = Some(reason);
                    tokio::task::yield_now().await;
                }
                _ => return Err(invalid("prefix acquisition returned an unexpected turn")),
            }
        }
        // Arm at the completed boundary, immediately before the one explicit
        // guarded capture; acquisition prefill cannot issue an interest-driven
        // second capture. Arming alone allocates/submits no checkpoint.
        let expires_at = budget.deadline().into_std();
        let clock = Arc::clone(
            &self
                .engine
                .inner
                .cost_runtime
                .as_ref()
                .ok_or_else(|| invalid("native acquisition cost runtime missing"))?
                .clock,
        );
        let capture_started_at_ns = clock
            .now_ns()
            .ok_or_else(|| invalid("native acquisition original clock missing"))?;
        let remaining = expires_at
            .checked_duration_since(std::time::Instant::now())
            .ok_or_else(|| invalid("native acquisition original deadline expired"))?;
        let expires_at_ns = capture_started_at_ns
            .checked_add(
                u64::try_from(remaining.as_nanos())
                    .map_err(|_| invalid("native acquisition expiry overflow"))?,
            )
            .ok_or_else(|| invalid("native acquisition expiry overflow"))?;
        let capture = || PrefixCaptureRequest {
            purpose: PrefixCapturePurpose::PrivateCalibration,
            source_request_id: source,
            source_tokens: &tokens,
            maximum_sequence_tokens,
            boundary: plan.boundary(),
            expires_at,
        };
        let Some(lease) = self
            .engine
            .inner
            .model_executor
            .retain_prefix_capture_interest(capture())?
        else {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::InterestUnavailable,
            ));
        };
        if lease.purpose() != PrefixCapturePurpose::PrivateCalibration {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::InterestUnavailable,
            ));
        }
        let guard = self.probe_prefix_guard(source, false, expires_at, None)?;
        charge.action(budget)?;
        if !self
            .engine
            .inner
            .model_executor
            .try_capture_plan_runtime_prefix_guarded(capture(), guard.clone())
            .await?
        {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::CaptureUnavailable,
            ));
        }
        if lease.purpose() != PrefixCapturePurpose::PrivateCalibration
            || lease.status() != PrefixCaptureStatus::Ready
            || lease.boundary() != plan.boundary()
        {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::LeaseUnavailable,
            ));
        }
        let captured_identity = guard.observed_transfer.lock().clone().ok_or_else(|| {
            invalid("acknowledged native capture has no original transfer identity")
        })?;
        let captured_at_ns = clock
            .now_ns()
            .filter(|n| *n >= capture_started_at_ns && *n < expires_at_ns)
            .ok_or_else(|| invalid("acknowledged native capture clock expired"))?;
        let input_tokens_sha256 = crate::continuous_engine::token_ids_digest(&tokens).into();
        let acquired = AcquiredProbePrefix {
            plan,
            tokens,
            lease,
            deadline: budget.deadline().into_std(),
            acknowledged_capture: true,
            maximum_guard_owners: self.limits.maximum_requests().get(),
            captured_identity,
            clock,
            captured_at_ns,
            expires_at_ns,
            input_tokens_sha256,
            charge,
        };
        if acquired.retained_payload_bytes()? > maximum_retained_host_bytes {
            return Ok(ProbePrefixAcquisition::ColdFallback(
                ProbePrefixFallback::HostCapacity,
            ));
        }
        Ok(ProbePrefixAcquisition::Ready(acquired))
    }

    fn probe_prefix_guard(
        &self,
        action_owner: &RequestId,
        restore: bool,
        deadline: std::time::Instant,
        expected_source_capture: Option<NativeCheckpointTransferIdentity>,
    ) -> Result<Arc<StartupPrefixGuard>> {
        let sequences = self.engine.inner.sequences.read();
        if sequences.is_empty() || sequences.len() > self.limits.maximum_requests().get() {
            return Err(invalid(
                "prefix acquisition guard has an invalid private cohort",
            ));
        }
        let mut owners = Vec::with_capacity(sequences.len());
        let mut selected = None;
        for (id, sequence) in sequences.iter() {
            if id == action_owner {
                selected = Some(owners.len());
            }
            owners.push(StartupOwner {
                id: id.clone(),
                identity: Arc::clone(&sequence.stream_projection_identity),
                frontier: sequence
                    .cost_frontier
                    .ok_or_else(|| invalid("prefix acquisition guard frontier missing"))?,
                offset: sequence.prefill_tokens_processed,
            });
        }
        Ok(Arc::new(StartupPrefixGuard {
            engine: Arc::downgrade(&self.engine.inner),
            owners,
            action_owner: selected
                .ok_or_else(|| invalid("prefix acquisition action owner missing"))?,
            deadline,
            restore,
            expected_source_capture,
            observed_transfer: parking_lot::Mutex::new(None),
        }))
    }

    /// Called after original source8 admission and before any output/prefill
    /// wave. A successful return follows scheduler+model+native acknowledgement.
    pub(in crate::continuous_engine::inner::calibration) async fn restore_acquired_probe_prefix(
        &mut self,
        target: &RequestId,
        acquired: &AcquiredProbePrefix,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<ProbePrefixRestore> {
        budget.require_time()?;
        if self.limits.maximum_requests().get() > acquired.maximum_guard_owners {
            return Err(invalid(
                "prefix acquisition guard capacity changed after capture",
            ));
        }
        if !acquired.ready() {
            return Ok(ProbePrefixRestore::ColdFallback(
                ProbePrefixFallback::LeaseUnavailable,
            ));
        }
        let runtime = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| invalid("native restore cost runtime missing"))?;
        if !Arc::ptr_eq(&runtime.clock, &acquired.clock) {
            return Err(invalid("native acquisition original clock domain changed"));
        }
        let (tokens, maximum_sequence_tokens, before) = {
            let sequences = self.engine.inner.sequences.read();
            let sequence = sequences
                .get(target)
                .ok_or_else(|| invalid("prefix restore target disappeared"))?;
            if sequence.prefill_tokens_processed != 0
                || sequence.prefill_complete
                || !sequence.generated_tokens.is_empty()
            {
                return Err(invalid(
                    "acquired prefix restore requires a fresh admitted owner",
                ));
            }
            if sequence.input_tokens.as_slice() != acquired.tokens.as_ref() {
                return Ok(ProbePrefixRestore::ColdFallback(
                    ProbePrefixFallback::PromptMismatch,
                ));
            }
            // A proper-prefix checkpoint contains no sampled output. Restore
            // validates this target's own admission and physical capacity; its
            // output allowance need not equal the retired producer's allowance.
            (
                Arc::clone(&acquired.tokens),
                sequence.model_maximum_sequence_tokens(),
                PrefixFrontierV1::capture(sequence)?,
            )
        };
        let Some(prepared) =
            self.engine
                .inner
                .scheduler
                .prepare_prefix_restore(target, 0, tokens.len())?
        else {
            return Ok(ProbePrefixRestore::ColdFallback(
                ProbePrefixFallback::RestoreAdmissionUnavailable,
            ));
        };
        // The original source block owns the FIFO cut and opening clock. Open
        // it before this maintenance action; doing so after ACK would move the
        // opening clock past the evidence it is meant to validate.
        self.prepared_owner_capture
            .as_mut()
            .ok_or_else(|| invalid("native restore has no original source8 collector"))?
            .prepare_native_prefix_restore()?;
        let guard = self.probe_prefix_guard(
            target,
            true,
            budget.deadline().into_std().min(acquired.deadline),
            Some(acquired.captured_identity.clone()),
        )?;
        acquired.charge.action(budget)?;
        match self
            .engine
            .inner
            .model_executor
            .try_restore_plan_runtime_prefix_guarded(
                PlanRuntimePrefixRestoreInput {
                    request_id: target,
                    input_tokens: &tokens,
                    maximum_sequence_tokens,
                    checkpoint: Some(acquired.lease.as_ref()),
                    retry: None,
                },
                guard.clone(),
            )
            .await?
        {
            PlanRuntimePrefixRestoreOutcome::Restored(output) => {
                if output.restored_tokens() != acquired.plan.boundary() {
                    // Physical publication cannot be silently treated as a cold
                    // miss. Its exact acknowledgement/retirement must be owned.
                    return Err(invalid(
                        "native prefix restored a different acquisition boundary",
                    ));
                }
                self.engine
                    .inner
                    .commit_prefix_restore_output(target, prepared, &tokens, output)?;
                let after =
                    {
                        let sequences = self.engine.inner.sequences.read();
                        PrefixFrontierV1::capture(sequences.get(target).ok_or_else(|| {
                            invalid("acknowledged native restore target missing")
                        })?)?
                    };
                let restore_identity = guard.observed_transfer.lock().clone().ok_or_else(|| {
                    invalid("acknowledged native restore transfer identity missing")
                })?;
                let acknowledged_at_ns = acquired
                    .clock
                    .now_ns()
                    .filter(|n| *n >= acquired.captured_at_ns && *n < acquired.expires_at_ns)
                    .ok_or_else(|| invalid("native restore ACK original clock expired"))?;
                let receipt = AcknowledgedProbePrefixRestore {
                    before,
                    after,
                    input_tokens_sha256: acquired.input_tokens_sha256,
                    capture: NativeTransferMetadata::from_identity(&acquired.captured_identity),
                    restore: NativeTransferMetadata::from_identity(&restore_identity),
                    captured_at_ns: acquired.captured_at_ns,
                    acknowledged_at_ns,
                    expires_at_ns: acquired.expires_at_ns,
                    acknowledged: true,
                    maintenance_fifo: self
                        .engine
                        .inner
                        .cost_runtime
                        .as_ref()
                        .ok_or_else(|| invalid("native restore cost runtime disappeared"))?
                        .take_prefix_fifo_receipt(&restore_identity),
                };
                self.accept_prepared_owner_native_restore(&receipt)?;
                Ok(ProbePrefixRestore::Acknowledged {
                    restored_tokens: acquired.plan.boundary(),
                })
            }
            PlanRuntimePrefixRestoreOutcome::Unavailable => Ok(ProbePrefixRestore::ColdFallback(
                ProbePrefixFallback::RestoreUnavailable,
            )),
            PlanRuntimePrefixRestoreOutcome::Deferred(_) => Ok(ProbePrefixRestore::ColdFallback(
                ProbePrefixFallback::RestoreDeferred,
            )),
        }
    }
}

fn native_scope(identity: &NativeCheckpointTransferIdentity) -> StructuredNativePrefixScopeV1 {
    StructuredNativePrefixScopeV1 {
        plan_hash: identity.plan_hash().as_str().to_owned(),
        layout_fingerprint: identity.layout_fingerprint().to_owned(),
        runtime_implementation_fingerprint: identity
            .runtime_implementation_fingerprint()
            .to_owned(),
        device_id: identity.device_id().as_str().to_owned(),
    }
}
#[derive(serde::Serialize)]
struct NativeTransferMetadata {
    slot: u64,
    checkpoint_coordinator: u64,
    checkpoint_serial: u64,
    sequence_sparse: u32,
    sequence_generation: u64,
    request_sparse: u32,
    request_generation: u64,
    boundary_tokens: u64,
    kind: &'static str,
    plan_hash: String,
    layout_fingerprint: String,
    runtime_implementation_fingerprint: String,
    device_id: String,
}
impl NativeTransferMetadata {
    fn from_identity(identity: &NativeCheckpointTransferIdentity) -> Self {
        Self {
            slot: identity.slot_id().get(),
            checkpoint_coordinator: identity.checkpoint_authority().coordinator_id().get(),
            checkpoint_serial: identity.checkpoint_authority().serial(),
            sequence_sparse: identity.sequence_authority().sparse_id(),
            sequence_generation: identity.sequence_authority().generation(),
            request_sparse: identity.request_authority().sparse_id(),
            request_generation: identity.request_authority().generation(),
            boundary_tokens: identity.boundary_tokens(),
            kind: match identity.kind() {
                NativeCheckpointTransferKind::Capture => "capture",
                NativeCheckpointTransferKind::Restore => "restore",
            },
            plan_hash: identity.plan_hash().as_str().to_owned(),
            layout_fingerprint: identity.layout_fingerprint().to_owned(),
            runtime_implementation_fingerprint: identity
                .runtime_implementation_fingerprint()
                .to_owned(),
            device_id: identity.device_id().as_str().to_owned(),
        }
    }
    fn matches_scope(&self, scope: &StructuredNativePrefixScopeV1) -> bool {
        self.plan_hash == scope.plan_hash
            && self.layout_fingerprint == scope.layout_fingerprint
            && self.runtime_implementation_fingerprint == scope.runtime_implementation_fingerprint
            && self.device_id == scope.device_id
    }
    fn heap_bytes(&self) -> Option<usize> {
        self.plan_hash
            .capacity()
            .checked_add(self.layout_fingerprint.capacity())?
            .checked_add(self.runtime_implementation_fingerprint.capacity())?
            .checked_add(self.device_id.capacity())
    }
}
/// Only the successful physical restore + commit_prefix_restore_output ACK
/// branch constructs this value. The source adapter accepts no diagnostic DTO.
#[derive(serde::Serialize)]
pub(in crate::continuous_engine::inner) struct AcknowledgedProbePrefixRestore {
    before: PrefixFrontierV1,
    after: PrefixFrontierV1,
    input_tokens_sha256: [u8; 32],
    capture: NativeTransferMetadata,
    restore: NativeTransferMetadata,
    captured_at_ns: u64,
    acknowledged_at_ns: u64,
    expires_at_ns: u64,
    acknowledged: bool,
    maintenance_fifo: Option<u64>,
}
impl AcknowledgedProbePrefixRestore {
    pub fn before(&self) -> &PrefixFrontierV1 {
        &self.before
    }
    pub fn maintenance_fifo(&self) -> Option<u64> {
        self.maintenance_fifo
    }
    pub fn matches_declaration(&self, plan: &StructuredNativePrefixAcquisitionCohortV1) -> bool {
        self.input_tokens_sha256 == plan.input_tokens_sha256
            && self.after.kv_tokens as u64 == plan.boundary_tokens
            && self.capture.matches_scope(&plan.native_scope)
            && self.restore.matches_scope(&plan.native_scope)
    }
    pub fn retained_payload_bytes(&self) -> Result<usize> {
        let mut bytes = std::mem::size_of::<Self>();
        for frontier in [&self.before, &self.after] {
            bytes = bytes
                .checked_add(frontier.request_id.to_string().len())
                .and_then(|n| n.checked_add(frontier.pending_utf8.capacity()))
                .and_then(|n| {
                    n.checked_add(frontier.model_cache_id.as_ref().map_or(0, String::capacity))
                })
                .ok_or_else(|| invalid("native ACK frontier payload overflow"))?;
        }
        bytes
            .checked_add(
                self.capture
                    .heap_bytes()
                    .ok_or_else(|| invalid("native capture payload overflow"))?,
            )
            .and_then(|n| n.checked_add(self.restore.heap_bytes()?))
            // JSON serializer retains a second copy plus bounded field/value
            // nodes while the original hashed collector handles this record.
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("native ACK record payload overflow"))
    }
    pub fn event(
        &self,
        phase: ferrum_scheduler::implementations::continuous::cost_profile::StructuredProfilePhaseV10,
        cohort: usize,
        slot: usize,
    ) -> Result<serde_json::Value> {
        let mut value = serde_json::to_value(self).map_err(|e| invalid(e.to_string()))?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| invalid("native ACK record is not an object"))?;
        object.insert("kind".into(), "native_prefix_restored".into());
        object.insert(
            "phase".into(),
            serde_json::to_value(phase).map_err(|e| invalid(e.to_string()))?,
        );
        object.insert("cohort".into(), cohort.into());
        object.insert("slot".into(), slot.into());
        Ok(value)
    }
}
impl std::fmt::Debug for AcquiredProbePrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcquiredProbePrefix")
            .field("plan", &self.plan)
            .field("captured_identity", &self.captured_identity)
            .field("deadline", &self.deadline)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acquisition_plan_separates_native_actions_from_real_numerical_waves() {
        let plan = ProbePrefixAcquisitionPlan::new(70, 69, NonZeroU32::new(32).unwrap()).unwrap();
        assert_eq!(
            plan.setup_work(),
            ProbePrefixWork {
                requests: 1,
                inference_waves: 3,
                native_captures: 1,
                native_restores: 0
            }
        );
        let cohort = plan
            .restored_cohort_work(
                NonZeroUsize::new(2).unwrap(),
                NonZeroUsize::new(3).unwrap(),
                ProbePrefillPlan::PreparedSequentialV1,
            )
            .unwrap();
        assert_eq!(
            cohort,
            ProbePrefixWork {
                requests: 2,
                inference_waves: 4,
                native_captures: 0,
                native_restores: 2
            }
        );
        assert_eq!(cohort.actions().unwrap(), 6);
        assert!(ProbePrefixAcquisitionPlan::new(1, 1, NonZeroU32::MIN).is_err());
        assert!(ProbePrefixAcquisitionPlan::new(3, 0, NonZeroU32::MIN).is_err());
    }
}
