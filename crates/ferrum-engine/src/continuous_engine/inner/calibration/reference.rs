//! Explicit discovery, frozen singleton trials, and original-source assembly.
//! This cold collector does not predict costs or grant permission to execute.
//! The decode unit freezes the actual route/output identity. Its opaque hash
//! is never interpreted as proof of a greedy or device-side sampling mode.
use super::*;
use ferrum_interfaces::execution_cost::{ActualRowWork, HostCostFeaturesV1};
use ferrum_scheduler::implementations::continuous::{
    cost_model::{CostBoundary, WaveCostObservation, WaveObservationOutcome},
    cost_profile::{self as profile, v2::ProfileWaveShapeV2},
    prefill_reference::*,
};
use ferrum_types::SloPrefillReferenceLimits;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    num::{NonZeroU32, NonZeroU64},
    path::{Path, PathBuf},
};

mod assemble;
mod evidence;
mod plan;
mod source;
mod trials;
pub use assemble::CalibrationReferenceArtifact;
pub use evidence::CalibrationReferenceDiscoverySample;
use evidence::Witness;
pub use plan::{CalibrationReferenceCurve, CalibrationReferencePlan};
pub use trials::CalibrationReferenceTrial;

pub struct CalibrationReferenceCollector {
    session: Arc<()>,
    plan: CalibrationReferencePlan,
    plan_sha256: [u8; 32],
    fingerprint: profile::ProfileFingerprint,
    frozen_accepted_ordinal: u64,
    frozen_at_ns: u64,
    discovery: Vec<CalibrationReferenceDiscoverySample>,
    trials: BTreeMap<CalibrationReferenceTrial, trials::Trial>,
    retained: usize,
}

impl CalibrationSession {
    /// Register only the live, unmodified owner initialized by add_request.
    /// A retained zero-prefix handle cannot stand in for a later restored or
    /// recomputed state, even when that state's visible prefix is zero again.
    pub fn begin_reference_trial(
        &mut self,
        collector: &mut CalibrationReferenceCollector,
        key: CalibrationReferenceTrial,
        frontier: &CalibrationFrontier,
    ) -> Result<()> {
        if self.pending.is_some()
            || self.indeterminate
            || !Arc::ptr_eq(&collector.session, &self.identity)
        {
            return Err(invalid(
                "reference registration requires this session with no pending wave",
            ));
        }
        let sequences = self.engine.inner.sequences.try_read().ok_or_else(|| {
            FerrumError::resource_exhausted("reference registration sequence view is busy")
        })?;
        let sequence = sequences
            .get(frontier.request_id())
            .ok_or_else(|| invalid("reference owner is no longer live"))?;
        let original = self
            .reference_origins
            .get(frontier.request_id())
            .ok_or_else(|| invalid("reference owner has no original session admission"))?;
        let current = sequence
            .cost_frontier
            .ok_or_else(|| invalid("reference owner frontier is unavailable"))?;
        if current != *original
            || current.owner_incarnation != frontier.owner_incarnation()
            || current.work_generation != frontier.work_generation()
            || sequence.prefill_complete
            || sequence.prefill_tokens_processed != 0
            || !sequence.generated_tokens.is_empty()
            || sequence
                .model_kv
                .as_ref()
                .is_some_and(|state| state.handle().num_tokens() != 0)
            || CalibrationRequestEvidence::capture(sequence) != *frontier.request_evidence()
        {
            return Err(invalid(
                "reference registration differs from fresh initialized owner/frontier",
            ));
        }
        collector.begin_trial(key, frontier)
    }

    /// Capture the actual discovery wave and its original tokenized input.
    /// Discovery is diagnostic and can never be reused as a reference trial.
    pub fn capture_reference_discovery(
        &self,
        before: &CalibrationFrontier,
        report: &CalibrationWaveReport,
    ) -> Result<CalibrationReferenceDiscoverySample> {
        if self.pending.is_some()
            || self.indeterminate
            || !Arc::ptr_eq(&before.session, &self.identity)
        {
            return Err(invalid(
                "discovery belongs to a different or unfinished session",
            ));
        }
        let witness = Witness::capture(report)?;
        if witness.commit.request_id != *before.request_id()
            || witness.commit.owner_incarnation != before.owner_incarnation().get()
            || witness.commit.work_generation != before.work_generation().get()
        {
            return Err(invalid(
                "discovery wave differs from its actual input frontier",
            ));
        }
        Ok(CalibrationReferenceDiscoverySample {
            session: Arc::clone(&self.identity),
            input: *before.request_evidence(),
            witness,
        })
    }

    /// Freeze the entire proposed protocol after bounded discovery and before
    /// any reference trial. A real queue barrier separates the two phases.
    pub async fn freeze_reference_plan(
        &mut self,
        plan: CalibrationReferencePlan,
        discovery: Vec<CalibrationReferenceDiscoverySample>,
    ) -> Result<CalibrationReferenceCollector> {
        if self.pending.is_some() || self.indeterminate {
            return Err(invalid(
                "reap calibration before freezing reference protocol",
            ));
        }
        let fingerprint = plan.validate_discovery(&self.identity, &discovery)?;
        let runtime = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| invalid("reference discovery requires cost runtime"))?;
        let checkpoint = runtime
            .request_checkpoint()
            .map_err(|error| FerrumError::resource_exhausted(format!("reference freeze: {error}")))?
            .wait()
            .await
            .map_err(|error| invalid(format!("reference freeze: {error}")))?;
        let frozen_at_ns = runtime
            .clock
            .now_ns()
            .ok_or_else(|| invalid("reference freeze clock is unavailable"))?;
        if discovery.iter().any(|row| {
            row.witness.accepted > checkpoint.accepted_ordinal
                || row.witness.sample.observed_at_ns > frozen_at_ns
        }) {
            return Err(invalid(
                "discovery is not before the actual freeze boundary",
            ));
        }
        let mut hash = Sha256::new();
        if plan.piecewise.is_some() {
            hash.update(b"ferrum.calibration-reference-plan.v2\0");
        } else {
            hash.update(b"ferrum.calibration-reference-plan.v1\0");
        }
        hash.update(serde_json::to_vec(&plan).map_err(json_error)?);
        for row in &discovery {
            hash.update(row.witness.accepted.to_le_bytes());
        }
        let retained = discovery.len();
        Ok(CalibrationReferenceCollector {
            session: Arc::clone(&self.identity),
            plan,
            plan_sha256: hash.finalize().into(),
            fingerprint,
            frozen_accepted_ordinal: checkpoint.accepted_ordinal,
            frozen_at_ns,
            discovery,
            trials: BTreeMap::new(),
            retained,
        })
    }
}

fn invalid(message: impl Into<String>) -> FerrumError {
    FerrumError::invalid_request(message)
}
fn json_error(error: serde_json::Error) -> FerrumError {
    invalid(format!("reference JSON: {error}"))
}
fn reference_error(error: ReferenceError) -> FerrumError {
    invalid(format!("reference evidence: {error}"))
}
