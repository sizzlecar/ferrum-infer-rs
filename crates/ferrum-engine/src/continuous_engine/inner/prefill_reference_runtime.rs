//! One immutable calibration per engine and request-local useful-work credit.
//! This state grants neither admission nor physical execution permission.
use super::*;
use ferrum_interfaces::execution_cost::{
    ExecutorCostIdentityAvailability, EXECUTOR_COST_IDENTITY_SCHEMA,
};
use ferrum_scheduler::implementations::continuous::{
    cost_model::ExecutionFingerprint,
    prefill_reference::{
        load_prefill_reference, LoadedPrefillReference, ReferenceError, ReferenceUnknown,
    },
};
use ferrum_types::SloPrefillReferenceConfig;
use std::num::{NonZeroU32, NonZeroU64};

mod binding;
pub(in crate::continuous_engine) use binding::*;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(in crate::continuous_engine) fn test_calibration_runtime() -> Arc<EnginePrefillReferenceRuntime>
{
    tests::calibration_runtime()
}

#[cfg(test)]
pub(in crate::continuous_engine) fn test_calibration_artifact(
) -> ferrum_scheduler::implementations::continuous::prefill_reference::ReferenceCalibrationV1 {
    tests::calibration_artifact()
}

#[cfg(test)]
pub(in crate::continuous_engine) fn test_piecewise_calibration_runtime(
) -> Arc<EnginePrefillReferenceRuntime> {
    tests::piecewise_runtime()
}

#[derive(Debug, thiserror::Error)]
pub(in crate::continuous_engine) enum PrefillReferenceLoadError {
    #[error("prefill reference configuration: {0}")]
    Configuration(String),
    #[error("configured prefill reference requires a known current executor identity")]
    IdentityUnavailable,
    #[error("prefill reference executor identity schema is unsupported")]
    IdentitySchema,
    #[error(transparent)]
    Artifact(#[from] ReferenceError),
}

#[derive(Debug)]
pub(in crate::continuous_engine) struct EnginePrefillReferenceRuntime {
    calibration: Arc<LoadedPrefillReference>,
    next_incarnation: AtomicU64,
}

impl EnginePrefillReferenceRuntime {
    /// Called once by the common run/serve engine constructor. An explicit
    /// invalid artifact fails construction even when no online model exists.
    pub fn load(
        config: Option<&SloPrefillReferenceConfig>,
        identity: impl FnOnce() -> ExecutorCostIdentityAvailability,
    ) -> std::result::Result<Option<Arc<Self>>, PrefillReferenceLoadError> {
        let Some(config) = config else {
            return Ok(None);
        };
        config
            .validate()
            .map_err(PrefillReferenceLoadError::Configuration)?;
        let ExecutorCostIdentityAvailability::Known(identity) = identity() else {
            return Err(PrefillReferenceLoadError::IdentityUnavailable);
        };
        if identity.schema_version != EXECUTOR_COST_IDENTITY_SCHEMA {
            return Err(PrefillReferenceLoadError::IdentitySchema);
        }
        let fingerprint = ExecutionFingerprint {
            model_weights: identity.model_weights,
            numerical_policy: identity.numerical_policy,
            device_runtime: identity.device_runtime,
            execution_config: identity.execution_config,
        };
        let calibration = load_prefill_reference(
            &config.artifact_path,
            &fingerprint,
            config.expected_protocol_sha256,
            &config.limits,
        )?;
        Ok(Some(Arc::new(Self {
            calibration,
            next_incarnation: AtomicU64::new(1),
        })))
    }

    pub fn calibration(&self) -> &Arc<LoadedPrefillReference> {
        &self.calibration
    }

    fn bind(
        &self,
        sequence: &SequenceState,
        admitted_at: Instant,
    ) -> std::result::Result<SequenceReferenceBinding, ReferenceBindingUnknown> {
        let incarnation = self
            .next_incarnation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .ok()
            .and_then(NonZeroU64::new)
            .ok_or(ReferenceBindingUnknown::IncarnationExhausted)?;
        let slo = sequence
            .slo
            .as_ref()
            .filter(|state| state.is_trusted())
            .ok_or(ReferenceBindingUnknown::UntrustedTiming)?;
        // This initialization is only a new admission, never a recompute or
        // restore hook. Existing sequences retain their original binding.
        if !sequence.generated_tokens.is_empty() || sequence.prefill_complete {
            return Err(ReferenceBindingUnknown::InvalidReceipt);
        }
        let total = u32::try_from(sequence.input_tokens.len())
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(ReferenceBindingUnknown::LengthOverflow)?;
        let prefix = u32::try_from(sequence.prefill_tokens_processed)
            .map_err(|_| ReferenceBindingUnknown::LengthOverflow)?;
        SequenceReferenceBinding::new(
            self.calibration.clone(),
            sequence.request_id.clone(),
            sequence.stream_projection_identity.clone(),
            incarnation,
            total,
            slo.ingress(),
            admitted_at,
            slo.first_deadline(),
            prefix,
        )
    }
}

impl EngineInner {
    /// Runs after scheduler submission succeeded and before releasing the
    /// admission iteration lock. It neither rejects work nor resets an owner.
    pub(in crate::continuous_engine) fn initialize_sequence_prefill_reference(
        &self,
        sequence: &mut SequenceState,
    ) {
        let Some(runtime) = &self.prefill_reference_runtime else {
            return;
        };
        if sequence.prefill_reference.is_some() {
            return;
        }
        sequence.prefill_reference = Some(match runtime.bind(sequence, slo_clock_now()) {
            Ok(binding) => SequencePrefillReference::Known(binding),
            Err(reason) => SequencePrefillReference::Unknown(reason),
        });
    }
}
