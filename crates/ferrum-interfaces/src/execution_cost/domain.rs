//! Finite workload configuration for an empirical cost model. This descriptor
//! and its public bounds checks confer no execution or projection authority.
//! A caller must first validate the original actual/future recipe and its live
//! resource, provider, policy and execution-identity binding.
use super::{
    project_host_cost_features, ActualRowWork, CanonicalCostRow, CanonicalWaveCostShape,
    CostIdentityUnknownReason, CostRowNumericFeatures, CostRowOutput, ExecutorCostIdentity,
    HostContentDomainV1, HostRowRoleV2, HostTerminalExpectationV1, StructuredHostRowV1,
    COST_NUMERIC_FEATURE_SCHEMA_V1, EXECUTOR_COST_IDENTITY_SCHEMA,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    num::{NonZeroU32, NonZeroU64},
    sync::Arc,
};

pub const COST_WORKLOAD_DOMAIN_SCHEMA_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostWorkloadProjectionV1 {
    /// Image of finite legal input ranges under the already bound VNext CPU
    /// recipe. A descriptor cannot grant support to an unknown provider route.
    ValidatedVNextRecipeV1,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostWorkloadProjectionV1 {
    /// Per-owner installed policy and decoder bounds stay in the original
    /// recipe. This tag does not authorize tool, grammar or hidden output work.
    InstalledPlainTextV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostWorkloadLimitsV1 {
    pub maximum_rows: NonZeroU32,
    pub maximum_context_tokens: NonZeroU32,
    /// Original scheduler units: one per decode row, `count` per prefill row;
    /// a mixed wave uses their checked sum. This is not a user output limit.
    pub maximum_scheduled_tokens_per_wave: NonZeroU64,
    pub output_vocabulary_elements: NonZeroU64,
    pub repetition_slot_capacity: u64,
    pub fixed_state_bytes_per_row: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityWire {
    schema_version: u32,
    model_weights: [u8; 32],
    numerical_policy: [u8; 32],
    device_runtime: [u8; 32],
    execution_config: [u8; 32],
}
impl From<&ExecutorCostIdentity> for IdentityWire {
    fn from(v: &ExecutorCostIdentity) -> Self {
        Self {
            schema_version: v.schema_version,
            model_weights: v.model_weights,
            numerical_policy: v.numerical_policy,
            device_runtime: v.device_runtime,
            execution_config: v.execution_config,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DomainWire {
    schema_version: u32,
    projection: CostWorkloadProjectionV1,
    host_projection: HostWorkloadProjectionV1,
    identity: IdentityWire,
    limits: CostWorkloadLimitsV1,
}

/// Checked typed metadata. Deserialization goes through the same validator as
/// cold construction. The digest is recomputed from canonical typed fields;
/// a free wire digest is never accepted as evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "DomainWire", into = "DomainWire")]
pub struct CostWorkloadDomainV1 {
    wire: DomainWire,
    digest: [u8; 32],
}
impl CostWorkloadDomainV1 {
    pub fn new_vnext(
        identity: &ExecutorCostIdentity,
        limits: CostWorkloadLimitsV1,
    ) -> Result<Self, CostWorkloadDomainError> {
        DomainWire {
            schema_version: COST_WORKLOAD_DOMAIN_SCHEMA_V1,
            projection: CostWorkloadProjectionV1::ValidatedVNextRecipeV1,
            host_projection: HostWorkloadProjectionV1::InstalledPlainTextV1,
            identity: identity.into(),
            limits,
        }
        .try_into()
    }
    pub fn limits(&self) -> &CostWorkloadLimitsV1 {
        &self.wire.limits
    }
    pub fn projection(&self) -> CostWorkloadProjectionV1 {
        self.wire.projection
    }
    pub fn host_projection(&self) -> HostWorkloadProjectionV1 {
        self.wire.host_projection
    }
    pub fn sha256(&self) -> &[u8; 32] {
        &self.digest
    }
    pub fn matches_execution_identity(&self, identity: &ExecutorCostIdentity) -> bool {
        self.wire.identity == IdentityWire::from(identity)
    }
    /// Runtime activation compares the complete cold descriptor, including its
    /// real limits. Matching only the execution identity is insufficient when
    /// reading an externally supplied declaration.
    pub fn matches_runtime_domain(&self, runtime: &Self) -> bool {
        self.wire == runtime.wire
    }

    /// Allocation-free scalar check after the original recipe validation.
    /// Public rows are not sealed evidence: success means only that these
    /// values satisfy this descriptor's limits, never that they can execute.
    /// Device axes remain the output of the original checked provider recipe.
    pub fn validate_workload_rows(
        &self,
        rows: &[CanonicalCostRow],
        recurrent_state_bytes: u64,
    ) -> Result<(), CostWorkloadDomainError> {
        self.validate_row_capacity(rows.len(), recurrent_state_bytes)?;
        let mut scheduled = 0u64;
        for row in rows {
            let final_prefill = match row.output {
                CostRowOutput::Prefill { final_logits } => Some(final_logits),
                CostRowOutput::Decode {
                    repetition_tokens,
                    repetition_penalty_bits,
                    ..
                } => {
                    if !matches!(row.work, ActualRowWork::Decode { .. }) {
                        return Err(CostWorkloadDomainError::UnsupportedWork);
                    }
                    self.validate_repetition(repetition_tokens, repetition_penalty_bits)?;
                    None
                }
            };
            self.add_scheduled_work(&mut scheduled, row.work, final_prefill)?;
            let host = row
                .host_features
                .filter(|v| v.supports_installed_plain_text_content())
                .ok_or(CostWorkloadDomainError::UnsupportedHostPolicy)?;
            let numeric = project_host_cost_features(host, row.work, row.output)
                .map_err(|_| CostWorkloadDomainError::InvalidHostWork)?;
            self.validate_host_limits(&numeric)?;
        }
        Ok(())
    }

    /// Checks the finite domain of an already validated actual/future recipe.
    /// The caller must first validate the original prepared shape and recipe
    /// binding. Public projected rows are not execution or projection authority.
    /// This reads the original numeric and physical host rows without allocating,
    /// synthesizing host state, rebuilding signatures, or selecting a provider.
    pub fn validate_projected_workload(
        &self,
        canonical: &CanonicalWaveCostShape,
        physical_host_rows: &[StructuredHostRowV1],
    ) -> Result<(), CostWorkloadDomainError> {
        self.validate_row_capacity(canonical.rows.len(), canonical.recurrent_state_bytes)?;
        let numeric = canonical
            .numeric_features
            .as_ref()
            .ok_or(CostWorkloadDomainError::InvalidHostWork)?;
        if numeric.schema_version != COST_NUMERIC_FEATURE_SCHEMA_V1 {
            return Err(CostWorkloadDomainError::UnsupportedSchema);
        }
        if physical_host_rows.len() != canonical.rows.len()
            || numeric.rows.len() != canonical.rows.len()
        {
            return Err(CostWorkloadDomainError::RowCapacity);
        }
        let mut scheduled = 0u64;
        for (position, ((work, numeric), host)) in canonical
            .rows
            .iter()
            .zip(&numeric.rows)
            .zip(physical_host_rows)
            .enumerate()
        {
            if usize::try_from(host.physical_position).ok() != Some(position) {
                return Err(CostWorkloadDomainError::InvalidHostWork);
            }
            if !matches!(
                host.installed_policy.empirical_content_domain,
                Some(HostContentDomainV1::PlainTextGreedyV1)
                    | Some(HostContentDomainV1::PlainTextInstalledV2(_))
            ) {
                return Err(CostWorkloadDomainError::UnsupportedHostPolicy);
            }
            let (final_prefill, emits_token) = match (work, host.role) {
                (ActualRowWork::Decode { .. }, HostRowRoleV2::Decode)
                    if !host.initial_prefill
                        && !host.final_prefill
                        && host.decode_requires_full_logits.is_some() =>
                {
                    self.validate_repetition(
                        numeric.repetition_tokens,
                        host.repetition_penalty_bits
                            .ok_or(CostWorkloadDomainError::InvalidHostWork)?,
                    )?;
                    (None, true)
                }
                (ActualRowWork::Prefill { offset, .. }, HostRowRoleV2::Prefill)
                    if host.initial_prefill == (*offset == 0)
                        && host.decode_requires_full_logits.is_none()
                        && host.repetition_penalty_bits.is_none()
                        && numeric.repetition_tokens == 0 =>
                {
                    (Some(host.final_prefill), host.final_prefill)
                }
                _ => return Err(CostWorkloadDomainError::InvalidHostWork),
            };
            self.add_scheduled_work(&mut scheduled, *work, final_prefill)?;
            self.validate_host_limits(numeric)?;
            let prefix = numeric
                .generated_tokens_before
                .checked_add(u64::from(emits_token))
                .ok_or(CostWorkloadDomainError::ArithmeticOverflow)?;
            let terminal = if !emits_token {
                HostTerminalExpectationV1::NoTokenProduced
            } else if prefix == numeric.maximum_output_tokens {
                HostTerminalExpectationV1::LengthBoundary
            } else {
                HostTerminalExpectationV1::TokenMayTerminate
            };
            if numeric.decoded_prefix_tokens != prefix
                || numeric.sampling_history_tokens != numeric.generated_tokens_before
                || host.no_generated_history != (numeric.generated_tokens_before == 0)
                || host.terminal_expectation != terminal
                || host.installed_policy.decoder_text_bytes_per_token == 0
                || host.installed_policy.raw_token_bytes_bound == 0
                || host
                    .installed_policy
                    .decoder_text_bytes_per_token
                    .checked_mul(prefix)
                    != Some(numeric.decoded_text_bytes_bound)
                || host
                    .installed_policy
                    .decoder_scratch_bytes_per_token
                    .checked_mul(prefix)
                    != Some(numeric.decode_scratch_bytes_bound)
            {
                return Err(CostWorkloadDomainError::InvalidHostWork);
            }
        }
        Ok(())
    }

    fn validate_row_capacity(
        &self,
        rows: usize,
        recurrent_state_bytes: u64,
    ) -> Result<(), CostWorkloadDomainError> {
        if rows == 0 || rows > self.limits().maximum_rows.get() as usize {
            return Err(CostWorkloadDomainError::RowCapacity);
        }
        let expected = self
            .limits()
            .fixed_state_bytes_per_row
            .checked_mul(
                u64::try_from(rows).map_err(|_| CostWorkloadDomainError::ArithmeticOverflow)?,
            )
            .ok_or(CostWorkloadDomainError::ArithmeticOverflow)?;
        if recurrent_state_bytes != expected {
            return Err(CostWorkloadDomainError::RecurrentStateMismatch);
        }
        Ok(())
    }

    fn add_scheduled_work(
        &self,
        scheduled: &mut u64,
        work: ActualRowWork,
        final_prefill: Option<bool>,
    ) -> Result<(), CostWorkloadDomainError> {
        let context = u64::from(self.limits().maximum_context_tokens.get());
        let immediate = match (work, final_prefill) {
            (ActualRowWork::Decode { kv_tokens }, None) => {
                if kv_tokens == 0
                    || u64::from(kv_tokens)
                        .checked_add(1)
                        .ok_or(CostWorkloadDomainError::ArithmeticOverflow)?
                        > context
                {
                    return Err(CostWorkloadDomainError::ContextCapacity);
                }
                1
            }
            (
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                },
                Some(final_logits),
            ) => {
                let end = offset
                    .checked_add(count)
                    .ok_or(CostWorkloadDomainError::ArithmeticOverflow)?;
                if count == 0
                    || end > total_prompt_tokens
                    || (end == total_prompt_tokens) != final_logits
                {
                    return Err(CostWorkloadDomainError::InvalidRow);
                }
                if u64::from(total_prompt_tokens) > context {
                    return Err(CostWorkloadDomainError::ContextCapacity);
                }
                u64::from(count)
            }
            _ => return Err(CostWorkloadDomainError::UnsupportedWork),
        };
        *scheduled = scheduled
            .checked_add(immediate)
            .ok_or(CostWorkloadDomainError::ArithmeticOverflow)?;
        if *scheduled > self.limits().maximum_scheduled_tokens_per_wave.get() {
            return Err(CostWorkloadDomainError::ScheduledTokenCapacity);
        }
        Ok(())
    }

    fn validate_repetition(
        &self,
        tokens: u64,
        penalty_bits: u32,
    ) -> Result<(), CostWorkloadDomainError> {
        let penalty = f32::from_bits(penalty_bits);
        if !penalty.is_finite() || penalty <= 0.0 {
            return Err(CostWorkloadDomainError::InvalidRow);
        }
        if tokens > self.limits().repetition_slot_capacity
            || tokens > self.limits().output_vocabulary_elements.get()
        {
            return Err(CostWorkloadDomainError::RepetitionCapacity);
        }
        Ok(())
    }

    fn validate_host_limits(
        &self,
        numeric: &CostRowNumericFeatures,
    ) -> Result<(), CostWorkloadDomainError> {
        numeric
            .validate()
            .map_err(|_| CostWorkloadDomainError::InvalidHostWork)?;
        let context = u64::from(self.limits().maximum_context_tokens.get());
        if numeric.generated_tokens_before > context
            || numeric.sampling_history_tokens > context
            || numeric.decoded_prefix_tokens > context
        {
            return Err(CostWorkloadDomainError::ContextCapacity);
        }
        // maximum_output_tokens is termination metadata and may exceed C.
        Ok(())
    }
}
impl TryFrom<DomainWire> for CostWorkloadDomainV1 {
    type Error = CostWorkloadDomainError;
    fn try_from(wire: DomainWire) -> Result<Self, Self::Error> {
        if wire.schema_version != COST_WORKLOAD_DOMAIN_SCHEMA_V1
            || wire.identity.schema_version != EXECUTOR_COST_IDENTITY_SCHEMA
        {
            return Err(CostWorkloadDomainError::UnsupportedSchema);
        }
        // NonZero fields are checked by their typed decoder. No numeric axis
        // maxima are invented here: the bounded recipe may remain partial.
        let mut h = Sha256::new();
        h.update(b"ferrum.cost-workload-domain.v1\0");
        h.update(wire.schema_version.to_le_bytes());
        h.update([match wire.projection {
            CostWorkloadProjectionV1::ValidatedVNextRecipeV1 => 1,
        }]);
        h.update([match wire.host_projection {
            HostWorkloadProjectionV1::InstalledPlainTextV1 => 1,
        }]);
        h.update(wire.identity.schema_version.to_le_bytes());
        for value in [
            wire.identity.model_weights,
            wire.identity.numerical_policy,
            wire.identity.device_runtime,
            wire.identity.execution_config,
        ] {
            h.update(value);
        }
        h.update(wire.limits.maximum_rows.get().to_le_bytes());
        h.update(wire.limits.maximum_context_tokens.get().to_le_bytes());
        for value in [
            wire.limits.maximum_scheduled_tokens_per_wave.get(),
            wire.limits.output_vocabulary_elements.get(),
            wire.limits.repetition_slot_capacity,
            wire.limits.fixed_state_bytes_per_row,
        ] {
            h.update(value.to_le_bytes());
        }
        Ok(Self {
            wire,
            digest: h.finalize().into(),
        })
    }
}
impl From<CostWorkloadDomainV1> for DomainWire {
    fn from(value: CostWorkloadDomainV1) -> Self {
        value.wire
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CostWorkloadDomainError {
    #[error("unsupported workload domain schema")]
    UnsupportedSchema,
    #[error("workload row count exceeds the executor capacity")]
    RowCapacity,
    #[error("workload context exceeds the executor capacity")]
    ContextCapacity,
    #[error("workload scheduled tokens exceed the compiled capacity")]
    ScheduledTokenCapacity,
    #[error("repetition work exceeds the compiled slots or vocabulary")]
    RepetitionCapacity,
    #[error("unsupported workload role")]
    UnsupportedWork,
    #[error("invalid workload row")]
    InvalidRow,
    #[error("unsupported installed host policy")]
    UnsupportedHostPolicy,
    #[error("invalid bounded host work")]
    InvalidHostWork,
    #[error("fixed state bytes differ from the compiled program")]
    RecurrentStateMismatch,
    #[error("workload arithmetic overflow")]
    ArithmeticOverflow,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostWorkloadDomainUnknown {
    Unsupported,
    MissingExecutionIdentity(CostIdentityUnknownReason),
    InvalidCapacity,
    UnsupportedTokenScaledState,
    InvalidDescriptor(CostWorkloadDomainError),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostWorkloadDomainAvailability {
    Known(Arc<CostWorkloadDomainV1>),
    Unknown(CostWorkloadDomainUnknown),
}
impl Default for CostWorkloadDomainAvailability {
    fn default() -> Self {
        Self::Unknown(CostWorkloadDomainUnknown::Unsupported)
    }
}

#[cfg(test)]
mod tests;
