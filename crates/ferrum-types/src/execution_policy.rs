//! Typed execution-policy and admission authority contracts.

use serde::{Deserialize, Serialize};

/// Current CUDA native-adaptive boundary between single- and multi-partition
/// addressed decode. The compiled provider and effective-config report consume
/// this same constant.
pub const CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS: u64 = 512;

/// Declares the authoritative owner of request-lifetime accelerator resources.
///
/// `PlanRuntime` owns admission, allocation, fences, backpressure, and release.
/// `LegacyEngine` remains only for executors that have not migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionResourceAuthority {
    LegacyEngine,
    PlanRuntime,
}

impl Default for ExecutionResourceAuthority {
    fn default() -> Self {
        Self::LegacyEngine
    }
}

/// Product policy for selecting an attention provider implementation.
///
/// The policy selects a provider family, not a fixed kernel. An adaptive
/// provider may select different physical variants for each invocation based
/// on the admitted shape and sequence frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionExecutionPolicy {
    /// Resolve to the best compiled provider family at composition time.
    Auto,
    /// Use the portable backend implementation and avoid optional native
    /// accelerator libraries.
    Portable,
    /// Use a compiled native provider with invocation-shape selection.
    NativeAdaptive,
}

impl AttentionExecutionPolicy {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Portable => "portable",
            Self::NativeAdaptive => "native-adaptive",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "auto" => Ok(Self::Auto),
            "portable" => Ok(Self::Portable),
            "native-adaptive" => Ok(Self::NativeAdaptive),
            _ => Err(format!(
                "expected auto, portable, or native-adaptive; got {raw:?}"
            )),
        }
    }

    /// Resolves the product request into the policy sealed into the runtime
    /// policy and provider descriptor.
    pub fn resolve(self, native_adaptive_supported: bool) -> Result<Self, String> {
        match self {
            Self::Auto if native_adaptive_supported => Ok(Self::NativeAdaptive),
            Self::Auto => Ok(Self::Portable),
            Self::NativeAdaptive if !native_adaptive_supported => Err(
                "native-adaptive attention was requested but the selected backend composition does not provide it"
                    .to_owned(),
            ),
            resolved => Ok(resolved),
        }
    }

    pub const fn is_resolved(self) -> bool {
        !matches!(self, Self::Auto)
    }
}

impl Default for AttentionExecutionPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Immutable admission ceilings compiled into one model executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorAdmissionLimits {
    maximum_active_sequences: u32,
    maximum_scheduled_tokens: u64,
}

impl ExecutorAdmissionLimits {
    pub fn new(
        maximum_active_sequences: u32,
        maximum_scheduled_tokens: u64,
    ) -> Result<Self, String> {
        if maximum_active_sequences == 0 {
            return Err("maximum_active_sequences must be non-zero".to_owned());
        }
        if maximum_scheduled_tokens == 0 {
            return Err("maximum_scheduled_tokens must be non-zero".to_owned());
        }
        Ok(Self {
            maximum_active_sequences,
            maximum_scheduled_tokens,
        })
    }

    pub const fn maximum_active_sequences(self) -> u32 {
        self.maximum_active_sequences
    }

    pub const fn maximum_scheduled_tokens(self) -> u64 {
        self.maximum_scheduled_tokens
    }
}

/// Runtime-owned admission state exposed to product health and validators.
///
/// This snapshot is deliberately separate from startup sizing estimates. Its
/// phase counts come from one scheduler-owned request index, so one request
/// cannot appear in both prefill and decode in the same observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorAdmissionSnapshot {
    schema_version: u32,
    resource_authority: ExecutionResourceAuthority,
    #[serde(flatten)]
    limits: ExecutorAdmissionLimits,
    waiting_requests: u32,
    active_sequences: u32,
    active_prefill_sequences: u32,
    active_decode_sequences: u32,
    current_batch_size: Option<u32>,
    capacity_blocked_requests: Option<u32>,
}

impl ExecutorAdmissionSnapshot {
    pub fn new(
        resource_authority: ExecutionResourceAuthority,
        limits: ExecutorAdmissionLimits,
        waiting_requests: u32,
        active_prefill_sequences: u32,
        active_decode_sequences: u32,
        current_batch_size: Option<u32>,
        capacity_blocked_requests: Option<u32>,
    ) -> Result<Self, String> {
        let active_sequences = active_prefill_sequences
            .checked_add(active_decode_sequences)
            .ok_or_else(|| "active sequence phase counts overflow u32".to_owned())?;
        if active_sequences > limits.maximum_active_sequences {
            return Err(format!(
                "active_sequences {active_sequences} exceeds maximum_active_sequences {}",
                limits.maximum_active_sequences
            ));
        }
        if current_batch_size.is_some_and(|batch| batch > active_sequences) {
            return Err(format!(
                "current_batch_size {} exceeds active_sequences {active_sequences}",
                current_batch_size.unwrap_or_default()
            ));
        }
        let observed_requests = waiting_requests
            .checked_add(active_sequences)
            .ok_or_else(|| "observed admission request counts overflow u32".to_owned())?;
        if capacity_blocked_requests.is_some_and(|blocked| blocked > observed_requests) {
            return Err(format!(
                "capacity_blocked_requests {} exceeds observed waiting plus active requests {observed_requests}",
                capacity_blocked_requests.unwrap_or_default()
            ));
        }
        Ok(Self {
            schema_version: 2,
            resource_authority,
            limits,
            waiting_requests,
            active_sequences,
            active_prefill_sequences,
            active_decode_sequences,
            current_batch_size,
            capacity_blocked_requests,
        })
    }

    pub const fn resource_authority(&self) -> ExecutionResourceAuthority {
        self.resource_authority
    }

    pub const fn maximum_active_sequences(&self) -> u32 {
        self.limits.maximum_active_sequences
    }

    pub const fn maximum_scheduled_tokens(&self) -> u64 {
        self.limits.maximum_scheduled_tokens
    }

    pub const fn active_sequences(&self) -> u32 {
        self.active_sequences
    }

    pub const fn waiting_requests(&self) -> u32 {
        self.waiting_requests
    }

    pub const fn active_prefill_sequences(&self) -> u32 {
        self.active_prefill_sequences
    }

    pub const fn active_decode_sequences(&self) -> u32 {
        self.active_decode_sequences
    }

    pub const fn current_batch_size(&self) -> Option<u32> {
        self.current_batch_size
    }

    pub const fn capacity_blocked_requests(&self) -> Option<u32> {
        self.capacity_blocked_requests
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionExecutionPolicy, ExecutionResourceAuthority, ExecutorAdmissionLimits,
        ExecutorAdmissionSnapshot,
    };

    #[test]
    fn attention_policy_resolves_auto_without_hiding_native_requirements() {
        assert_eq!(
            AttentionExecutionPolicy::Auto.resolve(true).unwrap(),
            AttentionExecutionPolicy::NativeAdaptive
        );
        assert_eq!(
            AttentionExecutionPolicy::Auto.resolve(false).unwrap(),
            AttentionExecutionPolicy::Portable
        );
        assert!(AttentionExecutionPolicy::NativeAdaptive
            .resolve(false)
            .is_err());
    }

    #[test]
    fn admission_snapshot_rejects_runtime_over_admission_ceiling() {
        let limits = ExecutorAdmissionLimits::new(16, 2048).unwrap();
        assert!(ExecutorAdmissionSnapshot::new(
            ExecutionResourceAuthority::PlanRuntime,
            limits,
            0,
            8,
            9,
            None,
            None,
        )
        .is_err());
    }

    #[test]
    fn admission_snapshot_reconciles_single_source_phase_counts() {
        let snapshot = ExecutorAdmissionSnapshot::new(
            ExecutionResourceAuthority::PlanRuntime,
            ExecutorAdmissionLimits::new(32, 4096).unwrap(),
            3,
            7,
            11,
            Some(8),
            Some(2),
        )
        .unwrap();
        assert_eq!(snapshot.active_sequences(), 18);
        assert_eq!(snapshot.active_prefill_sequences(), 7);
        assert_eq!(snapshot.active_decode_sequences(), 11);
        assert_eq!(snapshot.current_batch_size(), Some(8));
    }
}
