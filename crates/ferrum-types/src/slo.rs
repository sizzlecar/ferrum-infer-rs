//! Serializable SLO policy, independent of clocks, execution authority and benchmarks.
//!
//! The controller targets server ingress to committed model tokens. Optional
//! client-visible targets describe a separate acceptance boundary: they are not
//! inferred from internal budgets and are not guaranteed by a scheduling policy.
//! All defaults are bounded starting settings, not measured capacity promises.

use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    time::Duration,
};

mod cost;
mod reference;
pub use cost::{
    SloCostFeatureModel, SloCostModelConfig, SloCostObservationConfig, SloCostProfileExportConfig,
    SloCostProfileImportConfig, SloCostProfileReceipt, SloCostShapeLimits,
    SLO_COST_PROFILE_RECEIPT_RUNTIME_KEY,
};
pub use reference::*;

/// Typed snapshot entries; none of these keys is an environment-variable alias.
pub const SLO_CONFIG_RUNTIME_KEY: &str = "slo_config";
pub const SLO_CONFIG_PATH_RUNTIME_KEY: &str = "slo_config_path";
pub const SLO_CONFIG_DIGEST_RUNTIME_KEY: &str = "slo_config_sha256";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SloMode {
    #[default]
    Off,
    Observe,
    Enforce,
}

/// Relative internal budgets. Runtime ingress and commit timestamps belong in
/// a request context, never in this serializable policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SloLatencyBudgets {
    pub ttft_ms: NonZeroU64,
    pub tpot_ms: NonZeroU64,
    pub itl_ms: NonZeroU64,
}

impl SloLatencyBudgets {
    pub fn ttft(&self) -> Duration {
        Duration::from_millis(self.ttft_ms.get())
    }

    pub fn tpot(&self) -> Duration {
        Duration::from_millis(self.tpot_ms.get())
    }

    pub fn itl(&self) -> Duration {
        Duration::from_millis(self.itl_ms.get())
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("itl_ms", self.itl_ms),
        ] {
            validate_milliseconds(name, value)?;
        }
        Ok(())
    }
}

/// Population used for the independent ITL percentile target. Joint request
/// attainment always uses that request's maximum gap, regardless of this choice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SloItlPercentileScope {
    #[default]
    PooledGaps,
    RequestMaximumGap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloAttainmentTargets {
    /// Percentile values are expressed in (0, 100], not as fractions.
    pub ttft_percentile: f64,
    pub tpot_percentile: f64,
    pub itl_percentile: f64,
    pub itl_percentile_scope: SloItlPercentileScope,
    /// Fraction of explicitly accepted requests that succeed and meet all
    /// request-level timing rules. Failed, pending and unknown outcomes remain
    /// in the denominator. Acceptance means observable service acceptance (for
    /// example successful real output), not GPU admission or a time promise.
    /// It must not be inferred merely by subtracting rejections from offers.
    pub min_accepted_joint_attainment: f64,
    /// Optional independent target over all valid offered requests. Rejected,
    /// failed, pending and unknown requests must not count as a pass.
    pub min_offered_joint_attainment: Option<f64>,
    /// Rejected requests divided by valid offered requests.
    pub max_reject_rate: f64,
    /// Failed requests divided by accepted requests. Rejection is separate.
    pub max_error_rate: f64,
}

impl Default for SloAttainmentTargets {
    fn default() -> Self {
        Self {
            ttft_percentile: 99.0,
            tpot_percentile: 99.0,
            itl_percentile: 99.0,
            itl_percentile_scope: SloItlPercentileScope::PooledGaps,
            min_accepted_joint_attainment: 0.99,
            min_offered_joint_attainment: None,
            max_reject_rate: 0.001,
            max_error_rate: 0.001,
        }
    }
}

impl SloAttainmentTargets {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("ttft_percentile", self.ttft_percentile),
            ("tpot_percentile", self.tpot_percentile),
            ("itl_percentile", self.itl_percentile),
        ] {
            validate_fraction(name, value, 100.0, false)?;
        }
        validate_fraction(
            "min_accepted_joint_attainment",
            self.min_accepted_joint_attainment,
            1.0,
            false,
        )?;
        if let Some(target) = self.min_offered_joint_attainment {
            validate_fraction("min_offered_joint_attainment", target, 1.0, false)?;
        }
        validate_fraction("max_reject_rate", self.max_reject_rate, 1.0, true)?;
        validate_fraction("max_error_rate", self.max_error_rate, 1.0, true)
    }
}

/// Client request initiation to visible output. TTFT ends at the first nonempty
/// visible text update, and ITL is the interval between successive such updates
/// (SSE text-event gaps). Role-only, empty and finish-only updates are excluded.
/// TPOT uses actual output token usage and first/last visible output timestamps;
/// transport coalescing is reported rather than treated as strict token timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SloClientVisibleConfig {
    pub latency: SloLatencyBudgets,
    #[serde(default)]
    pub attainment: SloAttainmentTargets,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceSloConfig {
    pub id: String,
    /// Server ingress through model-token commit, including internal waiting.
    pub server_token_commit: SloLatencyBudgets,
    #[serde(default)]
    pub attainment: SloAttainmentTargets,
    /// An independent external acceptance target, never an implied guarantee.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_visible: Option<SloClientVisibleConfig>,
}

impl ServiceSloConfig {
    pub fn validate(&self) -> Result<(), String> {
        validate_class_id("service id", &self.id)?;
        self.server_token_commit
            .validate()
            .map_err(|reason| format!("service {} server_token_commit: {reason}", self.id))?;
        self.attainment
            .validate()
            .map_err(|reason| format!("service {} attainment: {reason}", self.id))?;
        if let Some(client) = &self.client_visible {
            client.latency.validate().map_err(|reason| {
                format!("service {} client_visible latency: {reason}", self.id)
            })?;
            client.attainment.validate().map_err(|reason| {
                format!("service {} client_visible attainment: {reason}", self.id)
            })?;
        }
        Ok(())
    }
}

/// Bounded experimental starting settings. They do not establish prediction
/// coverage: an Unknown cost/feasibility result must remain Unknown regardless
/// of the configured search limits or weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloPlannerConfig {
    pub candidate_limit: NonZeroUsize,
    pub beam_width: NonZeroUsize,
    pub lookahead_waves: NonZeroUsize,
    /// Complete reachable route/residency states retained per candidate prefix.
    /// Hitting this bound returns Unknown; it never drops a possible branch.
    pub max_route_states: NonZeroUsize,
    /// Complete whole-wave cost alternatives. Independent of candidate count.
    pub max_shape_alternatives: NonZeroUsize,
    pub max_planning_us: NonZeroU64,
    pub max_replan_attempts: NonZeroUsize,
    /// A new snapshot after transient lock contention or an exhausted compute
    /// slice. This timer never grants physical capacity or resets request time.
    pub retry_backoff_ms: NonZeroU64,
    /// Reference prefill work credit; not a contribution to output throughput.
    pub prefill_credit_beta: f64,
    pub prefill_debt_gamma: f64,
    pub enable_prefill_milestones: bool,
}

impl Default for SloPlannerConfig {
    fn default() -> Self {
        Self {
            candidate_limit: NonZeroUsize::new(16).unwrap(),
            beam_width: NonZeroUsize::new(4).unwrap(),
            lookahead_waves: NonZeroUsize::new(3).unwrap(),
            max_route_states: NonZeroUsize::new(16).unwrap(),
            max_shape_alternatives: NonZeroUsize::new(32).unwrap(),
            max_planning_us: NonZeroU64::new(2_000).unwrap(),
            max_replan_attempts: NonZeroUsize::new(2).unwrap(),
            retry_backoff_ms: NonZeroU64::new(1).unwrap(),
            prefill_credit_beta: 1.0,
            prefill_debt_gamma: 1.0,
            enable_prefill_milestones: true,
        }
    }
}

impl SloPlannerConfig {
    pub fn planning_budget(&self) -> Duration {
        Duration::from_micros(self.max_planning_us.get())
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_milliseconds("planner retry_backoff_ms", self.retry_backoff_ms)?;
        if self.max_route_states.get() > 256 || self.max_shape_alternatives.get() > 256 {
            return Err(
                "planner route-state and shape-alternative limits must not exceed 256".to_owned(),
            );
        }
        self.candidate_limit
            .get()
            .checked_mul(self.beam_width.get())
            .and_then(|count| count.checked_mul(self.lookahead_waves.get()))
            .and_then(|count| count.checked_mul(self.max_replan_attempts.get()))
            .ok_or_else(|| "planner candidate expansion bound overflows usize".to_owned())?;
        self.max_planning_us
            .get()
            .checked_mul(1_000)
            .ok_or_else(|| {
                "planner max_planning_us overflows a u64 nanosecond duration".to_owned()
            })?;
        for (name, value) in [
            ("prefill_credit_beta", self.prefill_credit_beta),
            ("prefill_debt_gamma", self.prefill_debt_gamma),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("planner {name} must be finite and nonnegative"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SloOutputLengthPolicy {
    /// Plan against the user's declared maximum; never shorten that maximum.
    #[default]
    ConservativeUpperBound,
    /// Requires independently calibrated length statistics in the cost profile.
    /// This is not authorization to alter maximum_sequence_tokens.
    StatisticalCapacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloAdmissionConfig {
    /// Whether failing a time promise may reject a new request. Physical
    /// capacity and input validity remain independent in both policies.
    pub time_policy: SloTimeAdmissionPolicy,
    /// Additional time-policy ceilings, not physical resource permits.
    pub max_active_requests: NonZeroUsize,
    pub max_waiting_requests: NonZeroUsize,
    pub max_waiting_prompt_tokens: NonZeroUsize,
    pub max_waiting_prompt_bytes: NonZeroUsize,
    /// Review the waiting request by this elapsed time from original ingress.
    /// This is a rejection expiry only under explicitly selected RequireSlo.
    pub max_wait_ms: NonZeroU64,
    /// Context range admitted by the controller's coverage policy. A request
    /// outside this range has no time promise; this is not a model limit and
    /// does not authorize rejection in CompleteRequests or output truncation.
    pub max_sequence_tokens: NonZeroUsize,
    pub output_length_policy: SloOutputLengthPolicy,
}

impl Default for SloAdmissionConfig {
    fn default() -> Self {
        Self {
            time_policy: SloTimeAdmissionPolicy::CompleteRequests,
            max_active_requests: NonZeroUsize::new(8).unwrap(),
            max_waiting_requests: NonZeroUsize::new(128).unwrap(),
            max_waiting_prompt_tokens: NonZeroUsize::new(1_048_576).unwrap(),
            max_waiting_prompt_bytes: NonZeroUsize::new(16 * 1_048_576).unwrap(),
            max_wait_ms: NonZeroU64::new(30_000).unwrap(),
            max_sequence_tokens: NonZeroUsize::new(32_768).unwrap(),
            output_length_policy: SloOutputLengthPolicy::ConservativeUpperBound,
        }
    }
}

/// Latency targets do not implicitly grant permission to discard work.
/// CompleteRequests retains late/unknown requests for safe, fair best-effort
/// service. RequireSlo may decline a new time promise before acceptance;
/// neither policy authorizes cancelling already accepted work for a SLO miss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SloTimeAdmissionPolicy {
    #[default]
    CompleteRequests,
    RequireSlo,
}

impl SloAdmissionConfig {
    pub fn max_wait(&self) -> Duration {
        Duration::from_millis(self.max_wait_ms.get())
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_milliseconds("admission max_wait_ms", self.max_wait_ms)?;
        self.max_active_requests
            .get()
            .checked_add(self.max_waiting_requests.get())
            .ok_or_else(|| "admission request count bound overflows usize".to_owned())?;
        self.max_waiting_prompt_tokens
            .get()
            .checked_mul(std::mem::size_of::<crate::TokenId>())
            .and_then(|bytes| bytes.checked_add(self.max_waiting_prompt_bytes.get()))
            .ok_or_else(|| "admission waiting prompt storage bound overflows usize".to_owned())?;
        Ok(())
    }
}

/// Product transport selection is independent of the scheduling policy.
/// Credited output requires a proved codec and keeps ownership through the
/// final transport byte owner; unsupported protocols must reject explicitly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SloOutputTransport {
    #[default]
    Legacy,
    Credited,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloOutputConfig {
    /// Legacy preserves Off/Observe output behavior unless explicitly changed.
    pub transport: SloOutputTransport,
    pub max_queued_events_per_request: NonZeroUsize,
    /// Includes reserved terminal metadata; event count alone does not bound text.
    pub max_queued_bytes_per_request: NonZeroUsize,
    pub max_projection_bytes_per_request: NonZeroUsize,
    pub terminal_reserve_bytes_per_request: NonZeroUsize,
    /// Total payload/projection budget, independent of KV/device capacity.
    pub max_total_buffer_bytes: NonZeroUsize,
    /// Time from continuous output blocking to cancellation. SLO clocks keep
    /// running while blocked and failure records remain in the denominator.
    pub slow_consumer_timeout_ms: NonZeroU64,
}

impl Default for SloOutputConfig {
    fn default() -> Self {
        Self {
            transport: SloOutputTransport::Legacy,
            max_queued_events_per_request: NonZeroUsize::new(256).unwrap(),
            max_queued_bytes_per_request: NonZeroUsize::new(1_048_576).unwrap(),
            max_projection_bytes_per_request: NonZeroUsize::new(1_048_576).unwrap(),
            terminal_reserve_bytes_per_request: NonZeroUsize::new(4_096).unwrap(),
            max_total_buffer_bytes: NonZeroUsize::new(64 * 1_048_576).unwrap(),
            slow_consumer_timeout_ms: NonZeroU64::new(30_000).unwrap(),
        }
    }
}

impl SloOutputConfig {
    pub fn slow_consumer_timeout(&self) -> Duration {
        Duration::from_millis(self.slow_consumer_timeout_ms.get())
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_milliseconds(
            "output slow_consumer_timeout_ms",
            self.slow_consumer_timeout_ms,
        )?;
        if self.terminal_reserve_bytes_per_request >= self.max_queued_bytes_per_request {
            return Err("output terminal reserve must leave room for queued data".to_owned());
        }
        let per_request = self
            .max_queued_bytes_per_request
            .get()
            .checked_add(self.max_projection_bytes_per_request.get())
            .ok_or_else(|| "output per-request storage bound overflows usize".to_owned())?;
        if per_request > self.max_total_buffer_bytes.get() {
            return Err(
                "output total buffer budget cannot hold one request's bounded buffers".to_owned(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloConfig {
    pub mode: SloMode,
    pub default_service_class: Option<String>,
    pub services: Vec<ServiceSloConfig>,
    /// Profile identity/content and execution fingerprint are verified when it
    /// is loaded; a nonempty path alone is not evidence of cost-model coverage.
    pub cost_profile: Option<PathBuf>,
    /// Independently frozen reference work and decode unit. Missing calibration
    /// is Unknown; it is never synthesized from online candidate estimates.
    pub prefill_reference: Option<SloPrefillReferenceConfig>,
    pub cost_observation: SloCostObservationConfig,
    pub planner: SloPlannerConfig,
    pub admission: SloAdmissionConfig,
    pub output: SloOutputConfig,
}

impl SloConfig {
    pub fn service(&self, id: &str) -> Option<&ServiceSloConfig> {
        self.services.iter().find(|service| service.id == id)
    }

    pub fn default_service(&self) -> Option<&ServiceSloConfig> {
        self.default_service_class
            .as_deref()
            .and_then(|id| self.service(id))
    }

    pub fn validate(&self) -> Result<(), String> {
        self.cost_observation.validate()?;
        if let Some(reference) = &self.prefill_reference {
            reference.validate()?;
        }
        if self.mode == SloMode::Off && self.cost_observation.profile_export.is_some() {
            return Err("cost profile export requires enabled SLO observation".into());
        }
        self.planner.validate()?;
        self.admission.validate()?;
        self.output.validate()?;
        // Each admitted request retains its own bounded share so a slow peer
        // cannot consume all global output capacity. These are payload budgets;
        // bounded channel/task overhead is accounted separately by the engine.
        let active_output_bytes = self
            .output
            .max_queued_bytes_per_request
            .get()
            .checked_add(self.output.max_projection_bytes_per_request.get())
            .and_then(|bytes| bytes.checked_mul(self.admission.max_active_requests.get()))
            .ok_or_else(|| "output active-request storage bound overflows usize".to_owned())?;
        if active_output_bytes > self.output.max_total_buffer_bytes.get() {
            return Err("output total buffer budget is smaller than the bounded shares of max_active_requests".to_owned());
        }
        if let Some(path) = &self.cost_profile {
            if path.as_os_str().is_empty() {
                return Err("cost_profile path must not be empty".to_owned());
            }
        }
        let mut ids = HashSet::new();
        for service in &self.services {
            service.validate()?;
            if !ids.insert(service.id.as_str()) {
                return Err(format!("duplicate service class {:?}", service.id));
            }
            let tokens = u64::try_from(self.admission.max_sequence_tokens.get())
                .map_err(|_| "max_sequence_tokens exceeds u64".to_owned())?;
            service
                .server_token_commit
                .tpot_ms
                .get()
                .checked_mul(tokens)
                .and_then(|ms| ms.checked_mul(1_000_000))
                .ok_or_else(|| {
                    format!(
                        "service {} cumulative TPOT budget overflows u64 nanoseconds",
                        service.id
                    )
                })?;
        }
        if let Some(id) = &self.default_service_class {
            validate_class_id("default_service_class", id)?;
            if !ids.contains(id.as_str()) {
                return Err(format!(
                    "default_service_class {id:?} has no service configuration"
                ));
            }
        } else if self.mode != SloMode::Off {
            return Err("Observe and Enforce require an explicit default_service_class".to_owned());
        }
        if self.mode == SloMode::Enforce
            && self.admission.time_policy == SloTimeAdmissionPolicy::RequireSlo
            && self.cost_profile.is_none()
        {
            return Err("RequireSlo Enforce requires an explicit cost_profile".to_owned());
        }
        Ok(())
    }
}

fn validate_class_id(field: &str, id: &str) -> Result<(), String> {
    if id.is_empty() || id.trim() != id || id.chars().any(char::is_control) {
        return Err(format!(
            "{field} must be nonempty without surrounding whitespace or control characters"
        ));
    }
    Ok(())
}

fn validate_milliseconds(field: &str, value: NonZeroU64) -> Result<(), String> {
    value
        .get()
        .checked_mul(1_000_000)
        .map(|_| ())
        .ok_or_else(|| format!("{field} overflows a u64 nanosecond duration"))
}

fn validate_fraction(
    field: &str,
    value: f64,
    maximum: f64,
    allow_zero: bool,
) -> Result<(), String> {
    if !value.is_finite() || value > maximum || value < 0.0 || (!allow_zero && value == 0.0) {
        let start = if allow_zero { "[0" } else { "(0" };
        return Err(format!("{field} must be finite and in {start}, {maximum}]"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
