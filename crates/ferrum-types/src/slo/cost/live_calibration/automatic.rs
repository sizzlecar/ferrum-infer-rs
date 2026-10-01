//! Runtime-owned calibration budgets. Numerical settings and readiness remain
//! native-engine concerns; these types neither import a scheduler nor turn a
//! resource allowance into a qualified model or completed reference protocol.
use super::*;
use crate::{SloSelectedFeedbackSettingsV1, SloSelectedFeedbackStorageV1};

#[cfg(test)]
mod tests;

mod cost_probe;
pub use cost_probe::*;
mod input_readiness;
pub use input_readiness::*;
mod prediction_margin;
pub use prediction_margin::*;
mod prediction_validity;
pub use prediction_validity::*;
mod numerical_strategy;
pub use numerical_strategy::*;
mod reuse;
pub use reuse::*;

const MAX_WINDOW_NS: u64 = 86_400_000_000_000;
const MAX_STATE_BYTES: usize = 512 * 1024 * 1024;
const MAX_RETAINED_GENERATIONS: usize = 128;
const MAX_COMBINED_RETAINED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_DIAGNOSTIC_TOTAL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_ENCODED_SOURCE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_PROBE_DURATION_MS: u64 = 3_600_000;

/// Population ordering is distinct from the permitted physical execution
/// routes. Both schedules retain the original offered attempts and require
/// independent fitting, residual calibration and qualification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloAutomaticCalibrationPopulationScheduleV1 {
    /// Historical source6 population: all owners share three fixed windows.
    FixedWindowsV1,
    /// Complete global offer blocks with independently advancing owner phases.
    /// A temporarily absent owner retains its phase until age or capacity
    /// expires; numerical failure remains terminal for that attempt.
    #[default]
    OwnerBlocksV1,
    /// Independent source7 attempts overlap on the same original offer blocks.
    /// The first completed Discovery declares one successor at the next block
    /// boundary. Each attempt retains its own original collection deadline.
    OwnerBlocksRollingV2,
}
impl SloAutomaticCalibrationPopulationScheduleV1 {
    pub fn uses_owner_blocks(self) -> bool {
        matches!(self, Self::OwnerBlocksV1 | Self::OwnerBlocksRollingV2)
    }
}

/// Bounded starting values, not measured coverage or latency guarantees.
/// Each owner's discovery freezes before its three fresh numerical phases.
/// Reaching a retention limit never means the lifetime generation limit was
/// reached: automatic calibration must continue with bounded retained state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloAutomaticCalibrationSettingsV1 {
    pub population_schedule: SloAutomaticCalibrationPopulationScheduleV1,
    /// Explicit candidate numerical contract; omission preserves the current estimator.
    pub numerical_strategy: SloAutomaticCalibrationNumericalStrategyV1,
    /// Input-only readiness for ordinary source7 owner blocks. FixedWindowsV1
    /// and private prepared source8 declarations keep their own stopping rules.
    pub input_readiness: SloAutomaticCalibrationInputReadinessV1,
    /// Extra prediction conservatism frozen from independent Residual inputs
    /// and durations, before Qualification, for automatic source7 and source8.
    /// Historical FixedWindowsV1 keeps its original native settings.
    pub prediction_margin: SloAutomaticCalibrationPredictionMarginV1,
    /// Collection always obeys maximum_window_ns. This policy separately
    /// declares qualified owner-block prediction lifetime; sample timestamps
    /// and maximum sample age are never renewed by publication or import.
    pub prediction_validity: SloAutomaticCalibrationPredictionValidityV1,
    /// Omitted runtime settings use the same automatic population as Default,
    /// including when only diagnostics or probe budgets are supplied. Explicit
    /// older populations retain their meaning; historical source/profile
    /// declarations have their own unchanged defaults.
    pub route_population: SloCalibrationRoutePopulationV1,
    /// Discovery attempts; also the complete global audit-block size under
    /// OwnerBlocksV1. Includes outside-route and proven unsubmitted attempts.
    pub discovery_offered_waves: NonZeroUsize,
    /// Fixed window sizes under FixedWindowsV1. Under OwnerBlocksV1 these
    /// are minimum global offered populations for each owner's phase; the
    /// phase also needs the independently configured numerical member minimum.
    pub phase_offered_waves: [NonZeroUsize; 3],
    /// Maximum elapsed time for one declared collection window. Engine-native
    /// numerical sample age must also permit the selected value.
    pub maximum_window_ns: NonZeroU64,
    pub maximum_owners: NonZeroUsize,
    pub maximum_discovery_bytes: NonZeroUsize,
    /// Cumulative canonical source bytes hashed per owner-block epoch or
    /// prepared startup source. This bounds encoding/hash work, not retained
    /// RAM or optional diagnostic files. Source6 FixedWindows keeps its
    /// historical import policy. Exhaustion fails the original source.
    pub maximum_encoded_source_bytes: NonZeroU64,
    /// Per retained generation, including its numerical/evidence state.
    pub maximum_retained_numeric_bytes: NonZeroUsize,
    /// Simultaneously retained origins; never a limit on lifetime generations.
    pub maximum_retained_generations: NonZeroUsize,
    pub reference_probe: SloAutomaticReferenceProbeSettingsV1,
    pub cost_probe: SloAutomaticCostProbeSettingsV1,
    /// Conservative process-local drift corrections. Independent fresh
    /// calibration is required before replacing the base or reducing margins.
    /// An explicit `structured_feedback` policy takes precedence over this
    /// automatic policy, including its declared storage and resource limits.
    /// With the automatic default policy, a fully validated new owner stays
    /// Unknown until independently qualified; it does not revoke unrelated
    /// owners. Validated inputs outside an existing owner's numerical support
    /// also remain Unknown and are counted separately. Missing evidence still
    /// counts as uncomparable. Identity, order and lag checks remain active.
    pub feedback: SloSelectedFeedbackSettingsV1,
    /// Automatic same-boot reuse is independent of optional diagnostics.
    /// A cache miss preserves ordinary automatic calibration.
    pub reuse: SloAutomaticCalibrationReuseV1,
    pub diagnostics: SloAutomaticCalibrationDiagnosticsV1,
}
impl Default for SloAutomaticCalibrationSettingsV1 {
    fn default() -> Self {
        Self {
            population_schedule: SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksV1,
            numerical_strategy: SloAutomaticCalibrationNumericalStrategyV1::default(),
            input_readiness: SloAutomaticCalibrationInputReadinessV1::default(),
            prediction_margin: SloAutomaticCalibrationPredictionMarginV1::default(),
            prediction_validity: SloAutomaticCalibrationPredictionValidityV1::default(),
            route_population:
                SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2,
            discovery_offered_waves: NonZeroUsize::new(256).unwrap(),
            phase_offered_waves: [NonZeroUsize::new(256).unwrap(); 3],
            maximum_window_ns: NonZeroU64::new(300_000_000_000).unwrap(),
            maximum_owners: NonZeroUsize::new(128).unwrap(),
            maximum_discovery_bytes: NonZeroUsize::new(8 * 1024 * 1024).unwrap(),
            // Bounded starting allowance: 16k offered waves at 64 KiB each.
            // It is neither a per-wave bound nor a completion guarantee.
            maximum_encoded_source_bytes: NonZeroU64::new(1024 * 1024 * 1024).unwrap(),
            maximum_retained_numeric_bytes: NonZeroUsize::new(128 * 1024 * 1024).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
            reference_probe: SloAutomaticReferenceProbeSettingsV1::default(),
            cost_probe: SloAutomaticCostProbeSettingsV1::default(),
            feedback: SloSelectedFeedbackSettingsV1 {
                window_samples: NonZeroUsize::new(32).unwrap(),
                minimum_underestimates: NonZeroUsize::new(2).unwrap(),
                minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
                trigger_excess_ns: NonZeroU64::new(100_000).unwrap(),
                correction_padding_ns: 100_000,
                maximum_family_margin_ns: NonZeroU64::new(60_000_000_000).unwrap(),
                maximum_consumption_lag_ns: NonZeroU64::new(1_000_000_000).unwrap(),
                maximum_uncomparable_observations: 0,
                maximum_failed_or_partial: 0,
                maximum_queue_drops: 0,
                maximum_state_bytes: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
            },
            reuse: SloAutomaticCalibrationReuseV1::default(),
            diagnostics: SloAutomaticCalibrationDiagnosticsV1::MemoryOnly,
        }
    }
}
impl SloAutomaticCalibrationSettingsV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.input_readiness.validate()?;
        if self.numerical_strategy
            == SloAutomaticCalibrationNumericalStrategyV1::IdentifiedFitGlobalResidualV1
        {
            if !self.population_schedule.uses_owner_blocks() {
                return Err("identified global residual requires owner blocks; FixedWindowsV1 does not support this strategy".into());
            }
            if !matches!(
                self.input_readiness,
                SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 { .. }
            ) {
                return Err("identified global residual requires work_axes_and_branches_v3 input readiness; earlier geometry combinations are unsupported".into());
            }
        }
        if self.numerical_strategy
            == SloAutomaticCalibrationNumericalStrategyV1::SameSourceJointCellsV1
            && (!self.population_schedule.uses_owner_blocks()
                || self.input_readiness
                    == (SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {}))
        {
            return Err("joint cells require owner blocks and declared input readiness".into());
        }
        if self.discovery_offered_waves.get() > 65_536
            || self
                .phase_offered_waves
                .iter()
                .any(|n| !(8..=4096).contains(&n.get()))
            || self.maximum_window_ns.get() > MAX_WINDOW_NS
            || self.maximum_owners.get() > 128
            || self.maximum_discovery_bytes.get() > MAX_STATE_BYTES
            || self.maximum_encoded_source_bytes.get() > MAX_ENCODED_SOURCE_BYTES
            || self.maximum_retained_numeric_bytes.get() > MAX_STATE_BYTES
            || self.maximum_retained_generations.get() > MAX_RETAINED_GENERATIONS
        {
            return Err("automatic calibration exceeds bounded discovery/phase/window/owner/retention capacities".into());
        }
        if self.population_schedule.uses_owner_blocks() {
            if self.population_schedule
                == SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2
                && self.input_readiness == (SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {})
            {
                return Err("rolling owner blocks require declared input readiness".into());
            }
            let block = self.discovery_offered_waves.get();
            self.input_readiness
                .validate_owner_blocks(block, self.phase_offered_waves.map(NonZeroUsize::get))?;
            // A phase may first reach its eight-member minimum part-way
            // through the final block. Capacity must include that whole block,
            // as well as rounding a declared global offer quota upward.
            if self.input_readiness == (SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {})
                && (block.checked_add(7).is_none_or(|members| members > 4096)
                    || self.phase_offered_waves.iter().any(|quota| {
                        quota
                            .get()
                            .div_ceil(block)
                            .checked_mul(block)
                            .is_none_or(|members| members > 4096)
                    }))
            {
                return Err(
                    "owner block population exceeds the complete-phase member capacity".into(),
                );
            }
        }
        self.reference_probe.validate()?;
        self.cost_probe.validate()?;
        self.reuse.validate()?;
        self.diagnostics.validate()?;
        super::super::feedback::validate_feedback_bounds(
            &self.feedback,
            &SloSelectedFeedbackStorageV1::MemoryOnly,
            self.maximum_owners.get(),
            128 * 4096,
            1 << 53,
            u64::MAX,
            self.maximum_retained_numeric_bytes.get(),
        )?;
        // Multiple retained origins and the independent discovery/reference
        // populations must not turn individually bounded fields into an
        // effectively unbounded combined allocation allowance. This is a sum
        // of configured state budgets, not total process memory or GPU memory.
        let combined = u64::try_from(self.maximum_retained_numeric_bytes.get())
            .ok()
            .and_then(|bytes| {
                bytes.checked_mul(u64::try_from(self.maximum_retained_generations.get()).ok()?)
            })
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(self.maximum_discovery_bytes.get()).ok()?)
            })
            .and_then(|bytes| bytes.checked_add(self.reference_probe.maximum_source_bytes.get()))
            .and_then(|bytes| bytes.checked_add(self.feedback.maximum_state_bytes.get() as u64))
            .ok_or_else(|| {
                "automatic calibration combined retention budget overflows".to_owned()
            })?;
        if combined > MAX_COMBINED_RETAINED_BYTES {
            return Err(
                "automatic calibration combined retained state budgets exceed 2 GiB".into(),
            );
        }
        Ok(())
    }
}

/// An independent reference probe has a complete protocol. Exhausting any
/// budget leaves reference coverage Unknown; it cannot shorten the protocol,
/// substitute estimated costs, or label fewer trials as a completed probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloAutomaticReferenceProbeSettingsV1 {
    pub fresh_trials_per_anchor: NonZeroUsize,
    pub maximum_duration_ms: NonZeroU64,
    /// Total retained raw reference source, independent of a diagnostic path.
    pub maximum_source_bytes: NonZeroU64,
    /// Total submitted probe requests, including repeated fresh trials.
    pub maximum_probe_requests: NonZeroUsize,
    /// Engineering headroom on measured probe-work estimates, not an SLO bound.
    pub work_estimate_margin_percent: u16,
    /// Reserve part of the probe deadline for source assembly and readiness
    /// finalization. In-flight work must still be drained after a hard timeout.
    pub finalization_reserve_percent: u8,
}
impl Default for SloAutomaticReferenceProbeSettingsV1 {
    fn default() -> Self {
        Self {
            fresh_trials_per_anchor: NonZeroUsize::new(3).unwrap(),
            maximum_duration_ms: NonZeroU64::new(120_000).unwrap(),
            maximum_source_bytes: NonZeroU64::new(64 * 1024 * 1024).unwrap(),
            maximum_probe_requests: NonZeroUsize::new(128).unwrap(),
            work_estimate_margin_percent: 25,
            finalization_reserve_percent: 10,
        }
    }
}
impl SloAutomaticReferenceProbeSettingsV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.fresh_trials_per_anchor.get() > crate::PREFILL_REFERENCE_MAX_REPETITIONS
            || self.maximum_duration_ms.get() > MAX_PROBE_DURATION_MS
            || self.maximum_source_bytes.get()
                > crate::SloCostProfileImportConfig::MAX_FILE_BYTES as u64
            || self.maximum_probe_requests.get() > crate::PREFILL_REFERENCE_MAX_SAMPLES
            || self.work_estimate_margin_percent > 1_000
            || self.finalization_reserve_percent == 0
            || self.finalization_reserve_percent >= 100
        {
            return Err(
                "automatic reference probe exceeds bounded trial/time/source/request capacities"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloAutomaticCalibrationDiagnosticsV1 {
    #[default]
    #[serde(deserialize_with = "super::super::feedback::deserialize_disabled")]
    MemoryOnly,
    /// Explicit optional diagnostic persistence. Total bytes include in-flight
    /// writes and retained source/profile/audit files owned by this component;
    /// restarting the process does not reset the directory quota.
    /// Storage failures are reported separately and never qualify a source or
    /// invalidate an otherwise qualified process-local reference/model.
    Directory {
        directory: PathBuf,
        maximum_source_bytes: NonZeroU64,
        maximum_total_bytes: NonZeroU64,
        maximum_retained_generations: NonZeroUsize,
    },
}
impl SloAutomaticCalibrationDiagnosticsV1 {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::MemoryOnly => Ok(()),
            Self::Directory {
                directory,
                maximum_source_bytes,
                maximum_total_bytes,
                maximum_retained_generations,
            } => {
                if directory.as_os_str().is_empty()
                    || maximum_source_bytes.get()
                        > crate::SloCostProfileImportConfig::MAX_FILE_BYTES as u64
                    || maximum_source_bytes > maximum_total_bytes
                    || maximum_total_bytes.get() > MAX_DIAGNOSTIC_TOTAL_BYTES
                    || maximum_retained_generations.get() > 65_536
                {
                    return Err("automatic diagnostic directory requires a nonempty path and bounded per-source/total-byte/retained-generation quotas".into());
                }
                Ok(())
            }
        }
    }
}
