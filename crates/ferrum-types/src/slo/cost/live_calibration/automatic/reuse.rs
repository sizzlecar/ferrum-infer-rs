//! Automatic cache policy. Identity, original age, clean shutdown and feedback
//! validation remain mandatory in the cache consumer.
use super::*;

mod location;
pub use location::*;

#[cfg(test)]
mod tests;

/// Runtime defaults apply to omitted and partial AutomaticV1 settings.
/// Diagnostics remain independent: MemoryOnly does not disable this cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloAutomaticCalibrationReuseV1 {
    /// No automatic cache reads or writes. Explicit profile import is separate.
    Disabled {},
    /// Reuse only a complete, cleanly closed cache from the same OS boot and
    /// monotonic namespace. Restoring a profile must not refresh its original
    /// TTL, discard feedback corrections, or reuse execution authorities.
    /// An unsupported clock, unavailable location, invalid cache or exceeded
    /// budget is a cache miss; ordinary automatic calibration remains enabled.
    SameBootCleanShutdownV1 {
        #[serde(default)]
        location: SloAutomaticCalibrationCacheLocationV1,
        #[serde(default)]
        limits: SloAutomaticCalibrationReuseLimitsV1,
    },
}

impl Default for SloAutomaticCalibrationReuseV1 {
    fn default() -> Self {
        Self::SameBootCleanShutdownV1 {
            location: SloAutomaticCalibrationCacheLocationV1::default(),
            limits: SloAutomaticCalibrationReuseLimitsV1::default(),
        }
    }
}

impl SloAutomaticCalibrationReuseV1 {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Disabled {} => Ok(()),
            Self::SameBootCleanShutdownV1 { location, limits } => {
                location.validate()?;
                limits.validate()
            }
        }
    }
}

/// Additional resources needed for persistence, not new numerical allowances.
///
/// The consumer must also use the existing automatic maximum retained
/// generations, encoded-source and retained-memory limits, and the existing
/// profile import sample/shape/field/age limits. Automatic source input bytes
/// are bounded by this cache's total byte allowance and the import format's
/// hard ceiling; the separate manual-import byte default is unchanged.
/// Source counts, bytes and loaded samples/rows are bounded across the whole
/// cache transaction, never granted afresh to each source. Same-boot reuse does
/// not require a fabricated wall clock error bound or renew original age.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloAutomaticCalibrationReuseLimitsV1 {
    /// Total cache-owned disk bytes, including retained generations, original
    /// source journals, feedback state, manifests and simultaneous staging.
    /// Unrelated user files are never eviction candidates.
    pub maximum_total_bytes: NonZeroU64,
    /// Wall budget for one entire cold load or clean-save transaction, including
    /// all sources, replay and publication checks; not reset per source/retry.
    /// Expiry prevents cache publication and cannot authorize partial replay.
    pub maximum_operation_duration_ms: NonZeroU64,
}

impl Default for SloAutomaticCalibrationReuseLimitsV1 {
    fn default() -> Self {
        Self {
            maximum_total_bytes: NonZeroU64::new(256 * 1024 * 1024).unwrap(),
            maximum_operation_duration_ms: NonZeroU64::new(30_000).unwrap(),
        }
    }
}

impl SloAutomaticCalibrationReuseLimitsV1 {
    pub fn validate(&self) -> Result<(), String> {
        // Reuse the existing diagnostic disk ceiling and cold probe duration
        // ceiling; this policy does not increase either class of resource cap.
        if self.maximum_total_bytes.get() > MAX_DIAGNOSTIC_TOTAL_BYTES
            || self.maximum_operation_duration_ms.get() > MAX_PROBE_DURATION_MS
        {
            return Err(
                "automatic reuse exceeds bounded disk or cold-transaction duration capacity".into(),
            );
        }
        Ok(())
    }
}
