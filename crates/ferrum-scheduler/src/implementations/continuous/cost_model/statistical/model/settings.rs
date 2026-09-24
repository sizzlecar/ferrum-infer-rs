use super::super::super::CostShapeLimits;
use super::*;
use std::num::{NonZeroU64, NonZeroUsize};

/// New-model settings contain only consumed limits and statistical parameters.
/// Exact/numeric legacy feature selectors and bucket widths are not reinterpreted.
#[derive(Debug, Clone, PartialEq)]
pub struct WholeWaveSettingsV1 {
    pub max_buckets: NonZeroUsize,
    pub max_samples_per_bucket: NonZeroUsize,
    pub min_samples: NonZeroUsize,
    pub max_retained_samples: NonZeroUsize,
    pub max_retained_shape_rows: NonZeroUsize,
    pub residual_quantile: f64,
    pub drift_margin_ns: u64,
    pub max_wave_ns: NonZeroU64,
    pub max_sample_age_ns: NonZeroU64,
    pub shape_limits: CostShapeLimits,
}
impl WholeWaveSettingsV1 {
    /// Explicitly copies the existing resource/error policy, never its model.
    pub fn from_policy_limits(s: &CostModelSettings) -> Self {
        Self {
            max_buckets: s.max_buckets,
            max_samples_per_bucket: s.max_samples_per_bucket,
            min_samples: s.min_samples,
            max_retained_samples: s.max_retained_samples,
            max_retained_shape_rows: s.max_retained_shape_rows,
            residual_quantile: s.residual_quantile,
            drift_margin_ns: s.drift_margin_ns,
            max_wave_ns: s.max_wave_ns,
            max_sample_age_ns: s.max_sample_age_ns,
            shape_limits: s.shape_limits.clone(),
        }
    }
    pub fn validate(&self) -> Result<(), ModelUnknown> {
        let old = CostModelSettings {
            max_buckets: self.max_buckets,
            max_samples_per_bucket: self.max_samples_per_bucket,
            min_samples: self.min_samples,
            max_retained_samples: self.max_retained_samples,
            max_retained_shape_rows: self.max_retained_shape_rows,
            residual_quantile: self.residual_quantile,
            drift_margin_ns: self.drift_margin_ns,
            max_wave_ns: self.max_wave_ns,
            max_sample_age_ns: self.max_sample_age_ns,
            shape_limits: self.shape_limits.clone(),
            ..CostModelSettings::default()
        };
        old.validate().map_err(|_| ModelUnknown::InvalidSettings)?;
        if self.min_samples.get() < 8
            || self.residual_quantile < 0.99
            || self.max_wave_ns.get() > (1u64 << 53)
        {
            return Err(ModelUnknown::InvalidSettings);
        }
        Ok(())
    }
}
impl Default for WholeWaveSettingsV1 {
    fn default() -> Self {
        Self::from_policy_limits(&CostModelSettings::default())
    }
}
