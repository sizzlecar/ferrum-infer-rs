use super::*;
use ferrum_engine::continuous_engine::StructuredCalibrationGroupLimitsV2;
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupLimitsV2 {
    pub maximum_children: NonZeroUsize,
    pub maximum_total_file_bytes: NonZeroU64,
    pub maximum_retained_numeric_bytes: NonZeroUsize,
    pub maximum_retained_coordinates: NonZeroUsize,
}
impl Default for GroupLimitsV2 {
    fn default() -> Self {
        let v = StructuredCalibrationGroupLimitsV2::default();
        Self {
            maximum_children: v.maximum_children,
            maximum_total_file_bytes: v.maximum_total_file_bytes,
            maximum_retained_numeric_bytes: v.maximum_retained_numeric_bytes,
            maximum_retained_coordinates: v.maximum_retained_coordinates,
        }
    }
}
impl GroupLimitsV2 {
    fn core(&self) -> StructuredCalibrationGroupLimitsV2 {
        StructuredCalibrationGroupLimitsV2 {
            maximum_children: self.maximum_children,
            maximum_total_file_bytes: self.maximum_total_file_bytes,
            maximum_retained_numeric_bytes: self.maximum_retained_numeric_bytes,
            maximum_retained_coordinates: self.maximum_retained_coordinates,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupCaptureConfigV2 {
    /// Source4 mode. Every child source must equal this path and its profile
    /// must equal catalog; those files are physically created only once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_source: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warmup: Vec<manifest::Cohort>,
    pub catalog: PathBuf,
    pub children: Vec<CaptureConfigV2>,
    #[serde(default)]
    pub limits: GroupLimitsV2,
}
impl GroupCaptureConfigV2 {
    pub(crate) fn validate(&self, manifest: &manifest::Manifest) -> Result<()> {
        if self.children.is_empty()
            || self.children.len() > self.limits.maximum_children.get()
            || self.limits.maximum_children.get() > 128
            || self.limits.maximum_total_file_bytes.get() > 8 * 1024 * 1024 * 1024
            || self.limits.maximum_retained_numeric_bytes.get() > 512 * 1024 * 1024
            || self.limits.maximum_retained_coordinates.get() > 16_777_216
            || self.catalog.as_os_str().is_empty()
        {
            return Err(FerrumError::config(
                "invalid structured group limits or catalog destination",
            ));
        }
        let mut files = 0_u64;
        let mut coordinates = 0_usize;
        let mut destinations = vec![self.catalog.as_path()];
        for (index, child) in self.children.iter().enumerate() {
            child.validate(manifest)?;
            if !child.warmup.is_empty()
                || child.maximum_offered_waves != self.children[0].maximum_offered_waves
                || self.children[..index]
                    .iter()
                    .any(|old| old.scope.owner == child.scope.owner)
            {
                return Err(FerrumError::config("group requires shared warmup and FIFO capacity with distinct predeclared owners"));
            }
            if let Some(shared) = &self.shared_source {
                if shared.as_os_str().is_empty()
                    || shared == &self.catalog
                    || &child.source != shared
                    || child.profile != self.catalog
                    || child.maximum_file_bytes != self.children[0].maximum_file_bytes
                {
                    return Err(FerrumError::config("source4 requires common source/byte limit and profile11 catalog for every declared child"));
                }
            } else {
                for destination in [&child.profile, &child.source] {
                    if destination.as_os_str().is_empty()
                        || destinations.contains(&destination.as_path())
                    {
                        return Err(FerrumError::config(
                            "group source/profile/catalog destinations must be distinct",
                        ));
                    }
                    destinations.push(destination.as_path());
                }
            }
            files = files
                .checked_add(if self.shared_source.is_none() || index == 0 {
                    child.maximum_file_bytes.get()
                } else {
                    0
                })
                .ok_or_else(|| FerrumError::config("group declared file bound overflow"))?;
            for count in child.phase_members {
                coordinates = count
                    .checked_mul(child.settings.max_axes)
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|n| coordinates.checked_add(n))
                    .ok_or_else(|| {
                        FerrumError::config("group declared coordinate bound overflow")
                    })?;
            }
        }
        if files > self.limits.maximum_total_file_bytes.get()
            || coordinates > self.limits.maximum_retained_coordinates.get()
        {
            return Err(FerrumError::config(
                "group aggregate file/coordinate bounds exceeded",
            ));
        }
        Ok(())
    }
    pub(crate) fn resolve_paths(&mut self, parent: &std::path::Path) {
        if let Some(path) = &mut self.shared_source {
            if path.is_relative() {
                *path = parent.join(&*path);
            }
        }
        for path in std::iter::once(&mut self.catalog).chain(
            self.children
                .iter_mut()
                .flat_map(|c| [&mut c.profile, &mut c.source]),
        ) {
            if path.is_relative() {
                *path = parent.join(&*path);
            }
        }
    }
    pub(crate) fn validate_policy(
        &self,
        manifest: &manifest::Manifest,
        policy: &ferrum_types::SloConfig,
    ) -> Result<()> {
        self.validate(manifest)?;
        let observation = &policy.cost_observation;
        let limits = &observation.profile_import;
        limits.validate().map_err(FerrumError::config)?;
        let samples = self
            .children
            .iter()
            .flat_map(|c| c.phase_members)
            .try_fold(0_usize, |sum, n| sum.checked_add(n))
            .ok_or_else(|| FerrumError::config("group sample population overflow"))?;
        let source_bytes = self
            .children
            .iter()
            .enumerate()
            .try_fold(0_u64, |sum, (index, c)| {
                sum.checked_add(if self.shared_source.is_none() || index == 0 {
                    c.maximum_file_bytes.get()
                } else {
                    0
                })
            })
            .ok_or_else(|| FerrumError::config("group source capacity overflow"))?;
        if observation.predictor != ferrum_types::SloCostPredictor::StructuredWholeWaveV2
            || observation.structured_capture
                != ferrum_types::SloStructuredCostCapture::HostSettledV1
            || observation.profile_export.is_some()
            || policy.cost_profile.is_some()
            || limits.declared_local_clock_max_error_ns.is_none()
            || self
                .children
                .iter()
                .any(|c| c.declared_source_clock_error_ns > limits.max_clock_error_ns)
            || source_bytes >= limits.max_file_bytes.get() as u64
            || samples > limits.max_samples.get()
        {
            return Err(FerrumError::config("group needs fresh HostSettledV1/V2 capture, explicit original-clock accuracy and aggregate import capacity for all sources/profile10/catalog envelopes"));
        }
        Ok(())
    }
    pub(super) fn options(
        &self,
        manifest: &manifest::Manifest,
        inputs: &inputs::PreparedInputs,
        session: &CalibrationSession,
    ) -> Result<StructuredCalibrationGroupOptionsV2> {
        self.validate(manifest)?;
        Ok(StructuredCalibrationGroupOptionsV2 {
            shared_source: self.shared_source.clone(),
            children: self
                .children
                .iter()
                .map(|c| c.options(manifest, inputs, session))
                .collect::<Result<_>>()?,
            limits: self.limits.core(),
        })
    }
}
