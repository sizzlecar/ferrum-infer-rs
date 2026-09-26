use super::*;

/// Bounds apply to the declared complete run, not to a post hoc subset of
/// successful children. File/numeric totals are checked before source creation.
#[derive(Debug, Clone)]
pub struct StructuredCalibrationGroupLimitsV2 {
    pub maximum_children: NonZeroUsize,
    pub maximum_total_file_bytes: NonZeroU64,
    pub maximum_retained_numeric_bytes: NonZeroUsize,
    pub maximum_retained_coordinates: NonZeroUsize,
}
impl Default for StructuredCalibrationGroupLimitsV2 {
    fn default() -> Self {
        Self {
            maximum_children: NonZeroUsize::new(8).unwrap(),
            maximum_total_file_bytes: NonZeroU64::new(512 * 1024 * 1024).unwrap(),
            maximum_retained_numeric_bytes: NonZeroUsize::new(128 * 1024 * 1024).unwrap(),
            maximum_retained_coordinates: NonZeroUsize::new(1_048_576).unwrap(),
        }
    }
}
#[derive(Debug, Clone)]
pub struct StructuredCalibrationGroupOptionsV2 {
    /// Explicit shared source4; None preserves independent source3 files.
    pub shared_source: Option<PathBuf>,
    pub children: Vec<StructuredCalibrationOptionsV2>,
    pub limits: StructuredCalibrationGroupLimitsV2,
}
impl StructuredCalibrationGroupOptionsV2 {
    pub(super) fn validate(&self) -> Result<(), ExportError> {
        if self.children.is_empty()
            || self.children.len() > self.limits.maximum_children.get()
            || self.limits.maximum_children.get() > 128
            || self.limits.maximum_total_file_bytes.get() > 8 * 1024 * 1024 * 1024
            || self.limits.maximum_retained_numeric_bytes.get() > 512 * 1024 * 1024
            || self.limits.maximum_retained_coordinates.get() > 16_777_216
        {
            return Err(ExportError::Config(
                "invalid structured group bounds".into(),
            ));
        }
        let first = &self.children[0];
        let mut files = 0_u64;
        let mut numeric = 0_usize;
        let mut coordinates = 0_usize;
        let mut destinations = Vec::with_capacity(self.children.len());
        for (index, child) in self.children.iter().enumerate() {
            child.validate()?;
            // Resolve parent aliases before any source file is created.
            let destination = files::destination(&child.observations_path)?;
            if self.shared_source.is_none() && destinations.contains(&destination) {
                return Err(ExportError::Config(
                    "group child files alias the same destination".into(),
                ));
            }
            destinations.push(destination);
            if child.cohort_plan != first.cohort_plan
                || child.cohort_manifest_payload != first.cohort_manifest_payload
                || child.maximum_offered_waves != first.maximum_offered_waves
                || self.children[..index].iter().any(|old| {
                    old.scope.owner == child.scope.owner
                        || (self.shared_source.is_none()
                            && old.observations_path == child.observations_path)
                })
            {
                return Err(ExportError::Config(
                    "group children require unique owners/files and identical complete cohorts"
                        .into(),
                ));
            }
            if let Some(shared) = &self.shared_source {
                if &child.observations_path != shared
                    || child.maximum_file_bytes != first.maximum_file_bytes
                {
                    return Err(ExportError::Config(
                        "shared children must declare the same physical source and byte limit"
                            .into(),
                    ));
                }
            }
            files = files
                .checked_add(if self.shared_source.is_none() || index == 0 {
                    child.maximum_file_bytes.get()
                } else {
                    0
                })
                .ok_or(ExportError::Source("group file capacity overflow"))?;
            numeric = numeric
                .checked_add(
                    child
                        .numeric_storage_bound()
                        .ok_or(ExportError::Source("child numeric capacity overflow"))?,
                )
                .ok_or(ExportError::Source("group numeric capacity overflow"))?;
            let child_coordinates = child
                .phase_members
                .iter()
                .try_fold(0_usize, |sum, count| {
                    count
                        .checked_mul(child.settings.max_axes)?
                        .checked_mul(2)?
                        .checked_add(sum)
                })
                .ok_or(ExportError::Source("child coordinate capacity overflow"))?;
            coordinates = coordinates
                .checked_add(child_coordinates)
                .ok_or(ExportError::Source("group coordinate capacity overflow"))?;
        }
        if files > self.limits.maximum_total_file_bytes.get()
            || numeric > self.limits.maximum_retained_numeric_bytes.get()
            || coordinates > self.limits.maximum_retained_coordinates.get()
        {
            return Err(ExportError::Config(
                "structured group aggregate budget exceeded".into(),
            ));
        }
        Ok(())
    }
}
