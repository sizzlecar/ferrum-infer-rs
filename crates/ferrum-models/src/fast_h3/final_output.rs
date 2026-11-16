//! One device graph for final BF16 normalization, modality selection, exact
//! promotion and the two F32 PDD heads. Text rows do not enter output heads.
use super::*;
use crate::tensor_program_fragment::{self as fragment, Fragment};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod family;
mod program;
mod source;
pub use family::FastH3FinalOutputFamily;
pub use source::{FastH3FinalOutputSource, OwnedFastH3FinalOutputSource};

#[derive(Clone, Copy, Debug)]
pub struct FastH3FinalOutputLimits {
    pub conditioning: FastH3OutputConditioningLimits,
    pub projection: FastH3PddProjectionLimits,
    pub maximum_metadata_bytes: usize,
    pub maximum_plan_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_pdd_steps: usize,
}
impl FastH3FinalOutputLimits {
    fn validate(self) -> Result<(), VNextError> {
        FastH3OutputNormFamily::new(self.conditioning)?;
        FastH3PddProjectionFamily::new(self.projection)?;
        if self.maximum_metadata_bytes == 0
            || self.maximum_plan_nodes == 0
            || self.maximum_stored_bytes == 0
            || self.maximum_pdd_steps < 2
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowRun {
    start: u64,
    rows: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Segment {
    kind: FastH3SegmentKind,
    rows: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3FinalOutputConfig {
    norm: FastH3OutputNormConfig,
    video: FastH3PddProjectionConfig,
    audio: FastH3PddProjectionConfig,
    schedule: FastH3PddSchedule,
    segments: Vec<Segment>,
    #[serde(default, skip_serializing_if = "FastH3PredictionInput::is_host")]
    input_residency: FastH3PredictionInput,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3PredictionInput {
    #[default]
    Host,
    RequestState,
}
impl FastH3PredictionInput {
    fn is_host(&self) -> bool {
        matches!(self, Self::Host)
    }
}
impl FastH3FinalOutputConfig {
    pub fn with_input_residency(mut self, residency: FastH3PredictionInput) -> Self {
        self.input_residency = residency;
        self
    }
    pub fn input_residency(&self) -> FastH3PredictionInput {
        self.input_residency
    }
    pub fn normalization(&self) -> &FastH3OutputNormConfig {
        &self.norm
    }
    pub fn projection(&self, modality: FastH3OutputModality) -> &FastH3PddProjectionConfig {
        match modality {
            FastH3OutputModality::Video => &self.video,
            FastH3OutputModality::Audio => &self.audio,
        }
    }
    pub fn compile_options(
        &self,
        limits: FastH3FinalOutputLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(program::compose(self, limits)?.options)
    }
    fn validate(&self, limits: FastH3FinalOutputLimits) -> Result<(), VNextError> {
        limits.validate()?;
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("configuration exceeds metadata capacity"));
        }
        if self.video.modality() != FastH3OutputModality::Video
            || self.audio.modality() != FastH3OutputModality::Audio
            || self.video.evaluation() != self.audio.evaluation()
            || self.video.heads() != self.audio.heads()
            || self.video.hidden_size() != self.norm.hidden_size()
            || self.audio.hidden_size() != self.norm.hidden_size()
        {
            return Err(invalid("normalization and modality heads disagree"));
        }
        use FastH3SegmentKind::*;
        let n = self.segments.len();
        if n < 4
            || self.segments[0].kind != Text
            || self.segments[n - 2].kind != TargetAudio
            || self.segments[n - 1].kind != TargetVideo
            || self.segments[1..n - 2]
                .iter()
                .any(|s| !matches!(s.kind, ReferenceImage | ReferenceVideo | ReferenceAudio))
            || self.segments.iter().any(|s| s.rows == 0)
            || self
                .segments
                .iter()
                .try_fold(0u64, |n, s| n.checked_add(s.rows))
                != Some(self.norm.rows())
        {
            return Err(invalid("invalid Ref2VA output segment partition"));
        }
        for (modality, expected) in [
            (FastH3OutputModality::Video, self.video.rows()),
            (FastH3OutputModality::Audio, self.audio.rows()),
        ] {
            if runs(&self.segments, modality)?
                .iter()
                .try_fold(0u64, |n, r| n.checked_add(r.rows))
                != Some(expected)
            {
                return Err(invalid("modality partition differs from head geometry"));
            }
        }
        let plan = FastH3PddPlan::from_schedule(
            self.schedule.clone(),
            self.video.output_size().try_into().map_err(invalid)?,
            self.audio.output_size().try_into().map_err(invalid)?,
            limits.maximum_pdd_steps,
        )
        .map_err(invalid)?;
        self.video.validate_schedule(&plan)?;
        self.audio.validate_schedule(&plan)?;
        let schedule = FastH3ConditioningSchedule::all_modalities(&plan)?;
        self.norm.validate_schedule(
            &schedule,
            self.video.evaluation(),
            self.segments.iter().map(|s| (s.kind, s.rows)),
        )?;
        Ok(())
    }
}

fn runs(segments: &[Segment], modality: FastH3OutputModality) -> Result<Vec<RowRun>, VNextError> {
    let mut runs = Vec::<RowRun>::new();
    let mut start = 0u64;
    for segment in segments {
        let next = start
            .checked_add(segment.rows)
            .ok_or_else(|| invalid("row extent overflow"))?;
        let selected = match modality {
            FastH3OutputModality::Video => matches!(
                segment.kind,
                FastH3SegmentKind::ReferenceImage
                    | FastH3SegmentKind::ReferenceVideo
                    | FastH3SegmentKind::TargetVideo
            ),
            FastH3OutputModality::Audio => matches!(
                segment.kind,
                FastH3SegmentKind::ReferenceAudio | FastH3SegmentKind::TargetAudio
            ),
        };
        if !selected {
            start = next;
            continue;
        }
        let rows = segment.rows;
        if let Some(last) = runs
            .last_mut()
            .filter(|r| r.start.checked_add(r.rows) == Some(start))
        {
            last.rows = last
                .rows
                .checked_add(rows)
                .ok_or_else(|| invalid("row extent overflow"))?;
        } else {
            runs.push(RowRun { start, rows });
        }
        start = next;
    }
    Ok(runs)
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 final output: {e}"),
    }
}
