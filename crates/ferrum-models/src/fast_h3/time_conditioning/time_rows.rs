use super::*;
use crate::fast_h3::{FastH3PackedLayout, FastH3PddPlan, FastH3PddStep, FastH3SegmentKind};

/// Compact row-time routing for one document. Text (including its vision
/// tokens) follows target video time; retained video uses max(t_video,.999),
/// retained audio uses one. Tags remain a separate conditioning input.
#[derive(Clone, Debug)]
pub struct FastH3ConditioningClock {
    times: Vec<f32>,
    video: f32,
    audio: f32,
}

/// Immutable union of row times for the complete declared sampling plan. A
/// consumer can precompute one small modulation table per block, then release
/// its full-rank projection. Request progress still belongs to the scheduler.
#[derive(Clone, Debug)]
pub struct FastH3ConditioningSchedule {
    clocks: Vec<FastH3ConditioningClock>,
    times: Vec<f32>,
}
impl FastH3ConditioningSchedule {
    /// Include each supported modality even when the eventual request omits a
    /// reference. This reusable cold table depends on the trained schedule,
    /// not on token counts, image dimensions or mutable request progress.
    pub fn all_modalities(plan: &FastH3PddPlan) -> Result<Self, VNextError> {
        use FastH3SegmentKind::*;
        Self::from_clocks(
            plan.steps()
                .iter()
                .map(|step| {
                    FastH3ConditioningClock::from_kinds(
                        [
                            Text,
                            ReferenceImage,
                            ReferenceVideo,
                            ReferenceAudio,
                            TargetVideo,
                            TargetAudio,
                        ],
                        step,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
    pub fn new(layout: &FastH3PackedLayout, plan: &FastH3PddPlan) -> Result<Self, VNextError> {
        let clocks = plan
            .steps()
            .iter()
            .map(|step| FastH3ConditioningClock::new(layout, step))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_clocks(clocks)
    }
    fn from_clocks(clocks: Vec<FastH3ConditioningClock>) -> Result<Self, VNextError> {
        let mut times: Vec<_> = clocks
            .iter()
            .flat_map(|c| c.timesteps().iter().copied())
            .collect();
        times.sort_by(f32::total_cmp);
        times.dedup();
        if times.is_empty() {
            return Err(invalid(
                "conditioning schedule requires a nonempty document and plan",
            ));
        }
        Ok(Self { clocks, times })
    }
    pub fn timesteps(&self) -> &[f32] {
        &self.times
    }
    pub fn evaluation_count(&self) -> usize {
        self.clocks.len()
    }
    pub fn segment_time_index(&self, evaluation: usize, kind: FastH3SegmentKind) -> Option<usize> {
        let clock = self.clocks.get(evaluation)?;
        clock.segment_time_index(kind)?;
        self.times.iter().position(|&time| time == clock.time(kind))
    }
}
impl FastH3ConditioningClock {
    pub fn new(layout: &FastH3PackedLayout, step: &FastH3PddStep) -> Result<Self, VNextError> {
        Self::from_kinds(layout.segments().iter().map(|s| s.kind), step)
    }
    fn from_kinds(
        kinds: impl IntoIterator<Item = FastH3SegmentKind>,
        step: &FastH3PddStep,
    ) -> Result<Self, VNextError> {
        let video = step.video.timestep();
        let audio = step.audio.timestep();
        if [video, audio]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(invalid("conditioning times must be finite flow times"));
        }
        let mut clock = Self {
            times: Vec::with_capacity(4),
            video,
            audio,
        };
        for kind in kinds {
            let time = clock.time(kind);
            if !clock.times.contains(&time) {
                clock.times.push(time);
            }
        }
        clock.times.sort_by(f32::total_cmp);
        Ok(clock)
    }
    pub fn timesteps(&self) -> &[f32] {
        &self.times
    }
    pub fn segment_time_index(&self, kind: FastH3SegmentKind) -> Option<usize> {
        self.times.iter().position(|&time| time == self.time(kind))
    }
    fn time(&self, kind: FastH3SegmentKind) -> f32 {
        match kind {
            FastH3SegmentKind::Text | FastH3SegmentKind::TargetVideo => self.video,
            FastH3SegmentKind::TargetAudio => self.audio,
            FastH3SegmentKind::ReferenceAudio => 1.,
            FastH3SegmentKind::ReferenceImage | FastH3SegmentKind::ReferenceVideo => {
                self.video.max(0.999_f32)
            }
        }
    }
}

impl FastH3TimeConditioningConfig {
    /// Explicit CPU libm-0.2.15 sinusoidal feature profile (cos then sin,
    /// max_period=10000, downscale_freq_shift=0, scale=1). Each ordinary
    /// arithmetic boundary is F32. This is not a CUDA/Torch bitwise claim.
    /// Its output is admitted before generation and feeds this same component.
    pub fn managed_features(
        &self,
        host: &HostMemoryAccount,
        clock: &FastH3ConditioningClock,
    ) -> anyhow::Result<ManagedHostTensor> {
        self.managed_features_for_times(host, clock.timesteps())
    }
    pub fn managed_schedule_features(
        &self,
        host: &HostMemoryAccount,
        schedule: &FastH3ConditioningSchedule,
    ) -> anyhow::Result<ManagedHostTensor> {
        self.managed_features_for_times(host, schedule.timesteps())
    }
    fn managed_features_for_times(
        &self,
        host: &HostMemoryAccount,
        times: &[f32],
    ) -> anyhow::Result<ManagedHostTensor> {
        if times.len() as u64 != self.rows || self.frequency == 0 || self.frequency % 2 != 0 {
            return Err(invalid("time table rows differ from conditioning clock").into());
        }
        let tensor = ResolvedTensorSpec::new(
            vec![self.rows, self.frequency],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let bytes = usize::try_from(tensor.minimum_storage_bytes()?)
            .ok()
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid("time features exceed Host address space"))?;
        let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let width =
            usize::try_from(self.frequency).map_err(|_| invalid("time width exceeds usize"))?;
        write_features(times, width, output.as_mut());
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(FastH3TimeConditioningFamily::INPUT_ID)?,
            tensor,
            output.freeze(),
        )?)
    }
}
fn write_features(times: &[f32], width: usize, bytes: &mut [u8]) {
    let half = width / 2;
    let coefficient = -(libm::log(10000.0) as f32);
    for (&time, row) in times.iter().zip(bytes.chunks_exact_mut(width * 4)) {
        for column in 0..half {
            let exponent = (coefficient * column as f32) / half as f32;
            let angle = time * libm::expf(exponent);
            row[column * 4..column * 4 + 4].copy_from_slice(&libm::cosf(angle).to_le_bytes());
            let offset = (column + half) * 4;
            row[offset..offset + 4].copy_from_slice(&libm::sinf(angle).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_h3::{
        FastH3Checkpoint, FastH3MetadataLimits, FastH3PddPlan, FastH3Reference, FastH3VideoGeometry,
    };
    #[test]
    fn reference_clocks_remain_fixed_while_text_tracks_each_requests_video_time() {
        let checkpoint = FastH3Checkpoint::parse(
            include_bytes!("../fixtures/inference.json"),
            include_bytes!("../fixtures/transformer.json"),
            include_bytes!("../fixtures/video_scheduler.json"),
            include_bytes!("../fixtures/audio_scheduler.json"),
            FastH3MetadataLimits {
                max_file_bytes: 4096,
                max_pdd_steps: 64,
            },
        )
        .unwrap();
        let plan = FastH3PddPlan::new(&checkpoint).unwrap();
        let layout = FastH3PackedLayout::ref2va(
            7,
            &[
                FastH3Reference::Image {
                    latent_height: 2,
                    latent_width: 2,
                },
                FastH3Reference::Audio {
                    latents_per_channel: 4,
                },
            ],
            &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
            128,
        )
        .unwrap();
        let schedule = FastH3ConditioningSchedule::all_modalities(&plan).unwrap();
        assert_eq!(schedule.evaluation_count(), plan.steps().len());
        assert!(schedule.timesteps().windows(2).all(|v| v[0] < v[1]));
        let document_schedule = FastH3ConditioningSchedule::new(&layout, &plan).unwrap();
        for (index, step) in plan.steps().iter().enumerate() {
            let clock = FastH3ConditioningClock::new(&layout, step).unwrap();
            for segment in layout.segments() {
                let expected = clock.timesteps()[clock.segment_time_index(segment.kind).unwrap()];
                assert_eq!(
                    schedule.timesteps()[schedule.segment_time_index(index, segment.kind).unwrap()],
                    expected
                );
                assert_eq!(
                    document_schedule.timesteps()[document_schedule
                        .segment_time_index(index, segment.kind)
                        .unwrap()],
                    expected
                );
            }
        }
        assert!(schedule
            .segment_time_index(plan.steps().len(), FastH3SegmentKind::TargetVideo)
            .is_none());
        for step in [&plan.steps()[1], &plan.steps()[6]] {
            let clock = FastH3ConditioningClock::new(&layout, step).unwrap();
            let time = |kind| clock.timesteps()[clock.segment_time_index(kind).unwrap()];
            assert_eq!(time(FastH3SegmentKind::Text), step.video.timestep());
            assert_eq!(time(FastH3SegmentKind::TargetVideo), step.video.timestep());
            assert_eq!(time(FastH3SegmentKind::TargetAudio), step.audio.timestep());
            assert_eq!(time(FastH3SegmentKind::ReferenceImage), 0.999_f32);
            assert_eq!(time(FastH3SegmentKind::ReferenceAudio), 1.);
            assert!(clock.timesteps().windows(2).all(|v| v[0] < v[1]));
        }
    }
    #[test]
    fn sinusoidal_order_and_frequency_scale_match_the_declared_analytic_basis() {
        let mut bytes = [0u8; 64];
        write_features(&[0., 1.], 8, &mut bytes);
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(&values[..8], &[1., 1., 1., 1., 0., 0., 0., 0.]);
        for (column, angle) in [1.0f64, 0.1, 0.01, 0.001].into_iter().enumerate() {
            assert!((f64::from(values[8 + column]) - angle.cos()).abs() < 1e-7);
            assert!((f64::from(values[12 + column]) - angle.sin()).abs() < 1e-7);
        }
    }
}
