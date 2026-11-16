//! Request-owned PDD continuation over the existing Host memory authority.
//!
//! Arithmetic follows FastVideo `866a3cdb91a85aef8561847365fc8564add9e4c8`
//! `models/schedulers/scheduling_minimax_h3.py` and its denoising stage:
//! F32 packed samples, output promoted to F32, `t=1-sigma`, separate multiply
//! and add operations, and independent video/audio sigma grids. This Host
//! execution does not establish CUDA trajectory parity or a completion fence.

use std::{ops::Range, sync::Arc};

use anyhow::{ensure, Context, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostBuffer, ManagedHostBytes,
    ManagedHostTensor, RequestIdentity, ResolvedTensorLayout, ResolvedTensorSpec,
};
use half::bf16;

use super::{FastH3ModalityStep, FastH3PackedLayout, FastH3PddPlan, FastH3PddStep};

/// A borrowed numerical input. The execution owner must establish completion
/// and bind the physical output to its submitted batch before creating this
/// view; shape, request and step labels alone do not prove device execution.
#[derive(Clone, Copy)]
pub struct FastH3VelocityInput<'a> {
    tensor: &'a ResolvedTensorSpec,
    bytes: &'a ManagedHostBytes,
}

impl<'a> FastH3VelocityInput<'a> {
    pub fn new(tensor: &'a ResolvedTensorSpec, bytes: &'a ManagedHostBytes) -> Self {
        Self { tensor, bytes }
    }
}

/// Per-request, per-step prediction view after the execution owner has applied
/// the exact batch member mapping. It must already contain this interval's
/// fused PDD head prediction, not every fine-grid head or another interval.
pub struct FastH3PredictionView<'a> {
    request: &'a RequestIdentity,
    step_index: usize,
    video: FastH3VelocityInput<'a>,
    audio: FastH3VelocityInput<'a>,
}

impl<'a> FastH3PredictionView<'a> {
    pub fn new(
        request: &'a RequestIdentity,
        step_index: usize,
        video: FastH3VelocityInput<'a>,
        audio: FastH3VelocityInput<'a>,
    ) -> Self {
        Self {
            request,
            step_index,
            video,
            audio,
        }
    }
}

struct StreamSpec {
    tensor: ResolvedTensorSpec,
    bytes: usize,
    target_bytes: Range<usize>,
}

impl StreamSpec {
    fn new(rows: Range<usize>, width: usize) -> Result<Self> {
        ensure!(
            rows.start < rows.end && width > 0,
            "empty FastH3 target stream"
        );
        let tensor = ResolvedTensorSpec::new(
            vec![u64::try_from(rows.end)?, u64::try_from(width)?],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let bytes = usize::try_from(tensor.minimum_storage_bytes()?)?;
        ensure!(
            bytes > 0 && bytes <= isize::MAX as usize,
            "FastH3 stream exceeds Host address range"
        );
        let start = rows
            .start
            .checked_mul(width)
            .and_then(|value| value.checked_mul(4))
            .context("FastH3 target byte offset overflow")?;
        Ok(Self {
            tensor,
            bytes,
            target_bytes: start..bytes,
        })
    }
}

/// Exact successor allocation demand. The caller must also account for all
/// live samples, output readbacks and its other work in the same Host root.
/// This is a description; it does not reserve or create a second memory ledger.
pub struct FastH3SamplingHostRequirements {
    video: StreamSpec,
    audio: StreamSpec,
}

impl FastH3SamplingHostRequirements {
    pub fn new(plan: &FastH3PddPlan, layout: &FastH3PackedLayout) -> Result<Self> {
        let video = StreamSpec::new(layout.target_video_rows(), plan.video_head_width())?;
        let audio = StreamSpec::new(layout.target_audio_rows(), plan.audio_head_width())?;
        video
            .bytes
            .checked_add(audio.bytes)
            .context("joint FastH3 successor byte count overflow")?;
        Ok(Self { video, audio })
    }
    pub fn video_tensor(&self) -> &ResolvedTensorSpec {
        &self.video.tensor
    }
    pub fn audio_tensor(&self) -> &ResolvedTensorSpec {
        &self.audio.tensor
    }
    pub fn video_successor_bytes(&self) -> usize {
        self.video.bytes
    }
    pub fn audio_successor_bytes(&self) -> usize {
        self.audio.bytes
    }
    pub fn successor_bytes(&self) -> usize {
        self.video.bytes + self.audio.bytes
    }
}

/// One quiescent request's committed solver frontier and latent owners. The
/// plan is immutable and shared; all changing data is local to this request.
pub struct FastH3ManagedContinuation {
    request: RequestIdentity,
    plan: Arc<FastH3PddPlan>,
    step_index: usize,
    host: HostMemoryAccount,
    requirements: FastH3SamplingHostRequirements,
    video: ManagedHostTensor,
    audio: ManagedHostTensor,
}

/// Both private successors and the exclusive frontier borrow. Preparing every
/// batch member before committing any member supports all-or-nothing solver
/// publication; dropping a preparation leaves its committed state unchanged.
#[must_use = "commit both streams or drop to preserve the committed frontier"]
pub struct PreparedFastH3Advance<'a> {
    continuation: &'a mut FastH3ManagedContinuation,
    video: ManagedHostTensor,
    audio: ManagedHostTensor,
}

impl PreparedFastH3Advance<'_> {
    pub fn step_index(&self) -> usize {
        self.continuation.step_index
    }
    pub fn step(&self) -> &FastH3PddStep {
        &self.continuation.plan.steps()[self.step_index()]
    }
    /// This publishes only solver state. Device retirement, downstream decode,
    /// output delivery and the enclosing batch transaction remain caller-owned.
    pub fn commit(self) {
        self.continuation.video = self.video;
        self.continuation.audio = self.audio;
        self.continuation.step_index += 1;
    }
}

impl FastH3ManagedContinuation {
    pub const NUMERICAL_PROFILE_ID: &'static str = "fast_h3.pdd.clean_time.f32.separate_mul_add.v1";

    /// Start from caller-generated unit target noise and already encoded frozen
    /// reference prefixes. Copies both admitted streams, scales only targets by
    /// their initial sigma, and consumes no RNG or step. On any error the inputs
    /// and Host account retain their original state.
    pub fn initialize(
        request: RequestIdentity,
        plan: Arc<FastH3PddPlan>,
        layout: &FastH3PackedLayout,
        host: HostMemoryAccount,
        video_noise: &ManagedHostTensor,
        audio_noise: &ManagedHostTensor,
    ) -> Result<Self> {
        let requirements = FastH3SamplingHostRequirements::new(&plan, layout)?;
        ensure!(
            video_noise.value_id() != audio_noise.value_id(),
            "FastH3 stream identities must differ"
        );
        validate_sample(&host, &requirements.video, video_noise)?;
        validate_sample(&host, &requirements.audio, audio_noise)?;
        let mut video_bytes =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, requirements.video.bytes)?;
        let mut audio_bytes =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, requirements.audio.bytes)?;
        initialize_stream(
            video_noise,
            &requirements.video,
            plan.initial_video_noise_scale(),
            &mut video_bytes,
        )?;
        initialize_stream(
            audio_noise,
            &requirements.audio,
            plan.initial_audio_noise_scale(),
            &mut audio_bytes,
        )?;
        let video = successor(video_noise, &requirements.video, video_bytes)?;
        let audio = successor(audio_noise, &requirements.audio, audio_bytes)?;
        Ok(Self {
            request,
            plan,
            step_index: 0,
            host,
            requirements,
            video,
            audio,
        })
    }
    pub fn request(&self) -> &RequestIdentity {
        &self.request
    }
    pub fn step_index(&self) -> usize {
        self.step_index
    }
    pub fn is_complete(&self) -> bool {
        self.step_index == self.plan.steps().len()
    }
    pub fn current_step(&self) -> Option<&FastH3PddStep> {
        self.plan.steps().get(self.step_index)
    }
    pub fn video(&self) -> &ManagedHostTensor {
        &self.video
    }
    pub fn audio(&self) -> &ManagedHostTensor {
        &self.audio
    }
    pub fn host_requirements(&self) -> &FastH3SamplingHostRequirements {
        &self.requirements
    }

    pub fn prepare_advance(
        &mut self,
        prediction: FastH3PredictionView<'_>,
    ) -> Result<PreparedFastH3Advance<'_>> {
        let step = self
            .current_step()
            .context("FastH3 sampling is already complete")?;
        ensure!(
            prediction.request == &self.request && prediction.step_index == self.step_index,
            "FastH3 prediction request or step does not match its continuation"
        );
        validate_velocity(&self.host, &self.requirements.video, prediction.video)?;
        validate_velocity(&self.host, &self.requirements.audio, prediction.audio)?;
        let mut video_bytes = self.host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            self.requirements.video.bytes,
        )?;
        let mut audio_bytes = self.host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            self.requirements.audio.bytes,
        )?;
        advance_stream(
            &self.video,
            prediction.video,
            &self.requirements.video,
            &step.video,
            &mut video_bytes,
        )?;
        advance_stream(
            &self.audio,
            prediction.audio,
            &self.requirements.audio,
            &step.audio,
            &mut audio_bytes,
        )?;
        let video = successor(&self.video, &self.requirements.video, video_bytes)?;
        let audio = successor(&self.audio, &self.requirements.audio, audio_bytes)?;
        Ok(PreparedFastH3Advance {
            continuation: self,
            video,
            audio,
        })
    }
}

fn validate_sample(
    host: &HostMemoryAccount,
    spec: &StreamSpec,
    sample: &ManagedHostTensor,
) -> Result<()> {
    ensure!(
        sample.tensor() == &spec.tensor && sample.bytes().len() == spec.bytes,
        "FastH3 sample tensor or byte extent differs"
    );
    ensure!(
        host.owns(sample.bytes()) && sample.bytes().class() == HostMemoryClass::InputPreprocess,
        "FastH3 sample has foreign Host ownership or wrong class"
    );
    Ok(())
}

fn validate_velocity(
    host: &HostMemoryAccount,
    spec: &StreamSpec,
    velocity: FastH3VelocityInput<'_>,
) -> Result<()> {
    ensure!(
        host.owns(velocity.bytes) && velocity.bytes.class() == HostMemoryClass::Output,
        "FastH3 prediction has foreign Host ownership or wrong class"
    );
    ensure!(
        velocity.tensor.dimensions() == spec.tensor.dimensions()
            && velocity.tensor.layout() == &ResolvedTensorLayout::Contiguous
            && matches!(
                velocity.tensor.element_type(),
                ElementType::F32 | ElementType::Bf16
            ),
        "FastH3 prediction must be a matching contiguous F32 or BF16 row matrix"
    );
    ensure!(
        velocity.tensor.minimum_storage_bytes()? == u64::try_from(velocity.bytes.len())?,
        "FastH3 prediction byte extent differs"
    );
    Ok(())
}

fn initialize_stream(
    sample: &ManagedHostTensor,
    spec: &StreamSpec,
    scale: f32,
    destination: &mut ManagedHostBuffer,
) -> Result<()> {
    destination.as_mut().copy_from_slice(sample.bytes().bytes());
    for (index, bytes) in destination.as_mut().chunks_exact_mut(4).enumerate() {
        let value = decode_f32(bytes);
        ensure!(
            value.is_finite(),
            "FastH3 initial sample {index} is nonfinite"
        );
        if index * 4 >= spec.target_bytes.start {
            let scaled = value * scale;
            ensure!(
                scaled.is_finite(),
                "FastH3 initial scaled target {index} is nonfinite"
            );
            bytes.copy_from_slice(&scaled.to_le_bytes());
        }
    }
    Ok(())
}

fn advance_stream(
    sample: &ManagedHostTensor,
    velocity: FastH3VelocityInput<'_>,
    spec: &StreamSpec,
    step: &FastH3ModalityStep,
    destination: &mut ManagedHostBuffer,
) -> Result<()> {
    let frozen = spec.target_bytes.start;
    destination.as_mut()[..frozen].copy_from_slice(&sample.bytes().bytes()[..frozen]);
    let element_bytes = if velocity.tensor.element_type() == ElementType::F32 {
        4
    } else {
        2
    };
    let velocity_start = frozen / 4 * element_bytes;
    for (index, ((sample, prediction), output)) in sample.bytes().bytes()[frozen..]
        .chunks_exact(4)
        .zip(velocity.bytes.bytes()[velocity_start..].chunks_exact(element_bytes))
        .zip(destination.as_mut()[frozen..].chunks_exact_mut(4))
        .enumerate()
    {
        let x = decode_f32(sample);
        let v = if element_bytes == 4 {
            decode_f32(prediction)
        } else {
            bf16::from_bits(u16::from_le_bytes([prediction[0], prediction[1]])).to_f32()
        };
        ensure!(
            x.is_finite() && v.is_finite(),
            "FastH3 target sample or prediction {index} is nonfinite"
        );
        // Preserve the executable reference's data-ward output convention and
        // its separate sigma round trip. No negative sign, direct dt shortcut,
        // extra PDD block-weight multiplier, or fused multiply-add is applied.
        let sigma_from_timestep = 1.0_f32 - step.timestep();
        let denoised = x + sigma_from_timestep * v;
        let ratio = step.next_sigma() / step.sigma();
        let updated = ratio * x + (1.0_f32 - ratio) * denoised;
        ensure!(
            updated.is_finite(),
            "FastH3 target successor {index} is nonfinite"
        );
        output.copy_from_slice(&updated.to_le_bytes());
    }
    Ok(())
}

fn successor(
    sample: &ManagedHostTensor,
    spec: &StreamSpec,
    bytes: ManagedHostBuffer,
) -> Result<ManagedHostTensor> {
    Ok(ManagedHostTensor::new(
        sample.value_id().clone(),
        spec.tensor.clone(),
        bytes.freeze(),
    )?)
}

fn decode_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests;
