//! The FP32 image/audio scheduler with the existing Host ownership authority.
//! No device residency or product continuous-batching capability is implied.
use super::{PH3ImageRequestSchedule, PH3ImageSamplingSchedule, PH3ImageSamplingStep};
use anyhow::{ensure, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_runtime::{
    CompiledTensorPort, ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostBuffer,
    ManagedHostBytes, ManagedHostTensor, RequestIdentity, ResolvedTensorLayout, ResolvedTensorSpec,
};
use half::bf16;
mod velocity_layout;
use velocity_layout::VelocityMapping;
pub use velocity_layout::{PH3ImageVelocityBatchLayout, PH3ImageVelocityRequest};

/// Borrowed readback with its existing Output-class owner. This is deliberately
/// separate from `ManagedHostTensor`, whose contract is an input allocation.
#[derive(Clone, Copy)]
pub struct PH3ImageSamplingVelocity<'a> {
    tensor: &'a ResolvedTensorSpec,
    bytes: &'a ManagedHostBytes,
    mapping: VelocityMapping<'a>,
}
impl<'a> PH3ImageSamplingVelocity<'a> {
    /// The sampling adapter validates shape, dtype, class, extent and account
    /// before allocating successors; this view neither copies nor reclassifies.
    pub fn new(tensor: &'a ResolvedTensorSpec, bytes: &'a ManagedHostBytes) -> Self {
        Self {
            tensor,
            bytes,
            mapping: VelocityMapping::Contiguous,
        }
    }
    /// The original physical matrix and Output owner remain visible. Mapped
    /// views never advertise a NCHW reshape of patch/row storage.
    pub fn physical_tensor(&self) -> &ResolvedTensorSpec {
        self.tensor
    }
    pub fn backing(&self) -> &ManagedHostBytes {
        self.bytes
    }
    pub fn from_output(
        declared: &CompiledTensorPort,
        output: &'a TensorExecutionOutput,
    ) -> Result<Self> {
        ensure!(
            output.value_id() == declared.value_id() && output.tensor() == declared.tensor(),
            "H3 velocity differs from its compiled output port"
        );
        Ok(Self::new(output.tensor(), output.bytes()))
    }
}

/// The completed model output's actual storage, promoted to F32 for integration.
/// Samples and successors always remain F32 in this sampling profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageVelocityPrecision {
    F32,
    Bf16,
}
impl PH3ImageVelocityPrecision {
    fn element_type(self) -> ElementType {
        match self {
            Self::F32 => ElementType::F32,
            Self::Bf16 => ElementType::Bf16,
        }
    }
    fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
    fn decode(self, bytes: &[u8]) -> f32 {
        match self {
            Self::F32 => decode_f32(bytes),
            Self::Bf16 => bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
        }
    }
}

struct StreamSpecs {
    sample: ResolvedTensorSpec,
    velocity: ResolvedTensorSpec,
    sample_bytes: usize,
    velocity_bytes: usize,
}
impl StreamSpecs {
    fn new(sample: ResolvedTensorSpec, precision: PH3ImageVelocityPrecision) -> Result<Self> {
        ensure!(
            sample.element_type() == ElementType::F32
                && sample.layout() == &ResolvedTensorLayout::Contiguous
                && !sample.dimensions().is_empty()
                && !sample.dimensions().contains(&0),
            "H3 managed sampling requires nonempty contiguous F32 samples"
        );
        let velocity = ResolvedTensorSpec::new(
            sample.dimensions().to_vec(),
            precision.element_type(),
            ResolvedTensorLayout::Contiguous,
        )?;
        let extent = |spec: &ResolvedTensorSpec| -> Result<usize> {
            let bytes = usize::try_from(spec.minimum_storage_bytes()?)?;
            ensure!(
                bytes > 0 && bytes <= isize::MAX as usize,
                "H3 sampling exceeds Host address range"
            );
            Ok(bytes)
        };
        Ok(Self {
            sample_bytes: extent(&sample)?,
            velocity_bytes: extent(&velocity)?,
            sample,
            velocity,
        })
    }
}

/// One request's cursor and permitted image/audio tensor shapes. The existing
/// Host account grants every physical backing; this object has no separate
/// capacity ledger and does not reserve resources for a future step itself.
pub struct PH3ImageManagedSampling {
    cursor: PH3ImageRequestSchedule,
    host: HostMemoryAccount,
    image: StreamSpecs,
    audio: StreamSpecs,
    precision: PH3ImageVelocityPrecision,
    request: Option<RequestIdentity>,
}

/// Both successor streams are private until commit. Dropping this value releases
/// both new backings and leaves the exclusively borrowed request cursor intact.
/// A physical batch can prepare every request before committing any of them.
#[must_use = "commit both streams, or drop to retain the original cursor"]
pub struct PreparedPH3ImageSamplingAdvance<'a> {
    sampling: &'a mut PH3ImageManagedSampling,
    step: PH3ImageSamplingStep,
    image: ManagedHostTensor,
    audio: ManagedHostTensor,
}
impl PreparedPH3ImageSamplingAdvance<'_> {
    pub fn step(&self) -> PH3ImageSamplingStep {
        self.step
    }
    pub fn commit(self) -> (PH3ImageSamplingStep, ManagedHostTensor, ManagedHostTensor) {
        // All fallible work has already completed. No partial image/audio
        // publication or capacity-dependent operation is possible here.
        self.sampling.cursor.step_index += 1;
        (self.step, self.image, self.audio)
    }
}

impl PH3ImageManagedSampling {
    pub const NUMERICAL_PROFILE_ID: &'static str = "p_h3_image.euler.f32.separate_mul_add.v1";

    pub fn new(
        schedule: &PH3ImageSamplingSchedule,
        host: HostMemoryAccount,
        image: ResolvedTensorSpec,
        audio: ResolvedTensorSpec,
        precision: PH3ImageVelocityPrecision,
    ) -> Result<Self> {
        Ok(Self {
            cursor: schedule.start(),
            host,
            image: StreamSpecs::new(image, precision)?,
            audio: StreamSpecs::new(audio, precision)?,
            precision,
            request: None,
        })
    }
    pub fn new_for_request(
        schedule: &PH3ImageSamplingSchedule,
        host: HostMemoryAccount,
        image: ResolvedTensorSpec,
        audio: ResolvedTensorSpec,
        precision: PH3ImageVelocityPrecision,
        request: RequestIdentity,
    ) -> Result<Self> {
        ensure!(
            matches!(image.dimensions(), [1, _, _, _] | [1, _, 1, _, _])
                && matches!(audio.dimensions(), [1, _, 2, 2]),
            "request-bound H3 sampling requires single-image NCHW/T=1 and stereo audio NCHW shapes"
        );
        let mut s = Self::new(schedule, host, image, audio, precision)?;
        s.request = Some(request);
        Ok(s)
    }
    pub fn step_index(&self) -> usize {
        self.cursor.step_index()
    }
    pub fn current_step(&self) -> Option<PH3ImageSamplingStep> {
        self.cursor.current_step()
    }
    pub fn is_complete(&self) -> bool {
        self.cursor.is_complete()
    }
    pub fn velocity_precision(&self) -> PH3ImageVelocityPrecision {
        self.precision
    }

    /// Consume completed Output-class velocities and InputPreprocess samples
    /// belonging to the same Host root. Both streams are mandatory, even when
    /// only the final image will be returned to a caller. The next samples keep
    /// their original value IDs and have independently charged physical owners.
    pub fn prepare_advance(
        &mut self,
        image: &ManagedHostTensor,
        image_velocity: PH3ImageSamplingVelocity<'_>,
        audio: &ManagedHostTensor,
        audio_velocity: PH3ImageSamplingVelocity<'_>,
    ) -> Result<PreparedPH3ImageSamplingAdvance<'_>> {
        let step = self
            .current_step()
            .context("H3 sampling is already complete")?;
        ensure!(
            image.value_id() != audio.value_id(),
            "H3 image/audio value identities must differ"
        );
        self.validate_stream("image", &self.image, image, image_velocity)?;
        self.validate_stream("audio", &self.audio, audio, audio_velocity)?;
        // Admission of the second stream may fail. The first private allocation
        // then drops normally, without changing either sample or the cursor.
        let mut image_bytes = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, self.image.sample_bytes)?;
        let mut audio_bytes = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, self.audio.sample_bytes)?;
        self.write_successor(step, image, image_velocity, &mut image_bytes)?;
        self.write_successor(step, audio, audio_velocity, &mut audio_bytes)?;
        let image = ManagedHostTensor::new(
            image.value_id().clone(),
            self.image.sample.clone(),
            image_bytes.freeze(),
        )?;
        let audio = ManagedHostTensor::new(
            audio.value_id().clone(),
            self.audio.sample.clone(),
            audio_bytes.freeze(),
        )?;
        Ok(PreparedPH3ImageSamplingAdvance {
            sampling: self,
            step,
            image,
            audio,
        })
    }

    fn validate_stream(
        &self,
        name: &str,
        specs: &StreamSpecs,
        sample: &ManagedHostTensor,
        velocity: PH3ImageSamplingVelocity<'_>,
    ) -> Result<()> {
        ensure!(
            sample.tensor() == &specs.sample && sample.bytes().len() == specs.sample_bytes,
            "H3 {name} sample shape/dtype/layout or byte extent differs"
        );
        ensure!(
            self.host.owns(sample.bytes())
                && sample.bytes().class() == HostMemoryClass::InputPreprocess,
            "H3 {name} sample has foreign Host owner or wrong class"
        );
        ensure!(
            self.host.owns(velocity.bytes) && velocity.bytes.class() == HostMemoryClass::Output,
            "H3 {name} velocity has foreign Host owner or wrong class"
        );
        ensure!(
            velocity.tensor.element_type() == self.precision.element_type()
                && velocity.tensor.layout() == &ResolvedTensorLayout::Contiguous
                && velocity.tensor.minimum_storage_bytes()? == u64::try_from(velocity.bytes.len())?,
            "H3 {name} velocity physical precision/layout/extent differs"
        );
        velocity
            .mapping
            .validate(name, self.request.as_ref(), &specs.sample)?;
        if matches!(velocity.mapping, VelocityMapping::Contiguous) {
            ensure!(
                velocity.tensor == &specs.velocity && velocity.bytes.len() == specs.velocity_bytes,
                "H3 {name} velocity shape/dtype/layout or byte extent differs"
            );
        }
        Ok(())
    }

    fn write_successor(
        &self,
        step: PH3ImageSamplingStep,
        sample: &ManagedHostTensor,
        velocity: PH3ImageSamplingVelocity<'_>,
        output: &mut ManagedHostBuffer,
    ) -> Result<()> {
        for (index, (x, y)) in sample
            .bytes()
            .bytes()
            .chunks_exact(4)
            .zip(output.as_mut().chunks_exact_mut(4))
            .enumerate()
        {
            let x = decode_f32(x);
            let source = velocity.mapping.source_element(index);
            let offset = source
                .checked_mul(self.precision.bytes())
                .context("velocity mapped byte offset overflow")?;
            let end = offset
                .checked_add(self.precision.bytes())
                .context("velocity mapped byte extent overflow")?;
            let v = self.precision.decode(
                velocity
                    .bytes
                    .bytes()
                    .get(offset..end)
                    .context("velocity mapped read exceeds original Output extent")?,
            );
            ensure!(
                x.is_finite() && v.is_finite(),
                "H3 sampling input {index} is nonfinite"
            );
            let updated = step.update(x, v);
            ensure!(
                updated.is_finite(),
                "H3 sampling result {index} is nonfinite"
            );
            y.copy_from_slice(&updated.to_le_bytes());
        }
        Ok(())
    }
}

fn decode_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests;
