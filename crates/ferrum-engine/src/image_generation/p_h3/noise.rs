//! Explicit request-local initial image/audio streams. The CPU profile follows
//! Torch's MT19937 draw layout with scalar F32/libm, not AVX bit equivalence.
use anyhow::{ensure, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily, NormalNoisePrecision};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};

use ferrum_models::normal_noise::cpu_mt19937;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageInitialNoisePolicy {
    /// One flat native F32 draw per request, image first and audio second.
    /// No global mutable RNG state is shared between requests, so arrival
    /// order and batching do not change a draw. Shape/device/provider may.
    NativeCudaPhiloxJointImageAudioV1,
    /// CPU MT19937: image then audio on one private state per request.
    /// Scalar F32/libm conversion is explicit, not Torch AVX bit identity.
    /// Each stream must contain at least sixteen F32 elements.
    #[default]
    CpuMt19937ImageAudioV1,
}
impl PH3ImageInitialNoisePolicy {
    pub const fn identity(self) -> &'static str {
        match self {
            Self::NativeCudaPhiloxJointImageAudioV1 => {
                "p_h3_image.noise.native_cuda_philox.joint_image_audio.v1"
            }
            Self::CpuMt19937ImageAudioV1 => {
                "p_h3_image.noise.cpu_mt19937.image_audio.scalar_f32_platform_libm.v1"
            }
        }
    }
    /// This is one native invocation, not a continuation offset for Torch's
    /// two shape-dependent CPU randn calls. Native offsets index each device
    /// thread's subsequence, and cannot be treated as flat normal indices.
    pub fn state(self, seed: u64) -> Result<PhiloxSeedOffset> {
        match self {
            Self::NativeCudaPhiloxJointImageAudioV1 => Ok(PhiloxSeedOffset::fresh(seed)),
            Self::CpuMt19937ImageAudioV1 => {
                anyhow::bail!("CPU MT19937 policy has no CUDA Philox state")
            }
        }
    }
}

pub(super) struct JointNoiseLayout {
    pub config: NormalNoiseConfig,
    image: ResolvedTensorSpec,
    audio: ResolvedTensorSpec,
    image_bytes: usize,
    audio_bytes: usize,
}
impl JointNoiseLayout {
    pub fn new(image: &ResolvedTensorSpec, audio: &ResolvedTensorSpec) -> Result<Self> {
        ensure!(
            matches!(image.dimensions(), [1, _, _, _] | [1, _, 1, _, _])
                && matches!(audio.dimensions(), [1, _, 2, 2]),
            "H3 initial samples require B1 image/T=1 and stereo audio geometry"
        );
        for spec in [image, audio] {
            ensure!(
                spec.element_type() == ElementType::F32
                    && spec.layout() == &ResolvedTensorLayout::Contiguous,
                "H3 initial sample precision must be contiguous F32"
            );
        }
        let image_bytes = usize::try_from(image.minimum_storage_bytes()?)?;
        let audio_bytes = usize::try_from(audio.minimum_storage_bytes()?)?;
        let total = image_bytes
            .checked_add(audio_bytes)
            .context("H3 joint noise extent overflows")?;
        // The existing native noise contract validates its real indexing
        // capacity here, before compilation or Host/device allocation.
        let config = NormalNoiseConfig::new_with_precision(
            1,
            1,
            u64::try_from(total / 4)?,
            NormalNoisePrecision::F32,
        )?;
        Ok(Self {
            config,
            image: image.clone(),
            audio: audio.clone(),
            image_bytes,
            audio_bytes,
        })
    }

    /// Each stream has its own physical owner, including independent padding.
    pub fn input_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        host.backing_bytes(self.image_bytes)?
            .checked_add(host.backing_bytes(self.audio_bytes)?)
            .context("H3 initial sample backing overflows")
    }

    pub fn validate_cpu(&self) -> Result<()> {
        ensure!(
            self.image_bytes >= 16 * 4 && self.audio_bytes >= 16 * 4,
            "CPU MT19937 initial noise requires at least sixteen F32 elements per stream"
        );
        Ok(())
    }

    pub fn cpu_work_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        self.validate_cpu()?;
        Ok(host.backing_bytes(cpu_mt19937::SCRATCH_BYTES)?)
    }

    /// Fill the two original persistent owners directly. The RNG state is an
    /// actual transient Input allocation from the same Host parent. No sample
    /// is published until both allocations and both draws succeed.
    pub fn cpu_samples(
        &self,
        persistent: &HostMemoryAccount,
        work: &HostMemoryAccount,
        seed: u64,
        image_id: ProgramValueId,
        audio_id: ProgramValueId,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor)> {
        self.validate_cpu()?;
        ensure!(
            persistent.same_parent(work),
            "CPU noise scopes require the same Host parent"
        );
        ensure!(
            image_id != audio_id,
            "H3 image/audio sample identities must differ"
        );
        let mut scratch =
            work.allocate_zeroed(HostMemoryClass::InputPreprocess, cpu_mt19937::SCRATCH_BYTES)?;
        let mut image =
            persistent.allocate_zeroed(HostMemoryClass::InputPreprocess, self.image_bytes)?;
        let mut audio =
            persistent.allocate_zeroed(HostMemoryClass::InputPreprocess, self.audio_bytes)?;
        cpu_mt19937::fill_initial_noise(seed, image.as_mut(), audio.as_mut(), scratch.as_mut())?;
        drop(scratch);
        Ok((
            ManagedHostTensor::new(image_id, self.image.clone(), image.freeze())?,
            ManagedHostTensor::new(audio_id, self.audio.clone(), audio.freeze())?,
        ))
    }

    /// Transfer the completed noise into the request's persistent Input scope.
    /// The original Output owner stays charged until dropped. A late allocation
    /// rejection drops the first private sample and publishes neither stream.
    pub fn samples(
        &self,
        host: &HostMemoryAccount,
        output: &TensorExecutionOutput,
        image_id: ProgramValueId,
        audio_id: ProgramValueId,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor)> {
        ensure!(
            output.value_id().as_str() == NormalNoiseFamily::OUTPUT_ID,
            "H3 initial noise has an unexpected public output identity"
        );
        self.copy_samples(host, output.tensor(), output.bytes(), image_id, audio_id)
    }

    fn copy_samples(
        &self,
        host: &HostMemoryAccount,
        physical: &ResolvedTensorSpec,
        output: &ManagedHostBytes,
        image_id: ProgramValueId,
        audio_id: ProgramValueId,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor)> {
        ensure!(
            image_id != audio_id,
            "H3 image/audio sample identities must differ"
        );
        ensure!(
            physical == &self.config.output_tensor()?
                && physical.minimum_storage_bytes()? == output.len() as u64,
            "H3 joint noise shape, precision or storage differs from its request"
        );
        ensure!(
            host.owns(output) && output.class() == HostMemoryClass::Output,
            "H3 noise readback requires an actual Output owner from the same Host parent"
        );
        let mut image = host.allocate_zeroed(HostMemoryClass::InputPreprocess, self.image_bytes)?;
        let mut audio = host.allocate_zeroed(HostMemoryClass::InputPreprocess, self.audio_bytes)?;
        image
            .as_mut()
            .copy_from_slice(&output.bytes()[..self.image_bytes]);
        audio
            .as_mut()
            .copy_from_slice(&output.bytes()[self.image_bytes..]);
        Ok((
            ManagedHostTensor::new(image_id, self.image.clone(), image.freeze())?,
            ManagedHostTensor::new(audio_id, self.audio.clone(), audio.freeze())?,
        ))
    }
}

pub(super) fn managed_state(
    host: &HostMemoryAccount,
    state: PhiloxSeedOffset,
) -> Result<ManagedHostTensor> {
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, 16)?;
    bytes.as_mut().copy_from_slice(&state.little_endian_bytes());
    Ok(ManagedHostTensor::new(
        ProgramValueId::new(NormalNoiseFamily::STATE_ID)?,
        NormalNoiseConfig::state_tensor()?,
        bytes.freeze(),
    )?)
}

#[cfg(test)]
mod tests;
