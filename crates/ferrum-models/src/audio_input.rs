//! Bounded PCM/IEEE-F32 WAVE presentation on the caller's Host account.
//! Container parsing, channel expansion and trimming do not infer a model,
//! resample the signal, or create a second resource owner.
use anyhow::{ensure, Context, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostBytes, ManagedHostTensor,
    ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
};
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
pub struct AudioInputLimits {
    pub maximum_encoded_bytes: usize,
    pub maximum_decoded_bytes: usize,
}
impl Default for AudioInputLimits {
    fn default() -> Self {
        Self {
            maximum_encoded_bytes: 33 << 20,
            maximum_decoded_bytes: 32 << 20,
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum SampleEncoding {
    Unsigned8,
    Signed16,
    Signed24,
    Signed32,
    Float32,
}
impl SampleEncoding {
    fn bytes(self) -> usize {
        match self {
            Self::Unsigned8 => 1,
            Self::Signed16 => 2,
            Self::Signed24 => 3,
            Self::Signed32 | Self::Float32 => 4,
        }
    }
    fn decode(self, b: &[u8]) -> f32 {
        match self {
            Self::Unsigned8 => (f32::from(b[0]) - 128.) / 128.,
            Self::Signed16 => f32::from(i16::from_le_bytes(b.try_into().unwrap())) / 32768.,
            Self::Signed24 => {
                ((i32::from_le_bytes([b[0], b[1], b[2], 0]) << 8) >> 8) as f32 / 8_388_608.
            }
            Self::Signed32 => i32::from_le_bytes(b.try_into().unwrap()) as f32 / 2_147_483_648.,
            Self::Float32 => f32::from_le_bytes(b.try_into().unwrap()),
        }
    }
}
#[derive(Clone, Copy, Debug)]
struct Format {
    channels: usize,
    rate: u32,
    encoding: SampleEncoding,
    block: usize,
}
fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes(b[i..i + 2].try_into().unwrap())
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn format(b: &[u8]) -> Result<Format> {
    ensure!(b.len() >= 16, "short WAVE format chunk");
    let mut tag = u16_at(b, 0);
    let channels = usize::from(u16_at(b, 2));
    let rate = u32_at(b, 4);
    let block = usize::from(u16_at(b, 12));
    let bits = u16_at(b, 14);
    ensure!(
        (1..=2).contains(&channels) && rate > 0,
        "WAVE must declare positive-rate mono or stereo"
    );
    if tag == 0xfffe {
        ensure!(
            b.len() >= 40 && u16_at(b, 16) >= 22 && usize::from(u16_at(b, 16)) + 18 == b.len(),
            "invalid extensible WAVE format"
        );
        ensure!(
            u16_at(b, 18) == bits,
            "packed valid-bit WAVE containers require a separate decoder"
        );
        let mask = u32_at(b, 20);
        ensure!(
            mask == 0 || mask == if channels == 1 { 4 } else { 3 },
            "WAVE speaker mask differs from mono/stereo order"
        );
        ensure!(
            b[28..40] == [0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71],
            "unsupported WAVE subformat GUID"
        );
        tag = u16::try_from(u32_at(b, 24)).context("unsupported WAVE encoding")?;
    } else {
        ensure!(
            b.len() == 16 || (b.len() >= 18 && usize::from(u16_at(b, 16)) + 18 == b.len()),
            "invalid WAVE format extension length"
        );
    }
    let encoding = match (tag, bits) {
        (1, 8) => SampleEncoding::Unsigned8,
        (1, 16) => SampleEncoding::Signed16,
        (1, 24) => SampleEncoding::Signed24,
        (1, 32) => SampleEncoding::Signed32,
        (3, 32) => SampleEncoding::Float32,
        _ => anyhow::bail!("unsupported WAVE sample encoding"),
    };
    ensure!(
        block == channels * encoding.bytes()
            && u64::from(u32_at(b, 8)) == u64::from(rate) * block as u64,
        "WAVE block alignment or byte rate differs"
    );
    Ok(Format {
        channels,
        rate,
        encoding,
        block,
    })
}

/// Checked container layout, retaining the original admitted encoded backing.
/// A long input may be trimmed before any sample-shaped allocation is made.
#[derive(Clone, Debug)]
pub struct WaveAudioPlan {
    encoded: ManagedHostBytes,
    data: Range<usize>,
    format: Format,
    samples: usize,
    limits: AudioInputLimits,
}
impl WaveAudioPlan {
    pub fn parse(encoded: ManagedHostBytes, limits: AudioInputLimits) -> Result<Self> {
        let b = encoded.bytes();
        ensure!(
            limits.maximum_encoded_bytes > 0
                && limits.maximum_decoded_bytes > 0
                && b.len() <= limits.maximum_encoded_bytes,
            "audio input exceeds declared capacity"
        );
        ensure!(
            b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WAVE",
            "audio input requires a RIFF WAVE container"
        );
        ensure!(
            u64::from(u32_at(b, 4)) + 8 == b.len() as u64,
            "WAVE RIFF extent differs from encoded payload"
        );
        let mut cursor = 12;
        let (mut fmt, mut data) = (None, None);
        while cursor < b.len() {
            ensure!(b.len() - cursor >= 8, "truncated WAVE chunk header");
            let count = u32_at(b, cursor + 4) as usize;
            let start = cursor + 8;
            let end = start
                .checked_add(count)
                .filter(|&n| n <= b.len())
                .context("WAVE chunk exceeds container")?;
            match &b[cursor..cursor + 4] {
                b"fmt " => {
                    ensure!(fmt.is_none(), "duplicate WAVE format chunk");
                    fmt = Some(format(&b[start..end])?);
                }
                b"data" => {
                    ensure!(
                        data.is_none(),
                        "multiple WAVE data chunks require explicit concatenation"
                    );
                    data = Some(start..end);
                }
                _ => {}
            }
            cursor = end
                .checked_add(count % 2)
                .filter(|&n| n <= b.len())
                .context("missing WAVE chunk padding")?;
        }
        let format = fmt.context("WAVE has no format chunk")?;
        let data = data.context("WAVE has no data chunk")?;
        ensure!(
            !data.is_empty() && data.len() % format.block == 0,
            "WAVE data does not contain complete sample frames"
        );
        let samples = data.len() / format.block;
        Ok(Self {
            encoded,
            data,
            format,
            samples,
            limits,
        })
    }
    pub fn sampling_rate(&self) -> u32 {
        self.format.rate
    }
    pub fn source_channels(&self) -> usize {
        self.format.channels
    }
    pub fn source_samples_per_channel(&self) -> usize {
        self.samples
    }
    pub fn stereo_spec(&self, maximum_samples: usize) -> Result<ResolvedTensorSpec> {
        ensure!(maximum_samples > 0, "positive audio trim extent required");
        let samples = self.samples.min(maximum_samples);
        let spec = ResolvedTensorSpec::new(
            vec![2, 1, samples as u64],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        ensure!(
            spec.minimum_storage_bytes()? <= self.limits.maximum_decoded_bytes as u64,
            "trimmed stereo audio exceeds declared decoded capacity"
        );
        Ok(spec)
    }
    pub fn stereo_backing_bytes(
        &self,
        maximum_samples: usize,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        Ok(host.backing_bytes(usize::try_from(
            self.stereo_spec(maximum_samples)?.minimum_storage_bytes()?,
        )?)?)
    }
    /// Validate only samples that survive trimming, before admitting a model
    /// job. Integer PCM is finite by construction; floating WAVE is not.
    pub fn validate_samples(&self, maximum_samples: usize) -> Result<()> {
        self.stereo_spec(maximum_samples)?;
        if matches!(self.format.encoding, SampleEncoding::Float32) {
            let end = self.data.start + self.samples.min(maximum_samples) * self.format.block;
            ensure!(
                self.encoded.bytes()[self.data.start..end]
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                "WAVE contains a nonfinite reference sample"
            );
        }
        Ok(())
    }
    /// Normalize PCM into independent F32 mono channels. Mono is copied into
    /// both stereo channels; stereo order is preserved. IEEE-F32 amplitudes are
    /// not clipped. Resampling, if requested, is a separate declared operation.
    pub fn prepare_stereo(
        &self,
        maximum_samples: usize,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            host.owns(&self.encoded),
            "audio encoded backing belongs to another Host account"
        );
        let spec = self.stereo_spec(maximum_samples)?;
        let samples = spec.dimensions()[2] as usize;
        let mut out = host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(spec.minimum_storage_bytes()?)?,
        )?;
        let size = self.format.encoding.bytes();
        for t in 0..samples {
            for c in 0..2 {
                let source =
                    self.data.start + t * self.format.block + (c % self.format.channels) * size;
                let x = self
                    .format
                    .encoding
                    .decode(&self.encoded.bytes()[source..source + size]);
                ensure!(x.is_finite(), "WAVE contains a nonfinite reference sample");
                let dst = (c * samples + t) * 4;
                out.as_mut()[dst..dst + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new("value.audio.reference.waveform")?,
            spec,
            out.freeze(),
        )?)
    }
}
#[cfg(test)]
mod tests;
