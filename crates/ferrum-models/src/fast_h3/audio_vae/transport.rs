use super::*;
use ferrum_execution::TensorExecutionOutput;
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FastH3AudioLatentLayout {
    /// [request*2*latent_samples, latent_channels], stereo channel then time.
    PackedStereoTokens,
    /// [request*2, latent_channels, latent_samples], two mono items per request.
    MonoBatchNcl,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FastH3AudioLatentUnits {
    Normalized,
    Decoder,
}
pub struct FastH3AudioWaveform {
    bytes: ManagedHostBytes,
    requests: u64,
    samples: u64,
    sampling_rate: u32,
}
impl FastH3AudioWaveform {
    pub fn requests(&self) -> u64 {
        self.requests
    }
    pub fn samples_per_channel(&self) -> u64 {
        self.samples
    }
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }
    /// Original admitted Output owner; clones and request slices retain its
    /// existing Host charge through an encoder or response body.
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
    pub fn into_bytes(self) -> ManagedHostBytes {
        self.bytes
    }
    /// One request's interleaved stereo F32 little-endian samples.
    pub fn request_bytes(&self, request: u64) -> Result<&[u8]> {
        ensure!(request < self.requests, "audio request outside output");
        let stride = usize::try_from(self.samples * 2 * 4)?;
        let start = usize::try_from(request)? * stride;
        Ok(&self.bytes.bytes()[start..start + stride])
    }
}
impl PreparedFastH3AudioDecoder {
    /// Explicit source-layout conversion and, when selected, separate F32
    /// std multiply then mean add. No prefix rows are silently dropped.
    pub fn managed_input(
        &self,
        source: &ManagedHostTensor,
        layout: FastH3AudioLatentLayout,
        units: FastH3AudioLatentUnits,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        let [batch, channels, _, length]: [u64; 4] = self.input_spec().dimensions().try_into()?;
        let expected = match layout {
            FastH3AudioLatentLayout::PackedStereoTokens => vec![batch * length, channels],
            FastH3AudioLatentLayout::MonoBatchNcl => vec![batch, channels, length],
        };
        ensure!(
            source.tensor().dimensions() == expected
                && source.tensor().element_type() == ElementType::F32
                && source.tensor().layout() == &ResolvedTensorLayout::Contiguous,
            "audio source layout differs from explicit latent declaration"
        );
        let bytes = usize::try_from(self.input_spec().minimum_storage_bytes()?)?;
        ensure!(
            source.bytes().len() == bytes,
            "audio source physical extent differs"
        );
        let mut out = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        for b in 0..batch {
            for c in 0..channels {
                for t in 0..length {
                    let destination = ((b * channels + c) * length + t) as usize;
                    let source_index = match layout {
                        FastH3AudioLatentLayout::MonoBatchNcl => destination,
                        FastH3AudioLatentLayout::PackedStereoTokens => {
                            ((b * length + t) * channels + c) as usize
                        }
                    };
                    let x = f32::from_le_bytes(
                        source.bytes().bytes()[4 * source_index..4 * source_index + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let x = match units {
                        FastH3AudioLatentUnits::Decoder => x,
                        FastH3AudioLatentUnits::Normalized => {
                            let scaled = x * self.architecture().latent_std()[c as usize];
                            scaled + self.architecture().latent_mean()[c as usize]
                        }
                    };
                    ensure!(x.is_finite(), "audio latent conversion is nonfinite");
                    out.as_mut()[4 * destination..4 * destination + 4]
                        .copy_from_slice(&x.to_le_bytes());
                }
            }
        }
        Ok(ManagedHostTensor::new(
            id(Self::INPUT_ID)?,
            self.input_spec().clone(),
            out.freeze(),
        )?)
    }
    /// Checked presentation copy from the actual execution output. The source
    /// output retains its own runtime Host owner; this allocation is separately
    /// charged as Output and does not mint an execution-completion proof.
    pub fn interleave_stereo(
        &self,
        output: &TensorExecutionOutput,
        host: &HostMemoryAccount,
    ) -> Result<FastH3AudioWaveform> {
        ensure!(
            output.value_id().as_str() == Self::OUTPUT_ID && output.tensor() == self.output_spec(),
            "audio output identity or geometry differs"
        );
        let [batch, channels, height, samples]: [u64; 4] =
            output.tensor().dimensions().try_into()?;
        ensure!(
            batch % 2 == 0 && channels == 1 && height == 1,
            "audio output requires paired independent mono batch items"
        );
        let bytes = output.bytes().bytes();
        ensure!(
            bytes.len() as u64 == output.tensor().minimum_storage_bytes()?,
            "audio output byte extent differs"
        );
        let mut interleaved = host.allocate_zeroed(HostMemoryClass::Output, bytes.len())?;
        for request in 0..batch / 2 {
            for t in 0..samples {
                for channel in 0..2 {
                    let source = (((request * 2 + channel) * samples + t) * 4) as usize;
                    let destination = ((request * samples + t) * 2 + channel) as usize * 4;
                    let value = f32::from_le_bytes(bytes[source..source + 4].try_into().unwrap());
                    ensure!(
                        value.is_finite() && (-1.0..=1.0).contains(&value),
                        "audio decoder output is outside finite clamped domain"
                    );
                    interleaved.as_mut()[destination..destination + 4]
                        .copy_from_slice(&bytes[source..source + 4]);
                }
            }
        }
        Ok(FastH3AudioWaveform {
            bytes: interleaved.freeze(),
            requests: batch / 2,
            samples,
            sampling_rate: self.architecture().sampling_rate(),
        })
    }
}
