use super::*;

impl VideoContainerPlan {
    /// libavcodec float PCM, source mono duplication, and explicit libswresample
    /// rate conversion. Same-rate PCM is unchanged. Rate conversion is a codec
    /// profile, not a claim of bitwise equivalence to torchaudio's sinc filter.
    pub const AUDIO_PROFILE: &'static str = "ffmpeg.f32_pcm.swr32_phase10_cutoff097.mono_repeat.v1";

    /// One maximum-length planar backing is compacted in place after EOF. A
    /// shorter stream retains its original physical credit through the view.
    pub fn prepare_audio(
        &self,
        sample_rate: u32,
        maximum_samples: usize,
        host: &HostMemoryAccount,
        cancelled: &AtomicBool,
    ) -> Result<ManagedHostTensor> {
        let source = self.audio.context("reference video has no soundtrack")?;
        ensure!(
            host.owns(&self.encoded)
                && (1..=2).contains(&source.channels)
                && sample_rate > 0
                && sample_rate <= 384000
                && maximum_samples > 0,
            "soundtrack requires admitted mono/stereo input and positive bounded sample rate"
        );
        let duration = maximum_samples as f64 / f64::from(sample_rate);
        ensure!(
            duration <= self.codec.config().maximum_duration.as_secs_f64(),
            "soundtrack duration exceeds codec bound"
        );
        let bytes = maximum_samples
            .checked_mul(8)
            .context("soundtrack backing overflow")?;
        let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let filter = format!("aresample={sample_rate}:resampler=swr:filter_size=32:phase_shift=10:linear_interp=1:cutoff=0.97,atrim=end_sample={maximum_samples}");
        let args = vec![
            "-map".into(),
            "0:a:0".into(),
            "-vn".into(),
            "-sn".into(),
            "-dn".into(),
            "-t".into(),
            duration.to_string().into(),
            "-af".into(),
            filter.into(),
            "-c:a".into(),
            "pcm_f32le".into(),
            "-f".into(),
            "f32le".into(),
            "pipe:1".into(),
        ];
        let samples = process::run(
            &self.codec,
            &self.codec.config().ffmpeg,
            &self.encoded,
            args,
            true,
            cancelled,
            |stdout| {
                let mut stdout = std::io::BufReader::with_capacity(16 << 10, stdout);
                let channels = source.channels as usize;
                let mut sample = [0u8; 8];
                let mut count = 0;
                loop {
                    if stdout.read(&mut sample[..1])? == 0 {
                        break;
                    }
                    stdout
                        .read_exact(&mut sample[1..channels * 4])
                        .context("partial soundtrack sample")?;
                    ensure!(
                        count < maximum_samples,
                        "soundtrack exceeded admitted sample extent"
                    );
                    for channel in 0..2 {
                        let at = (channel % channels) * 4;
                        let value = f32::from_le_bytes(sample[at..at + 4].try_into()?);
                        ensure!(value.is_finite(), "soundtrack contains a nonfinite sample");
                        let destination = (channel * maximum_samples + count) * 4;
                        output.as_mut()[destination..destination + 4]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                    count += 1;
                }
                ensure!(count > 0, "soundtrack contains no samples");
                Ok(count)
            },
        )?;
        output.as_mut().copy_within(
            maximum_samples * 4..(maximum_samples + samples) * 4,
            samples * 4,
        );
        let spec = ResolvedTensorSpec::new(
            vec![2, 1, samples as u64],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        Ok(ManagedHostTensor::new(
            ProgramValueId::new("value.audio.reference.waveform")?,
            spec,
            output.freeze().slice(0..samples * 8)?,
        )?)
    }
}
