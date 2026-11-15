use super::*;
use source::{Entry, EntryKind, PreparedAudioWeights};
mod family;
mod program;
mod projection;
pub use projection::PreparedFastH3AudioEncoderProjection;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct EncoderConfig {
    architecture: FastH3AudioVaeConfig,
    mono_batch: u64,
    samples: u64,
    weights: Vec<AudioWeight>,
}
impl family::Component for EncoderConfig {
    const FAMILY_ID: &'static str = "family.fast_h3.audio_encoder.trunk.original";
    const PROFILE_ID: &'static str =
        "fast_h3.audio_encoder.trunk.full_f32.tree256_weight_norm.direct_snake.v1";
    const OUTPUT_ID: &'static str = PreparedFastH3AudioEncoderTrunk::OUTPUT_ID;
    fn validate(&self, limits: FastH3AudioComponentLimits) -> Result<(), VNextError> {
        self.validate(limits)
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        self.schema()
    }
    fn build(
        &self,
        limits: FastH3AudioComponentLimits,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        program::build(self, limits)
    }
}
impl EncoderConfig {
    fn padded_samples(&self) -> Result<u64, VNextError> {
        self.architecture
            .encoder_padded_samples(self.samples)
            .map_err(invalid)
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        weight_schema(
            &self.weights,
            "weight-layout.fast_h3.audio_encoder.trunk.original_f32.v1",
        )
    }
    fn validate(&self, limits: FastH3AudioComponentLimits) -> Result<(), VNextError> {
        self.architecture.validate().map_err(invalid)?;
        if self.mono_batch == 0 || self.samples == 0 {
            return Err(invalid("positive encoder batch and samples required"));
        }
        self.padded_samples()?;
        config::validate_inventory(
            &self.weights,
            entries(&self.architecture).map_err(invalid)?,
            limits,
        )?;
        self.schema()?
            .validate(&id(family::Family::<EncoderConfig>::FAMILY_ID)?)?;
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("audio encoder metadata capacity exceeded"));
        }
        Ok(())
    }
}

fn entries(a: &FastH3AudioVaeConfig) -> Result<Vec<Entry>> {
    fn snake(out: &mut Vec<Entry>, prefix: &str, channels: u64) {
        let name = format!("{prefix}.alpha");
        out.push(Entry {
            name: name.clone(),
            physical: vec![channels],
            kind: EntryKind::Original {
                source: name,
                shape: vec![1, channels, 1],
            },
        });
    }
    a.validate()?;
    let mut out = Vec::new();
    let mut channels = a.encoder_dim;
    source::conv(
        &mut out,
        "encoder.block.0",
        1,
        channels,
        7,
        false,
        true,
        true,
    );
    for (stage, &stride) in a.encoder_rates.iter().enumerate() {
        let prefix = format!("encoder.block.{}.block", stage + 1);
        for unit in 0..3 {
            let residual = format!("{prefix}.{unit}.block");
            snake(&mut out, &format!("{residual}.0"), channels);
            source::conv(
                &mut out,
                &format!("{residual}.1"),
                channels,
                channels,
                7,
                false,
                true,
                true,
            );
            snake(&mut out, &format!("{residual}.2"), channels);
            source::conv(
                &mut out,
                &format!("{residual}.3"),
                channels,
                channels,
                1,
                false,
                true,
                true,
            );
        }
        snake(&mut out, &format!("{prefix}.3"), channels);
        source::conv(
            &mut out,
            &format!("{prefix}.4"),
            channels,
            channels * 2,
            stride * 2,
            false,
            true,
            true,
        );
        channels *= 2;
    }
    snake(
        &mut out,
        &format!("encoder.block.{}", a.encoder_rates.len() + 1),
        channels,
    );
    source::conv(
        &mut out,
        &format!("encoder.block.{}", a.encoder_rates.len() + 2),
        channels,
        a.latent_dim,
        3,
        false,
        true,
        true,
    );
    Ok(out)
}

/// Original weight-normalized waveform CNN, before the causal projection.
/// The caller explicitly supplies independent mono channels; padding and every
/// intermediate activation are covered by the original component authority.
pub struct PreparedFastH3AudioEncoderTrunk {
    selected: PreparedAudioWeights,
    samples: u64,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    input: ResolvedTensorSpec,
    output: ResolvedTensorSpec,
}
impl PreparedFastH3AudioEncoderTrunk {
    pub fn host_weight_backing_bytes(
        architecture: &FastH3AudioVaeConfig,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        source::derived_backing_bytes(&entries(architecture)?, host)
    }
    pub const INPUT_ID: &'static str = "value.fast_h3.audio.encoder.input";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.audio.encoder.trunk";
    pub fn prepare(
        architecture: &FastH3AudioVaeConfig,
        source: Arc<SafetensorsArchive>,
        mono_batch: u64,
        samples: u64,
        host: &HostMemoryAccount,
        limits: FastH3AudioComponentLimits,
    ) -> Result<Self> {
        architecture.validate()?;
        ensure!(mono_batch > 0, "positive audio encoder batch required");
        let padded = architecture.encoder_padded_samples(samples)?;
        let input = tensor(vec![mono_batch, 1, 1, padded]);
        ensure!(
            input.byte_len()? <= limits.maximum_activation_bytes,
            "audio encoder input exceeds capacity"
        );
        let selected = PreparedAudioWeights::prepare(source, entries(architecture)?, host, limits)?;
        let config = EncoderConfig {
            architecture: architecture.clone(),
            mono_batch,
            samples,
            weights: selected.weights.clone(),
        };
        config.validate(limits)?;
        let family = TypedFamilyRegistration::new(family::Family::<EncoderConfig>::new(limits)?)
            .prepare_with_profile(
                &serde_json::to_value(&config)?,
                &NumericalProfileId::new(family::Family::<EncoderConfig>::PROFILE_ID)
                    .map_err(anyhow::Error::msg)?,
            )?;
        let (program, specs) = program::build(&config, limits)?;
        ensure!(
            program == *family.program(),
            "prepared audio encoder program changed"
        );
        let resolve = |s: &ProgramTensorSpec| {
            ResolvedTensorSpec::new(s.dimensions.clone(), s.element_type, s.layout.clone())
        };
        let input = resolve(&specs[&id::<ProgramValueId>(Self::INPUT_ID)?])?;
        let output = resolve(&specs[&id::<ProgramValueId>(Self::OUTPUT_ID)?])?;
        Ok(Self {
            selected,
            samples,
            family,
            options: ProgramPlanCompileOptions::new(specs)?,
            input,
            output,
        })
    }
    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        &self.options
    }
    pub fn input_spec(&self) -> &ResolvedTensorSpec {
        &self.input
    }
    pub fn output_spec(&self) -> &ResolvedTensorSpec {
        &self.output
    }
    pub fn original_source_bytes(&self) -> u64 {
        self.selected.original_source_bytes
    }
    pub fn original_source_tensors(&self) -> usize {
        self.selected.original_source_tensors
    }

    /// Source is F32 NCL [mono_batch, 1, samples] at the declared VAE sample
    /// rate. This operation only adds the original encoder's right zero pad;
    /// it does not resample, mix stereo channels or infer a source layout.
    pub fn managed_input(
        &self,
        source: &ManagedHostTensor,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        let batch = self.input.dimensions()[0];
        let padded = self.input.dimensions()[3];
        ensure!(
            source.tensor().dimensions() == [batch, 1, self.samples]
                && source.tensor().element_type() == ElementType::F32
                && source.tensor().layout() == &ResolvedTensorLayout::Contiguous,
            "audio encoder requires declared independent mono NCL input"
        );
        let mut out = host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(self.input.minimum_storage_bytes()?)?,
        )?;
        for b in 0..batch as usize {
            let start = b * self.samples as usize * 4;
            let bytes = &source.bytes().bytes()[start..start + self.samples as usize * 4];
            ensure!(
                bytes
                    .chunks_exact(4)
                    .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                "nonfinite audio reference sample"
            );
            let dst = b * padded as usize * 4;
            out.as_mut()[dst..dst + bytes.len()].copy_from_slice(bytes);
        }
        Ok(ManagedHostTensor::new(
            id(Self::INPUT_ID)?,
            self.input.clone(),
            out.freeze(),
        )?)
    }
}
impl WeightComponentSource for PreparedFastH3AudioEncoderTrunk {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.selected.component(c)
    }
}
