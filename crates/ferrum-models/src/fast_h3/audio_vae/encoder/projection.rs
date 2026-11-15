use super::*;
use family::Component;
use ferrum_execution::TensorExecutionOutput;
mod program;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProjectionConfig {
    architecture: FastH3AudioVaeConfig,
    mono_batch: u64,
    latent_samples: u64,
    weights: Vec<AudioWeight>,
}
impl Component for ProjectionConfig {
    const FAMILY_ID: &'static str = "family.fast_h3.audio_encoder.projection.original";
    const PROFILE_ID: &'static str = "fast_h3.audio_encoder.projection.full_f32.causal.head_mean.adaptive_pool.geglu_tanh.mode.v1";
    const OUTPUT_ID: &'static str = PreparedFastH3AudioEncoderProjection::OUTPUT_ID;
    fn validate(&self, limits: FastH3AudioComponentLimits) -> Result<(), VNextError> {
        self.architecture.validate().map_err(invalid)?;
        if self.mono_batch == 0
            || self.latent_samples == 0
            || self.mono_batch > STRUCTURED_ATTENTION_BF16_MAX_BATCH as u64
            || self.latent_samples > u32::MAX as u64
        {
            return Err(invalid(
                "audio encoder projection batch/sequence outside bounded attention domain",
            ));
        }
        let head = self.architecture.latent_dim / self.architecture.num_attention_heads;
        if !(2..=STRUCTURED_ATTENTION_BF16_MAX_HEAD_DIM).contains(&head) {
            return Err(invalid(
                "audio encoder attention head width outside F32 provider domain",
            ));
        }
        self.mono_batch
            .checked_mul(self.latent_samples)
            .ok_or_else(|| invalid("audio projection rows overflow"))?;
        config::validate_inventory(
            &self.weights,
            entries(&self.architecture).map_err(invalid)?,
            limits,
        )?;
        self.schema()?.validate(&id(Self::FAMILY_ID)?)?;
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("audio projection metadata capacity exceeded"));
        }
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        weight_schema(
            &self.weights,
            "weight-layout.fast_h3.audio_encoder.projection.original_f32.v1",
        )
    }
    fn build(
        &self,
        limits: FastH3AudioComponentLimits,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        program::build(self, limits)
    }
}

fn entries(a: &FastH3AudioVaeConfig) -> Result<Vec<Entry>> {
    fn original(out: &mut Vec<Entry>, name: String, shape: Vec<u64>, physical: Vec<u64>) {
        out.push(Entry {
            name: name.clone(),
            physical,
            kind: EntryKind::Original {
                source: name,
                shape,
            },
        });
    }
    fn affine(out: &mut Vec<Entry>, prefix: &str, inputs: u64, outputs: u64) {
        original(
            out,
            format!("{prefix}.weight"),
            vec![outputs, inputs],
            vec![outputs, inputs],
        );
        original(out, format!("{prefix}.bias"), vec![outputs], vec![outputs]);
    }
    a.validate()?;
    let mut out = Vec::new();
    let d = a.latent_dim;
    let c = a.latent_channels;
    let hidden = c.checked_mul(2).context("audio MLP width overflow")?;
    let qkv = d.checked_mul(3).context("audio QKV width overflow")?;
    for (prefix, width) in [
        ("pre_block.norm1", d),
        ("pre_block.norm3", d),
        ("pre_block.norm2", c),
        ("pre_block.mlp.norm", c),
    ] {
        for suffix in ["weight", "bias"] {
            original(
                &mut out,
                format!("{prefix}.{suffix}"),
                vec![width],
                vec![width],
            );
        }
    }
    original(
        &mut out,
        "pre_block.attn.qkv.weight".into(),
        vec![qkv, d],
        vec![qkv, d],
    );
    out.push(Entry {
        name: "pre_block.attn.qkv.bias".into(),
        physical: vec![qkv],
        kind: EntryKind::ConcatVectors {
            sources: ["q_bias", "zero_k_bias", "v_bias"]
                .map(|n| format!("pre_block.attn.{n}"))
                .into(),
            width: d,
        },
    });
    for (prefix, inputs, outputs) in [
        ("pre_block.proj", d, c),
        ("pre_block.attn.proj", c, c),
        ("pre_block.mlp.w0", c, hidden),
        ("pre_block.mlp.w1", c, hidden),
        ("pre_block.mlp.w2", hidden, c),
    ] {
        affine(&mut out, prefix, inputs, outputs);
    }
    // A pointwise Conv1d is an affine map per time row. Keep its original
    // O,I,1 source shape checked; the graph declares its explicit O,I layout.
    original(
        &mut out,
        "mean_proj.weight".into(),
        vec![c, c, 1],
        vec![c, c],
    );
    original(&mut out, "mean_proj.bias".into(), vec![c], vec![c]);
    Ok(out)
}

/// Original causal attention projection and posterior mean. The reference
/// consumer uses mode(), so logs_proj and posterior sampling are not executed.
pub struct PreparedFastH3AudioEncoderProjection {
    selected: PreparedAudioWeights,
    architecture: FastH3AudioVaeConfig,
    mono_batch: u64,
    latent_samples: u64,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    input: ResolvedTensorSpec,
    output: ResolvedTensorSpec,
}
impl PreparedFastH3AudioEncoderProjection {
    pub fn host_weight_backing_bytes(
        architecture: &FastH3AudioVaeConfig,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        source::derived_backing_bytes(&entries(architecture)?, host)
    }
    pub const INPUT_ID: &'static str = "value.fast_h3.audio.encoder.projection.input";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.audio.encoder.mode";
    pub fn prepare(
        architecture: &FastH3AudioVaeConfig,
        source: Arc<SafetensorsArchive>,
        mono_batch: u64,
        latent_samples: u64,
        host: &HostMemoryAccount,
        limits: FastH3AudioComponentLimits,
    ) -> Result<Self> {
        architecture.validate()?;
        let rows = mono_batch
            .checked_mul(latent_samples)
            .context("audio encoder projection row overflow")?;
        ensure!(
            rows > 0
                && tensor(vec![rows, architecture.latent_dim]).byte_len()?
                    <= limits.maximum_activation_bytes,
            "audio projection input exceeds capacity"
        );
        let selected = PreparedAudioWeights::prepare(source, entries(architecture)?, host, limits)?;
        let config = ProjectionConfig {
            architecture: architecture.clone(),
            mono_batch,
            latent_samples,
            weights: selected.weights.clone(),
        };
        config.validate(limits)?;
        let family = TypedFamilyRegistration::new(family::Family::<ProjectionConfig>::new(limits)?)
            .prepare_with_profile(
                &serde_json::to_value(&config)?,
                &NumericalProfileId::new(ProjectionConfig::PROFILE_ID)
                    .map_err(anyhow::Error::msg)?,
            )?;
        let (program, specs) = config.build(limits)?;
        ensure!(
            program == *family.program(),
            "prepared audio projection program changed"
        );
        let resolve = |s: &ProgramTensorSpec| {
            ResolvedTensorSpec::new(s.dimensions.clone(), s.element_type, s.layout.clone())
        };
        let input = resolve(&specs[&id::<ProgramValueId>(Self::INPUT_ID)?])?;
        let output = resolve(&specs[&id::<ProgramValueId>(Self::OUTPUT_ID)?])?;
        Ok(Self {
            selected,
            architecture: architecture.clone(),
            mono_batch,
            latent_samples,
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

    /// A separately admitted presentation copy bridges the completed CNN's
    /// channel-major output to the causal transformer's time-major rows. It
    /// preserves every F32 bit and does not retain CNN device allocations.
    pub fn managed_input(
        &self,
        trunk: &TensorExecutionOutput,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        let (b, d, t) = (
            self.mono_batch,
            self.architecture.latent_dim,
            self.latent_samples,
        );
        ensure!(
            trunk.value_id().as_str() == PreparedFastH3AudioEncoderTrunk::OUTPUT_ID
                && trunk.tensor().dimensions() == [b, d, 1, t]
                && trunk.tensor().element_type() == ElementType::F32
                && trunk.tensor().layout() == &ResolvedTensorLayout::Contiguous,
            "audio projection requires the matching completed encoder trunk"
        );
        let mut bytes = host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(self.input.minimum_storage_bytes()?)?,
        )?;
        for batch in 0..b {
            for time in 0..t {
                for channel in 0..d {
                    let src = ((batch * d + channel) * t + time) as usize * 4;
                    let dst = ((batch * t + time) * d + channel) as usize * 4;
                    bytes.as_mut()[dst..dst + 4]
                        .copy_from_slice(&trunk.bytes().bytes()[src..src + 4]);
                }
            }
        }
        Ok(ManagedHostTensor::new(
            id(Self::INPUT_ID)?,
            self.input.clone(),
            bytes.freeze(),
        )?)
    }

    /// Original mode normalization, with separate F32 subtraction/division,
    /// and explicit channel-major [mono_batch,C,T] layout for reference packing.
    pub fn normalized_mode(
        &self,
        output: &TensorExecutionOutput,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            output.value_id().as_str() == Self::OUTPUT_ID && output.tensor() == &self.output,
            "audio posterior mean output identity differs"
        );
        let (b, c, t) = (
            self.mono_batch,
            self.architecture.latent_channels,
            self.latent_samples,
        );
        let mut bytes = host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(self.output.minimum_storage_bytes()?)?,
        )?;
        for batch in 0..b {
            for channel in 0..c {
                for time in 0..t {
                    let src = ((batch * t + time) * c + channel) as usize * 4;
                    let dst = ((batch * c + channel) * t + time) as usize * 4;
                    let x = f32::from_le_bytes(
                        output.bytes().bytes()[src..src + 4].try_into().unwrap(),
                    );
                    let x = (x - self.architecture.latent_mean()[channel as usize])
                        / self.architecture.latent_std()[channel as usize];
                    ensure!(x.is_finite(), "nonfinite normalized audio reference");
                    bytes.as_mut()[dst..dst + 4].copy_from_slice(&x.to_le_bytes());
                }
            }
        }
        Ok(ManagedHostTensor::new(
            id("value.fast_h3.audio.reference.normalized")?,
            ResolvedTensorSpec::new(
                vec![b, c, t],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            bytes.freeze(),
        )?)
    }
}
impl WeightComponentSource for PreparedFastH3AudioEncoderProjection {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.selected.component(c)
    }
}
