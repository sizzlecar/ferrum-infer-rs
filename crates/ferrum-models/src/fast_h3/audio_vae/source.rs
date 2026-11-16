use super::*;
const WEIGHT_NORM_RECIPE: &str =
    "fast_h3.audio.weight_norm.dim0.f32.fma256.tree128_to1.sqrt_rn.div_rn.two_mul.v1";
#[derive(Clone)]
pub(super) struct Entry {
    pub name: String,
    pub physical: Vec<u64>,
    pub(super) kind: EntryKind,
}
#[derive(Clone)]
pub(super) enum EntryKind {
    Original {
        source: String,
        shape: Vec<u64>,
    },
    WeightNorm {
        v: String,
        g: String,
        shape: Vec<u64>,
    },
    ConcatVectors {
        sources: Vec<String>,
        width: u64,
    },
    Zero,
}
impl Entry {
    pub(super) fn names(&self) -> Vec<String> {
        match &self.kind {
            EntryKind::Original { source, .. } => vec![source.clone()],
            EntryKind::WeightNorm { v, g, .. } => vec![g.clone(), v.clone()],
            EntryKind::ConcatVectors { sources, .. } => sources.clone(),
            EntryKind::Zero => vec!["derived.fast_h3.audio.conv_post.zero_bias.f32".into()],
        }
    }
}
pub(super) fn conv(
    out: &mut Vec<Entry>,
    prefix: &str,
    inputs: u64,
    outputs: u64,
    kernel: u64,
    transpose: bool,
    wn: bool,
    bias: bool,
) {
    let shape = if transpose {
        vec![inputs, outputs, kernel]
    } else {
        vec![outputs, inputs, kernel]
    };
    let physical = vec![shape[0], shape[1], 1, shape[2]];
    let name = format!("{prefix}.weight");
    let kind = if wn {
        EntryKind::WeightNorm {
            v: format!("{prefix}.weight_v"),
            g: format!("{prefix}.weight_g"),
            shape,
        }
    } else {
        EntryKind::Original {
            source: name.clone(),
            shape,
        }
    };
    out.push(Entry {
        name,
        physical,
        kind,
    });
    let name = format!("{prefix}.bias");
    out.push(Entry {
        name: name.clone(),
        physical: vec![outputs],
        kind: if bias {
            EntryKind::Original {
                source: name,
                shape: vec![outputs],
            }
        } else {
            EntryKind::Zero
        },
    });
}
pub(super) fn entries(a: &FastH3AudioVaeConfig) -> Result<Vec<Entry>> {
    fn activation(out: &mut Vec<Entry>, prefix: &str, channels: u64) {
        for suffix in ["act.alpha", "act.beta"] {
            let name = format!("{prefix}.{suffix}");
            out.push(Entry {
                name: name.clone(),
                physical: vec![channels],
                kind: EntryKind::Original {
                    source: name,
                    shape: vec![channels],
                },
            });
        }
        for suffix in ["upsample.filter", "downsample.lowpass.filter"] {
            let name = format!("{prefix}.{suffix}");
            out.push(Entry {
                name: name.clone(),
                physical: vec![12],
                kind: EntryKind::Original {
                    source: name,
                    shape: vec![1, 1, 12],
                },
            });
        }
    }
    a.validate()?;
    let mut out = Vec::new();
    conv(
        &mut out,
        "dec_in_proj",
        a.latent_channels,
        a.latent_dim,
        1,
        false,
        false,
        true,
    );
    conv(
        &mut out,
        "decoder.conv_pre",
        a.latent_dim,
        a.decoder_dim,
        7,
        false,
        true,
        true,
    );
    let branches = a.resblock_kernel_sizes.len();
    let mut channels = a.decoder_dim;
    for stage in 0..a.decoder_rates.len() {
        conv(
            &mut out,
            &format!("decoder.ups.{stage}.0"),
            channels,
            channels / 2,
            a.decoder_kernel_sizes[stage],
            true,
            true,
            true,
        );
        channels /= 2;
        for branch in 0..branches {
            let prefix = format!("decoder.resblocks.{}", stage * branches + branch);
            for step in 0..a.resblock_dilation_sizes[branch].len() {
                for name in ["convs1", "convs2"] {
                    conv(
                        &mut out,
                        &format!("{prefix}.{name}.{step}"),
                        channels,
                        channels,
                        a.resblock_kernel_sizes[branch],
                        false,
                        true,
                        true,
                    );
                }
                activation(
                    &mut out,
                    &format!("{prefix}.activations.{}", 2 * step),
                    channels,
                );
                activation(
                    &mut out,
                    &format!("{prefix}.activations.{}", 2 * step + 1),
                    channels,
                );
            }
        }
    }
    activation(&mut out, "decoder.activation_post", channels);
    conv(
        &mut out,
        "decoder.conv_post",
        channels,
        1,
        7,
        false,
        true,
        false,
    );
    Ok(out)
}
enum Stored {
    Original {
        component: WeightComponentSpec,
        name: String,
        shape: Vec<u64>,
    },
    Derived {
        component: WeightComponentSpec,
        bytes: ManagedHostBytes,
        files: Vec<String>,
    },
}
impl Stored {
    fn component(&self) -> &WeightComponentSpec {
        match self {
            Self::Original { component, .. } | Self::Derived { component, .. } => component,
        }
    }
}

pub(super) struct PreparedAudioWeights {
    source: Arc<SafetensorsArchive>,
    stored: BTreeMap<WeightId, Stored>,
    pub(super) weights: Vec<AudioWeight>,
    pub(super) original_source_bytes: u64,
    pub(super) original_source_tensors: usize,
}

/// Only derived payloads acquire persistent WeightBacking; original tensors
/// keep their immutable archive mapping. Forecast each real backing including
/// allocator padding, in the same order/shape used by preparation below.
pub(super) fn derived_backing_bytes(entries: &[Entry], host: &HostMemoryAccount) -> Result<usize> {
    entries.iter().try_fold(0usize, |sum, entry| {
        if matches!(entry.kind, EntryKind::Original { .. }) {
            return Ok(sum);
        }
        let bytes = entry
            .physical
            .iter()
            .try_fold(4u64, |n, &d| n.checked_mul(d))
            .context("audio derived extent overflow")?;
        sum.checked_add(host.backing_bytes(usize::try_from(bytes)?)?)
            .context("audio derived Host backing overflow")
    })
}
impl PreparedAudioWeights {
    pub(super) fn prepare(
        source: Arc<SafetensorsArchive>,
        entries: Vec<Entry>,
        host: &HostMemoryAccount,
        limits: FastH3AudioComponentLimits,
    ) -> Result<Self> {
        ensure!(
            entries.len() <= limits.maximum_metadata_bytes / 32,
            "audio inventory exceeds metadata bound"
        );
        let stored_bytes = entries.iter().try_fold(0u64, |total, e| {
            let bytes = e
                .physical
                .iter()
                .try_fold(4u64, |n, &d| n.checked_mul(d))
                .context("audio weight extent overflow")?;
            total
                .checked_add(bytes)
                .context("audio stored extent overflow")
        })?;
        ensure!(
            stored_bytes <= limits.maximum_stored_bytes,
            "audio selected stored capacity exceeded"
        );
        let mut originals = BTreeSet::new();
        let mut original_source_bytes = 0u64;
        for e in &entries {
            let names = match &e.kind {
                EntryKind::Original { source, .. } => vec![source],
                EntryKind::WeightNorm { v, g, .. } => vec![v, g],
                EntryKind::ConcatVectors { sources, .. } => sources.iter().collect(),
                EntryKind::Zero => vec![],
            };
            for name in names {
                if originals.insert(name.clone()) {
                    let t = source.tensor(name)?;
                    ensure!(
                        t.element_type() == Some(ElementType::F32),
                        "audio tensor must retain original F32: {name}"
                    );
                    original_source_bytes = original_source_bytes
                        .checked_add(t.bytes().len() as u64)
                        .context("audio source extent overflow")?;
                }
            }
        }
        ensure!(
            original_source_bytes <= limits.maximum_source_bytes,
            "audio original source capacity exceeded"
        );
        let mut stored = BTreeMap::new();
        let mut weights = Vec::new();
        for e in entries {
            let mut digest = Sha256::new();
            digest.update(b"ferrum.fast_h3.audio.source.v1\0");
            digest.update(e.name.as_bytes());
            digest.update(serde_json::to_vec(&e.physical)?);
            let (backing, files) = match &e.kind {
                EntryKind::Original {
                    source: name,
                    shape,
                } => {
                    let t = source.tensor(name)?;
                    ensure!(
                        t.shape() == shape,
                        "audio original tensor shape differs: {name}"
                    );
                    finite(t.bytes(), name)?;
                    digest.update(t.bytes());
                    (None, vec![])
                }
                EntryKind::WeightNorm { v, g, shape } => {
                    let vt = source.tensor(v)?;
                    let gt = source.tensor(g)?;
                    ensure!(
                        vt.shape() == shape && gt.shape() == [shape[0], 1, 1],
                        "audio weight_norm original shapes differ: {v}"
                    );
                    finite(vt.bytes(), v)?;
                    finite(gt.bytes(), g)?;
                    digest.update(WEIGHT_NORM_RECIPE.as_bytes());
                    digest.update(gt.bytes());
                    digest.update(vt.bytes());
                    let bytes = weight_norm(host, vt.bytes(), gt.bytes(), shape[0])?;
                    (
                        Some(bytes),
                        // One file per ordered original name (g, v). Both
                        // originals can live in the same archive shard.
                        vec![gt.source_file().to_owned(), vt.source_file().to_owned()],
                    )
                }
                EntryKind::Zero => {
                    digest.update(b"constant-positive-zero-f32");
                    (
                        Some(
                            host.allocate_zeroed(HostMemoryClass::WeightBacking, 4)?
                                .freeze(),
                        ),
                        vec!["derived/constant-zero-f32".into()],
                    )
                }
                EntryKind::ConcatVectors { sources, width } => {
                    ensure!(
                        *width > 0 && !sources.is_empty(),
                        "empty audio vector concatenation"
                    );
                    let elements = width
                        .checked_mul(sources.len() as u64)
                        .context("audio concatenation overflow")?;
                    ensure!(
                        e.physical == [elements],
                        "audio concatenation must declare its exact flat vector"
                    );
                    let length = usize::try_from(
                        elements
                            .checked_mul(4)
                            .context("audio concatenation byte overflow")?,
                    )?;
                    let mut bytes = host.allocate_zeroed(HostMemoryClass::WeightBacking, length)?;
                    let mut files = Vec::new();
                    digest.update(b"ordered_original_f32_vectors.concat.v1");
                    for (i, name) in sources.iter().enumerate() {
                        let t = source.tensor(name)?;
                        ensure!(
                            t.shape() == [*width],
                            "audio concatenation source shape differs: {name}"
                        );
                        finite(t.bytes(), name)?;
                        digest.update(name.as_bytes());
                        digest.update(t.bytes());
                        let start = i * t.bytes().len();
                        bytes.as_mut()[start..start + t.bytes().len()].copy_from_slice(t.bytes());
                        files.push(t.source_file().to_owned());
                    }
                    (Some(bytes.freeze()), files)
                }
            };
            let component = WeightComponentSpec {
                id: id(format!("component.fast_h3.audio.{:x}", digest.finalize()))?,
                role: WeightComponentRole::Values,
                external_names: e.names(),
                dimensions: e.physical,
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            };
            weights.push(AudioWeight {
                name: e.name,
                component: component.clone(),
            });
            let retained = match (backing, e.kind) {
                (Some(bytes), _) => Stored::Derived {
                    component: component.clone(),
                    bytes,
                    files,
                },
                (
                    None,
                    EntryKind::Original {
                        source: name,
                        shape,
                    },
                ) => Stored::Original {
                    component: component.clone(),
                    name,
                    shape,
                },
                _ => unreachable!("declared source representation"),
            };
            ensure!(
                stored.insert(component.id.clone(), retained).is_none(),
                "duplicate source identity"
            );
        }
        Ok(Self {
            source,
            stored,
            weights,
            original_source_bytes,
            original_source_tensors: originals.len(),
        })
    }
}

/// Complete original decoder and its selected immutable source. The 135
/// weight-normalized matrices are derived once inside admitted WeightBacking;
/// ordinary F32 vectors and filters retain their original checkpoint mmap.
pub struct PreparedFastH3AudioDecoder {
    selected: PreparedAudioWeights,
    architecture: FastH3AudioVaeConfig,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    input: ResolvedTensorSpec,
    output: ResolvedTensorSpec,
}
impl PreparedFastH3AudioDecoder {
    pub const INPUT_ID: &'static str = "value.fast_h3.audio.input";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.audio.waveform";
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        architecture: &FastH3AudioVaeConfig,
        source: Arc<SafetensorsArchive>,
        requests: u64,
        latent_samples: u64,
        host: &HostMemoryAccount,
        limits: FastH3AudioDecoderLimits,
    ) -> Result<Self> {
        architecture.validate()?;
        ensure!(
            requests > 0 && latent_samples > 0,
            "positive request and latent sample counts required"
        );
        let batch = requests
            .checked_mul(2)
            .context("audio stereo batch overflow")?;
        let output_samples = latent_samples
            .checked_mul(architecture.hop_length()?)
            .context("audio sample extent overflow")?;
        for shape in [
            [batch, architecture.latent_channels, 1, latent_samples],
            [batch, 1, 1, output_samples],
        ] {
            let bytes = shape
                .into_iter()
                .try_fold(4u64, |n, d| n.checked_mul(d))
                .context("audio activation extent overflow")?;
            ensure!(
                bytes <= limits.maximum_activation_bytes && bytes <= isize::MAX as u64,
                "audio input/output exceeds capacity"
            );
        }
        let entries = entries(architecture)?;
        let selected = PreparedAudioWeights::prepare(source, entries, host, limits)?;
        let config = DecoderConfig {
            architecture: architecture.clone(),
            requests,
            latent_samples,
            weights: selected.weights.clone(),
        };
        config.validate(limits)?;
        let family = TypedFamilyRegistration::new(family::Family::new(limits)?)
            .prepare_with_profile(
                &serde_json::to_value(&config)?,
                &NumericalProfileId::new(family::Family::PROFILE_ID).map_err(anyhow::Error::msg)?,
            )?;
        let (program, specs) = program::build(&config, limits)?;
        let resolve = |s: &ProgramTensorSpec| {
            ResolvedTensorSpec::new(s.dimensions.clone(), s.element_type, s.layout.clone())
        };
        let input = resolve(&specs[&id::<ProgramValueId>(Self::INPUT_ID)?])?;
        let output = resolve(&specs[&id::<ProgramValueId>(Self::OUTPUT_ID)?])?;
        let options = ProgramPlanCompileOptions::new(specs)?;
        ensure!(
            program == *family.program(),
            "prepared audio program changed"
        );
        Ok(Self {
            selected,
            architecture: architecture.clone(),
            family,
            options,
            input,
            output,
        })
    }
    pub fn architecture(&self) -> &FastH3AudioVaeConfig {
        &self.architecture
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
}
impl WeightComponentSource for PreparedFastH3AudioDecoder {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.selected.component(c)
    }
}
impl WeightComponentSource for PreparedAudioWeights {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let stored = self
            .stored
            .get(&c.id)
            .filter(|s| s.component() == c)
            .ok_or_else(|| invalid("audio component differs from checked source"))?;
        match stored {
            Stored::Original { name, shape, .. } => {
                let t = self.source.tensor(name).map_err(invalid)?;
                if t.shape() != shape || t.element_type() != Some(ElementType::F32) {
                    return Err(invalid("immutable original audio source differs"));
                }
                WeightComponentPayload::new(
                    c,
                    name.clone(),
                    t.source_file(),
                    c.dimensions.clone(),
                    ElementType::F32,
                    t.bytes(),
                )?
                .with_retained_host_memory(t.retained_host_memory().clone())
            }
            Stored::Derived { bytes, files, .. } => WeightComponentPayload::from_managed_host(
                c,
                c.external_names.clone(),
                files.clone(),
                c.dimensions.clone(),
                ElementType::F32,
                bytes.clone(),
            ),
        }
    }
}
fn finite(bytes: &[u8], name: &str) -> Result<()> {
    ensure!(
        bytes
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
        "nonfinite original audio tensor: {name}"
    );
    Ok(())
}
fn weight_norm(
    host: &HostMemoryAccount,
    v: &[u8],
    g: &[u8],
    rows: u64,
) -> Result<ManagedHostBytes> {
    let rows = usize::try_from(rows)?;
    ensure!(
        rows > 0 && v.len() % 4 == 0 && v.len() / 4 % rows == 0 && g.len() == rows * 4,
        "invalid weight_norm physical extent"
    );
    let width = v.len() / 4 / rows;
    let mut out = host.allocate_zeroed(HostMemoryClass::WeightBacking, v.len())?;
    let f =
        |bytes: &[u8], i: usize| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    for row in 0..rows {
        let mut sums = [0f32; 256];
        for (lane, sum) in sums.iter_mut().enumerate() {
            for col in (lane..width).step_by(256) {
                let x = f(v, row * width + col);
                *sum = x.mul_add(x, *sum);
            }
        }
        for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
            for i in 0..stride {
                sums[i] += sums[i + stride];
            }
        }
        let norm = sums[0].sqrt();
        ensure!(
            norm.is_finite() && norm > 0.,
            "weight_norm norm is not finite positive"
        );
        let reciprocal = 1f32 / norm;
        let gain = f(g, row);
        for col in 0..width {
            let x = (gain * f(v, row * width + col)) * reciprocal;
            ensure!(
                x.is_finite(),
                "derived normalized audio weight is nonfinite"
            );
            out.as_mut()[4 * (row * width + col)..4 * (row * width + col + 1)]
                .copy_from_slice(&x.to_le_bytes());
        }
    }
    Ok(out.freeze())
}
