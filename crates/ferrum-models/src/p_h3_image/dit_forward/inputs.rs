use super::*;
use anyhow::{ensure, Context, Result};
use half::bf16;

#[derive(Clone, Copy)]
pub(crate) struct PH3ImageDiTSample<'a> {
    pub image: &'a ManagedHostTensor,
    pub audio: &'a ManagedHostTensor,
}
/// Noise is supplied in patch order. This does not invent a replacement for
/// Comfy's separately seeded CPU Torch RNG or silently round references to B.
#[derive(Clone, Copy)]
pub(crate) struct PH3ImageDiTReference<'a> {
    pub latent: &'a ManagedHostTensor,
    pub patch_noise: Option<&'a ManagedHostTensor>,
    /// Python scalar precision is retained until each F32 tensor multiplier.
    pub augmentation: f64,
}
impl<'a> PH3ImageDiTReference<'a> {
    pub fn default_augmented(
        latent: &'a ManagedHostTensor,
        patch_noise: &'a ManagedHostTensor,
    ) -> Self {
        Self {
            latent,
            patch_noise: Some(patch_noise),
            augmentation: 0.999,
        }
    }
}
pub(crate) enum PH3ImageDiTTextInput<'a> {
    WithTextRefiner {
        connector: &'a ManagedHostTensor,
        keys: &'a [cache::PH3ImageDiTConditioningKey],
    },
    RefinedText {
        caches: &'a [PH3ImageRefinedTextCache],
        keys: &'a [cache::PH3ImageDiTConditioningKey],
    },
}
pub(crate) struct PreparedPH3ImageDiTInputs {
    pub inputs: Vec<ManagedHostTensor>,
    pub(super) conditioning: Option<Vec<cache::PH3ImageDiTConditioningKey>>,
}
pub(crate) struct PH3ImageDiTInputAdapter<'a> {
    config: &'a PH3ImageDiTConfig,
    host: &'a HostMemoryAccount,
}
impl<'a> PH3ImageDiTInputAdapter<'a> {
    pub const NUMERICAL_PROFILE_ID:&'static str="p_h3_image.dit_inputs.target_f32_after_bf16.reference_f32_supplied_patch_noise.f64_scalar_complement.separate_mul_add.v2";
    pub fn new(config: &'a PH3ImageDiTConfig, host: &'a HostMemoryAccount) -> Result<Self> {
        config.validate()?;
        ensure!(
            cfg!(target_endian = "little"),
            "DiT adapter requires little-endian Host"
        );
        Ok(Self { config, host })
    }
    /// All new large buffers are charged to the existing Host. Failure drops
    /// every local allocation and publishes neither a partial input set nor a
    /// conditioning cache. Original F32 solver samples are only borrowed.
    pub fn prepare(
        &self,
        samples: &[PH3ImageDiTSample<'_>],
        references: &[Option<PH3ImageDiTReference<'_>>],
        text: PH3ImageDiTTextInput<'_>,
        basis: ManagedHostTensor,
        cosine: ManagedHostTensor,
        sine: ManagedHostTensor,
    ) -> Result<PreparedPH3ImageDiTInputs> {
        let references = references
            .iter()
            .map(|reference| reference.iter().copied().collect())
            .collect::<Vec<Vec<_>>>();
        self.prepare_references(samples, &references, text, basis, cosine, sine)
    }

    pub fn prepare_references(
        &self,
        samples: &[PH3ImageDiTSample<'_>],
        references: &[Vec<PH3ImageDiTReference<'_>>],
        text: PH3ImageDiTTextInput<'_>,
        basis: ManagedHostTensor,
        cosine: ManagedHostTensor,
        sine: ManagedHostTensor,
    ) -> Result<PreparedPH3ImageDiTInputs> {
        let c = self.config;
        ensure!(
            c.phase != PH3ImageDiTPhase::RefineOnly,
            "RefineOnly uses the fresh text-only session consumer"
        );
        ensure!(
            samples.len() == c.requests.len() && references.len() == samples.len(),
            "DiT request/sample/reference counts differ"
        );
        for ((r, s), reference) in c.requests.iter().zip(samples).zip(references) {
            check(
                self.host,
                s.image,
                &[1, c.channels, 1, r.target.height, r.target.width],
                ElementType::F32,
            )?;
            check(
                self.host,
                s.audio,
                &[1, c.audio_channels, 2, 2],
                ElementType::F32,
            )?;
            ensure!(
                r.reference_extents().count() == reference.len(),
                "ordered reference extents and provided values differ"
            );
            for (extent, reference) in r.reference_extents().zip(reference) {
                ensure!(
                    reference.augmentation.is_finite()
                        && (0.0..=1.0).contains(&reference.augmentation),
                    "reference augmentation must lie in [0,1]"
                );
                check(
                    self.host,
                    reference.latent,
                    &[1, c.channels, 1, extent.height, extent.width],
                    ElementType::F32,
                )?;
                if reference.augmentation < 1.0 {
                    let noise = reference
                        .patch_noise
                        .context("reference augmentation requires explicit patch-order noise")?;
                    check(
                        self.host,
                        noise,
                        &[config::patch_rows(extent, c)?, c.image_width()?],
                        ElementType::F32,
                    )?;
                } else if let Some(noise) = reference.patch_noise {
                    check(
                        self.host,
                        noise,
                        &[config::patch_rows(extent, c)?, c.image_width()?],
                        ElementType::F32,
                    )?;
                }
            }
        }
        let image_spec = config::tensor(vec![c.image_rows()?, c.image_width()?], ElementType::F32);
        let audio_spec = config::tensor(vec![c.audio_rows()?, c.audio_channels], ElementType::F32);
        let mut image = self.host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(image_spec.byte_len()?)?,
        )?;
        let mut audio = self.host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(audio_spec.byte_len()?)?,
        )?;
        let mut image_offset = 0usize;
        for (index, ((r, s), reference)) in
            c.requests.iter().zip(samples).zip(references).enumerate()
        {
            for (extent, reference) in r.reference_extents().zip(reference) {
                let elements = usize::try_from(
                    config::patch_rows(extent, c)?
                        .checked_mul(c.image_width()?)
                        .context("reference elements overflow")?,
                )?;
                patchify(c, extent, reference.latent, |i, v| {
                    let v = if reference.augmentation == 1.0 {
                        v
                    } else {
                        let n = read(reference.patch_noise.unwrap(), i);
                        // Comfy subtracts its Python float before either scalar
                        // becomes a F32 tensor multiplier. Keep the products
                        // and addition separate, without FMA contraction.
                        let signal = (reference.augmentation as f32) * v;
                        let noise = ((1.0_f64 - reference.augmentation) as f32) * n;
                        signal + noise
                    };
                    image.as_mut()[4 * (image_offset + i)..4 * (image_offset + i + 1)]
                        .copy_from_slice(&v.to_le_bytes());
                })?;
                image_offset += elements;
            }
            let elements = usize::try_from(
                r.target_rows(c)?
                    .checked_mul(c.image_width()?)
                    .context("target elements overflow")?,
            )?;
            patchify(c, r.target, s.image, |i, v| {
                image.as_mut()[4 * (image_offset + i)..4 * (image_offset + i + 1)]
                    .copy_from_slice(&bf16::from_f32(v).to_f32().to_le_bytes())
            })?;
            image_offset += elements;
            for row in 0..4usize {
                for channel in 0..usize::try_from(c.audio_channels)? {
                    let v = bf16::from_f32(read(s.audio, 4 * channel + row)).to_f32();
                    let at = (4 * index + row) * usize::try_from(c.audio_channels)? + channel;
                    audio.as_mut()[4 * at..4 * at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        ensure!(
            image
                .as_ref()
                .chunks_exact(4)
                .chain(audio.as_ref().chunks_exact(4))
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "prepared patch values must remain finite F32"
        );
        let (text, conditioning) = match (c.phase, text) {
            (
                PH3ImageDiTPhase::WithTextRefiner,
                PH3ImageDiTTextInput::WithTextRefiner { connector, keys },
            ) => {
                cache::validate_keys(c, keys)?;
                check(
                    self.host,
                    connector,
                    &[c.text_rows()?, c.text_dim],
                    ElementType::F32,
                )?;
                cache::validate_connector_keys(c, connector, keys)?;
                let mut out = self.host.allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    usize::try_from(
                        c.text_rows()?
                            .checked_mul(c.text_dim)
                            .and_then(|n| n.checked_mul(2))
                            .context("text bytes overflow")?,
                    )?,
                )?;
                for (source, target) in connector
                    .bytes()
                    .bytes()
                    .chunks_exact(4)
                    .zip(out.as_mut().chunks_exact_mut(2))
                {
                    target.copy_from_slice(
                        &bf16::from_f32(f32::from_le_bytes(source.try_into().unwrap()))
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
                (
                    managed(
                        PH3ImageDiTFamily::RAW_TEXT_ID,
                        config::tensor(vec![c.text_rows()?, c.text_dim], ElementType::Bf16),
                        out.freeze(),
                    )?,
                    Some(keys.to_vec()),
                )
            }
            (PH3ImageDiTPhase::RefinedText, PH3ImageDiTTextInput::RefinedText { caches, keys }) => {
                (cache::concatenate(c, self.host, caches, keys)?, None)
            }
            _ => anyhow::bail!("text phase and supplied input differ"),
        };
        let mut inputs = vec![
            managed(
                PH3ImageDiTFamily::IMAGE_PATCHES_ID,
                image_spec,
                image.freeze(),
            )?,
            managed(
                PH3ImageDiTFamily::AUDIO_PATCHES_ID,
                audio_spec,
                audio.freeze(),
            )?,
            text,
            basis,
            cosine,
            sine,
        ];
        let expected = c.input_tensor_specs()?;
        ensure!(inputs.len() == expected.len(), "DiT input arity differs");
        for input in &inputs {
            let spec = expected
                .get(input.value_id())
                .context("DiT input has an undeclared ID")?;
            check(self.host, input, &spec.dimensions, spec.element_type)?;
        }
        inputs.sort_by(|a, b| a.value_id().cmp(b.value_id()));
        ensure!(
            !inputs
                .windows(2)
                .any(|p| p[0].value_id() == p[1].value_id()),
            "duplicate DiT input"
        );
        Ok(PreparedPH3ImageDiTInputs {
            inputs,
            conditioning,
        })
    }
}
pub(super) fn check(
    host: &HostMemoryAccount,
    t: &ManagedHostTensor,
    shape: &[u64],
    dtype: ElementType,
) -> Result<()> {
    ensure!(
        host.owns(t.bytes())
            && t.bytes().class() == HostMemoryClass::InputPreprocess
            && t.tensor().dimensions() == shape
            && t.tensor().element_type() == dtype
            && t.tensor().layout() == &ResolvedTensorLayout::Contiguous,
        "DiT tensor owner, shape, dtype or layout differs"
    );
    let finite = match dtype {
        ElementType::F32 => t
            .bytes()
            .bytes()
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
        ElementType::Bf16 => t
            .bytes()
            .bytes()
            .chunks_exact(2)
            .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),
        _ => false,
    };
    ensure!(finite, "DiT input must be finite");
    Ok(())
}
pub(super) fn managed(
    id: &str,
    spec: ProgramTensorSpec,
    bytes: ManagedHostBytes,
) -> Result<ManagedHostTensor> {
    Ok(ManagedHostTensor::new(
        ProgramValueId::new(id)?,
        ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)?,
        bytes,
    )?)
}
fn read(t: &ManagedHostTensor, i: usize) -> f32 {
    f32::from_le_bytes(t.bytes().bytes()[4 * i..4 * i + 4].try_into().unwrap())
}
fn patchify(
    c: &PH3ImageDiTConfig,
    e: config::Extent,
    t: &ManagedHostTensor,
    mut write: impl FnMut(usize, f32),
) -> Result<()> {
    let [height, width, ph, pw, channels] =
        [e.height, e.width, c.patch_h, c.patch_w, c.channels].map(usize::try_from);
    let (height, width, ph, pw, channels) = (height?, width?, ph?, pw?, channels?);
    let mut index = 0;
    // Official rearrange: b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw), pt=1.
    for y in 0..height / ph {
        for x in 0..width / pw {
            for channel in 0..channels {
                for dy in 0..ph {
                    for dx in 0..pw {
                        write(
                            index,
                            read(t, (channel * height + y * ph + dy) * width + x * pw + dx),
                        );
                        index += 1;
                    }
                }
            }
        }
    }
    Ok(())
}
