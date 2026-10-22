use super::{artifacts, shared::id};
use anyhow::{ensure, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21ModelTimestep, QwenImage21Timestep256, QwenImage21TransformerConfig,
    QwenImage21TransformerFamily as Family,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, path::Path};

#[derive(Debug, Deserialize, Serialize)]
pub struct TextMetadata {
    pub sha256: String,
    pub bytes: usize,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub text_tokens: u32,
    pub valid_text_lengths: Vec<u32>,
    pub nonfinite_count: u64,
}
impl TextMetadata {
    pub fn validate(
        &self,
        config: &QwenImage21TransformerConfig,
        budget: &HostBudget,
    ) -> Result<()> {
        let tensor = tensor(config, Family::TEXT_ID)?;
        ensure!(
            self.dtype == "BF16"
                && self.nonfinite_count == 0
                && self.shape == tensor.dimensions()
                && self.text_tokens == config.layout().text_tokens()
                && self.valid_text_lengths == config.layout().valid_text_lengths()
                && self.bytes == budget.inputs[1]
                && self.sha256.len() == 64
                && self.sha256.bytes().all(|x| x.is_ascii_hexdigit()),
            "TE artifact metadata does not match finite padded BF16 text geometry"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HostBudget {
    pub inputs: [usize; 4],
    pub input_peak: usize,
    pub velocity: usize,
    pub staging: usize,
    pub total: usize,
}
impl HostBudget {
    pub fn new(
        config: &QwenImage21TransformerConfig,
        staging: usize,
        maximum: usize,
    ) -> Result<Self> {
        let lengths = Family::INPUT_IDS.map(|name| {
            tensor(config, name).and_then(|s| Ok(usize::try_from(s.minimum_storage_bytes()?)?))
        });
        let [a, b, c, d] = lengths;
        let inputs = [a?, b?, c?, d?];
        ensure!(
            config.transformer().in_channels() == config.transformer().out_channels(),
            "Euler requires velocity and latent channel counts to match"
        );
        let input_peak = inputs
            .iter()
            .try_fold(inputs[0], |sum, n| sum.checked_add(*n))
            .context("Host input peak overflow")?;
        let velocity = inputs[0];
        let total = input_peak
            .checked_add(velocity)
            .and_then(|n| n.checked_add(staging))
            .context("Host peak overflow")?;
        ensure!(
            staging > 0 && total <= maximum && total <= isize::MAX as usize,
            "denoise Host peak {total} exceeds --maximum-host-bytes before payload reads"
        );
        Ok(Self {
            inputs,
            input_peak,
            velocity,
            staging,
            total,
        })
    }
    pub fn account(&self) -> Result<HostMemoryAccount> {
        Ok(HostMemoryAccount::new(HostMemoryLimits::new(
            self.total,
            HostMemoryClassLimits {
                input_preprocess_bytes: self.input_peak,
                output_bytes: self.velocity,
                transfer_staging_bytes: self.staging,
                weight_backing_bytes: 0,
            },
        )?))
    }
}

pub struct Inputs {
    pub sample: ManagedHostTensor,
    pub text: ManagedHostTensor,
    pub frequencies: ManagedHostTensor,
}
impl Inputs {
    pub fn read(
        config: &QwenImage21TransformerConfig,
        host: &HostMemoryAccount,
        budget: &HostBudget,
        latent: &Path,
        text: &Path,
        metadata: &TextMetadata,
    ) -> Result<Self> {
        metadata.validate(config, budget)?;
        let reservations = host.reserve_many(
            &[0, 1, 3].map(|i| (HostMemoryClass::InputPreprocess, budget.inputs[i])),
        )?;
        let mut buffers = reservations
            .into_iter()
            .map(|r| r.allocate_zeroed())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        read_exact_finite(latent, buffers[0].as_mut())?;
        read_exact_finite(text, buffers[1].as_mut())?;
        ensure!(
            artifacts::sha256(buffers[1].as_ref()) == metadata.sha256.to_ascii_lowercase(),
            "TE payload SHA differs from text.json"
        );
        let values = bytemuck::try_cast_slice_mut::<u8, f32>(buffers[2].as_mut())
            .map_err(|e| anyhow::anyhow!("managed rotary buffer alignment: {e}"))?;
        config
            .rotary_frequencies()?
            .write_into(config.layout(), values)?;
        for value in values {
            *value = f32::from_bits(value.to_bits().to_le());
        }
        let tensors = buffers
            .into_iter()
            .zip([Family::LATENTS_ID, Family::TEXT_ID, Family::FREQUENCIES_ID])
            .map(|(b, name)| {
                ManagedHostTensor::new(id(name)?, tensor(config, name)?, b.freeze())
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        let [sample, text, frequencies]: [ManagedHostTensor; 3] = tensors
            .try_into()
            .map_err(|_| anyhow::anyhow!("input arity"))?;
        Ok(Self {
            sample,
            text,
            frequencies,
        })
    }
}

pub fn time_input(
    config: &QwenImage21TransformerConfig,
    host: &HostMemoryAccount,
    adapter: &QwenImage21Timestep256,
    timestep: f32,
) -> Result<ManagedHostTensor> {
    let spec = tensor(config, Family::TIME_FEATURES_ID)?;
    let mut storage = host.allocate_zeroed(
        HostMemoryClass::InputPreprocess,
        usize::try_from(spec.minimum_storage_bytes()?)?,
    )?;
    let values = bytemuck::try_cast_slice_mut::<u8, u16>(storage.as_mut())
        .map_err(|e| anyhow::anyhow!("managed timestep buffer alignment: {e}"))?;
    let time = QwenImage21ModelTimestep::from_scheduler_timestep(timestep)?;
    adapter.write_into(
        config,
        &vec![time; usize::try_from(config.layout().batch())?],
        values,
    )?;
    for value in values {
        *value = value.to_le();
    }
    Ok(ManagedHostTensor::new(
        id(Family::TIME_FEATURES_ID)?,
        spec,
        storage.freeze(),
    )?)
}
pub fn tensor(config: &QwenImage21TransformerConfig, name: &str) -> Result<ResolvedTensorSpec> {
    let specs = config.input_tensor_specs()?;
    let s = specs
        .get(&id(name)?)
        .context("missing typed transformer input")?;
    Ok(ResolvedTensorSpec::new(
        s.dimensions.clone(),
        s.element_type,
        s.layout.clone(),
    )?)
}
pub fn finite(bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() % 2 == 0 && !bytes.is_empty(),
        "BF16 requires a nonempty even byte extent"
    );
    ensure!(
        !bytes
            .chunks_exact(2)
            .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80),
        "nonfinite BF16 payload"
    );
    Ok(())
}
fn read_exact_finite(path: &Path, bytes: &mut [u8]) -> Result<()> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.len() == bytes.len() as u64,
        "{} differs from exact typed byte extent",
        path.display()
    );
    file.read_exact(bytes)?;
    let mut extra = [0];
    ensure!(
        file.read(&mut extra)? == 0,
        "input changed past its expected extent"
    );
    finite(bytes)
}
