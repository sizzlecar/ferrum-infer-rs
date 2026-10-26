//! Fixed upstream c0551d pipeline_qwen_image_21.py `_pack_latents` and
//! `_decode_latents`: [B,HW,C], channel mean/std converted F32 then BF16,
//! B(B(x * B(std)) + B(mean)). The file reader writes directly to a charged
//! NCHW backing; its only payload scratch is a fixed 8 KiB stack buffer.
use super::{
    arguments::{LatentDomain, LatentLayout},
    shared::id,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21VaeDecoderConfig as Config, QwenImage21VaeDecoderFamily as Family,
};
use ferrum_runtime::*;
use half::bf16;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

pub struct Statistics {
    pub mean: Vec<bf16>,
    pub std: Vec<bf16>,
}
impl Statistics {
    pub fn from_config(raw: &Value, channels: usize) -> Result<Self> {
        let parse = |key: &str, positive: bool| -> Result<Vec<bf16>> {
            let values = raw[key]
                .as_array()
                .with_context(|| format!("missing {key}"))?;
            ensure!(
                values.len() == channels,
                "{key} must contain exactly {channels} channels"
            );
            values
                .iter()
                .map(|value| {
                    let f = value
                        .as_f64()
                        .context("latent statistics must be numeric")?
                        as f32;
                    let b = bf16::from_f32(f);
                    ensure!(
                        f.is_finite() && b.is_finite() && (!positive || b.to_f32() > 0.0),
                        "invalid finite/positive BF16 latent statistic"
                    );
                    Ok(b)
                })
                .collect()
        };
        Ok(Self {
            mean: parse("latents_mean", false)?,
            std: parse("latents_std", true)?,
        })
    }
}
#[derive(Debug, Serialize)]
pub struct Budget {
    pub input: usize,
    pub output: usize,
    pub staging: usize,
    pub total: usize,
}
impl Budget {
    pub fn new(config: &Config, staging: usize, maximum: usize) -> Result<Self> {
        let input = usize::try_from(config.input_tensor()?.minimum_storage_bytes()?)?;
        let output = usize::try_from(config.output_tensor()?.minimum_storage_bytes()?)?;
        let total = input
            .checked_add(output)
            .and_then(|n| n.checked_add(staging))
            .context("Host budget overflow")?;
        ensure!(
            staging > 0 && total <= maximum && total <= isize::MAX as usize,
            "required Host bytes {total} exceed budget {maximum}"
        );
        Ok(Self {
            input,
            output,
            staging,
            total,
        })
    }
    pub fn account(&self) -> Result<HostMemoryAccount> {
        Ok(HostMemoryAccount::new(HostMemoryLimits::new(
            self.total,
            HostMemoryClassLimits {
                input_preprocess_bytes: self.input,
                output_bytes: self.output,
                transfer_staging_bytes: self.staging,
                weight_backing_bytes: 0,
            },
        )?))
    }

    #[cfg(any(test, all(feature = "metal", target_os = "macos")))]
    pub fn aligned_limits(
        &self,
        account: &HostMemoryAccount,
        maximum: usize,
    ) -> Result<HostMemoryLimits> {
        let input = account.backing_bytes(self.input)?;
        let output = account.backing_bytes(self.output)?;
        let staging = account.backing_bytes(self.staging)?;
        let total = input
            .checked_add(output)
            .and_then(|n| n.checked_add(staging))
            .context("aligned Host budget overflow")?;
        ensure!(
            total <= maximum,
            "aligned Host backing {total} exceeds budget {maximum}"
        );
        Ok(HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: input,
                output_bytes: output,
                transfer_staging_bytes: staging,
                weight_backing_bytes: 0,
            },
        )?)
    }
}
pub struct Loaded {
    pub tensor: ManagedHostTensor,
    pub source_sha256: String,
    pub prepared_sha256: String,
}
pub fn load(
    path: &Path,
    config: &Config,
    layout: LatentLayout,
    domain: LatentDomain,
    statistics: Option<&Statistics>,
    host: &HostMemoryAccount,
) -> Result<Loaded> {
    let tensor = config.input_tensor()?;
    let bytes = usize::try_from(tensor.minimum_storage_bytes()?)?;
    let channels = usize::try_from(config.latent_channels())?;
    let plane = usize::try_from(
        config
            .height()
            .checked_mul(config.width())
            .context("latent plane overflow")?,
    )?;
    if domain == LatentDomain::Normalized {
        let stats = statistics.context("normalized input requires channel statistics")?;
        ensure!(
            stats.mean.len() == channels && stats.std.len() == channels,
            "statistics differ from latent channels"
        );
    }
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() == u64::try_from(bytes)?,
        "latent file must have exact little-endian BF16 extent {bytes}"
    );
    let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
    let mut scratch = [0_u8; 8192];
    let mut digest = Sha256::new();
    let mut consumed = 0_usize;
    while consumed < bytes {
        let count = (bytes - consumed).min(scratch.len());
        file.read_exact(&mut scratch[..count])?;
        digest.update(&scratch[..count]);
        for (offset, raw) in scratch[..count].chunks_exact(2).enumerate() {
            let index = consumed / 2 + offset;
            let (channel, destination) = match layout {
                LatentLayout::Nchw => ((index / plane) % channels, index),
                LatentLayout::Packed => {
                    let c = index % channels;
                    let p = (index / channels) % plane;
                    let b = index / (channels * plane);
                    (c, (b * channels + c) * plane + p)
                }
            };
            let input = bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]]));
            ensure!(
                input.is_finite(),
                "nonfinite latent at source element {index}"
            );
            let value = match domain {
                LatentDomain::Denormalized => input,
                LatentDomain::Normalized => {
                    let stats = statistics.context("missing normalized statistics")?;
                    let product = bf16::from_f32(input.to_f32() * stats.std[channel].to_f32());
                    ensure!(
                        product.is_finite(),
                        "latent product overflows BF16 at element {index}"
                    );
                    bf16::from_f32(product.to_f32() + stats.mean[channel].to_f32())
                }
            };
            ensure!(
                value.is_finite(),
                "denormalized latent overflows BF16 at element {index}"
            );
            output.as_mut()[destination * 2..destination * 2 + 2]
                .copy_from_slice(&value.to_bits().to_le_bytes());
        }
        consumed += count;
    }
    ensure!(
        file.read(&mut scratch[..1])? == 0,
        "latent file grew while reading"
    );
    let output = output.freeze();
    let prepared_sha256 = format!("{:x}", Sha256::digest(output.bytes()));
    Ok(Loaded {
        tensor: ManagedHostTensor::new(id(Family::INPUT_ID)?, tensor, output)?,
        source_sha256: format!("{:x}", digest.finalize()),
        prepared_sha256,
    })
}
