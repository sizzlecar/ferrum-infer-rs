use super::backend::Backend;
use anyhow::{bail, Context, Result};
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

pub const USAGE: &str = "verify_qwen_image21_transformer --config CONFIG --model-dir TRANSFORMER_DIRECTORY --latent-file PACKED_BF16 --text-file PADDED_TEXT_BF16 --text-tokens T --valid-text-lengths L0,L1 --latent-height H --latent-width W --scheduler-timesteps T0,T1 --artifact-dir NEW_EXTERNAL_DIRECTORY --backend reference|cuda [--index-file diffusion_pytorch_model.safetensors.index.json] [--maximum-input-bytes N (268435456)] [--reference-memory-bytes N (1073741824)] [explicit CUDA budget options]\nLatents are [B*H*W,Cin], text is [B*T,context_dim], little-endian BF16; timestep values are scheduler units in [0,1000]. This executes one transformer velocity evaluation, not a denoising loop or image generation. CUDA needs a command limit covering every graph node plus four input uploads (698 for 32 layers).";

pub struct Arguments {
    pub config: PathBuf,
    pub model_dir: PathBuf,
    pub latent_file: PathBuf,
    pub text_file: PathBuf,
    pub artifact_dir: PathBuf,
    pub index_file: String,
    pub text_tokens: u32,
    pub valid_text_lengths: Vec<u32>,
    pub latent_height: u32,
    pub latent_width: u32,
    pub scheduler_timesteps: Vec<f32>,
    pub maximum_input_bytes: usize,
    pub reference_memory_bytes: u64,
    pub backend: Backend,
}

impl Arguments {
    pub fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut arguments = arguments.into_iter().peekable();
        if arguments
            .peek()
            .is_some_and(|value| value == "--help" || value == "-h")
        {
            arguments.next();
            if arguments.next().is_some() {
                bail!("--help accepts no other arguments");
            }
            return Ok(None);
        }
        let mut values = BTreeMap::new();
        while let Some(flag) = arguments.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option names require UTF-8"))?;
            let value = arguments
                .next()
                .context("every option requires one value")?;
            if values.insert(flag.clone(), value).is_some() {
                bail!("duplicate option {flag}");
            }
        }
        let config = values.remove("--config").context(USAGE)?.into();
        let model_dir = values.remove("--model-dir").context(USAGE)?.into();
        let latent_file = values.remove("--latent-file").context(USAGE)?.into();
        let text_file = values.remove("--text-file").context(USAGE)?.into();
        let artifact_dir = values.remove("--artifact-dir").context(USAGE)?.into();
        let index_file = values
            .remove("--index-file")
            .unwrap_or_else(|| "diffusion_pytorch_model.safetensors.index.json".into())
            .into_string()
            .map_err(|_| anyhow::anyhow!("index filename requires UTF-8"))?;
        let text_tokens = number(&mut values, "--text-tokens")?;
        let latent_height = number(&mut values, "--latent-height")?;
        let latent_width = number(&mut values, "--latent-width")?;
        let valid_text_lengths = list(&mut values, "--valid-text-lengths")?;
        let scheduler_timesteps: Vec<f32> = list(&mut values, "--scheduler-timesteps")?;
        if scheduler_timesteps.len() != valid_text_lengths.len()
            || scheduler_timesteps
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1000.0).contains(value))
        {
            bail!("one finite scheduler timestep in [0,1000] is required per batch sample");
        }
        let maximum_input_bytes = if values.contains_key("--maximum-input-bytes") {
            number(&mut values, "--maximum-input-bytes")?
        } else {
            268_435_456_usize
        };
        let reference_memory_bytes = if values.contains_key("--reference-memory-bytes") {
            number(&mut values, "--reference-memory-bytes")?
        } else {
            1_073_741_824_u64
        };
        if maximum_input_bytes == 0
            || maximum_input_bytes > isize::MAX as usize
            || reference_memory_bytes == 0
            || reference_memory_bytes > isize::MAX as u64
        {
            bail!(
                "Host input and Reference memory limits must be positive addressable byte counts"
            );
        }
        if !values.contains_key("--backend") {
            bail!("choose --backend explicitly; a full Reference transformer can be slow");
        }
        let backend = Backend::parse_options(&mut values)?;
        Ok(Some(Self {
            config,
            model_dir,
            latent_file,
            text_file,
            artifact_dir,
            index_file,
            text_tokens,
            valid_text_lengths,
            latent_height,
            latent_width,
            scheduler_timesteps,
            maximum_input_bytes,
            reference_memory_bytes,
            backend,
        }))
    }
}

fn text(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<String> {
    values
        .remove(key)
        .with_context(|| format!("missing {key}; {USAGE}"))?
        .into_string()
        .map_err(|_| anyhow::anyhow!("{key} requires UTF-8"))
}
fn number<T: std::str::FromStr>(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<T> {
    text(values, key)?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid numeric {key}"))
}
fn list<T: std::str::FromStr>(
    values: &mut BTreeMap<String, OsString>,
    key: &str,
) -> Result<Vec<T>> {
    let value = text(values, key)?;
    let result = value
        .split(',')
        .map(|part| {
            part.parse()
                .map_err(|_| anyhow::anyhow!("invalid comma-separated {key}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if result.is_empty() || result.len() > 16 {
        bail!("{key} requires 1..16 samples");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn base() -> Vec<OsString> {
        [
            "--config",
            "c",
            "--model-dir",
            "m",
            "--latent-file",
            "l",
            "--text-file",
            "t",
            "--text-tokens",
            "3",
            "--valid-text-lengths",
            "3,1",
            "--latent-height",
            "2",
            "--latent-width",
            "4",
            "--scheduler-timesteps",
            "1000,499.015625",
            "--artifact-dir",
            "out",
            "--backend",
            "reference",
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }
    #[test]
    fn actual_inputs_and_per_sample_scheduler_values_are_required() {
        let args = Arguments::parse(base()).unwrap().unwrap();
        assert_eq!(args.valid_text_lengths, [3, 1]);
        assert_eq!(args.scheduler_timesteps, [1000.0, 499.015625]);
        for (flag, value) in [
            ("--scheduler-timesteps", "NaN,1"),
            ("--scheduler-timesteps", "1"),
            ("--scheduler-timesteps", "1001,1"),
            ("--maximum-input-bytes", "0"),
            ("--unknown", "1"),
        ] {
            let mut args = base();
            if let Some(index) = args.iter().position(|part| part == flag) {
                args[index + 1] = value.into();
            } else {
                args.extend([flag.into(), value.into()]);
            }
            assert!(Arguments::parse(args).is_err());
        }
        let mut args = base();
        args.truncate(args.len() - 2);
        assert!(Arguments::parse(args).is_err());
    }
}
