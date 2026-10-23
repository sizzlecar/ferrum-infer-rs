use super::backend::Backend;
use anyhow::{ensure, Context, Result};
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

pub const USAGE:&str="verify_qwen_image21_denoise --config TRANSFORMER_CONFIG --model-dir TRANSFORMER_DIRECTORY --model-index MODEL_INDEX --scheduler-config SCHEDULER_CONFIG --latent-file INITIAL_PACKED_BF16 --text-file TEXT_BF16 --text-metadata TEXT_JSON --latent-height H --latent-width W --artifact-dir NEW_EXTERNAL_DIRECTORY --backend reference|cuda [--index-file diffusion_pytorch_model.safetensors.index.json] [--maximum-host-bytes N (536870912)] [--reference-memory-bytes N (1073741824)] [explicit CUDA budget options]\nExecutes every explicit sample_sigma (eight for the target metadata), initializing DiT once. Input latent is caller-supplied normalized packed BF16; no RNG is generated or authenticated. Output is normalized packed BF16, not a decoded image or Torch parity proof.";

pub struct Arguments {
    pub config: PathBuf,
    pub model_dir: PathBuf,
    pub model_index: PathBuf,
    pub scheduler_config: PathBuf,
    pub latent_file: PathBuf,
    pub text_file: PathBuf,
    pub text_metadata: PathBuf,
    pub artifact_dir: PathBuf,
    pub index_file: String,
    pub latent_height: u32,
    pub latent_width: u32,
    pub maximum_host_bytes: usize,
    pub reference_memory_bytes: u64,
    pub backend: Backend,
}
impl Arguments {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = args.into_iter().peekable();
        if args.peek().is_some_and(|x| x == "--help" || x == "-h") {
            args.next();
            ensure!(args.next().is_none(), "--help takes no other options");
            return Ok(None);
        }
        let mut v = BTreeMap::new();
        while let Some(k) = args.next() {
            let k = k
                .into_string()
                .map_err(|_| anyhow::anyhow!("option must be UTF-8"))?;
            let value = args.next().context("every option requires one value")?;
            ensure!(v.insert(k.clone(), value).is_none(), "duplicate option {k}");
        }
        let config = path(&mut v, "--config")?;
        let model_dir = path(&mut v, "--model-dir")?;
        let model_index = path(&mut v, "--model-index")?;
        let scheduler_config = path(&mut v, "--scheduler-config")?;
        let latent_file = path(&mut v, "--latent-file")?;
        let text_file = path(&mut v, "--text-file")?;
        let text_metadata = path(&mut v, "--text-metadata")?;
        let artifact_dir = path(&mut v, "--artifact-dir")?;
        let index_file = v
            .remove("--index-file")
            .unwrap_or_else(|| "diffusion_pytorch_model.safetensors.index.json".into())
            .into_string()
            .map_err(|_| anyhow::anyhow!("index filename must be UTF-8"))?;
        let latent_height = number(&mut v, "--latent-height", None)?;
        let latent_width = number(&mut v, "--latent-width", None)?;
        let maximum_host_bytes = number(&mut v, "--maximum-host-bytes", Some(536_870_912_usize))?;
        let reference_memory_bytes =
            number(&mut v, "--reference-memory-bytes", Some(1_073_741_824_u64))?;
        ensure!(
            maximum_host_bytes > 0
                && maximum_host_bytes <= isize::MAX as usize
                && reference_memory_bytes > 0
                && reference_memory_bytes <= isize::MAX as u64,
            "memory limits must be positive and addressable"
        );
        ensure!(
            v.contains_key("--backend"),
            "choose --backend explicitly; full Reference denoising can be slow"
        );
        let backend = Backend::parse_options(&mut v)?;
        Ok(Some(Self {
            config,
            model_dir,
            model_index,
            scheduler_config,
            latent_file,
            text_file,
            text_metadata,
            artifact_dir,
            index_file,
            latent_height,
            latent_width,
            maximum_host_bytes,
            reference_memory_bytes,
            backend,
        }))
    }
    pub fn staging_bytes(&self) -> usize {
        match &self.backend {
            Backend::Reference => 65536,
            Backend::Cuda(c) => c.transfer_staging_bytes,
        }
    }
    pub fn device_bytes(&self) -> u64 {
        match &self.backend {
            Backend::Reference => self.reference_memory_bytes,
            Backend::Cuda(c) => c.usable_memory_bytes,
        }
    }
}
fn path(v: &mut BTreeMap<String, OsString>, key: &str) -> Result<PathBuf> {
    Ok(v.remove(key)
        .with_context(|| format!("missing {key}; {USAGE}"))?
        .into())
}
fn number<T: std::str::FromStr>(
    v: &mut BTreeMap<String, OsString>,
    key: &str,
    default: Option<T>,
) -> Result<T> {
    let Some(value) = v.remove(key) else {
        return default.with_context(|| format!("missing {key}; {USAGE}"));
    };
    value
        .to_str()
        .context("number requires UTF-8")?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args() -> Vec<OsString> {
        [
            "--config",
            "c",
            "--model-dir",
            "m",
            "--model-index",
            "i",
            "--scheduler-config",
            "s",
            "--latent-file",
            "l",
            "--text-file",
            "t",
            "--text-metadata",
            "j",
            "--latent-height",
            "2",
            "--latent-width",
            "3",
            "--artifact-dir",
            "out",
            "--backend",
            "reference",
        ]
        .into_iter()
        .map(Into::into)
        .collect()
    }
    #[test]
    fn external_latent_and_metadata_are_explicit_no_seed_substitution() {
        assert!(Arguments::parse(args()).unwrap().is_some());
        let mut missing = args();
        let i = missing.iter().position(|s| s == "--latent-file").unwrap();
        missing.drain(i..i + 2);
        assert!(Arguments::parse(missing).is_err());
        for (flag, value) in [
            ("--seed", "42"),
            ("--steps", "8"),
            ("--maximum-host-bytes", "0"),
        ] {
            let mut invalid = args();
            invalid.extend([flag.into(), value.into()]);
            assert!(Arguments::parse(invalid).is_err());
        }
    }
}
