use super::backend::Backend;
use anyhow::{bail, ensure, Context, Result};
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

pub const USAGE: &str = "verify_qwen3_vl_text --model-dir MODEL_ROOT --prompt TEXT --artifact-dir NEW_EXTERNAL_DIRECTORY --backend reference|cuda [--maximum-user-tokens N (8192)] [--maximum-host-bytes N (268435456)] [--reference-memory-bytes N (1073741824)] [explicit CUDA budgets]\nPure T2I: actual processor tokenizer/template and text_encoder/model.safetensors; emits right-padded pre-final-norm text.bf16. No vision branch, Torch parity, denoising or image claim. CUDA command limit must cover 17*layers+3 (615 for 36 layers; use 1024).";
pub struct Arguments {
    pub model_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub prompt: String,
    pub maximum_user_tokens: usize,
    pub maximum_host_bytes: usize,
    pub reference_memory_bytes: u64,
    pub backend: Backend,
}
impl Arguments {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = args.into_iter().peekable();
        if args.peek().is_some_and(|a| a == "--help" || a == "-h") {
            args.next();
            ensure!(args.next().is_none(), "--help accepts no other arguments");
            return Ok(None);
        }
        let mut values = BTreeMap::new();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option names require UTF-8"))?;
            let value = args.next().context("every option requires a value")?;
            ensure!(
                values.insert(flag.clone(), value).is_none(),
                "duplicate option {flag}"
            );
        }
        let model_dir = values.remove("--model-dir").context(USAGE)?.into();
        let artifact_dir = values.remove("--artifact-dir").context(USAGE)?.into();
        let prompt = values
            .remove("--prompt")
            .context(USAGE)?
            .into_string()
            .map_err(|_| anyhow::anyhow!("prompt requires UTF-8"))?;
        let maximum_user_tokens = number(&mut values, "--maximum-user-tokens", 8192_usize)?;
        let maximum_host_bytes = number(&mut values, "--maximum-host-bytes", 268_435_456_usize)?;
        let reference_memory_bytes =
            number(&mut values, "--reference-memory-bytes", 1_073_741_824_u64)?;
        ensure!(
            (1..=8192).contains(&maximum_user_tokens),
            "user token budget must be 1..=8192"
        );
        ensure!(
            maximum_host_bytes > 0
                && maximum_host_bytes <= isize::MAX as usize
                && reference_memory_bytes > 0
                && reference_memory_bytes <= isize::MAX as u64,
            "memory limits require positive addressable byte counts"
        );
        if !values.contains_key("--backend") {
            bail!("choose --backend explicitly; a full Reference text tower can be slow");
        }
        let backend = Backend::parse_options(&mut values)?;
        Ok(Some(Self {
            model_dir,
            artifact_dir,
            prompt,
            maximum_user_tokens,
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
fn number<T: std::str::FromStr>(
    values: &mut BTreeMap<String, OsString>,
    key: &str,
    default: T,
) -> Result<T> {
    match values.remove(key) {
        None => Ok(default),
        Some(v) => v
            .into_string()
            .ok()
            .and_then(|s| s.parse().ok())
            .with_context(|| format!("invalid {key}")),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_source_and_backend_are_explicit_and_limits_are_checked() {
        let base = [
            "--model-dir",
            "m",
            "--prompt",
            "你好",
            "--artifact-dir",
            "o",
            "--backend",
            "reference",
        ]
        .map(OsString::from);
        let args = Arguments::parse(base.clone()).unwrap().unwrap();
        assert_eq!(args.prompt, "你好");
        assert!(Arguments::parse(base[..6].iter().cloned()).is_err());
        for (key, value) in [
            ("--maximum-user-tokens", "8193"),
            ("--maximum-host-bytes", "0"),
            ("--reference-memory-bytes", "0"),
            ("--unknown", "1"),
        ] {
            let mut v = base.to_vec();
            v.extend([key.into(), value.into()]);
            assert!(Arguments::parse(v).is_err());
        }
    }
}
