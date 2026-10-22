use super::backend::Backend;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

pub const USAGE: &str = "verify_qwen_image21_vae --model-root MODEL_ROOT --latent-file RAW_LITTLE_ENDIAN_BF16 --latent-layout packed|nchw --latent-domain normalized|denormalized --latent-height H --latent-width W --batch B --artifact-dir NEW_EXTERNAL_DIRECTORY --backend reference|cuda [--config VAE_CONFIG] [--index-file BASENAME] [--maximum-host-bytes N (268435456)] [--reference-memory-bytes N (2147483648)] [CUDA budgets] [--cudnn-library ABSOLUTE_PATH --cudnn-opaque-memory-bytes N --cudnn-workspace-bytes N --cudnn-memory-profile NAME]\nH/W are LATENT spatial dimensions. Packed is [B,H*W,C], not 2x2 patches. Output is raw BF16 NCHW RGBA plus per-sample interleaved RGBA8 and PNG (straight alpha retained; PNG rows limited to 65536 bytes). No denoising, numerical parity or throughput claim. Real 148-node decoder requires a CUDA command budget covering compute and input upload (use 256).";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentLayout {
    Packed,
    Nchw,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentDomain {
    Normalized,
    Denormalized,
}
#[derive(Debug, Serialize)]
pub struct CudnnOptions {
    pub library: PathBuf,
    pub opaque_memory_bytes: u64,
    pub workspace_bytes: u64,
    pub memory_profile: String,
}
pub struct Arguments {
    pub model_root: PathBuf,
    pub config: PathBuf,
    pub index_file: Option<String>,
    pub latent_file: PathBuf,
    pub layout: LatentLayout,
    pub domain: LatentDomain,
    pub height: u64,
    pub width: u64,
    pub batch: u64,
    pub artifact_dir: PathBuf,
    pub maximum_host_bytes: usize,
    pub reference_memory_bytes: u64,
    pub backend: Backend,
    pub cudnn: Option<CudnnOptions>,
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
        let model_root: PathBuf = values.remove("--model-root").context(USAGE)?.into();
        let config = values
            .remove("--config")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_root.join("vae/config.json"));
        let index_file = values
            .remove("--index-file")
            .map(|s| {
                s.into_string()
                    .map_err(|_| anyhow::anyhow!("index requires UTF-8"))
            })
            .transpose()?;
        if let Some(name) = &index_file {
            ensure!(
                !name.is_empty()
                    && !matches!(name.as_str(), "." | "..")
                    && !name.contains(['/', '\\', ':']),
                "index must be a basename"
            );
        }
        let latent_file = values.remove("--latent-file").context(USAGE)?.into();
        let artifact_dir = values.remove("--artifact-dir").context(USAGE)?.into();
        let layout = match text(&mut values, "--latent-layout")?.as_str() {
            "packed" => LatentLayout::Packed,
            "nchw" => LatentLayout::Nchw,
            _ => anyhow::bail!("latent layout must be packed or nchw"),
        };
        let domain = match text(&mut values, "--latent-domain")?.as_str() {
            "normalized" => LatentDomain::Normalized,
            "denormalized" => LatentDomain::Denormalized,
            _ => anyhow::bail!("latent domain must be normalized or denormalized"),
        };
        let height = required_number(&mut values, "--latent-height")?;
        let width = required_number(&mut values, "--latent-width")?;
        let batch = required_number(&mut values, "--batch")?;
        ensure!(
            height > 0 && width > 0 && batch > 0,
            "latent dimensions must be positive"
        );
        let maximum_host_bytes = number(&mut values, "--maximum-host-bytes", 268_435_456_usize)?;
        let explicit_reference = values.contains_key("--reference-memory-bytes");
        let reference_memory_bytes =
            number(&mut values, "--reference-memory-bytes", 2_147_483_648_u64)?;
        ensure!(
            maximum_host_bytes > 0
                && maximum_host_bytes <= isize::MAX as usize
                && reference_memory_bytes > 0
                && reference_memory_bytes <= isize::MAX as u64,
            "memory budgets must be positive/addressable"
        );
        let backend_name = values
            .get("--backend")
            .and_then(|v| v.to_str())
            .context("choose --backend reference|cuda explicitly")?;
        let cudnn = if backend_name == "cuda" {
            ensure!(
                !explicit_reference,
                "Reference budget applies only to Reference"
            );
            let library: PathBuf = values
                .remove("--cudnn-library")
                .context("CUDA requires --cudnn-library")?
                .into();
            let opaque_memory_bytes = required_number(&mut values, "--cudnn-opaque-memory-bytes")?;
            let workspace_bytes = required_number(&mut values, "--cudnn-workspace-bytes")?;
            let memory_profile = text(&mut values, "--cudnn-memory-profile")?;
            ensure!(
                library.is_absolute()
                    && opaque_memory_bytes > 0
                    && workspace_bytes <= isize::MAX as u64,
                "cuDNN requires absolute library and bounded explicit allowances"
            );
            ensure!(
                !memory_profile.is_empty()
                    && memory_profile.len() <= 256
                    && memory_profile.is_ascii()
                    && !memory_profile.bytes().any(|v| v.is_ascii_control()),
                "invalid cuDNN memory profile"
            );
            Some(CudnnOptions {
                library,
                opaque_memory_bytes,
                workspace_bytes,
                memory_profile,
            })
        } else {
            None
        };
        let backend = Backend::parse_options(&mut values)?;
        Ok(Some(Self {
            model_root,
            config,
            index_file,
            latent_file,
            layout,
            domain,
            height,
            width,
            batch,
            artifact_dir,
            maximum_host_bytes,
            reference_memory_bytes,
            backend,
            cudnn,
        }))
    }
    pub fn staging_bytes(&self) -> usize {
        match &self.backend {
            Backend::Reference => 65_536,
            Backend::Cuda(c) => c.transfer_staging_bytes,
        }
    }
}
fn text(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<String> {
    values
        .remove(key)
        .with_context(|| format!("missing {key}"))?
        .into_string()
        .map_err(|_| anyhow::anyhow!("{key} requires UTF-8"))
}
fn required_number<T: std::str::FromStr>(
    values: &mut BTreeMap<String, OsString>,
    key: &str,
) -> Result<T> {
    text(values, key)?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {key}"))
}
fn number<T: std::str::FromStr>(
    values: &mut BTreeMap<String, OsString>,
    key: &str,
    default: T,
) -> Result<T> {
    if values.contains_key(key) {
        required_number(values, key)
    } else {
        Ok(default)
    }
}
