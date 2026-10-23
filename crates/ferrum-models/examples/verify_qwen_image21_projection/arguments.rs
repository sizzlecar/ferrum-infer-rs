use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::Serialize;

pub const USAGE: &str = "verify_qwen_image21_projection --config CONFIG --weight-fragment BF16 --weight-sha256 HEX --artifact-dir NEW_DIRECTORY_OUTSIDE_A_GIT_WORKTREE [--backend reference|cuda]\nCUDA requires --cuda-usable-memory-bytes N --cuda-bootstrap-memory-bytes N --cuda-stream-memory-bytes N --cuda-opaque-memory-profile NAME; optional --cuda-device N (0), --cuda-transfer-staging-bytes N (65536), --cuda-maximum-streams N (1), --cuda-maximum-in-flight-per-stream N (1), --cuda-maximum-commands-per-submission N (64, including upload/init/compute). Budgets are experimental declarations, not measured CUDA memory qualifications.";

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum Backend {
    Reference,
    Cuda(CudaOptions),
}

#[derive(Debug, Clone, Serialize)]
pub struct CudaOptions {
    pub device: usize,
    pub usable_memory_bytes: u64,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub transfer_staging_bytes: usize,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
}

impl Backend {
    pub fn ensure_available(&self) -> Result<()> {
        if matches!(self, Self::Cuda(_)) && !cfg!(feature = "cuda") {
            bail!("--backend cuda requires building this example with --features cuda");
        }
        Ok(())
    }
}

// Sibling examples reuse Backend/CudaOptions without the projection CLI parser.
#[allow(dead_code)]
pub struct Arguments {
    pub config: PathBuf,
    pub weight_fragment: PathBuf,
    pub weight_sha256: String,
    pub artifact_dir: PathBuf,
    pub backend: Backend,
}

impl Arguments {
    #[allow(dead_code)] // Used by the projection CLI; siblings parse their own inputs.
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
                .map_err(|_| anyhow::anyhow!("option names must be ASCII"))?;
            if !matches!(
                flag.as_str(),
                "--config"
                    | "--weight-fragment"
                    | "--weight-sha256"
                    | "--artifact-dir"
                    | "--backend"
                    | "--cuda-device"
                    | "--cuda-usable-memory-bytes"
                    | "--cuda-bootstrap-memory-bytes"
                    | "--cuda-stream-memory-bytes"
                    | "--cuda-transfer-staging-bytes"
                    | "--cuda-maximum-streams"
                    | "--cuda-maximum-in-flight-per-stream"
                    | "--cuda-maximum-commands-per-submission"
                    | "--cuda-opaque-memory-profile"
            ) {
                bail!("unknown option {flag:?}; {USAGE}");
            }
            let value = arguments.next().context("every option requires a value")?;
            if values.insert(flag.clone(), value).is_some() {
                bail!("duplicate option {flag:?}");
            }
        }
        let weight_sha256 = take_text(&mut values, "--weight-sha256")?;
        if weight_sha256.len() != 64 || !weight_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("--weight-sha256 requires exactly 64 hexadecimal digits");
        }
        let config = values.remove("--config").context(USAGE)?.into();
        let weight_fragment = values.remove("--weight-fragment").context(USAGE)?.into();
        let artifact_dir = values.remove("--artifact-dir").context(USAGE)?.into();
        let backend = Backend::parse_options(&mut values)?;
        Ok(Some(Self {
            config,
            weight_fragment,
            weight_sha256: weight_sha256.to_ascii_lowercase(),
            artifact_dir,
            backend,
        }))
    }
}

impl Backend {
    /// Shared typed backend options for the standalone verification examples.
    pub fn parse_options(mut values: &mut BTreeMap<String, OsString>) -> Result<Self> {
        let backend_name = values
            .remove("--backend")
            .unwrap_or_else(|| "reference".into());
        let backend = match backend_name.to_str() {
            Some("reference") => Backend::Reference,
            Some("cuda") => {
                let device = optional_number(&mut values, "--cuda-device", 0_usize)?;
                let usable_memory_bytes = number(&mut values, "--cuda-usable-memory-bytes")?;
                let bootstrap_memory_bytes = number(&mut values, "--cuda-bootstrap-memory-bytes")?;
                let stream_memory_bytes: u64 = number(&mut values, "--cuda-stream-memory-bytes")?;
                let transfer_staging_bytes =
                    optional_number(&mut values, "--cuda-transfer-staging-bytes", 65_536_usize)?;
                let maximum_streams =
                    optional_number(&mut values, "--cuda-maximum-streams", 1_usize)?;
                let maximum_in_flight_per_stream =
                    optional_number(&mut values, "--cuda-maximum-in-flight-per-stream", 1_usize)?;
                let maximum_commands_per_submission = optional_number(
                    &mut values,
                    "--cuda-maximum-commands-per-submission",
                    64_usize,
                )?;
                let opaque_memory_profile = take_text(&mut values, "--cuda-opaque-memory-profile")?;
                let overhead = stream_memory_bytes
                    .checked_mul(u64::try_from(maximum_streams)?)
                    .and_then(|bytes| bytes.checked_add(bootstrap_memory_bytes));
                if usable_memory_bytes == 0
                    || bootstrap_memory_bytes == 0
                    || stream_memory_bytes == 0
                    || maximum_streams == 0
                    || maximum_in_flight_per_stream == 0
                    || maximum_commands_per_submission == 0
                    || maximum_streams
                        .checked_mul(maximum_in_flight_per_stream)
                        .and_then(|flights| flights.checked_mul(maximum_commands_per_submission))
                        .is_none()
                    || overhead.is_none_or(|bytes| bytes >= usable_memory_bytes)
                    || transfer_staging_bytes < 2
                    || transfer_staging_bytes > isize::MAX as usize
                    || transfer_staging_bytes % 2 != 0
                    || opaque_memory_profile.is_empty()
                    || opaque_memory_profile.len() > 256
                    || !opaque_memory_profile.is_ascii()
                    || opaque_memory_profile
                        .bytes()
                        .any(|byte| byte.is_ascii_control())
                {
                    bail!("CUDA requires positive bounded allowances leaving device storage, finite lane/flight limits, an even BF16 staging budget, and an ASCII profile identity");
                }
                Backend::Cuda(CudaOptions {
                    device,
                    usable_memory_bytes,
                    bootstrap_memory_bytes,
                    stream_memory_bytes,
                    transfer_staging_bytes,
                    maximum_streams,
                    maximum_in_flight_per_stream,
                    maximum_commands_per_submission,
                    opaque_memory_profile,
                })
            }
            _ => bail!("--backend must be reference or cuda"),
        };
        if !values.is_empty() {
            bail!("CUDA options require --backend cuda");
        }
        Ok(backend)
    }
}

fn take_text(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<String> {
    values
        .remove(key)
        .with_context(|| format!("missing {key}; {USAGE}"))?
        .into_string()
        .map_err(|_| anyhow::anyhow!("{key} requires UTF-8 text"))
}

fn number<T: std::str::FromStr>(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<T> {
    take_text(values, key)?
        .parse()
        .map_err(|_| anyhow::anyhow!("{key} requires an unsigned integer within range"))
}

fn optional_number<T: std::str::FromStr>(
    values: &mut BTreeMap<String, OsString>,
    key: &str,
    default: T,
) -> Result<T> {
    if values.contains_key(key) {
        number(values, key)
    } else {
        Ok(default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Vec<OsString> {
        [
            "--config",
            "config.json",
            "--weight-fragment",
            "weight.bin",
            "--weight-sha256",
            &"a".repeat(64),
            "--artifact-dir",
            "out",
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    fn cuda() -> Vec<OsString> {
        let mut args = base();
        args.extend(
            [
                "--backend",
                "cuda",
                "--cuda-usable-memory-bytes",
                "2147483648",
                "--cuda-bootstrap-memory-bytes",
                "1073741824",
                "--cuda-stream-memory-bytes",
                "67108864",
                "--cuda-opaque-memory-profile",
                "experimental-unqualified-smoke-v1",
            ]
            .into_iter()
            .map(OsString::from),
        );
        args
    }

    #[test]
    fn reference_default_and_cuda_budgets_are_explicit() {
        assert!(matches!(
            Arguments::parse(base()).unwrap().unwrap().backend,
            Backend::Reference
        ));
        let Backend::Cuda(options) = Arguments::parse(cuda()).unwrap().unwrap().backend else {
            panic!("expected CUDA options")
        };
        assert_eq!(options.transfer_staging_bytes, 65_536);
        assert_eq!(options.maximum_streams, 1);
        assert_eq!(options.maximum_in_flight_per_stream, 1);
        assert_eq!(options.maximum_commands_per_submission, 64);
        let mut larger = cuda();
        larger.extend(
            ["--cuda-maximum-commands-per-submission", "1024"]
                .into_iter()
                .map(OsString::from),
        );
        let Backend::Cuda(larger) = Arguments::parse(larger).unwrap().unwrap().backend else {
            panic!("expected CUDA")
        };
        assert_eq!(larger.maximum_commands_per_submission, 1024);
        let mut wrong_backend = base();
        wrong_backend.extend(["--cuda-device", "0"].into_iter().map(OsString::from));
        assert!(Arguments::parse(wrong_backend).is_err());
    }

    #[test]
    fn cuda_rejects_missing_or_impossible_capacity_before_native_creation() {
        let mut incomplete = base();
        incomplete.extend(["--backend", "cuda"].into_iter().map(OsString::from));
        assert!(Arguments::parse(incomplete).is_err());
        for (flag, value) in [
            ("--cuda-transfer-staging-bytes", "1"),
            ("--cuda-maximum-streams", "0"),
            ("--cuda-maximum-streams", "18446744073709551615"),
            ("--cuda-maximum-in-flight-per-stream", "0"),
            ("--cuda-maximum-commands-per-submission", "0"),
        ] {
            let mut args = cuda();
            args.extend([flag, value].into_iter().map(OsString::from));
            assert!(Arguments::parse(args).is_err());
        }
    }
}
