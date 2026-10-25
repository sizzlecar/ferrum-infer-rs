//! Typed backend declarations for the standalone complete decoder diagnostic.
#[path = "../verify_qwen_image21_projection/arguments.rs"]
mod shared;
pub use shared::USAGE as CUDA_USAGE;

use anyhow::{bail, ensure, Result};
use serde::Serialize;
use std::{collections::BTreeMap, ffi::OsString};

pub const USAGE: &str = "Metal requires --features metal on macOS and --metal-usable-memory-bytes N. Optional --metal-bootstrap-memory-bytes N (134217728), --metal-stream-memory-bytes N (2097152), --metal-maximum-streams N (2), --metal-maximum-in-flight-per-stream N (1), --metal-maximum-commands-per-submission N (256), --metal-transfer-staging-bytes N (16777216). All are declared limits, not measured peak memory. CUDA accepts the shared explicit --cuda-* budgets.";

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum Backend {
    Reference,
    Cuda(shared::CudaOptions),
    Metal(MetalOptions),
}

#[derive(Debug, Clone, Serialize)]
pub struct MetalOptions {
    pub usable_memory_bytes: u64,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub transfer_staging_bytes: usize,
}

pub struct Runtime {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub metal:
        Option<std::sync::Arc<ferrum_kernels::backend::metal::vnext_runtime::MetalDeviceRuntime>>,
}

impl Runtime {
    pub fn new(backend: &Backend) -> Result<Self> {
        backend.ensure_available()?;
        Ok(Self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal: match backend {
                Backend::Metal(options) => Some(super::metal::create(options)?),
                _ => None,
            },
        })
    }

    pub fn host_account(
        &self,
        budget: &super::latent::Budget,
        maximum: usize,
    ) -> Result<ferrum_runtime::HostMemoryAccount> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(runtime) = &self.metal {
            let prototype = runtime.tensor_host_account(budget.account()?.limits())?;
            return Ok(runtime.tensor_host_account(budget.aligned_limits(&prototype, maximum)?)?);
        }
        let _ = maximum;
        budget.account()
    }
}

impl Backend {
    pub fn ensure_available(&self) -> Result<()> {
        match self {
            Self::Reference => Ok(()),
            Self::Cuda(c) => shared::Backend::Cuda(c.clone()).ensure_available(),
            Self::Metal(_) if cfg!(all(feature = "metal", target_os = "macos")) => Ok(()),
            Self::Metal(_) => bail!("--backend metal requires macOS and --features metal"),
        }
    }

    pub fn parse_options(values: &mut BTreeMap<String, OsString>) -> Result<Self> {
        if values.get("--backend").is_none_or(|v| v != "metal") {
            return Ok(match shared::Backend::parse_options(values)? {
                shared::Backend::Reference => Self::Reference,
                shared::Backend::Cuda(c) => Self::Cuda(c),
            });
        }
        values.remove("--backend");
        use super::arguments::{number, required_number};
        let c = MetalOptions {
            usable_memory_bytes: required_number(values, "--metal-usable-memory-bytes")?,
            bootstrap_memory_bytes: number(values, "--metal-bootstrap-memory-bytes", 128 << 20)?,
            stream_memory_bytes: number(values, "--metal-stream-memory-bytes", 2 << 20)?,
            maximum_streams: number(values, "--metal-maximum-streams", 2)?,
            maximum_in_flight_per_stream: number(
                values,
                "--metal-maximum-in-flight-per-stream",
                1,
            )?,
            maximum_commands_per_submission: number(
                values,
                "--metal-maximum-commands-per-submission",
                256,
            )?,
            transfer_staging_bytes: number(values, "--metal-transfer-staging-bytes", 16 << 20)?,
        };
        ensure!(
            values.is_empty(),
            "unknown or incompatible Metal backend options"
        );
        let overhead = u64::try_from(c.maximum_streams)
            .ok()
            .and_then(|n| n.checked_mul(c.stream_memory_bytes))
            .and_then(|n| n.checked_add(c.bootstrap_memory_bytes));
        ensure!(
            c.usable_memory_bytes <= isize::MAX as u64
                && c.bootstrap_memory_bytes > 0 && c.stream_memory_bytes > 0
                && c.maximum_streams > 0 && c.maximum_in_flight_per_stream > 0
                && c.maximum_commands_per_submission > 0
                && overhead.is_some_and(|n| n < c.usable_memory_bytes)
                && c.maximum_streams.checked_mul(c.maximum_in_flight_per_stream)
                    .and_then(|n| n.checked_mul(c.maximum_commands_per_submission)).is_some()
                && c.transfer_staging_bytes >= 2
                && c.transfer_staging_bytes <= isize::MAX as usize
                && c.transfer_staging_bytes % 2 == 0,
            "Metal requires positive bounded allowances leaving device storage, bounded submission counts and even BF16 staging bytes"
        );
        Ok(Self::Metal(c))
    }
}
