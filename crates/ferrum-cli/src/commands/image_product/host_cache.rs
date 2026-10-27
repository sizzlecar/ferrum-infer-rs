//! Complete converted-TE caching for the CUDA directory product default.
//! Availability is an observation; the live Host account authorizes allocations.
use super::configuration::HostConfig;
use anyhow::{Context, Result};
use ferrum_engine::image_generation::{
    ImageFp8HostCacheBudgets, ImageFp8HostCachePolicy, ImageTextEncoderFp8CacheRequirements,
};
use serde::Serialize;
use std::num::NonZeroU64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum AutoHostCacheDecision {
    CompleteTextEncoderEnabled,
    InsufficientAvailableMemory,
    AvailableMemoryUnknown,
    TextEncoderNotFp8,
}

#[derive(Debug, Serialize)]
pub(super) struct AutoHostCacheReceipt {
    pub policy: &'static str,
    pub decision: AutoHostCacheDecision,
    pub observed_available_bytes: Option<u64>,
    pub required_cache_bytes: Option<u64>,
    pub required_host_total_bytes: Option<usize>,
    pub effective_host_total_bytes: usize,
    pub effective_weight_backing_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_error: Option<String>,
    pub observation_scope: &'static str,
}

pub(super) struct AutoHostCacheSelection {
    pub host: HostConfig,
    pub cache: ImageFp8HostCachePolicy,
    pub receipt: AutoHostCacheReceipt,
}

#[cfg(feature = "cuda")]
pub(super) fn for_directory(
    options: &super::directory::DirectoryOptions,
    assets: &ferrum_engine::image_generation::QwenImage21Assets,
) -> Result<AutoHostCacheSelection> {
    let host = options.host();
    let requirements = options
        .precision
        .stage_uses_fp8(ferrum_engine::image_generation::ImageStage::TextEncoder)
        .then(|| assets.text_encoder_fp8_host_cache_requirements(host.limits()?))
        .transpose()?;
    // Observe after archive mapping/metadata preparation, before creating CUDA
    // or a managed payload. BF16 needs neither this query nor an expanded pool.
    let available = if requirements.is_some() {
        ferrum_kernels::backend::cpu::vnext_runtime::host_memory_available()
            .map_err(|error| error.to_string())
    } else {
        Ok(0)
    };
    select(host, requirements, available)
}

fn select(
    mut host: HostConfig,
    requirements: Option<ImageTextEncoderFp8CacheRequirements>,
    available: Result<u64, String>,
) -> Result<AutoHostCacheSelection> {
    let mut receipt = AutoHostCacheReceipt {
        policy: "auto_complete_text_encoder",
        decision: AutoHostCacheDecision::TextEncoderNotFp8,
        observed_available_bytes: None,
        required_cache_bytes: None,
        required_host_total_bytes: None,
        effective_host_total_bytes: host.total_bytes,
        effective_weight_backing_bytes: host.weight_backing_bytes,
        observation_error: None,
        observation_scope: "available physical memory limited by remaining process address space and cgroup headroom; snapshot only, no OS reservation; managed Host allocations remain checked; mmap/page-cache, allocator and native-library memory are outside the Host ledger",
    };
    let mut cache = ImageFp8HostCachePolicy::Off;
    if let Some(requirements) = requirements {
        let cache_bytes = NonZeroU64::new(requirements.converted_cache_bytes)
            .context("complete text encoder FP8 cache has no converted weights")?;
        let conversion_bytes = usize::try_from(requirements.maximum_conversion_group_bytes)
            .context("FP8 conversion group exceeds Host address space")?;
        // Preserve the existing transient allowance; a larger declared group
        // must also coexist with all retained weights. Never select a partial
        // cache that continually evicts the model it is meant to retain.
        let backing = host
            .weight_backing_bytes
            .max(conversion_bytes)
            .checked_add(usize::try_from(cache_bytes.get())?)
            .context("complete FP8 Host cache allowance overflows")?;
        let total = [
            host.input_preprocess_bytes,
            host.transfer_staging_bytes,
            host.output_bytes,
        ]
        .into_iter()
        .try_fold(backing, |total, bytes| total.checked_add(bytes))
        .context("complete FP8 Host coexistence allowance overflows")?
        .max(host.total_bytes);
        receipt.required_cache_bytes = Some(cache_bytes.get());
        receipt.required_host_total_bytes = Some(total);
        match available {
            Ok(bytes) => {
                receipt.observed_available_bytes = Some(bytes);
                if bytes >= u64::try_from(total)? {
                    host.total_bytes = total;
                    host.weight_backing_bytes = backing;
                    host.limits()?;
                    cache = ImageFp8HostCachePolicy::StageBudgets {
                        maximum_bytes_by_stage: ImageFp8HostCacheBudgets {
                            text_encoder: Some(cache_bytes),
                            denoiser: None,
                        },
                    };
                    receipt.decision = AutoHostCacheDecision::CompleteTextEncoderEnabled;
                    receipt.effective_host_total_bytes = total;
                    receipt.effective_weight_backing_bytes = backing;
                } else {
                    receipt.decision = AutoHostCacheDecision::InsufficientAvailableMemory;
                }
            }
            Err(error) => {
                receipt.decision = AutoHostCacheDecision::AvailableMemoryUnknown;
                receipt.observation_error = Some(error);
            }
        }
    }
    Ok(AutoHostCacheSelection {
        host,
        cache,
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> HostConfig {
        HostConfig {
            total_bytes: 1000,
            input_preprocess_bytes: 600,
            transfer_staging_bytes: 100,
            output_bytes: 300,
            weight_backing_bytes: 200,
        }
    }

    #[test]
    fn complete_cache_requires_all_coexisting_allowances_and_falls_back_without_partial_cache() {
        let requirements = Some(ImageTextEncoderFp8CacheRequirements {
            converted_cache_bytes: 4000,
            maximum_conversion_group_bytes: 100,
        });
        let enabled = select(host(), requirements, Ok(5200)).unwrap();
        assert_eq!(enabled.host.total_bytes, 5200);
        assert_eq!(enabled.host.weight_backing_bytes, 4200);
        assert_eq!(
            enabled
                .cache
                .maximum_bytes(ferrum_engine::image_generation::ImageStage::TextEncoder),
            NonZeroU64::new(4000)
        );
        assert_eq!(
            enabled.receipt.decision,
            AutoHostCacheDecision::CompleteTextEncoderEnabled
        );
        for observation in [Ok(5199), Err("headroom unavailable".to_owned())] {
            let selected = select(host(), requirements, observation.clone()).unwrap();
            assert_eq!(selected.cache, ImageFp8HostCachePolicy::Off);
            assert_eq!(selected.host.total_bytes, 1000);
            assert_eq!(selected.host.weight_backing_bytes, 200);
            assert_eq!(selected.receipt.required_host_total_bytes, Some(5200));
            assert_eq!(
                selected.receipt.decision,
                if observation.is_ok() {
                    AutoHostCacheDecision::InsufficientAvailableMemory
                } else {
                    AutoHostCacheDecision::AvailableMemoryUnknown
                }
            );
        }
    }

    #[test]
    fn conversion_peak_is_metadata_derived_and_non_fp8_keeps_original_host_limits() {
        let selected = select(
            host(),
            Some(ImageTextEncoderFp8CacheRequirements {
                converted_cache_bytes: 4000,
                maximum_conversion_group_bytes: 500,
            }),
            Ok(5500),
        )
        .unwrap();
        assert_eq!(selected.host.total_bytes, 5500);
        assert_eq!(selected.host.weight_backing_bytes, 4500);
        let unchanged = select(host(), None, Err("must not matter".to_owned())).unwrap();
        assert_eq!(unchanged.cache, ImageFp8HostCachePolicy::Off);
        assert_eq!(unchanged.host.total_bytes, 1000);
        assert_eq!(
            unchanged.receipt.decision,
            AutoHostCacheDecision::TextEncoderNotFp8
        );
        assert!(unchanged.receipt.observed_available_bytes.is_none());
        assert!(unchanged.receipt.observation_error.is_none());
    }
}
