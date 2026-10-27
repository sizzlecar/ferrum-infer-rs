//! Complete converted-TE caching for the CUDA directory product default.
//! Availability is an observation; the live Host account authorizes allocations.
use super::configuration::HostConfig;
use anyhow::{Context, Result};
use ferrum_engine::image_generation::{
    ImageFp8HostCacheBudgets, ImageFp8HostCachePolicy, ImageOutputRequirements,
    ImageTextEncoderFp8CacheRequirements,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum AutoOutputDecision {
    FullOverlap,
    PartialAvailableMemory,
    InsufficientAvailableMemory,
    AvailableMemoryUnknown,
}

#[derive(Debug, Serialize)]
pub(super) struct AutoOutputReceipt {
    pub decision: AutoOutputDecision,
    pub pipeline: ImageOutputRequirements,
    pub response_batch_bytes: usize,
    pub required_output_bytes: usize,
    pub effective_output_bytes: usize,
    pub ownership_scope: &'static str,
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
    pub output_selection: Option<AutoOutputReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_error: Option<String>,
    pub observation_scope: &'static str,
}

pub(super) struct AutoHostCacheSelection {
    pub host: HostConfig,
    pub cache: ImageFp8HostCachePolicy,
    pub receipt: AutoHostCacheReceipt,
}

#[cfg(any(feature = "cuda", test))]
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
    let account = ferrum_runtime::HostMemoryAccount::new(host.limits()?);
    let output = assets.output_requirements(&options.pipeline().resolved_limits()?, &account)?;
    let serving = options.serving()?;
    let http = ferrum_server::ImageHttpConfig {
        output: serving.output,
        default_width: serving.default_width,
        default_height: serving.default_height,
        default_seed: serving.default_seed,
        rgba_scratch_bytes: serving.rgba_scratch_bytes,
        png_limits: ferrum_output::png::PngLimits::default(),
    };
    let response_bytes = http.response_backing_bytes(
        &account,
        ferrum_output::png::PngDimensions::new(
            output.maximum_width.into(),
            output.maximum_height.into(),
        )?,
    )?;
    // One snapshot covers both decisions after archive mapping, before CUDA
    // construction or managed payload allocation. BF16 also sizes output.
    let available = ferrum_kernels::backend::cpu::vnext_runtime::host_memory_available()
        .map_err(|error| error.to_string());
    select(
        host,
        requirements,
        Some((output, response_bytes)),
        available,
    )
}

fn select(
    mut host: HostConfig,
    requirements: Option<ImageTextEncoderFp8CacheRequirements>,
    output: Option<(ImageOutputRequirements, usize)>,
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
        output_selection: None,
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
        match &available {
            Ok(bytes) => {
                receipt.observed_available_bytes = Some(*bytes);
                if *bytes >= u64::try_from(total)? {
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
                receipt.observation_error = Some(error.clone());
            }
        }
    }
    if let Some((pipeline, response_bytes)) = output {
        let response_batch_bytes = response_bytes
            .checked_mul(pipeline.batch_size)
            .context("HTTP output batch backing overflows")?;
        // The completed batch can still own raw readbacks and all reserved
        // encoding backing while the next batch owns its execution scope and
        // response gates. These are lifecycle owners, not a queue multiplier.
        let required_output_bytes = [
            pipeline.raw_batch_bytes,
            pipeline.execution_output_bytes,
            response_batch_bytes,
            response_batch_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |n, bytes| n.checked_add(bytes))
        .context("overlapping image output backing overflows")?;
        let decision = match available {
            Ok(bytes) => {
                receipt.observed_available_bytes = Some(bytes);
                let other = [
                    host.input_preprocess_bytes,
                    host.transfer_staging_bytes,
                    host.weight_backing_bytes,
                ]
                .into_iter()
                .try_fold(0usize, |n, bytes| n.checked_add(bytes))
                .context("non-output Host allowance overflows")?;
                let remaining = usize::try_from(bytes.saturating_sub(u64::try_from(other)?))
                    .unwrap_or(usize::MAX);
                let selected = required_output_bytes.min(remaining).max(host.output_bytes);
                let decision = if selected >= required_output_bytes {
                    AutoOutputDecision::FullOverlap
                } else if selected > host.output_bytes {
                    AutoOutputDecision::PartialAvailableMemory
                } else {
                    AutoOutputDecision::InsufficientAvailableMemory
                };
                if selected > host.output_bytes {
                    host.output_bytes = selected;
                    host.total_bytes = host.total_bytes.max(
                        other
                            .checked_add(selected)
                            .context("selected Host output total overflows")?,
                    );
                }
                decision
            }
            Err(error) => {
                receipt.observation_error = Some(error);
                AutoOutputDecision::AvailableMemoryUnknown
            }
        };
        receipt.output_selection = Some(AutoOutputReceipt {
            decision, pipeline, response_batch_bytes, required_output_bytes,
            effective_output_bytes: host.output_bytes,
            ownership_scope: "conservative metadata bound: previous completed batch raw + HTTP backing, next execution readback + HTTP backing; allowance only, no batch preallocation; exact live Host grants remain authoritative",
        });
    }
    host.limits()?;
    receipt.effective_host_total_bytes = host.total_bytes;
    receipt.effective_weight_backing_bytes = host.weight_backing_bytes;
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
        let enabled = select(host(), requirements, None, Ok(5200)).unwrap();
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
            let selected = select(host(), requirements, None, observation.clone()).unwrap();
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
            None,
            Ok(5500),
        )
        .unwrap();
        assert_eq!(selected.host.total_bytes, 5500);
        assert_eq!(selected.host.weight_backing_bytes, 4500);
        let unchanged = select(host(), None, None, Err("must not matter".to_owned())).unwrap();
        assert_eq!(unchanged.cache, ImageFp8HostCachePolicy::Off);
        assert_eq!(unchanged.host.total_bytes, 1000);
        assert_eq!(
            unchanged.receipt.decision,
            AutoHostCacheDecision::TextEncoderNotFp8
        );
        assert!(unchanged.receipt.observed_available_bytes.is_none());
        assert!(unchanged.receipt.observation_error.is_none());
    }

    #[test]
    fn output_overlap_uses_remaining_shared_headroom_after_complete_cache() {
        let cache = Some(ImageTextEncoderFp8CacheRequirements {
            converted_cache_bytes: 4000,
            maximum_conversion_group_bytes: 100,
        });
        let output = Some((
            ImageOutputRequirements {
                batch_size: 2,
                maximum_width: 32,
                maximum_height: 32,
                raw_batch_bytes: 400,
                execution_output_bytes: 600,
            },
            50,
        ));
        // Other classes with the complete cache cost 4900; the lifecycle
        // target costs 400 + 600 + 2*100 = 1200, all on the same observation.
        for (available, expected, decision) in [
            (Ok(6100), 1200, AutoOutputDecision::FullOverlap),
            (Ok(5700), 800, AutoOutputDecision::PartialAvailableMemory),
            (
                Ok(5200),
                300,
                AutoOutputDecision::InsufficientAvailableMemory,
            ),
            (
                Err("unknown".to_owned()),
                300,
                AutoOutputDecision::AvailableMemoryUnknown,
            ),
        ] {
            let selected = select(host(), cache, output, available.clone()).unwrap();
            let receipt = selected.receipt.output_selection.unwrap();
            assert_eq!(receipt.required_output_bytes, 1200);
            assert_eq!(receipt.decision, decision);
            assert_eq!(selected.host.output_bytes, expected);
            if let Ok(bytes) = available {
                assert_eq!(selected.host.total_bytes as u64, bytes);
                assert_eq!(selected.host.weight_backing_bytes, 4200);
            } else {
                assert_eq!(selected.cache, ImageFp8HostCachePolicy::Off);
                assert_eq!(selected.host.total_bytes, 1000);
            }
        }
        let bf16 = select(host(), None, output, Ok(2100)).unwrap();
        assert_eq!(bf16.host.output_bytes, 1200);
        assert_eq!(bf16.cache, ImageFp8HostCachePolicy::Off);
    }
}
