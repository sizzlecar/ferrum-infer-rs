//! Explicit Host residency for converted weights; independent of device residency.
use super::{ImageCudaPrecisionPolicy, ImageStage};
use anyhow::{ensure, Context};
pub use ferrum_quantization::bf16_whole_tensor_fp8::ComposedWeightHostRequirements as ImageLoraHostRequirements;
use ferrum_runtime::{HostMemoryClass, HostMemoryLimits};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

/// Metadata-only demand for a complete converted text encoder on the CUDA
/// composition's byte-aligned Host account. No tensor payload is allocated.
#[derive(Debug, Clone, Copy)]
pub struct ImageTextEncoderFp8CacheRequirements {
    pub converted_cache_bytes: u64,
    pub maximum_conversion_group_bytes: u64,
}

impl super::QwenImage21Assets {
    /// Complete fixed-adapter DiT retention, including BF16 projections that
    /// are not quantized. No payload scan, Host claim or device work occurs.
    pub fn lora_host_requirements(
        &self,
        precision: ImageCudaPrecisionPolicy,
        host_limits: HostMemoryLimits,
    ) -> anyhow::Result<Option<ImageLoraHostRequirements>> {
        use ferrum_models::qwen_image21::{
            QwenImage21T2iLayout, QwenImage21TransformerConfig, QwenImage21TransformerFamily,
        };
        use ferrum_runtime::{
            HostMemoryAccount, ModelFamilyRegistration, NumericalProfileId, TypedFamilyRegistration,
        };
        let Some(adapter) = &self.lora else {
            return Ok(None);
        };
        let fp8 = precision.stage_uses_fp8(ImageStage::Denoiser);
        let config = QwenImage21TransformerConfig::new(
            self.transformer.clone(),
            QwenImage21T2iLayout::new(1, 2, 2, vec![1])?,
        )?;
        let profile = if fp8 {
            QwenImage21TransformerFamily::CUDA_FP8_PROFILE_ID
        } else {
            QwenImage21TransformerFamily::CUDA_PROFILE_ID
        };
        let family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
            .prepare_with_profile(
                &serde_json::to_value(&config)?,
                &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
            )?;
        let groups = if fp8 {
            QwenImage21TransformerFamily::fp8_weight_groups(&config)?
        } else {
            Vec::new()
        };
        Ok(Some(
            ferrum_quantization::Bf16WholeTensorFp8Materializer::composed_host_requirements(
                &HostMemoryAccount::new(host_limits),
                &family,
                &groups,
                adapter,
            )?,
        ))
    }

    pub fn text_encoder_fp8_host_cache_requirements(
        &self,
        host_limits: HostMemoryLimits,
    ) -> anyhow::Result<ImageTextEncoderFp8CacheRequirements> {
        text_encoder_cache_requirements(&self.text, host_limits)
    }
}

fn text_encoder_cache_requirements(
    architecture: &ferrum_models::qwen3_vl_text::Qwen3VLTextArchitecture,
    host_limits: HostMemoryLimits,
) -> anyhow::Result<ImageTextEncoderFp8CacheRequirements> {
    use ferrum_models::qwen3_vl_text::{
        Qwen3VLTextConfig, Qwen3VLTextEncoderFamily, Qwen3VLTextKeyRun, Qwen3VLTextLayout,
    };
    use ferrum_runtime::{
        HostMemoryAccount, ModelFamilyRegistration, NumericalProfileId, TypedFamilyRegistration,
    };
    let text = Qwen3VLTextConfig::new(
        architecture.clone(),
        Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
    )?;
    let family = TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
        .prepare_with_profile(
            &serde_json::to_value(&text)?,
            &NumericalProfileId::new(Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID)
                .map_err(anyhow::Error::msg)?,
        )?;
    // CUDA creates this same byte-aligned account. Conversion metadata is
    // independent of request batch/text shape and never reads source payloads.
    let host = HostMemoryAccount::new(host_limits);
    let materializer = super::precision::materializer(
        super::precision::TEXT_MATERIALIZER,
        &host,
        Qwen3VLTextEncoderFamily::fp8_weight_groups(&text)?,
    )?;
    Ok(ImageTextEncoderFp8CacheRequirements {
        converted_cache_bytes: materializer.converted_cache_bytes(&family)?,
        maximum_conversion_group_bytes: materializer.maximum_group_accounted_bytes(&family)?,
    })
}

/// Only stages with an FP8 weight materializer accept a converted-weight cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageFp8HostCacheBudgets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_encoder: Option<NonZeroU64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denoiser: Option<NonZeroU64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageFp8HostCachePolicy {
    #[default]
    Off,
    StageBudgets {
        maximum_bytes_by_stage: ImageFp8HostCacheBudgets,
    },
}

impl ImageFp8HostCachePolicy {
    pub fn maximum_bytes(self, stage: ImageStage) -> Option<NonZeroU64> {
        match (self, stage) {
            (
                Self::StageBudgets {
                    maximum_bytes_by_stage,
                },
                ImageStage::TextEncoder,
            ) => maximum_bytes_by_stage.text_encoder,
            (
                Self::StageBudgets {
                    maximum_bytes_by_stage,
                },
                ImageStage::Denoiser,
            ) => maximum_bytes_by_stage.denoiser,
            _ => None,
        }
    }

    /// Pure product validation; exact conversion-group demand is checked again
    /// from prepared family metadata before creating the CUDA runtime.
    pub fn validate(
        self,
        precision: ImageCudaPrecisionPolicy,
        host: HostMemoryLimits,
    ) -> anyhow::Result<()> {
        if self == Self::Off {
            return Ok(());
        }
        for stage in [ImageStage::TextEncoder, ImageStage::Denoiser] {
            ensure!(
                self.maximum_bytes(stage).is_none() || precision.stage_uses_fp8(stage),
                "{stage:?} converted FP8 Host caching requires an FP8 materializer for that stage"
            );
        }
        ensure!(
            self.retained_ceiling()? > 0,
            "FP8 Host cache must select a stage"
        );
        self.validate_peak(host, 0)
    }

    fn retained_ceiling(self) -> anyhow::Result<u64> {
        [ImageStage::TextEncoder, ImageStage::Denoiser]
            .into_iter()
            .try_fold(0_u64, |total, stage| {
                total
                    .checked_add(self.maximum_bytes(stage).map_or(0, NonZeroU64::get))
                    .context("FP8 Host cache stage ceilings overflow")
            })
    }

    #[cfg(any(test, feature = "cuda"))]
    pub(super) fn validate_stage(
        self,
        stage: ImageStage,
        converted_bytes: u64,
    ) -> anyhow::Result<()> {
        if let Some(capacity) = self.maximum_bytes(stage) {
            ensure!(
                converted_bytes <= capacity.get(),
                "{stage:?} FP8 Host cache needs {converted_bytes} bytes to retain all converted groups, exceeding configured {}",
                capacity.get()
            );
        }
        Ok(())
    }

    pub(super) fn validate_peak(
        self,
        host: HostMemoryLimits,
        maximum_active_group_bytes: u64,
    ) -> anyhow::Result<()> {
        self.validate_peak_with_composition(host, maximum_active_group_bytes, 0)
    }

    /// LoRA's complete retained cache and mapped files coexist with configured
    /// FP8 caches; they are mandatory even when the legacy FP8 policy is Off.
    pub(super) fn validate_peak_with_composition(
        self,
        host: HostMemoryLimits,
        maximum_active_group_bytes: u64,
        composition_retained_bytes: u64,
    ) -> anyhow::Result<()> {
        if self == Self::Off && composition_retained_bytes == 0 {
            return Ok(());
        }
        let backing = self
            .retained_ceiling()?
            .checked_add(composition_retained_bytes)
            .context("composed Host retained bytes overflow")?
            .checked_add(maximum_active_group_bytes)
            .context("FP8 Host retained plus active conversion bytes overflow")?;
        ensure!(
            backing <= host.class_bytes(HostMemoryClass::WeightBacking) as u64,
            "FP8 Host caching needs {backing} WeightBacking bytes including the active conversion group"
        );
        let total = [
            HostMemoryClass::InputPreprocess,
            HostMemoryClass::TransferStaging,
            HostMemoryClass::Output,
        ]
        .into_iter()
        .try_fold(backing, |bytes, class| {
            bytes
                .checked_add(host.class_bytes(class) as u64)
                .context("FP8 Host coexistence bytes overflow")
        })?;
        ensure!(
            total <= host.total_bytes() as u64,
            "FP8 Host caching needs {total} total Host bytes including input, transfer and output allowances"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::HostMemoryClassLimits;

    #[test]
    fn text_cache_demand_comes_from_complete_family_metadata_without_payloads() {
        use ferrum_models::qwen3_vl_text::Qwen3VLTextArchitecture;
        let architecture = |layers| {
            Qwen3VLTextArchitecture::from_hf_value(&serde_json::json!({
                "model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],
                "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16",
                    "hidden_act":"silu","attention_bias":false,"attention_dropout":0.0,
                    "num_hidden_layers":layers,"hidden_size":8,"intermediate_size":16,
                    "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,
                    "vocab_size":32,"max_position_embeddings":1024,"rms_norm_eps":0.000001,
                    "rope_theta":100,"rope_scaling":{"rope_type":"default",
                        "mrope_interleaved":true,"mrope_section":[2,1,1]}}
            }))
            .unwrap()
        };
        // Even an account far too small to materialize a group can inspect it.
        let one = text_encoder_cache_requirements(&architecture(1), host(1024, 1)).unwrap();
        let two = text_encoder_cache_requirements(&architecture(2), host(1024, 1)).unwrap();
        assert!(one.converted_cache_bytes > 0);
        assert_eq!(two.converted_cache_bytes, one.converted_cache_bytes * 2);
        assert_eq!(
            two.maximum_conversion_group_bytes,
            one.maximum_conversion_group_bytes
        );
        assert!(one.maximum_conversion_group_bytes > 1);
    }

    fn policy() -> ImageFp8HostCachePolicy {
        ImageFp8HostCachePolicy::StageBudgets {
            maximum_bytes_by_stage: ImageFp8HostCacheBudgets {
                text_encoder: NonZeroU64::new(400),
                denoiser: NonZeroU64::new(200),
            },
        }
    }
    fn host(total: usize, backing: usize) -> HostMemoryLimits {
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64,
                transfer_staging_bytes: 32,
                output_bytes: 128,
                weight_backing_bytes: backing,
            },
        )
        .unwrap()
    }

    #[test]
    fn fp8_host_cache_defaults_and_wire_require_supported_positive_stages() {
        assert_eq!(
            ImageFp8HostCachePolicy::default(),
            ImageFp8HostCachePolicy::Off
        );
        let wire = serde_json::json!({"stage_budgets":{"maximum_bytes_by_stage":{
            "text_encoder":400,"denoiser":200
        }}});
        assert_eq!(serde_json::to_value(policy()).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<ImageFp8HostCachePolicy>(wire).unwrap(),
            policy()
        );
        for stages in [
            serde_json::json!({"text_encoder":0}),
            serde_json::json!({"denoiser":-1}),
            serde_json::json!({"decoder":400}),
            serde_json::json!({"vision_encoder":400}),
            serde_json::json!({"noise":400}),
        ] {
            assert!(
                serde_json::from_value::<ImageFp8HostCachePolicy>(serde_json::json!({
                    "stage_budgets":{"maximum_bytes_by_stage":stages}
                }))
                .is_err()
            );
        }
        assert!(policy()
            .validate(ImageCudaPrecisionPolicy::Bf16, host(1024, 800))
            .is_err());
        assert!(ImageFp8HostCachePolicy::StageBudgets {
            maximum_bytes_by_stage: ImageFp8HostCacheBudgets::default(),
        }
        .validate(
            ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1,
            host(1024, 800)
        )
        .is_err());
        ImageFp8HostCachePolicy::Off
            .validate(ImageCudaPrecisionPolicy::Bf16, host(224, 0))
            .unwrap();
    }

    #[test]
    fn fp8_host_cache_checks_retained_active_and_other_host_classes_at_exact_boundaries() {
        let policy = policy();
        policy.validate_stage(ImageStage::TextEncoder, 400).unwrap();
        assert!(policy.validate_stage(ImageStage::TextEncoder, 401).is_err());
        policy.validate_peak(host(924, 700), 100).unwrap();
        assert!(policy.validate_peak(host(924, 699), 100).is_err());
        assert!(policy.validate_peak(host(923, 700), 100).is_err());
        assert!(policy.validate_peak(host(1024, 800), u64::MAX).is_err());
        let overflowing = ImageFp8HostCachePolicy::StageBudgets {
            maximum_bytes_by_stage: ImageFp8HostCacheBudgets {
                text_encoder: NonZeroU64::new(u64::MAX),
                denoiser: NonZeroU64::new(1),
            },
        };
        assert!(overflowing
            .validate(
                ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1,
                host(1024, 800)
            )
            .is_err());
    }

    #[test]
    fn native_denoiser_allows_only_the_text_converted_weight_cache() {
        use super::super::{ImageComponentPrecision, ImageDenoiserPrecision, ImageTextPrecision};
        let precision = ImageCudaPrecisionPolicy::Components(ImageComponentPrecision {
            text_encoder: ImageTextPrecision::Fp8LinearExperimentalV1,
            denoiser: ImageDenoiserPrecision::NativeSource,
        });
        let text_only = ImageFp8HostCachePolicy::StageBudgets {
            maximum_bytes_by_stage: ImageFp8HostCacheBudgets {
                text_encoder: NonZeroU64::new(400),
                denoiser: None,
            },
        };
        text_only.validate(precision, host(1024, 800)).unwrap();
        assert!(policy().validate(precision, host(1024, 800)).is_err());
    }
}
