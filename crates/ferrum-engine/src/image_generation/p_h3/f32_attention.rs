//! Explicit numerical selection shared by image run and serve. Availability
//! means an advertised native capability, not a model-name guess.
#[cfg(any(feature = "cuda", test))]
use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageF32AttentionPolicy {
    /// Use fused FastF32 when the backend supplies it; otherwise retain the
    /// original strict F32 operation and family profiles.
    #[default]
    Auto,
    Sequential,
    /// Require fused capability; fail before execution if unavailable.
    CutlassFused,
}

impl PH3ImageF32AttentionPolicy {
    /// Separate native module reservation, charged inside the device ceiling.
    /// This is a declared allowance, not a claim about measured peak memory.
    pub fn additional_opaque_bytes(self) -> u64 {
        if cfg!(feature = "cuda-cutlass-fused-f32-attention") && self != Self::Sequential {
            64 << 20
        } else {
            0
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn resolve(self, available: bool) -> Result<bool> {
        match self {
            Self::Sequential => Ok(false),
            Self::Auto => Ok(available),
            Self::CutlassFused => {
                ensure!(
                    available,
                    "fused F32 attention requires its native CUDA capability"
                );
                Ok(true)
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl super::PH3ImageAssets {
    pub(super) fn configure_f32_attention(&mut self, fused: bool) -> Result<()> {
        use ferrum_models::qwen3_vl_vision::{
            comfy::Qwen3VLVisionComfyFamily, hf::Qwen3VLVisionHfFamily,
        };
        self.main_vision_profile = if fused {
            Qwen3VLVisionComfyFamily::CUDA_FUSED_F32_PROFILE_ID
        } else {
            Qwen3VLVisionComfyFamily::CUDA_PROFILE_ID
        };
        if let Some(expansion) = &mut self.expansion {
            let profile = if fused {
                Qwen3VLVisionHfFamily::CUDA_FUSED_F32_PROFILE_ID
            } else {
                Qwen3VLVisionHfFamily::CUDA_PROFILE_ID
            };
            if expansion
                .assets
                .vision_family()
                .is_some_and(|family| family.numerical_profile().id.as_str() != profile)
            {
                let assets = std::sync::Arc::get_mut(&mut expansion.assets).ok_or_else(|| {
                    anyhow::anyhow!("select F32 attention before sharing expansion assets")
                })?;
                assets.configure_visual_profile(
                    ferrum_runtime::NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_requires_capability_and_respects_strict_arithmetic() {
        assert!(!PH3ImageF32AttentionPolicy::Auto.resolve(false).unwrap());
        assert!(PH3ImageF32AttentionPolicy::Auto.resolve(true).unwrap());
        assert!(!PH3ImageF32AttentionPolicy::Sequential
            .resolve(true)
            .unwrap());
        assert!(PH3ImageF32AttentionPolicy::CutlassFused
            .resolve(false)
            .is_err());
        assert!(PH3ImageF32AttentionPolicy::CutlassFused
            .resolve(true)
            .unwrap());
    }
}
