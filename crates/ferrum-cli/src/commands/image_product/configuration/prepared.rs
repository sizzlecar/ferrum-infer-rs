use super::*;
use ferrum_engine::image_generation::ImageStage;
use ferrum_models::{
    qwen3_vl_text::Qwen3VLTextEncoderFamily, qwen_image21::QwenImage21TransformerFamily,
};
use ferrum_runtime::NumericalProfileId;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedImageWeights {
    pub text_encoder: Option<PreparedWeightPath>,
    pub denoiser: Option<PreparedWeightPath>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedWeightPath {
    pub manifest: PathBuf,
    pub sha256: String,
}

impl ImageBackendConfig {
    pub fn prepared_fp8_profile(&self, stage: ImageStage) -> Result<NumericalProfileId> {
        let precision = match self {
            Self::Cuda(c) => c.precision_policy,
            Self::Metal(c) => c.precision_policy,
        };
        ensure!(
            precision.stage_uses_fp8(stage),
            "prepared weights require FP8 precision for the selected stage"
        );
        let profile = match (self, stage) {
            (Self::Cuda(_), ImageStage::TextEncoder) => {
                Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID
            }
            (Self::Metal(_), ImageStage::TextEncoder) => {
                Qwen3VLTextEncoderFamily::METAL_FP8_PROFILE_ID
            }
            (Self::Cuda(_), ImageStage::Denoiser) => {
                QwenImage21TransformerFamily::CUDA_FP8_PROFILE_ID
            }
            (Self::Metal(_), ImageStage::Denoiser) => {
                QwenImage21TransformerFamily::METAL_FP8_PROFILE_ID
            }
            _ => anyhow::bail!("selected stage has no FP8 preparation contract"),
        };
        NumericalProfileId::new(profile).map_err(anyhow::Error::msg)
    }
}

impl PreparedImageWeights {
    pub fn entries(&self) -> impl Iterator<Item = (ImageStage, &PreparedWeightPath)> {
        [
            (ImageStage::TextEncoder, self.text_encoder.as_ref()),
            (ImageStage::Denoiser, self.denoiser.as_ref()),
        ]
        .into_iter()
        .filter_map(|(stage, path)| path.map(|path| (stage, path)))
    }

    pub(super) fn validate(&self, backend: &ImageBackendConfig) -> Result<()> {
        for (stage, artifact) in self.entries() {
            backend.prepared_fp8_profile(stage)?;
            ensure!(
                !artifact.manifest.as_os_str().is_empty(),
                "prepared manifest path is empty"
            );
            ensure!(
                artifact.sha256.len() == 64
                    && artifact
                        .sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "prepared manifest sha256 must contain exactly 64 lowercase hexadecimal digits"
            );
            if let ImageBackendConfig::Cuda(cuda) = backend {
                ensure!(cuda.fp8_host_cache.maximum_bytes(stage).is_none(),
                    "prepared weights and a retained FP8 Host cache are mutually exclusive for the same stage");
            }
        }
        Ok(())
    }

    pub(super) fn resolve_paths(&mut self, base: &Path) {
        for artifact in [&mut self.text_encoder, &mut self.denoiser]
            .into_iter()
            .flatten()
        {
            if artifact.manifest.is_relative() {
                artifact.manifest = base.join(&artifact.manifest);
            }
        }
    }
}
