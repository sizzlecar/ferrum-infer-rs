//! Explicit physical weight artifacts, independent of device allocation.
use super::*;
use ferrum_quantization::{
    bf16_whole_tensor_fp8::ExactBf16TensorSource, Bf16WholeTensorFp8Materializer,
    PreparedFp8Archive, PreparedFp8ExportReceipt, PreparedFp8TensorArchive,
};
use std::path::Path;

impl QwenImage21Assets {
    fn fp8_family(
        &self,
        stage: ImageStage,
        profile: &NumericalProfileId,
    ) -> anyhow::Result<(PreparedModelFamily, Vec<Vec<WeightId>>)> {
        match stage {
            ImageStage::TextEncoder => {
                ensure!(
                    Qwen3VLTextEncoderFamily::is_fp8_profile(profile.as_str()),
                    "prepared text weights require a declared FP8 text profile"
                );
                let config = Qwen3VLTextConfig::new(
                    self.text.clone(),
                    Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
                )?;
                Ok((
                    TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
                        .prepare_with_profile(&serde_json::to_value(&config)?, profile)?,
                    Qwen3VLTextEncoderFamily::fp8_weight_groups(&config)?,
                ))
            }
            ImageStage::Denoiser => {
                ensure!(
                    QwenImage21TransformerFamily::is_fp8_profile(profile.as_str()),
                    "prepared denoiser weights require a declared FP8 denoiser profile"
                );
                ensure!(
                    self.transformer_encodings.is_empty(),
                    "native denoiser storage cannot use prepared BF16-to-FP8 weights"
                );
                let config = QwenImage21TransformerConfig::new(
                    self.transformer.clone(),
                    QwenImage21T2iLayout::new(1, 2, 2, vec![1])?,
                )?;
                Ok((
                    TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
                        .prepare_with_profile(&serde_json::to_value(&config)?, profile)?,
                    QwenImage21TransformerFamily::fp8_weight_groups(&config)?,
                ))
            }
            _ => anyhow::bail!("this stage has no declared FP8 weight materializer"),
        }
    }

    /// Select a verified prepared artifact before constructing any device.
    /// The original archive still supplies embeddings and optional vision.
    pub fn with_prepared_fp8(
        mut self,
        stage: ImageStage,
        profile: &NumericalProfileId,
        archive: Arc<PreparedFp8Archive>,
    ) -> anyhow::Result<Self> {
        ensure!(
            self.prepared_fp8(stage).is_none(),
            "prepared stage already selected"
        );
        let (family, groups) = self.fp8_family(stage, profile)?;
        archive.validate_recipe(&family, &groups)?;
        let (source, selected, receipt_key) = match stage {
            ImageStage::TextEncoder => (
                &mut self.sources.text_encoder,
                &mut self.prepared_text_encoder,
                "prepared_text_encoder_manifest",
            ),
            ImageStage::Denoiser => (
                &mut self.sources.transformer,
                &mut self.prepared_denoiser,
                "prepared_denoiser_manifest",
            ),
            _ => unreachable!("FP8 family validation restricted the stage"),
        };
        *source = Arc::new(PreparedFp8TensorArchive::new(
            source.clone(),
            archive.clone(),
        ));
        self.receipt
            .metadata_sha256
            .insert(receipt_key.into(), archive.manifest_sha256().into());
        self.receipt.payload_verification =
            "selected prepared artifacts hash-verified; original sources header-validated";
        *selected = Some(archive);
        Ok(self)
    }

    pub(in crate::image_generation) fn prepared_fp8(
        &self,
        stage: ImageStage,
    ) -> Option<&Arc<PreparedFp8Archive>> {
        match stage {
            ImageStage::TextEncoder => self.prepared_text_encoder.as_ref(),
            ImageStage::Denoiser => self.prepared_denoiser.as_ref(),
            _ => None,
        }
    }

    pub(in crate::image_generation) fn bind_prepared_fp8(
        &self,
        stage: ImageStage,
        family: &PreparedModelFamily,
        materializer: Bf16WholeTensorFp8Materializer,
    ) -> anyhow::Result<Bf16WholeTensorFp8Materializer> {
        match self.prepared_fp8(stage) {
            Some(archive) => Ok(materializer.with_prepared_archive(family, archive.clone())?),
            None => Ok(materializer),
        }
    }

    /// CPU-only export. Each conversion group uses the supplied Host account;
    /// no device, whole-stage host cache or unbounded payload Vec is created.
    pub fn export_prepared_fp8(
        &self,
        stage: ImageStage,
        profile: &NumericalProfileId,
        host: &HostMemoryAccount,
        destination: &Path,
    ) -> anyhow::Result<PreparedFp8ExportReceipt> {
        ensure!(
            self.prepared_fp8(stage).is_none(),
            "export requires the original weight source"
        );
        let (family, groups) = self.fp8_family(stage, profile)?;
        let (id, source) = match stage {
            ImageStage::TextEncoder => (precision::TEXT_MATERIALIZER, &self.sources.text_encoder),
            ImageStage::Denoiser => (precision::DIT_MATERIALIZER, &self.sources.transformer),
            _ => unreachable!("FP8 family validation restricted the stage"),
        };
        let materializer = precision::materializer(id, host, groups)?;
        precision::validate_backing(&family, &materializer, host)?;
        let source = ExactBf16TensorSource::new(source.as_ref());
        Ok(materializer.export_prepared(&family, &source, destination)?)
    }
}
