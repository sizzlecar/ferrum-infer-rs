//! Explicit model precision, separate from attention and lifetime policies.
use super::*;
use ferrum_quantization::Bf16WholeTensorFp8Materializer;

/// Storage/compute policy for each weighted stage. Legacy string presets retain
/// their behavior; component selection does not imply a change in residency.
/// VAE, vision and activation boundaries remain BF16 in every current policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageCudaPrecisionPolicy {
    #[default]
    Bf16,
    Fp8LinearExperimentalV1,
    Components(ImageComponentPrecision),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageComponentPrecision {
    pub text_encoder: ImageTextPrecision,
    pub denoiser: ImageDenoiserPrecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTextPrecision {
    Bf16,
    Fp8LinearExperimentalV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDenoiserPrecision {
    Bf16,
    Fp8LinearExperimentalV1,
    /// Preserve the bound archive encodings; native blocks decode into
    /// plan-admitted scratch, with BF16 activation/weight compute boundaries.
    NativeSource,
}
impl ImageCudaPrecisionPolicy {
    /// Canonical CLI spelling, including an explicit object for component policies.
    pub fn as_str(self) -> std::borrow::Cow<'static, str> {
        match self {
            Self::Bf16 => "bf16".into(),
            Self::Fp8LinearExperimentalV1 => "fp8_linear_experimental_v1".into(),
            Self::Components(_) => serde_json::to_string(&self)
                .expect("precision contains only enum and struct values")
                .into(),
        }
    }
    pub const fn text_encoder(self) -> ImageTextPrecision {
        match self {
            Self::Bf16 => ImageTextPrecision::Bf16,
            Self::Fp8LinearExperimentalV1 => ImageTextPrecision::Fp8LinearExperimentalV1,
            Self::Components(components) => components.text_encoder,
        }
    }
    pub const fn denoiser(self) -> ImageDenoiserPrecision {
        match self {
            Self::Bf16 => ImageDenoiserPrecision::Bf16,
            Self::Fp8LinearExperimentalV1 => ImageDenoiserPrecision::Fp8LinearExperimentalV1,
            Self::Components(components) => components.denoiser,
        }
    }
    pub const fn stage_uses_fp8(self, stage: ImageStage) -> bool {
        match stage {
            ImageStage::TextEncoder => matches!(
                self.text_encoder(),
                ImageTextPrecision::Fp8LinearExperimentalV1
            ),
            ImageStage::Denoiser => matches!(
                self.denoiser(),
                ImageDenoiserPrecision::Fp8LinearExperimentalV1
            ),
            _ => false,
        }
    }
    pub const fn uses_fp8(self) -> bool {
        self.stage_uses_fp8(ImageStage::TextEncoder) || self.stage_uses_fp8(ImageStage::Denoiser)
    }
    /// Reject a conversion policy incompatible with the declared source before
    /// creating a device runtime. Legacy BF16 still means BF16 compute on native
    /// source storage when that inventory is explicitly supplied.
    pub fn validate_denoiser_source(self, native: bool) -> anyhow::Result<()> {
        ensure!(
            !native || self.denoiser() != ImageDenoiserPrecision::Fp8LinearExperimentalV1,
            "native denoiser encodings cannot be replaced by an FP8 weight materializer"
        );
        ensure!(
            native || self.denoiser() != ImageDenoiserPrecision::NativeSource,
            "native_source denoiser precision requires a native encoding inventory"
        );
        Ok(())
    }
    /// Declared process-retained CUTLASS module allowance, not measured VRAM.
    pub const fn additional_opaque_bytes(self) -> u64 {
        if self.uses_fp8() {
            128 * 1024 * 1024
        } else {
            0
        }
    }
}
impl std::str::FromStr for ImageCudaPrecisionPolicy {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8_linear_experimental_v1" => Ok(Self::Fp8LinearExperimentalV1),
            value if value.trim_start().starts_with('{') =>
                serde_json::from_str(value).map_err(|error| format!("invalid component precision: {error}")),
            _ => Err("image precision must be bf16, fp8_linear_experimental_v1, or a components JSON object".into()),
        }
    }
}
pub(super) const TEXT_MATERIALIZER: &str = "weight-materializer.cpu.qwen_image21.text_fp8.v1";
pub(super) const DIT_MATERIALIZER: &str = "weight-materializer.cpu.qwen_image21.dit_fp8.v1";

pub(super) fn materializer(
    id: &str,
    host: &HostMemoryAccount,
    groups: Vec<Vec<WeightId>>,
) -> Result<Bf16WholeTensorFp8Materializer, VNextError> {
    Bf16WholeTensorFp8Materializer::new_with_id(
        WeightMaterializerId::new(id)?,
        host.clone(),
        groups,
    )
}

pub(super) fn select(
    family: &PreparedModelFamily,
    options: &mut ProgramPlanCompileOptions,
    materializer: &Bf16WholeTensorFp8Materializer,
    host: &HostMemoryAccount,
) -> anyhow::Result<()> {
    validate_backing(family, materializer, host)?;
    options.require_weight_materializer_selection(materializer.selection(family)?);
    Ok(())
}

pub(super) fn validate_backing(
    family: &PreparedModelFamily,
    materializer: &Bf16WholeTensorFp8Materializer,
    host: &HostMemoryAccount,
) -> anyhow::Result<()> {
    let peak = materializer.maximum_group_backing_bytes(family)?;
    ensure!(
        peak <= host.limits().class_bytes(HostMemoryClass::WeightBacking) as u64,
        "FP8 conversion group needs {peak} WeightBacking bytes before transfer staging"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn precision_is_explicit_and_wire_roundtrips_without_omni_equivalence_alias() {
        assert_eq!(
            ImageCudaPrecisionPolicy::default(),
            ImageCudaPrecisionPolicy::Bf16
        );
        for policy in [
            ImageCudaPrecisionPolicy::Bf16,
            ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1,
        ] {
            assert_eq!(
                policy.as_str().parse::<ImageCudaPrecisionPolicy>().unwrap(),
                policy
            );
            assert_eq!(
                serde_json::from_value::<ImageCudaPrecisionPolicy>(
                    serde_json::to_value(policy).unwrap()
                )
                .unwrap(),
                policy
            );
        }
        for value in ["fp8", "omni", "auto", "BF16"] {
            assert!(value.parse::<ImageCudaPrecisionPolicy>().is_err());
        }
    }

    #[test]
    fn component_precision_roundtrips_and_rejects_incompatible_source_materialization() {
        let mixed = ImageCudaPrecisionPolicy::Components(ImageComponentPrecision {
            text_encoder: ImageTextPrecision::Fp8LinearExperimentalV1,
            denoiser: ImageDenoiserPrecision::NativeSource,
        });
        let wire = serde_json::json!({"components": {
            "text_encoder": "fp8_linear_experimental_v1", "denoiser": "native_source"
        }});
        assert_eq!(serde_json::to_value(mixed).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<ImageCudaPrecisionPolicy>(wire).unwrap(),
            mixed
        );
        assert_eq!(
            mixed.as_str().parse::<ImageCudaPrecisionPolicy>().unwrap(),
            mixed
        );
        assert!(mixed.stage_uses_fp8(ImageStage::TextEncoder));
        for stage in [ImageStage::Denoiser, ImageStage::Decoder] {
            assert!(!mixed.stage_uses_fp8(stage));
        }
        mixed.validate_denoiser_source(true).unwrap();
        assert!(mixed.validate_denoiser_source(false).is_err());
        assert!(ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1
            .validate_denoiser_source(true)
            .is_err());
        ImageCudaPrecisionPolicy::Bf16
            .validate_denoiser_source(true)
            .unwrap();
        for invalid in [
            serde_json::json!({"text_encoder":"native_source", "denoiser":"bf16"}),
            serde_json::json!({"text_encoder":"bf16"}),
            serde_json::json!({"text_encoder":"bf16", "denoiser":"native_source", "decoder":"fp8"}),
        ] {
            assert!(serde_json::from_value::<ImageCudaPrecisionPolicy>(
                serde_json::json!({"components": invalid})
            )
            .is_err());
        }
    }
}
