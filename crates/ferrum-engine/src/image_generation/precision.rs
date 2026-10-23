//! Explicit model precision, separate from attention and lifetime policies.
use super::*;
use ferrum_quantization::Bf16WholeTensorFp8Materializer;

/// FP8 selects text Linear weights independently and all DiT block Linear
/// weights (Q/K/V share a scale). VAE and non-block DiT weights remain BF16.
/// This is an experimental scope, not proof that an Omni ignore rule matched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageCudaPrecisionPolicy {
    #[default]
    Bf16,
    Fp8LinearExperimentalV1,
}
impl ImageCudaPrecisionPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8LinearExperimentalV1 => "fp8_linear_experimental_v1",
        }
    }
    /// Declared process-retained CUTLASS module allowance, not measured VRAM.
    pub const fn additional_opaque_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 0,
            Self::Fp8LinearExperimentalV1 => 128 * 1024 * 1024,
        }
    }
}
impl std::str::FromStr for ImageCudaPrecisionPolicy {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8_linear_experimental_v1" => Ok(Self::Fp8LinearExperimentalV1),
            _ => Err("image precision must be bf16 or fp8_linear_experimental_v1".into()),
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
}
