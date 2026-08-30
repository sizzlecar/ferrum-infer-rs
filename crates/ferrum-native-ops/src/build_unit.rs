//! Typed coverage contract between Cargo feature units and native artifacts.

use std::collections::BTreeSet;

use ferrum_types::NativeOperatorBackend;
use thiserror::Error;

use crate::ResolvedNativeOperatorArtifactSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CudaNativeBuildUnit {
    Marlin,
    VllmMarlin,
    VllmMoeMarlin,
    VllmPagedAttentionV2,
}

impl CudaNativeBuildUnit {
    pub const ALL: [Self; 4] = [
        Self::Marlin,
        Self::VllmMarlin,
        Self::VllmMoeMarlin,
        Self::VllmPagedAttentionV2,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Marlin => "marlin",
            Self::VllmMarlin => "vllm_marlin",
            Self::VllmMoeMarlin => "vllm_moe_marlin",
            Self::VllmPagedAttentionV2 => "vllm_paged_attention_v2",
        }
    }

    pub const fn artifact_operator(self) -> &'static str {
        match self {
            Self::Marlin => "ferrum.cuda.marlin",
            Self::VllmMarlin => "ferrum.cuda.vllm_marlin",
            Self::VllmMoeMarlin => "ferrum.cuda.vllm_moe_marlin",
            Self::VllmPagedAttentionV2 => "ferrum.cuda.vllm_paged_attention_v2",
        }
    }

    pub fn from_artifact_operator(operator: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|unit| unit.artifact_operator() == operator)
    }

    pub const fn required_exports(self) -> &'static [&'static str] {
        match self {
            Self::Marlin => &["marlin_cuda", "marlin_cuda_moe"],
            Self::VllmMarlin => &[
                "ferrum_block_fp8_group128_repack",
                "ferrum_block_fp8_group128_scales",
                "ferrum_marlin_mm",
                "ferrum_marlin_mm_f16_u4b8",
                "ferrum_vllm_gptq_marlin_repack",
            ],
            Self::VllmMoeMarlin => &[
                "ferrum_vllm_marlin_moe_clear_profile_config",
                "ferrum_vllm_marlin_moe_f16",
                "ferrum_vllm_marlin_moe_fp8_f16",
                "ferrum_vllm_marlin_moe_set_profile_config",
            ],
            Self::VllmPagedAttentionV2 => &[
                "ferrum_vllm_paged_attention_v1_f16_h128_b16",
                "ferrum_vllm_paged_attention_v1_f16_h256_b16",
                "ferrum_vllm_paged_attention_v2_f16_h128_b16",
                "ferrum_vllm_paged_attention_v2_f16_h256_b16",
                "ferrum_vnext_vllm_paged_attention_v1_f16_h128_b16_addressed",
                "ferrum_vnext_vllm_paged_attention_v1_f16_h256_b16_addressed",
                "ferrum_vnext_vllm_paged_attention_v2_f16_h128_b16_addressed",
                "ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_addressed",
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCudaNativeBuildCoverage {
    units: BTreeSet<CudaNativeBuildUnit>,
}

impl ResolvedCudaNativeBuildCoverage {
    pub fn resolve(
        artifacts: &ResolvedNativeOperatorArtifactSet,
        required: impl IntoIterator<Item = CudaNativeBuildUnit>,
    ) -> Result<Self, CudaNativeBuildCoverageError> {
        let views = artifacts
            .artifacts
            .iter()
            .map(|artifact| ArtifactView {
                operator: artifact.resolved.manifest.operator.as_str(),
                backend: artifact.resolved.manifest.backend,
                exports: artifact.resolved.manifest.exports.as_slice(),
            })
            .collect::<Vec<_>>();
        resolve_views(&views, required)
    }

    pub fn contains(&self, unit: CudaNativeBuildUnit) -> bool {
        self.units.contains(&unit)
    }

    pub fn iter(&self) -> impl Iterator<Item = CudaNativeBuildUnit> + '_ {
        self.units.iter().copied()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CudaNativeBuildCoverageError {
    #[error(
        "native artifact set does not provide CUDA build unit {unit} (required operator={operator})"
    )]
    MissingArtifact {
        unit: &'static str,
        operator: &'static str,
    },
    #[error("native artifact for CUDA build unit {unit} has backend {actual:?}, expected cuda")]
    WrongBackend {
        unit: &'static str,
        actual: NativeOperatorBackend,
    },
    #[error("native artifact for CUDA build unit {unit} is missing required export {export}")]
    MissingExport {
        unit: &'static str,
        export: &'static str,
    },
}

struct ArtifactView<'a> {
    operator: &'a str,
    backend: NativeOperatorBackend,
    exports: &'a [String],
}

fn resolve_views(
    artifacts: &[ArtifactView<'_>],
    required: impl IntoIterator<Item = CudaNativeBuildUnit>,
) -> Result<ResolvedCudaNativeBuildCoverage, CudaNativeBuildCoverageError> {
    let required = required.into_iter().collect::<BTreeSet<_>>();
    for unit in &required {
        let artifact = artifacts
            .iter()
            .find(|artifact| artifact.operator == unit.artifact_operator())
            .ok_or(CudaNativeBuildCoverageError::MissingArtifact {
                unit: unit.as_str(),
                operator: unit.artifact_operator(),
            })?;
        if artifact.backend != NativeOperatorBackend::Cuda {
            return Err(CudaNativeBuildCoverageError::WrongBackend {
                unit: unit.as_str(),
                actual: artifact.backend,
            });
        }
        for required_export in unit.required_exports() {
            if !artifact
                .exports
                .iter()
                .any(|export| export == required_export)
            {
                return Err(CudaNativeBuildCoverageError::MissingExport {
                    unit: unit.as_str(),
                    export: required_export,
                });
            }
        }
    }
    Ok(ResolvedCudaNativeBuildCoverage { units: required })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exports(unit: CudaNativeBuildUnit) -> Vec<String> {
        unit.required_exports()
            .iter()
            .map(|value| (*value).to_string())
            .collect()
    }

    #[test]
    fn resolves_exact_required_build_units() {
        let operator_exports = CudaNativeBuildUnit::ALL.map(exports);
        let artifacts = CudaNativeBuildUnit::ALL
            .iter()
            .enumerate()
            .map(|(index, unit)| ArtifactView {
                operator: unit.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                exports: &operator_exports[index],
            })
            .collect::<Vec<_>>();

        let coverage =
            resolve_views(&artifacts, CudaNativeBuildUnit::ALL).expect("complete artifact set");

        assert!(CudaNativeBuildUnit::ALL
            .into_iter()
            .all(|unit| coverage.contains(unit)));
    }

    #[test]
    fn rejects_missing_artifact_before_source_build_can_fallback() {
        let error = resolve_views(&[], [CudaNativeBuildUnit::VllmMoeMarlin]).unwrap_err();

        assert_eq!(
            error,
            CudaNativeBuildCoverageError::MissingArtifact {
                unit: "vllm_moe_marlin",
                operator: "ferrum.cuda.vllm_moe_marlin",
            }
        );
    }

    #[test]
    fn rejects_artifact_that_does_not_export_the_linked_rust_abi() {
        let exports = vec!["ferrum_vllm_marlin_moe_f16".to_string()];
        let artifacts = [ArtifactView {
            operator: CudaNativeBuildUnit::VllmMoeMarlin.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            exports: &exports,
        }];

        let error = resolve_views(&artifacts, [CudaNativeBuildUnit::VllmMoeMarlin]).unwrap_err();

        assert_eq!(
            error,
            CudaNativeBuildCoverageError::MissingExport {
                unit: "vllm_moe_marlin",
                export: "ferrum_vllm_marlin_moe_clear_profile_config",
            }
        );
    }

    #[test]
    fn rejects_moe_artifact_without_fp8_entrypoint_before_link() {
        let exports = CudaNativeBuildUnit::VllmMoeMarlin
            .required_exports()
            .iter()
            .copied()
            .filter(|export| *export != "ferrum_vllm_marlin_moe_fp8_f16")
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let artifacts = [ArtifactView {
            operator: CudaNativeBuildUnit::VllmMoeMarlin.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            exports: &exports,
        }];

        let error = resolve_views(&artifacts, [CudaNativeBuildUnit::VllmMoeMarlin]).unwrap_err();

        assert_eq!(
            error,
            CudaNativeBuildCoverageError::MissingExport {
                unit: "vllm_moe_marlin",
                export: "ferrum_vllm_marlin_moe_fp8_f16",
            }
        );
    }
}
