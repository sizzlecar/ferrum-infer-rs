//! Typed coverage contract between Cargo feature units and native artifacts.

use std::collections::BTreeSet;

use ferrum_types::NativeOperatorBackend;
use thiserror::Error;

use crate::ResolvedNativeOperatorArtifactSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CudaNativeBuildUnit {
    CutlassFp8ScaledMm,
    Marlin,
    VllmMarlin,
    VllmMoeMarlin,
    VllmPagedAttentionV2,
    VllmIndexedPagedAttentionV2,
    UpstreamLinear,
    UpstreamExtraLinear,
    UpstreamQ6F32Linear,
}

impl CudaNativeBuildUnit {
    pub const ALL: [Self; 9] = [
        Self::CutlassFp8ScaledMm,
        Self::Marlin,
        Self::VllmMarlin,
        Self::VllmMoeMarlin,
        Self::VllmPagedAttentionV2,
        Self::VllmIndexedPagedAttentionV2,
        Self::UpstreamLinear,
        Self::UpstreamExtraLinear,
        Self::UpstreamQ6F32Linear,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CutlassFp8ScaledMm => "cutlass_fp8_scaled_mm",
            Self::Marlin => "marlin",
            Self::VllmMarlin => "vllm_marlin",
            Self::VllmMoeMarlin => "vllm_moe_marlin",
            Self::VllmPagedAttentionV2 => "vllm_paged_attention_v2",
            Self::VllmIndexedPagedAttentionV2 => "vllm_indexed_paged_attention_v2",
            Self::UpstreamLinear => "upstream_linear",
            Self::UpstreamExtraLinear => "upstream_extra_linear",
            Self::UpstreamQ6F32Linear => "upstream_q6_f32_linear",
        }
    }

    pub const fn artifact_operator(self) -> &'static str {
        match self {
            Self::CutlassFp8ScaledMm => crate::cutlass_fp8::CUTLASS_FP8_OPERATOR,
            Self::Marlin => "ferrum.cuda.marlin",
            Self::VllmMarlin => "ferrum.cuda.vllm_marlin",
            Self::VllmMoeMarlin => "ferrum.cuda.vllm_moe_marlin",
            Self::VllmPagedAttentionV2 | Self::VllmIndexedPagedAttentionV2 => {
                "ferrum.cuda.vllm_paged_attention_v2"
            }
            Self::UpstreamLinear => "ferrum.cuda.upstream_linear",
            Self::UpstreamExtraLinear => "ferrum.cuda.upstream_extra_linear",
            Self::UpstreamQ6F32Linear => "ferrum.cuda.upstream-q6-f32-linear",
        }
    }

    /// Return the baseline build unit for an operator. Optional extensions of
    /// that same artifact are requested explicitly by the active Cargo graph.
    pub fn from_artifact_operator(operator: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|unit| unit.artifact_operator() == operator)
    }

    /// Package revisions can retire ambient policy or exports without changing
    /// the retained mathematical entrypoint signatures.
    pub const fn required_operator_abi_version(self) -> Option<&'static str> {
        match self {
            Self::CutlassFp8ScaledMm => Some("1"),
            Self::Marlin => Some("2"),
            Self::VllmMoeMarlin => Some("3"),
            _ => None,
        }
    }

    pub const fn forbidden_exports(self) -> &'static [&'static str] {
        match self {
            Self::VllmMoeMarlin => &[
                "ferrum_vllm_marlin_moe_clear_profile_config",
                "ferrum_vllm_marlin_moe_set_profile_config",
            ],
            _ => &[],
        }
    }

    /// Validate the declared contract before packaging or accepting an artifact.
    pub fn validate_package_contract(
        self,
        operator_abi_version: &str,
        exports: &[String],
    ) -> Result<(), CudaNativeBuildCoverageError> {
        if let Some(expected) = self.required_operator_abi_version() {
            if operator_abi_version != expected {
                return Err(CudaNativeBuildCoverageError::WrongOperatorAbi {
                    unit: self.as_str(),
                    expected,
                    actual: operator_abi_version.to_owned(),
                });
            }
        }
        self.validate_binary_exports(exports)?;
        for required in self.required_exports() {
            if !exports.iter().any(|export| export == required) {
                return Err(CudaNativeBuildCoverageError::MissingExport {
                    unit: self.as_str(),
                    export: required,
                });
            }
        }
        Ok(())
    }

    /// Inspect actual archive symbols as well as manifest claims: omitting a
    /// retired export from the manifest must not hide a still-linked writer.
    pub fn validate_binary_exports(
        self,
        symbols: &[String],
    ) -> Result<(), CudaNativeBuildCoverageError> {
        for forbidden in self.forbidden_exports() {
            if symbols.iter().any(|symbol| symbol == forbidden) {
                return Err(CudaNativeBuildCoverageError::ForbiddenExport {
                    unit: self.as_str(),
                    export: forbidden,
                });
            }
        }
        Ok(())
    }

    pub const fn required_exports(self) -> &'static [&'static str] {
        match self {
            Self::CutlassFp8ScaledMm => &crate::cutlass_fp8::CUTLASS_FP8_EXPORTS,
            Self::UpstreamQ6F32Linear => &[
                "ferrum_upstream_q6_f32_check_weights_v1",
                "ferrum_upstream_q6_f32_dot_v1",
                "ferrum_upstream_q6_f32_pack_v1",
                "ferrum_upstream_q6_f32_plan_v1",
                "ferrum_upstream_q6_f32_publish_v1",
            ],
            Self::UpstreamLinear => &[
                "ferrum_upstream_mmq_cast_v1",
                "ferrum_upstream_mmq_cast_v2",
                "ferrum_upstream_mmq_check_weights_v2",
                "ferrum_upstream_mmq_dot_v1",
                "ferrum_upstream_mmq_pack_v1",
                "ferrum_upstream_mmq_pack_v2",
                "ferrum_upstream_mmq_plan_v1",
                "ferrum_upstream_mmvq_cast_v1",
                "ferrum_upstream_mmvq_cast_v2",
                "ferrum_upstream_mmvq_check_weights_v2",
                "ferrum_upstream_mmvq_dot_v1",
                "ferrum_upstream_mmvq_pack_v1",
                "ferrum_upstream_mmvq_pack_v2",
                "ferrum_upstream_mmvq_plan_v1",
            ],
            Self::UpstreamExtraLinear => &[
                "ferrum_upstream_extra_mmq_cast_v1",
                "ferrum_upstream_extra_mmq_cast_v2",
                "ferrum_upstream_extra_mmq_check_weights_v2",
                "ferrum_upstream_extra_mmq_dot_v1",
                "ferrum_upstream_extra_mmq_pack_v1",
                "ferrum_upstream_extra_mmq_pack_v2",
                "ferrum_upstream_extra_mmq_plan_v1",
                "ferrum_upstream_extra_mmvq_cast_v1",
                "ferrum_upstream_extra_mmvq_cast_v2",
                "ferrum_upstream_extra_mmvq_check_weights_v2",
                "ferrum_upstream_extra_mmvq_dot_v1",
                "ferrum_upstream_extra_mmvq_pack_v1",
                "ferrum_upstream_extra_mmvq_pack_v2",
                "ferrum_upstream_extra_mmvq_plan_v1",
            ],
            Self::Marlin => &["marlin_cuda", "marlin_cuda_moe"],
            Self::VllmMarlin => &[
                "ferrum_block_fp8_group128_repack",
                "ferrum_block_fp8_group128_scales",
                "ferrum_marlin_mm",
                "ferrum_marlin_mm_f16_u4b8",
                "ferrum_vllm_gptq_marlin_repack",
            ],
            Self::VllmMoeMarlin => &[
                "ferrum_vllm_marlin_moe_f16",
                "ferrum_vllm_marlin_moe_fp8_f16",
                "ferrum_vllm_marlin_moe_mxfp4_bf16",
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
            Self::VllmIndexedPagedAttentionV2 => &[
                "ferrum_vllm_paged_attention_v1_f16_h128_b16",
                "ferrum_vllm_paged_attention_v1_f16_h256_b16",
                "ferrum_vllm_paged_attention_v2_f16_h128_b16",
                "ferrum_vllm_paged_attention_v2_f16_h256_b16",
                "ferrum_vnext_vllm_paged_attention_v1_f16_h128_b16_addressed",
                "ferrum_vnext_vllm_paged_attention_v1_f16_h128_b16_addressed_indexed_v1",
                "ferrum_vnext_vllm_paged_attention_v1_f16_h256_b16_addressed",
                "ferrum_vnext_vllm_paged_attention_v1_f16_h256_b16_addressed_indexed_v1",
                "ferrum_vnext_vllm_paged_attention_v2_f16_h128_b16_addressed",
                "ferrum_vnext_vllm_paged_attention_v2_f16_h128_b16_addressed_indexed_v1",
                "ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_addressed",
                "ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_addressed_indexed_v1",
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
                operator_abi_version: artifact.resolved.manifest.operator_abi_version.as_str(),
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
    #[error(
        "native artifact for CUDA build unit {unit} has operator ABI {actual}, expected {expected}"
    )]
    WrongOperatorAbi {
        unit: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("native artifact for CUDA build unit {unit} retains forbidden export {export}")]
    ForbiddenExport {
        unit: &'static str,
        export: &'static str,
    },
}

struct ArtifactView<'a> {
    operator: &'a str,
    backend: NativeOperatorBackend,
    operator_abi_version: &'a str,
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
        unit.validate_package_contract(artifact.operator_abi_version, artifact.exports)?;
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
        // Several build capabilities may share one artifact; materialize one
        // view with the union of that artifact's requested exports.
        let mut by_operator = std::collections::BTreeMap::<&str, BTreeSet<String>>::new();
        for unit in CudaNativeBuildUnit::ALL {
            by_operator
                .entry(unit.artifact_operator())
                .or_default()
                .extend(exports(unit));
        }
        let operator_exports = by_operator
            .into_iter()
            .map(|(operator, symbols)| (operator, symbols.into_iter().collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let artifacts = operator_exports
            .iter()
            .map(|(operator, symbols)| ArtifactView {
                operator,
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version: CudaNativeBuildUnit::from_artifact_operator(operator)
                    .and_then(CudaNativeBuildUnit::required_operator_abi_version)
                    .unwrap_or("1"),
                exports: symbols,
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
            operator_abi_version: "3",
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
            operator_abi_version: "3",
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

    #[test]
    fn moe_profiler_retirement_requires_new_abi_and_rejects_old_exports() {
        let unit = CudaNativeBuildUnit::VllmMoeMarlin;
        let symbols = exports(unit);
        for operator_abi_version in ["1", "2"] {
            let old = [ArtifactView {
                operator: unit.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version,
                exports: &symbols,
            }];
            assert!(matches!(
                resolve_views(&old, [unit]),
                Err(CudaNativeBuildCoverageError::WrongOperatorAbi { expected: "3", .. })
            ));
        }
        for retired in unit.forbidden_exports() {
            let mut symbols = symbols.clone();
            symbols.push((*retired).to_owned());
            let artifact = [ArtifactView {
                operator: unit.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version: "3",
                exports: &symbols,
            }];
            assert!(matches!(resolve_views(&artifact, [unit]),
                Err(CudaNativeBuildCoverageError::ForbiddenExport { export, .. })
                    if export == *retired));
        }
        let current = [ArtifactView {
            operator: unit.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version: "3",
            exports: &symbols,
        }];
        assert!(resolve_views(&current, [unit]).is_ok());
    }

    #[test]
    fn marlin_ambient_policy_retirement_rejects_old_abi_with_complete_math_exports() {
        let unit = CudaNativeBuildUnit::Marlin;
        let symbols = exports(unit);
        let artifact = |operator_abi_version| ArtifactView {
            operator: unit.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version,
            exports: &symbols,
        };
        assert!(matches!(
            resolve_views(&[artifact("1")], [unit]),
            Err(CudaNativeBuildCoverageError::WrongOperatorAbi { expected: "2", .. })
        ));
        assert!(resolve_views(&[artifact("2")], [unit]).is_ok());
    }
    #[test]
    fn extra_artifact_cannot_be_satisfied_by_original_operator_or_symbols() {
        let old = exports(CudaNativeBuildUnit::UpstreamLinear);
        let extra = CudaNativeBuildUnit::UpstreamExtraLinear;
        let old_operator = [ArtifactView {
            operator: CudaNativeBuildUnit::UpstreamLinear.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version: "1",
            exports: &old,
        }];
        assert!(matches!(
            resolve_views(&old_operator, [extra]),
            Err(CudaNativeBuildCoverageError::MissingArtifact { .. })
        ));
        let wrong_symbols = [ArtifactView {
            operator: extra.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version: "1",
            exports: &old,
        }];
        assert!(matches!(
            resolve_views(&wrong_symbols, [extra]),
            Err(CudaNativeBuildCoverageError::MissingExport { .. })
        ));
        let new = exports(extra);
        let both = [
            ArtifactView {
                operator: CudaNativeBuildUnit::UpstreamLinear.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version: "1",
                exports: &old,
            },
            ArtifactView {
                operator: extra.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version: "1",
                exports: &new,
            },
        ];
        assert!(resolve_views(&both, [CudaNativeBuildUnit::UpstreamLinear, extra]).is_ok());
    }

    #[test]
    fn q6_f32_requires_its_own_operator_and_every_export() {
        let q6 = CudaNativeBuildUnit::UpstreamQ6F32Linear;
        let old = exports(CudaNativeBuildUnit::UpstreamLinear);
        let old_artifact = [ArtifactView {
            operator: CudaNativeBuildUnit::UpstreamLinear.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version: "1",
            exports: &old,
        }];
        assert!(matches!(
            resolve_views(&old_artifact, [q6]),
            Err(CudaNativeBuildCoverageError::MissingArtifact { .. })
        ));
        for missing in q6.required_exports() {
            let available = q6
                .required_exports()
                .iter()
                .filter(|e| *e != missing)
                .map(|e| (*e).to_owned())
                .collect::<Vec<_>>();
            let artifact = [ArtifactView {
                operator: q6.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version: "1",
                exports: &available,
            }];
            assert!(
                matches!(resolve_views(&artifact,[q6]), Err(CudaNativeBuildCoverageError::MissingExport { export, .. }) if export == *missing)
            );
        }
    }
    #[test]
    fn indexed_paged_attention_requires_indexed_and_retained_addressed_exports() {
        let unit = CudaNativeBuildUnit::VllmIndexedPagedAttentionV2;
        let complete = exports(unit);
        assert!(resolve_views(
            &[ArtifactView {
                operator: unit.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version: "1",
                exports: &complete,
            }],
            [unit]
        )
        .is_ok());
        for missing in unit.required_exports() {
            let available = complete
                .iter()
                .filter(|symbol| symbol.as_str() != *missing)
                .cloned()
                .collect::<Vec<_>>();
            let artifact = [ArtifactView {
                operator: unit.artifact_operator(),
                backend: NativeOperatorBackend::Cuda,
                operator_abi_version: "1",
                exports: &available,
            }];
            assert!(matches!(resolve_views(&artifact, [unit]),
                Err(CudaNativeBuildCoverageError::MissingExport { export, .. }) if export == *missing));
        }
    }
    #[test]
    fn original_paged_attention_artifact_remains_valid_without_indexed_capability() {
        let original = CudaNativeBuildUnit::VllmPagedAttentionV2;
        let indexed = CudaNativeBuildUnit::VllmIndexedPagedAttentionV2;
        let symbols = exports(original);
        let artifact = [ArtifactView {
            operator: original.artifact_operator(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version: "1",
            exports: &symbols,
        }];
        assert!(resolve_views(&artifact, [original]).is_ok());
        assert!(matches!(resolve_views(&artifact, [original, indexed]),
            Err(CudaNativeBuildCoverageError::MissingExport { unit, export })
                if unit == indexed.as_str() && export.ends_with("_addressed_indexed_v1")));
        assert_eq!(
            CudaNativeBuildUnit::from_artifact_operator(original.artifact_operator()),
            Some(original)
        );
    }
}
