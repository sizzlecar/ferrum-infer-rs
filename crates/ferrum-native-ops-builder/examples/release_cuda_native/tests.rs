use super::*;
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorBinding, NativeOperatorContractVersion,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};

#[test]
fn production_moe_closure_rejects_missing_dense_repacking_artifact() {
    // Exercise the resolver's actual missing-artifact boundary for the MoE
    // feature closure, independently of unrelated production units.
    let set = ResolvedNativeOperatorArtifactSet {
        lock_path: PathBuf::from("fixture.lock.json"),
        g03_catalog_sha256: "a".repeat(64),
        artifacts: Vec::new(),
    };
    let required = PRODUCTION.iter().copied().filter(|unit| {
        matches!(
            unit,
            CudaNativeBuildUnit::VllmMarlin | CudaNativeBuildUnit::VllmMoeMarlin
        )
    });
    assert_eq!(
        ResolvedCudaNativeBuildCoverage::resolve(&set, required).unwrap_err(),
        ferrum_native_ops::CudaNativeBuildCoverageError::MissingArtifact {
            unit: "vllm_marlin",
            operator: "ferrum.cuda.vllm_marlin",
        }
    );
}

#[test]
fn both_release_architectures_require_quantized_projection_and_indexed_attention() {
    let set = ResolvedNativeOperatorArtifactSet {
        lock_path: PathBuf::from("fixture.lock.json"),
        g03_catalog_sha256: "a".repeat(64),
        artifacts: Vec::new(),
    };
    for architecture in [Architecture::Sm89, Architecture::Sm120] {
        for final_set in [false, true] {
            for required in [
                CudaNativeBuildUnit::UpstreamLinear,
                CudaNativeBuildUnit::UpstreamExtraLinear,
                CudaNativeBuildUnit::UpstreamQ6F32Linear,
                CudaNativeBuildUnit::VllmIndexedPagedAttentionV2,
            ] {
                let units = required_units_for_set(architecture, final_set)
                    .filter(|unit| *unit == required);
                assert_eq!(
                    ResolvedCudaNativeBuildCoverage::resolve(&set, units).unwrap_err(),
                    ferrum_native_ops::CudaNativeBuildCoverageError::MissingArtifact {
                        unit: required.as_str(),
                        operator: required.artifact_operator(),
                    }
                );
            }
        }
    }
}

#[test]
fn f32_only_q6_package_cannot_satisfy_the_production_f16_boundary() {
    let f32 = CudaNativeBuildUnit::UpstreamQ6F32Linear
        .required_exports()
        .iter()
        .map(|export| (*export).to_owned())
        .collect::<Vec<_>>();
    assert!(require_q6_f16_exports(&f32, false).is_err());
    let f16 = ferrum_native_ops::upstream_q6_f16_linear::UPSTREAM_Q6_F16_LINEAR_EXPORTS;
    let complete = f32
        .into_iter()
        .chain(f16.iter().map(|export| (*export).to_owned()))
        .collect::<Vec<_>>();
    require_q6_f16_exports(&complete, false).unwrap();
    assert!(require_q6_f16_exports(&complete, true).is_err());
    let mut prefill = complete.clone();
    prefill.push(
        ferrum_native_ops::upstream_q6_f16_linear::UPSTREAM_Q6_F16_PREFILL_PLAN_EXPORT.into(),
    );
    require_q6_f16_exports(&prefill, true).unwrap();
    for missing in f16 {
        let incomplete = complete
            .iter()
            .filter(|export| *export != missing)
            .cloned()
            .collect::<Vec<_>>();
        assert!(require_q6_f16_exports(&incomplete, false)
            .unwrap_err()
            .to_string()
            .contains(missing));
    }
}

#[test]
fn final_image_release_rejects_missing_fused_native_artifacts() {
    let set = ResolvedNativeOperatorArtifactSet {
        lock_path: PathBuf::from("fixture.lock.json"),
        g03_catalog_sha256: "a".repeat(64),
        artifacts: Vec::new(),
    };
    for architecture in [Architecture::Sm89, Architecture::Sm120] {
        for unit in [
            CudaNativeBuildUnit::CutlassFusedF32Attention,
            CudaNativeBuildUnit::CutlassInt8FusedOutput,
        ] {
            let required = |final_set| {
                required_units_for_set(architecture, final_set)
                    .filter(move |candidate| *candidate == unit)
            };
            assert!(ResolvedCudaNativeBuildCoverage::resolve(&set, required(false)).is_ok());
            assert!(
                matches!(ResolvedCudaNativeBuildCoverage::resolve(&set, required(true)), Err(ferrum_native_ops::CudaNativeBuildCoverageError::MissingArtifact { operator, .. }) if operator == unit.artifact_operator())
            );
        }
    }
}

#[test]
fn cross_architecture_package_keeps_provider_contract_and_uses_the_build_target() {
    let exports = vec!["ferrum_upstream_q6_f16_dot_v1".to_owned()];
    let reference = NativeOperatorPackageSpec {
        schema_version: ferrum_native_ops_builder::NATIVE_OPERATOR_PACKAGE_SPEC_SCHEMA_VERSION,
        operator: CudaNativeBuildUnit::UpstreamQ6F32Linear
            .artifact_operator()
            .to_owned(),
        operator_abi_version: "1".to_owned(),
        backend: NativeOperatorBackend::Cuda,
        compute_capabilities: vec!["sm_120".to_owned()],
        operation_bindings: vec![NativeOperatorBinding {
            operation_id: "operation.dense_swiglu.upstream-q6-f16-marker-v1".to_owned(),
            operation_contract_version: NativeOperatorContractVersion::new(1, 0),
            provider_id: "provider.cuda.dense_swiglu.upstream-q6-f16-marker-v1".to_owned(),
            provider_version: NativeOperatorContractVersion::new(1, 0),
            provider_implementation_fingerprint: "b".repeat(64),
            entrypoints: exports.clone(),
        }],
        required_exports: exports,
        license_files: vec![ferrum_native_ops_builder::NativeOperatorLicenseInput {
            source_path: "LICENSE".to_owned(),
            output_path: "licenses/Ferrum-MIT.txt".to_owned(),
        }],
        cuda_toolkit: Some("12.8".to_owned()),
        cuda_runtime_min: Some("12.8".to_owned()),
        system_libraries: vec![ferrum_native_ops::NativeOperatorSystemLibrary::CudaRuntime],
    };
    let rebuilt = package_spec_for_target(reference.clone(), Architecture::Sm89);
    assert_eq!(rebuilt.compute_capabilities, ["sm_89"]);
    assert_eq!(rebuilt.cuda_toolkit.as_deref(), Some("12.4"));
    assert_eq!(rebuilt.cuda_runtime_min.as_deref(), Some("12.4"));
    // Returning the target metadata to the reference architecture must recover
    // the entire original provider, ABI, license and linker contract.
    assert_eq!(
        package_spec_for_target(rebuilt, Architecture::Sm120),
        reference
    );
}

fn row(operation: &str, provider: &str) -> NativeOperatorProviderCatalogRow {
    NativeOperatorProviderCatalogRow {
        operation_id: operation.into(),
        operation_contract_version: NativeOperatorContractVersion::new(1, 0),
        operation_fingerprint: "a".repeat(64),
        provider_id: provider.into(),
        provider_version: NativeOperatorContractVersion::new(1, 0),
        provider_implementation_fingerprint: "b".repeat(64),
    }
}

fn image_provider_rows() -> Vec<NativeOperatorProviderCatalogRow> {
    vec![
        row(
            "operation.structured_gqa_attention_fused.f32",
            "provider.cuda.structured_gqa_attention_fused.f32.cutlass",
        ),
        row(
            "operation.comfy_convrot_codebook_w4a8.bf16",
            "provider.cuda.comfy_convrot_codebook_w4a8.bf16.cutlass_int8_fused_output",
        ),
        row(
            "operation.comfy_convrot_codebook_w4a8.swiglu_f32.bf16",
            "provider.cuda.comfy_convrot_w4a8.swiglu_f32.bf16.cutlass_int8_fused_output",
        ),
        row(
            "operation.comfy_convrot_w8a8.bf16",
            "provider.cuda.comfy_convrot_w8a8.bf16.cutlass_int8_fused_output",
        ),
        row(
            "operation.comfy_convrot_w8a8.swiglu_f32.bf16",
            "provider.cuda.comfy_convrot_w8a8.swiglu_f32.bf16.cutlass_int8_fused_output",
        ),
    ]
}

#[test]
fn catalog_extension_preserves_text_contracts_and_rejects_missing_or_changed_rows() {
    let original = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers: vec![row("operation.text", "provider.cuda.text")],
    };
    let mut combined = original.clone();
    for suffix in ["", "_pair", "_triple"] {
        combined.providers.push(row(
            &format!("operation.dense_linear_fp8{suffix}.bf16"),
            &format!("provider.cuda.dense_linear_fp8{suffix}.bf16.cutlass"),
        ));
    }
    combined.providers.extend(image_provider_rows());
    combined
        .providers
        .sort_by(|a, b| (&a.operation_id, &a.provider_id).cmp(&(&b.operation_id, &b.provider_id)));
    validate_catalog_extension(&original, &combined, None).unwrap();
    for image in image_provider_rows() {
        let mut incomplete = combined.clone();
        incomplete
            .providers
            .retain(|row| row.provider_id != image.provider_id);
        assert!(validate_catalog_extension(&original, &incomplete, None).is_err());
    }
    assert!(validate_catalog_extension(&original, &original, None).is_err());
    let mut missing_text = combined.clone();
    missing_text.providers.pop();
    assert!(validate_catalog_extension(&original, &missing_text, None).is_err());
    let mut missing_group = combined.clone();
    missing_group.providers.remove(1);
    assert!(validate_catalog_extension(&original, &missing_group, None).is_err());
    combined
        .providers
        .iter_mut()
        .find(|row| row.operation_id == "operation.text")
        .unwrap()
        .provider_implementation_fingerprint = "c".repeat(64);
    assert!(validate_catalog_extension(&original, &combined, None).is_err());
}

#[test]
fn sm120_final_coverage_requires_xqa_but_historical_and_sm89_sets_do_not() {
    let set = ResolvedNativeOperatorArtifactSet {
        lock_path: PathBuf::from("fixture.lock.json"),
        g03_catalog_sha256: "a".repeat(64),
        artifacts: Vec::new(),
    };
    let xqa = |architecture, final_set| {
        required_units_for_set(architecture, final_set)
            .filter(|unit| *unit == CudaNativeBuildUnit::FlashinferXqa)
    };
    assert!(
        ResolvedCudaNativeBuildCoverage::resolve(&set, xqa(Architecture::Sm120, false)).is_ok()
    );
    assert!(ResolvedCudaNativeBuildCoverage::resolve(&set, xqa(Architecture::Sm89, true)).is_ok());
    assert_eq!(
        ResolvedCudaNativeBuildCoverage::resolve(&set, xqa(Architecture::Sm120, true)).unwrap_err(),
        ferrum_native_ops::CudaNativeBuildCoverageError::MissingArtifact {
            unit: "flashinfer_xqa",
            operator: "ferrum.cuda.flashinfer_xqa_decode",
        }
    );
}

#[test]
fn gpt_catalog_refresh_requires_exact_source_row_and_preserves_contracts() {
    let original = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers: vec![
            row(
                "operation.gpt_oss.causal_paged_attention",
                "provider.cuda.gpt_oss.causal_paged_attention.f16",
            ),
            row("operation.text", "provider.cuda.text"),
        ],
    };
    let mut source = original.clone();
    source.providers[0].provider_implementation_fingerprint = "c".repeat(64);
    let mut combined = source.clone();
    for suffix in ["", "_pair", "_triple"] {
        combined.providers.push(row(
            &format!("operation.dense_linear_fp8{suffix}.bf16"),
            &format!("provider.cuda.dense_linear_fp8{suffix}.bf16.cutlass"),
        ));
    }
    combined.providers.extend(image_provider_rows());
    combined.providers.sort();
    validate_catalog_extension(&original, &combined, Some(&source)).unwrap();
    assert!(validate_catalog_extension(&original, &combined, None).is_err());
    let mut wrong = combined.clone();
    wrong
        .providers
        .iter_mut()
        .find(|r| r.operation_id == "operation.gpt_oss.causal_paged_attention")
        .unwrap()
        .provider_implementation_fingerprint = "d".repeat(64);
    assert!(validate_catalog_extension(&original, &wrong, Some(&source)).is_err());
    source.providers[0].operation_fingerprint = "e".repeat(64);
    assert!(validate_catalog_extension(&original, &combined, Some(&source)).is_err());
    source = original.clone();
    source.providers[1].provider_implementation_fingerprint = "f".repeat(64);
    assert!(validate_catalog_extension(&original, &combined, Some(&source)).is_err());
}

#[test]
fn source_materialization_selects_locked_bytes_and_refuses_substitution_or_overwrite() {
    let temp = tempfile::tempdir().unwrap();
    let old = temp.path().join("old");
    let candidate = temp.path().join("candidate");
    fs::create_dir(&old).unwrap();
    fs::create_dir(&candidate).unwrap();
    fs::write(old.join("kernel.cu"), b"pinned kernel").unwrap();
    fs::write(candidate.join("kernel.cu"), b"changed kernel").unwrap();
    let sha = digest_file(&old.join("kernel.cu")).unwrap();
    let inputs = [
        (candidate.as_path(), PathBuf::from("kernel.cu")),
        (old.as_path(), PathBuf::from("kernel.cu")),
    ];
    let output = temp.path().join("output/kernel.cu");
    copy_matching_source("kernel.cu", &sha, &inputs, &output).unwrap();
    assert_eq!(fs::read(&output).unwrap(), b"pinned kernel");
    assert!(copy_matching_source("kernel.cu", &sha, &inputs, &output).is_err());
    assert!(copy_matching_source(
        "kernel.cu",
        &"0".repeat(64),
        &inputs,
        &temp.path().join("bad")
    )
    .is_err());
    for invalid in [
        "",
        "../escape",
        "/absolute",
        "a/../escape",
        "C:drive",
        "a\\b",
    ] {
        assert!(checked_relative(invalid).is_err(), "{invalid}");
    }
}

#[cfg(unix)]
#[test]
fn source_materialization_rejects_symlink_escape_even_if_digest_matches() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("source");
    fs::create_dir(&root).unwrap();
    let outside = temp.path().join("outside");
    fs::write(&outside, b"locked").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("kernel.cu")).unwrap();
    assert!(copy_matching_source(
        "kernel.cu",
        &digest_file(&outside).unwrap(),
        &[(root.as_path(), PathBuf::from("kernel.cu"))],
        &temp.path().join("output")
    )
    .is_err());
}
