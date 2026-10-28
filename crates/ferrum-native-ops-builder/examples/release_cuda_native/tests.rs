use super::*;
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalogRow,
    NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
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

#[test]
fn catalog_extension_preserves_text_contracts_and_rejects_missing_or_changed_rows() {
    let original = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers: vec![row("operation.text", "provider.cuda.text")],
    };
    let mut combined = original.clone();
    combined.providers.insert(
        0,
        row(
            "operation.dense_linear_fp8.bf16",
            "provider.cuda.dense_linear_fp8.bf16.cutlass",
        ),
    );
    validate_catalog_extension(&original, &combined).unwrap();
    assert!(validate_catalog_extension(&original, &original).is_err());
    let mut missing_text = combined.clone();
    missing_text.providers.pop();
    assert!(validate_catalog_extension(&original, &missing_text).is_err());
    combined.providers[1].provider_implementation_fingerprint = "c".repeat(64);
    assert!(validate_catalog_extension(&original, &combined).is_err());
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
