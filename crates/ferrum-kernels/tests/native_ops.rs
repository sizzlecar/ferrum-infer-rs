use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ferrum_kernels::native_ops::{
    compiled_fa2_native_operator_artifact, compiled_fa2_native_operator_artifact_linked,
    compiled_fa2_native_operator_artifact_state, resolve_cuda_fa2_native_operator,
    validate_compiled_native_operator_provider_catalog, NativeOperatorArtifactSpec,
    CUDA_NATIVE_SOURCE_BUNDLE_ID, FA2_NATIVE_OPERATOR,
};
use ferrum_native_ops::{NativeOperatorArtifactFormat, NativeOperatorResolveError};
use ferrum_types::{
    CompiledNativeOperatorIdentity, NativeOperatorBackend, NativeOperatorBinding,
    NativeOperatorBuildSummary, NativeOperatorContractVersion, NativeOperatorLinkage,
    NativeOperatorManifest, NativeOperatorProviderCatalog, NativeOperatorProviderCatalogRow,
    NativeOperatorSourcePackage, FERRUM_NATIVE_OPERATOR_ABI_VERSION,
    NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

const CUDA_SOURCE_BUNDLE_MANIFEST: &str =
    include_str!("../../../native-operators/cuda/source-bundles/ferrum-native-cuda-v1.json");

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn compiled_cuda_source_bundle_identity_matches_the_distribution_manifest() {
    let manifest: serde_json::Value =
        serde_json::from_str(CUDA_SOURCE_BUNDLE_MANIFEST).expect("valid source bundle manifest");

    assert_eq!(
        manifest["bundle_id"].as_str(),
        Some(CUDA_NATIVE_SOURCE_BUNDLE_ID)
    );
}

#[test]
fn product_workspace_does_not_vendor_native_source_bundle_members() {
    let manifest: serde_json::Value =
        serde_json::from_str(CUDA_SOURCE_BUNDLE_MANIFEST).expect("valid source bundle manifest");
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let members = manifest["members"]
        .as_array()
        .expect("source bundle members");

    assert_eq!(members.len(), 53);
    for member in members {
        let relative = member["path"].as_str().expect("source bundle member path");
        assert!(
            !manifest_dir.join(relative).exists(),
            "product workspace retained bundled native source {relative}"
        );
    }
    for removed_tree in [
        "vllm_marlin",
        "kernels/vllm_marlin_moe",
        "kernels/vllm_attn",
    ] {
        assert!(
            !manifest_dir.join(removed_tree).exists(),
            "product workspace retained native source tree {removed_tree}"
        );
    }
}

struct TestDir(PathBuf);

impl TestDir {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct NativeOpFixture {
    _dir: TestDir,
    manifest: PathBuf,
    artifact: PathBuf,
    artifact_sha256: String,
    source_package_sha256: String,
    inputs_sha256: String,
}

fn temp_dir(name: &str) -> TestDir {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "ferrum-kernels-native-op-{name}-{}-{counter}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    TestDir(dir)
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest(ch: char) -> String {
    std::iter::repeat(ch).take(64).collect()
}

fn write_static_archive(root: &Path, include_descriptor: bool) -> PathBuf {
    let source = root.join("native_op.c");
    let mut source_text = String::from("int ferrum_native_fa2_execute_v1(void) { return 0; }\n");
    if include_descriptor {
        source_text
            .push_str("const char *ferrum_native_fa2_descriptor_v2(void) { return \"fa2\"; }\n");
    }
    std::fs::write(&source, source_text).unwrap();
    let object = root.join("native_op.o");
    let archive = root.join("libferrum_native_fa2.a");
    let cc_status = Command::new("cc")
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .unwrap();
    assert!(cc_status.success());
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&archive)
        .arg(&object)
        .status()
        .unwrap();
    assert!(ar_status.success());
    archive
}

fn write_manifest(
    path: &Path,
    binary_sha256: String,
    source_package_sha256: String,
    inputs_sha256: String,
) {
    let manifest = NativeOperatorManifest {
        schema_version: NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
        operator: FA2_NATIVE_OPERATOR.to_string(),
        operator_abi_version: "1".to_string(),
        ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
        backend: NativeOperatorBackend::Cuda,
        cuda_toolkit: Some("12.4".to_string()),
        cuda_runtime_min: Some("12.4".to_string()),
        compute_capabilities: vec!["sm_89".to_string()],
        source_package: NativeOperatorSourcePackage {
            kind: "external_archive".to_string(),
            revision: "test-revision".to_string(),
            sha256: source_package_sha256,
        },
        inputs_sha256,
        binary_sha256,
        linkage: NativeOperatorLinkage::Static,
        g03_catalog_sha256: Some(digest('c')),
        abi_contract_sha256: Some(digest('d')),
        descriptor_export: Some("ferrum_native_fa2_descriptor_v2".to_string()),
        operation_bindings: vec![NativeOperatorBinding {
            operation_id: "operation.causal_paged_attention".to_string(),
            operation_contract_version: ferrum_types::NativeOperatorContractVersion::new(1, 0),
            provider_id: "provider.cuda.fa2".to_string(),
            provider_version: ferrum_types::NativeOperatorContractVersion::new(1, 0),
            provider_implementation_fingerprint: digest('e'),
            entrypoints: vec!["ferrum_native_fa2_execute_v1".to_string()],
        }],
        exports: vec![
            "ferrum_native_fa2_descriptor_v2".to_string(),
            "ferrum_native_fa2_execute_v1".to_string(),
        ],
        license_files: vec!["LICENSE".to_string()],
        build_summary: NativeOperatorBuildSummary {
            builder_sha: digest('7'),
            elapsed_ms: 1,
            nvcc_version: Some("12.4".to_string()),
            host_compiler: "cc".to_string(),
        },
    };
    std::fs::write(path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
}

fn fixture(include_descriptor: bool) -> NativeOpFixture {
    let dir = temp_dir("fa2");
    let artifact = write_static_archive(dir.path(), include_descriptor);
    let artifact_sha256 = digest_bytes(&std::fs::read(&artifact).unwrap());
    let source_package_sha256 = digest('a');
    let inputs_sha256 = digest('b');
    let manifest = dir.path().join("native_operator_manifest.json");
    write_manifest(
        &manifest,
        artifact_sha256.clone(),
        source_package_sha256.clone(),
        inputs_sha256.clone(),
    );
    NativeOpFixture {
        _dir: dir,
        manifest,
        artifact,
        artifact_sha256,
        source_package_sha256,
        inputs_sha256,
    }
}

fn spec(fixture: &NativeOpFixture) -> NativeOperatorArtifactSpec {
    NativeOperatorArtifactSpec::cuda_fa2(&fixture.manifest, &fixture.artifact, "sm_89")
        .with_source_package_sha256(fixture.source_package_sha256.clone())
        .with_inputs_sha256(fixture.inputs_sha256.clone())
        .with_binary_sha256(fixture.artifact_sha256.clone())
}

fn live_provider_catalog() -> NativeOperatorProviderCatalog {
    NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers: vec![
            NativeOperatorProviderCatalogRow {
                operation_id: "operation.causal_paged_attention".to_string(),
                operation_contract_version: NativeOperatorContractVersion::new(1, 2),
                operation_fingerprint: digest('a'),
                provider_id: "provider.cuda.causal_paged_attention.f16".to_string(),
                provider_version: NativeOperatorContractVersion::new(3, 4),
                provider_implementation_fingerprint: digest('b'),
            },
            NativeOperatorProviderCatalogRow {
                operation_id: "operation.gated_delta_recurrent_attention".to_string(),
                operation_contract_version: NativeOperatorContractVersion::new(1, 0),
                operation_fingerprint: digest('c'),
                provider_id: "provider.cuda.gated_delta_recurrent_attention.f16".to_string(),
                provider_version: NativeOperatorContractVersion::new(1, 0),
                provider_implementation_fingerprint: digest('d'),
            },
        ],
    }
}

fn compiled_provider(catalog: &NativeOperatorProviderCatalog) -> CompiledNativeOperatorIdentity {
    let provider = &catalog.providers[0];
    CompiledNativeOperatorIdentity {
        schema_version: NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
        operator: "ferrum.cuda.vllm_paged_attention_v2".to_string(),
        operator_abi_version: "1".to_string(),
        ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
        backend: NativeOperatorBackend::Cuda,
        linkage: NativeOperatorLinkage::Static,
        g03_catalog_sha256: Some(catalog.canonical_sha256().unwrap()),
        abi_contract_sha256: Some(digest('c')),
        descriptor_export: Some(
            "ferrum_native_ferrum_cuda_vllm_paged_attention_v2_descriptor_v2".to_string(),
        ),
        operation_bindings: vec![NativeOperatorBinding {
            operation_id: provider.operation_id.clone(),
            operation_contract_version: provider.operation_contract_version,
            provider_id: provider.provider_id.clone(),
            provider_version: provider.provider_version,
            provider_implementation_fingerprint: provider
                .provider_implementation_fingerprint
                .clone(),
            entrypoints: vec!["ferrum_vnext_paged_attention".to_string()],
        }],
        exports: vec!["ferrum_vnext_paged_attention".to_string()],
        source_package_sha256: digest('d'),
        inputs_sha256: digest('e'),
        binary_sha256: digest('f'),
    }
}

#[test]
fn normal_test_build_does_not_report_fa2_native_artifact_linked() {
    assert!(!compiled_fa2_native_operator_artifact_linked());
    assert_eq!(
        compiled_fa2_native_operator_artifact_state(),
        "not_configured"
    );
    assert!(compiled_fa2_native_operator_artifact().is_none());
}

#[test]
fn compiled_native_operator_bindings_match_the_exact_live_catalog() {
    let catalog = live_provider_catalog();
    let artifact = compiled_provider(&catalog);
    validate_compiled_native_operator_provider_catalog(&catalog, &[artifact]).unwrap();
    validate_compiled_native_operator_provider_catalog(&catalog, &[]).unwrap();
}

#[test]
fn compiled_native_operator_bindings_accept_unrelated_catalog_provenance_change() {
    let catalog = live_provider_catalog();
    let artifact = compiled_provider(&catalog);
    let mut changed_catalog = catalog.clone();
    changed_catalog.providers[1].provider_implementation_fingerprint = digest('0');
    let changed_catalog_sha256 = changed_catalog.canonical_sha256().unwrap();
    assert_ne!(
        artifact.g03_catalog_sha256.as_deref(),
        Some(changed_catalog_sha256.as_str())
    );

    validate_compiled_native_operator_provider_catalog(&changed_catalog, &[artifact]).unwrap();
}

#[test]
fn compiled_native_operator_bindings_reject_changed_bound_provider_identity() {
    let catalog = live_provider_catalog();
    let mut stale_provider = compiled_provider(&catalog);
    stale_provider.operation_bindings[0].provider_implementation_fingerprint = digest('0');
    assert!(
        validate_compiled_native_operator_provider_catalog(&catalog, &[stale_provider])
            .unwrap_err()
            .contains("differs from the live provider identity")
    );
}

#[test]
fn compiled_native_operator_bindings_reject_missing_bound_provider() {
    let catalog = live_provider_catalog();
    let artifact = compiled_provider(&catalog);
    let mut incomplete_catalog = catalog;
    incomplete_catalog.providers.remove(0);

    assert!(
        validate_compiled_native_operator_provider_catalog(&incomplete_catalog, &[artifact])
            .unwrap_err()
            .contains("binds missing live provider")
    );
}

#[test]
fn compiled_native_operator_set_rejects_missing_bindings_or_provenance() {
    let catalog = live_provider_catalog();
    let mut unbound = compiled_provider(&catalog);
    unbound.operation_bindings.clear();
    assert!(
        validate_compiled_native_operator_provider_catalog(&catalog, &[unbound])
            .unwrap_err()
            .contains("does not bind any live G03 operation/provider")
    );

    let mut missing_provenance = compiled_provider(&catalog);
    missing_provenance.g03_catalog_sha256 = None;
    assert!(
        validate_compiled_native_operator_provider_catalog(&catalog, &[missing_provenance])
            .unwrap_err()
            .contains("has no provider catalog provenance")
    );
}

#[test]
fn resolves_cuda_fa2_native_operator_with_pinned_hashes_and_exports() {
    let fixture = fixture(true);
    let selection = resolve_cuda_fa2_native_operator(&spec(&fixture)).unwrap();

    assert_eq!(selection.operator, FA2_NATIVE_OPERATOR);
    assert_eq!(selection.backend, NativeOperatorBackend::Cuda);
    assert_eq!(selection.compute_capability.as_deref(), Some("sm_89"));
    assert_eq!(selection.linkage, NativeOperatorLinkage::Static);
    assert_eq!(selection.binary_sha256, fixture.artifact_sha256);
    assert_eq!(
        selection.source_package_sha256,
        fixture.source_package_sha256
    );
    assert_eq!(selection.inputs_sha256, fixture.inputs_sha256);
    assert_eq!(
        selection.artifact_format,
        NativeOperatorArtifactFormat::StaticArchive
    );
    assert!(selection
        .archive_members
        .iter()
        .any(|member| member == "native_op.o"));
    assert_eq!(
        selection.required_exports,
        vec![
            "ferrum_native_fa2_descriptor_v2".to_string(),
            "ferrum_native_fa2_execute_v1".to_string(),
        ]
    );
}

#[test]
fn rejects_source_hash_mismatch_before_runtime_selection() {
    let fixture = fixture(true);
    let bad_spec = spec(&fixture).with_source_package_sha256(
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    );

    let err = resolve_cuda_fa2_native_operator(&bad_spec).unwrap_err();
    assert!(
        matches!(err, NativeOperatorResolveError::ManifestInvalid(_)),
        "{err:?}"
    );
    assert!(err.to_string().contains("source_package.sha256 mismatch"));
}

#[test]
fn rejects_non_fa2_selection_for_fa2_helper() {
    let fixture = fixture(true);
    let mut bad_spec = spec(&fixture);
    bad_spec.operator = "dummy".to_string();

    let err = resolve_cuda_fa2_native_operator(&bad_spec).unwrap_err();
    assert!(
        matches!(err, NativeOperatorResolveError::ManifestInvalid(_)),
        "{err:?}"
    );
}

#[test]
fn rejects_archive_missing_declared_descriptor_export() {
    let fixture = fixture(false);

    let err = resolve_cuda_fa2_native_operator(&spec(&fixture)).unwrap_err();
    assert!(
        matches!(
            err,
            NativeOperatorResolveError::ArtifactMissingExports { .. }
        ),
        "{err:?}"
    );
}
