use super::*;

const LDD: &str = "ferrum: ELF 64-bit LSB pie executable, x86-64, dynamically linked\n\tlinux-vdso.so.1 (0x00007fff)\n\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00001234)\n\t/lib64/ld-linux-x86-64.so.2 (0x00005678)\n";
const OTOOL: &str = "ferrum: Mach-O 64-bit executable arm64\nferrum:\n\t/System/Library/Frameworks/Metal.framework/Versions/A/Metal (compatibility version 1.0.0, current version 1.0.0)\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1351.0.0)\n";

fn candidate(version: &str) -> CandidateInput {
    CandidateInput {
        version: version.into(),
        release_candidate_sha: "a".repeat(40),
        release_candidate_tag: format!("v{version}-rc.2"),
        staging_label: format!("v{version}-candidate"),
        workflow_run_id: "12345".into(),
        workflow_run_attempt: "2".into(),
    }
}

fn abi(backend: Backend) -> AbiInput {
    match backend {
        Backend::Cpu => AbiInput {
            backend,
            target_triple: "x86_64-unknown-linux-gnu".into(),
            cargo_features: vec![],
            cuda_compute_capability: None,
            cuda_toolkit_image: None,
        },
        Backend::Metal => AbiInput {
            backend,
            target_triple: "aarch64-apple-darwin".into(),
            cargo_features: vec!["metal".into()],
            cuda_compute_capability: None,
            cuda_toolkit_image: None,
        },
        Backend::Cuda => AbiInput {
            backend,
            target_triple: "x86_64-unknown-linux-gnu".into(),
            cargo_features: vec![
                "cuda".into(),
                "vllm-moe-marlin".into(),
                "vllm-paged-attn-v2".into(),
            ],
            cuda_compute_capability: Some("89".into()),
            cuda_toolkit_image: Some("nvidia/cuda:12.4.0-devel-ubuntu22.04".into()),
        },
    }
}

fn generate(backend: Backend, audit: &str) -> Result<StagedManifests, String> {
    generate_manifests(
        &candidate("1.2.3"),
        &abi(backend),
        "asset.tar.gz",
        b"archive",
        b"binary",
        "dependencies.txt",
        audit,
    )
}

#[test]
fn candidate_version_is_semantic_and_not_pinned_to_a_release() {
    for version in ["0.1.0", "2.34.567"] {
        validate_candidate(&candidate(version)).unwrap();
    }
    for version in ["01.2.3", "1.2", "v1.2.3", "1.2.3-beta.1", "1.2.3+build"] {
        assert!(validate_candidate(&candidate(version)).is_err());
    }
    let mut input = candidate("1.2.3");
    for tag in [
        "v1.2.4-rc.2",
        "v1.2.3",
        "v1.2.3-rc.0",
        "v1.2.3-rc.01",
        "v1.2.3-rc.2.extra",
    ] {
        input.release_candidate_tag = tag.into();
        assert!(validate_candidate(&input).is_err());
    }
}

#[test]
fn candidate_metadata_rejects_injection_but_never_compares_to_a_trusted_commit() {
    for field in ["$(touch x)", "a\nb", "../../outside", "", "x;y"] {
        let mut input = candidate("1.2.3");
        input.staging_label = field.into();
        assert!(validate_candidate(&input).is_err());
    }
    for commit in ["0".repeat(40), "f".repeat(64)] {
        let mut input = candidate("1.2.3");
        input.release_candidate_sha = commit;
        validate_candidate(&input).unwrap();
    }
    let mut input = candidate("1.2.3");
    input.release_candidate_sha = "a;echo x".into();
    assert!(validate_candidate(&input).is_err());
}

#[test]
fn workspace_member_versions_must_all_match_and_dependencies_are_not_members() {
    let mut metadata = json!({
        "workspace_members": ["cli", "types"],
        "packages": [
            {"id": "cli", "name": "ferrum-cli", "version": "2.0.1"},
            {"id": "types", "name": "ferrum-types", "version": "2.0.1"},
            {"id": "dependency", "version": "9.9.9"}
        ]
    });
    validate_workspace_versions(&metadata, "2.0.1").unwrap();
    metadata["packages"][1]["version"] = json!("2.0.0");
    assert!(validate_workspace_versions(&metadata, "2.0.1").is_err());
    metadata["workspace_members"] = json!(["missing"]);
    assert!(validate_workspace_versions(&metadata, "2.0.1").is_err());
    metadata["workspace_members"] = json!([]);
    assert!(validate_workspace_versions(&metadata, "2.0.1").is_err());
}

#[test]
fn manifests_hash_actual_bytes_and_preserve_adjacent_manifest_contract() {
    let input = candidate("3.4.5");
    let manifest = generate_manifests(
        &input,
        &abi(Backend::Cpu),
        "asset.tar.gz",
        b"abc",
        b"abc",
        "dependencies.txt",
        LDD,
    )
    .unwrap();
    let known_sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(
        manifest.asset_checksum,
        format!("{known_sha}  asset.tar.gz\n")
    );
    assert_eq!(manifest.binary_checksum, format!("{known_sha}  ferrum\n"));
    assert_eq!(manifest.version["version"], "3.4.5");
    assert_eq!(manifest.version["asset_sha256"], known_sha);
    assert_eq!(manifest.dependency["audit_sha256"], digest(LDD.as_bytes()));
    assert_eq!(
        manifest.abi["dependency_audit_sha256"],
        manifest.dependency["audit_sha256"]
    );
    assert_eq!(manifest.abi["backend"], "cpu");
    assert_eq!(
        manifest.dependency["forbidden_runtime_linkage_found"],
        false
    );
    let changed = generate_manifests(
        &input,
        &abi(Backend::Cpu),
        "asset.tar.gz",
        b"abd",
        b"binary",
        "dependencies.txt",
        LDD,
    )
    .unwrap();
    assert_ne!(
        changed.version["asset_sha256"],
        manifest.version["asset_sha256"]
    );
    assert_ne!(
        changed.version["binary_sha256"],
        manifest.version["binary_sha256"]
    );
}

#[test]
fn absent_or_claimed_success_is_not_a_dependency_audit() {
    for text in [
        "",
        "passed",
        r#"{"passed":true}"#,
        "ferrum: ELF 64-bit LSB pie executable\n",
        "libc.so.6 => not found\n",
        "libc.so.6 => /lib/libc.so.6\n",
        "libc.so.6 => /lib/libc.so.6 (garbage)\n",
    ] {
        assert!(generate(Backend::Cpu, text).is_err(), "{text:?}");
    }
}

#[test]
fn forbidden_dynamic_libraries_are_rejected_from_names_and_resolved_paths() {
    for library in ["libpython3.12.so", "libTorch.so", "libvllm.so"] {
        let audit = format!("{LDD}\t{library} => /lib/{library} (0x1234)\n");
        assert!(generate(Backend::Cuda, &audit).is_err());
    }
    let aliased = format!("{LDD}\tlibneutral.so => /lib/libtorch.so (0x1234)\n");
    assert!(generate(Backend::Cpu, &aliased).is_err());
    let framework = format!("{OTOOL}\t/Library/Frameworks/Python.framework/Versions/3.12/Python (compatibility version 3.12.0, current version 3.12.0)\n");
    assert!(generate(Backend::Metal, &framework).is_err());
    // Native static operators are declared features, not linked shared libraries.
    let manifest = generate(Backend::Cuda, LDD).unwrap();
    assert_eq!(manifest.abi["cargo_features"][1], "vllm-moe-marlin");
}

#[test]
fn only_cuda_driver_resolution_can_be_deferred_to_the_runtime_host() {
    let missing_driver = format!("{LDD}\tlibcuda.so.1 => not found\n");
    let manifest = generate(Backend::Cuda, &missing_driver).unwrap();
    assert_eq!(
        manifest.dependency["unresolved_runtime_libraries"],
        json!(["libcuda.so.1"])
    );
    assert_eq!(
        manifest.dependency["deferred_runtime_dependencies"],
        json!(["libcuda.so.1"])
    );
    assert!(generate(Backend::Cpu, &missing_driver).is_err());
    assert!(generate(
        Backend::Cuda,
        &format!("{LDD}\tlibcudart.so.12 => not found\n")
    )
    .is_err());
    assert!(generate(Backend::Cuda, &format!("{LDD}\tlibtorch.so => not found\n")).is_err());
}

#[test]
fn abi_features_and_raw_audit_format_must_match_the_declared_backend() {
    generate(Backend::Metal, OTOOL).unwrap();
    assert!(generate(Backend::Metal, LDD).is_err());
    assert!(generate(Backend::Cpu, OTOOL).is_err());
    let mut wrong = abi(Backend::Cpu);
    wrong.cargo_features = vec!["cuda".into()];
    assert!(validate_abi(&wrong).is_err());
    let mut wrong = abi(Backend::Cuda);
    wrong.cuda_compute_capability = None;
    assert!(validate_abi(&wrong).is_err());
    let mut wrong = abi(Backend::Metal);
    wrong.target_triple = "x86_64-unknown-linux-gnu".into();
    assert!(validate_abi(&wrong).is_err());
}

#[test]
fn platform_and_backend_mismatches_cannot_generate_manifests() {
    for (backend, target, text) in [
        (Backend::Cuda, "aarch64-apple-darwin", OTOOL),
        (Backend::Metal, "aarch64-unknown-linux-gnu", LDD),
        (Backend::Cuda, "x86_64-pc-windows-msvc", LDD),
        (Backend::Cpu, "x86_64-linux-unknown-gnu", LDD),
        (Backend::Cuda, "x86_64-unknown-linux-gnu-extra", LDD),
        (Backend::Cpu, "-unknown-linux-gnu", LDD),
    ] {
        let mut input = abi(backend);
        input.target_triple = target.into();
        let error = generate_manifests(
            &candidate("1.2.3"),
            &input,
            "asset.tar.gz",
            b"archive",
            b"binary",
            "dependencies.txt",
            text,
        )
        .unwrap_err();
        assert!(error.contains("target"), "{backend:?} / {target}: {error}");
    }
}

#[test]
fn platform_classification_does_not_pin_architecture_or_environment() {
    for (backend, target, expected) in [
        (
            Backend::Cpu,
            "aarch64-unknown-linux-gnu",
            audit::Format::Ldd,
        ),
        (
            Backend::Cpu,
            "x86_64-unknown-linux-musl",
            audit::Format::Ldd,
        ),
        (
            Backend::Cuda,
            "aarch64-unknown-linux-gnu",
            audit::Format::Ldd,
        ),
        (Backend::Metal, "x86_64-apple-darwin", audit::Format::Otool),
        (Backend::Cpu, "aarch64-apple-darwin", audit::Format::Otool),
        (Backend::Cpu, "x86_64-apple-darwin", audit::Format::Otool),
    ] {
        let mut input = abi(backend);
        input.target_triple = target.into();
        assert_eq!(validate_abi(&input).unwrap(), expected);
    }
}

#[test]
fn cpu_assets_use_native_platform_audits_without_gpu_runtime_dependencies() {
    let cpu_otool = OTOOL
        .lines()
        .filter(|line| !line.contains("Metal.framework"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut cpu = abi(Backend::Cpu);
    cpu.target_triple = "aarch64-apple-darwin".into();
    let manifests = generate_manifests(
        &candidate("1.2.3"),
        &cpu,
        "cpu.tar.gz",
        b"archive",
        b"binary",
        "dependencies.txt",
        &cpu_otool,
    )
    .unwrap();
    assert_eq!(manifests.abi["backend"], "cpu");
    assert_eq!(
        manifests.dependency["unresolved_runtime_libraries"],
        json!([])
    );
    let error = generate_manifests(
        &candidate("1.2.3"),
        &cpu,
        "cpu.tar.gz",
        b"archive",
        b"binary",
        "dependencies.txt",
        OTOOL,
    )
    .unwrap_err();
    assert!(error.contains("GPU runtime"), "{error}");
    for library in [
        "libcuda.so.1",
        "libcudart.so.12",
        "libcublas.so.12",
        "libnccl.so.2",
    ] {
        let text = format!("{LDD}\t{library} => /usr/lib/{library} (0x00001234)\n");
        assert!(generate(Backend::Cpu, &text)
            .unwrap_err()
            .contains("GPU runtime"));
        generate(Backend::Cuda, &text).unwrap();
    }
}

#[test]
fn raw_dependency_format_cannot_be_swapped_without_file_description() {
    // Reject the wrong dependency grammar even when no `file` header supplies
    // a convenient ELF/Mach-O hint. The legitimate grammar still succeeds.
    let ldd = LDD.lines().skip(1).collect::<Vec<_>>().join("\n");
    let otool = OTOOL.lines().skip(2).collect::<Vec<_>>().join("\n");
    generate(Backend::Cpu, &ldd).unwrap();
    generate(Backend::Cuda, &ldd).unwrap();
    generate(Backend::Metal, &otool).unwrap();
    assert!(generate(Backend::Cpu, &otool).is_err());
    assert!(generate(Backend::Cuda, &otool).is_err());
    assert!(generate(Backend::Metal, &ldd).is_err());
}

#[test]
fn checksum_filenames_and_empty_payloads_cannot_create_misleading_manifests() {
    for name in ["../../asset", "asset\nfake", ".", "..", "asset file"] {
        assert!(generate_manifests(
            &candidate("1.2.3"),
            &abi(Backend::Cpu),
            name,
            b"a",
            b"b",
            "dependencies.txt",
            LDD
        )
        .is_err());
    }
    assert!(generate_manifests(
        &candidate("1.2.3"),
        &abi(Backend::Cpu),
        "asset.tar.gz",
        b"",
        b"b",
        "dependencies.txt",
        LDD
    )
    .is_err());
}

#[test]
fn formal_version_progression_uses_semver_and_rejects_equal_backward_or_injected_inputs() {
    for (previous, target) in [
        ("0.8.9", "0.8.10"),
        ("0.99.99", "1.0.0"),
        ("9.1.0", "10.0.0"),
    ] {
        validate_version_progression(previous, target).unwrap();
    }
    for (previous, target) in [
        ("0.8.9", "0.8.9"),
        ("0.8.10", "0.8.9"),
        ("1.0.0", "0.99.99"),
    ] {
        assert!(validate_version_progression(previous, target)
            .unwrap_err()
            .contains("must be newer"));
    }
    for invalid in [
        "v1.2.3",
        "1.2",
        "01.2.3",
        "1.2.3-rc.1",
        "1.2.3+build",
        "1.2.3\n",
        "$(touch x)",
    ] {
        assert!(validate_version_progression(invalid, "9.0.0").is_err());
        assert!(validate_version_progression("0.1.0", invalid).is_err());
    }
    // A retry of 1.2.4 still advances its unchanged previous release, 1.2.3.
    validate_version_progression("1.2.3", "1.2.4").unwrap();
    validate_version_progression("1.2.3", "1.2.4").unwrap();
}
