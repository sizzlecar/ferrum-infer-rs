use super::*;
use sha2::{Digest, Sha256};
use std::io::Cursor;

// Structural PE identity fixtures only: these are never executed and do not
// purport to model import closure or a working CUDA binary.
fn pe(dll: bool) -> Vec<u8> {
    let mut bytes = vec![0u8; 512];
    bytes[..2].copy_from_slice(b"MZ");
    bytes[60..64].copy_from_slice(&64u32.to_le_bytes());
    bytes[64..68].copy_from_slice(b"PE\0\0");
    bytes[68..70].copy_from_slice(&0x8664u16.to_le_bytes());
    bytes[84..86].copy_from_slice(&240u16.to_le_bytes());
    bytes[86..88].copy_from_slice(&(if dll { 0x2002u16 } else { 2u16 }).to_le_bytes());
    bytes[88..90].copy_from_slice(&0x20bu16.to_le_bytes());
    bytes
}

struct Fixture {
    root: tempfile::TempDir,
    spec: PackSpec,
    archive: PathBuf,
    receipt: PathBuf,
}

impl Fixture {
    fn cpu() -> Self {
        let mut fixture = Self::new();
        fixture.spec.manifest.backend = Backend::Cpu;
        fixture.spec.manifest.cuda_compute_capability.clear();
        fixture.spec.manifest.cargo_features.clear();
        fixture
            .spec
            .manifest
            .files
            .retain(|file| file.role != Role::CudaRuntime);
        fixture.spec.sources.remove("cudart64_12.dll");
        fixture.archive = fixture.root.path().join("ferrum-windows-x86_64-cpu.zip");
        fixture
    }

    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let mut files = Vec::new();
        let mut sources = BTreeMap::new();
        for (name, role, bytes) in [
            ("ferrum.exe", Role::Binary, pe(false)),
            ("cudart64_12.dll", Role::CudaRuntime, pe(true)),
            ("vcruntime140.dll", Role::MsvcRuntime, pe(true)),
            (
                "licenses/Ferrum.txt",
                Role::License,
                b"literal license\r\n".to_vec(),
            ),
        ] {
            let source = root.path().join("inputs").join(name);
            fs::create_dir_all(source.parent().unwrap()).unwrap();
            fs::write(&source, &bytes).unwrap();
            files.push(PayloadFile {
                path: name.into(),
                role,
                sha256: format!("{:x}", Sha256::digest(&bytes)),
                size_bytes: bytes.len() as u64,
                license: (role != Role::License).then(|| "licenses/Ferrum.txt".into()),
            });
            sources.insert(name.into(), source);
        }
        Self {
            archive: root.path().join("windows-cuda.zip"),
            receipt: root.path().join("receipt.json"),
            root,
            spec: PackSpec {
                manifest: Manifest {
                    schema_version: 1,
                    build: BuildIdentity {
                        version: "2.3.4".into(),
                        source_commit: "a".repeat(40),
                        build_id: "local-native-fixture".into(),
                    },
                    backend: Backend::Cuda,
                    target_triple: TARGET.into(),
                    cuda_compute_capability: "89".into(),
                    cargo_features: vec!["cuda".into()],
                    files,
                },
                sources,
            },
        }
    }

    fn pack(&self) -> Receipt {
        pack(&self.spec, &self.archive, &self.receipt).unwrap();
        read(&self.receipt).unwrap()
    }

    fn extracted(&self) -> PathBuf {
        self.root.path().join("portable 中文 directory")
    }

    fn replace_zip(&self, entries: Vec<(String, Vec<u8>, bool)>, receipt: &mut Receipt) {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes, symlink) in entries {
            let options = zip::write::SimpleFileOptions::default();
            if symlink {
                writer.add_symlink(name, "outside", options).unwrap();
            } else {
                writer.start_file(name, options).unwrap();
                writer.write_all(&bytes).unwrap();
            }
        }
        let bytes = writer.finish().unwrap().into_inner();
        fs::write(&self.archive, &bytes).unwrap();
        receipt.archive_sha256 = format!("{:x}", Sha256::digest(&bytes));
        receipt.archive_size_bytes = bytes.len() as u64;
    }

    fn entries(&self) -> Vec<(String, Vec<u8>, bool)> {
        let mut result = vec![(
            MANIFEST.into(),
            serde_json::to_vec(&self.spec.manifest).unwrap(),
            false,
        )];
        result.extend(self.spec.manifest.files.iter().map(|f| {
            (
                f.path.clone(),
                fs::read(&self.spec.sources[&f.path]).unwrap(),
                false,
            )
        }));
        result
    }
}

#[test]
fn cpu_payload_roundtrips_without_cuda_runtime_files() {
    let fixture = Fixture::cpu();
    let receipt = fixture.pack();
    fs::create_dir(&fixture.extracted()).unwrap();
    extract(&fixture.archive, &receipt, &fixture.extracted()).unwrap();
    assert_eq!(receipt.manifest.backend, Backend::Cpu);
    assert!(fixture.extracted().join("ferrum.exe").is_file());
    assert!(fixture.extracted().join("vcruntime140.dll").is_file());
    assert!(!fixture.extracted().join("cudart64_12.dll").exists());
}

#[test]
fn cpu_payload_rejects_gpu_build_identity_and_cuda_runtime_inputs() {
    let fixture = Fixture::cpu();
    for feature in ["cuda", "metal", "vllm-moe-marlin", "vllm-paged-attn-v2"] {
        let mut manifest = fixture.spec.manifest.clone();
        manifest.cargo_features.push(feature.into());
        assert!(
            validate(&manifest).is_err(),
            "accepted GPU feature {feature}"
        );
    }
    let mut manifest = fixture.spec.manifest.clone();
    manifest.cuda_compute_capability = "89".into();
    assert!(validate(&manifest).is_err());
    let mut manifest = fixture.spec.manifest.clone();
    manifest.files.push(
        Fixture::new()
            .spec
            .manifest
            .files
            .into_iter()
            .find(|file| file.role == Role::CudaRuntime)
            .unwrap(),
    );
    assert!(validate(&manifest).is_err());
    manifest = fixture.spec.manifest.clone();
    manifest.backend = Backend::Cuda;
    assert!(validate(&manifest).is_err());
}

// Reuse the real ZIP producer with structural PE fixtures. These tests verify
// asset/receipt binding and never execute or claim a working installer or GPU.
fn staged_fixture() -> Fixture {
    staged_fixture_backend(Backend::Cuda)
}

fn staged_fixture_backend(backend: Backend) -> Fixture {
    let mut fixture = if backend == Backend::Cpu {
        Fixture::cpu()
    } else {
        Fixture::new()
    };
    let suffix = if backend == Backend::Cpu {
        "cpu"
    } else {
        "cuda-sm89"
    };
    let archive_name = format!("ferrum-windows-x86_64-{suffix}.zip");
    fixture.archive = fixture.root.path().join(&archive_name);
    fixture.receipt = fixture
        .root
        .path()
        .join(format!("{archive_name}.receipt.json"));
    if backend == Backend::Cuda {
        fixture
            .spec
            .manifest
            .cargo_features
            .extend(["vllm-moe-marlin".into(), "vllm-paged-attn-v2".into()]);
    }
    let receipt = fixture.pack();
    let mut inspection = Inspection {
        schema_version: 1,
        status: "not_run".into(),
        scope: "archive_bytes_and_startup_only".into(),
        receipt,
        host_os: "windows".into(),
        host_arch: "x86_64".into(),
        directory: fixture.extracted(),
        startup_environment: BTreeMap::new(),
        startup_directory: None,
        observations: vec![],
        error: None,
    };
    if backend == Backend::Cpu {
        // Synthetic receipt observations exercise evidence binding only; the
        // PE fixture is never executed. Native package QA is a separate flow.
        inspection.status = "passed".into();
        inspection.startup_directory = Some(fixture.extracted());
        inspection.startup_environment = BTreeMap::from([
            ("SystemRoot".into(), "C:\\Windows".into()),
            ("WINDIR".into(), "C:\\Windows".into()),
            ("PATH".into(), "C:\\Windows\\System32".into()),
        ]);
        inspection.observations = [
            vec!["--version"],
            vec!["--help"],
            vec!["run", "--help"],
            vec!["serve", "--help"],
        ]
        .into_iter()
        .map(|args| Observation {
            stdout: if args == ["--version"] {
                "ferrum 2.3.4".into()
            } else {
                "fixture help".into()
            },
            arguments: args.into_iter().map(str::to_owned).collect(),
            exit_code: Some(0),
            stderr: String::new(),
        })
        .collect();
    }
    fs::write(
        fixture.root.path().join("portable.staging-inspection.json"),
        serde_json::to_vec(&inspection).unwrap(),
    )
    .unwrap();
    let launcher = "ferrum-windows-launcher-v1.exe";
    let setup = format!("ferrum-2.3.4-windows-x86_64-{suffix}-setup.exe");
    fs::write(fixture.root.path().join(launcher), pe(false)).unwrap();
    fs::write(
        fixture.root.path().join(&setup),
        b"opaque installer byte identity fixture",
    )
    .unwrap();
    let identity = |name: &str| {
        let path = fixture.root.path().join(name);
        let sha = installation::sha256(&path).unwrap();
        fs::write(
            fixture.root.path().join(format!("{name}.sha256")),
            format!("{sha}  {name}\r\n"),
        )
        .unwrap();
        serde_json::json!({"name":name,"sha256":sha,"size_bytes":fs::metadata(path).unwrap().len()})
    };
    let stage = serde_json::json!({
        "schema_version":1,"version":"2.3.4","candidate_sha":"a".repeat(40),
        "candidate_tag":"v2.3.4-rc.7","staging_label":"run-17-2","workflow_run_id":17,"workflow_run_attempt":2,
        "scope":if backend == Backend::Cpu { "staged_bytes_and_startup" } else { "staged_bytes_only" },"binary_sha256":inspection.receipt.manifest.files[0].sha256,
        "launcher_sha256":installation::sha256(&fixture.root.path().join(launcher)).unwrap(),
        "manifest_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&fixture.spec.manifest).unwrap())),
        "archive":identity(&archive_name),"setup":identity(&setup),"launcher":identity(launcher)
    });
    fs::write(
        fixture.root.path().join("windows-staging.json"),
        serde_json::to_vec(&stage).unwrap(),
    )
    .unwrap();
    fs::write(fixture.root.path().join("launcher-origin.json"), serde_json::to_vec(&serde_json::json!({
        "mode":"first_issuance","source_commit":"a".repeat(40),"sha256":stage["launcher_sha256"],"dependencies":["kernel32.dll"]
    })).unwrap()).unwrap();
    fixture
}

fn accept_staged(
    fixture: &Fixture,
) -> Result<super::super::gate::windows_assets::VerifiedWindows, String> {
    super::super::gate::windows_assets::verify_one(
        fixture.root.path(),
        Backend::Cuda,
        "2.3.4",
        &"a".repeat(40),
        17,
        "v2.3.3",
        "owner/repo",
    )
}

#[test]
fn windows_release_requires_cpu_startup_and_the_shared_stable_launcher() {
    let cuda = staged_fixture();
    let accept = || {
        super::super::gate::windows_assets::verify(
            cuda.root.path(),
            "2.3.4",
            &"a".repeat(40),
            17,
            "v2.3.3",
            "owner/repo",
        )
    };
    assert!(
        accept().is_err(),
        "accepted a release without the CPU package"
    );
    let cpu = staged_fixture_backend(Backend::Cpu);
    let cpu_dir = cuda.root.path().join("cpu");
    fs::create_dir(&cpu_dir).unwrap();
    for entry in fs::read_dir(cpu.root.path()).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            fs::copy(entry.path(), cpu_dir.join(entry.file_name())).unwrap();
        }
    }
    let release = accept().unwrap();
    assert!(release
        .assets
        .iter()
        .any(|asset| asset.name == "ferrum-windows-x86_64-cpu.zip"));
    assert!(release
        .assets
        .iter()
        .any(|asset| asset.name == "ferrum-2.3.4-windows-x86_64-cpu-setup.exe"));
    let launcher: Vec<_> = release
        .assets
        .iter()
        .filter(|asset| asset.name == "ferrum-windows-launcher-v1.exe")
        .collect();
    assert_eq!(launcher.len(), 1);
    let path = cpu_dir.join("portable.staging-inspection.json");
    let mut report: Inspection = read(&path).unwrap();
    report.status = "not_run".into();
    report.observations.clear();
    report.startup_directory = None;
    report.startup_environment.clear();
    fs::write(path, serde_json::to_vec(&report).unwrap()).unwrap();
    assert!(accept()
        .unwrap_err()
        .contains("CPU assets require executed startup"));
}

#[test]
fn windows_staging_accepts_exact_payload_setup_and_stable_launcher_without_runtime_claims() {
    let fixture = staged_fixture();
    let accepted = accept_staged(&fixture).unwrap();
    let names: BTreeSet<_> = accepted.assets.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains("ferrum-windows-x86_64-cuda-sm89.zip"));
    assert!(names.contains("ferrum-2.3.4-windows-x86_64-cuda-sm89-setup.exe"));
    assert!(names.contains("ferrum-windows-launcher-v1.exe"));
    assert!(names.contains("ferrum-windows-x86_64-cuda-sm89.zip.receipt.json"));
    assert_eq!(accepted.attempt, 2);
    assert!(!names.contains("portable.staging-inspection.json"));
    let origin = fixture.root.path().join("launcher-origin.json");
    let launcher =
        installation::sha256(&fixture.root.path().join("ferrum-windows-launcher-v1.exe")).unwrap();
    fs::write(&origin, serde_json::to_vec(&serde_json::json!({"mode":"released",
        "url":"https://github.com/owner/repo/releases/download/v2.3.3/ferrum-windows-launcher-v1.exe",
        "sha256":launcher,"dependencies":["kernel32.dll"]})).unwrap()).unwrap();
    accept_staged(&fixture).unwrap();
}

#[test]
fn windows_staging_rejects_wrong_inputs_corruption_and_invented_startup_success() {
    let fixture = staged_fixture();
    let path = fixture.root.path().join("windows-staging.json");
    let original: serde_json::Value = read(&path).unwrap();
    for (field, value) in [
        ("version", serde_json::json!("2.3.3")),
        ("candidate_sha", serde_json::json!("b".repeat(40))),
        ("workflow_run_id", serde_json::json!(18)),
        ("manifest_sha256", serde_json::json!("c".repeat(64))),
        ("binary_sha256", serde_json::json!("d".repeat(64))),
        ("scope", serde_json::json!("staged_bytes_and_startup")),
    ] {
        let mut changed = original.clone();
        changed[field] = value;
        fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(accept_staged(&fixture).is_err(), "accepted changed {field}");
    }
    fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    for name in [
        "ferrum-windows-x86_64-cuda-sm89.zip",
        "ferrum-windows-launcher-v1.exe",
        "ferrum-2.3.4-windows-x86_64-cuda-sm89-setup.exe",
    ] {
        let path = fixture.root.path().join(name);
        let bytes = fs::read(&path).unwrap();
        let mut corrupt = bytes.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        fs::write(&path, corrupt).unwrap();
        assert!(accept_staged(&fixture).is_err(), "accepted changed {name}");
        fs::remove_file(&path).unwrap();
        assert!(accept_staged(&fixture).is_err(), "accepted missing {name}");
        fs::write(&path, bytes).unwrap();
    }
    let origin_path = fixture.root.path().join("launcher-origin.json");
    let origin: serde_json::Value = read(&origin_path).unwrap();
    for (field, value) in [
        ("source_commit", serde_json::json!("b".repeat(40))),
        ("sha256", serde_json::json!("c".repeat(64))),
        ("dependencies", serde_json::json!(["vcruntime140.dll"])),
        ("dependencies", serde_json::json!([])),
    ] {
        let mut changed = origin.clone();
        changed[field] = value;
        fs::write(&origin_path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(
            accept_staged(&fixture).is_err(),
            "accepted changed origin {field}"
        );
    }
    fs::write(&origin_path, serde_json::to_vec(&origin).unwrap()).unwrap();
    let inspection = fixture.root.path().join("portable.staging-inspection.json");
    let mut report: serde_json::Value = read(&inspection).unwrap();
    report["status"] = serde_json::json!("passed");
    fs::write(&inspection, serde_json::to_vec(&report).unwrap()).unwrap();
    assert!(accept_staged(&fixture).unwrap_err().contains("probes"));
}

#[test]
fn portable_local_build_identity_preserves_provenance_without_a_release_candidate() {
    let fixture = Fixture::new();
    let receipt = fixture.pack();
    assert_eq!(receipt.manifest.build, fixture.spec.manifest.build);
    let wire = serde_json::to_value(&receipt.manifest).unwrap();
    assert!(wire.get("candidate").is_none());
    assert_eq!(wire["build"]["build_id"], "local-native-fixture");
    for (field, invalid) in [
        ("version", "2.3"),
        ("source_commit", "abcdef"),
        ("source_commit", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
        ("build_id", ""),
        ("build_id", "build\nforged"),
    ] {
        let mut changed = wire.clone();
        changed["build"][field] = invalid.into();
        let manifest: Manifest = serde_json::from_value(changed).unwrap();
        assert!(validate(&manifest).is_err(), "accepted {field}={invalid:?}");
    }
    let mut candidate = wire;
    candidate["candidate"] = serde_json::json!({"release_candidate_tag":"v2.3.4-rc.1"});
    assert!(serde_json::from_value::<Manifest>(candidate).is_err());
}

#[tokio::test]
async fn portable_roundtrip_preserves_all_runtime_and_license_bytes_without_claiming_execution() {
    let fixture = Fixture::new();
    let receipt = fixture.pack();
    let output = fixture.root.path().join("inspection.json");
    // Original files need not exist after packaging; the accepted archive is the input.
    fs::remove_dir_all(fixture.root.path().join("inputs")).unwrap();
    inspect(
        &fixture.archive,
        receipt,
        &fixture.extracted(),
        &output,
        true,
    )
    .await
    .unwrap();
    let report: Inspection = read(&output).unwrap();
    assert_eq!(report.status, "not_run");
    assert!(report.observations.is_empty());
    assert_eq!(report.receipt.manifest.backend, Backend::Cuda);
    assert_eq!(report.receipt.manifest.target_triple, TARGET);
    assert_eq!(
        fs::read(fixture.extracted().join(MANIFEST)).unwrap(),
        serde_json::to_vec(&report.receipt.manifest).unwrap()
    );
    for file in &report.receipt.manifest.files {
        verify_file(&fixture.extracted().join(&file.path), file).unwrap();
    }
    assert_eq!(
        fs::read(fixture.extracted().join("licenses/Ferrum.txt")).unwrap(),
        b"literal license\r\n"
    );
    assert!(pack(&fixture.spec, &fixture.archive, &fixture.receipt).is_err());
}

#[test]
fn portable_rejects_linux_or_arm_identity_and_wrong_pe_roles_before_packing() {
    for target in ["x86_64-unknown-linux-gnu", "aarch64-pc-windows-msvc"] {
        let mut fixture = Fixture::new();
        fixture.spec.manifest.target_triple = target.into();
        assert!(pack(&fixture.spec, &fixture.archive, &fixture.receipt).is_err());
        assert!(!fixture.archive.exists());
    }
    for bytes in [b"\x7fELF Linux CUDA".to_vec(), pe(true), {
        let mut bytes = pe(false);
        bytes[68..70].copy_from_slice(&0xaa64u16.to_le_bytes());
        bytes
    }] {
        let mut fixture = Fixture::new();
        fs::write(&fixture.spec.sources["ferrum.exe"], &bytes).unwrap();
        let file = &mut fixture.spec.manifest.files[0];
        file.sha256 = format!("{:x}", Sha256::digest(&bytes));
        file.size_bytes = bytes.len() as u64;
        assert!(pack(&fixture.spec, &fixture.archive, &fixture.receipt).is_err());
        assert!(!fixture.archive.exists());
    }
}

#[test]
fn portable_requires_exact_sources_runtime_licenses_and_case_unique_windows_names() {
    let fixture = Fixture::new();
    for path in [
        "../evil.dll",
        "C:/evil.dll",
        "bad\\evil.dll",
        "bad:stream.dll",
        "CON.dll",
        "nul.txt",
        "x.dll.",
        "x.dll ",
    ] {
        let mut manifest = fixture.spec.manifest.clone();
        manifest.files[1].path = path.into();
        assert!(validate(&manifest).is_err(), "{path}");
    }
    let mut manifest = fixture.spec.manifest.clone();
    manifest.files[2].path = "CUDART64_12.DLL".into();
    assert!(validate(&manifest).is_err());
    manifest = fixture.spec.manifest.clone();
    manifest.files[1].license = Some("licenses/missing.txt".into());
    assert!(validate(&manifest).is_err());
    manifest = fixture.spec.manifest.clone();
    manifest.files[1].path = "nvcuda.dll".into();
    assert!(validate(&manifest).is_err());
    manifest = fixture.spec.manifest.clone();
    manifest.files.remove(2);
    assert!(validate(&manifest).is_err());
    let mut fixture = Fixture::new();
    fixture.spec.sources.remove("vcruntime140.dll");
    assert!(pack(&fixture.spec, &fixture.archive, &fixture.receipt).is_err());
}

#[test]
fn portable_zip_rejects_missing_extra_traversal_symlink_and_changed_dll_entries() {
    for mutation in 0..6 {
        let fixture = Fixture::new();
        let mut receipt = fixture.pack();
        let mut entries = fixture.entries();
        match mutation {
            0 => {
                entries.pop();
            }
            1 => {
                entries.push(("extra.dll".into(), pe(true), false));
            }
            2 => {
                entries[2].0 = "../escape.dll".into();
            }
            3 => {
                entries[2].2 = true;
            }
            4 => {
                entries[2].1[400] ^= 1;
            }
            5 => {
                entries[0].1 = serde_json::to_vec(&{
                    let mut manifest = receipt.manifest.clone();
                    manifest.target_triple = "x86_64-unknown-linux-gnu".into();
                    manifest
                })
                .unwrap();
            }
            _ => unreachable!(),
        }
        fixture.replace_zip(entries, &mut receipt);
        fs::create_dir(fixture.extracted()).unwrap();
        assert!(
            extract(&fixture.archive, &receipt, &fixture.extracted()).is_err(),
            "mutation {mutation}"
        );
        assert!(!fixture.root.path().join("escape.dll").exists());
    }
}

#[test]
fn portable_archive_tampering_is_rejected_before_extraction() {
    let fixture = Fixture::new();
    let receipt = fixture.pack();
    fs::OpenOptions::new()
        .append(true)
        .open(&fixture.archive)
        .unwrap()
        .write_all(b"tamper")
        .unwrap();
    assert!(verify_receipt(&fixture.archive, &receipt).is_err());
}

#[test]
fn portable_rejects_exact_duplicate_hidden_by_zip_name_index() {
    let fixture = Fixture::new();
    let mut receipt = fixture.pack();
    let mut entries = fixture.entries();
    entries.push(("CUDART64_12.DLL".into(), pe(true), false));
    fixture.replace_zip(entries, &mut receipt);
    let mut bytes = fs::read(&fixture.archive).unwrap();
    // Keep all ZIP offsets and lengths valid while making the additional local
    // and central names identical to the already declared DLL.
    let old = b"CUDART64_12.DLL";
    for index in 0..=bytes.len() - old.len() {
        if &bytes[index..index + old.len()] == old {
            bytes[index..index + old.len()].copy_from_slice(b"cudart64_12.dll");
        }
    }
    fs::write(&fixture.archive, &bytes).unwrap();
    receipt.archive_sha256 = format!("{:x}", Sha256::digest(&bytes));
    receipt.archive_size_bytes = bytes.len() as u64;
    let reader = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
    assert_eq!(
        reader.len(),
        receipt.manifest.files.len() + 1,
        "zip collapses the duplicate before our inventory loop"
    );
    fs::create_dir(fixture.extracted()).unwrap();
    let error = extract(&fixture.archive, &receipt, &fixture.extracted()).unwrap_err();
    assert!(error.contains("hidden or duplicate"), "{error}");
    assert!(!fixture.extracted().join("ferrum.exe").exists());
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
#[tokio::test]
async fn portable_startup_cannot_be_claimed_by_a_non_windows_host() {
    let fixture = Fixture::new();
    let receipt = fixture.pack();
    let output = fixture.root.path().join("inspection.json");
    let error = inspect(
        &fixture.archive,
        receipt,
        &fixture.extracted(),
        &output,
        false,
    )
    .await
    .unwrap_err();
    assert!(error.contains("native Windows"));
    let report: Inspection = read(&output).unwrap();
    assert_eq!(report.status, "failed");
    assert!(report.observations.is_empty());
}
