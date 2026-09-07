use super::*;

fn evidence(path: &str) -> NativeOperatorEvidenceFile {
    NativeOperatorEvidenceFile {
        path: path.into(),
        sha256: "a".repeat(64),
        size_bytes: 32,
    }
}

fn msvc_toolchain() -> NativeOperatorPackageToolchain {
    let tool = |name| NativeOperatorToolIdentity {
        path: format!("C:/VS/bin/{name}.exe"),
        sha256: "b".repeat(64),
        version: "19.38.33145 x64".into(),
    };
    NativeOperatorPackageToolchain {
        descriptor_compiler: tool("cl"),
        archiver: tool("lib"),
        descriptor_target: platform::MSVC_TARGET.into(),
        host_abi: Some(NativeOperatorHostAbi::for_target(platform::MSVC_TARGET).unwrap()),
        environment: BTreeMap::from([
            ("INCLUDE".into(), "C:/VS/include;C:/SDK/include".into()),
            ("LIB".into(), "C:/VS/lib/x64;C:/SDK/lib/x64".into()),
            ("LIBPATH".into(), "C:/VS/lib/x64".into()),
            ("SystemRoot".into(), "C:/Windows".into()),
            ("TEMP".into(), "C:/Temp".into()),
            ("TMP".into(), "C:/Temp".into()),
            (
                "WindowsSdkVerBinPath".into(),
                "C:/SDK/bin/10.0.22621.0".into(),
            ),
        ]),
    }
}

fn member(name: &str) -> NativeOperatorArchiveMemberEvidence {
    NativeOperatorArchiveMemberEvidence {
        member: name.into(),
        sha256: "c".repeat(64),
        size_bytes: 64,
        object_identity: NativeOperatorObjectIdentity {
            format: NativeOperatorObjectFormat::Coff,
            class_bits: 64,
            endianness: NativeOperatorObjectEndianness::Little,
            machine: 0x8664,
        },
    }
}

fn receipt() -> NativeOperatorPackageReceipt {
    let toolchain = msvc_toolchain();
    let source_members = vec![member("0000_kernel.obj")];
    let descriptor = member("descriptor.obj");
    let artifact = package_artifact_file("test", "abcdef012345", true);
    let commands = [
        (
            &toolchain.descriptor_compiler.path,
            descriptor_compile_args(true),
            "descriptor-compile",
        ),
        (
            &toolchain.archiver.path,
            descriptor_archive_args(&artifact, &source_members, true),
            "descriptor-archive",
        ),
    ]
    .into_iter()
    .map(|(program, args, name)| NativeOperatorPackageCommand {
        argv: std::iter::once(program.clone()).chain(args).collect(),
        working_directory: ".".into(),
        stdout_log: format!("build-logs/{name}.stdout.log"),
        stderr_log: format!("build-logs/{name}.stderr.log"),
        return_code: 0,
        elapsed_ms: 1,
    })
    .collect::<Vec<_>>();
    let source_verification = evidence("build-logs/source-archive-verify.log");
    let final_verification = evidence("build-logs/final-archive-verify.log");
    let mut logs = vec![source_verification.clone(), final_verification.clone()];
    logs.extend(
        commands
            .iter()
            .flat_map(|command| [&command.stdout_log, &command.stderr_log])
            .map(|path| evidence(path)),
    );
    logs.sort_by(|left, right| left.path.cmp(&right.path));
    NativeOperatorPackageReceipt {
        schema_version: NATIVE_OPERATOR_PACKAGE_RECEIPT_SCHEMA_VERSION,
        operator: "test".into(),
        host_abi: toolchain.host_abi.clone(),
        package_spec: evidence("provenance/package.spec.json"),
        g03_catalog: evidence("provenance/g03-provider-catalog.json"),
        abi_contract: evidence("provenance/native-abi-contract.json"),
        source_build_receipt: evidence("provenance/source-build.receipt.json"),
        source_build_plan: evidence("provenance/source-build.plan.json"),
        source_build_inputs: vec![evidence("provenance/toolchain/host-static-manifest.json")],
        source_build_logs: vec![evidence("provenance/build-logs/source.stdout.log")],
        source_archive_sha256: "a".repeat(64),
        source_archive_members: source_members.clone(),
        source_archive_verification: source_verification,
        descriptor_object: descriptor.clone(),
        final_archive_members: source_members.into_iter().chain([descriptor]).collect(),
        final_archive_verification: final_verification,
        manifest_file: "native_operator_manifest.json".into(),
        artifact_file: artifact,
        manifest_sha256: "a".repeat(64),
        binary_sha256: "a".repeat(64),
        g03_catalog_sha256: "a".repeat(64),
        abi_contract_sha256: "a".repeat(64),
        descriptor_export: "ferrum_native_test_abcdef012345_descriptor_v2".into(),
        license_files: vec![evidence("licenses/LICENSE")],
        system_libraries: vec![
            NativeOperatorSystemLibrary::CudaRuntime,
            NativeOperatorSystemLibrary::MsvcRuntime,
        ],
        package_environment: package_build_environment(&toolchain).unwrap(),
        package_toolchain: toolchain,
        package_commands: commands,
        package_build_logs: logs,
    }
}

#[test]
fn windows_package_receipt_binds_target_crt_and_actual_object_kind() {
    let valid = receipt();
    validate_package_receipt(&valid).unwrap();
    let mut missing = valid.clone();
    missing.host_abi = None;
    missing.package_toolchain.host_abi = None;
    assert!(validate_package_receipt(&missing).is_err());
    let mut wrong_machine = valid.clone();
    wrong_machine.source_archive_members[0]
        .object_identity
        .machine = 0xaa64;
    wrong_machine.descriptor_object.object_identity.machine = 0xaa64;
    for object in &mut wrong_machine.final_archive_members {
        object.object_identity.machine = 0xaa64;
    }
    assert!(validate_package_receipt(&wrong_machine).is_err());
    let mut wrong_crt = valid.clone();
    let crt_index = wrong_crt.package_commands[0]
        .argv
        .iter()
        .position(|arg| arg == "/MD")
        .unwrap();
    wrong_crt.package_commands[0].argv[crt_index] = "/MT".into();
    assert!(validate_package_receipt(&wrong_crt).is_err());
    let mut gnu_runtime = valid;
    gnu_runtime.system_libraries = vec![NativeOperatorSystemLibrary::StdCxx];
    assert!(validate_package_receipt(&gnu_runtime).is_err());
}

#[test]
fn windows_package_receipt_rejects_partial_rebuild_and_environment_drift() {
    let valid = receipt();
    let mut omitted_source = valid.clone();
    omitted_source.package_commands[1]
        .argv
        .retain(|arg| arg != "0000_kernel.obj");
    assert!(validate_package_receipt(&omitted_source).is_err());
    let mut changed_source = valid.clone();
    changed_source.final_archive_members[0].sha256 = "d".repeat(64);
    assert!(validate_package_receipt(&changed_source).is_err());
    let mut environment = valid;
    environment
        .package_environment
        .insert("CL".into(), "/MT".into());
    assert!(validate_package_receipt(&environment).is_err());
}

#[test]
fn runtime_projection_preserves_legacy_declarations_and_requires_explicit_windows_abi() {
    let declared = vec![
        NativeOperatorSystemLibrary::CudaRuntime,
        NativeOperatorSystemLibrary::StdCxx,
    ];
    assert_eq!(package_system_libraries(&declared, None).unwrap(), declared);
    let abi = NativeOperatorHostAbi::for_target(platform::MSVC_TARGET).unwrap();
    assert_eq!(
        package_system_libraries(&declared, Some(&abi)).unwrap(),
        [
            NativeOperatorSystemLibrary::CudaRuntime,
            NativeOperatorSystemLibrary::MsvcRuntime
        ]
    );
    assert!(package_system_libraries(&[NativeOperatorSystemLibrary::MsvcRuntime], None).is_err());
    let source =
        render_descriptor_source_for_host("descriptor", "test", "1", "catalog", "abi", true);
    assert!(!source.contains("__attribute__"));
    assert!(source.contains("*descriptor(void)"));
    assert_eq!(
        render_descriptor_source_for_host("descriptor", "test", "1", "catalog", "abi", false),
        render_descriptor_source("descriptor", "test", "1", "catalog", "abi")
    );
}

#[test]
fn legacy_package_toolchain_serialization_does_not_gain_windows_fields() {
    let tool = NativeOperatorToolIdentity {
        path: "/usr/bin/cc".into(),
        version: "cc".into(),
        sha256: "a".repeat(64),
    };
    let legacy = NativeOperatorPackageToolchain {
        descriptor_compiler: tool.clone(),
        archiver: tool,
        descriptor_target: "x86_64-linux-gnu".into(),
        host_abi: None,
        environment: BTreeMap::new(),
    };
    let value = serde_json::to_value(&legacy).unwrap();
    assert!(value.get("host_abi").is_none());
    assert!(value.get("environment").is_none());
    assert_eq!(
        serde_json::from_value::<NativeOperatorPackageToolchain>(value).unwrap(),
        legacy
    );
}

#[test]
fn archive_members_reject_paths_duplicates_and_unbound_coff() {
    for name in [
        "../kernel.obj",
        "C:\\build\\kernel.obj",
        "nested/kernel.obj",
        "kernel.obj:stream",
    ] {
        assert!(validate_archive_member_evidence("test", &[member(name)]).is_err());
    }
    assert!(validate_archive_member_evidence(
        "test",
        &[member("kernel.obj"), member("kernel.obj")]
    )
    .is_err());
    assert!(validate_package_member_host(&[member("kernel.obj")], None).is_err());
    assert_eq!(
        portable_file_name(r"C:\build\kernel.obj").unwrap(),
        "kernel.obj"
    );
}
