use super::*;

fn host_environment() -> BTreeMap<String, String> {
    [
        ("INCLUDE", "C:/MSVC/include;C:/SDK/include"),
        ("LIB", "C:/MSVC/lib/x64;C:/SDK/lib/x64"),
        ("LIBPATH", "C:/MSVC/lib/x64"),
        ("SystemRoot", "C:/Windows"),
        ("TEMP", "C:/Temp"),
        ("TMP", "C:/Temp"),
        ("WindowsSdkVerBinPath", "C:/SDK/bin/10.0.22621.0/"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect()
}

fn tool(path: &str) -> NativeOperatorToolFileIdentity {
    NativeOperatorToolFileIdentity {
        path: path.to_string(),
        sha256: "a".repeat(64),
        size_bytes: 128,
    }
}

fn evidence(path: &str) -> NativeOperatorEvidenceFile {
    NativeOperatorEvidenceFile {
        path: path.to_string(),
        sha256: "b".repeat(64),
        size_bytes: 128,
    }
}

fn toolchain() -> NativeOperatorSourceBuildToolchain {
    NativeOperatorSourceBuildToolchain {
        static_identity: NativeOperatorSourceBuildStaticToolchain {
            backend: NativeOperatorBackend::Cuda,
            compiler_driver: NativeOperatorSourceCompilerDriver::CudaNvcc,
            cuda_toolkit: NativeOperatorCudaToolkitIdentity {
                canonical_root: "C:/CUDA".to_string(),
                invocation_root: "C:/CUDA".to_string(),
                release_version: "12.4.0".to_string(),
                nvcc: tool("C:/CUDA/bin/nvcc.exe"),
                manifest: evidence("toolchain/cuda-static-manifest.json"),
            },
            host_toolchain: NativeOperatorHostToolchainIdentity {
                compiler: tool("C:/MSVC/bin/Hostx64/x64/cl.exe"),
                compiler_version: "MSVC 1938 full 193833130".to_string(),
                target: platform::MSVC_TARGET.to_string(),
                host_abi: Some(NativeOperatorHostAbi::for_target(platform::MSVC_TARGET).unwrap()),
                environment: host_environment(),
                manifest: evidence("toolchain/host-static-manifest.json"),
            },
            archiver: tool("C:/MSVC/bin/Hostx64/x64/lib.exe"),
        },
        miss_probe: None,
    }
}

fn plan(root: &Path) -> (NativeOperatorSourceBuildPlan, PathBuf) {
    let source = root.join("source");
    fs::create_dir_all(source.join("kernels")).unwrap();
    fs::write(source.join("kernels/marlin.cu"), "#include \"marlin.h\"\n").unwrap();
    fs::write(source.join("kernels/marlin.h"), "void fixture();\n").unwrap();
    let definition = NativeOperatorSourceDefinition {
        schema_version: NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION,
        operator: CudaNativeBuildUnit::Marlin.artifact_operator().to_string(),
        source_package_kind: "ferrum-native-source-bundle".to_string(),
        source_package_revision: "fixture".to_string(),
        upstream_sources: vec![NativeOperatorUpstreamSource {
            repository: "https://example.invalid/source".to_string(),
            revision: "fixture".to_string(),
            license: "Apache-2.0".to_string(),
        }],
        translation_units: vec!["kernels/marlin.cu".to_string()],
        headers: vec!["kernels/marlin.h".to_string()],
        dependency_closures: vec![NativeOperatorTranslationUnitDependencies {
            translation_unit: "kernels/marlin.cu".to_string(),
            headers: vec!["kernels/marlin.h".to_string()],
        }],
        include_dirs: vec!["kernels".to_string()],
        defines: vec![],
        nvcc_policy: NativeOperatorNvccPolicy {
            cpp_standard: NativeOperatorCppStandard::Cpp17,
            optimization: NativeOperatorOptimization::O3,
            use_fast_math: true,
            relaxed_constexpr: true,
            extended_lambda: true,
            host_position_independent_code: true,
            host_default_visibility: true,
        },
        architecture: NativeOperatorCudaArchitecture::DeviceComputeCapability,
        archive_file: "libmarlin.a".to_string(),
    };
    let definition_path = root.join("definition.json");
    let plan_path = root.join("plan.json");
    write_json(&definition_path, &definition).unwrap();
    (
        lock_native_operator_source_definition(&definition_path, &source, &plan_path).unwrap(),
        plan_path,
    )
}

fn receipt(
    plan: &NativeOperatorSourceBuildPlan,
    plan_path: &Path,
) -> NativeOperatorSourceBuildReceipt {
    let mut toolchain = toolchain();
    let request = NativeOperatorSourceBuildRequest {
        plan_path: plan_path.to_path_buf(),
        source_root: PathBuf::from("C:/source"),
        output_dir: PathBuf::from("C:/build"),
        compute_capability: "sm_89".to_string(),
        builder_sha: "c".repeat(40),
        nvcc_path: PathBuf::from("C:/CUDA/bin/nvcc.exe"),
        ccbin_path: PathBuf::from("C:/MSVC/bin/Hostx64/x64/cl.exe"),
        ar_path: PathBuf::from("C:/MSVC/bin/Hostx64/x64/lib.exe"),
        cuda_toolkit_root: PathBuf::from("C:/CUDA"),
        nvcc_threads: 4,
        object_cache_dir: PathBuf::from("C:/cache"),
        plan_only: false,
    };
    let environment = effective_build_environment(&request, Some(&toolchain)).unwrap();
    let architecture = architecture_argument(plan.architecture, "sm_89");
    let mut commands = build_commands(
        &request,
        plan,
        &request.source_root,
        &architecture,
        &request.output_dir.join("objects"),
        &request.output_dir.join("logs"),
        Some(&toolchain),
        &environment,
    );
    let specs = build_object_cache_specs(
        plan,
        &architecture,
        &toolchain.static_identity,
        &environment,
    )
    .unwrap();
    for (index, command) in commands
        .iter_mut()
        .take(plan.translation_units.len())
        .enumerate()
    {
        command.object_cache_key = Some(specs[index].input_signature_sha256().to_string());
        command.object_cache_status = Some(NativeOperatorSourceObjectCacheStatus::Published);
        command.object_sha256 = Some("d".repeat(64));
        command.object_size_bytes = Some(128);
        command.object_identity = Some(NativeOperatorObjectIdentity {
            format: NativeOperatorObjectFormat::Coff,
            class_bits: 64,
            endianness: NativeOperatorObjectEndianness::Little,
            machine: 0x8664,
        });
        command.dependency_validation = Some(NativeOperatorDependencyValidation::Depfile);
        command.compiler_depfile_sha256 = Some("e".repeat(64));
        command.depfile_sha256 = Some("f".repeat(64));
        command.depfile_producer_working_directory = Some(command.working_directory.clone());
        command.depfile_producer_object_file = command.object_file.clone();
        command.observed_dependencies = expected_source_dependencies(
            &plan.translation_units[index],
            &plan.dependency_closures[index],
        )
        .into_iter()
        .collect();
        command.depfile_bindings = command
            .observed_dependencies
            .iter()
            .map(|dependency| NativeOperatorDepfileDependencyBinding {
                producer_path: dependency.path.clone(),
                portable_path: dependency.path.clone(),
                dependency: dependency.clone(),
            })
            .collect();
        command.compiler_executed = true;
        command.return_code = Some(0);
    }
    let compiled = plan
        .translation_units
        .iter()
        .map(|unit| unit.path.clone())
        .collect::<Vec<_>>();
    toolchain.miss_probe = Some(NativeOperatorSourceBuildToolchainProbe {
        nvcc_version: "Cuda compilation tools, release 12.4, V12.4.127".to_string(),
        host_compiler_version: toolchain
            .static_identity
            .host_toolchain
            .compiler_version
            .clone(),
        host_target: platform::MSVC_TARGET.to_string(),
        archiver_version: "Microsoft Library Manager 14.38".to_string(),
        probed_for_misses: compiled.clone(),
    });
    let plan_sha256 = sha256_file(plan_path).unwrap();
    let inputs_sha256 = build_inputs_sha256(
        &plan_sha256,
        &plan.source_package.sha256,
        &architecture,
        &environment,
        Some(&toolchain.static_identity),
        plan_path,
    )
    .unwrap();
    NativeOperatorSourceBuildReceipt {
        schema_version: NATIVE_OPERATOR_SOURCE_BUILD_RECEIPT_SCHEMA_VERSION,
        status: NativeOperatorSourceBuildStatus::Pass,
        operator: plan.operator.clone(),
        plan_only: false,
        plan_sha256,
        source_package: plan.source_package.clone(),
        builder_sha: request.builder_sha,
        compute_capability: request.compute_capability,
        architecture_argument: architecture,
        nvcc_threads: request.nvcc_threads,
        object_cache_root: request.object_cache_dir.display().to_string(),
        toolchain: Some(toolchain),
        effective_environment: environment,
        inputs_sha256,
        commands,
        compiled_translation_units: compiled,
        cache_hit_translation_units: vec![],
        archive_file: Some("marlin.lib".to_string()),
        archive_sha256: Some("a".repeat(64)),
        started_unix_ms: 0,
        elapsed_ms: 0,
        failure_class: None,
    }
}

#[test]
fn msvc_receipt_replays_native_commands_without_the_build_machine() {
    let root = tempfile::tempdir().unwrap();
    let (plan, path) = plan(root.path());
    let receipt = receipt(&plan, &path);
    verify_source_build_receipt_against_plan_portable(&receipt, &path).unwrap();
    let compile = &receipt.commands[0];
    assert!(compile.object_file.as_ref().unwrap().ends_with(".obj"));
    assert!(compile.argv.iter().any(|arg| arg == "-MD"));
    assert!(compile.argv.iter().any(|arg| arg == "/MD,/EHsc,/bigobj"));
    assert!(!compile
        .argv
        .iter()
        .any(|arg| matches!(arg.as_str(), "-MMD" | "-fPIC" | "-fvisibility=default")));
    assert_eq!(
        receipt.commands.last().unwrap().argv[3],
        platform::basename(compile.object_file.as_ref().unwrap())
    );
    for mutation in 0..5 {
        let mut invalid = receipt.clone();
        match mutation {
            0 => invalid.commands[0]
                .argv
                .iter_mut()
                .find(|arg| arg.as_str() == "/MD,/EHsc,/bigobj")
                .unwrap()
                .replace_range(.., "/MT"),
            1 => {
                invalid
                    .toolchain
                    .as_mut()
                    .unwrap()
                    .static_identity
                    .host_toolchain
                    .host_abi = None;
            }
            2 => {
                invalid
                    .effective_environment
                    .insert("CL".to_string(), "/DUNRECORDED=1".to_string());
            }
            3 => {
                invalid.commands.last_mut().unwrap().argv[3] = "C:/other/member.obj".to_string();
            }
            _ => {
                invalid.commands[0]
                    .object_identity
                    .as_mut()
                    .unwrap()
                    .machine = 0x14c;
            }
        }
        assert!(
            verify_source_build_receipt_against_plan_portable(&invalid, &path).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn legacy_host_identity_serialization_stays_unchanged() {
    let legacy = serde_json::json!({"compiler":{"path":"/usr/bin/g++","sha256":"a".repeat(64),"size_bytes":128},"compiler_version":"g++ 12","target":"x86_64-linux-gnu","manifest":{"path":"toolchain/host-static-manifest.json","sha256":"b".repeat(64),"size_bytes":128}});
    let identity: NativeOperatorHostToolchainIdentity =
        serde_json::from_value(legacy.clone()).unwrap();
    assert!(identity.host_abi.is_none());
    assert!(identity.environment.is_empty());
    assert_eq!(serde_json::to_value(identity).unwrap(), legacy);
}

#[test]
fn msvc_environment_rejects_untracked_flags_and_mixed_tool_installations() {
    let environment = host_environment();
    let tools = [
        "C:/MSVC/bin/Hostx64/x64/cl.exe",
        "C:/MSVC/bin/Hostx64/x64/lib.exe",
    ];
    let effective = platform::msvc_environment_for_package_tools(tools, &environment).unwrap();
    assert!(!effective["PATH"].contains("/usr/bin"));
    assert!(!effective.contains_key("CL"));
    let mut invalid = environment;
    invalid.insert("CL".to_string(), "/MT".to_string());
    assert!(platform::msvc_environment_for_package_tools(tools, &invalid).is_err());
    invalid.remove("CL");
    assert!(
        platform::msvc_environment_for_package_tools([tools[0], "C:/Other/lib.exe"], &invalid)
            .is_err()
    );
    invalid.insert("INCLUDE".to_string(), "relative/include".to_string());
    assert!(platform::msvc_environment_for_package_tools(tools, &invalid).is_err());
}
