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
                nvcc: tool(r"\\?\C:\CUDA\bin\nvcc.exe"),
                manifest: evidence("toolchain/cuda-static-manifest.json"),
            },
            host_toolchain: NativeOperatorHostToolchainIdentity {
                compiler: tool(r"\\?\C:\MSVC\bin\Hostx64\x64\cl.exe"),
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
    plan_with_source(root, "#include \"marlin.h\"\n")
}

fn plan_with_source(root: &Path, contents: &str) -> (NativeOperatorSourceBuildPlan, PathBuf) {
    let source = root.join("source");
    fs::create_dir_all(source.join("kernels")).unwrap();
    fs::write(source.join("kernels/marlin.cu"), contents).unwrap();
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

#[test]
fn dependency_proof_publication_reopens_and_rejects_damaged_members() {
    let root = tempfile::tempdir().unwrap();
    let (plan, _) = plan(root.path());
    let source = root.path().join("source").canonicalize().unwrap();
    let output = root.path().join("build");
    let cache_entry = root.path().join("cache-entry");
    fs::create_dir(&output).unwrap();
    fs::create_dir(&cache_entry).unwrap();
    let object = output.join("fixture.obj");
    fs::write(&object, b"object bytes bound by the dependency proof").unwrap();
    let object_sha256 = sha256_file(&object).unwrap();
    let cache_key = sha256_bytes(b"dependency-proof-publication-fixture");
    let compiler_depfile = output.join("compiler.raw.d");
    let portable_depfile = output.join("dependency.d");
    let translation_unit = &plan.translation_units[0];
    let closure = &plan.dependency_closures[0];
    let dependencies = std::iter::once(translation_unit.path.clone())
        .chain(closure.headers.iter().map(|header| header.path.clone()))
        .collect::<Vec<_>>();
    fs::write(
        &compiler_depfile,
        serialize_portable_depfile(&object.display().to_string(), &dependencies).unwrap(),
    )
    .unwrap();
    let toolchain_scope = NativeOperatorToolchainDependencyScope {
        by_absolute_path: BTreeMap::new(),
    };
    let validated = validate_translation_unit_depfile(
        &compiler_depfile,
        &portable_depfile,
        &object,
        &source,
        translation_unit,
        closure,
        &toolchain_scope,
    )
    .unwrap();
    publish_object_dependency_proof(
        &cache_entry,
        &cache_key,
        &object_sha256,
        translation_unit,
        closure,
        &validated.compiler_raw,
        &validated.compiler_sha256,
        &validated.portable_raw,
        &validated.portable_sha256,
        &source.display().to_string(),
        &object.display().to_string(),
        &validated.bindings,
        &validated.observed_dependencies,
        &toolchain_scope,
    )
    .unwrap();

    let restored = root.path().join("restored");
    let restored_compiler = restored.join("compiler.raw.d");
    let restored_portable = restored.join("dependency.d");
    let restore = || {
        restore_object_dependency_proof(
            &cache_entry,
            &cache_key,
            &object_sha256,
            closure,
            translation_unit,
            &restored.join("fixture.obj"),
            &restored_compiler,
            &restored_portable,
            &toolchain_scope,
        )
    };
    let proof = restore().unwrap().expect("published proof is reusable");
    assert_eq!(proof.observed_dependencies, validated.observed_dependencies);
    assert_eq!(
        fs::read(&restored_compiler).unwrap(),
        validated.compiler_raw
    );
    assert_eq!(
        fs::read(&restored_portable).unwrap(),
        validated.portable_raw
    );

    for member in ["compiler-dependency.raw.d", "dependency.d", "proof.json"] {
        let path = cache_entry.join("dependency-proof").join(member);
        let original = fs::read(&path).unwrap();
        fs::write(&path, b"damaged cached evidence\n").unwrap();
        fs::remove_file(&restored_compiler).unwrap();
        fs::remove_file(&restored_portable).unwrap();
        assert!(restore().is_err(), "damaged {member} must be rejected");
        assert!(!restored_compiler.exists());
        assert!(!restored_portable.exists());
        fs::write(&path, original).unwrap();
        assert!(restore().unwrap().is_some());
    }
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
    )
    .unwrap();
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
    assert!(compile.argv.iter().any(|arg| arg == "--use-local-env"));
    assert!(!compile
        .argv
        .iter()
        .any(|arg| matches!(arg.as_str(), "-MMD" | "-fPIC" | "-fvisibility=default")));
    assert_eq!(
        receipt.commands.last().unwrap().argv[3],
        platform::basename(compile.object_file.as_ref().unwrap())
    );
    for mutation in 0..6 {
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
            4 => {
                invalid.commands[0]
                    .object_identity
                    .as_mut()
                    .unwrap()
                    .machine = 0x14c;
            }
            _ => invalid.commands[0]
                .argv
                .retain(|arg| arg != "--use-local-env"),
        }
        assert!(
            verify_source_build_receipt_against_plan_portable(&invalid, &path).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn msvc_cache_identity_rejects_objects_from_nvcc_reinitialized_environments() {
    let root = tempfile::tempdir().unwrap();
    let (plan, path) = plan(root.path());
    let mut receipt = receipt(&plan, &path);
    let specs = build_object_cache_specs(
        &plan,
        &architecture_argument(plan.architecture, "sm_89"),
        &receipt.toolchain.as_ref().unwrap().static_identity,
        &receipt.effective_environment,
    )
    .unwrap();
    let signature = specs[0].input_signature();
    let value: serde_json::Value = serde_json::from_str(signature).unwrap();
    assert_eq!(value["msvc_environment_option"], "--use-local-env");
    // This field is appended and omitted for legacy Unix identities. Removing
    // it reconstructs the previous Windows signature with every other input
    // unchanged, including the compiler/SDK and declared INCLUDE order.
    let legacy_signature =
        signature.replace(",\"msvc_environment_option\":\"--use-local-env\"", "");
    assert_ne!(signature, legacy_signature);
    receipt.commands[0].object_cache_key = Some(sha256_bytes(legacy_signature.as_bytes()));
    assert!(verify_source_build_receipt_against_plan_portable(&receipt, &path).is_err());
}

#[test]
fn msvc_nvcc_ccbin_invocation_is_bound_by_receipt_and_cache() {
    let root = tempfile::tempdir().unwrap();
    let (plan, path) = plan(root.path());
    let receipt = receipt(&plan, &path);
    verify_source_build_receipt_against_plan_portable(&receipt, &path).unwrap();
    let identity = &receipt.toolchain.as_ref().unwrap().static_identity;
    assert_eq!(
        identity.host_toolchain.compiler.path,
        r"\\?\C:\MSVC\bin\Hostx64\x64\cl.exe"
    );
    let argument = &receipt.commands[0].argv[7];
    assert_eq!(argument, "C:/MSVC/bin/Hostx64/x64/cl.exe");
    assert!(receipt.effective_environment["PATH"]
        .split(';')
        .any(|path| { Some(path) == platform::parent(argument) }));
    let specs = build_object_cache_specs(
        &plan,
        &receipt.architecture_argument,
        identity,
        &receipt.effective_environment,
    )
    .unwrap();
    let signature = specs[0].input_signature();
    let value: serde_json::Value = serde_json::from_str(signature).unwrap();
    assert_eq!(value["msvc_nvcc_ccbin"], *argument);
    let legacy_signature = signature.replace(
        &format!(
            ",\"msvc_nvcc_ccbin\":{}",
            serde_json::to_string(argument).unwrap()
        ),
        "",
    );
    assert_ne!(signature, legacy_signature);
    let mut legacy_cache = receipt.clone();
    legacy_cache.commands[0].object_cache_key = Some(sha256_bytes(legacy_signature.as_bytes()));
    assert!(verify_source_build_receipt_against_plan_portable(&legacy_cache, &path).is_err());
    for changed in [
        identity.host_toolchain.compiler.path.as_str(),
        "C:/Other/cl.exe",
    ] {
        let mut invalid = receipt.clone();
        invalid.commands[0].argv[7] = changed.to_string();
        assert!(verify_source_build_receipt_against_plan_portable(&invalid, &path).is_err());
    }
    let linux = "/usr/bin/../bin/g++";
    assert_eq!(platform::nvcc_ccbin_argument(linux, false).unwrap(), linux);
}

#[test]
fn msvc_nvcc_program_is_bound_by_receipt_and_cache() {
    let root = tempfile::tempdir().unwrap();
    let (plan, path) = plan(root.path());
    let receipt = receipt(&plan, &path);
    verify_source_build_receipt_against_plan_portable(&receipt, &path).unwrap();
    let identity = &receipt.toolchain.as_ref().unwrap().static_identity;
    assert_eq!(identity.cuda_toolkit.nvcc.path, r"\\?\C:\CUDA\bin\nvcc.exe");
    let program = &receipt.commands[0].argv[0];
    assert_eq!(program, "C:/CUDA/bin/nvcc.exe");
    let specs = build_object_cache_specs(
        &plan,
        &receipt.architecture_argument,
        identity,
        &receipt.effective_environment,
    )
    .unwrap();
    let signature = specs[0].input_signature();
    let value: serde_json::Value = serde_json::from_str(signature).unwrap();
    assert_eq!(value["msvc_nvcc_program"], *program);
    let legacy_signature = signature.replace(
        &format!(
            ",\"msvc_nvcc_program\":{}",
            serde_json::to_string(program).unwrap()
        ),
        "",
    );
    assert_ne!(signature, legacy_signature);
    let mut invalid = receipt.clone();
    invalid.commands[0].object_cache_key = Some(sha256_bytes(legacy_signature.as_bytes()));
    assert!(verify_source_build_receipt_against_plan_portable(&invalid, &path).is_err());

    let previous_inputs = NativeOperatorBuildInputIdentity {
        plan_sha256: &receipt.plan_sha256,
        source_package_sha256: &receipt.source_package.sha256,
        builder_contract_version: NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
        architecture_argument: &receipt.architecture_argument,
        effective_environment: &receipt.effective_environment,
        toolchain: Some(identity),
        msvc_environment_option: Some("--use-local-env"),
        msvc_nvcc_ccbin: Some(receipt.commands[0].argv[7].clone()),
        msvc_nvcc_program: None,
    };
    let mut invalid = receipt.clone();
    invalid.inputs_sha256 = sha256_bytes(&serde_json::to_vec(&previous_inputs).unwrap());
    assert!(verify_source_build_receipt_against_plan_portable(&invalid, &path).is_err());
    for changed in [
        identity.cuda_toolkit.nvcc.path.as_str(),
        "C:/Other/nvcc.exe",
        "C:/CUDA/bin/cl.exe",
    ] {
        let mut invalid = receipt.clone();
        invalid.commands[0].argv[0] = changed.to_string();
        assert!(verify_source_build_receipt_against_plan_portable(&invalid, &path).is_err());
    }
    assert!(platform::nvcc_program("C:/CUDA/bin/cl.exe", true).is_err());
    assert!(platform::nvcc_program("nvcc.exe", true).is_err());
    assert_eq!(
        platform::nvcc_program(r"\\?\UNC\server\CUDA Tools\bin\nvcc.exe", true).unwrap(),
        "//server/CUDA Tools/bin/nvcc.exe"
    );
    let unix = "/opt/cuda/../cuda/bin/nvcc";
    assert_eq!(platform::nvcc_program(unix, false).unwrap(), unix);
}

#[test]
fn unix_cache_inputs_omit_msvc_nvcc_program() {
    let root = tempfile::tempdir().unwrap();
    let (plan, path) = plan(root.path());
    let environment = BTreeMap::new();
    let previous_inputs = NativeOperatorBuildInputIdentity {
        plan_sha256: "plan",
        source_package_sha256: "source",
        builder_contract_version: NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
        architecture_argument: "-arch=sm_89",
        effective_environment: &environment,
        toolchain: None,
        msvc_environment_option: None,
        msvc_nvcc_ccbin: None,
        msvc_nvcc_program: None,
    };
    let previous_bytes = format!(
        "{{\"plan_sha256\":\"plan\",\"source_package_sha256\":\"source\",\"builder_contract_version\":{},\"architecture_argument\":\"-arch=sm_89\",\"effective_environment\":{{}},\"toolchain\":null}}",
        NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION
    );
    assert_eq!(
        serde_json::to_string(&previous_inputs).unwrap(),
        previous_bytes
    );
    assert_eq!(
        build_inputs_sha256("plan", "source", "-arch=sm_89", &environment, None, &path).unwrap(),
        sha256_bytes(previous_bytes.as_bytes())
    );
    let mut identity = toolchain().static_identity;
    identity.cuda_toolkit.canonical_root = "/opt/cuda".to_string();
    identity.cuda_toolkit.invocation_root = "/opt/cuda".to_string();
    identity.cuda_toolkit.nvcc.path = "/opt/cuda/bin/nvcc".to_string();
    identity.host_toolchain.compiler.path = "/usr/bin/g++".to_string();
    identity.host_toolchain.compiler_version = "g++ 12".to_string();
    identity.host_toolchain.target = "x86_64-linux-gnu".to_string();
    identity.host_toolchain.host_abi = None;
    identity.host_toolchain.environment.clear();
    identity.archiver.path = "/usr/bin/ar".to_string();
    validate_static_toolchain_identity(&plan.operator, &identity).unwrap();
    let specs = build_object_cache_specs(&plan, "-arch=sm_89", &identity, &environment).unwrap();
    let signature: serde_json::Value = serde_json::from_str(specs[0].input_signature()).unwrap();
    assert!(signature.get("msvc_nvcc_program").is_none());
}

#[cfg(windows)]
#[test]
fn msvc_nvcc_program_projection_resolves_to_the_recorded_file() {
    let directory = tempfile::Builder::new()
        .prefix("nvcc 中文 tools ")
        .tempdir()
        .unwrap();
    let nvcc = directory.path().join("nvcc.exe");
    fs::write(&nvcc, b"selected NVCC identity").unwrap();
    let identity = tool_file_identity(&nvcc).unwrap();
    assert!(identity.path.starts_with(r"\\?\"));
    platform::validate_nvcc_program_identity(&identity).unwrap();
    let program = platform::nvcc_program(&identity.path, true).unwrap();
    assert!(!program.starts_with(r"\\?\"));
    assert_eq!(tool_file_identity(Path::new(&program)).unwrap(), identity);
    fs::write(&nvcc, b"changed NVCC identity").unwrap();
    assert!(platform::validate_nvcc_program_identity(&identity).is_err());
}

#[cfg(windows)]
#[test]
fn msvc_nvcc_ccbin_projection_resolves_to_the_recorded_file() {
    let directory = tempfile::Builder::new()
        .prefix("nvcc host 中文 ")
        .tempdir()
        .unwrap();
    let compiler = directory.path().join("cl.exe");
    fs::write(&compiler, b"selected compiler identity").unwrap();
    let identity = tool_file_identity(&compiler).unwrap();
    assert!(identity.path.starts_with(r"\\?\"));
    platform::validate_nvcc_ccbin_identity(&identity).unwrap();
    let argument = platform::nvcc_ccbin_argument(&identity.path, true).unwrap();
    assert!(!argument.starts_with(r"\\?\"));
    assert_eq!(
        Path::new(&argument).canonicalize().unwrap(),
        Path::new(&identity.path)
    );
    fs::write(&compiler, b"changed compiler identity").unwrap();
    assert!(platform::validate_nvcc_ccbin_identity(&identity).is_err());
}

#[cfg(windows)]
pub(super) fn configured_nvcc_host_compile() {
    let root = tempfile::Builder::new()
        .prefix("ferrum nvcc host ")
        .tempdir()
        .unwrap();
    let (plan, plan_path) = plan_with_source(root.path(),
        "#include \"marlin.h\"\nextern \"C\" __global__ void ferrum_host_probe(int* values) { values[threadIdx.x] = 7; }\n");
    let toolkit =
        PathBuf::from(std::env::var_os("CUDA_PATH").expect("CUDA_PATH must select a Toolkit"));
    let nvcc = tool_file_identity(&toolkit.join("bin/nvcc.exe")).unwrap();
    let compiler = tool_file_identity(Path::new(
        &std::env::var_os("NVCC_CCBIN").expect("NVCC_CCBIN must select cl.exe"),
    ))
    .unwrap();
    let archiver = tool_file_identity(Path::new(
        &std::env::var_os("FERRUM_MSVC_LIB").expect("FERRUM_MSVC_LIB must select lib.exe"),
    ))
    .unwrap();
    platform::validate_nvcc_ccbin_identity(&compiler).unwrap();
    platform::validate_nvcc_program_identity(&nvcc).unwrap();
    let environment = platform::msvc_environment_for_tools(
        [&nvcc.path, &compiler.path, &archiver.path],
        &platform::capture_msvc_environment().unwrap(),
    )
    .unwrap();
    let output = root.path().join("build");
    for directory in ["objects", "logs", "depfiles"] {
        fs::create_dir_all(output.join(directory)).unwrap();
    }
    let request = NativeOperatorSourceBuildRequest {
        plan_path,
        source_root: root.path().join("source").canonicalize().unwrap(),
        output_dir: output,
        compute_capability: "sm_89".to_string(),
        builder_sha: "0".repeat(40),
        nvcc_path: PathBuf::from(&nvcc.path),
        ccbin_path: PathBuf::from(&compiler.path),
        ar_path: PathBuf::from(&archiver.path),
        cuda_toolkit_root: toolkit,
        nvcc_threads: 1,
        object_cache_dir: root.path().join("cache"),
        plan_only: false,
    };
    // Use the production argv builder from canonical identities. This exercises
    // the actual NVCC program and PATH/-ccbin together, beyond --version probes.
    let commands = build_commands(
        &request,
        &plan,
        &request.source_root,
        &architecture_argument(plan.architecture, &request.compute_capability),
        &request.output_dir.join("objects"),
        &request.output_dir.join("logs"),
        None,
        &environment,
    )
    .unwrap();
    let command = &commands[0];
    let stdout = request.output_dir.join(&command.stdout_log);
    let stderr = request.output_dir.join(&command.stderr_log);
    write_command_stream(
        &stdout,
        "stdout",
        &command.argv,
        b"configured compile probe\n",
    )
    .unwrap();
    write_command_stream(
        &stderr,
        "stderr",
        &command.argv,
        b"configured compile probe\n",
    )
    .unwrap();
    let status = run_logged_command(
        &command.argv,
        &stdout,
        &stderr,
        &command.working_directory,
        &environment,
    )
    .unwrap();
    platform::validate_nvcc_program_identity(&nvcc).unwrap();
    eprintln!(
        "production NVCC host compile: argv={:?} status={status} stdout={} stderr={}",
        command.argv,
        fs::read_to_string(stdout)
            .unwrap()
            .chars()
            .take(8000)
            .collect::<String>(),
        fs::read_to_string(stderr)
            .unwrap()
            .chars()
            .take(8000)
            .collect::<String>()
    );
    if !status.success() {
        for relative in ["include/cuda_runtime.h", "bin/nvcc.profile"] {
            let path = request.cuda_toolkit_root.join(relative);
            eprintln!(
                "NVCC compile diagnostic input: path={} exists={} identity={:?}",
                path.display(),
                path.exists(),
                tool_file_identity(&path)
            );
        }
        let bounded_log = |path: &Path| {
            let mut bytes = Vec::new();
            match fs::File::open(path).and_then(|file| file.take(12001).read_to_end(&mut bytes)) {
                Ok(_) => {
                    let truncated = bytes.len() > 12000;
                    bytes.truncate(12000);
                    format!(
                        "{}{}",
                        String::from_utf8_lossy(&bytes),
                        if truncated { "\n[truncated]" } else { "" }
                    )
                }
                Err(error) => format!("cannot read {}: {error}", path.display()),
            }
        };
        let run_diagnostic = |name: &str, argv: &[String]| {
            let stdout = request.output_dir.join(format!("logs/{name}.stdout.log"));
            let stderr = request.output_dir.join(format!("logs/{name}.stderr.log"));
            let result = (|| {
                write_command_stream(&stdout, "stdout", argv, b"compile failure diagnostic\n")?;
                write_command_stream(&stderr, "stderr", argv, b"compile failure diagnostic\n")?;
                run_logged_command(
                    argv,
                    &stdout,
                    &stderr,
                    &command.working_directory,
                    &environment,
                )
            })();
            eprintln!(
                "NVCC compile diagnostic {name}: argv={argv:?} result={result:?} stdout={} stderr={}",
                bounded_log(&stdout),
                bounded_log(&stderr)
            );
            result
        };
        let diagnosis = (|| -> Result<()> {
            let ordinary = platform::normalize_windows_path(&nvcc.path)?;
            let ordinary_identity = tool_file_identity(Path::new(&ordinary))?;
            if ordinary_identity != nvcc {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "ordinary NVCC path has a different physical identity: {ordinary_identity:?} != {nvcc:?}"
                )));
            }
            eprintln!("NVCC compile diagnostic verified ordinary={ordinary:?} identity={nvcc:?}");
            for (name, program) in [
                ("canonical-dryrun", &nvcc.path),
                ("ordinary-dryrun", &ordinary),
            ] {
                let mut argv = command.argv.clone();
                argv[0] = program.clone();
                argv.push("--dryrun".to_string());
                // A failed dryrun remains diagnostic; still collect the next comparison.
                let _ = run_diagnostic(name, &argv);
            }
            let object = Path::new(command.object_file.as_ref().unwrap());
            let depfile = request
                .output_dir
                .join(command.compiler_depfile.as_ref().unwrap());
            for path in [object, depfile.as_path()] {
                if !path.starts_with(&request.output_dir)
                    || !request.output_dir.starts_with(root.path())
                {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "diagnostic output is outside the probe TempDir: {}",
                        path.display()
                    )));
                }
                match fs::remove_file(path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(source) => {
                        return Err(NativeOperatorBuilderError::Io {
                            path: path.to_path_buf(),
                            source,
                        });
                    }
                }
            }
            let mut argv = command.argv.clone();
            argv[0] = ordinary;
            let comparison = run_diagnostic("ordinary-compile", &argv)?;
            if comparison.success() {
                let object_identity = tool_file_identity(object)?;
                let inspection = ferrum_native_ops::inspect_msvc_object(
                    &fs::read(object).map_err(|source| NativeOperatorBuilderError::Io {
                        path: object.to_path_buf(),
                        source,
                    })?,
                    platform::MSVC_TARGET,
                )
                .map_err(NativeOperatorBuilderError::Invalid)?;
                let depfile_identity = tool_file_identity(&depfile)?;
                let contents = fs::read_to_string(&depfile).map_err(|source| {
                    NativeOperatorBuilderError::Io {
                        path: depfile.clone(),
                        source,
                    }
                })?;
                eprintln!(
                    "ordinary NVCC compile raw evidence: object={object_identity:?} COFF={:?} depfile={depfile_identity:?} contents={:?}",
                    inspection.identity,
                    contents.chars().take(16000).collect::<String>()
                );
                let (target, dependencies) = parse_make_depfile(&contents, &depfile)?;
                let target_matches = platform::same_path(&target, &object.display().to_string());
                let source_matches = dependencies.iter().any(|dependency| {
                    platform::source_relative_path(dependency, &command.working_directory)
                        .is_ok_and(|relative| relative == plan.translation_units[0].path)
                });
                eprintln!(
                    "ordinary NVCC compile evidence: object={object_identity:?} COFF={:?} depfile={depfile_identity:?} target={target:?} dependencies={} target_matches={target_matches} source_matches={source_matches}",
                    inspection.identity,
                    dependencies.len()
                );
                if inspection.identity.machine != 0x8664 || !target_matches || !source_matches {
                    return Err(NativeOperatorBuilderError::Invalid(
                        "ordinary NVCC compile did not produce the expected AMD64 object and source-bound depfile".to_string(),
                    ));
                }
            }
            Ok(())
        })();
        eprintln!("NVCC compile failure diagnosis: {diagnosis:?}");
        // Check both spellings again even when a diagnostic command or inspection failed.
        for path in [
            Ok(nvcc.path.clone()),
            platform::normalize_windows_path(&nvcc.path),
        ] {
            let identity = path.and_then(|path| tool_file_identity(Path::new(&path)));
            let unchanged = identity.as_ref().is_ok_and(|identity| identity == &nvcc);
            eprintln!(
                "NVCC compile diagnostic final identity: unchanged={unchanged} identity={identity:?}"
            );
        }
    }
    assert!(
        status.success(),
        "production NVCC host compile failed: {status}"
    );
    let object = Path::new(command.object_file.as_ref().unwrap());
    let inspection =
        ferrum_native_ops::inspect_msvc_object(&fs::read(object).unwrap(), platform::MSVC_TARGET)
            .unwrap();
    assert_eq!(inspection.identity.machine, 0x8664);
    let depfile = request
        .output_dir
        .join(command.compiler_depfile.as_ref().unwrap());
    let contents = fs::read_to_string(&depfile).unwrap();
    eprintln!(
        "production NVCC compile raw evidence: object={:?} COFF={:?} depfile={:?} contents={:?}",
        tool_file_identity(object),
        inspection.identity,
        tool_file_identity(&depfile),
        contents.chars().take(16000).collect::<String>()
    );
    let (target, dependencies) = parse_make_depfile(&contents, &depfile).unwrap();
    assert!(platform::same_path(&target, &object.display().to_string()));
    assert!(dependencies.iter().any(|dependency| {
        platform::source_relative_path(dependency, &command.working_directory)
            .is_ok_and(|relative| relative == plan.translation_units[0].path)
    }));
    platform::validate_nvcc_ccbin_identity(&compiler).unwrap();
    platform::validate_nvcc_program_identity(&nvcc).unwrap();
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
