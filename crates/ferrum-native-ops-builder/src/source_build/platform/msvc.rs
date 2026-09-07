//! Probe the selected MSVC/SDK installation using its real C++ preprocessor.

use super::*;

const PROBE: &[u8] = b"#include <stddef.h>\n#include <windows.h>\nFERRUM_MSVC_VERSION _MSC_VER\nFERRUM_MSVC_FULL_VERSION _MSC_FULL_VER\nFERRUM_MSVC_X64 _M_X64\nFERRUM_MSVC_DYNAMIC_CRT _DLL\n";

pub(crate) fn probe_output(
    compiler: &Path,
    environment: &BTreeMap<String, String>,
    version_details: bool,
) -> Result<std::process::Output> {
    let temporary_root = environment.get("TEMP").ok_or_else(|| {
        NativeOperatorBuilderError::Invalid("MSVC probe is missing TEMP".to_string())
    })?;
    let directory =
        tempfile::tempdir_in(temporary_root).map_err(|source| NativeOperatorBuilderError::Io {
            path: PathBuf::from(temporary_root),
            source,
        })?;
    let source_path = directory.path().join("ferrum-host-probe.cpp");
    fs::write(&source_path, PROBE).map_err(|source| NativeOperatorBuilderError::Io {
        path: source_path.clone(),
        source,
    })?;
    let mut command = Command::new(compiler);
    command.args(["/nologo", "/EP", "/TP", "/MD"]);
    if version_details {
        command.arg("/Bv");
    }
    let output = command
        .arg("ferrum-host-probe.cpp")
        .current_dir(directory.path())
        .env_clear()
        .envs(environment)
        .output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: compiler.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "MSVC preprocessing probe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    validate_probe(&output.stdout)?;
    Ok(output)
}

fn probe_number(text: &str, marker: &str) -> Result<u32> {
    text.lines()
        .filter_map(|line| line.trim().strip_prefix(marker))
        .find_map(|suffix| suffix.trim().parse().ok())
        .ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!("MSVC compiler did not establish {marker}"))
        })
}

fn validate_probe(stdout: &[u8]) -> Result<(u32, u32)> {
    let text = String::from_utf8_lossy(stdout);
    let version = probe_number(&text, "FERRUM_MSVC_VERSION ")?;
    let full_version = probe_number(&text, "FERRUM_MSVC_FULL_VERSION ")?;
    if version < 1900
        || full_version / 100_000 != version
        || probe_number(&text, "FERRUM_MSVC_X64 ")? != 100
        || probe_number(&text, "FERRUM_MSVC_DYNAMIC_CRT ")? != 1
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "native Windows build requires real x64 MSVC with /MD dynamic CRT".to_string(),
        ));
    }
    Ok((version, full_version))
}

pub(crate) fn probe_manifest(
    compiler: &NativeOperatorToolFileIdentity,
    environment: &BTreeMap<String, String>,
) -> Result<NativeOperatorHostToolchainManifest> {
    let include = probe_output(Path::new(&compiler.path), environment, false)?;
    let driver = probe_output(Path::new(&compiler.path), environment, true)?;
    let (version, full_version) = validate_probe(&include.stdout)?;
    let mut recorded = BTreeMap::new();
    for key in MSVC_ENVIRONMENT_KEYS {
        recorded.insert(
            (*key).to_string(),
            environment
                .get(*key)
                .ok_or_else(|| NativeOperatorBuilderError::Invalid(format!("missing MSVC {key}")))?
                .clone(),
        );
    }
    validate_msvc_environment(&recorded)?;
    let include_roots = environment_roots(&recorded, &["INCLUDE"])?;
    let mut discovery_roots = environment_roots(&recorded, &["LIB", "LIBPATH"])?;
    let compiler_parent = Path::new(&compiler.path).parent().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid("MSVC compiler has no parent".to_string())
    })?;
    let sdk_bin = Path::new(&recorded["WindowsSdkVerBinPath"]).join("x64");
    let mut executable_inputs = vec![compiler.clone()];
    for name in ["lib.exe", "link.exe"] {
        executable_inputs.push(tool_file_identity(&compiler_parent.join(name))?);
    }
    for name in ["rc.exe", "mt.exe"] {
        executable_inputs.push(tool_file_identity(&sdk_bin.join(name))?);
    }
    executable_inputs.sort_by(|left, right| left.path.cmp(&right.path));
    discovery_roots.extend(
        executable_inputs
            .iter()
            .filter_map(|tool| Path::new(&tool.path).parent())
            .map(|path| path.display().to_string()),
    );
    discovery_roots.sort();
    discovery_roots.dedup();
    let scopes = include_roots
        .iter()
        .chain(&discovery_roots)
        .cloned()
        .collect::<Vec<_>>();
    let manifest = NativeOperatorHostToolchainManifest {
        schema_version: NATIVE_OPERATOR_HOST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION,
        compiler: compiler.clone(),
        compiler_version: format!("MSVC {version} full {full_version}"),
        target: MSVC_TARGET.to_string(),
        host_abi: Some(
            NativeOperatorHostAbi::for_target(MSVC_TARGET)
                .map_err(NativeOperatorBuilderError::Invalid)?,
        ),
        environment: recorded,
        executable_inputs,
        include_roots,
        include_probe_sha256: compiler_probe_sha256(&include),
        driver_probe_sha256: compiler_probe_sha256(&driver),
        discovery_roots,
        files: collect_host_toolchain_scope_files(&scopes)?,
    };
    validate_host_toolchain_manifest("<msvc-probe>", &manifest)?;
    Ok(manifest)
}

fn environment_roots(environment: &BTreeMap<String, String>, keys: &[&str]) -> Result<Vec<String>> {
    let mut roots = Vec::new();
    for key in keys {
        for value in environment[*key].split(';').filter(|part| !part.is_empty()) {
            let root = Path::new(value).canonicalize().map_err(|source| {
                NativeOperatorBuilderError::Io {
                    path: PathBuf::from(value),
                    source,
                }
            })?;
            if !root.is_dir() {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "MSVC {key} search root is not a directory: {value}"
                )));
            }
            // Bind the spelling searched by the compiler. Canonicalization above
            // validates the directory; the inventory separately records each
            // file's resolved path and bytes without replacing its logical root.
            let root = normalize_windows_path(value)?;
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
    }
    Ok(roots)
}

pub(crate) fn validate_manifest_roots(
    manifest: &NativeOperatorHostToolchainManifest,
) -> Result<()> {
    let environment_paths = |keys: &[&str]| -> Result<Vec<String>> {
        let mut seen = BTreeSet::new();
        keys.iter()
            .flat_map(|key| {
                manifest.environment[*key]
                    .split(';')
                    .filter(|part| !part.is_empty())
            })
            .map(normalize_windows_path)
            .filter(|path| path.as_ref().map_or(true, |path| seen.insert(path.clone())))
            .collect()
    };
    let includes = manifest
        .include_roots
        .iter()
        .map(|path| normalize_windows_path(path))
        .collect::<Result<Vec<_>>>()?;
    let discovered = manifest
        .discovery_roots
        .iter()
        .map(|path| normalize_windows_path(path))
        .collect::<Result<BTreeSet<_>>>()?;
    let mut expected = environment_paths(&["LIB", "LIBPATH"])?
        .into_iter()
        .collect::<BTreeSet<_>>();
    expected.extend(
        manifest
            .executable_inputs
            .iter()
            .map(|tool| {
                parent(&tool.path)
                    .ok_or_else(|| {
                        NativeOperatorBuilderError::Invalid("MSVC tool has no parent".to_string())
                    })
                    .and_then(normalize_windows_path)
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let expected_includes = environment_paths(&["INCLUDE"])?;
    if includes != expected_includes {
        let index = includes
            .iter()
            .zip(&expected_includes)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(includes.len().min(expected_includes.len()));
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "MSVC manifest INCLUDE search roots differ at index {index}: expected={:?} recorded={:?}",
            expected_includes.get(index),
            includes.get(index)
        )));
    }
    if discovered != expected {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "MSVC manifest LIB/LIBPATH and tool discovery roots differ: first_missing={:?} first_extra={:?}",
            expected.difference(&discovered).next(),
            discovered.difference(&expected).next()
        )));
    }
    Ok(())
}

pub(crate) fn target(compiler: &Path) -> Result<String> {
    let captured = capture_msvc_environment()?;
    let compiler = compiler.to_str().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid("MSVC compiler path is not UTF-8".to_string())
    })?;
    let compiler_parent = parent(compiler).ok_or_else(|| {
        NativeOperatorBuilderError::Invalid("MSVC compiler has no parent".to_string())
    })?;
    // The target probe needs only cl.exe and the selected SDK, not a CUDA installation.
    let mut environment = captured;
    environment.insert(
        "PATH".to_string(),
        format!(
            "{compiler_parent};{}/x64;{}/System32",
            environment["WindowsSdkVerBinPath"], environment["SystemRoot"]
        ),
    );
    environment.insert("VSLANG".to_string(), "1033".to_string());
    probe_output(Path::new(compiler), &environment, false)?;
    Ok(MSVC_TARGET.to_string())
}

fn tool_version_command(path: &Path) -> Result<Command> {
    let mut command = Command::new(path);
    command.arg(
        if basename(&path.display().to_string()).eq_ignore_ascii_case("nvcc.exe") {
            "--version"
        } else {
            "/?"
        },
    );
    command.env_clear().env("VSLANG", "1033");
    for key in ["SystemRoot", "TEMP", "TMP"] {
        let value = std::env::var(key).map_err(|_| {
            NativeOperatorBuilderError::Invalid(format!(
                "Windows tool probe is missing {key}: program={:?} args={:?} status=not_started stdout=\"\" stderr=\"\"",
                command.get_program(),
                command.get_args().collect::<Vec<_>>()
            ))
        })?;
        command.env(key, value);
    }
    Ok(command)
}

fn bounded_probe_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut characters = text.chars();
    let mut bounded: String = characters.by_ref().take(4000).collect();
    if characters.next().is_some() {
        bounded.push_str(" [output truncated]");
    }
    bounded
}

fn probe_output_diagnostic(command: &Command, output: &std::process::Output) -> String {
    format!(
        "program={:?} args={:?} status={} code={:?} stdout={:?} stderr={:?}",
        command.get_program(),
        command.get_args().collect::<Vec<_>>(),
        output.status,
        output.status.code(),
        bounded_probe_text(&output.stdout),
        bounded_probe_text(&output.stderr)
    )
}

pub(crate) fn tool_version(path: &Path) -> Result<String> {
    let mut command = tool_version_command(path)?;
    let output = command
        .output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() || text.trim().is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "Windows tool version probe failed: {}",
            probe_output_diagnostic(&command, &output)
        )));
    }
    Ok(text.trim().chars().take(4000).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_version_failure_diagnostic_keeps_status_arguments_and_bounded_channels() {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt;

        #[cfg(unix)]
        let status = std::process::ExitStatus::from_raw(1 << 8);
        #[cfg(windows)]
        let status = std::process::ExitStatus::from_raw(1);

        let mut command = Command::new("nvcc.exe");
        command.arg("--version");
        let output = std::process::Output {
            status,
            stdout: b"compiler version output\n".to_vec(),
            stderr: format!("missing dependency: {}unbounded-tail", "x".repeat(4100)).into_bytes(),
        };
        let diagnostic = probe_output_diagnostic(&command, &output);
        for expected in [
            "program=\"nvcc.exe\"",
            "args=[\"--version\"]",
            "code=Some(1)",
            "stdout=\"compiler version output\\n\"",
            "stderr=\"missing dependency:",
            "[output truncated]",
        ] {
            assert!(diagnostic.contains(expected), "{diagnostic}");
        }
        assert!(!diagnostic.contains("unbounded-tail"));
        assert_eq!(bounded_probe_text(b"short stderr"), "short stderr");
        assert_eq!(
            bounded_probe_text("界".repeat(4001).as_bytes()),
            format!("{} [output truncated]", "界".repeat(4000))
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires configured CUDA Toolkit"]
    fn configured_nvcc_version_probe() {
        let toolkit = std::env::var_os("CUDA_PATH").expect("CUDA_PATH must select a real Toolkit");
        let input_path = PathBuf::from(toolkit).join("bin").join("nvcc.exe");
        let canonical_path = input_path
            .canonicalize()
            .unwrap_or_else(|error| panic!("resolve selected NVCC {:?}: {error}", input_path));
        match tool_version(&canonical_path) {
            Ok(version) => {
                assert!(!version.is_empty());
                eprintln!(
                    "production NVCC version probe succeeded: input={input_path:?} canonical={canonical_path:?} version={version:?}"
                );
            }
            Err(error) => {
                eprintln!("production NVCC version probe failed: {error}");
                // Compare only the selected executable's two path spellings and
                // the current versus production environment. No environment
                // values are printed or changed in the parent test process.
                for (path_label, path) in [
                    ("input", input_path.as_path()),
                    ("canonical", canonical_path.as_path()),
                ] {
                    for isolated in [false, true] {
                        let environment_label = if isolated { "production" } else { "inherited" };
                        let command = if isolated {
                            tool_version_command(path)
                        } else {
                            let mut command = Command::new(path);
                            command.arg("--version");
                            Ok(command)
                        };
                        let label = format!(
                            "NVCC comparison path={path_label} environment={environment_label}"
                        );
                        match command {
                            Ok(mut command) => match command.output() {
                                Ok(output) => eprintln!(
                                    "{label}: {}",
                                    probe_output_diagnostic(&command, &output)
                                ),
                                Err(error) => eprintln!(
                                    "{label}: program={:?} args={:?} status=not_started stdout=\"\" stderr=\"\" error={error}",
                                    command.get_program(),
                                    command.get_args().collect::<Vec<_>>()
                                ),
                            },
                            Err(error) => eprintln!(
                                "{label}: program={path:?} args=[\"--version\"] status=not_started stdout=\"\" stderr=\"\" error={error}"
                            ),
                        }
                    }
                }
                panic!("selected NVCC must pass the production version probe: {error}");
            }
        }
    }

    fn manifest_fixture() -> NativeOperatorHostToolchainManifest {
        let compiler = NativeOperatorToolFileIdentity {
            path: "C:/MSVC/bin/cl.exe".to_string(),
            sha256: "a".repeat(64),
            size_bytes: 128,
        };
        NativeOperatorHostToolchainManifest {
            schema_version: NATIVE_OPERATOR_HOST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION,
            compiler: compiler.clone(),
            compiler_version: "MSVC 1938 full 193833130".to_string(),
            target: MSVC_TARGET.to_string(),
            host_abi: Some(NativeOperatorHostAbi::for_target(MSVC_TARGET).unwrap()),
            environment: [
                ("INCLUDE", "C:/MSVC/include;C:/SDK/include;C:/MSVC/include"),
                ("LIB", "C:/MSVC/lib"),
                ("LIBPATH", "C:/MSVC/lib"),
                ("SystemRoot", "C:/Windows"),
                ("TEMP", "C:/Temp"),
                ("TMP", "C:/Temp"),
                ("WindowsSdkVerBinPath", "C:/SDK/bin/10.0.22621.0"),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
            executable_inputs: vec![compiler],
            include_roots: vec!["C:/MSVC/include".to_string(), "C:/SDK/include".to_string()],
            include_probe_sha256: "b".repeat(64),
            driver_probe_sha256: "c".repeat(64),
            discovery_roots: vec!["C:/MSVC/bin".to_string(), "C:/MSVC/lib".to_string()],
            files: vec![NativeOperatorHostToolchainFileIdentity {
                logical_path: "C:/MSVC/include/stddef.h".to_string(),
                resolved_path: "C:/MSVC/include/stddef.h".to_string(),
                sha256: "d".repeat(64),
                size_bytes: 128,
            }],
        }
    }

    #[test]
    fn manifest_include_order_preserves_first_search_priority() {
        let mut manifest = manifest_fixture();
        validate_host_toolchain_manifest("ordered-include", &manifest).unwrap();
        manifest.include_roots.swap(0, 1);
        assert!(validate_host_toolchain_manifest("reordered-include", &manifest).is_err());
        manifest.include_roots.swap(0, 1);
        manifest.environment.insert(
            "INCLUDE".to_string(),
            "C:/SDK/include;C:/MSVC/include".to_string(),
        );
        assert!(validate_host_toolchain_manifest("changed-search-priority", &manifest).is_err());
    }

    #[test]
    fn manifest_search_scope_validation_rejects_missing_extra_and_changed_roots() {
        let manifest = manifest_fixture();
        let mut missing_include = manifest.clone();
        missing_include.include_roots.pop();
        let error = validate_manifest_roots(&missing_include)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("INCLUDE search roots differ at index 1"),
            "{error}"
        );

        let mut changed_spelling = manifest.clone();
        changed_spelling.include_roots[0] = "C:/msvc/include".to_string();
        assert!(validate_manifest_roots(&changed_spelling).is_err());

        let mut missing_library = manifest.clone();
        missing_library
            .discovery_roots
            .retain(|root| root != "C:/MSVC/lib");
        let error = validate_manifest_roots(&missing_library)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("first_missing=Some(\"C:/MSVC/lib\")"),
            "{error}"
        );

        let mut extra_library = manifest;
        extra_library
            .discovery_roots
            .push("C:/Undeclared/lib".to_string());
        let error = validate_manifest_roots(&extra_library)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("first_extra=Some(\"C:/Undeclared/lib\")"),
            "{error}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn environment_scope_inventory_preserves_windows_search_spelling() {
        let workspace = tempfile::tempdir().unwrap();
        let include = workspace.path().join("SDK-Include");
        let library = workspace.path().join("SDK-Lib");
        let bin = workspace.path().join("Bin");
        for path in [&include, &library, &bin] {
            fs::create_dir(path).unwrap();
        }
        let header = include.join("scope_fixture.h");
        fs::write(&header, b"// recorded header bytes\n").unwrap();
        fs::write(library.join("fixture.lib"), b"recorded library bytes").unwrap();
        let compiler = bin.join("cl.exe");
        fs::write(&compiler, b"scope fixture, never executed").unwrap();

        let include_alias = workspace.path().join("sdk-include");
        let library_alias = workspace.path().join("sdk-lib");
        assert_eq!(
            include_alias.canonicalize().unwrap(),
            include.canonicalize().unwrap()
        );
        assert_eq!(
            library_alias.canonicalize().unwrap(),
            library.canonicalize().unwrap()
        );

        let mut manifest = manifest_fixture();
        manifest.compiler = tool_file_identity(&compiler).unwrap();
        manifest.executable_inputs = vec![manifest.compiler.clone()];
        manifest
            .environment
            .insert("INCLUDE".to_string(), include_alias.display().to_string());
        for key in ["LIB", "LIBPATH"] {
            manifest
                .environment
                .insert(key.to_string(), library_alias.display().to_string());
        }
        manifest.include_roots = environment_roots(&manifest.environment, &["INCLUDE"]).unwrap();
        manifest.discovery_roots =
            environment_roots(&manifest.environment, &["LIB", "LIBPATH"]).unwrap();
        manifest
            .discovery_roots
            .push(parent(&manifest.compiler.path).unwrap().to_string());
        manifest.discovery_roots.sort();
        manifest.discovery_roots.dedup();
        let scopes = manifest
            .include_roots
            .iter()
            .chain(&manifest.discovery_roots)
            .cloned()
            .collect::<Vec<_>>();
        manifest.files = collect_host_toolchain_scope_files(&scopes).unwrap();

        validate_host_toolchain_manifest("actual-windows-search-roots", &manifest).unwrap();
        let recorded_header = manifest
            .files
            .iter()
            .find(|file| basename(&file.logical_path) == "scope_fixture.h")
            .unwrap();
        assert_eq!(
            comparison_path(&recorded_header.logical_path).unwrap(),
            normalize_windows_path(&include_alias.join("scope_fixture.h").display().to_string())
                .unwrap()
        );
        assert_eq!(
            recorded_header.resolved_path,
            header.canonicalize().unwrap().display().to_string()
        );
        assert_eq!(
            recorded_header.sha256,
            sha256_bytes(b"// recorded header bytes\n")
        );
        assert_eq!(
            rebuild_host_toolchain_manifest(&manifest).unwrap(),
            manifest
        );
    }

    #[test]
    fn compiler_probe_requires_actual_x64_and_dynamic_crt_macros() {
        let valid = b"FERRUM_MSVC_VERSION 1938\nFERRUM_MSVC_FULL_VERSION 193833130\nFERRUM_MSVC_X64 100\nFERRUM_MSVC_DYNAMIC_CRT 1\n";
        assert_eq!(validate_probe(valid).unwrap(), (1938, 193833130));
        for bad in [
            String::from_utf8_lossy(valid).replace("X64 100", "X64 _M_X64"),
            String::from_utf8_lossy(valid).replace("CRT 1", "CRT _DLL"),
            String::from_utf8_lossy(valid).replace("193833130", "194433130"),
        ] {
            assert!(validate_probe(bad.as_bytes()).is_err());
        }
    }
}
