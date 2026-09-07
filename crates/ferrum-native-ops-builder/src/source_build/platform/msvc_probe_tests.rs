use super::*;

fn environment() -> BTreeMap<String, String> {
    let recorded = [
        ("INCLUDE", "C:/Selected/include"),
        ("LIB", "C:/Selected/lib"),
        ("LIBPATH", "C:/Selected/lib"),
        ("SystemRoot", "C:/Selected/Windows"),
        ("TEMP", "C:/Selected/temp"),
        ("TMP", "C:/Selected/tmp"),
        ("WindowsSdkVerBinPath", "C:/Selected/sdk/bin"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect();
    msvc_environment_for_tools(
        [
            "C:/Selected/cuda/bin/nvcc.exe",
            "C:/Selected/msvc/bin/cl.exe",
            "C:/Selected/msvc/bin/lib.exe",
        ],
        &recorded,
    )
    .unwrap()
}

#[test]
fn version_probe_uses_the_supplied_build_environment() {
    let environment = environment();
    for (program, argument) in [("nvcc.exe", "--version"), ("cl.exe", "/?")] {
        let command = tool_version_command(Path::new(program), &environment).unwrap();
        assert_eq!(command.get_program(), program);
        assert_eq!(command.get_args().collect::<Vec<_>>(), [argument]);
        let actual = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_str().unwrap().to_string(),
                    value.unwrap().to_str().unwrap().to_string(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(actual, environment);
        assert_eq!(actual["SystemRoot"], "C:/Selected/Windows");
        assert_eq!(actual["TEMP"], "C:/Selected/temp");
        assert_eq!(actual["TMP"], "C:/Selected/tmp");
    }
}

#[test]
fn version_probe_rejects_missing_or_undeclared_environment() {
    let complete = environment();
    for key in ["INCLUDE", "TEMP", "PATH"] {
        let mut missing = complete.clone();
        missing.remove(key);
        assert!(tool_version_command(Path::new("nvcc.exe"), &missing).is_err());
    }
    let mut injected = complete;
    injected.insert(
        "NVCC_PREPEND_FLAGS".to_string(),
        "--use_fast_math".to_string(),
    );
    assert!(tool_version_command(Path::new("nvcc.exe"), &injected).is_err());
    let error = crate::source_build::tool_version(Path::new("nvcc.exe"), None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("requires the controlled build environment"),
        "{error}"
    );
}

fn output(code: i32, stdout: &[u8], stderr: &[u8]) -> std::process::Output {
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    #[cfg(windows)]
    use std::os::windows::process::ExitStatusExt;

    #[cfg(unix)]
    let status = std::process::ExitStatus::from_raw(code << 8);
    #[cfg(windows)]
    let status = std::process::ExitStatus::from_raw(code as u32);
    std::process::Output {
        status,
        stdout: stdout.to_vec(),
        stderr: stderr.to_vec(),
    }
}

#[test]
fn version_probe_preserves_output_and_requires_successful_nonempty_version() {
    let command = tool_version_command(Path::new("nvcc.exe"), &environment()).unwrap();
    assert_eq!(
        version_from_output(
            &command,
            &output(0, b" CUDA version\n", b"build identity\n")
        )
        .unwrap(),
        "CUDA version\nbuild identity"
    );
    assert!(version_from_output(&command, &output(0, b" \n", b"")).is_err());
    let error = version_from_output(&command, &output(9, b"version banner", b"probe failed"))
        .unwrap_err()
        .to_string();
    for expected in [
        "code=Some(9)",
        "version banner",
        "probe failed",
        "--version",
    ] {
        assert!(error.contains(expected), "{error}");
    }
}
