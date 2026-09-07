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
    for (program, argument) in [
        ("nvcc.exe", "--version"),
        ("cl.exe", "/?"),
        ("lib.exe", "/?"),
    ] {
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

fn lib_help(version: &str) -> String {
    format!(
        "Microsoft (R) Library Manager Version {version}\r\n\
         Copyright (C) Microsoft Corporation. All rights reserved.\r\n\r\n\
         usage: LIB [options] [files]\r\n\r\n\
            options:\r\n\r\n\
               /LIST[:filename]\r\n\
               /OUT:filename\r\n"
    )
}

#[test]
fn version_probe_accepts_recognized_lib_help_with_windows_status_1100() {
    let command = tool_version_command(
        Path::new(r"\\?\C:\Selected\MSVC\bin\LIB.EXE"),
        &environment(),
    )
    .unwrap();
    for version in ["14.38.33145.0", "14.39.33523.0"] {
        let help = lib_help(version);
        // Pass the full Windows code directly: a Unix ExitStatus cannot
        // represent 1100 without truncating it to a different exit code.
        assert_eq!(
            version_text_from_probe(&command, Some(1100), help.as_bytes(), b" \r\n\t"),
            Some(help.trim().to_string())
        );
    }
}

#[test]
fn version_probe_keeps_lib_help_status_specific_to_tool_and_arguments() {
    let help = lib_help("14.38.33145.0");
    let command = tool_version_command(Path::new("lib.exe"), &environment()).unwrap();
    for code in [
        None,
        Some(1),
        Some(76),
        Some(1099),
        Some(1101),
        Some(-1073741819),
    ] {
        assert!(
            version_text_from_probe(&command, code, help.as_bytes(), b"").is_none(),
            "unexpectedly accepted status {code:?}"
        );
    }
    for (program, arguments) in [
        ("nvcc.exe", vec!["--version"]),
        ("cl.exe", vec!["/?"]),
        ("link.exe", vec!["/?"]),
        ("llvm-lib.exe", vec!["/?"]),
        ("lib.exe", vec![]),
        ("lib.exe", vec!["--version"]),
        ("lib.exe", vec!["/?", "/OUT:output.lib"]),
        ("lib.exe", vec!["/OUT:output.lib", "input.obj"]),
    ] {
        let mut command = Command::new(program);
        command.args(arguments);
        assert!(
            version_text_from_probe(&command, Some(1100), help.as_bytes(), b"").is_none(),
            "unexpectedly accepted {command:?}"
        );
    }
}

#[test]
fn version_probe_rejects_unrecognized_or_damaged_lib_help() {
    let command = tool_version_command(Path::new("lib.exe"), &environment()).unwrap();
    for version in ["", "unknown", "143833145", "14..38", "14.38.beta", "14.38."] {
        assert!(
            version_text_from_probe(&command, Some(1100), lib_help(version).as_bytes(), b"")
                .is_none(),
            "unexpectedly accepted version {version:?}"
        );
    }
    let help = lib_help("14.38.33145.0");
    for damaged in [
        help.replace("Microsoft (R) Library Manager", "Other Library Manager"),
        help.replace("usage: LIB [options] [files]", ""),
        help.replace("/OUT:filename", ""),
        help.replace("/OUT:filename", "/OUT:"),
        format!("{help}LIB : fatal error LNK1104: cannot open file\r\n"),
        format!("{help}LIB : error LNK1104: cannot open file\r\n"),
    ] {
        assert!(
            version_text_from_probe(&command, Some(1100), damaged.as_bytes(), b"").is_none(),
            "unexpectedly accepted {damaged:?}"
        );
    }
    assert!(version_text_from_probe(
        &command,
        Some(1100),
        help.as_bytes(),
        b"LIB : fatal error LNK1104: cannot open file"
    )
    .is_none());
    let mut damaged_encoding = help.into_bytes();
    damaged_encoding.push(0xff);
    assert!(version_text_from_probe(&command, Some(1100), &damaged_encoding, b"").is_none());
}
