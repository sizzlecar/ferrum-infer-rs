//! Exercise the actual product PowerShell functions. No alternate installer or
//! copied release-selection implementation is used by these Rust fixtures.
#![cfg(windows)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

#[path = "support/http_fixture.rs"]
mod http_fixture;
use http_fixture::Server;

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn powershell(body: &str, environment: &[(&str, &str)]) -> std::process::Output {
    powershell_in("System32", body, environment)
}

fn powershell_in(
    system_directory: &str,
    body: &str,
    environment: &[(&str, &str)],
) -> std::process::Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/install.ps1")
        .canonicalize()
        .unwrap();
    invoke_powershell_in(
        system_directory,
        &format!(". {}; {body}", quote(script.to_str().unwrap())),
        environment,
    )
}

fn invoke_powershell_in(
    system_directory: &str,
    body: &str,
    environment: &[(&str, &str)],
) -> std::process::Output {
    let program = Path::new(&std::env::var_os("SystemRoot").unwrap())
        .join(system_directory)
        .join("WindowsPowerShell/v1.0/powershell.exe");
    Command::new(program)
        // Load the repository's unsigned script only in this test process,
        // independently of the invoking shell's execution policy.
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
        ])
        .arg(format!(
            "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); {body}"
        ))
        .envs(environment.iter().copied())
        // Let Windows PowerShell build its own module paths. A PowerShell 7
        // parent passes incompatible modules through intermediate Rust processes.
        .env_remove("PSModulePath")
        .output()
        .unwrap()
}

#[test]
fn downloaded_bootstrap_pipeline_selects_cpu_without_driver_tools() {
    // Serve unmodified candidate bytes and execute the README's IRM/IEX pipeline.
    // Replace driver-file lookup and release metadata at the network boundary.
    // Missing CPU assets stop before setup execution; this is a startup/selection
    // regression, not a complete installation from the public website.
    let script =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/install.ps1")).unwrap();
    let server = Server::new(BTreeMap::from([("/install.ps1".into(), script)]));
    let body = format!(
        r#"
        $cpu = @(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Architecture -Unique)
        function Test-Path {{ param($LiteralPath, $PathType); return $false }}
        function Invoke-RestMethod {{
            param($Uri, $Headers)
            if ($Uri -eq {url}) {{ return Microsoft.PowerShell.Utility\Invoke-RestMethod -Uri $Uri }}
            if ($Uri -ne 'https://api.github.com/repos/sizzlecar/ferrum-infer-rs/releases/latest') {{ throw 'unexpected release endpoint' }}
            return [pscustomobject]@{{tag_name='v2.3.4'; draft=$false; prerelease=$false; assets=@()}}
        }}
        $failure = $null
        try {{ irm {url} | iex }}
        catch {{ $failure = [ordered]@{{message=$_.Exception.Message; error_id=$_.FullyQualifiedErrorId; cpu=$cpu; process_bits=([IntPtr]::Size*8)}} }}
        if ($null -eq $failure) {{ throw 'installer unexpectedly accepted missing release assets' }}
        $failure | ConvertTo-Json -Compress
        "#,
        url = quote(&format!("{}/install.ps1", server.url)),
    );
    let system_root = std::env::var_os("SystemRoot").unwrap();
    for directory in ["System32", "SysWOW64"] {
        if directory == "SysWOW64"
            && !Path::new(&system_root)
                .join(directory)
                .join("WindowsPowerShell/v1.0/powershell.exe")
                .is_file()
        {
            continue;
        }
        let output = invoke_powershell_in(directory, &body, &[]);
        require_success(&output);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let observed: Value = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
        let message = observed["message"].as_str().unwrap();
        if observed["cpu"] == json!([9]) {
            assert!(
                message.contains("ferrum-2.3.4-windows-x86_64-cpu-setup.exe"),
                "{observed}"
            );
        } else {
            assert!(message.contains("native Windows x64 only"), "{observed}");
        }
        if directory == "SysWOW64" {
            assert_eq!(observed["process_bits"], 32);
        }
    }
}

#[test]
fn downloaded_bootstrap_pipeline_reports_http_failure() {
    let server = Server::new(BTreeMap::new());
    let output = invoke_powershell_in(
        "System32",
        &format!(
            "irm {} | iex",
            quote(&format!("{}/install.ps1", server.url))
        ),
        &[],
    );
    assert!(
        !output.status.success(),
        "a missing download must fail the pipeline"
    );
}

#[test]
fn download_stream_preserves_bytes_and_reports_progress_before_completion() {
    let payload: Vec<u8> = (0..2 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let server = Server::new(BTreeMap::from([("/setup.exe".into(), payload.clone())]));
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("download 中文.exe");
    let receipt = temp.path().join("progress.json");
    let body = format!(
        r#"
        $events = [Collections.Generic.List[object]]::new()
        function Write-Progress {{
            param($Id, $Activity, $Status, $PercentComplete, [switch]$Completed)
            $events.Add([ordered]@{{percent=$PercentComplete; completed=$Completed.IsPresent}})
        }}
        Invoke-FerrumDownload -Uri {url} -Destination {destination} -ExpectedSize {size}
        [IO.File]::WriteAllText({receipt}, (ConvertTo-Json -InputObject @($events.ToArray()) -Compress))
        "#,
        url = quote(&format!("{}/setup.exe", server.url)),
        destination = quote(destination.to_str().unwrap()),
        size = payload.len(),
        receipt = quote(receipt.to_str().unwrap()),
    );
    let output = powershell(&body, &[]);
    require_success(&output);
    assert_eq!(fs::read(destination).unwrap(), payload);
    let events: Vec<Value> = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
    assert_eq!(events.first().unwrap()["percent"], 0);
    assert!(events.iter().any(|event| event["percent"]
        .as_u64()
        .is_some_and(|percent| percent > 0 && percent < 100)));
    assert!(events.iter().any(|event| event["percent"] == 100));
    assert_eq!(events.last().unwrap()["completed"], true);
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(log.contains("Downloading download 中文.exe"), "{log}");
    assert!(
        log.contains("Downloaded download 中文.exe (2097152 bytes)"),
        "{log}"
    );
}

#[test]
fn download_http_and_size_failures_do_not_leave_an_installable_payload() {
    let server = Server::responses(BTreeMap::from([
        ("/setup.exe".into(), b"payload".to_vec().into()),
        (
            "/truncated.exe".into(),
            http_fixture::Response {
                body: b"partial".to_vec(),
                declared_length: Some(128),
            },
        ),
        (
            "/too-large.exe".into(),
            http_fixture::Response {
                body: b"larger than expected".to_vec(),
                declared_length: None,
            },
        ),
    ]));
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("setup.exe");
    for (path, size) in [
        ("/missing.exe", 7),
        ("/setup.exe", 8),
        ("/truncated.exe", 128),
        ("/too-large.exe", 7),
    ] {
        let output = powershell(
            &format!(
                "Invoke-FerrumDownload -Uri {} -Destination {} -ExpectedSize {size}",
                quote(&format!("{}{path}", server.url)),
                quote(destination.to_str().unwrap()),
            ),
            &[],
        );
        assert!(!output.status.success());
        assert!(!destination.exists());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Downloaded setup.exe"));
    }
    fs::write(&destination, b"existing user file").unwrap();
    let output = powershell(
        &format!(
            "Invoke-FerrumDownload -Uri {} -Destination {} -ExpectedSize 7",
            quote(&format!("{}/setup.exe", server.url)),
            quote(destination.to_str().unwrap()),
        ),
        &[],
    );
    assert!(!output.status.success());
    assert_eq!(fs::read(destination).unwrap(), b"existing user file");
}

#[test]
fn bootstrap_detects_native_architecture_in_windows_powershell_and_wow64() {
    // An independent OS query is the oracle; process bitness is deliberately
    // different in SysWOW64. Do not assume the CI host's CPU architecture.
    let body = r#"
        $cpu = @(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Architecture -Unique)
        [ordered]@{architecture=(Get-FerrumWindowsArchitecture); cpu=$cpu; process_bits=([IntPtr]::Size*8)} | ConvertTo-Json -Compress
    "#;
    let output = powershell(body, &[]);
    require_success(&output);
    let native: Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = match native["cpu"].as_array().unwrap().as_slice() {
        [value] => match value.as_u64().unwrap() {
            0 => "X86",
            5 => "Arm",
            9 => "X64",
            12 => "Arm64",
            other => panic!("unsupported processor architecture {other}"),
        },
        other => panic!("inconsistent processor architectures {other:?}"),
    };
    assert_eq!(native["architecture"], expected);

    // Parent-process environment overrides must not change native detection.
    let overridden = powershell(
        "Get-FerrumWindowsArchitecture",
        &[
            ("PROCESSOR_ARCHITECTURE", "unknown"),
            ("PROCESSOR_ARCHITEW6432", "unknown"),
        ],
    );
    require_success(&overridden);
    assert_eq!(
        String::from_utf8(overridden.stdout).unwrap().trim(),
        expected
    );

    if expected == "X64" {
        let output = powershell_in("SysWOW64", body, &[]);
        require_success(&output);
        let wow64: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(wow64["process_bits"], 32);
        assert_eq!(wow64["architecture"], native["architecture"]);
    }
}

#[test]
fn gpu_probe_directories_resolve_native_executables_from_both_powershells() {
    // Use an OS executable present even on driverless hosts to prove that the
    // GPU tool's directory resolves to native bytes, including under WOW64.
    let body = r#"
        $candidates = @(Get-FerrumNvidiaSmiCandidates)
        $nativeCommand = Join-Path ([IO.Path]::GetDirectoryName($candidates[0])) 'cmd.exe'
        [ordered]@{
            candidates=$candidates
            command_sha256=(Get-FileHash -LiteralPath $nativeCommand -Algorithm SHA256).Hash
            is64_os=[Environment]::Is64BitOperatingSystem
            process_bits=([IntPtr]::Size*8)
        } | ConvertTo-Json -Compress
    "#;
    let native_output = powershell(body, &[]);
    require_success(&native_output);
    let native: Value = serde_json::from_slice(&native_output.stdout).unwrap();
    if native["is64_os"] == true {
        let wow64_output = powershell_in("SysWOW64", body, &[]);
        require_success(&wow64_output);
        let wow64: Value = serde_json::from_slice(&wow64_output.stdout).unwrap();
        assert_eq!(wow64["process_bits"], 32);
        assert_eq!(wow64["command_sha256"], native["command_sha256"]);
        assert_eq!(wow64["candidates"][1], native["candidates"][1]);
    }
}

#[test]
fn bootstrap_startup_reaches_gpu_probe_after_real_architecture_detection() {
    // Exercise the install entrypoint, stopping at the first hardware probe.
    // The only replaced boundaries are driver-file lookup and process launch;
    // no network request or installer execution is needed for this regression.
    let output = powershell(
        r#"
        $architecture = Get-FerrumWindowsArchitecture
        function Test-Path { param($LiteralPath, $PathType); return $true }
        function Invoke-FerrumProcess { param($Program, $Arguments); throw 'fixture: reached GPU probe' }
        try { Install-FerrumRelease -RequestedBackend cuda; throw 'startup did not stop at hardware detection' }
        catch {
            $expected = if ($architecture -eq 'X64') { 'fixture: reached GPU probe' } else { 'This installer supports native Windows x64 only.' }
            if ($_.Exception.Message -cne $expected) { throw }
            $architecture
        }
        "#,
        &[],
    );
    require_success(&output);
}

fn require_success(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "status={} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn release() -> Value {
    let name = "ferrum-2.3.4-windows-x86_64-cuda-sm89-setup.exe";
    let assets = [name.to_owned(), format!("{name}.sha256")]
        .into_iter()
        .map(|name| {
            json!({"browser_download_url":format!("https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v2.3.4/{name}"),"name":name,"size":512})
        })
        .collect::<Vec<_>>();
    json!({
        "tag_name":"v2.3.4", "draft":false, "prerelease":false,
        "assets":assets
    })
}

#[test]
fn bootstrap_selects_same_release_setup_and_rejects_missing_or_foreign_assets() {
    let temp = tempfile::tempdir().unwrap();
    let fixture = temp.path().join("release.json");
    let request = format!(
        "Select-FerrumRelease -Release (Get-Content -Raw -LiteralPath {} | ConvertFrom-Json) -RequestedVersion '2.3.4' | ConvertTo-Json -Depth 5",
        quote(fixture.to_str().unwrap())
    );
    fs::write(&fixture, serde_json::to_vec(&release()).unwrap()).unwrap();
    let result = powershell(&request, &[]);
    require_success(&result);
    let selected: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(selected["version"], "2.3.4");
    assert_eq!(
        selected["setup"]["name"],
        "ferrum-2.3.4-windows-x86_64-cuda-sm89-setup.exe"
    );
    assert_eq!(
        selected["checksum"]["name"],
        "ferrum-2.3.4-windows-x86_64-cuda-sm89-setup.exe.sha256"
    );

    let mut cases = Vec::new();
    for field in ["draft", "prerelease"] {
        let mut item = release();
        item[field] = true.into();
        cases.push(item);
    }
    for tag in ["v2.3.5", "v2.3.4-rc.1", "v2.03.4", "v2.3.4;whoami"] {
        let mut item = release();
        item["tag_name"] = tag.into();
        cases.push(item);
    }
    let mut absent = release();
    absent["assets"].as_array_mut().unwrap().pop();
    cases.push(absent);
    let mut duplicate = release();
    let first = duplicate["assets"][0].clone();
    duplicate["assets"].as_array_mut().unwrap().push(first);
    cases.push(duplicate);
    for url in [
        "https://example.com/setup.exe",
        "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v2.3.5/ferrum-2.3.4-windows-x86_64-cuda-sm89-setup.exe",
    ] {
        let mut item = release();
        item["assets"][0]["browser_download_url"] = url.into();
        cases.push(item);
    }
    for item in cases {
        fs::write(&fixture, serde_json::to_vec(&item).unwrap()).unwrap();
        assert!(
            !powershell(&request, &[]).status.success(),
            "accepted {item}"
        );
    }
}

#[test]
fn bootstrap_checksum_binds_filename_and_file_bytes_before_execution() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("setup with spaces.exe");
    fs::write(&file, b"fixture bytes, never executable").unwrap();
    let hash = format!("{:x}", Sha256::digest(fs::read(&file).unwrap()));
    let command = format!(
        "$hash=ConvertFrom-FerrumChecksum -Text {} -AssetName 'setup.exe'; Confirm-FerrumFile -Path {} -Sha256 $hash; $hash",
        quote(&format!("{hash}  setup.exe\r\n")), quote(file.to_str().unwrap())
    );
    let result = powershell(&command, &[]);
    require_success(&result);
    assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), hash);
    fs::write(&file, b"different bytes, same file path").unwrap();
    assert!(!powershell(&command, &[]).status.success());
    for checksum in [
        format!("{hash}  another.exe"),
        format!("{hash}\n{hash}"),
        "1234".into(),
    ] {
        let result = powershell(
            &format!(
                "ConvertFrom-FerrumChecksum -Text {} -AssetName 'setup.exe'",
                quote(&checksum)
            ),
            &[],
        );
        assert!(!result.status.success());
    }
    let output = powershell(
        &format!(
            "Install-FerrumSetup -SetupPath {} -Sha256 {} -ExpectedVersion '2.3.4'",
            quote(file.to_str().unwrap()),
            quote(&hash)
        ),
        &[],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("SHA256 mismatch"));
}

#[test]
fn bootstrap_hardware_contract_rejects_other_architectures_or_old_drivers() {
    require_success(&powershell(
        "Assert-FerrumHardware -Architecture 'X64' -GpuCsv '8.9, 560.94'",
        &[],
    ));
    for (architecture, csv) in [
        ("Arm64", "8.9, 560.94"),
        ("X64", "8.6, 560.94"),
        ("X64", "8.9, 528.33"),
        ("X64", "8.9, unknown"),
    ] {
        assert!(!powershell(
            &format!(
                "Assert-FerrumHardware -Architecture {} -GpuCsv {}",
                quote(architecture),
                quote(csv)
            ),
            &[]
        )
        .status
        .success());
    }
}

#[test]
fn bootstrap_backend_selection_falls_back_only_for_automatic_requests() {
    for (requested, csv, probe_error, expected) in [
        ("auto", "8.9, 560.94", "", "cuda"),
        ("auto", "", "driver tools absent", "cpu"),
        ("auto", "8.6, 560.94", "", "cpu"),
        ("auto", "8.9, 528.33", "", "cpu"),
        ("auto", "invalid", "", "cpu"),
        ("cpu", "8.9, 560.94", "", "cpu"),
        ("cpu", "", "driver tools absent", "cpu"),
    ] {
        let result = powershell(&format!(
            "Resolve-FerrumBackend -RequestedBackend {} -Architecture X64 -GpuCsv {} -ProbeError {}",
            quote(requested), quote(csv), quote(probe_error)
        ), &[]);
        require_success(&result);
        assert_eq!(
            String::from_utf8_lossy(&result.stdout).lines().last(),
            Some(expected)
        );
    }
    for csv in ["", "8.6, 560.94", "8.9, 528.33", "invalid"] {
        assert!(!powershell(
            &format!(
                "Resolve-FerrumBackend -RequestedBackend cuda -Architecture X64 -GpuCsv {}",
                quote(csv)
            ),
            &[]
        )
        .status
        .success());
    }
    assert!(!powershell(
        "Resolve-FerrumBackend -RequestedBackend cpu -Architecture Arm64",
        &[]
    )
    .status
    .success());
}

#[test]
fn bootstrap_selects_cpu_assets_from_the_requested_release() {
    let temp = tempfile::tempdir().unwrap();
    let fixture = temp.path().join("cpu-release.json");
    let mut cpu_release = release();
    for asset in cpu_release["assets"].as_array_mut().unwrap() {
        for key in ["name", "browser_download_url"] {
            asset[key] = asset[key]
                .as_str()
                .unwrap()
                .replace("cuda-sm89", "cpu")
                .into();
        }
    }
    fs::write(&fixture, serde_json::to_vec(&cpu_release).unwrap()).unwrap();
    let result = powershell(&format!(
        "Select-FerrumRelease -Release (Get-Content -Raw -LiteralPath {} | ConvertFrom-Json) -RequestedVersion '2.3.4' -SelectedBackend cpu | ConvertTo-Json -Depth 5",
        quote(fixture.to_str().unwrap())
    ), &[]);
    require_success(&result);
    let selected: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        selected["setup"]["name"],
        "ferrum-2.3.4-windows-x86_64-cpu-setup.exe"
    );
    assert_eq!(
        selected["checksum"]["name"],
        "ferrum-2.3.4-windows-x86_64-cpu-setup.exe.sha256"
    );
}

#[test]
fn bootstrap_process_path_selects_installed_executable_and_preserves_other_commands() {
    let temp = tempfile::tempdir().unwrap();
    let installed = temp.path().join("Ferrum 中文 path");
    let previous = temp.path().join("previous portable");
    fs::create_dir(&installed).unwrap();
    fs::create_dir(&previous).unwrap();
    let executable = std::env::current_exe().unwrap();
    let selected = installed.join("ferrum.exe");
    let old = previous.join("ferrum.exe");
    let other = previous.join("existing-tool.exe");
    for path in [&selected, &old, &other] {
        fs::copy(&executable, path).unwrap();
    }
    let receipt = temp.path().join("selected-child.json");
    let directory = installed.to_str().unwrap();
    let body = format!(
        "$before=(Get-Command ferrum).Path; Add-FerrumProcessPath -Directory {}; Confirm-FerrumCommand -Program {}; $once=[string]$env:Path; Add-FerrumProcessPath -Directory {}; Confirm-FerrumCommand -Program {}; $null=& ferrum --exact bootstrap_child --ignored --nocapture; if ($LASTEXITCODE -ne 0) {{throw 'selected child failed'}}; [ordered]@{{before=$before;selected=(Get-Command ferrum).Path;other=(Get-Command existing-tool).Path;once=$once;twice=[string]$env:Path}} | ConvertTo-Json",
        quote(directory), quote(selected.to_str().unwrap()),
        quote(&format!("{directory}\\")), quote(selected.to_str().unwrap())
    );
    for original in [
        previous.display().to_string(),
        format!("{};;C:\\keep second;", previous.display()),
        format!(
            "{};{};C:\\keep second",
            previous.display(),
            directory.to_uppercase()
        ),
    ] {
        let output = powershell(
            &body,
            &[
                ("Path", &original),
                ("FERRUM_BOOTSTRAP_CHILD_RECEIPT", receipt.to_str().unwrap()),
                ("FERRUM_BOOTSTRAP_CHILD_EXIT", "0"),
            ],
        );
        require_success(&output);
        let observed: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(observed["before"], old.to_str().unwrap());
        assert_eq!(observed["selected"], selected.to_str().unwrap());
        assert_eq!(observed["other"], other.to_str().unwrap());
        assert_eq!(observed["once"], observed["twice"]);
        let child: Vec<String> = serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(
            Path::new(&child[0]).canonicalize().unwrap(),
            selected.canonicalize().unwrap()
        );
    }
}

#[test]
fn bootstrap_preserves_alias_and_function_conflicts_instead_of_claiming_ready() {
    let temp = tempfile::tempdir().unwrap();
    let installed = temp.path().join("Ferrum 中文 path");
    fs::create_dir(&installed).unwrap();
    let selected = installed.join("ferrum.exe");
    fs::copy(std::env::current_exe().unwrap(), &selected).unwrap();
    for (definition, kind) in [
        ("Set-Alias -Name ferrum -Value Write-Output", "Alias"),
        ("function ferrum { param($Value) $Value }", "Function"),
    ] {
        let body = format!(
            "{definition}; Add-FerrumProcessPath -Directory {}; $message=$null; try {{Confirm-FerrumCommand -Program {}}} catch {{$message=$_.Exception.Message}}; [ordered]@{{message=$message;kind=(Get-Command ferrum).CommandType.ToString();value=(ferrum 'original command survives')}} | ConvertTo-Json",
            quote(installed.to_str().unwrap()), quote(selected.to_str().unwrap())
        );
        let output = powershell(&body, &[]);
        require_success(&output);
        let observed: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(observed["message"]
            .as_str()
            .unwrap()
            .contains("resolves to another command"));
        assert_eq!(observed["kind"], kind);
        assert_eq!(observed["value"], "original command survives");
    }
}

#[test]
fn bootstrap_child_arguments_remain_literal_and_nonzero_exit_is_an_error() {
    let temp = tempfile::tempdir().unwrap();
    let receipt = temp.path().join("child.json");
    let executable = std::env::current_exe().unwrap();
    let values = [
        "with spaces",
        "中文; $(not-a-command)",
        "quote\" and trailing\\",
    ];
    let arguments = ["--exact", "bootstrap_child", "--ignored", "--nocapture"]
        .into_iter()
        .map(str::to_owned)
        .chain(
            values
                .iter()
                .flat_map(|v| ["--skip".to_owned(), (*v).to_owned()]),
        )
        .collect::<Vec<_>>();
    let command = format!(
        "Invoke-FerrumProcess -Program {} -Arguments @({}) | ConvertTo-Json",
        quote(executable.to_str().unwrap()),
        arguments
            .iter()
            .map(|s| quote(s))
            .collect::<Vec<_>>()
            .join(",")
    );
    let environment = [
        ("FERRUM_BOOTSTRAP_CHILD_RECEIPT", receipt.to_str().unwrap()),
        ("FERRUM_BOOTSTRAP_CHILD_EXIT", "0"),
    ];
    require_success(&powershell(&command, &environment));
    let actual: Vec<String> = serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(&actual[1..], arguments);
    let failure = powershell(
        &command,
        &[
            ("FERRUM_BOOTSTRAP_CHILD_RECEIPT", receipt.to_str().unwrap()),
            ("FERRUM_BOOTSTRAP_CHILD_EXIT", "7"),
        ],
    );
    assert!(!failure.status.success());
    assert!(String::from_utf8_lossy(&failure.stderr).contains("exited 7"));
}

#[test]
#[ignore = "child process fixture, invoked only by bootstrap_child_arguments_remain_literal_and_nonzero_exit_is_an_error"]
fn bootstrap_child() {
    let path =
        std::env::var_os("FERRUM_BOOTSTRAP_CHILD_RECEIPT").expect("parent-owned receipt path");
    fs::write(
        path,
        serde_json::to_vec(&std::env::args().collect::<Vec<_>>()).unwrap(),
    )
    .unwrap();
    let exit = std::env::var("FERRUM_BOOTSTRAP_CHILD_EXIT")
        .unwrap()
        .parse::<i32>()
        .unwrap();
    std::process::exit(exit);
}
