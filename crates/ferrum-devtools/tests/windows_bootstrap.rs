//! Exercise the actual product PowerShell functions. No alternate installer or
//! copied release-selection implementation is used by these Rust fixtures.
#![cfg(windows)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn powershell(body: &str, environment: &[(&str, &str)]) -> std::process::Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/install.ps1")
        .canonicalize()
        .unwrap();
    let program = Path::new(&std::env::var_os("SystemRoot").unwrap())
        .join("System32/WindowsPowerShell/v1.0/powershell.exe");
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
            "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); . {}; {body}",
            quote(script.to_str().unwrap())
        ))
        .envs(environment.iter().copied())
        // Let Windows PowerShell build its own module paths. A PowerShell 7
        // parent passes incompatible modules through intermediate Rust processes.
        .env_remove("PSModulePath")
        .output()
        .unwrap()
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
