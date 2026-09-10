use super::*;

struct Installation {
    _temporary: tempfile::TempDir,
    root: std::path::PathBuf,
    payload: std::path::PathBuf,
    release: Value,
}

impl Installation {
    fn new(backend: &str) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("Ferrum 中文 path");
        fs::create_dir(&root).unwrap();
        let launcher = b"fixture launcher; execution is observed separately";
        fs::write(root.join("ferrum.exe"), launcher).unwrap();
        let files = [
            ("ferrum.exe", b"fixture core".as_slice()),
            ("vcruntime140.dll", b"fixture runtime".as_slice()),
            ("licenses/Ferrum.txt", b"fixture license".as_slice()),
        ];
        let manifest = serde_json::to_vec(&json!({
            "schema_version":1,
            "build":{"version":"2.3.4"},
            "backend":backend,
            "target_triple":"x86_64-pc-windows-msvc",
            "cuda_compute_capability":if backend == "cuda" {"89"} else {""},
            "files":files.iter().map(|(path,bytes)|json!({
                "path":path,"size_bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))
            })).collect::<Vec<_>>()
        }))
        .unwrap();
        let directory = format!("2.3.4-{:x}", Sha256::digest(&manifest));
        let payload = root.join("versions").join(&directory);
        fs::create_dir_all(payload.join("licenses")).unwrap();
        for (path, bytes) in files {
            fs::write(payload.join(path), bytes).unwrap();
        }
        fs::write(payload.join("ferrum-portable.json"), manifest).unwrap();
        fs::write(
            root.join("current.json"),
            json!({"schema_version":1,"version_dir":directory}).to_string(),
        )
        .unwrap();
        let mut release = release();
        if backend == "cpu" {
            for asset in release["assets"].as_array_mut().unwrap() {
                for key in ["name", "browser_download_url"] {
                    asset[key] = asset[key]
                        .as_str()
                        .unwrap()
                        .replace("cuda-sm89", "cpu")
                        .into();
                }
            }
        }
        release["assets"].as_array_mut().unwrap().push(json!({
            "name":"ferrum-windows-launcher-v1.exe",
            "size":launcher.len(),
            "digest":format!("sha256:{:x}",Sha256::digest(launcher))
        }));
        Self {
            _temporary: temporary,
            root,
            payload,
            release,
        }
    }

    fn current(&self, version: &str, backend: &str, process: &str) -> bool {
        let output = powershell(
            &format!(
                r#"
                function Invoke-FerrumProcess {{ param($Program,$Arguments); {process} }}
                $release = {} | ConvertFrom-Json
                Test-FerrumInstalledRelease -Directory {} -Release $release -ExpectedVersion {} -SelectedBackend {} | ConvertTo-Json -Compress
                "#,
                quote(&self.release.to_string()),
                quote(self.root.to_str().unwrap()),
                quote(version),
                quote(backend),
            ),
            &[],
        );
        require_success(&output);
        serde_json::from_slice(&output.stdout).unwrap()
    }
}

const RUNNING_VERSION: &str = "[pscustomobject]@{stdout='ferrum 2.3.4'}";

#[test]
fn repeat_install_checks_latest_and_skips_download_and_setup_for_intact_payload() {
    let fixture = Installation::new("cpu");
    let server = Server::new(BTreeMap::from([(
        "/release.json".into(),
        serde_json::to_vec(&fixture.release).unwrap(),
    )]));
    let output = powershell(
        &format!(
            r#"
            $script:lookups=0; $script:downloads=0; $script:setups=0; $script:starts=0
            function Get-FerrumInstallDirectory {{ return {root} }}
            function Invoke-RestMethod {{
                param($Uri,$Headers)
                if ($Uri -ne 'https://api.github.com/repos/sizzlecar/ferrum-infer-rs/releases/latest') {{ throw 'unexpected release lookup' }}
                $script:lookups++
                Microsoft.PowerShell.Utility\Invoke-RestMethod -Uri {url}
            }}
            function Invoke-FerrumDownloadWithFallback {{ $script:downloads++; throw 'unexpected installer download' }}
            function Install-FerrumSetup {{ $script:setups++; throw 'unexpected setup execution' }}
            function Invoke-FerrumProcess {{ param($Program,$Arguments); $script:starts++; {RUNNING_VERSION} }}
            Install-FerrumRelease -RequestedBackend cpu
            $once=[string]$env:Path
            Install-FerrumRelease -RequestedBackend cpu
            [ordered]@{{lookups=$script:lookups;downloads=$script:downloads;setups=$script:setups;starts=$script:starts;selected=(Get-Command ferrum).Path;stable_path=($once -ceq [string]$env:Path)}} | ConvertTo-Json -Compress
            "#,
            root = quote(fixture.root.to_str().unwrap()),
            url = quote(&format!("{}/release.json", server.url)),
        ),
        &[],
    );
    require_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: Value = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
    assert_eq!(result["lookups"], 2);
    assert_eq!(result["downloads"], 0);
    assert_eq!(result["setups"], 0);
    assert_eq!(result["starts"], 2);
    assert_eq!(result["stable_path"], true);
    assert_eq!(
        result["selected"],
        fixture.root.join("ferrum.exe").to_str().unwrap()
    );
}

#[test]
fn reuse_requires_selected_version_backend_inventory_and_a_working_launcher() {
    for backend in ["cpu", "cuda"] {
        let fixture = Installation::new(backend);
        assert!(fixture.current("2.3.4", backend, RUNNING_VERSION));
        assert!(!fixture.current("2.3.5", backend, RUNNING_VERSION));
        let other = if backend == "cpu" { "cuda" } else { "cpu" };
        assert!(!fixture.current("2.3.4", other, RUNNING_VERSION));
        assert!(!fixture.current("2.3.4", backend, "throw 'loader failure'"));
        assert!(!fixture.current("2.3.4", backend, "[pscustomobject]@{stdout='ferrum 2.3.3'}"));
    }
    for relative in [
        "ferrum.exe",
        "vcruntime140.dll",
        "licenses/Ferrum.txt",
        "ferrum-portable.json",
    ] {
        let fixture = Installation::new("cpu");
        let path = fixture.payload.join(relative);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert!(
            !fixture.current("2.3.4", "cpu", RUNNING_VERSION),
            "changed {relative}"
        );
        fs::remove_file(&path).unwrap();
        assert!(
            !fixture.current("2.3.4", "cpu", RUNNING_VERSION),
            "missing {relative}"
        );
    }
    let fixture = Installation::new("cpu");
    fs::write(fixture.root.join("ferrum.exe"), b"changed launcher").unwrap();
    assert!(!fixture.current("2.3.4", "cpu", RUNNING_VERSION));
    let fixture = Installation::new("cpu");
    fs::write(fixture.payload.join("extra.dll"), b"unlisted runtime").unwrap();
    assert!(!fixture.current("2.3.4", "cpu", RUNNING_VERSION));
    let fixture = Installation::new("cpu");
    fs::write(
        fixture.root.join("current.json"),
        r#"{"schema_version":1,"version_dir":"../elsewhere"}"#,
    )
    .unwrap();
    assert!(!fixture.current("2.3.4", "cpu", RUNNING_VERSION));
}

#[test]
fn first_install_and_new_release_download_verified_setup_without_changing_old_payload() {
    for first_install in [true, false] {
        let fixture = Installation::new("cpu");
        let root = if first_install {
            fixture._temporary.path().join("not installed yet")
        } else {
            fixture.root.clone()
        };
        let version = if first_install { "2.3.4" } else { "2.3.5" };
        let name = format!("ferrum-{version}-windows-x86_64-cpu-setup.exe");
        let setup = b"fixture installer: execution is checked at the setup boundary";
        let hash = format!("{:x}", Sha256::digest(setup));
        let checksum = format!("{hash}  {name}\n");
        let mut release = fixture.release.clone();
        release["tag_name"] = format!("v{version}").into();
        for (index, size) in [setup.len(), checksum.len()].into_iter().enumerate() {
            let asset_name = format!("{name}{}", if index == 0 { "" } else { ".sha256" });
            release["assets"][index] = json!({
                "name":asset_name,
                "size":size,
                "browser_download_url":format!("https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v{version}/{asset_name}")
            });
        }
        let server = Server::new(BTreeMap::from([
            (
                "/release.json".into(),
                serde_json::to_vec(&release).unwrap(),
            ),
            (format!("/{name}"), setup.to_vec()),
            (format!("/{name}.sha256"), checksum.into_bytes()),
        ]));
        let pointer = fs::read(fixture.root.join("current.json")).unwrap();
        let core = fs::read(fixture.payload.join("ferrum.exe")).unwrap();
        let output = powershell(
            &format!(
                r#"
                $script:downloads=@(); $script:setup=$null
                function Get-FerrumInstallDirectory {{ return {root} }}
                function Invoke-RestMethod {{ param($Uri,$Headers); Microsoft.PowerShell.Utility\Invoke-RestMethod -Uri ({origin}+'/release.json') }}
                function Invoke-FerrumDownloadWithFallback {{
                    param($Uri,$FallbackUri,$Destination,$ExpectedSize)
                    $name=[IO.Path]::GetFileName($Destination)
                    if ($Uri -cne ('https://ferrum.pandaailabs.com/downloads/v'+{version}+'/'+$name)) {{ throw 'wrong CDN release' }}
                    if ($FallbackUri -cne ('https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v'+{version}+'/'+$name)) {{ throw 'wrong fallback release' }}
                    Invoke-FerrumDownload -Uri ({origin}+'/'+$name) -Destination $Destination -ExpectedSize $ExpectedSize
                    $script:downloads+=@($name)
                }}
                function Install-FerrumSetup {{
                    param($SetupPath,$Sha256,$ExpectedVersion,$SelectedBackend)
                    Confirm-FerrumFile -Path $SetupPath -Sha256 $Sha256
                    $script:setup=[ordered]@{{sha256=$Sha256;version=$ExpectedVersion;backend=$SelectedBackend;temporary=$SetupPath}}
                }}
                Install-FerrumRelease -RequestedBackend cpu
                [ordered]@{{downloads=$script:downloads;setup=$script:setup;temporary_removed=(-not (Test-Path -LiteralPath $script:setup.temporary))}} | ConvertTo-Json -Compress
                "#,
                root = quote(root.to_str().unwrap()),
                origin = quote(&server.url),
                version = quote(version),
            ),
            &[],
        );
        require_success(&output);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: Value = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
        assert_eq!(result["downloads"], json!([name, format!("{name}.sha256")]));
        assert_eq!(result["setup"]["version"], version);
        assert_eq!(result["setup"]["backend"], "cpu");
        assert_eq!(result["setup"]["sha256"], hash);
        assert_eq!(result["temporary_removed"], true);
        assert_eq!(
            fs::read(fixture.root.join("current.json")).unwrap(),
            pointer
        );
        assert_eq!(fs::read(fixture.payload.join("ferrum.exe")).unwrap(), core);
    }
}
