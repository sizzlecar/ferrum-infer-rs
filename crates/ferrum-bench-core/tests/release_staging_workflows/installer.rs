//! Compile the actual Windows workflow wrapper with Inno Setup. The small
//! payload files are packaging fixtures, never installed or executed.
use super::workflows;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const VERSION: &str = "2.3.4";

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct FileIdentity {
    name: String,
    size_bytes: u64,
    sha256: String,
}

fn identity(path: &Path) -> FileIdentity {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    FileIdentity {
        name: path.file_name().unwrap().to_str().unwrap().into(),
        size_bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Provenance {
    version: String,
    backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    iscc_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    archive: Option<FileIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    setup: Option<FileIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    launcher: Option<FileIdentity>,
}

struct Fixture {
    root: PathBuf,
    assets: PathBuf,
    payload: PathBuf,
    launcher: PathBuf,
    archive: PathBuf,
    setup: PathBuf,
}

impl Fixture {
    fn new(evidence: &Path, backend: &str, suffix: &str) -> Self {
        let root = evidence.join(backend);
        let assets = root.join("windows-assets");
        let payload = root.join("payload with spaces");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&assets).unwrap();
        fs::create_dir_all(payload.join("licenses")).unwrap();
        fs::write(payload.join("ferrum.exe"), b"packaging fixture core").unwrap();
        fs::write(
            payload.join("vcruntime140.dll"),
            b"packaging fixture runtime",
        )
        .unwrap();
        fs::write(
            payload.join("licenses/Ferrum.txt"),
            b"packaging fixture license",
        )
        .unwrap();
        // The template hashes this file; portable-manifest validation belongs to
        // the preceding staging step and its separate typed fixture tests.
        let manifest = serde_json::json!({
            "schema_version": 1,
            "build": {"version": VERSION},
            "backend": backend,
            "target_triple": "x86_64-pc-windows-msvc",
        });
        fs::write(
            payload.join("ferrum-portable.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let launcher = assets.join("ferrum-windows-launcher-v1.exe");
        fs::write(&launcher, b"packaging fixture launcher").unwrap();
        let archive = assets.join(format!("ferrum-windows-x86_64-{suffix}.zip"));
        fs::write(&archive, b"preceding archive step fixture").unwrap();
        let setup = assets.join(format!(
            "ferrum-{VERSION}-windows-x86_64-{suffix}-setup.exe"
        ));
        let provenance = Provenance {
            version: VERSION.into(),
            backend: backend.into(),
            iscc_version: None,
            archive: None,
            setup: None,
            launcher: None,
        };
        fs::write(
            assets.join("windows-staging.json"),
            serde_json::to_vec_pretty(&provenance).unwrap(),
        )
        .unwrap();
        Self {
            root,
            assets,
            payload,
            launcher,
            archive,
            setup,
        }
    }

    fn verify(&self, backend: &str) {
        let setup = fs::read(&self.setup).expect("workflow must produce the named installer");
        assert_eq!(setup.get(..2), Some(b"MZ".as_slice()));
        let pe_offset = u32::from_le_bytes(setup[0x3c..0x40].try_into().unwrap()) as usize;
        assert_eq!(
            setup.get(pe_offset..pe_offset + 4),
            Some(b"PE\0\0".as_slice())
        );
        let report: Provenance =
            serde_json::from_slice(&fs::read(self.assets.join("windows-staging.json")).unwrap())
                .unwrap();
        assert_eq!(report.version, VERSION);
        assert_eq!(report.backend, backend);
        let compiler_log = fs::read_to_string(self.assets.join("iscc-build.log")).unwrap();
        let compiler_version = compiler_log
            .lines()
            .find_map(|line| line.strip_prefix("Compiler engine version: Inno Setup "))
            .expect("actual compiler must identify its version");
        semver::Version::parse(compiler_version).expect("compiler reports a semantic version");
        assert_eq!(report.iscc_version.as_deref(), Some(compiler_version));
        for (reported, path) in [
            (report.archive, &self.archive),
            (report.setup, &self.setup),
            (report.launcher, &self.launcher),
        ] {
            let actual = identity(path);
            let checksum =
                fs::read_to_string(path.with_file_name(format!("{}.sha256", actual.name))).unwrap();
            assert_eq!(
                checksum.trim(),
                format!("{}  {}", actual.sha256, actual.name)
            );
            assert_eq!(reported.as_ref(), Some(&actual));
        }
    }
}

#[test]
#[ignore = "requires native Windows, PowerShell 7 on PATH, ISCC, and a fresh FERRUM_INNO_EVIDENCE_DIR"]
fn windows_installer_wrapper_compiles_cpu_and_cuda() {
    let compiler = PathBuf::from(std::env::var_os("ISCC").expect("set ISCC to the real ISCC.exe"));
    assert!(compiler.is_file(), "ISCC must name an existing compiler");
    let evidence = PathBuf::from(
        std::env::var_os("FERRUM_INNO_EVIDENCE_DIR")
            .expect("set FERRUM_INNO_EVIDENCE_DIR to a new external directory"),
    );
    assert!(evidence.is_absolute(), "evidence path must be absolute");
    fs::create_dir(&evidence)
        .expect("reserve a fresh evidence directory; existing files preserved");
    eprintln!("Inno compilation evidence: {}", evidence.display());
    fs::write(
        evidence.join("compiler.json"),
        serde_json::to_vec_pretty(&identity(&compiler)).unwrap(),
    )
    .unwrap();
    let shell = Command::new("pwsh.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$PSVersionTable.PSVersion.Major",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("PowerShell 7 must be installed as pwsh.exe on PATH");
    fs::write(evidence.join("powershell.stdout.log"), &shell.stdout).unwrap();
    fs::write(evidence.join("powershell.stderr.log"), &shell.stderr).unwrap();
    assert!(shell.status.success(), "PowerShell version probe failed");
    assert!(
        String::from_utf8(shell.stdout)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap()
            >= 7,
        "the real workflow requires PowerShell 7"
    );
    let (_, workflow) = workflows()
        .into_iter()
        .find(|(name, _)| name == "release-windows.yml")
        .unwrap();
    let wrapper = workflow["jobs"]["build"]["steps"]
        .as_sequence()
        .unwrap()
        .iter()
        .find(|step| {
            step["name"].as_str()
                == Some("Wrap the verified payload in the versioned current-user installer")
        })
        .expect("Windows workflow must have its installer packaging step")["run"]
        .as_str()
        .expect("packaging step must execute a script");
    fs::write(evidence.join("executed-wrapper.txt"), wrapper).unwrap();
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (backend, suffix) in [("cpu", "cpu"), ("cuda", "cuda-sm89")] {
        let fixture = Fixture::new(&evidence, backend, suffix);
        let output = Command::new("pwsh.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", wrapper])
            .current_dir(&workspace)
            .env("ISCC", &compiler)
            .env("BACKEND", backend)
            .env("VERSION", VERSION)
            .env("RUNNER_TEMP", &fixture.root)
            .env("WINDOWS_PAYLOAD", &fixture.payload)
            .env("WINDOWS_LAUNCHER_PATH", &fixture.launcher)
            .stdin(Stdio::null())
            .output()
            .expect("execute the actual Windows packaging step with PowerShell 7");
        fs::write(fixture.root.join("wrapper.stdout.log"), &output.stdout).unwrap();
        fs::write(fixture.root.join("wrapper.stderr.log"), &output.stderr).unwrap();
        assert!(
            output.status.success(),
            "{backend} wrapper failed; inspect {}: {}",
            fixture.root.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        fixture.verify(backend);
        eprintln!(
            "{backend}: compiled and verified {}",
            fixture.setup.display()
        );
    }
}
