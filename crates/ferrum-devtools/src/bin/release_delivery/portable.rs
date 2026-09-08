//! Windows portable payload bytes and startup inspection. This does not prove
//! PE import closure, model correctness, an installer, or publication readiness.
use super::installation::{self, Observation};
use clap::{Args, Subcommand};
use ferrum_bench_core::release_candidate::staging::Backend;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const MANIFEST: &str = "ferrum-portable.json";
const TARGET: &str = "x86_64-pc-windows-msvc";
const MAX_FILES: usize = 128;
const MAX_PAYLOAD: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Args)]
pub struct PortableArgs {
    #[command(subcommand)]
    action: Action,
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Create a fresh ZIP and receipt from explicitly hashed runtime inputs.
    Pack {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        archive: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Extract the whole accepted payload and optionally run startup probes.
    Inspect {
        #[arg(long)]
        archive: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long)]
        extract_dir: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Check bytes on another OS without claiming Windows execution.
        #[arg(long)]
        extract_only: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    Binary,
    CudaRuntime,
    MsvcRuntime,
    License,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadFile {
    path: String,
    role: Role,
    sha256: String,
    size_bytes: u64,
    /// Exact payload license path for each executable or runtime file.
    license: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildIdentity {
    version: String,
    source_commit: String,
    /// Actual local or CI build identifier; does not assert a published RC tag.
    build_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    build: BuildIdentity,
    backend: Backend,
    target_triple: String,
    cuda_compute_capability: String,
    cargo_features: Vec<String>,
    files: Vec<PayloadFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackSpec {
    manifest: Manifest,
    /// Keys are exact archive paths, values are the original local input files.
    sources: BTreeMap<String, PathBuf>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u32,
    archive_name: String,
    archive_sha256: String,
    archive_size_bytes: u64,
    manifest: Manifest,
}

/// Revalidated package identity for the formal asset inventory; no model claim.
pub(super) struct StagedPortable {
    pub archive_sha256: String,
    pub archive_size_bytes: u64,
    pub binary_sha256: String,
    pub manifest_sha256: String,
    pub startup_executed: bool,
}

pub(super) fn verify_staged(
    archive: &Path,
    receipt_path: &Path,
    inspection_path: &Path,
    version: &str,
    candidate: &str,
) -> Result<StagedPortable, String> {
    let receipt: Receipt = read(receipt_path)?;
    if receipt.archive_name != "ferrum-windows-x86_64-cuda-sm89.zip"
        || receipt.manifest.build.version != version
        || receipt.manifest.build.source_commit != candidate
        || receipt.manifest.cuda_compute_capability != "89"
        || ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"]
            .iter()
            .any(|required| {
                !receipt
                    .manifest
                    .cargo_features
                    .iter()
                    .any(|f| f == required)
            })
    {
        return Err("Windows portable identity differs from the formal candidate".into());
    }
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    extract(archive, &receipt, directory.path())?;
    let inspection: Inspection = read(inspection_path)?;
    if inspection.schema_version != 1
        || inspection.receipt != receipt
        || inspection.scope != "archive_bytes_and_startup_only"
        || inspection.host_os != "windows"
        || inspection.host_arch != "x86_64"
        || inspection.error.is_some()
    {
        return Err("Windows inspection does not describe these staged bytes".into());
    }
    let startup_executed = match inspection.status.as_str() {
        "not_run"
            if inspection.observations.is_empty()
                && inspection.startup_environment.is_empty()
                && inspection.startup_directory.is_none() =>
        {
            false
        }
        "passed" => {
            let expected = [
                vec!["--version"],
                vec!["--help"],
                vec!["run", "--help"],
                vec!["serve", "--help"],
            ];
            if inspection.observations.len() != expected.len()
                || inspection
                    .observations
                    .iter()
                    .zip(expected)
                    .any(|(actual, args)| {
                        actual.arguments != args
                            || actual.exit_code != Some(0)
                            || actual.stdout.trim().is_empty()
                            || (args == ["--version"]
                                && actual.stdout.trim() != format!("ferrum {version}"))
                    })
                || inspection.startup_directory.is_none()
                || inspection
                    .startup_environment
                    .get("SystemRoot")
                    .is_none_or(String::is_empty)
            {
                return Err("Windows startup report omits successful executable probes".into());
            }
            let system = &inspection.startup_environment["SystemRoot"];
            if inspection.startup_environment.get("WINDIR") != Some(system)
                || inspection.startup_environment.get("PATH")
                    != Some(&format!("{system}\\System32"))
                || inspection.startup_environment.keys().any(|key| {
                    !matches!(
                        key.as_str(),
                        "SystemRoot" | "WINDIR" | "PATH" | "TEMP" | "TMP"
                    )
                })
            {
                return Err("Windows startup environment includes unverified loader paths".into());
            }
            true
        }
        _ => return Err("Windows startup is neither passed nor explicitly deferred".into()),
    };
    Ok(StagedPortable {
        archive_sha256: receipt.archive_sha256,
        archive_size_bytes: receipt.archive_size_bytes,
        binary_sha256: installation::sha256(&directory.path().join("ferrum.exe"))?,
        manifest_sha256: installation::sha256(&directory.path().join(MANIFEST))?,
        startup_executed,
    })
}

pub(super) fn verify_launcher(path: &Path) -> Result<(), String> {
    verify_pe(path, false)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Inspection {
    schema_version: u32,
    /// passed means these archive/startup checks, not full release acceptance.
    status: String,
    scope: String,
    receipt: Receipt,
    host_os: String,
    host_arch: String,
    directory: PathBuf,
    startup_environment: BTreeMap<String, String>,
    startup_directory: Option<PathBuf>,
    observations: Vec<Observation>,
    error: Option<String>,
}

fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

// Apply Windows filename rules even when packaging or auditing on Unix.
fn component(value: &str) -> bool {
    let stem = value.split('.').next().unwrap_or("").to_ascii_uppercase();
    !value.is_empty()
        && value.is_ascii()
        && value != "."
        && value != ".."
        && !value.ends_with(['.', ' '])
        && !value.starts_with('-')
        && !value
            .chars()
            .any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
        && !matches!(
            stem.as_str(),
            "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
        )
        && !(stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}

fn validate(manifest: &Manifest) -> Result<(), String> {
    let build = &manifest.build;
    let version = semver::Version::parse(&build.version)
        .map_err(|error| format!("invalid portable build version: {error}"))?;
    if version.to_string() != build.version
        || !matches!(build.source_commit.len(), 40 | 64)
        || !build
            .source_commit
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || build.build_id.is_empty()
        || !build
            .build_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(
            "portable build requires a canonical version, full source commit and build identifier"
                .into(),
        );
    }
    if manifest.schema_version != 1
        || manifest.backend != Backend::Cuda
        || manifest.target_triple != TARGET
    {
        return Err("portable payload requires CUDA and x86_64-pc-windows-msvc identity".into());
    }
    if manifest.cuda_compute_capability.is_empty()
        || !manifest
            .cuda_compute_capability
            .bytes()
            .all(|b| b.is_ascii_digit())
        || !manifest.cargo_features.iter().any(|f| f == "cuda")
        || manifest.cargo_features.iter().any(|f| f == "metal")
    {
        return Err("portable CUDA build declaration is incomplete or inconsistent".into());
    }
    let mut features = BTreeSet::new();
    if manifest.cargo_features.iter().any(|f| {
        f.is_empty()
            || !f
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
            || !features.insert(f)
    }) {
        return Err("invalid or duplicate Cargo feature".into());
    }
    if manifest.files.is_empty() || manifest.files.len() > MAX_FILES {
        return Err("portable file inventory is empty or exceeds its bound".into());
    }
    let mut names = BTreeSet::from([MANIFEST.to_owned()]);
    let mut binary_count = 0;
    let mut roles = BTreeSet::new();
    let mut total = 0u64;
    for file in &manifest.files {
        let parts: Vec<_> = file.path.split('/').collect();
        let valid_path = match (file.role, parts.as_slice()) {
            (Role::License, ["licenses", name]) => component(name),
            (Role::Binary, ["ferrum.exe"]) => {
                binary_count += 1;
                true
            }
            (Role::CudaRuntime | Role::MsvcRuntime, [name]) => {
                component(name) && name.to_ascii_lowercase().ends_with(".dll")
            }
            _ => false,
        };
        if !valid_path || !names.insert(file.path.to_lowercase()) {
            return Err(format!(
                "unsafe or case-colliding portable path: {}",
                file.path
            ));
        }
        if !hash(&file.sha256) || file.size_bytes == 0 {
            return Err(format!(
                "missing portable file byte identity: {}",
                file.path
            ));
        }
        total = total
            .checked_add(file.size_bytes)
            .ok_or("payload size overflow")?;
        if total > MAX_PAYLOAD {
            return Err("portable payload exceeds its size bound".into());
        }
        if file.role != Role::License {
            let license = file
                .license
                .as_ref()
                .ok_or("runtime input has no license")?;
            if !manifest
                .files
                .iter()
                .any(|f| f.role == Role::License && &f.path == license)
            {
                return Err(format!("runtime input license is absent: {}", file.path));
            }
        } else if file.license.is_some() {
            return Err("license entries must not reference another license".into());
        }
        let lower = file.path.to_ascii_lowercase();
        if lower == "nvcuda.dll" || lower.contains("nvcc") {
            return Err("driver/developer files are not portable runtime inputs".into());
        }
        roles.insert(match file.role {
            Role::CudaRuntime => 1,
            Role::MsvcRuntime => 2,
            _ => 0,
        });
    }
    if binary_count != 1 || !roles.contains(&1) || !roles.contains(&2) {
        return Err(
            "portable CUDA payload requires one ferrum.exe, CUDA and MSVC runtime files".into(),
        );
    }
    Ok(())
}

fn verify_file(path: &Path, expected: &PayloadFile) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file()
        || metadata.len() != expected.size_bytes
        || installation::sha256(path)? != expected.sha256
    {
        return Err(format!(
            "portable file identity mismatch: {}",
            expected.path
        ));
    }
    if expected.role != Role::License {
        verify_pe(path, expected.role != Role::Binary)?;
    }
    Ok(())
}

/// Inspect just the target identity, not imports or DLL dependency closure.
fn verify_pe(path: &Path, dll: bool) -> Result<(), String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut dos = [0u8; 64];
    let length = file.metadata().map_err(|e| e.to_string())?.len();
    file.read_exact(&mut dos)
        .map_err(|e| format!("PE DOS header: {e}"))?;
    if &dos[..2] != b"MZ" {
        return Err("payload is not a Windows PE image".into());
    }
    let offset = u32::from_le_bytes(dos[60..64].try_into().unwrap()) as u64;
    if offset < 64 || offset.checked_add(24 + 112).is_none_or(|end| end > length) {
        return Err("invalid PE header offset".into());
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    let mut header = [0u8; 26];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let word = |at| u16::from_le_bytes([header[at], header[at + 1]]);
    if &header[..4] != b"PE\0\0"
        || word(4) != 0x8664
        || word(20) < 112
        || offset + 24 + u64::from(word(20)) > length
        || word(24) != 0x20b
        || word(22) & 0x2 == 0
        || (word(22) & 0x2000 != 0) != dll
    {
        return Err("payload PE machine/type disagrees with AMD64 executable/DLL role".into());
    }
    Ok(())
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn write_new(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut file, value).map_err(|e| e.to_string())?;
    file.write_all(b"\n")
        .and_then(|()| file.sync_all())
        .map_err(|e| e.to_string())
}

fn pack(spec: &PackSpec, archive: &Path, receipt: &Path) -> Result<(), String> {
    validate(&spec.manifest)?;
    if spec.sources.keys().collect::<BTreeSet<_>>()
        != spec.manifest.files.iter().map(|f| &f.path).collect()
    {
        return Err("portable sources must match the exact declared file inventory".into());
    }
    let name = archive
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| component(s) && s.ends_with(".zip"))
        .ok_or("portable archive must have a safe .zip basename")?;
    if archive.exists() || receipt.exists() {
        return Err("portable outputs must be new".into());
    }
    for file in &spec.manifest.files {
        verify_file(&spec.sources[&file.path], file)?;
    }
    let parent = archive
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    {
        let mut zip = zip::ZipWriter::new(temporary.as_file_mut());
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .last_modified_time(zip::DateTime::default())
            .unix_permissions(0o644);
        zip.start_file(MANIFEST, options)
            .map_err(|e| e.to_string())?;
        zip.write_all(&serde_json::to_vec(&spec.manifest).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        for file in &spec.manifest.files {
            zip.start_file(
                &file.path,
                options.large_file(file.size_bytes >= u32::MAX as u64),
            )
            .map_err(|e| e.to_string())?;
            let source = fs::File::open(&spec.sources[&file.path]).map_err(|e| e.to_string())?;
            let count = std::io::copy(&mut source.take(file.size_bytes + 1), &mut zip)
                .map_err(|e| e.to_string())?;
            if count != file.size_bytes {
                return Err("portable source size changed while packing".into());
            }
            verify_file(&spec.sources[&file.path], file)?;
        }
        zip.finish().map_err(|e| e.to_string())?;
    }
    temporary.as_file().sync_all().map_err(|e| e.to_string())?;
    let record = Receipt {
        schema_version: 1,
        archive_name: name.into(),
        archive_sha256: installation::sha256(temporary.path())?,
        archive_size_bytes: temporary
            .as_file()
            .metadata()
            .map_err(|e| e.to_string())?
            .len(),
        manifest: spec.manifest.clone(),
    };
    // Reopen and verify the written ZIP, not just its original source files.
    let check = tempfile::tempdir().map_err(|e| e.to_string())?;
    extract(temporary.path(), &record, check.path())?;
    temporary
        .persist_noclobber(archive)
        .map_err(|e| e.to_string())?;
    write_new(receipt, &record)
}

fn verify_receipt(archive: &Path, receipt: &Receipt) -> Result<(), String> {
    validate(&receipt.manifest)?;
    if receipt.schema_version != 1
        || !component(&receipt.archive_name)
        || !receipt.archive_name.ends_with(".zip")
        || !hash(&receipt.archive_sha256)
        || receipt.archive_size_bytes == 0
        || fs::metadata(archive).map_err(|e| e.to_string())?.len() != receipt.archive_size_bytes
        || installation::sha256(archive)? != receipt.archive_sha256
    {
        return Err("portable archive identity mismatch".into());
    }
    Ok(())
}

fn extract(archive: &Path, receipt: &Receipt, directory: &Path) -> Result<(), String> {
    verify_receipt(archive, receipt)?;
    let mut zip = zip::ZipArchive::new(fs::File::open(archive).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let expected: BTreeMap<_, _> = receipt
        .manifest
        .files
        .iter()
        .map(|f| (f.path.as_str(), f))
        .collect();
    if zip.len() != expected.len() + 1 {
        return Err("ZIP entry inventory differs from the receipt".into());
    }
    verify_central_inventory(archive, &mut zip)?;
    let mut names = BTreeSet::new();
    // Validate all names/types before allowing any entry to create a file.
    for i in 0..zip.len() {
        let entry = zip.by_index(i).map_err(|e| e.to_string())?;
        if !names.insert(entry.name().to_lowercase())
            || entry.is_dir()
            || entry.is_symlink()
            || entry.encrypted()
            || entry
                .unix_mode()
                .is_some_and(|m| !matches!(m & 0o170000, 0 | 0o100000))
            || (entry.name() != MANIFEST && !expected.contains_key(entry.name()))
        {
            return Err("ZIP contains an undeclared, duplicate or unsafe entry".into());
        }
        let size = expected.get(entry.name()).map(|f| f.size_bytes);
        if size.is_some_and(|size| entry.size() != size)
            || (entry.name() == MANIFEST && entry.size() > 1024 * 1024)
        {
            return Err("ZIP entry size differs from the declared payload".into());
        }
    }
    let mut embedded = Vec::new();
    zip.by_name(MANIFEST)
        .map_err(|e| e.to_string())?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut embedded)
        .map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_slice(&embedded).map_err(|e| e.to_string())?;
    if manifest != receipt.manifest {
        return Err("embedded portable manifest differs from receipt".into());
    }
    let mut manifest_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(MANIFEST))
        .map_err(|e| e.to_string())?;
    manifest_file
        .write_all(&embedded)
        .and_then(|()| manifest_file.sync_all())
        .map_err(|e| e.to_string())?;
    fs::create_dir(directory.join("licenses")).map_err(|e| e.to_string())?;
    for file in &receipt.manifest.files {
        let mut entry = zip.by_name(&file.path).map_err(|e| e.to_string())?;
        let output = directory.join(&file.path);
        let mut destination = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|e| e.to_string())?;
        let count = std::io::copy(
            &mut entry.by_ref().take(file.size_bytes + 1),
            &mut destination,
        )
        .map_err(|e| e.to_string())?;
        destination.sync_all().map_err(|e| e.to_string())?;
        if count != file.size_bytes {
            return Err("ZIP decompressed size differs from payload".into());
        }
        verify_file(&output, file)?;
    }
    verify_receipt(archive, receipt)
}

// zip 7.2 indexes the central directory by name and silently collapses exact
// duplicates. Check the framing of its parsed entries so a hidden duplicate
// cannot disappear from our exact inventory. Decoding remains zip's job.
fn verify_central_inventory(
    path: &Path,
    zip: &mut zip::ZipArchive<fs::File>,
) -> Result<(), String> {
    let mut positions = Vec::with_capacity(zip.len());
    for index in 0..zip.len() {
        positions.push(
            zip.by_index(index)
                .map_err(|e| e.to_string())?
                .central_header_start(),
        );
    }
    positions.sort_unstable();
    let mut position = zip.central_directory_start();
    let mut raw = fs::File::open(path).map_err(|e| e.to_string())?;
    for recorded in positions {
        if recorded != position {
            return Err("ZIP central inventory contains a hidden or duplicate entry".into());
        }
        raw.seek(SeekFrom::Start(position))
            .map_err(|e| e.to_string())?;
        let mut header = [0u8; 46];
        raw.read_exact(&mut header).map_err(|e| e.to_string())?;
        if &header[..4] != b"PK\x01\x02" {
            return Err("invalid ZIP central entry framing".into());
        }
        position = position
            .checked_add(
                46 + [28, 30, 32]
                    .into_iter()
                    .map(|at| u64::from(u16::from_le_bytes([header[at], header[at + 1]])))
                    .sum::<u64>(),
            )
            .ok_or("ZIP central size overflow")?;
    }
    raw.seek(SeekFrom::Start(position))
        .map_err(|e| e.to_string())?;
    let mut terminal = [0u8; 4];
    raw.read_exact(&mut terminal).map_err(|e| e.to_string())?;
    if terminal != *b"PK\x05\x06" && terminal != *b"PK\x06\x06" {
        return Err("ZIP central inventory has undeclared entries after the accepted files".into());
    }
    Ok(())
}

async fn inspect(
    archive: &Path,
    receipt: Receipt,
    directory: &Path,
    output: &Path,
    extract_only: bool,
) -> Result<(), String> {
    verify_receipt(archive, &receipt)?;
    fs::create_dir(directory).map_err(|e| e.to_string())?;
    let mut report = Inspection {
        schema_version: 1,
        status: "failed".into(),
        scope: "archive_bytes_and_startup_only".into(),
        receipt,
        host_os: std::env::consts::OS.into(),
        host_arch: std::env::consts::ARCH.into(),
        directory: directory.canonicalize().map_err(|e| e.to_string())?,
        startup_environment: BTreeMap::new(),
        startup_directory: None,
        observations: Vec::new(),
        error: None,
    };
    let result = async {
        extract(archive, &report.receipt, &report.directory)?;
        if extract_only {
            report.status = "not_run".into();
            return Ok(());
        }
        if !cfg!(all(windows, target_arch = "x86_64")) {
            return Err(
                "Windows AMD64 startup must run on a native Windows AMD64 host; use --extract-only"
                    .into(),
            );
        }
        let environment = startup_environment()?;
        // An empty independent cwd cannot accidentally supply a developer DLL.
        let working_directory = tempfile::tempdir().map_err(|e| e.to_string())?;
        report.startup_directory = Some(
            working_directory
                .path()
                .canonicalize()
                .map_err(|e| e.to_string())?,
        );
        report.startup_environment = environment
            .iter()
            .map(|(key, value)| {
                Ok((
                    key.to_str()
                        .ok_or("non-UTF8 startup environment key")?
                        .to_owned(),
                    value
                        .to_str()
                        .ok_or("non-UTF8 startup environment value")?
                        .to_owned(),
                ))
            })
            .collect::<Result<_, String>>()?;
        for arguments in [
            &["--version"][..],
            &["--help"],
            &["run", "--help"],
            &["serve", "--help"],
        ] {
            for file in &report.receipt.manifest.files {
                verify_file(&report.directory.join(&file.path), file)?;
            }
            let observation = installation::probe_with_environment(
                &report.directory.join("ferrum.exe"),
                arguments,
                Some(&environment),
                Some(working_directory.path()),
            )
            .await?;
            let valid = observation.exit_code == Some(0)
                && !observation.stdout.trim().is_empty()
                && (arguments != ["--version"]
                    || observation.stdout.trim()
                        == format!("ferrum {}", report.receipt.manifest.build.version));
            report.observations.push(observation);
            if !valid {
                return Err(format!("portable startup probe {arguments:?} failed"));
            }
        }
        for file in &report.receipt.manifest.files {
            verify_file(&report.directory.join(&file.path), file)?;
        }
        report.status = "passed".into();
        Ok::<(), String>(())
    }
    .await;
    if let Err(error) = &result {
        report.error = Some(error.clone());
    }
    write_new(output, &report)?;
    result
}

fn startup_environment() -> Result<BTreeMap<std::ffi::OsString, std::ffi::OsString>, String> {
    let root = std::env::var_os("SystemRoot").ok_or("Windows SystemRoot is missing")?;
    let root_path = Path::new(&root);
    if !root_path.is_absolute() || !root_path.join("System32").is_dir() {
        return Err("Windows SystemRoot is not an existing absolute system directory".into());
    }
    let system_path = root_path.join("System32").into_os_string();
    let mut environment = BTreeMap::from([
        ("SystemRoot".into(), root.clone()),
        ("WINDIR".into(), root),
        ("PATH".into(), system_path),
    ]);
    for key in ["TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), value);
        }
    }
    Ok(environment)
}

pub async fn execute(args: PortableArgs) -> Result<(), String> {
    match args.action {
        Action::Pack {
            spec,
            archive,
            receipt,
        } => pack(&read(&spec)?, &archive, &receipt),
        Action::Inspect {
            archive,
            receipt,
            extract_dir,
            output,
            extract_only,
        } => {
            inspect(
                &archive,
                read(&receipt)?,
                &extract_dir,
                &output,
                extract_only,
            )
            .await
        }
    }
}

#[cfg(test)]
#[path = "portable_tests.rs"]
mod tests;
