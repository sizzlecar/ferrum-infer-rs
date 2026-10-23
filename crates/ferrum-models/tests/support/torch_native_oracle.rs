//! Test-only native deployment oracle. Neither compilation nor LibTorch is a
//! product dependency. All paths, ABI, loader directories and expected versions
//! are explicit; the child does not inherit Ferrum's CUDA toolkit library path.
use anyhow::{bail, ensure, Context, Result};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBytes};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const SOURCE: &str = include_str!("../native/torch_philox_oracle.cpp");
pub(crate) const METADATA_LIMIT: u64 = 64 * 1024;
/// Fixed test sources only; callers do not supply executable source from a
/// request or environment. Source identity remains in the build receipt.
#[derive(Clone, Copy)]
pub(crate) struct NativeSource {
    pub filename: &'static str,
    pub text: &'static str,
    pub required_header: &'static str,
    pub cache_prefix: &'static str,
}
const PHILOX_SOURCE: NativeSource = NativeSource {
    filename: "torch_philox_oracle.cpp",
    text: SOURCE,
    required_header: "ATen/cuda/CUDAGeneratorImpl.h",
    cache_prefix: "torch-philox",
};
const TORCH_LIBRARIES: [&str; 5] = [
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libc10_cuda.so",
    "libc10.so",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub torch_root: PathBuf,
    pub library_directories: Vec<PathBuf>,
    pub executable: Executable,
    pub cxx11_abi: u8,
    pub expected_torch_version: String,
    pub expected_cuda_runtime_soname: String,
    pub deployment_profile: String,
    pub device: u32,
    pub timeout_seconds: u64,
    pub maximum_payload_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum Executable {
    Build {
        compiler: PathBuf,
        cache_directory: PathBuf,
    },
    Prebuilt {
        path: PathBuf,
        sha256: String,
    },
}

impl Config {
    pub fn from_test_environment() -> Result<Self> {
        let path = std::env::var_os("FERRUM_TEST_TORCH_ORACLE_CONFIG")
            .context("explicit test configuration FERRUM_TEST_TORCH_ORACLE_CONFIG is required")?;
        let config: Self =
            serde_json::from_slice(&read_bounded(Path::new(&path), METADATA_LIMIT)?)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.cxx11_abi <= 1, "cxx11_abi must explicitly be 0 or 1");
        ensure!(self.torch_root.is_absolute(), "torch_root must be absolute");
        ensure!(
            !self.library_directories.is_empty() && self.library_directories.len() <= 32,
            "explicit bounded library_directories are required"
        );
        ensure!(
            self.library_directories.iter().all(|p| p.is_absolute()),
            "loader directories must be absolute"
        );
        ensure!(
            !self.expected_torch_version.is_empty() && !self.deployment_profile.is_empty(),
            "version and deployment profile are required"
        );
        ensure!(
            self.expected_cuda_runtime_soname
                .starts_with("libcudart.so.")
                && !self.expected_cuda_runtime_soname.contains('/'),
            "expected CUDA runtime must be an explicit SONAME"
        );
        ensure!(
            self.device <= i8::MAX as u32,
            "Torch device index is out of range"
        );
        ensure!(
            (1..=600).contains(&self.timeout_seconds),
            "timeout must be 1..=600 seconds"
        );
        ensure!(
            (2..=16 * 1024 * 1024).contains(&self.maximum_payload_bytes),
            "oracle payload bound must be 2 bytes..=16 MiB"
        );
        match &self.executable {
            Executable::Build {
                compiler,
                cache_directory,
            } => {
                ensure!(
                    compiler.is_absolute() && cache_directory.is_absolute(),
                    "compiler and cache directory must be absolute"
                );
            }
            Executable::Prebuilt { path, sha256 } => {
                ensure!(
                    path.is_absolute() && valid_sha(sha256),
                    "prebuilt executable requires absolute path and lowercase SHA256"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn child(&self, executable: &Path) -> Result<Command> {
        let mut command = Command::new(executable);
        // Keep CUDA_VISIBLE_DEVICES: oracle and Ferrum must see the same ordinal.
        command.env(
            "LD_LIBRARY_PATH",
            std::env::join_paths(&self.library_directories)?,
        );
        command.env_remove("LD_PRELOAD").env_remove("LD_AUDIT");
        // Both the compiler scan/build and oracle execution use this path.
        // Do not let an inherited SDK search path silently satisfy headers or
        // libraries absent from the explicit configuration and compiler.
        for name in [
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "C_INCLUDE_PATH",
            "LIBRARY_PATH",
        ] {
            command.env_remove(name);
        }
        command.stdin(Stdio::null());
        Ok(command)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BuildReceipt {
    pub source_sha256: String,
    pub input_identity_sha256: String,
    pub executable_sha256: String,
    pub compiler: PathBuf,
    pub cxx11_abi: u8,
    pub flags: Vec<String>,
    pub input_sha256: BTreeMap<PathBuf, String>,
}

pub struct Oracle {
    pub config: Config,
    pub executable: PathBuf,
    pub executable_sha256: String,
    pub build_receipt: Option<BuildReceipt>,
    library_hashes: BTreeMap<PathBuf, String>,
}

impl Oracle {
    /// Build mode is suitable for a build-only ignored Rust test inside the
    /// configured Linux toolchain container. Prebuilt mode needs no compiler.
    pub fn prepare(config: Config) -> Result<Self> {
        Self::prepare_source(config, PHILOX_SOURCE)
    }

    pub(crate) fn prepare_source(config: Config, source: NativeSource) -> Result<Self> {
        config.validate()?;
        ensure!(
            cfg!(target_os = "linux"),
            "native Torch oracle requires Linux /proc/self/maps"
        );
        let (executable, executable_sha256, build_receipt) = match &config.executable {
            Executable::Prebuilt { path, sha256 } => {
                let actual = file_sha(path)?;
                ensure!(
                    &actual == sha256,
                    "prebuilt oracle executable SHA256 mismatch"
                );
                (path.canonicalize()?, actual, None)
            }
            Executable::Build {
                compiler,
                cache_directory,
            } => {
                let (path, receipt) = build(&config, compiler, cache_directory, source)?;
                (path, receipt.executable_sha256.clone(), Some(receipt))
            }
        };
        Ok(Self {
            config,
            executable,
            executable_sha256,
            build_receipt,
            library_hashes: BTreeMap::new(),
        })
    }

    pub fn generate(&mut self, case: Case, host: &HostMemoryAccount) -> Result<Output> {
        let bytes = case.payload_bytes()?;
        ensure!(
            bytes <= self.config.maximum_payload_bytes,
            "case exceeds explicitly configured payload bound"
        );
        let directory = tempfile::tempdir()?;
        let output_path = directory.path().join("stdout.bin");
        let stderr_path = directory.path().join("stderr.txt");
        let mut command = self.config.child(&self.executable)?;
        command.args([
            self.config.device.to_string(),
            case.seed.to_string(),
            case.offset.to_string(),
            case.channels.to_string(),
            case.height.to_string(),
            case.width.to_string(),
            case.draws.to_string(),
        ]);
        let output_limit = 16 + METADATA_LIMIT + u64::from(case.draws) * (24 + bytes);
        run(
            &mut command,
            &output_path,
            &stderr_path,
            self.config.timeout_seconds,
            output_limit,
        )?;
        let mut output = parse_output(&output_path, &self.config, case, host)?;
        output.library_sha256 =
            self.validate_mapped_libraries(&output.metadata.loaded_libraries)?;
        Ok(output)
    }

    pub(crate) fn validate_mapped_libraries(
        &mut self,
        libraries: &[PathBuf],
    ) -> Result<BTreeMap<PathBuf, String>> {
        ensure!(
            libraries.len() <= 1024 && libraries.iter().all(|p| p.is_absolute()),
            "invalid mapped-library metadata"
        );
        let mapped: BTreeSet<PathBuf> = libraries
            .iter()
            .map(|path| {
                path.canonicalize()
                    .with_context(|| format!("mapped library is unavailable: {}", path.display()))
            })
            .collect::<Result<_>>()?;
        for name in TORCH_LIBRARIES {
            let expected = self
                .config
                .torch_root
                .join("lib")
                .join(name)
                .canonicalize()?;
            ensure!(
                mapped.contains(&expected),
                "oracle did not map configured {name}: {}",
                expected.display()
            );
        }
        ensure!(
            mapped.iter().any(|p| p
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == self.config.expected_cuda_runtime_soname
                    || n.starts_with(&format!("{}.", self.config.expected_cuda_runtime_soname)))),
            "oracle did not map expected CUDA runtime SONAME"
        );
        // Hash actual shared objects once per oracle instance; stream their
        // contents, rather than allocating a library-sized payload buffer.
        let mut hashes = BTreeMap::new();
        for path in mapped {
            if !self.library_hashes.contains_key(&path) {
                self.library_hashes.insert(path.clone(), file_sha(&path)?);
            }
            hashes.insert(path.clone(), self.library_hashes[&path].clone());
        }
        Ok(hashes)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub seed: u64,
    pub offset: u64,
    pub channels: u64,
    pub height: u64,
    pub width: u64,
    pub draws: u32,
}
impl Case {
    pub fn payload_bytes(self) -> Result<u64> {
        ensure!(
            self.channels != 0 && self.height != 0 && self.width != 0,
            "zero dimension"
        );
        ensure!(
            (1..=2).contains(&self.draws) && self.offset % 4 == 0,
            "unsupported draw count or Torch offset"
        );
        let bytes = self
            .channels
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(self.width))
            .and_then(|n| n.checked_mul(2))
            .context("extent overflow")?;
        ensure!(
            bytes <= i32::MAX as u64,
            "oracle scope is 32-bit contiguous TensorIterator"
        );
        Ok(bytes)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub torch_version: String,
    pub torch_config: String,
    pub cxx11_abi: u8,
    pub device: u32,
    pub seed: u64,
    pub offset: u64,
    pub draws: u32,
    pub shape: [u64; 5],
    pub dtype: String,
    pub loaded_libraries: Vec<PathBuf>,
}
pub struct Draw {
    pub offset_before: u64,
    pub offset_after: u64,
    pub bytes: ManagedHostBytes,
}
pub struct Output {
    pub metadata: Metadata,
    pub draws: Vec<Draw>,
    pub library_sha256: BTreeMap<PathBuf, String>,
}

fn parse_output(
    path: &Path,
    config: &Config,
    case: Case,
    host: &HostMemoryAccount,
) -> Result<Output> {
    let expected_bytes = case.payload_bytes()?;
    let mut file = File::open(path)?;
    let mut header = [0_u8; 16];
    file.read_exact(&mut header)?;
    ensure!(
        &header[..8] == b"FRNGOR01" && u32::from_le_bytes(header[8..12].try_into()?) == 1,
        "invalid oracle protocol"
    );
    let metadata_len = u32::from_le_bytes(header[12..16].try_into()?) as u64;
    ensure!(
        metadata_len <= METADATA_LIMIT,
        "oracle metadata exceeds limit"
    );
    ensure!(
        file.metadata()?.len() == 16 + metadata_len + u64::from(case.draws) * (24 + expected_bytes),
        "oracle file is truncated or has trailing data"
    );
    let mut encoded = vec![0; metadata_len as usize];
    file.read_exact(&mut encoded)?;
    let metadata: Metadata = serde_json::from_slice(&encoded)?;
    ensure!(
        metadata.torch_version == config.expected_torch_version
            && metadata.cxx11_abi == config.cxx11_abi,
        "deployed Torch version or ABI mismatch"
    );
    ensure!(
        metadata.device == config.device
            && metadata.seed == case.seed
            && metadata.offset == case.offset
            && metadata.draws == case.draws,
        "oracle generator metadata mismatch"
    );
    ensure!(
        metadata.shape == [1, 1, case.channels, case.height, case.width]
            && metadata.dtype == "BF16",
        "oracle tensor metadata mismatch"
    );
    ensure!(
        metadata.loaded_libraries.len() <= 1024
            && metadata.loaded_libraries.iter().all(|p| p.is_absolute()),
        "invalid mapped-library metadata"
    );
    let mut draws = Vec::new();
    let mut offset = case.offset;
    for _ in 0..case.draws {
        let mut record = [0_u8; 24];
        file.read_exact(&mut record)?;
        let before = u64::from_le_bytes(record[..8].try_into()?);
        let after = u64::from_le_bytes(record[8..16].try_into()?);
        let len = u64::from_le_bytes(record[16..].try_into()?);
        ensure!(
            before == offset && after > before && len == expected_bytes,
            "invalid oracle draw offsets or extent"
        );
        let mut storage = host.allocate_zeroed(HostMemoryClass::Output, usize::try_from(len)?)?;
        file.read_exact(storage.as_mut())?;
        draws.push(Draw {
            offset_before: before,
            offset_after: after,
            bytes: storage.freeze(),
        });
        offset = after;
    }
    Ok(Output {
        metadata,
        draws,
        library_sha256: BTreeMap::new(),
    })
}

fn build(
    config: &Config,
    compiler: &Path,
    cache: &Path,
    program: NativeSource,
) -> Result<(PathBuf, BuildReceipt)> {
    fs::create_dir_all(cache)?;
    let directory = tempfile::tempdir_in(cache)?;
    let source = directory.path().join(program.filename);
    fs::write(&source, program.text)?;
    let compiler = compiler.canonicalize()?;
    let flags = vec![
        "-std=c++20".into(),
        "-O2".into(),
        format!("-D_GLIBCXX_USE_CXX11_ABI={}", config.cxx11_abi),
        format!("-I{}", config.torch_root.join("include").display()),
        format!(
            "-I{}",
            config
                .torch_root
                .join("include/torch/csrc/api/include")
                .display()
        ),
    ];
    let mut scan = config.child(&compiler)?;
    scan.args(&flags)
        .args(["-M", "-MT", "ferrum_torch_oracle"])
        .arg(&source);
    let deps = directory.path().join("dependencies.txt");
    let errors = directory.path().join("stderr.txt");
    run(
        &mut scan,
        &deps,
        &errors,
        config.timeout_seconds,
        4 * 1024 * 1024,
    )?;
    let mut inputs = BTreeMap::new();
    for path in dependencies(&String::from_utf8(read_bounded(&deps, 4 * 1024 * 1024)?)?)? {
        let path = path.canonicalize()?;
        if path != source.canonicalize()? {
            inputs.insert(path.clone(), file_sha(&path)?);
        }
    }
    ensure!(
        inputs.keys().any(|p| p.ends_with(program.required_header)),
        "dependency scan omitted required native oracle header"
    );
    inputs.insert(compiler.clone(), file_sha(&compiler)?);
    let libraries = TORCH_LIBRARIES.map(|name| config.torch_root.join("lib").join(name));
    for path in &libraries {
        inputs.insert(path.canonicalize()?, file_sha(path)?);
    }
    let source_sha256 = bytes_sha(program.text.as_bytes());
    let mut link_flags = vec!["-Xlinker".into(), "--no-as-needed".into()];
    link_flags.extend(libraries.iter().map(|p| p.to_string_lossy().into_owned()));
    link_flags.extend(["-Xlinker".into(), "--as-needed".into(), "-pthread".into()]);
    for path in &config.library_directories {
        link_flags.extend([
            "-Xlinker".into(),
            "-rpath".into(),
            "-Xlinker".into(),
            path.to_string_lossy().into_owned(),
            "-Xlinker".into(),
            "-rpath-link".into(),
            "-Xlinker".into(),
            path.to_string_lossy().into_owned(),
        ]);
    }
    let all_flags = flags.iter().chain(&link_flags).cloned().collect::<Vec<_>>();
    let identity = bytes_sha(&serde_json::to_vec(&(
        "torch-oracle-build-v1",
        &source_sha256,
        &inputs,
        &all_flags,
    ))?);
    let destination = cache.join(format!("{}-{identity}", program.cache_prefix));
    let receipt_path = destination.with_extension("json");
    if destination.is_file() && receipt_path.is_file() {
        let receipt: BuildReceipt =
            serde_json::from_slice(&read_bounded(&receipt_path, 4 * 1024 * 1024)?)?;
        ensure!(
            receipt.input_identity_sha256 == identity
                && receipt.executable_sha256 == file_sha(&destination)?,
            "cached oracle identity mismatch"
        );
        return Ok((destination, receipt));
    }
    let binary = directory.path().join("oracle");
    let mut compile = config.child(&compiler)?;
    compile
        .args(&flags)
        .arg(&source)
        .args(&link_flags)
        .arg("-o")
        .arg(&binary);
    run(
        &mut compile,
        &directory.path().join("compile.stdout"),
        &errors,
        config.timeout_seconds,
        METADATA_LIMIT,
    )?;
    let receipt = BuildReceipt {
        source_sha256,
        input_identity_sha256: identity,
        executable_sha256: file_sha(&binary)?,
        compiler,
        cxx11_abi: config.cxx11_abi,
        flags: all_flags,
        input_sha256: inputs,
    };
    fs::rename(binary, &destination)?;
    fs::write(receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok((destination, receipt))
}

pub(crate) fn run(
    command: &mut Command,
    stdout: &Path,
    stderr: &Path,
    seconds: u64,
    output_limit: u64,
) -> Result<()> {
    // A compiler driver forks cc1plus/as/ld. Give every invocation a private
    // group so timeout/error cleanup also reaches those descendants. This is
    // test-only Linux glue; no new product dependency or shell wrapper.
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    ensure!(
        cfg!(target_os = "linux"),
        "process-group oracle execution requires Linux"
    );
    struct ChildGuard {
        child: std::process::Child,
        reaped: bool,
    }
    impl ChildGuard {
        fn terminate(&mut self) -> std::io::Result<()> {
            if self.reaped {
                return Ok(());
            }
            #[cfg(target_os = "linux")]
            let signal = {
                unsafe extern "C" {
                    fn kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
                }
                let pid = i32::try_from(self.child.id())
                    .map_err(|_| std::io::Error::other("child PID out of range"))?;
                // SAFETY: process_group(0) made this unreaped child the group
                // leader. Its PID cannot be reused while owned and unreaped.
                let result = unsafe { kill(-pid, 9) };
                if result == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            };
            #[cfg(not(target_os = "linux"))]
            let signal = self.child.kill();
            if signal.is_err() {
                let _ = self.child.kill();
            }
            let waited = self.child.wait();
            self.reaped = waited.is_ok();
            signal.and(waited.map(|_| ()))
        }
    }
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            // Covers polling/file-I/O failures as well as ordinary timeout.
            // A reaped successful child is disarmed: do not signal a reused PID.
            let _ = self.terminate();
        }
    }
    command
        .stdout(Stdio::from(File::create(stdout)?))
        .stderr(Stdio::from(File::create(stderr)?));
    let mut child = ChildGuard {
        child: command
            .spawn()
            .context("launch native oracle/build process")?,
        reaped: false,
    };
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.child.try_wait()? {
            child.reaped = true;
            break status;
        }
        if start.elapsed() >= Duration::from_secs(seconds)
            || fs::metadata(stdout)?.len() > output_limit
            || fs::metadata(stderr)?.len() > 4 * 1024 * 1024
        {
            child
                .terminate()
                .context("terminate and reap oracle/compiler process group")?;
            bail!("native oracle/build exceeded timeout or output limit");
        }
        thread::sleep(Duration::from_millis(25));
    };
    ensure!(
        fs::metadata(stdout)?.len() <= output_limit,
        "native stdout exceeded bound"
    );
    ensure!(
        fs::metadata(stderr)?.len() <= 4 * 1024 * 1024,
        "native stderr exceeded bound"
    );
    if !status.success() {
        let mut errors = String::new();
        File::open(stderr)?
            .take(METADATA_LIMIT)
            .read_to_string(&mut errors)?;
        bail!("native oracle/build failed ({status}): {errors}");
    }
    Ok(())
}

fn dependencies(source: &str) -> Result<Vec<PathBuf>> {
    let normalized = source.replace("\\\r\n", " ").replace("\\\n", " ");
    let (target, body) = normalized
        .split_once(':')
        .context("missing dependency target")?;
    ensure!(
        target.trim() == "ferrum_torch_oracle",
        "unexpected dependency target"
    );
    let mut words = Vec::new();
    let mut word = String::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\'
            && chars
                .peek()
                .is_some_and(|n| n.is_whitespace() || matches!(*n, '#' | ':' | '\\'))
        {
            word.push(chars.next().unwrap());
        } else if c.is_whitespace() {
            if !word.is_empty() {
                words.push(PathBuf::from(std::mem::take(&mut word)));
            }
        } else if c == '$' && chars.peek() == Some(&'$') {
            chars.next();
            word.push('$');
        } else {
            word.push(c);
        }
    }
    if !word.is_empty() {
        words.push(PathBuf::from(word));
    }
    ensure!(!words.is_empty(), "empty dependency list");
    Ok(words)
}

pub(crate) fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= limit,
        "{} exceeds size limit",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "file grew beyond size limit");
    Ok(bytes)
}
pub fn bytes_sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_sha(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("hash {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

    #[test]
    fn child_removes_implicit_search_paths_and_keeps_explicit_loader_directories() {
        let mut config = config();
        config
            .library_directories
            .push("/deployed/nvidia/cu13/lib".into());
        let command = config.child(Path::new("/test/oracle")).unwrap();
        for name in [
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "C_INCLUDE_PATH",
            "LIBRARY_PATH",
            "LD_PRELOAD",
            "LD_AUDIT",
        ] {
            let entry = command
                .get_envs()
                .find(|(key, _)| *key == std::ffi::OsStr::new(name));
            assert!(
                matches!(entry, Some((_, None))),
                "{name} was not explicitly removed"
            );
        }
        let loader = command
            .get_envs()
            .find(|(key, _)| *key == std::ffi::OsStr::new("LD_LIBRARY_PATH"))
            .and_then(|(_, value)| value)
            .unwrap();
        let expected_loader = std::env::join_paths(&config.library_directories).unwrap();
        assert_eq!(loader, expected_loader.as_os_str());
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != std::ffi::OsStr::new("PATH")),
            "the configured compiler may still need ordinary toolchain PATH"
        );
    }

    #[cfg(target_os = "linux")]
    fn worker_name() -> String {
        // Rust's test filter omits the integration-test crate name.
        format!(
            "{}::process_group_worker",
            module_path!().split_once("::").unwrap().1
        )
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "internal Rust subprocess fixture; invoked only by timeout cleanup test"]
    fn process_group_worker() {
        let directory = PathBuf::from(
            std::env::var_os("FERRUM_TEST_TORCH_PROCESS_WORKER_DIRECTORY")
                .expect("subprocess fixture directory"),
        );
        let descendant = std::env::var_os("FERRUM_TEST_TORCH_PROCESS_WORKER_DESCENDANT").is_some();
        let name = if descendant {
            "descendant.pid"
        } else {
            "leader.pid"
        };
        fs::write(directory.join(name), std::process::id().to_string()).unwrap();
        let _child = if descendant {
            None
        } else {
            // Inherit the private group made by run(); do not create a second
            // group. This models g++ launching a compiler/linker subprocess.
            Some(
                Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--ignored",
                        "--exact",
                        &worker_name(),
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env("FERRUM_TEST_TORCH_PROCESS_WORKER_DESCENDANT", "1")
                    .stdin(Stdio::null())
                    .spawn()
                    .unwrap(),
            )
        };
        loop {
            thread::sleep(Duration::from_millis(100));
        }
    }

    #[cfg(target_os = "linux")]
    fn process_still_running(pid: u32) -> bool {
        // A killed descendant may briefly be an init-owned zombie. That has no
        // executing code/GPU context; the direct leader is reaped by run().
        match fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat
                .rsplit_once(") ")
                .is_some_and(|(_, fields)| !fields.starts_with('Z') && !fields.starts_with('X')),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => panic!("inspect child process: {error}"),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn timeout_terminates_compiler_style_descendants_and_reaps_leader() {
        let directory = tempfile::tempdir().unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--ignored",
                "--exact",
                &worker_name(),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                "FERRUM_TEST_TORCH_PROCESS_WORKER_DIRECTORY",
                directory.path(),
            )
            .env_remove("FERRUM_TEST_TORCH_PROCESS_WORKER_DESCENDANT");
        let error = run(
            &mut command,
            &directory.path().join("stdout"),
            &directory.path().join("stderr"),
            3,
            64 * 1024,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("timeout or output limit"),
            "{error:#}"
        );
        let pid = |name| {
            fs::read_to_string(directory.path().join(name))
                .unwrap()
                .parse::<u32>()
                .unwrap()
        };
        let leader = pid("leader.pid");
        let descendant = pid("descendant.pid");
        let deadline = Instant::now() + Duration::from_secs(2);
        while process_still_running(descendant) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !Path::new(&format!("/proc/{leader}")).exists(),
            "leader was not reaped"
        );
        assert!(
            !process_still_running(descendant),
            "compiler-style descendant survived group timeout"
        );
    }

    fn config() -> Config {
        Config {
            torch_root: "/deployed/torch".into(),
            library_directories: vec!["/deployed/torch/lib".into()],
            executable: Executable::Prebuilt {
                path: "/test/oracle".into(),
                sha256: "a".repeat(64),
            },
            cxx11_abi: 1,
            expected_torch_version: "2.13.0".into(),
            expected_cuda_runtime_soname: "libcudart.so.13".into(),
            deployment_profile: "test-explicit-deployment".into(),
            device: 0,
            timeout_seconds: 30,
            maximum_payload_bytes: 1024,
        }
    }

    fn host(bytes: usize) -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    output_bytes: bytes,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    }

    // Protocol fixture only: these literal bits do not pretend to be a random
    // draw or an independent numerical oracle.
    fn frame(case: Case) -> Vec<u8> {
        let encoded = serde_json::to_vec(&Metadata {
            torch_version: "2.13.0".into(),
            torch_config: "fixture".into(),
            cxx11_abi: 1,
            device: 0,
            seed: case.seed,
            offset: case.offset,
            draws: case.draws,
            shape: [1, 1, case.channels, case.height, case.width],
            dtype: "BF16".into(),
            loaded_libraries: vec![],
        })
        .unwrap();
        let mut bytes = b"FRNGOR01".to_vec();
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&encoded);
        for draw in 0..case.draws {
            bytes.extend_from_slice(&(case.offset + u64::from(draw) * 4).to_le_bytes());
            bytes.extend_from_slice(&(case.offset + u64::from(draw + 1) * 4).to_le_bytes());
            bytes.extend_from_slice(&case.payload_bytes().unwrap().to_le_bytes());
            bytes.extend_from_slice(&[0, 0x80]);
        }
        bytes
    }

    #[test]
    fn configuration_requires_explicit_abi_and_pinned_prebuilt_identity() {
        let config = config();
        config.validate().unwrap();
        let mut value = serde_json::to_value(&config).unwrap();
        value.as_object_mut().unwrap().remove("cxx11_abi");
        assert!(serde_json::from_value::<Config>(value).is_err());
        assert!(Config {
            cxx11_abi: 2,
            ..config.clone()
        }
        .validate()
        .is_err());
        assert!(Config {
            executable: Executable::Prebuilt {
                path: "relative".into(),
                sha256: "wrong".into()
            },
            ..config
        }
        .validate()
        .is_err());
    }

    #[test]
    fn protocol_exact_extent_identity_and_host_failure_retire_partial_results() {
        let case = Case {
            seed: 42,
            offset: 0,
            channels: 1,
            height: 1,
            width: 1,
            draws: 2,
        };
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oracle.bin");
        let bytes = frame(case);
        fs::write(&path, &bytes).unwrap();
        let account = host(4);
        let output = parse_output(&path, &config(), case, &account).unwrap();
        assert_eq!(output.draws[0].bytes.bytes(), [0, 0x80]);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 4);
        drop(output);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
        let too_small = host(2);
        assert!(parse_output(&path, &config(), case, &too_small).is_err());
        assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
        assert!(parse_output(&path, &config(), Case { seed: 43, ..case }, &account).is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
        fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(parse_output(&path, &config(), case, &account).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        fs::write(&path, trailing).unwrap();
        assert!(parse_output(&path, &config(), case, &account).is_err());
    }
    #[test]
    fn make_dependencies_preserve_spaces_and_escaped_names() {
        assert_eq!(
            dependencies("ferrum_torch_oracle: a.cpp \\\n /torch\\ headers/a.h a$$b\\#c.h")
                .unwrap(),
            vec![
                PathBuf::from("a.cpp"),
                PathBuf::from("/torch headers/a.h"),
                PathBuf::from("a$b#c.h")
            ]
        );
        assert!(dependencies("other: a.h").is_err());
        assert!(dependencies("ferrum_torch_oracle:").is_err());
        assert_eq!(
            dependencies("ferrum_torch_oracle : a.h").unwrap(),
            [PathBuf::from("a.h")]
        );
    }
    #[test]
    fn case_rejects_offset_alignment_zero_extent_and_32_bit_overflow() {
        let valid = Case {
            seed: 1,
            offset: 4,
            channels: 64,
            height: 64,
            width: 64,
            draws: 2,
        };
        assert_eq!(valid.payload_bytes().unwrap(), 524288);
        assert!(Case { offset: 1, ..valid }.payload_bytes().is_err());
        assert!(Case { width: 0, ..valid }.payload_bytes().is_err());
        assert!(Case {
            width: u64::MAX,
            ..valid
        }
        .payload_bytes()
        .is_err());
        assert!(Case {
            channels: 1,
            height: 1,
            width: i32::MAX as u64,
            ..valid
        }
        .payload_bytes()
        .is_err());
    }
}
