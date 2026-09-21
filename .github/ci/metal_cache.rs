//! Standalone Metal CI cache lease and bounded package cleanup.
//! Build with rustc, not Cargo: maintenance must not first fill the cache.
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        unix::{
            fs::{MetadataExt, OpenOptionsExt},
            process::ExitStatusExt,
        },
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const GIB: u64 = 1024 * 1024 * 1024;
#[cfg(target_os = "macos")]
const NOFOLLOW: i32 = 0x100;
#[cfg(not(target_os = "macos"))]
const NOFOLLOW: i32 = 0x20000;
unsafe extern "C" {
    fn fcntl(fd: i32, command: i32, ...) -> i32;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CleanupMode {
    DepInfo,
    Cargo,
}

struct Options {
    workspace: PathBuf,
    target: PathBuf,
    report: PathBuf,
    max_bytes: u64,
    min_free: u64,
    cleanup_mode: CleanupMode,
    command: Vec<std::ffi::OsString>,
}

fn options() -> Result<Options, String> {
    let mut args = env::args_os().skip(1);
    let (mut workspace, mut target, mut report) = (None, None, None);
    let (mut max_bytes, mut min_free) = (48 * GIB, 16 * GIB);
    let mut cleanup_mode = CleanupMode::DepInfo;
    while let Some(arg) = args.next() {
        if arg == "--" {
            let command: Vec<_> = args.collect();
            if command.is_empty() {
                return Err("missing command after --".into());
            }
            return Ok(Options {
                workspace: workspace.ok_or("missing --workspace")?,
                target: target.ok_or("missing --target")?,
                report: report.ok_or("missing --report")?,
                max_bytes,
                min_free,
                cleanup_mode,
                command,
            });
        }
        let value = args.next().ok_or("missing option value")?;
        match arg.to_str() {
            Some("--workspace") => workspace = Some(PathBuf::from(value)),
            Some("--target") => target = Some(PathBuf::from(value)),
            Some("--report") => report = Some(PathBuf::from(value)),
            Some("--cleanup-mode") => {
                cleanup_mode = match value.to_str() {
                    Some("dep-info") => CleanupMode::DepInfo,
                    Some("cargo") => CleanupMode::Cargo,
                    _ => return Err("cleanup mode must be dep-info or cargo".into()),
                }
            }
            Some("--max-gib" | "--min-free-gib") => {
                let bytes = value
                    .to_str()
                    .and_then(|s| s.parse::<u64>().ok())
                    .and_then(|n| n.checked_mul(GIB))
                    .ok_or("invalid GiB limit")?;
                if arg == "--max-gib" {
                    max_bytes = bytes;
                } else {
                    min_free = bytes;
                }
            }
            _ => return Err(format!("unknown option {arg:?}")),
        }
    }
    Err("missing -- command".into())
}

fn note(report: &mut File, message: impl std::fmt::Display) -> Result<(), String> {
    eprintln!("Metal cache: {message}");
    writeln!(report, "{message}").map_err(|e| e.to_string())
}

fn roots(workspace: &Path, target: &Path) -> Result<(PathBuf, PathBuf), String> {
    if !workspace.is_absolute() || !target.is_absolute() {
        return Err("workspace and target must be absolute".into());
    }
    if fs::symlink_metadata(target).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err("target must be the real cache directory, not a symlink".into());
    }
    let workspace = workspace.canonicalize().map_err(|e| e.to_string())?;
    fs::create_dir_all(target).map_err(|e| e.to_string())?;
    let target = target.canonicalize().map_err(|e| e.to_string())?;
    if workspace.parent().is_none()
        || target.parent().is_none()
        || workspace.starts_with(&target)
        || target.starts_with(&workspace)
    {
        return Err("cache and workspace must be disjoint non-root directories".into());
    }
    Ok((workspace, target))
}

fn lease(target: &Path) -> Result<File, String> {
    // Never unlink this inode: replacing it would create two independent locks.
    let lock_path = target.with_file_name(format!(
        ".{}.lease",
        target.file_name().unwrap().to_string_lossy()
    ));
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .custom_flags(NOFOLLOW)
        .open(lock_path)
        .map_err(|e| e.to_string())?;
    if !file.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("lease is not a regular file".into());
    }
    file.lock().map_err(|e| e.to_string())?;
    // Rust opens files CLOEXEC. Keep the lock in the command AND descendants,
    // so killing this supervisor cannot expose a still-running test to cleanup.
    // F_SETFD=2 on supported macOS/Linux runners; zero clears FD_CLOEXEC.
    if unsafe { fcntl(file.as_raw_fd(), 2, 0) } == -1 {
        return Err(io::Error::last_os_error().to_string());
    }
    // Do not explicitly unlock. The last inherited descriptor releases it.
    Ok(file)
}

fn output(command: &mut Command) -> Result<Vec<u8>, String> {
    let result = command.output().map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!(
            "{command:?}: {}: {}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    Ok(result.stdout)
}

#[derive(Clone, Copy, Debug)]
struct Usage {
    bytes: u64,
    free: u64,
}

fn usage(target: &Path) -> Result<Usage, String> {
    let size = output(
        Command::new("du")
            .args(["-sk"])
            .arg(target)
            .env("LC_ALL", "C"),
    )?;
    let disk = output(
        Command::new("df")
            .args(["-Pk"])
            .arg(target)
            .env("LC_ALL", "C"),
    )?;
    let parse = |s: Option<&str>| {
        s.and_then(|v| v.parse::<u64>().ok())
            .and_then(|n| n.checked_mul(1024))
            .ok_or_else(|| "invalid du/df byte count".to_owned())
    };
    Ok(Usage {
        bytes: parse(
            std::str::from_utf8(&size)
                .map_err(|e| e.to_string())?
                .split_whitespace()
                .next(),
        )?,
        free: parse(
            std::str::from_utf8(&disk)
                .map_err(|e| e.to_string())?
                .lines()
                .last()
                .and_then(|line| line.split_whitespace().nth(3)),
        )?,
    })
}

struct Package {
    name: String,
    manifest_dir: PathBuf,
}

fn packages(workspace: &Path) -> Result<Vec<Package>, String> {
    let metadata = output(
        Command::new("cargo")
            .args([
                "metadata",
                "--no-deps",
                "--offline",
                "--locked",
                "--format-version",
                "1",
                "--manifest-path",
            ])
            .arg(workspace.join("Cargo.toml")),
    )?;
    // jq is already a dependency of these workflows. Avoid an incomplete TOML
    // parser or filename globs pretending to identify Cargo package ownership.
    let mut jq = Command::new("jq").args(["-er", ". as $m | .packages[] | select(.id as $id | $m.workspace_members | index($id)) | select(.name | startswith(\"ferrum-\")) | (.name + \"\t\" + .manifest_path)"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    jq.stdin
        .take()
        .unwrap()
        .write_all(&metadata)
        .map_err(|e| e.to_string())?;
    let result = jq.wait_with_output().map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!(
            "select workspace packages: {}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    let packages = String::from_utf8(result.stdout)
        .map_err(|e| e.to_string())?
        .lines()
        .map(|line| {
            let (name, manifest) = line.split_once('\t').ok_or("missing package manifest")?;
            if !name.starts_with("ferrum-")
                || name.chars().chain(manifest.chars()).any(char::is_control)
            {
                return Err("ambiguous package metadata".to_owned());
            }
            let manifest = Path::new(manifest);
            let directory = manifest.parent().ok_or("missing manifest directory")?;
            if manifest.file_name() != Some(std::ffi::OsStr::new("Cargo.toml"))
                || !directory.starts_with(workspace)
                || directory.canonicalize().map_err(|e| e.to_string())? != directory
            {
                return Err("package manifest is outside the canonical workspace".into());
            }
            Ok(Package {
                name: name.into(),
                manifest_dir: directory.into(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if packages.is_empty() {
        return Err("no Ferrum workspace packages".into());
    }
    Ok(packages)
}

// Parse only a single dep-info rule's targets, never prerequisite paths or
// filename patterns. Unknown Make syntax fails closed for this dep-info file.
fn rule_targets(line: &str) -> Option<Vec<PathBuf>> {
    let mut chars = line.chars();
    let mut token = String::new();
    let mut targets = Vec::new();
    while let Some(ch) = chars.next() {
        match ch {
            ':' => {
                if !token.is_empty() {
                    targets.push(PathBuf::from(token));
                }
                return Some(targets);
            }
            '\\' => match chars.next()? {
                escaped @ (' ' | '\t' | '\\' | '#' | ':') => token.push(escaped),
                _ => return None,
            },
            '$' if chars.next()? == '$' => token.push('$'),
            '$' | '#' => return None,
            ch if ch.is_whitespace() => {
                if !token.is_empty() {
                    targets.push(PathBuf::from(std::mem::take(&mut token)));
                }
            }
            ch => token.push(ch),
        }
    }
    None
}

fn owned_outputs(
    text: &str,
    depfile: &Path,
    directory: &Path,
    packages: &[Package],
) -> Option<BTreeSet<PathBuf>> {
    let owners: Vec<_> = text
        .lines()
        .filter_map(|line| line.strip_prefix("# env-dep:CARGO_MANIFEST_DIR="))
        .collect();
    if owners.len() != 1
        || !packages
            .iter()
            .any(|package| package.manifest_dir == Path::new(owners[0]))
    {
        return None;
    }
    let mut outputs = BTreeSet::new();
    for line in text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        for path in rule_targets(line)? {
            if path.is_absolute()
                && path.parent() == Some(directory)
                && !path
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                outputs.insert(path);
            }
        }
    }
    // rustc lists the dep-info file itself as a target. Its presence ties this
    // ownership record to this physical directory; no hash/suffix is inferred.
    outputs.contains(depfile).then_some(outputs)
}

fn clean_dep_info(target: &Path, packages: &[Package], report: &mut File) -> Result<(), String> {
    let directory = target.join("debug/deps");
    for directory in [target.join("debug"), directory.clone()] {
        match fs::symlink_metadata(&directory) {
            Ok(metadata) if metadata.is_dir() => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            _ => return Err("dep-info cleanup requires real debug/deps directories".into()),
        }
    }
    let started = Instant::now();
    let mut progress = started;
    let (mut scanned, mut read_bytes, mut matched) = (0_u64, 0_u64, 0_u64);
    let mut artifacts = BTreeMap::new();
    let mut complete = true;
    // One unsorted traversal, bounded metadata reads, and no nested targets.
    // Large legacy caches must not repeat Cargo's per-package directory sorts.
    for entry in fs::read_dir(&directory).map_err(|e| e.to_string())? {
        if started.elapsed() >= Duration::from_secs(30)
            || read_bytes >= 256 * 1024 * 1024
            || scanned >= 50_000
        {
            complete = false;
            break;
        }
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension() != Some(std::ffi::OsStr::new("d"))
            || !entry.file_type().map_err(|e| e.to_string())?.is_file()
        {
            continue;
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(NOFOLLOW)
            .open(&path)
            .map_err(|e| e.to_string())?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() > 4 * 1024 * 1024 {
            continue;
        }
        let mut bytes = Vec::new();
        file.take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        scanned += 1;
        read_bytes += bytes.len() as u64;
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        if bytes.len() <= 4 * 1024 * 1024 {
            if let Some(outputs) = owned_outputs(text, &path, &directory, packages) {
                matched += 1;
                for output in outputs {
                    if let Ok(metadata) = fs::symlink_metadata(&output) {
                        if metadata.is_file() {
                            artifacts.entry(output).or_insert((metadata, path.clone()));
                        }
                    }
                }
            }
        }
        if progress.elapsed() >= Duration::from_secs(5) {
            note(
                report,
                format!("dep_info_scan files={scanned} matched={matched} bytes={read_bytes}"),
            )?;
            progress = Instant::now();
        }
    }
    note(report, format!("dep_info_scan_complete={complete} files={scanned} matched={matched} metadata_bytes={read_bytes} selected_files={}", artifacts.len()))?;
    let mut artifacts: Vec<_> = artifacts.into_iter().collect();
    // Keep the ownership records until their large outputs have been removed,
    // including when the bounded pass is interrupted or cancelled.
    artifacts.sort_unstable_by_key(|(path, (metadata, _))| {
        (
            path.extension() == Some(std::ffi::OsStr::new("d")),
            std::cmp::Reverse(metadata.len()),
        )
    });
    let (mut removed, mut logical_bytes) = (0, 0_u64);
    for (path, (observed, depfile)) in artifacts {
        if started.elapsed() >= Duration::from_secs(120) {
            note(
                report,
                "dep_info_cleanup_budget_exhausted=true remaining_artifacts_retained",
            )?;
            break;
        }
        let current = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !current.is_file()
            || (current.dev(), current.ino(), current.len())
                != (observed.dev(), observed.ino(), observed.len())
        {
            return Err(format!("artifact changed while cache lease held: {path:?}"));
        }
        note(
            report,
            format!(
                "remove_owned_artifact_intent={path:?} dep_info={depfile:?} logical_bytes={}",
                current.len()
            ),
        )?;
        fs::remove_file(&path).map_err(|e| e.to_string())?;
        removed += 1;
        logical_bytes = logical_bytes.saturating_add(current.len());
    }
    note(report, format!("dep_info_cleanup removed_files={removed} logical_bytes={logical_bytes} elapsed_ms={} actual_free_space_checked_afterward=true", started.elapsed().as_millis()))
}

fn clean(
    workspace: &Path,
    target: &Path,
    names: &[String],
    report: &mut File,
) -> Result<(), String> {
    if names.is_empty() || names.iter().any(|name| !name.starts_with("ferrum-")) {
        return Err("refusing cleanup without explicit Ferrum package selection".into());
    }
    let mut command = Command::new("cargo");
    command
        .args(["clean", "--profile", "dev", "--offline", "--manifest-path"])
        .arg(workspace.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target);
    for name in names {
        command.arg("--package").arg(name);
    }
    note(report, format!("cleanup_command={command:?}"))?;
    // Cargo clean ignores package URL/version qualifiers, so this deliberately
    // selects all dev versions of the current workspace's Ferrum package names.
    // Cargo owns package artifact selection. Never rm debug/deps, release,
    // incremental, other target triples, or nested trybuild targets ourselves.
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            note(report, format!("cleanup_exit={status}"))?;
            return if status.success() {
                Ok(())
            } else {
                Err(format!("package cleanup failed: {status}"))
            };
        }
        if started.elapsed() >= Duration::from_secs(600) {
            child.kill().map_err(|e| e.to_string())?;
            let status = child.wait().map_err(|e| e.to_string())?;
            return Err(format!(
                "package cleanup exceeded 600 seconds; stopped: {status}"
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn size_cleanup_needed(bytes: u64, limit: u64, previous: Option<(u64, u64)>) -> bool {
    bytes > limit
        && !previous.is_some_and(|(old_limit, floor)| {
            old_limit == limit && bytes.saturating_sub(floor) < (limit / 4).max(GIB)
        })
}

fn maintain(
    options: &Options,
    workspace: &Path,
    target: &Path,
    report: &mut File,
) -> Result<(), String> {
    // A retained residual larger than the size goal cannot be fixed by
    // repeatedly deleting the same Ferrum artifacts on every job. Remember
    // that condition under the lease; disk-reserve enforcement stays active.
    let marker = target.with_file_name(format!(
        ".{}.capacity-limit",
        target.file_name().unwrap().to_string_lossy()
    ));
    let previous_limit = match OpenOptions::new()
        .read(true)
        .custom_flags(NOFOLLOW)
        .open(&marker)
    {
        Ok(file) => {
            let mut value = String::new();
            file.take(64)
                .read_to_string(&mut value)
                .map_err(|e| e.to_string())?;
            let numbers = value
                .split_whitespace()
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "invalid cache capacity marker")?;
            match numbers.as_slice() {
                [limit, floor] => Some((*limit, *floor)),
                _ => return Err("invalid cache capacity marker".into()),
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.to_string()),
    };
    let before = usage(target)?;
    note(
        report,
        format!(
            "before={before:?} max_bytes={} min_free={}",
            options.max_bytes, options.min_free
        ),
    )?;
    if before.bytes <= options.max_bytes && previous_limit.is_some() {
        fs::remove_file(&marker).map_err(|e| e.to_string())?;
    }
    let size_trigger = size_cleanup_needed(before.bytes, options.max_bytes, previous_limit);
    if size_trigger || before.free < options.min_free {
        let packages = packages(workspace)?;
        let cleanup = match options.cleanup_mode {
            CleanupMode::DepInfo => clean_dep_info(target, &packages, report),
            CleanupMode::Cargo => clean(
                workspace,
                target,
                &packages
                    .iter()
                    .map(|package| package.name.clone())
                    .collect::<Vec<_>>(),
                report,
            ),
        };
        if let Err(error) = cleanup {
            // A timeout or cancellation can have removed some files. Preserve
            // that evidence without replacing the original cleanup failure.
            let _ = note(
                report,
                format!(
                    "cleanup_failed={error:?} usage_after_failure={:?}",
                    usage(target)
                ),
            );
            return Err(error);
        }
        let after = usage(target)?;
        note(report, format!("after={after:?}"))?;
        if after.bytes > options.max_bytes {
            let pending = marker.with_extension(format!("pending-{}", std::process::id()));
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .custom_flags(NOFOLLOW)
                .open(&pending)
                .map_err(|e| e.to_string())?;
            writeln!(file, "{} {}", options.max_bytes, after.bytes).map_err(|e| e.to_string())?;
            fs::rename(pending, &marker).map_err(|e| e.to_string())?;
            note(
                report,
                format!(
                    "retained_residual={} size_retrigger_growth={} disk_reserve=active",
                    after.bytes,
                    (options.max_bytes / 4).max(GIB)
                ),
            )?;
        } else if marker.exists() {
            fs::remove_file(&marker).map_err(|e| e.to_string())?;
        }
        if after.free < options.min_free {
            return Err(
                "free disk remains below reserve; protected artifacts were not removed".into(),
            );
        }
    } else if before.bytes > options.max_bytes {
        note(
            report,
            "cleanup=skipped_retained_residual_growth_below_margin disk_reserve=satisfied",
        )?;
    } else {
        note(report, "cleanup=not_needed")?;
    }
    Ok(())
}

fn run(options: Options) -> Result<i32, String> {
    if let Some(parent) = options.report.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut report = File::create(&options.report).map_err(|e| e.to_string())?;
    let result = (|| {
        let (workspace, target) = roots(&options.workspace, &options.target)?;
        note(
            &mut report,
            format!("workspace={workspace:?} target={target:?} phase=waiting_for_lease"),
        )?;
        let _lease = lease(&target)?;
        maintain(&options, &workspace, &target, &mut report)?;
        note(
            &mut report,
            format!("phase=running command={:?}", options.command),
        )?;
        let status = Command::new(&options.command[0])
            .args(&options.command[1..])
            .current_dir(workspace)
            .env("CARGO_TARGET_DIR", target)
            .status()
            .map_err(|e| e.to_string())?;
        let code = status
            .code()
            .unwrap_or_else(|| 128 + status.signal().unwrap_or(1));
        if let Err(error) = note(&mut report, format!("phase=completed command_exit={code}")) {
            // A full report filesystem must not replace the command's failure.
            if code == 0 {
                return Err(error);
            }
            eprintln!("Metal cache: could not record failed command: {error}");
        }
        Ok(code)
    })();
    if let Err(error) = &result {
        let _ = note(&mut report, format!("phase=failed error={error:?}"));
    }
    result
}

fn main() {
    let result = options().and_then(run);
    let code = match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("Metal cache: {error}");
            1
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
#[path = "metal_cache_tests.rs"]
mod tests;
