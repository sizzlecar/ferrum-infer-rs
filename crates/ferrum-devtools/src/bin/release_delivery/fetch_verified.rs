//! Content-addressed downloads for pinned release inputs, independent of model caches.
use clap::Args;
use reqwest::{Client, StatusCode, Url};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tempfile::NamedTempFile;
use tokio::time::Instant;

#[derive(Args, Debug)]
pub struct FetchArgs {
    #[arg(long)]
    pub url: String,
    #[arg(long)]
    pub sha256: String,
    #[arg(long)]
    pub cache_dir: PathBuf,
    /// Replaced regular file; failures after setup leave this path absent.
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 15)]
    pub connect_timeout_secs: u64,
    #[arg(long, default_value_t = 30)]
    pub read_timeout_secs: u64,
    /// Total deadline across cache verification, downloads, retries and output copy.
    #[arg(long, default_value_t = 300)]
    pub timeout_secs: u64,
    /// Total download attempts, including the initial request.
    #[arg(long, default_value_t = 3)]
    pub max_attempts: u8,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema_version: u32,
    sha256: String,
    bytes: u64,
    cache_hit: bool,
    discarded_corrupt_cache: bool,
    attempts: u8,
    /// Entire operation through verified output publication, excluding stdout.
    elapsed_ms: u64,
    cache_file: PathBuf,
    output: PathBuf,
}

struct Paths {
    cache_file: PathBuf,
    cache_lock: PathBuf,
    output: PathBuf,
}

enum AttemptError {
    Retry(String),
    Reject(String),
}

pub async fn execute(args: FetchArgs) -> Result<(), String> {
    let receipt = fetch(args).await?;
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|error| error.to_string())?
    );
    Ok(())
}

async fn fetch(mut args: FetchArgs) -> Result<Receipt, String> {
    let started = Instant::now();
    let url = validate(&args)?;
    args.sha256.make_ascii_lowercase();
    let deadline = started
        .checked_add(Duration::from_secs(args.timeout_secs))
        .ok_or("fetch deadline overflow")?;
    let paths = prepare_paths(&args)?;
    let mut receipt = tokio::time::timeout_at(deadline, fetch_inner(&args, url, &paths, deadline))
        .await
        .map_err(|_| "verified fetch overall deadline exceeded".to_string())??;
    receipt.elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    Ok(receipt)
}

fn validate(args: &FetchArgs) -> Result<Url, String> {
    if args.sha256.len() != 64 || !args.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("expected SHA-256 must be exactly 64 hexadecimal digits".into());
    }
    if args.connect_timeout_secs == 0
        || args.read_timeout_secs == 0
        || args.timeout_secs == 0
        || !(1..=10).contains(&args.max_attempts)
    {
        return Err("fetch timeouts must be positive and max-attempts must be in 1..=10".into());
    }
    for seconds in [
        args.connect_timeout_secs,
        args.read_timeout_secs,
        args.timeout_secs,
    ] {
        if Instant::now()
            .checked_add(Duration::from_secs(seconds))
            .is_none()
        {
            return Err("fetch timeout exceeds the supported clock range".into());
        }
    }
    let url = Url::parse(&args.url).map_err(|_| "invalid download URL")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err("download URL must be HTTP(S), without credentials or a fragment".into());
    }
    Ok(url)
}

fn prepare_paths(args: &FetchArgs) -> Result<Paths, String> {
    fs::create_dir_all(&args.cache_dir).map_err(|error| error.to_string())?;
    let cache_root = args.cache_dir.canonicalize().map_err(|e| e.to_string())?;
    let cache_parent = cache_root.join("sha256");
    let lock_parent = cache_root.join("locks");
    for directory in [&cache_parent, &lock_parent] {
        fs::create_dir_all(directory).map_err(|error| error.to_string())?;
        if fs::symlink_metadata(directory)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("cache namespace must not be a symbolic link".into());
        }
    }
    let name = args.output.file_name().ok_or("output must name a file")?;
    let parent = args
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let parent = parent.canonicalize().map_err(|e| e.to_string())?;
    if parent.starts_with(&cache_root) {
        return Err("output must be outside the persistent cache directory".into());
    }
    let output = parent.join(name);
    if regular_file_exists(&output)? {
        // A stale output is not evidence of a successful download. Never publish
        // partial bytes here, and never remove an unrelated directory/symlink.
        fs::remove_file(&output).map_err(|error| error.to_string())?;
    }
    Ok(Paths {
        cache_file: cache_parent.join(&args.sha256),
        cache_lock: lock_parent.join(&args.sha256),
        output,
    })
}

fn regular_file_exists(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(format!(
            "expected a regular file, not a link/directory: {}",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.to_string()),
    }
}

async fn fetch_inner(
    args: &FetchArgs,
    url: Url,
    paths: &Paths,
    deadline: Instant,
) -> Result<Receipt, String> {
    // Serialize inspection/removal/publication for this digest. Otherwise a
    // stale corrupt observation could unlink another fetch's verified winner.
    let _cache_lock = acquire_cache_lock(paths, deadline).await?;
    let mut discarded_corrupt_cache = false;
    if regular_file_exists(&paths.cache_file)? {
        if let Some((temporary, bytes)) = verified_copy(paths, &args.sha256, deadline)? {
            publish_output(temporary, &paths.output, deadline)?;
            return Ok(receipt(args, paths, bytes, true, false, 0));
        }
        fs::remove_file(&paths.cache_file).map_err(|error| error.to_string())?;
        discarded_corrupt_cache = true;
        eprintln!("Discarded cache entry with incorrect SHA-256; fetching pinned bytes.");
    }

    // Construct the client only after the verified cache miss. Cache hits do not
    // perform DNS, proxy discovery, HEAD requests or any other network access.
    let require_https = url.scheme() == "https";
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout_secs))
        .read_timeout(Duration::from_secs(args.read_timeout_secs))
        .timeout(Duration::from_secs(args.timeout_secs))
        .redirect(reqwest::redirect::Policy::custom(
            move |attempt| match redirect_allowed(
                require_https,
                attempt.url(),
                attempt.previous().len(),
            ) {
                Ok(()) => attempt.follow(),
                Err(error) => attempt.error(error),
            },
        ))
        .build()
        .map_err(|error| error.without_url().to_string())?;
    for attempt in 1..=args.max_attempts {
        check_deadline(deadline)?;
        let result = download(&client, url.clone(), paths, &args.sha256, deadline).await;
        check_deadline(deadline)?;
        match result {
            Ok(temporary) => {
                // Both names are on the same filesystem. A concurrent fetch of
                // this content may also publish the same fully verified bytes.
                temporary
                    .persist(&paths.cache_file)
                    .map_err(|error| error.to_string())?;
                let (output, bytes) = verified_copy(paths, &args.sha256, deadline)?
                    .ok_or("cache bytes changed after verified download")?;
                publish_output(output, &paths.output, deadline)?;
                return Ok(receipt(
                    args,
                    paths,
                    bytes,
                    false,
                    discarded_corrupt_cache,
                    attempt,
                ));
            }
            Err(AttemptError::Reject(error)) => return Err(error),
            Err(AttemptError::Retry(error)) if attempt == args.max_attempts => {
                return Err(format!("{error}; exhausted {attempt} download attempts"));
            }
            Err(AttemptError::Retry(error)) => {
                eprintln!(
                    "{error}; retrying after attempt {attempt}/{}",
                    args.max_attempts
                );
                tokio::time::sleep(Duration::from_millis(250 * u64::from(attempt))).await;
            }
        }
    }
    unreachable!("nonzero bounded attempts return")
}

async fn acquire_cache_lock(paths: &Paths, deadline: Instant) -> Result<File, String> {
    regular_file_exists(&paths.cache_lock)?;
    let lock = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&paths.cache_lock)
        .map_err(|error| error.to_string())?;
    // Keep this zero-byte lock file across runs; unlinking it would allow two
    // different inodes to be locked for the same key. The OS releases the lock
    // when this handle closes, including process exit; no stale-PID guessing.
    loop {
        check_deadline(deadline)?;
        match lock.try_lock() {
            Ok(()) => return Ok(lock),
            Err(fs::TryLockError::WouldBlock) => {
                tokio::time::sleep(Duration::from_millis(25)).await
            }
            Err(fs::TryLockError::Error(error)) => return Err(error.to_string()),
        }
    }
}

fn receipt(
    args: &FetchArgs,
    paths: &Paths,
    bytes: u64,
    cache_hit: bool,
    discarded_corrupt_cache: bool,
    attempts: u8,
) -> Receipt {
    Receipt {
        schema_version: 1,
        sha256: args.sha256.clone(),
        bytes,
        cache_hit,
        discarded_corrupt_cache,
        attempts,
        elapsed_ms: 0,
        cache_file: paths.cache_file.clone(),
        output: paths.output.clone(),
    }
}

fn redirect_allowed(
    require_https: bool,
    target: &Url,
    previous: usize,
) -> Result<(), &'static str> {
    if require_https && target.scheme() != "https" {
        Err("HTTPS download cannot redirect to a non-HTTPS URL")
    } else if previous >= 5 {
        Err("download redirect limit exceeded")
    } else {
        Ok(())
    }
}

async fn download(
    client: &Client,
    url: Url,
    paths: &Paths,
    expected: &str,
    deadline: Instant,
) -> Result<NamedTempFile, AttemptError> {
    let mut response = client.get(url).send().await.map_err(network_error)?;
    let status = response.status();
    if status != StatusCode::OK {
        let message = format!("download returned HTTP {status}");
        return Err(
            if status.is_server_error() || matches!(status.as_u16(), 408 | 429) {
                AttemptError::Retry(message)
            } else {
                AttemptError::Reject(message)
            },
        );
    }
    let mut temporary = NamedTempFile::new_in(paths.cache_file.parent().unwrap())
        .map_err(|error| AttemptError::Reject(error.to_string()))?;
    let mut digest = Sha256::new();
    while let Some(chunk) = response.chunk().await.map_err(network_error)? {
        check_deadline(deadline).map_err(AttemptError::Reject)?;
        temporary
            .write_all(&chunk)
            .map_err(|error| AttemptError::Reject(error.to_string()))?;
        digest.update(&chunk);
    }
    if format!("{:x}", digest.finalize()) != expected {
        return Err(AttemptError::Reject(
            "download SHA-256 mismatch; no bytes published".into(),
        ));
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| AttemptError::Reject(error.to_string()))?;
    check_deadline(deadline).map_err(AttemptError::Reject)?;
    Ok(temporary)
}

fn network_error(error: reqwest::Error) -> AttemptError {
    // Redirects may carry signed queries; do not print the URL on errors.
    let permanent = error.is_builder() || error.is_redirect();
    let message = format!("download transport failed: {}", error.without_url());
    if permanent {
        AttemptError::Reject(message)
    } else {
        AttemptError::Retry(message)
    }
}

/// Hash the actual copied bytes, not a metadata stamp or a previously opened path.
fn verified_copy(
    paths: &Paths,
    expected: &str,
    deadline: Instant,
) -> Result<Option<(NamedTempFile, u64)>, String> {
    let mut source = File::open(&paths.cache_file).map_err(|error| error.to_string())?;
    let mut destination =
        NamedTempFile::new_in(paths.output.parent().unwrap()).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        check_deadline(deadline)?;
        let count = source
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        destination
            .write_all(&buffer[..count])
            .map_err(|error| error.to_string())?;
        digest.update(&buffer[..count]);
        bytes = bytes
            .checked_add(count as u64)
            .ok_or("download length overflow")?;
    }
    if format!("{:x}", digest.finalize()) != expected {
        return Ok(None);
    }
    destination
        .as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    Ok(Some((destination, bytes)))
}

fn publish_output(
    temporary: NamedTempFile,
    output: &Path,
    deadline: Instant,
) -> Result<(), String> {
    check_deadline(deadline)?;
    temporary
        .persist_noclobber(output)
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn check_deadline(deadline: Instant) -> Result<(), String> {
    if Instant::now() >= deadline {
        Err("verified fetch overall deadline exceeded".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "fetch_verified/tests.rs"]
mod tests;
