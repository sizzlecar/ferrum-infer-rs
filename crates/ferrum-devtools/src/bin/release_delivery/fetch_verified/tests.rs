use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};

enum Reply {
    Complete(Vec<u8>),
    Truncated(Vec<u8>),
    Status(u16),
    Redirect,
    StallBody,
}

struct Server {
    url: String,
    requests: Arc<AtomicUsize>,
    worker: JoinHandle<()>,
}

impl Server {
    async fn new(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/asset", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let count = requests.clone();
        let worker = tokio::spawn(async move {
            for reply in replies {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut chunk = [0; 1024];
                while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                    let n = stream.read(&mut chunk).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&chunk[..n]);
                }
                assert!(request.starts_with(b"GET /asset HTTP/1.1\r\n"));
                count.fetch_add(1, Ordering::SeqCst);
                let (status, body, extra, stall, headers) = match reply {
                    Reply::Complete(body) => (200, body, 0, false, ""),
                    Reply::Truncated(body) => (200, body, 17, false, ""),
                    Reply::Status(status) => (status, Vec::new(), 0, false, ""),
                    Reply::Redirect => (302, Vec::new(), 0, false, "Location: /asset\r\n"),
                    Reply::StallBody => (200, b"partial".to_vec(), 17, true, ""),
                };
                let header = format!(
                    "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n",
                    body.len() + extra
                );
                stream.write_all(header.as_bytes()).await.unwrap();
                stream.write_all(&body).await.unwrap();
                if stall {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
        });
        Self {
            url,
            requests,
            worker,
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

fn args(root: &Path, url: &str, bytes: &[u8]) -> FetchArgs {
    FetchArgs {
        url: url.into(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        cache_dir: root.join("cache"),
        output: root.join("output/asset.zip"),
        connect_timeout_secs: 2,
        read_timeout_secs: 2,
        timeout_secs: 5,
        max_attempts: 3,
    }
}

fn assert_no_unverified_files(root: &Path) {
    for directory in [root.join("cache/sha256"), root.join("output")] {
        assert_eq!(fs::read_dir(directory).unwrap().count(), 0);
    }
}

#[tokio::test]
async fn interrupted_body_is_retried_and_only_complete_verified_bytes_are_published() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"pinned release input\0with binary bytes\xff";
    let server = Server::new(vec![
        Reply::Truncated(bytes[..7].to_vec()),
        Reply::Complete(bytes.to_vec()),
    ])
    .await;
    let receipt = fetch(args(root.path(), &server.url, bytes)).await.unwrap();
    assert_eq!(receipt.attempts, 2);
    assert!(
        receipt.elapsed_ms >= 250,
        "elapsed interval includes retry delay"
    );
    assert!(!receipt.cache_hit);
    assert_eq!(receipt.bytes, bytes.len() as u64);
    assert_eq!(fs::read(&receipt.cache_file).unwrap(), bytes);
    assert_eq!(fs::read(&receipt.output).unwrap(), bytes);
    assert_eq!(server.requests.load(Ordering::SeqCst), 2);
    assert_eq!(
        fs::read_dir(root.path().join("cache/sha256"))
            .unwrap()
            .count(),
        1
    );
    assert_eq!(fs::read_dir(root.path().join("output")).unwrap().count(), 1);
}

#[tokio::test]
async fn verified_cache_hit_is_offline_and_output_does_not_alias_cache() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"immutable source archive";
    let server = Server::new(vec![Reply::Complete(bytes.to_vec())]).await;
    let first = fetch(args(root.path(), &server.url, bytes)).await.unwrap();
    let url = server.url.clone();
    drop(server);
    fs::write(&first.output, b"stale destination").unwrap();
    let mut options = args(root.path(), &url, bytes);
    options.sha256.make_ascii_uppercase();
    let hit = fetch(options).await.unwrap();
    assert!(hit.cache_hit);
    assert_eq!(hit.attempts, 0);
    assert_eq!(fs::read(&hit.output).unwrap(), bytes);
    fs::write(&hit.output, b"caller changes its own copy").unwrap();
    assert_eq!(fs::read(&hit.cache_file).unwrap(), bytes);
}

#[tokio::test]
async fn same_size_corrupt_cache_with_unchanged_mtime_is_rehashed_and_replaced() {
    let root = tempfile::tempdir().unwrap();
    let bytes = vec![b'x'; 128 * 1024 + 17];
    let server = Server::new(vec![Reply::Complete(bytes.to_vec())]).await;
    let options = args(root.path(), &server.url, &bytes);
    let cached = options.cache_dir.join("sha256").join(&options.sha256);
    fs::create_dir_all(cached.parent().unwrap()).unwrap();
    fs::write(&cached, &bytes).unwrap();
    let modified = fs::metadata(&cached).unwrap().modified().unwrap();
    let mut corrupted = bytes.clone();
    *corrupted.last_mut().unwrap() ^= 1;
    fs::write(&cached, corrupted).unwrap();
    File::options()
        .write(true)
        .open(&cached)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    let receipt = fetch(options).await.unwrap();
    assert!(!receipt.cache_hit);
    assert!(receipt.discarded_corrupt_cache);
    assert_eq!(receipt.attempts, 1);
    assert_eq!(server.requests.load(Ordering::SeqCst), 1);
    assert_eq!(fs::read(cached).unwrap(), bytes);
    assert_eq!(fs::read(receipt.output).unwrap(), bytes);
}

#[tokio::test]
async fn hash_mismatch_leaves_no_cache_partial_or_stale_destination() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::new(vec![Reply::Complete(b"wrong SHA".to_vec())]).await;
    let options = args(root.path(), &server.url, b"expected SHA");
    fs::create_dir_all(options.output.parent().unwrap()).unwrap();
    // Even a previously correct destination cannot rescue this failed fetch.
    fs::write(&options.output, b"expected SHA").unwrap();
    let error = fetch(options).await.unwrap_err();
    assert!(error.contains("SHA-256 mismatch"), "{error}");
    assert_eq!(server.requests.load(Ordering::SeqCst), 1);
    assert_no_unverified_files(root.path());
}

#[tokio::test]
async fn retryable_http_status_recovers_but_attempt_limit_is_enforced() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"reference license";
    let server = Server::new(vec![Reply::Status(503), Reply::Complete(bytes.to_vec())]).await;
    assert_eq!(
        fetch(args(root.path(), &server.url, bytes))
            .await
            .unwrap()
            .attempts,
        2
    );

    let failed = tempfile::tempdir().unwrap();
    let server = Server::new(vec![Reply::Status(503), Reply::Status(503)]).await;
    let mut options = args(failed.path(), &server.url, bytes);
    options.max_attempts = 2;
    let error = fetch(options).await.unwrap_err();
    assert!(error.contains("exhausted 2 download attempts"), "{error}");
    assert_eq!(server.requests.load(Ordering::SeqCst), 2);
    assert_no_unverified_files(failed.path());
}

#[tokio::test]
async fn permanent_http_error_does_not_retry_or_reuse_corrupt_cache() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::new(vec![Reply::Status(404)]).await;
    let options = args(root.path(), &server.url, b"expected bytes");
    let cache = options.cache_dir.join("sha256").join(&options.sha256);
    fs::create_dir_all(cache.parent().unwrap()).unwrap();
    fs::write(cache, b"bad cached bytes").unwrap();
    let error = fetch(options).await.unwrap_err();
    assert!(error.contains("HTTP 404"), "{error}");
    assert_eq!(server.requests.load(Ordering::SeqCst), 1);
    assert_no_unverified_files(root.path());
}

#[tokio::test]
async fn stalled_body_obeys_read_and_overall_deadlines_and_cleans_partials() {
    for overall_first in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::new(vec![Reply::StallBody]).await;
        let mut options = args(root.path(), &server.url, b"expected bytes");
        options.max_attempts = 1;
        options.read_timeout_secs = if overall_first { 5 } else { 1 };
        options.timeout_secs = if overall_first { 1 } else { 5 };
        let error = tokio::time::timeout(Duration::from_secs(4), fetch(options))
            .await
            .expect("configured deadline must bound stalled body")
            .unwrap_err();
        assert!(
            if overall_first {
                error.contains("overall deadline")
            } else {
                error.contains("transport failed")
            },
            "{error}"
        );
        assert_no_unverified_files(root.path());
    }
}

#[tokio::test]
async fn unsafe_paths_and_invalid_policy_are_rejected_without_clobbering_cache() {
    let root = tempfile::tempdir().unwrap();
    let mut options = args(root.path(), "http://127.0.0.1:1/asset", b"bytes");
    options.output = options.cache_dir.join("sha256").join(&options.sha256);
    fs::create_dir_all(options.output.parent().unwrap()).unwrap();
    fs::write(&options.output, b"bytes").unwrap();
    let cached = options.output.clone();
    assert!(fetch(options).await.unwrap_err().contains("outside"));
    assert_eq!(fs::read(cached).unwrap(), b"bytes");
    let mut options = args(root.path(), "http://127.0.0.1:1/asset", b"bytes");
    options.output = root.path().join("directory");
    fs::create_dir(&options.output).unwrap();
    assert!(fetch(options).await.unwrap_err().contains("regular file"));
    for mutation in [0, 1, 2] {
        let mut options = args(root.path(), "http://127.0.0.1:1/asset", b"bytes");
        match mutation {
            0 => options.sha256 = "../invalid".into(),
            1 => options.timeout_secs = 0,
            _ => options.max_attempts = 0,
        }
        assert!(fetch(options).await.is_err());
    }
}

#[cfg(unix)]
#[tokio::test]
async fn output_symlink_is_rejected_without_touching_its_target() {
    let root = tempfile::tempdir().unwrap();
    let options = args(root.path(), "http://127.0.0.1:1/asset", b"bytes");
    let target = root.path().join("unrelated");
    fs::write(&target, b"keep").unwrap();
    fs::create_dir_all(options.output.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(&target, &options.output).unwrap();
    assert!(fetch(options).await.unwrap_err().contains("regular file"));
    assert_eq!(fs::read(target).unwrap(), b"keep");
}

#[test]
fn command_line_exposes_bounded_fetch_defaults() {
    use clap::Parser;
    let parsed = crate::Args::try_parse_from([
        "release_delivery",
        "fetch-verified",
        "--url",
        "https://example.org/input",
        "--sha256",
        &"a".repeat(64),
        "--cache-dir",
        "cache",
        "--output",
        "input.zip",
    ])
    .unwrap();
    let crate::Action::FetchVerified(options) = parsed.action else {
        panic!("fetch subcommand");
    };
    assert_eq!(options.connect_timeout_secs, 15);
    assert_eq!(options.read_timeout_secs, 30);
    assert_eq!(options.timeout_secs, 300);
    assert_eq!(options.max_attempts, 3);
}

#[test]
fn https_redirect_policy_preserves_transport_and_bounds_redirects() {
    let https = Url::parse("https://cdn.example.org/asset").unwrap();
    let http = Url::parse("http://cdn.example.org/asset").unwrap();
    assert!(redirect_allowed(true, &https, 1).is_ok());
    assert!(redirect_allowed(true, &http, 1)
        .unwrap_err()
        .contains("non-HTTPS"));
    assert!(redirect_allowed(false, &http, 1).is_ok());
    assert!(redirect_allowed(true, &https, 5).is_err());
}

#[tokio::test]
async fn explicit_http_redirects_fetch_final_bytes_but_loops_do_not_retry() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"redirected pinned input";
    let server = Server::new(vec![Reply::Redirect, Reply::Complete(bytes.to_vec())]).await;
    let receipt = fetch(args(root.path(), &server.url, bytes)).await.unwrap();
    assert_eq!(receipt.attempts, 1);
    assert_eq!(fs::read(receipt.output).unwrap(), bytes);
    assert_eq!(server.requests.load(Ordering::SeqCst), 2);
    let root = tempfile::tempdir().unwrap();
    let server = Server::new((0..5).map(|_| Reply::Redirect).collect()).await;
    assert!(fetch(args(root.path(), &server.url, bytes))
        .await
        .unwrap_err()
        .contains("transport failed"));
    assert_eq!(server.requests.load(Ordering::SeqCst), 5);
    assert_no_unverified_files(root.path());
}

#[tokio::test]
async fn concurrent_fetches_repair_corrupt_cache_without_deleting_a_verified_winner() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"shared immutable input";
    let server = Server::new(vec![Reply::Complete(bytes.to_vec())]).await;
    let first = args(root.path(), &server.url, bytes);
    let mut second = args(root.path(), &server.url, bytes);
    second.output = root.path().join("other-output.zip");
    let cached = first.cache_dir.join("sha256").join(&first.sha256);
    fs::create_dir_all(cached.parent().unwrap()).unwrap();
    fs::write(&cached, b"corrupt old object").unwrap();
    let (first, second) = tokio::join!(fetch(first), fetch(second));
    let receipts = [first.unwrap(), second.unwrap()];
    assert_eq!(receipts.iter().filter(|r| r.cache_hit).count(), 1);
    assert_eq!(
        receipts
            .iter()
            .filter(|r| r.discarded_corrupt_cache)
            .count(),
        1
    );
    assert_eq!(server.requests.load(Ordering::SeqCst), 1);
    for receipt in receipts {
        assert_eq!(fs::read(receipt.output).unwrap(), bytes);
    }
    assert_eq!(fs::read(cached).unwrap(), bytes);
}

#[tokio::test]
async fn cache_lock_wait_uses_overall_deadline_then_reuses_released_cache_offline() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"valid locked object";
    let server = Server::new(vec![Reply::Complete(bytes.to_vec())]).await;
    let mut options = args(root.path(), &server.url, bytes);
    options.timeout_secs = 1;
    let paths = prepare_paths(&options).unwrap();
    fs::write(&paths.cache_file, bytes).unwrap();
    let held = acquire_cache_lock(&paths, Instant::now() + Duration::from_secs(3))
        .await
        .unwrap();
    let error = fetch(options).await.unwrap_err();
    assert!(error.contains("overall deadline"), "{error}");
    assert!(!paths.output.exists());
    assert_eq!(fs::read(&paths.cache_file).unwrap(), bytes);
    assert_eq!(server.requests.load(Ordering::SeqCst), 0);
    drop(held);
    let hit = fetch(args(root.path(), &server.url, bytes)).await.unwrap();
    assert!(hit.cache_hit);
    assert_eq!(server.requests.load(Ordering::SeqCst), 0);
    assert_eq!(fs::read(hit.output).unwrap(), bytes);
}
