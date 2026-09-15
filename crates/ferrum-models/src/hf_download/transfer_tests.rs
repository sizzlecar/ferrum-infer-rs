//! Real HTTP framing and on-disk recovery, without weights or an external Hub.
use super::{DownloadRetryPolicy, HfDownloader};
use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const MODEL: &str = "fixture/recoverable";
const REVISION: &str = "1234567890abcdef1234567890abcdef12345678";
const FILE: &str = "model.safetensors";
const ETAG: &str = "immutable-payload";
const BYTES: &[u8] = b"immutable model bytes: abcdefghijklmnopqrstuvwxyz";

enum BodyFraming {
    ContentLength,
    Chunked { chunk_size: usize },
}

#[derive(Clone, Copy)]
enum BodyEnding {
    Complete,
    Interrupted,
    Pending,
}

struct Reply {
    status: &'static str,
    range: Option<String>,
    length: usize,
    body: Vec<u8>,
    delay: Duration,
    framing: BodyFraming,
    ending: BodyEnding,
}

impl Reply {
    fn full() -> Self {
        Self {
            status: "200 OK",
            range: None,
            length: BYTES.len(),
            body: BYTES.to_vec(),
            delay: Duration::ZERO,
            framing: BodyFraming::ContentLength,
            ending: BodyEnding::Complete,
        }
    }

    fn partial(offset: usize) -> Self {
        Self {
            status: "206 Partial Content",
            range: Some(format!(
                "bytes {offset}-{}/{}",
                BYTES.len() - 1,
                BYTES.len()
            )),
            length: BYTES.len() - offset,
            body: BYTES[offset..].to_vec(),
            ..Self::full()
        }
    }

    fn error(status: &'static str) -> Self {
        Self {
            status,
            length: 0,
            body: vec![],
            ..Self::full()
        }
    }

    fn chunked(mut self, chunk_size: usize, ending: BodyEnding) -> Self {
        assert!(chunk_size > 0);
        self.framing = BodyFraming::Chunked { chunk_size };
        self.ending = ending;
        self
    }
}

#[derive(Debug, Clone)]
struct Request {
    method: String,
    path: String,
    range: Option<String>,
}

struct TransferFixture {
    cache: TempDir,
    endpoint: String,
    requests: Arc<Mutex<Vec<Request>>>,
    server: JoinHandle<()>,
}

impl TransferFixture {
    async fn start(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let mut replies: VecDeque<_> = replies.into();
        let server = tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                let mut line = String::new();
                socket.read_line(&mut line).await.unwrap();
                let mut fields = line.split_whitespace();
                let method = fields.next().unwrap().to_owned();
                let path = fields.next().unwrap().to_owned();
                let mut range = None;
                let mut header_bytes = line.len();
                loop {
                    line.clear();
                    let count = socket.read_line(&mut line).await.unwrap();
                    header_bytes += count;
                    assert!(header_bytes < 16_384);
                    if count == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("range") {
                            range = Some(value.trim().to_owned());
                        }
                    }
                }
                recorded.lock().unwrap().push(Request {
                    method: method.clone(),
                    path,
                    range,
                });
                let reply = if method == "HEAD" {
                    Reply::full()
                } else {
                    replies
                        .pop_front()
                        .unwrap_or_else(|| Reply::error("503 Service Unavailable"))
                };
                tokio::time::sleep(reply.delay).await;
                let mut headers = format!(
                    "HTTP/1.1 {}\r\nETag: \"{ETAG}\"\r\nConnection: close\r\n",
                    reply.status
                );
                match &reply.framing {
                    BodyFraming::ContentLength => {
                        headers.push_str(&format!("Content-Length: {}\r\n", reply.length));
                    }
                    BodyFraming::Chunked { .. } => {
                        headers.push_str("Transfer-Encoding: chunked\r\n");
                    }
                }
                if let Some(range) = &reply.range {
                    headers.push_str(&format!("Content-Range: {range}\r\n"));
                }
                headers.push_str("\r\n");
                // A timeout may close the client before this intentionally
                // delayed reply. That is not a server-task panic.
                if socket.get_mut().write_all(headers.as_bytes()).await.is_ok() && method != "HEAD"
                {
                    match reply.framing {
                        BodyFraming::ContentLength => {
                            let _ = socket.get_mut().write_all(&reply.body).await;
                        }
                        BodyFraming::Chunked { chunk_size } => {
                            for chunk in reply.body.chunks(chunk_size) {
                                let framing = format!("{:x}\r\n", chunk.len());
                                let _ = socket.get_mut().write_all(framing.as_bytes()).await;
                                let _ = socket.get_mut().write_all(chunk).await;
                                let _ = socket.get_mut().write_all(b"\r\n").await;
                                let _ = socket.get_mut().flush().await;
                                tokio::task::yield_now().await;
                            }
                            if matches!(reply.ending, BodyEnding::Complete) {
                                let _ = socket.get_mut().write_all(b"0\r\n\r\n").await;
                            }
                        }
                    }
                    let _ = socket.get_mut().flush().await;
                    if matches!(reply.ending, BodyEnding::Pending) {
                        // Keep the HTTP body open until the fixture is dropped;
                        // only the downloader's whole-file budget can end it.
                        std::future::pending::<()>().await;
                    }
                    if reply.body.len() < reply.length {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
                let _ = socket.get_mut().shutdown().await;
            }
        });
        let cache = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(cache.path().join("blobs")).unwrap();
        std::fs::create_dir_all(cache.path().join("snapshot")).unwrap();
        Self {
            cache,
            endpoint,
            requests,
            server,
        }
    }

    fn incomplete(&self) -> PathBuf {
        self.cache
            .path()
            .join("blobs")
            .join(format!("{ETAG}.incomplete"))
    }

    fn gets(&self) -> Vec<Request> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.method == "GET")
            .cloned()
            .collect()
    }

    async fn download(&self, policy: DownloadRetryPolicy) -> ferrum_types::Result<()> {
        let downloader = HfDownloader {
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            cache_dir: self.cache.path().to_path_buf(),
            token: None,
            endpoint: self.endpoint.clone(),
            retry_policy: DownloadRetryPolicy::default(),
        }
        .with_retry_policy(policy)?;
        downloader
            .download_file_concurrent(
                MODEL,
                REVISION,
                FILE,
                BYTES.len() as u64,
                &self.cache.path().join("blobs"),
                &self.cache.path().join("snapshot"),
                None,
            )
            .await
    }

    fn assert_complete(&self) {
        assert_eq!(
            std::fs::read(self.cache.path().join("snapshot").join(FILE)).unwrap(),
            BYTES
        );
        assert_eq!(
            std::fs::read(self.cache.path().join("blobs").join(ETAG)).unwrap(),
            BYTES
        );
        assert!(!self.incomplete().exists());
        let expected_path = format!("/{MODEL}/resolve/{REVISION}/{FILE}");
        assert!(self
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|request| request.path == expected_path));
    }

    fn assert_unpublished(&self) {
        assert!(!self.cache.path().join("snapshot").join(FILE).exists());
        assert!(!self.cache.path().join("blobs").join(ETAG).exists());
    }
}

impl Drop for TransferFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn policy(attempts: u32) -> DownloadRetryPolicy {
    DownloadRetryPolicy {
        max_attempts: NonZeroU32::new(attempts).unwrap(),
        initial_backoff: Duration::ZERO,
        max_elapsed: Duration::from_secs(3),
    }
}

#[tokio::test]
async fn interrupted_body_resumes_the_saved_prefix_within_one_download() {
    let prefix = 13;
    let mut first = Reply::full();
    first.body.truncate(prefix);
    let fixture = TransferFixture::start(vec![first, Reply::partial(prefix)]).await;
    fixture.download(policy(2)).await.unwrap();
    fixture.assert_complete();
    let gets = fixture.gets();
    assert_eq!(gets[0].range, None);
    assert_eq!(gets[1].range.as_deref(), Some("bytes=13-"));
}

#[tokio::test]
async fn ignored_range_replaces_partial_bytes_instead_of_appending_a_full_body() {
    let fixture = TransferFixture::start(vec![Reply::full()]).await;
    std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
    fixture.download(policy(1)).await.unwrap();
    fixture.assert_complete();
    assert_eq!(fixture.gets()[0].range.as_deref(), Some("bytes=9-"));
}

#[tokio::test]
async fn invalid_partial_response_never_recombines_or_publishes_bytes() {
    for range in [
        None,
        Some(format!("bytes 8-{}/{}", BYTES.len() - 1, BYTES.len())),
        Some(format!("bytes 9-8/{}", BYTES.len())),
        Some(format!("bytes 9-{}/{}", BYTES.len() - 1, BYTES.len() + 1)),
        Some("unknown-unit 9-47/48".into()),
    ] {
        let mut reply = Reply::partial(9);
        reply.range = range;
        let fixture = TransferFixture::start(vec![reply]).await;
        std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
        assert!(fixture.download(policy(3)).await.is_err());
        fixture.assert_unpublished();
        assert_eq!(std::fs::read(fixture.incomplete()).unwrap(), &BYTES[..9]);
        assert_eq!(fixture.gets().len(), 1, "invalid framing is not transient");
    }
}

#[tokio::test]
async fn permanent_http_error_preserves_partial_without_retry_or_publication() {
    for status in ["401 Unauthorized", "404 Not Found"] {
        let fixture = TransferFixture::start(vec![Reply::error(status)]).await;
        std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
        assert!(fixture.download(policy(3)).await.is_err());
        fixture.assert_unpublished();
        assert_eq!(std::fs::read(fixture.incomplete()).unwrap(), &BYTES[..9]);
        assert_eq!(fixture.gets().len(), 1);
    }
}

#[tokio::test]
async fn transient_http_statuses_retry_the_same_saved_range_and_complete() {
    for status in [
        "408 Request Timeout",
        "429 Too Many Requests",
        "500 Internal Server Error",
        "503 Service Unavailable",
    ] {
        let fixture = TransferFixture::start(vec![Reply::error(status), Reply::partial(9)]).await;
        std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
        fixture.download(policy(2)).await.unwrap();
        fixture.assert_complete();
        let gets = fixture.gets();
        assert_eq!(
            gets.len(),
            2,
            "{status} must recover within the attempt budget"
        );
        assert!(gets
            .iter()
            .all(|request| request.range.as_deref() == Some("bytes=9-")));
    }
}

#[tokio::test]
async fn oversized_chunked_partial_body_rolls_back_to_the_original_prefix() {
    let mut reply = Reply::partial(9);
    reply.body.push(b'!');
    // Valid bytes and the excess byte use separate HTTP chunks. Even if valid
    // chunks have already reached disk, the invalid attempt must be rolled back.
    let reply = reply.chunked(BYTES.len() - 9, BodyEnding::Complete);
    let fixture = TransferFixture::start(vec![reply]).await;
    std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
    assert!(fixture.download(policy(3)).await.is_err());
    fixture.assert_unpublished();
    assert_eq!(std::fs::read(fixture.incomplete()).unwrap(), &BYTES[..9]);
    assert_eq!(
        fixture.gets().len(),
        1,
        "an oversized body is not transient"
    );
}

#[tokio::test]
async fn short_chunked_partial_body_resumes_after_newly_saved_bytes() {
    for ending in [BodyEnding::Complete, BodyEnding::Interrupted] {
        let mut reply = Reply::partial(9).chunked(4, ending);
        reply.body.truncate(7);
        let fixture = TransferFixture::start(vec![reply, Reply::partial(16)]).await;
        std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
        fixture.download(policy(2)).await.unwrap();
        fixture.assert_complete();
        let gets = fixture.gets();
        assert_eq!(gets.len(), 2);
        assert_eq!(gets[0].range.as_deref(), Some("bytes=9-"));
        assert_eq!(gets[1].range.as_deref(), Some("bytes=16-"));
    }
}

#[tokio::test]
async fn pending_body_exhausts_total_budget_and_flushes_new_partial_bytes() {
    let mut reply = Reply::partial(9).chunked(4, BodyEnding::Pending);
    reply.body.truncate(7);
    let fixture = TransferFixture::start(vec![reply]).await;
    std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
    let mut bounded = policy(3);
    bounded.max_elapsed = Duration::from_millis(500);
    assert!(
        tokio::time::timeout(Duration::from_secs(2), fixture.download(bounded))
            .await
            .expect("the whole-file deadline must also bound a pending body")
            .is_err()
    );
    fixture.assert_unpublished();
    assert_eq!(std::fs::read(fixture.incomplete()).unwrap(), &BYTES[..16]);
    assert_eq!(
        fixture.gets().len(),
        1,
        "retries cannot extend the total budget"
    );
}

#[tokio::test]
async fn zero_elapsed_budget_is_rejected_before_any_request_or_cache_write() {
    let fixture = TransferFixture::start(vec![Reply::full()]).await;
    let mut invalid = policy(3);
    invalid.max_elapsed = Duration::ZERO;
    assert!(matches!(
        fixture.download(invalid).await,
        Err(ferrum_types::FerrumError::Config { .. })
    ));
    assert!(fixture.requests.lock().unwrap().is_empty());
    assert!(!fixture.incomplete().exists());
    fixture.assert_unpublished();
}

#[tokio::test]
async fn backoff_exhausting_total_budget_never_starts_another_request() {
    let fixture = TransferFixture::start(vec![
        Reply::error("503 Service Unavailable"),
        Reply::partial(9),
    ])
    .await;
    std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
    let mut bounded = policy(3);
    bounded.initial_backoff = Duration::from_secs(10);
    bounded.max_elapsed = Duration::from_millis(500);
    assert!(
        tokio::time::timeout(Duration::from_secs(2), fixture.download(bounded))
            .await
            .expect("backoff must not extend the whole-file budget")
            .is_err()
    );
    fixture.assert_unpublished();
    assert_eq!(std::fs::read(fixture.incomplete()).unwrap(), &BYTES[..9]);
    assert_eq!(
        fixture.gets().len(),
        1,
        "no attempt remains after the deadline"
    );
}

#[tokio::test]
async fn retry_and_elapsed_budgets_leave_incomplete_bytes_unpublished() {
    let mut first = Reply::full();
    first.body.truncate(9);
    let fixture = TransferFixture::start(vec![first]).await;
    assert!(fixture.download(policy(2)).await.is_err());
    fixture.assert_unpublished();
    assert_eq!(std::fs::read(fixture.incomplete()).unwrap(), &BYTES[..9]);
    assert_eq!(fixture.gets().len(), 2, "finite attempt budget is enforced");

    let mut delayed = Reply::partial(9);
    delayed.delay = Duration::from_secs(2);
    let fixture = TransferFixture::start(vec![delayed]).await;
    std::fs::write(fixture.incomplete(), &BYTES[..9]).unwrap();
    let mut bounded = policy(3);
    bounded.max_elapsed = Duration::from_millis(100);
    assert!(
        tokio::time::timeout(Duration::from_secs(1), fixture.download(bounded))
            .await
            .expect("whole-file deadline must bound attempts and backoff")
            .is_err()
    );
    fixture.assert_unpublished();
    assert_eq!(std::fs::read(fixture.incomplete()).unwrap(), &BYTES[..9]);
}
