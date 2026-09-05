use super::HfDownloader;
use crate::vnext::source::ProductionModelSourceBundle;
use safetensors::tensor::{serialize, Dtype, TensorView};
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const MODEL_ID: &str = "fixture/standalone-template";
const REVISION: &str = "1234567890abcdef1234567890abcdef12345678";
const TEMPLATE: &str = "{% for message in messages %}{{ message.content }}\n{% endfor %}";

/// A tiny local Hub exercises the real tree, HEAD, GET, blob and snapshot path.
/// No process environment or shared Hugging Face cache is changed by these tests.
struct HubFixture {
    cache: TempDir,
    endpoint: String,
    fail_template_get: Arc<AtomicBool>,
    template_gets: Arc<AtomicUsize>,
    server: JoinHandle<()>,
}

impl HubFixture {
    async fn start(with_template: bool) -> Self {
        let weight = 1.0_f32.to_le_bytes();
        let tensor = TensorView::new(Dtype::F32, vec![1], &weight).unwrap();
        let weights = serialize([("weight", tensor)], &None).unwrap();
        let mut files = vec![
            ("config.json", b"{}".to_vec()),
            ("tokenizer.json", b"{}".to_vec()),
            ("tokenizer_config.json", b"{}".to_vec()),
            ("model.safetensors", weights),
        ];
        // Last in the tree so all other downloads finish before a template error
        // is returned by download(), which awaits its handles in tree order.
        if with_template {
            files.push(("chat_template.jinja", TEMPLATE.as_bytes().to_vec()));
        }
        let tree = serde_json::to_vec(
            &files
                .iter()
                .map(|(path, bytes)| json!({"path": path, "size": bytes.len(), "type": "file"}))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let revision = serde_json::to_vec(&json!({"sha": REVISION})).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let fail_template_get = Arc::new(AtomicBool::new(false));
        let template_gets = Arc::new(AtomicUsize::new(0));
        let fail = fail_template_get.clone();
        let gets = template_gets.clone();
        let server = tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                // Each response closes its connection. The files are tiny, so a
                // single accept loop suffices for concurrent downloader requests.
                tokio::time::timeout(Duration::from_secs(5), async {
                    let mut socket = BufReader::new(socket);
                    let mut request_line = String::new();
                    socket.read_line(&mut request_line).await.unwrap();
                    let mut fields = request_line.split_whitespace();
                    let method = fields.next().unwrap();
                    let path = fields.next().unwrap();
                    let mut header_bytes = 0;
                    loop {
                        let mut line = String::new();
                        let n = socket.read_line(&mut line).await.unwrap();
                        header_bytes += n;
                        assert!(header_bytes < 16_384, "unexpectedly large fixture request");
                        if n == 0 || line == "\r\n" {
                            break;
                        }
                    }
                    let resolve_prefix = format!("/{MODEL_ID}/resolve/main/");
                    let filename = path.strip_prefix(&resolve_prefix);
                    let (mut status, body, etag) = if path == format!("/api/models/{MODEL_ID}/tree/main") {
                        ("200 OK", tree.as_slice(), "tree")
                    } else if path == format!("/api/models/{MODEL_ID}/revision/main") {
                        ("200 OK", revision.as_slice(), "revision")
                    } else if let Some((name, bytes)) = files.iter().find(|(name, _)| Some(*name) == filename) {
                        ("200 OK", bytes.as_slice(), *name)
                    } else {
                        ("404 Not Found", b"missing".as_slice(), "missing")
                    };
                    if method == "GET" && filename == Some("chat_template.jinja") {
                        gets.fetch_add(1, Ordering::SeqCst);
                        if fail.load(Ordering::SeqCst) {
                            status = "503 Service Unavailable";
                        }
                    }
                    let headers = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nETag: \"{etag}\"\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    socket.get_mut().write_all(headers.as_bytes()).await.unwrap();
                    if method != "HEAD" {
                        socket.get_mut().write_all(body).await.unwrap();
                    }
                    socket.get_mut().shutdown().await.unwrap();
                })
                .await
                .expect("fixture request timed out");
            }
        });
        Self {
            cache: tempfile::tempdir().unwrap(),
            endpoint,
            fail_template_get,
            template_gets,
            server,
        }
    }

    fn downloader(&self) -> HfDownloader {
        HfDownloader {
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            cache_dir: self.cache.path().to_path_buf(),
            token: None,
            endpoint: self.endpoint.clone(),
        }
    }

    fn main_ref(&self) -> std::path::PathBuf {
        self.cache
            .path()
            .join("hub/models--fixture--standalone-template/refs/main")
    }
}

impl Drop for HubFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[tokio::test]
async fn fresh_download_preserves_standalone_chat_template_in_source_bundle() {
    let hub = HubFixture::start(true).await;
    let snapshot = hub.downloader().download(MODEL_ID, None).await.unwrap();
    let sources = ProductionModelSourceBundle::open_colocated_safetensors(&snapshot).unwrap();
    let config: serde_json::Value =
        serde_json::from_slice(sources.tokenizer_config_json().unwrap()).unwrap();
    assert!(config.get("chat_template").is_none());
    assert_eq!(sources.chat_template_jinja(), Some(TEMPLATE.as_bytes()));
    assert!(hub.template_gets.load(Ordering::SeqCst) > 0);
    let published_revision = std::fs::read_to_string(hub.main_ref()).unwrap();
    assert_eq!(
        snapshot.file_name().unwrap().to_str().unwrap(),
        published_revision
    );
}

#[tokio::test]
async fn repository_without_standalone_template_still_downloads() {
    let hub = HubFixture::start(false).await;
    let snapshot = hub.downloader().download(MODEL_ID, None).await.unwrap();
    let sources = ProductionModelSourceBundle::open_colocated_safetensors(snapshot).unwrap();
    assert!(sources.chat_template_jinja().is_none());
    assert_eq!(hub.template_gets.load(Ordering::SeqCst), 0);
    assert!(hub.main_ref().is_file());
}

#[tokio::test]
async fn failed_template_download_does_not_publish_ref_and_can_retry() {
    let hub = HubFixture::start(true).await;
    hub.fail_template_get.store(true, Ordering::SeqCst);
    let downloader = hub.downloader();
    let error = downloader.download(MODEL_ID, None).await.unwrap_err();
    assert!(error.to_string().contains("chat_template.jinja"), "{error}");
    assert!(!hub.main_ref().exists());

    hub.fail_template_get.store(false, Ordering::SeqCst);
    let snapshot = downloader.download(MODEL_ID, None).await.unwrap();
    let sources = ProductionModelSourceBundle::open_colocated_safetensors(snapshot).unwrap();
    assert_eq!(sources.chat_template_jinja(), Some(TEMPLATE.as_bytes()));
    assert!(hub.main_ref().is_file());
}
