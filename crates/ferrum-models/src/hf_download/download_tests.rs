use super::HfDownloader;
use crate::vnext::source::ProductionModelSourceBundle;
use ferrum_quantization::SafetensorsArchive;
use safetensors::tensor::{serialize, Dtype, TensorView};
use serde_json::json;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const MODEL_ID: &str = "fixture/standalone-template";
const REVISION: &str = "1234567890abcdef1234567890abcdef12345678";
const TEMPLATE: &str = "{% for message in messages %}{{ message.content }}\n{% endfor %}";
const INDEX: &str = "model.safetensors.index.json";
const SHARD_A: &str = "model-00001-of-00002.safetensors";
const SHARD_B: &str = "model-00002-of-00002.safetensors";
const EXTRA_WEIGHTS: &[&str] = &[
    "unused.safetensors",
    "original/unused.safetensors",
    "pytorch_model.bin",
    "alternative/pytorch_model.bin",
    "weights.pt",
    "alternative/weights.pt",
    "weights.onnx",
    "alternative/weights.onnx",
];

#[derive(Clone)]
struct HubFile {
    path: &'static str,
    bytes: Vec<u8>,
}

impl HubFile {
    fn new(path: &'static str, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            path,
            bytes: bytes.into(),
        }
    }
}

#[derive(Clone, Debug)]
struct HubRequest {
    method: String,
    path: String,
}

fn tensor_file(tensors: &[(&str, f32)]) -> Vec<u8> {
    let bytes: Vec<_> = tensors
        .iter()
        .map(|(_, value)| value.to_le_bytes())
        .collect();
    let views = tensors
        .iter()
        .zip(&bytes)
        .map(|((name, _), bytes)| (*name, TensorView::new(Dtype::F32, vec![1], bytes).unwrap()));
    serialize(views, &None).unwrap()
}

fn single_files(with_template: bool) -> Vec<HubFile> {
    let mut files = vec![
        HubFile::new("config.json", b"{}"),
        HubFile::new("tokenizer.json", b"{}"),
        HubFile::new("tokenizer_config.json", b"{}"),
        HubFile::new("model.safetensors", tensor_file(&[("weight", 1.0)])),
    ];
    // Keep the original template-error case last in transfer order.
    if with_template {
        files.push(HubFile::new("chat_template.jinja", TEMPLATE.as_bytes()));
    }
    files
}

fn indexed_files() -> Vec<HubFile> {
    let mut files = vec![
        HubFile::new("config.json", b"{}"),
        HubFile::new("tokenizer.json", b"{}"),
        HubFile::new("tokenizer_config.json", b"{}"),
        HubFile::new("generation_config.json", b"{}"),
        HubFile::new("chat_template.jinja", TEMPLATE.as_bytes()),
        HubFile::new("tokenizer.model", b"tokenizer sidecar"),
        HubFile::new("vocab.txt", b"a\nb\n"),
        HubFile::new("preprocessor_config.json", b"{}"),
        HubFile::new(
            INDEX,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 12},
                "weight_map": {
                    "weight.first": SHARD_A,
                    "weight.second": SHARD_A,
                    "weight.third": SHARD_B
                }
            }))
            .unwrap(),
        ),
    ];
    for path in EXTRA_WEIGHTS {
        files.push(HubFile::new(path, b"unselected alternative weights"));
    }
    files.push(HubFile::new(
        SHARD_A,
        tensor_file(&[("weight.first", 1.0), ("weight.second", 2.0)]),
    ));
    // Last so the shard-error regression awaits earlier concurrent transfers
    // before retrying the failed shard through the same cache.
    files.push(HubFile::new(SHARD_B, tensor_file(&[("weight.third", 3.0)])));
    files
}

fn tree_response(files: &[HubFile], directory: &str) -> Vec<u8> {
    let prefix = if directory.is_empty() {
        String::new()
    } else {
        format!("{directory}/")
    };
    let mut directories = BTreeSet::new();
    let mut entries = Vec::new();
    for file in files {
        let Some(relative) = file.path.strip_prefix(&prefix) else {
            continue;
        };
        if let Some((child, _)) = relative.split_once('/') {
            let path = format!("{prefix}{child}");
            if directories.insert(path.clone()) {
                entries.push(json!({"path": path, "type": "directory"}));
            }
        } else {
            entries.push(json!({"path": file.path, "size": file.bytes.len(), "type": "file"}));
        }
    }
    serde_json::to_vec(&entries).unwrap()
}

/// A tiny local Hub exercises real revision/tree/HEAD/GET/blob/snapshot paths.
/// No process environment or shared Hugging Face cache is changed.
struct HubFixture {
    cache: TempDir,
    endpoint: String,
    fail_template_get: Arc<AtomicBool>,
    template_gets: Arc<AtomicUsize>,
    failed_gets: Arc<Mutex<BTreeSet<String>>>,
    requests: Arc<Mutex<Vec<HubRequest>>>,
    server: JoinHandle<()>,
}

impl HubFixture {
    async fn start(with_template: bool) -> Self {
        Self::with_files(single_files(with_template), None).await
    }

    /// `main_files` simulates a moving branch after metadata resolved REVISION.
    /// None keeps legacy main routes equivalent to the immutable snapshot, so
    /// unrelated GGUF/sidecar download paths can still use this fixture.
    async fn with_files(files: Vec<HubFile>, main_files: Option<Vec<HubFile>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let fail_template_get = Arc::new(AtomicBool::new(false));
        let template_gets = Arc::new(AtomicUsize::new(0));
        let failed_gets = Arc::new(Mutex::new(BTreeSet::<String>::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let fail = fail_template_get.clone();
        let gets = template_gets.clone();
        let failures = failed_gets.clone();
        let recorded = requests.clone();
        let server = tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                tokio::time::timeout(Duration::from_secs(5), async {
                    let mut socket = BufReader::new(socket);
                    let mut request_line = String::new();
                    socket.read_line(&mut request_line).await.unwrap();
                    let mut fields = request_line.split_whitespace();
                    let method = fields.next().unwrap();
                    let path = fields.next().unwrap();
                    recorded.lock().unwrap().push(HubRequest {
                        method: method.to_owned(), path: path.to_owned(),
                    });
                    let mut header_bytes = 0;
                    loop {
                        let mut line = String::new();
                        let n = socket.read_line(&mut line).await.unwrap();
                        header_bytes += n;
                        assert!(header_bytes < 16_384, "unexpectedly large fixture request");
                        if n == 0 || line == "\r\n" { break; }
                    }
                    let path = path.split('?').next().unwrap();
                    let tree = path.strip_prefix(&format!("/api/models/{MODEL_ID}/tree/"));
                    let metadata = path.strip_prefix(&format!("/api/models/{MODEL_ID}/revision/"));
                    let resolve = path.strip_prefix(&format!("/{MODEL_ID}/resolve/"))
                        .and_then(|path| path.split_once('/'));
                    let inventory = |revision| match revision {
                        REVISION => Some(files.as_slice()),
                        "main" => Some(main_files.as_deref().unwrap_or(&files)),
                        _ => None,
                    };
                    let (mut status, body, etag) = if matches!(metadata, Some("main") | Some(REVISION)) {
                        ("200 OK", serde_json::to_vec(&json!({"sha": REVISION})).unwrap(), "revision".to_owned())
                    } else if let Some(tree) = tree {
                        let (revision, directory) = tree.split_once('/').unwrap_or((tree, ""));
                        if let Some(files) = inventory(revision) {
                            ("200 OK", tree_response(files, directory), "tree".to_owned())
                        } else {
                            ("404 Not Found", b"missing tree".to_vec(), "missing".to_owned())
                        }
                    } else if let Some((revision, filename)) = resolve {
                        if let Some(file) = inventory(revision).and_then(|files| files.iter().find(|f| f.path == filename)) {
                            ("200 OK", file.bytes.clone(), format!("{revision}-{}", filename.replace('/', "_")))
                        } else {
                            ("404 Not Found", b"missing file".to_vec(), "missing".to_owned())
                        }
                    } else {
                        ("404 Not Found", b"missing".to_vec(), "missing".to_owned())
                    };
                    if let Some((_, filename)) = resolve {
                        if method == "GET" {
                            if filename == "chat_template.jinja" {
                                gets.fetch_add(1, Ordering::SeqCst);
                                if fail.load(Ordering::SeqCst) { status = "503 Service Unavailable"; }
                            }
                            if failures.lock().unwrap().contains(filename) {
                                status = "503 Service Unavailable";
                            }
                        }
                    }
                    let headers = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nETag: \"{etag}\"\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    socket.get_mut().write_all(headers.as_bytes()).await.unwrap();
                    if method != "HEAD" { socket.get_mut().write_all(&body).await.unwrap(); }
                    socket.get_mut().shutdown().await.unwrap();
                }).await.expect("fixture request timed out");
            }
        });
        Self {
            cache: tempfile::tempdir().unwrap(),
            endpoint,
            fail_template_get,
            template_gets,
            failed_gets,
            requests,
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

    fn reference(&self, revision: &str) -> PathBuf {
        self.cache
            .path()
            .join("hub/models--fixture--standalone-template/refs")
            .join(revision)
    }

    fn main_ref(&self) -> PathBuf {
        self.reference("main")
    }

    fn fail_get(&self, filename: &str, fail: bool) {
        let mut failures = self.failed_gets.lock().unwrap();
        if fail {
            failures.insert(filename.to_owned());
        } else {
            failures.remove(filename);
        }
    }

    fn file_requests(&self, method: &str, filename: &str) -> usize {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| {
                request.method == method
                    && request
                        .path
                        .strip_prefix(&format!("/{MODEL_ID}/resolve/"))
                        .and_then(|path| path.split_once('/'))
                        .is_some_and(|(_, file)| file == filename)
            })
            .count()
    }

    fn assert_pinned_reads(&self) {
        let tree_prefix = format!("/api/models/{MODEL_ID}/tree/");
        let resolve_prefix = format!("/{MODEL_ID}/resolve/");
        let requests = self.requests.lock().unwrap();
        let mut saw_tree = false;
        let mut saw_file = false;
        for request in requests.iter() {
            if let Some(path) = request.path.strip_prefix(&tree_prefix) {
                saw_tree = true;
                assert_eq!(path.split('/').next().unwrap(), REVISION, "{request:?}");
            }
            if let Some(path) = request.path.strip_prefix(&resolve_prefix) {
                saw_file = true;
                assert_eq!(path.split('/').next().unwrap(), REVISION, "{request:?}");
            }
        }
        assert!(saw_tree && saw_file, "{requests:?}");
    }
}

impl Drop for HubFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[tokio::test]
async fn gguf_download_selects_one_file_from_the_resolved_immutable_snapshot() {
    let filename = "weights/model-Q4_K_M.gguf";
    for revision in [None, Some(REVISION)] {
        let hub = HubFixture::with_files(
            vec![
                HubFile::new(filename, b"frozen GGUF payload"),
                HubFile::new("other.gguf", b"other recipe"),
            ],
            Some(vec![HubFile::new(filename, b"changed branch payload")]),
        )
        .await;
        let path = hub
            .downloader()
            .download_gguf(MODEL_ID, revision, filename)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"frozen GGUF payload");
        assert_eq!(hub.file_requests("GET", "other.gguf"), 0);
        hub.assert_pinned_reads();
        assert_eq!(
            std::fs::read_to_string(hub.reference(revision.unwrap_or("main"))).unwrap(),
            REVISION
        );
        if revision.is_some() {
            assert!(!hub.main_ref().exists());
        }
    }
}

#[tokio::test]
async fn fresh_single_gguf_download_and_repeat_reuse_the_selected_payload() {
    let hub = HubFixture::with_files(
        vec![
            HubFile::new("model.gguf", b"GGUF payload"),
            HubFile::new("config.json", b"{}"),
        ],
        None,
    )
    .await;
    let snapshot = hub.downloader().download(MODEL_ID, None).await.unwrap();
    assert_eq!(
        std::fs::read(snapshot.join("model.gguf")).unwrap(),
        b"GGUF payload"
    );
    assert_eq!(hub.file_requests("GET", "config.json"), 1);
    assert_eq!(hub.file_requests("GET", "model.gguf"), 1);
    assert_eq!(
        hub.downloader()
            .download_safetensors(MODEL_ID, None)
            .await
            .unwrap(),
        snapshot
    );
    assert_eq!(hub.file_requests("GET", "model.gguf"), 1);
    hub.assert_pinned_reads();
}

#[tokio::test]
async fn default_and_explicit_quantization_download_only_the_requested_variant() {
    let hub = HubFixture::with_files(
        vec![
            HubFile::new("weights/model-Q8_0.gguf", b"eight bit"),
            HubFile::new("weights/model-Q4_K_M.gguf", b"four bit"),
            HubFile::new("mmproj-F16.gguf", b"projector"),
            HubFile::new("config.json", b"{}"),
        ],
        None,
    )
    .await;
    let root = hub.downloader().download(MODEL_ID, None).await.unwrap();
    assert_eq!(
        std::fs::read(root.join("weights/model-Q4_K_M.gguf")).unwrap(),
        b"four bit"
    );
    assert_eq!(hub.file_requests("GET", "weights/model-Q8_0.gguf"), 0);
    let path = hub
        .downloader()
        .download_gguf_quantization(MODEL_ID, None, "q8_0")
        .await
        .unwrap();
    assert_eq!(std::fs::read(path).unwrap(), b"eight bit");
    assert_eq!(hub.file_requests("GET", "weights/model-Q4_K_M.gguf"), 1);
    assert_eq!(hub.file_requests("GET", "mmproj-F16.gguf"), 0);
    hub.assert_pinned_reads();
}

#[tokio::test]
async fn ambiguous_gguf_repository_reports_choices_before_downloading_sidecars() {
    let hub = HubFixture::with_files(
        vec![
            HubFile::new("first.gguf", b"GGUF payload"),
            HubFile::new("second.gguf", b"another model"),
            HubFile::new("config.json", b"{}"),
        ],
        None,
    )
    .await;
    let error = hub
        .downloader()
        .download(MODEL_ID, None)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("--gguf-file"), "{error}");
    assert_eq!(hub.file_requests("GET", "config.json"), 0);
    assert_eq!(hub.file_requests("GET", "first.gguf"), 0);
    assert!(!hub.main_ref().exists());
}

#[tokio::test]
async fn selected_gguf_repairs_failed_metadata_without_refetching_weights() {
    let filename = "weights/model-Q4_K_M.gguf";
    let hub = HubFixture::with_files(
        vec![
            HubFile::new(filename, b"selected GGUF payload"),
            HubFile::new("weights/model-Q8_0.gguf", b"unselected GGUF payload"),
            HubFile::new("config.json", b"{}"),
            HubFile::new("tokenizer.json", b"{}"),
            HubFile::new("chat_template.jinja", TEMPLATE.as_bytes()),
        ],
        None,
    )
    .await;
    let metadata = ["config.json", "tokenizer.json", "chat_template.jinja"];
    hub.fail_get("chat_template.jinja", true);
    let error = hub
        .downloader()
        .download_selected_gguf(
            MODEL_ID,
            None,
            super::GgufRequest::Quantization("q4_k_m"),
            &metadata,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("chat_template.jinja"), "{error}");
    assert!(!hub.main_ref().exists());
    let snapshot = hub
        .cache
        .path()
        .join("hub/models--fixture--standalone-template/snapshots")
        .join(REVISION);
    assert!(
        super::inspect_cached_metadata_selection(&snapshot, &metadata)
            .unwrap()
            .is_some()
    );
    hub.fail_get("chat_template.jinja", false);
    let downloaded = hub
        .downloader()
        .download_selected_gguf(
            MODEL_ID,
            None,
            super::GgufRequest::Quantization("Q4_K_M"),
            &metadata,
        )
        .await
        .unwrap();
    assert_eq!(downloaded, snapshot.join(filename));
    assert!(
        super::inspect_cached_metadata_selection(&snapshot, &metadata)
            .unwrap()
            .is_none()
    );
    assert_eq!(hub.file_requests("GET", filename), 1);
    assert_eq!(hub.file_requests("GET", "config.json"), 1);
    assert_eq!(hub.file_requests("GET", "weights/model-Q8_0.gguf"), 0);
    assert_eq!(hub.file_requests("GET", "chat_template.jinja"), 2);
    assert_eq!(std::fs::read_to_string(hub.main_ref()).unwrap(), REVISION);
    hub.assert_pinned_reads();
}

#[tokio::test]
async fn selected_gguf_does_not_fetch_metadata_roles_supplied_by_the_caller() {
    for metadata in [&[][..], &["config.json"][..]] {
        let hub = HubFixture::with_files(
            vec![
                HubFile::new("model.gguf", b"GGUF payload"),
                HubFile::new("config.json", b"{}"),
                HubFile::new("tokenizer.json", b"unused tokenizer"),
            ],
            None,
        )
        .await;
        hub.fail_get("tokenizer.json", true);
        hub.downloader()
            .download_selected_gguf(
                MODEL_ID,
                None,
                super::GgufRequest::File("model.gguf"),
                metadata,
            )
            .await
            .unwrap();
        assert_eq!(hub.file_requests("GET", "model.gguf"), 1);
        assert_eq!(
            hub.file_requests("GET", "config.json"),
            usize::from(!metadata.is_empty())
        );
        assert_eq!(hub.file_requests("HEAD", "tokenizer.json"), 0);
        assert_eq!(hub.file_requests("GET", "tokenizer.json"), 0);
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

fn assert_extra_weights_untouched(hub: &HubFixture, snapshot: &Path) {
    for path in EXTRA_WEIGHTS {
        for method in ["HEAD", "GET"] {
            assert_eq!(
                hub.file_requests(method, path),
                0,
                "unexpected {method} for {path}"
            );
        }
        assert!(
            !snapshot.join(path).exists(),
            "unselected snapshot file: {path}"
        );
    }
}

fn assert_indexed_snapshot(snapshot: &Path, expected: &[HubFile]) {
    for file in expected
        .iter()
        .filter(|file| !EXTRA_WEIGHTS.contains(&file.path))
    {
        assert_eq!(
            std::fs::read(snapshot.join(file.path)).unwrap(),
            file.bytes,
            "{}",
            file.path
        );
    }
    let archive = SafetensorsArchive::open(snapshot).unwrap();
    assert_eq!(archive.tensor_count(), 3);
    for (name, value, shard) in [
        ("weight.first", 1.0_f32, SHARD_A),
        ("weight.second", 2.0_f32, SHARD_A),
        ("weight.third", 3.0_f32, SHARD_B),
    ] {
        let tensor = archive.tensor(name).unwrap();
        assert_eq!(tensor.source_file(), shard);
        assert_eq!(tensor.bytes(), value.to_le_bytes());
    }
    let sources = ProductionModelSourceBundle::open_colocated_safetensors(snapshot).unwrap();
    assert_eq!(sources.chat_template_jinja(), Some(TEMPLATE.as_bytes()));
}

#[tokio::test]
async fn indexed_fresh_download_fetches_only_referenced_shards_and_sidecars() {
    let files = indexed_files();
    let hub = HubFixture::with_files(files.clone(), None).await;
    let snapshot = hub
        .downloader()
        .download_safetensors(MODEL_ID, None)
        .await
        .unwrap();
    // Transfer behavior is the oracle: an extra file must never even receive HEAD.
    assert_extra_weights_untouched(&hub, &snapshot);
    assert_indexed_snapshot(&snapshot, &files);
    for file in files
        .iter()
        .filter(|file| !EXTRA_WEIGHTS.contains(&file.path))
    {
        assert_eq!(hub.file_requests("GET", file.path), 1, "{}", file.path);
    }
    assert_eq!(std::fs::read_to_string(hub.main_ref()).unwrap(), REVISION);
    assert_eq!(snapshot.file_name().unwrap(), REVISION);
}

#[tokio::test]
async fn indexed_download_uses_resolved_snapshot_when_main_moves() {
    for explicit_revision in [None, Some(REVISION)] {
        let files = indexed_files();
        let mut branch_files = single_files(true);
        branch_files
            .iter_mut()
            .find(|f| f.path == "model.safetensors")
            .unwrap()
            .bytes = tensor_file(&[("branch_b.weight", 9.0)]);
        branch_files
            .iter_mut()
            .find(|f| f.path == "config.json")
            .unwrap()
            .bytes = br#"{"branch":"B"}"#.to_vec();
        let hub = HubFixture::with_files(files.clone(), Some(branch_files)).await;
        let result = hub
            .downloader()
            .download_safetensors(MODEL_ID, explicit_revision)
            .await;
        // main's tree/bytes are B and its A-index/shard URLs return 404. A cache
        // directory named A cannot conceal a read accidentally made against main.
        hub.assert_pinned_reads();
        let snapshot = result.unwrap();
        assert_indexed_snapshot(&snapshot, &files);
        let requested = explicit_revision.unwrap_or("main");
        assert_eq!(
            std::fs::read_to_string(hub.reference(requested)).unwrap(),
            REVISION
        );
        if explicit_revision.is_some() {
            assert!(
                !hub.main_ref().exists(),
                "explicit SHA must not publish refs/main"
            );
            assert!(hub.requests.lock().unwrap().iter().all(|request| {
                request.path != format!("/api/models/{MODEL_ID}/revision/main")
            }));
        }
    }
}

#[tokio::test]
async fn indexed_transfer_failure_preserves_ref_and_retry_completes() {
    const PREVIOUS: &str = "fedcba0987654321fedcba0987654321fedcba09";
    for (failed_file, previous_ref) in [(INDEX, None), (SHARD_B, Some(PREVIOUS))] {
        let files = indexed_files();
        let hub = HubFixture::with_files(files.clone(), None).await;
        if let Some(previous) = previous_ref {
            std::fs::create_dir_all(hub.main_ref().parent().unwrap()).unwrap();
            std::fs::write(hub.main_ref(), previous).unwrap();
        }
        hub.fail_get(failed_file, true);
        let downloader = hub.downloader();
        let error = downloader
            .download_safetensors(MODEL_ID, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(failed_file), "{error}");
        assert_eq!(
            std::fs::read_to_string(hub.main_ref()).ok().as_deref(),
            previous_ref
        );
        if failed_file == INDEX {
            for path in [SHARD_A, SHARD_B]
                .into_iter()
                .chain(EXTRA_WEIGHTS.iter().copied())
            {
                for method in ["HEAD", "GET"] {
                    assert_eq!(
                        hub.file_requests(method, path),
                        0,
                        "index failed but {method} {path} started"
                    );
                }
            }
        }
        hub.fail_get(failed_file, false);
        let snapshot = downloader
            .download_safetensors(MODEL_ID, None)
            .await
            .unwrap();
        assert_indexed_snapshot(&snapshot, &files);
        assert_extra_weights_untouched(&hub, &snapshot);
        assert_eq!(std::fs::read_to_string(hub.main_ref()).unwrap(), REVISION);
        assert_eq!(
            hub.file_requests("GET", failed_file),
            2,
            "failed file must be fetched again"
        );
    }
}

#[tokio::test]
async fn sidecar_inventory_and_payloads_keep_the_resolved_commit_when_main_moves() {
    let files = single_files(true);
    let mut branch_files = files.clone();
    branch_files
        .iter_mut()
        .find(|file| file.path == "chat_template.jinja")
        .unwrap()
        .bytes = b"different branch template and length".to_vec();
    let hub = HubFixture::with_files(files, Some(branch_files)).await;
    let downloader = hub.downloader();
    let selected = ["config.json", "tokenizer.json", "chat_template.jinja"];
    let snapshot = downloader
        .download_sidecar_files(MODEL_ID, None, &selected)
        .await
        .unwrap();
    hub.assert_pinned_reads();
    assert_eq!(
        std::fs::read(snapshot.join("chat_template.jinja")).unwrap(),
        TEMPLATE.as_bytes()
    );
    assert_eq!(std::fs::read_to_string(hub.main_ref()).unwrap(), REVISION);
    assert_eq!(
        super::inspect_cached_metadata_inventory(&snapshot).unwrap(),
        None
    );
    assert_eq!(hub.file_requests("GET", "model.safetensors"), 0);

    // Repair the same commit after its optional template link is lost. A record
    // of main's different size must not survive and make this cache incomplete.
    std::fs::remove_file(snapshot.join("chat_template.jinja")).unwrap();
    let reason = super::inspect_cached_metadata_inventory(&snapshot)
        .unwrap()
        .unwrap();
    assert!(reason.contains("chat_template.jinja"), "{reason}");
    let restored = downloader
        .download_sidecar_files(MODEL_ID, None, &selected)
        .await
        .unwrap();
    assert_eq!(restored, snapshot);
    hub.assert_pinned_reads();
    assert_eq!(
        std::fs::read(restored.join("chat_template.jinja")).unwrap(),
        TEMPLATE.as_bytes()
    );
    assert_eq!(
        super::inspect_cached_metadata_inventory(&restored).unwrap(),
        None
    );
}

#[tokio::test]
async fn ordinary_download_keeps_indexed_and_auxiliary_model_weights() {
    let mut files = indexed_files();
    files.push(HubFile::new(
        "speech_tokenizer/model.safetensors",
        tensor_file(&[("speech.weight", 9.0)]),
    ));
    let hub = HubFixture::with_files(files.clone(), None).await;
    let snapshot = hub.downloader().download(MODEL_ID, None).await.unwrap();
    // Ordinary downloads also serve composite models. A root index must not
    // prune an auxiliary model simply because it is absent from that index.
    for file in &files {
        assert_eq!(
            std::fs::read(snapshot.join(file.path)).unwrap(),
            file.bytes,
            "{}",
            file.path
        );
        assert_eq!(hub.file_requests("GET", file.path), 1, "{}", file.path);
    }
    assert_eq!(std::fs::read_to_string(hub.main_ref()).unwrap(), REVISION);
}

#[tokio::test]
async fn indexed_invalid_index_stops_before_weights_and_preserves_ref() {
    const PREVIOUS: &str = "fedcba0987654321fedcba0987654321fedcba09";
    const MISSING_SHARD: &str = "missing-part.safetensors";
    let cases = [
        ("malformed JSON", b"{".to_vec(), None),
        (
            "missing referenced shard",
            serde_json::to_vec(&json!({
                "weight_map": {
                    "weight.first": SHARD_A,
                    "weight.missing": MISSING_SHARD
                }
            }))
            .unwrap(),
            Some(PREVIOUS),
        ),
    ];
    for (label, index_bytes, previous_ref) in cases {
        let mut files = indexed_files();
        files
            .iter_mut()
            .find(|file| file.path == INDEX)
            .unwrap()
            .bytes = index_bytes;
        let hub = HubFixture::with_files(files, None).await;
        if let Some(previous) = previous_ref {
            std::fs::create_dir_all(hub.main_ref().parent().unwrap()).unwrap();
            std::fs::write(hub.main_ref(), previous).unwrap();
        }
        let error = hub
            .downloader()
            .download_safetensors(MODEL_ID, None)
            .await
            .expect_err(label);
        assert!(error.to_string().contains(INDEX), "{label}: {error}");
        assert_eq!(hub.file_requests("GET", INDEX), 1, "{label}");
        // Even a valid first reference must wait until the whole index validates.
        // Check the absent path too: a failed HEAD is already an unwanted transfer.
        for path in [SHARD_A, SHARD_B, MISSING_SHARD]
            .into_iter()
            .chain(EXTRA_WEIGHTS.iter().copied())
        {
            for method in ["HEAD", "GET"] {
                assert_eq!(
                    hub.file_requests(method, path),
                    0,
                    "{label}: unexpected {method} {path}"
                );
            }
        }
        assert_eq!(
            std::fs::read_to_string(hub.main_ref()).ok().as_deref(),
            previous_ref,
            "{label}: invalid index must not publish or replace the ref"
        );
    }
}
