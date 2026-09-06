use axum::{
    body::Body,
    extract::State,
    http::{Method, Response, Uri},
    Router,
};
use candle_core::quantized::gguf_file::{self, Value as GgufValue};
use safetensors::tensor::{serialize, Dtype, TensorView};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};
use tokio::{net::TcpListener, task::JoinHandle};

pub const REVISION: &str = "1234567890abcdef1234567890abcdef12345678";
type Files = BTreeMap<String, Vec<u8>>;
pub type ModelFiles = BTreeMap<String, Files>;

struct HubState {
    files: ModelFiles,
    requests: Mutex<Vec<(Method, String)>>,
}

pub struct Hub {
    pub endpoint: String,
    state: Arc<HubState>,
    server: JoinHandle<()>,
}

impl Hub {
    pub async fn start(files: ModelFiles) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(HubState {
            files,
            requests: Mutex::new(Vec::new()),
        });
        let router = Router::new().fallback(handle).with_state(state.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        Self {
            endpoint,
            state,
            server,
        }
    }

    pub fn requested(&self, method: &str, repo: &str, filename: &str) -> bool {
        self.state
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|(actual, path)| {
                actual.as_str() == method
                    && ["main", REVISION]
                        .iter()
                        .any(|revision| path == &format!("/{repo}/resolve/{revision}/{filename}"))
            })
    }

    pub fn clear_requests(&self) {
        self.state.requests.lock().unwrap().clear();
    }
}

impl Drop for Hub {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn handle(State(state): State<Arc<HubState>>, method: Method, uri: Uri) -> Response<Body> {
    let path = uri.path();
    state
        .requests
        .lock()
        .unwrap()
        .push((method.clone(), path.to_owned()));
    for (repo, files) in &state.files {
        for revision in ["main", REVISION] {
            if path == format!("/api/models/{repo}/revision/{revision}") {
                return response(
                    &method,
                    serde_json::to_vec(&json!({"sha": REVISION})).unwrap(),
                );
            }
            if path == format!("/api/models/{repo}/tree/{revision}") {
                let entries: Vec<_> = files
                    .iter()
                    .map(|(name, bytes)| json!({"path": name, "size": bytes.len(), "type": "file"}))
                    .collect();
                return response(&method, serde_json::to_vec(&entries).unwrap());
            }
            if let Some(name) = path.strip_prefix(&format!("/{repo}/resolve/{revision}/")) {
                if let Some(bytes) = files.get(name) {
                    return response(&method, bytes.clone());
                }
            }
        }
    }
    Response::builder()
        .status(404)
        .body(Body::from("unregistered fixture path"))
        .unwrap()
}

fn response(method: &Method, bytes: Vec<u8>) -> Response<Body> {
    Response::builder()
        .status(200)
        .header("content-length", bytes.len().to_string())
        .header("etag", format!("\"{:x}\"", Sha256::digest(&bytes)))
        .body(if method == Method::HEAD {
            Body::empty()
        } else {
            Body::from(bytes)
        })
        .unwrap()
}

pub fn sidecar_files() -> Files {
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_owned(), 0),
                    ("[BOS]".to_owned(), 1),
                    ("[EOS]".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    [
        (
            "config.json".into(),
            serde_json::to_vec(&json!({
                "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2,
                "vocab_size": 3, "max_position_embeddings": 32,
                "rms_norm_eps": 0.00001, "rope_theta": 10000.0,
                "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": false
            }))
            .unwrap(),
        ),
        (
            "tokenizer.json".into(),
            tokenizer.to_string(false).unwrap().into_bytes(),
        ),
        (
            "tokenizer_config.json".into(),
            serde_json::to_vec(&json!({
                "bos_token": "[BOS]", "eos_token": "[EOS]", "unk_token": "[UNK]",
                "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"
            }))
            .unwrap(),
        ),
    ]
    .into()
}

pub fn safetensors_files() -> Files {
    let bytes = 1.0f32.to_le_bytes();
    let tensor = TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
    let mut files = sidecar_files();
    files.insert(
        "model.safetensors".into(),
        serialize([("fixture.sentinel", tensor)], &None).unwrap(),
    );
    files
}

pub fn gguf_without_weights() -> Vec<u8> {
    let metadata = [
        ("general.architecture", GgufValue::String("llama".into())),
        ("llama.block_count", GgufValue::U32(1)),
        ("llama.embedding_length", GgufValue::U32(8)),
        ("llama.feed_forward_length", GgufValue::U32(16)),
        ("llama.attention.head_count", GgufValue::U32(2)),
        ("llama.attention.head_count_kv", GgufValue::U32(2)),
        (
            "llama.attention.layer_norm_rms_epsilon",
            GgufValue::F32(0.00001),
        ),
        ("llama.context_length", GgufValue::U32(32)),
        ("llama.vocab_size", GgufValue::U32(3)),
    ];
    let metadata: Vec<_> = metadata
        .iter()
        .map(|(name, value)| (*name, value))
        .collect();
    let mut out = Cursor::new(Vec::new());
    gguf_file::write(&mut out, &metadata, &[]).unwrap();
    out.into_inner()
}
