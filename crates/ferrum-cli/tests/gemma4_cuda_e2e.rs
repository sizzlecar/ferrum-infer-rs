//! End-to-end CUDA canary for the typed Gemma 4 Unified text package.
//!
//! The fixture is generated entirely in Rust inside a temporary directory. It
//! keeps Gemma 4's hybrid sliding/full attention and compressed-tensors W4A16
//! source ABI, while using small dimensions that still satisfy Marlin's thread
//! tiles. The test then exercises both public product entrypoints.

#![recursion_limit = "256"]

use reqwest::Client;
use safetensors::tensor::{serialize_to_file, Dtype, View};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "gemma4-unified-tiny-w4a16";
const HIDDEN_SIZE: usize = 512;
const INTERMEDIATE_SIZE: usize = 512;
const LOCAL_HEAD_DIM: usize = 256;
const GLOBAL_HEAD_DIM: usize = 512;
const QUERY_HEADS: usize = 16;
const LOCAL_KV_HEADS: usize = 8;
const GLOBAL_KV_HEADS: usize = 1;
const VOCABULARY_SIZE: usize = 256;
const INPUT_TOKEN_ID: usize = 5;
const MAX_LIVE_CHILDREN: usize = 1;
const PACKED_SERVE_CONCURRENCIES: [usize; 2] = [8, 32];
const RUN_DEADLINE: Duration = Duration::from_secs(120);
const STARTUP_DEADLINE: Duration = Duration::from_secs(180);
const REQUEST_DEADLINE: Duration = Duration::from_secs(120);

static LIVE_CHILDREN: AtomicUsize = AtomicUsize::new(0);

struct FixtureTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for FixtureTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Process guard with an independent, source-level cap. A failed spawn gives
/// the slot back; every other exit path either observes completion or kills and
/// joins the child before releasing it.
struct BoundedChild {
    child: Option<Child>,
    slot_held: bool,
}

impl BoundedChild {
    fn spawn(command: &mut Command) -> Self {
        LIVE_CHILDREN
            .compare_exchange(0, MAX_LIVE_CHILDREN, Ordering::AcqRel, Ordering::Acquire)
            .unwrap_or_else(|active| {
                panic!(
                    "Gemma 4 CUDA E2E child cap exceeded: active={active}, cap={MAX_LIVE_CHILDREN}"
                )
            });
        match command.spawn() {
            Ok(child) => Self {
                child: Some(child),
                slot_held: true,
            },
            Err(error) => {
                LIVE_CHILDREN.store(0, Ordering::Release);
                panic!("spawn {:?}: {error}", command);
            }
        }
    }

    fn poll(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let status = self
            .child
            .as_mut()
            .expect("bounded child already reaped")
            .try_wait()?;
        if status.is_some() {
            self.child.take();
            self.release_slot();
        }
        Ok(status)
    }

    fn kill_and_wait(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.release_slot();
    }

    fn release_slot(&mut self) {
        if self.slot_held {
            let previous = LIVE_CHILDREN.swap(0, Ordering::AcqRel);
            assert_eq!(previous, MAX_LIVE_CHILDREN, "child slot accounting drift");
            self.slot_held = false;
        }
    }
}

impl Drop for BoundedChild {
    fn drop(&mut self) {
        self.kill_and_wait();
    }
}

struct ServerFixture {
    url: String,
    child: BoundedChild,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    profile_path: PathBuf,
}

impl ServerFixture {
    async fn spawn(model_dir: &Path, log_root: &Path) -> Self {
        let port = free_port();
        let url = format!("http://127.0.0.1:{port}");
        let stdout_path = log_root.join("serve.stdout.log");
        let stderr_path = log_root.join("serve.stderr.log");
        let profile_path = log_root.join("serve.profile.jsonl");
        let stdout = File::create(&stdout_path).expect("create serve stdout log");
        let stderr = File::create(&stderr_path).expect("create serve stderr log");
        let mut command = Command::new(ferrum_bin());
        command
            .args(["serve", model_dir.to_str().expect("UTF-8 fixture path")])
            .args(["--backend", "cuda"])
            .args(["--disable-thinking", "--port", &port.to_string()])
            .args(["--served-model-name", MODEL_NAME])
            .args(["--max-num-seqs", "32", "--max-num-batched-tokens", "64"])
            .args(["--profile-detail", "replay", "--profile-jsonl"])
            .arg(&profile_path)
            .args(["--profile-concurrency", "32"])
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        let mut child = BoundedChild::spawn(&mut command);

        let client = http_client();
        let health_url = format!("{url}/health");
        let started = Instant::now();
        loop {
            if let Some(status) = child.poll().expect("poll ferrum serve") {
                panic!(
                    "ferrum serve exited before health with {status}; stdout:\n{}\nstderr:\n{}",
                    read_log(&stdout_path),
                    read_log(&stderr_path)
                );
            }
            if started.elapsed() >= STARTUP_DEADLINE {
                child.kill_and_wait();
                panic!(
                    "ferrum serve missed {STARTUP_DEADLINE:?} startup deadline; stdout:\n{}\nstderr:\n{}",
                    read_log(&stdout_path),
                    read_log(&stderr_path)
                );
            }
            let ready = client
                .get(&health_url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .map(|response| response.status().is_success())
                .unwrap_or(false);
            if ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Self {
            url,
            child,
            stdout_path,
            stderr_path,
            profile_path,
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.url)
    }

    fn logs(&self) -> String {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            read_log(&self.stdout_path),
            read_log(&self.stderr_path)
        )
    }

    fn profile(&self) -> String {
        read_log(&self.profile_path)
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        self.child.kill_and_wait();
    }
}

fn ferrum_bin() -> PathBuf {
    if let Ok(binary) = std::env::var("CARGO_BIN_EXE_ferrum") {
        return PathBuf::from(binary);
    }
    let current = std::env::current_exe().expect("test executable path");
    let target_profile = current
        .parent()
        .and_then(Path::parent)
        .expect("target profile directory");
    let mut binary = target_profile.join("ferrum");
    if cfg!(windows) {
        binary.set_extension("exe");
    }
    assert!(
        binary.is_file(),
        "ferrum binary not found at {}",
        binary.display()
    );
    binary
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("ephemeral local address")
        .port()
}

fn http_client() -> Client {
    Client::builder()
        .timeout(REQUEST_DEADLINE)
        .build()
        .expect("build HTTP client")
}

fn read_log(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| format!("<read {}: {error}>", path.display()))
}

fn assert_clean_logs(label: &str, text: &str) {
    let lowercase = text.to_ascii_lowercase();
    for marker in [
        "panicked at",
        "out of memory",
        "cuda error",
        "invalid utf-8",
    ] {
        assert!(
            !lowercase.contains(marker),
            "{label} contains fatal marker {marker:?}:\n{text}"
        );
    }
    for marker in ["<unk>", "[PAD]"] {
        assert!(
            !text.contains(marker),
            "{label} leaked tokenizer marker {marker:?}:\n{text}"
        );
    }
}

fn wait_for_run(child: &mut BoundedChild, stdout_path: &Path, stderr_path: &Path) -> ExitStatus {
    let started = Instant::now();
    loop {
        if let Some(status) = child.poll().expect("poll ferrum run") {
            return status;
        }
        if started.elapsed() >= RUN_DEADLINE {
            child.kill_and_wait();
            panic!(
                "ferrum run missed {RUN_DEADLINE:?} deadline; stdout:\n{}\nstderr:\n{}",
                read_log(stdout_path),
                read_log(stderr_path)
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn run_entrypoint(model_dir: &Path, log_root: &Path) {
    let stdout_path = log_root.join("run.stdout.jsonl");
    let stderr_path = log_root.join("run.stderr.log");
    let stdout = File::create(&stdout_path).expect("create run stdout log");
    let stderr = File::create(&stderr_path).expect("create run stderr log");
    let mut command = Command::new(ferrum_bin());
    command
        .args(["run", model_dir.to_str().expect("UTF-8 fixture path")])
        .args(["--backend", "cuda"])
        .args(["--disable-thinking", "--output-format", "jsonl"])
        .args(["--temperature", "0", "--max-tokens", "2"])
        .args(["--prompt", "hello"])
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    let mut child = BoundedChild::spawn(&mut command);
    let status = wait_for_run(&mut child, &stdout_path, &stderr_path);
    assert!(
        status.success(),
        "ferrum run exited with {status}; stdout:\n{}\nstderr:\n{}",
        read_log(&stdout_path),
        read_log(&stderr_path)
    );

    let records = read_log(&stdout_path)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<Value>(line)
                .unwrap_or_else(|error| panic!("invalid run JSONL {line:?}: {error}"))
        })
        .collect::<Vec<_>>();
    let assistant = records
        .iter()
        .find(|record| record["event"] == "assistant")
        .unwrap_or_else(|| panic!("run emitted no assistant event: {records:?}"));
    assert!(
        assistant["content"]
            .as_str()
            .is_some_and(|content| !content.trim().is_empty()),
        "run assistant content is empty: {assistant:?}"
    );
    assert_clean_logs("ferrum run stdout", &read_log(&stdout_path));
    assert_clean_logs("ferrum run stderr", &read_log(&stderr_path));
}

fn parse_stream(body: &str) -> (usize, String, u64) {
    let mut done_count = 0;
    let mut content = String::new();
    let mut completion_tokens = 0;
    for line in body.lines() {
        let Some(data) = line.trim().strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            done_count += 1;
            continue;
        }
        if data.is_empty() {
            continue;
        }
        let chunk: Value = serde_json::from_str(data)
            .unwrap_or_else(|error| panic!("invalid SSE JSON {data:?}: {error}"));
        if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
            content.push_str(delta);
        }
        if let Some(tokens) = chunk["usage"]["completion_tokens"].as_u64() {
            completion_tokens = completion_tokens.max(tokens);
        }
    }
    (done_count, content, completion_tokens)
}

async fn stream_chat(chat_url: String) -> Result<(reqwest::StatusCode, String), String> {
    let response = http_client()
        .post(chat_url)
        .json(&json!({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .send()
        .await
        .map_err(|error| format!("stream request failed: {error}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("read stream body: {error}"))?;
    Ok((status, body))
}

fn assert_stream_response(label: &str, status: reqwest::StatusCode, body: &str) -> (String, u64) {
    assert_eq!(status, 200, "{label} returned non-200: {body}");
    let (done_count, content, completion_tokens) = parse_stream(body);
    assert_eq!(
        done_count, 1,
        "{label} must contain exactly one [DONE]: {body}"
    );
    assert!(
        !content.trim().is_empty(),
        "{label} streamed empty content: {body}"
    );
    assert!(
        completion_tokens > 0,
        "{label} usage has no completion tokens: {body}"
    );
    assert_clean_logs(label, body);
    (content, completion_tokens)
}

fn maximum_profile_participants(profile: &str) -> usize {
    profile
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|event| event["phase"] == "vnext.device_physical_submission")
        .filter_map(|event| event["shape"]["participant_count"].as_u64())
        .filter_map(|count| usize::try_from(count).ok())
        .max()
        .unwrap_or(0)
}

fn packed_native_work_summary(
    profile: &str,
    participant_count: usize,
) -> Vec<(String, usize, u64)> {
    let mut summary = BTreeMap::<String, (usize, u64)>::new();
    for event in profile
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|event| event["phase"] == "vnext.device_native_work")
        .filter(|event| {
            event["shape"]["participant_count"].as_u64() == u64::try_from(participant_count).ok()
                && event["attributes"]["batching_form"] == "packed"
        })
    {
        let Some(operation) = event["attributes"]["native_op_id"].as_str() else {
            continue;
        };
        let timing = summary.entry(operation.to_owned()).or_default();
        timing.0 += 1;
        timing.1 += event["shape"]["device_elapsed_ns"].as_u64().unwrap_or(0);
    }
    let mut summary = summary
        .into_iter()
        .map(|(operation, (samples, elapsed_ns))| (operation, samples, elapsed_ns))
        .collect::<Vec<_>>();
    summary.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
    summary
}

async fn wait_for_profile_participants(server: &ServerFixture, expected: usize) -> usize {
    let started = Instant::now();
    loop {
        let maximum = maximum_profile_participants(&server.profile());
        if maximum >= expected || started.elapsed() >= Duration::from_secs(5) {
            return maximum;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn bf16_constant(shape: &[usize], bits: u16) -> FixtureTensor {
    let elements = shape.iter().product::<usize>();
    let bytes = (0..elements).flat_map(|_| bits.to_le_bytes()).collect();
    FixtureTensor {
        dtype: Dtype::BF16,
        shape: shape.to_vec(),
        bytes,
    }
}

fn deterministic_embedding() -> FixtureTensor {
    let mut tensor = bf16_constant(&[VOCABULARY_SIZE, HIDDEN_SIZE], 0);
    let row_start = INPUT_TOKEN_ID * HIDDEN_SIZE * 2;
    for element in tensor.bytes[row_start..row_start + HIDDEN_SIZE * 2].chunks_exact_mut(2) {
        element.copy_from_slice(&0x3f80_u16.to_le_bytes());
    }
    tensor
}

fn insert_quantized_projection(
    tensors: &mut BTreeMap<String, FixtureTensor>,
    stem: impl Into<String>,
    n: usize,
    k: usize,
) {
    let stem = stem.into();
    assert!(
        (n.is_multiple_of(64) && k.is_multiple_of(128))
            || (n.is_multiple_of(128) && k.is_multiple_of(64)),
        "projection {stem} N={n} K={k} is outside Marlin thread tiles"
    );
    assert!(k.is_multiple_of(32), "projection K must be group32");

    // compressed-tensors pack-quantized symmetric INT4 uses biased nibble 8
    // for zero. Alternating 8/9 nibbles plus a 2^-6 scale keeps this fixture
    // numerically bounded while ensuring attention projections are non-zero.
    tensors.insert(
        format!("{stem}.weight_packed"),
        FixtureTensor {
            dtype: Dtype::I32,
            shape: vec![n, k / 8],
            bytes: vec![0x98; n * k / 2],
        },
    );
    tensors.insert(
        format!("{stem}.weight_scale"),
        bf16_constant(&[n, k / 32], 0x3c80),
    );
    tensors.insert(
        format!("{stem}.weight_shape"),
        FixtureTensor {
            dtype: Dtype::I64,
            shape: vec![2],
            bytes: [n as i64, k as i64]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
        },
    );
}

fn insert_layer(tensors: &mut BTreeMap<String, FixtureTensor>, layer: usize, full_attention: bool) {
    let prefix = format!("model.language_model.layers.{layer}");
    let head_dim = if full_attention {
        GLOBAL_HEAD_DIM
    } else {
        LOCAL_HEAD_DIM
    };
    let kv_heads = if full_attention {
        GLOBAL_KV_HEADS
    } else {
        LOCAL_KV_HEADS
    };
    let query_features = QUERY_HEADS * head_dim;
    let kv_features = kv_heads * head_dim;
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
    ] {
        tensors.insert(
            format!("{prefix}.{suffix}"),
            bf16_constant(&[HIDDEN_SIZE], 0x3f80),
        );
    }
    for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
        tensors.insert(
            format!("{prefix}.{suffix}"),
            bf16_constant(&[head_dim], 0x3f80),
        );
    }
    tensors.insert(
        format!("{prefix}.layer_scalar"),
        bf16_constant(&[1], if full_attention { 0x3f40 } else { 0x3f00 }),
    );

    insert_quantized_projection(
        tensors,
        format!("{prefix}.self_attn.q_proj"),
        query_features,
        HIDDEN_SIZE,
    );
    insert_quantized_projection(
        tensors,
        format!("{prefix}.self_attn.k_proj"),
        kv_features,
        HIDDEN_SIZE,
    );
    if !full_attention {
        insert_quantized_projection(
            tensors,
            format!("{prefix}.self_attn.v_proj"),
            kv_features,
            HIDDEN_SIZE,
        );
    }
    insert_quantized_projection(
        tensors,
        format!("{prefix}.self_attn.o_proj"),
        HIDDEN_SIZE,
        query_features,
    );
    for projection in ["gate_proj", "up_proj"] {
        insert_quantized_projection(
            tensors,
            format!("{prefix}.mlp.{projection}"),
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        );
    }
    insert_quantized_projection(
        tensors,
        format!("{prefix}.mlp.down_proj"),
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
    );
}

fn write_tokenizer(model_dir: &Path) {
    let vocab = [
        ("ordinary".to_owned(), 0),
        ("<pad>".to_owned(), 1),
        ("<eos>".to_owned(), 2),
        ("<bos>".to_owned(), 3),
        ("<unk>".to_owned(), 4),
        ("hello".to_owned(), INPUT_TOKEN_ID as u32),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("build WordLevel tokenizer");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    assert_eq!(
        tokenizer.encode("hello", false).unwrap().get_ids(),
        &[INPUT_TOKEN_ID as u32]
    );
    assert_eq!(
        tokenizer.decode(&[INPUT_TOKEN_ID as u32], false).unwrap(),
        "hello"
    );
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .expect("write tokenizer.json");
    fs::write(
        model_dir.join("tokenizer_config.json"),
        serde_json::to_vec_pretty(&json!({
            "bos_token_id": 3,
            "eos_token_id": 2,
            "pad_token_id": 1,
            "model_max_length": 1024,
            "chat_template": null
        }))
        .unwrap(),
    )
    .expect("write tokenizer_config.json");
    fs::write(
        model_dir.join("chat_template.jinja"),
        "{%- for message in messages -%}{{ message['content'] }}{%- endfor -%}",
    )
    .expect("write chat_template.jinja");
    fs::write(
        model_dir.join("generation_config.json"),
        serde_json::to_vec_pretty(&json!({
            "bos_token_id": 3,
            "eos_token_id": 2,
            "pad_token_id": 1
        }))
        .unwrap(),
    )
    .expect("write generation_config.json");
}

fn write_config(model_dir: &Path) {
    let config = json!({
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "model_type": "gemma4_unified",
        "moe_intermediate_size": null,
        "num_experts": null,
        "num_experts_per_tok": null,
        "tie_word_embeddings": true,
        "quantization_config": {
            "config_groups": {"group_0": {
                "format": "pack-quantized",
                "input_activations": null,
                "output_activations": null,
                "targets": ["Linear"],
                "weights": {
                    "actorder": null,
                    "block_structure": null,
                    "dynamic": false,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "memoryless_minmax",
                    "observer_kwargs": {},
                    "scale_dtype": null,
                    "strategy": "group",
                    "symmetric": true,
                    "type": "int",
                    "zp_dtype": null
                }
            }},
            "format": "pack-quantized",
            "ignore": [
                "lm_head",
                "model.embed_vision.patch_dense",
                "model.embed_vision.multimodal_embedder.embedding_projection",
                "model.embed_audio.embedding_projection"
            ],
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.17.1.ferrum-tiny-e2e"
        },
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": true,
            "dtype": "bfloat16",
            "enable_moe_block": false,
            "final_logit_softcapping": 30.0,
            "global_head_dim": GLOBAL_HEAD_DIM,
            "head_dim": LOCAL_HEAD_DIM,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": HIDDEN_SIZE,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": INTERMEDIATE_SIZE,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 1024,
            "model_type": "gemma4_unified_text",
            "moe_intermediate_size": null,
            "num_attention_heads": QUERY_HEADS,
            "num_experts": null,
            "num_global_key_value_heads": GLOBAL_KV_HEADS,
            "num_hidden_layers": 2,
            "num_key_value_heads": LOCAL_KV_HEADS,
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "sliding_window": 1024,
            "tie_word_embeddings": true,
            "top_k_experts": null,
            "use_bidirectional_attention": "vision",
            "use_cache": true,
            "use_double_wide_mlp": false,
            "vocab_size": VOCABULARY_SIZE,
            "vocab_size_per_layer_input": 0
        }
    });
    fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .expect("write config.json");
}

fn write_weights(model_dir: &Path) {
    let mut tensors = BTreeMap::new();
    let embedding = deterministic_embedding();
    let lm_head = FixtureTensor {
        dtype: embedding.dtype,
        shape: embedding.shape.clone(),
        bytes: embedding.bytes.clone(),
    };
    tensors.insert(
        "model.language_model.embed_tokens.weight".to_owned(),
        embedding,
    );
    tensors.insert(
        "model.language_model.norm.weight".to_owned(),
        bf16_constant(&[HIDDEN_SIZE], 0x3f80),
    );
    tensors.insert("lm_head.weight".to_owned(), lm_head);
    insert_layer(&mut tensors, 0, false);
    insert_layer(&mut tensors, 1, true);

    for name in [
        "model.embed_audio.embedding_projection.weight",
        "model.embed_vision.embedding_projection.weight",
        "model.vision_embedder.patch_dense.bias",
        "model.vision_embedder.patch_dense.weight",
        "model.vision_embedder.patch_ln1.bias",
        "model.vision_embedder.patch_ln1.weight",
        "model.vision_embedder.patch_ln2.bias",
        "model.vision_embedder.patch_ln2.weight",
        "model.vision_embedder.pos_embedding",
        "model.vision_embedder.pos_norm.bias",
        "model.vision_embedder.pos_norm.weight",
    ] {
        tensors.insert(name.to_owned(), bf16_constant(&[1], 0));
    }
    assert_eq!(tensors.len(), 67, "typed Gemma 4 fixture tensor count");

    serialize_to_file(
        tensors,
        &Some(HashMap::from([
            ("format".to_owned(), "pt".to_owned()),
            (
                "ferrum_generator".to_owned(),
                "gemma4-unified-tiny-w4a16-rust-e2e".to_owned(),
            ),
        ])),
        &model_dir.join("model.safetensors"),
    )
    .expect("write model.safetensors");
}

fn generate_fixture() -> (TempDir, PathBuf) {
    let root = tempfile::tempdir().expect("create Gemma 4 fixture root");
    let model_dir = root.path().join(MODEL_NAME);
    fs::create_dir(&model_dir).expect("create Gemma 4 fixture model directory");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    write_weights(&model_dir);
    (root, model_dir)
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a CUDA GPU and vllm-moe-marlin; run explicitly on panda-pad"]
async fn gemma4_tiny_cuda_run_and_serve_e2e() {
    assert!(
        cfg!(all(feature = "cuda", feature = "vllm-moe-marlin")),
        "build this test with --features cuda,vllm-moe-marlin,vllm-paged-attn-v2"
    );
    let (fixture, model_dir) = generate_fixture();
    run_entrypoint(&model_dir, fixture.path());
    assert_eq!(LIVE_CHILDREN.load(Ordering::Acquire), 0);

    let server = ServerFixture::spawn(&model_dir, fixture.path()).await;
    let c1_started = Instant::now();
    let (status, body) = stream_chat(server.chat_url())
        .await
        .unwrap_or_else(|error| panic!("{error}; {}", server.logs()));
    let (expected_content, expected_completion_tokens) =
        assert_stream_response("ferrum serve c=1 stream", status, &body);
    eprintln!(
        "GEMMA4 CUDA E2E c=1 elapsed_ms={}",
        c1_started.elapsed().as_millis()
    );

    for concurrency in PACKED_SERVE_CONCURRENCIES {
        let batch_started = Instant::now();
        let mut requests = tokio::task::JoinSet::new();
        for _ in 0..concurrency {
            requests.spawn(stream_chat(server.chat_url()));
        }
        let mut completed = 0;
        while let Some(result) = requests.join_next().await {
            let (status, body) = result
                .unwrap_or_else(|error| {
                    panic!(
                        "c={concurrency} stream task failed: {error}; {}",
                        server.logs()
                    )
                })
                .unwrap_or_else(|error| panic!("{error}; {}", server.logs()));
            let (content, completion_tokens) = assert_stream_response(
                &format!("ferrum serve c={concurrency} stream"),
                status,
                &body,
            );
            assert_eq!(
                content, expected_content,
                "c={concurrency} packed output differs from c=1 scalar output"
            );
            assert_eq!(
                completion_tokens, expected_completion_tokens,
                "c={concurrency} packed usage differs from c=1 scalar usage"
            );
            completed += 1;
        }
        assert_eq!(completed, concurrency);
        eprintln!(
            "GEMMA4 CUDA E2E c={concurrency} elapsed_ms={}",
            batch_started.elapsed().as_millis()
        );
    }
    let maximum_participants = wait_for_profile_participants(&server, 32).await;
    assert_eq!(
        maximum_participants,
        32,
        "profile never observed a true c=32 physical submission: {}",
        server.profile()
    );
    eprintln!("GEMMA4 CUDA E2E max_profile_participants={maximum_participants}");
    let native_work = packed_native_work_summary(&server.profile(), 32);
    assert!(
        native_work
            .iter()
            .any(|(operation, _, _)| operation == "vnext.causal_attention.token_major_fallback"),
        "profile never observed packed Gemma4 attention work: {native_work:?}"
    );
    eprintln!("GEMMA4 CUDA E2E c=32 packed_native_work={native_work:?}");
    assert_clean_logs("ferrum serve logs", &server.logs());
    drop(server);
    assert_eq!(LIVE_CHILDREN.load(Ordering::Acquire), 0);
}
