use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX_EPOCH")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ferrum-qwen35-{prefix}-{nanos}"));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn ferrum_bin() -> PathBuf {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ferrum") {
        return PathBuf::from(bin);
    }

    let current = std::env::current_exe().expect("failed to get current test executable path");
    let target_debug_dir = current
        .parent()
        .and_then(|p| p.parent())
        .expect("failed to locate target/debug directory from test executable");

    let mut bin = target_debug_dir.join("ferrum");
    if cfg!(windows) {
        bin.set_extension("exe");
    }

    assert!(bin.exists(), "ferrum binary not found at {}", bin.display());
    bin
}

fn run(cmd: &mut Command) -> Output {
    cmd.output().expect("failed to run ferrum command")
}

fn write_qwen35_reference_config(dir: &Path) {
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vocab_size": 3,
        "max_position_embeddings": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": 1,
            "linear_conv_kernel_dim": 1,
            "head_dim": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "tie_word_embeddings": false
        }
    });
    fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn write_qwen35_reference_tokenizer(dir: &Path) {
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "<unk>": 0,
                "alpha": 1,
                "beta": 2
            },
            "unk_token": "<unk>"
        }
    });
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer).unwrap(),
    )
    .unwrap();
}

fn write_qwen35_reference_safetensors(dir: &Path) {
    let tensors: Vec<(String, Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        ("model.norm.weight".to_string(), vec![0.0, 0.0]),
        (
            "model.lm_head.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        (
            "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
            vec![1.0, -1.0],
        ),
        (
            "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
            vec![0.5, 0.25],
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
            vec![-0.25, 0.75],
        ),
        (
            "model.layers.0.linear_attn.conv1d.weight".to_string(),
            vec![1.0, 1.0, 1.0],
        ),
        ("model.layers.0.linear_attn.A_log".to_string(), vec![0.0]),
        ("model.layers.0.linear_attn.dt_bias".to_string(), vec![0.0]),
        (
            "model.layers.0.linear_attn.norm.weight".to_string(),
            vec![1.0],
        ),
        (
            "model.layers.0.linear_attn.out_proj.weight".to_string(),
            vec![1.0, -0.5],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![0.2, 0.1, -0.1, 0.3],
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            vec![0.4, -0.2, 0.3, 0.5],
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.1.input_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.1.post_attention_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.1.self_attn.k_proj.weight".to_string(),
            vec![0.5, 0.0, 0.0, 0.5],
        ),
        (
            "model.layers.1.self_attn.v_proj.weight".to_string(),
            vec![1.0, 1.0, -0.5, 0.5],
        ),
        (
            "model.layers.1.self_attn.o_proj.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.1.self_attn.q_norm.weight".to_string(),
            vec![1.0, 1.0],
        ),
        (
            "model.layers.1.self_attn.k_norm.weight".to_string(),
            vec![1.0, 1.0],
        ),
        (
            "model.layers.1.mlp.gate_proj.weight".to_string(),
            vec![-0.2, 0.2, 0.1, 0.3],
        ),
        (
            "model.layers.1.mlp.up_proj.weight".to_string(),
            vec![0.25, 0.5, -0.3, 0.4],
        ),
        (
            "model.layers.1.mlp.down_proj.weight".to_string(),
            vec![0.5, 0.25, -0.2, 0.75],
        ),
    ];
    let views = tensors
        .into_iter()
        .map(|(name, values)| {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let bytes: &'static [u8] = Box::leak(bytes);
            (
                name,
                TensorView::new(Dtype::F32, vec![values.len()], bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        views,
        &None::<HashMap<String, String>>,
        &dir.join("model.safetensors"),
    )
    .unwrap();
}

fn write_qwen35_reference_model_dir(root: &Path) -> PathBuf {
    let model_dir = root.join("qwen35-reference-model");
    fs::create_dir_all(&model_dir).unwrap();
    write_qwen35_reference_config(&model_dir);
    write_qwen35_reference_tokenizer(&model_dir);
    write_qwen35_reference_safetensors(&model_dir);
    model_dir
}

#[test]
fn ferrum_run_qwen35_reference_oneshot_uses_product_path() {
    let workspace = TempDirGuard::new("run-reference");
    let model_dir = write_qwen35_reference_model_dir(workspace.path());
    let mut cmd = Command::new(ferrum_bin());
    cmd.current_dir(workspace.path())
        .arg("run")
        .arg(&model_dir)
        .arg("--backend")
        .arg("cpu")
        .arg("--qwen35-reference")
        .arg("--output-format")
        .arg("jsonl")
        .arg("--temperature")
        .arg("0")
        .arg("--max-tokens")
        .arg("2")
        .arg("--prompt")
        .arg("hello")
        .env("NO_COLOR", "1")
        .env("HF_HOME", workspace.path().join("hf-cache"));

    let output = run(&mut cmd);

    assert!(
        output.status.success(),
        "ferrum run failed\nstatus={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let assistant = stdout
        .lines()
        .find_map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).ok()?;
            (value.get("event").and_then(|v| v.as_str()) == Some("assistant")).then_some(value)
        })
        .expect("missing assistant JSONL event");
    assert_eq!(
        assistant["finish_reason"].as_str(),
        Some("length"),
        "assistant event={assistant}\nstdout={stdout}\nstderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(assistant["n_tokens"].as_u64(), Some(2));
    assert!(
        assistant["content"]
            .as_str()
            .is_some_and(|content| !content.trim().is_empty()),
        "assistant content should be non-empty: {assistant}"
    );
}
