//! Embed command - Generate embeddings using BERT models

use crate::config::CliConfig;
use candle_core::Device as CandleDevice;
use clap::Args;
use colored::Colorize;
use ferrum_models::source::{ModelFormat, ResolvedModelSource};
use ferrum_models::HfDownloader;
use ferrum_models::{BertModelExecutor, ConfigManager};
use ferrum_types::Result;
use std::io::{self, BufRead};
use std::path::PathBuf;

/// Generate embeddings using a BERT model
#[derive(Args, Debug)]
pub struct EmbedCommand {
    /// Model name (e.g., google-bert/bert-base-chinese)
    #[arg(required = true)]
    pub model: String,

    /// Text to embed (if not provided, reads from stdin)
    #[arg(short, long)]
    pub text: Option<String>,

    /// Output format: json, csv, or raw
    #[arg(short, long, default_value = "json")]
    pub format: String,

    /// Normalize embeddings to unit length
    #[arg(short, long, default_value = "true")]
    pub normalize: bool,
}

pub async fn execute(cmd: EmbedCommand, config: CliConfig) -> Result<()> {
    eprintln!("{}", format!("Loading {}...", cmd.model).dimmed());

    // Resolve model path using same logic as list/run commands
    let model_id = cmd.model.clone();
    let cache_dir = get_hf_cache_dir(&config);

    let source = match find_cached_model(&cache_dir, &model_id) {
        Some(source) => source,
        None => {
            eprintln!(
                "{} Model '{}' not found locally, downloading...",
                "📥".cyan(),
                model_id
            );

            let token = std::env::var("HF_TOKEN")
                .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                .ok();

            let downloader = HfDownloader::new(cache_dir, token)?;
            let snapshot_path = downloader.download(&model_id, None).await?;

            let format = detect_format(&snapshot_path);
            if format == ModelFormat::Unknown {
                return Err(ferrum_types::FerrumError::model(
                    "Downloaded model has unknown format",
                ));
            }

            ResolvedModelSource {
                original: model_id.clone(),
                local_path: snapshot_path,
                format,
                from_cache: false,
            }
        }
    };

    let model_path = source.local_path.to_string_lossy().to_string();
    eprintln!("{}", "Using CPU backend".dimmed());

    // Load model definition
    let mut config_manager = ConfigManager::new();
    let model_def = config_manager.load_from_path(&source.local_path).await?;

    // Load BERT executor
    let device = CandleDevice::Cpu;
    let executor = BertModelExecutor::from_path(&model_path, &model_def, device).await?;

    eprintln!("{}", "Model loaded. Ready for embedding.".green());

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(source.local_path.join("tokenizer.json"))
        .map_err(|e| {
            ferrum_types::FerrumError::model(format!("Failed to load tokenizer: {}", e))
        })?;

    // Process input text
    let texts: Vec<String> = if let Some(text) = cmd.text {
        vec![text]
    } else {
        eprintln!(
            "{}",
            "Reading text from stdin (one text per line, Ctrl+D to finish):".dimmed()
        );
        let stdin = io::stdin();
        stdin.lock().lines().filter_map(|l| l.ok()).collect()
    };

    if texts.is_empty() {
        eprintln!("{}", "No text provided.".yellow());
        return Ok(());
    }

    // Generate embeddings for each text
    let mut all_embeddings = Vec::new();

    for text in &texts {
        // Tokenize
        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| ferrum_types::FerrumError::model(format!("Tokenization failed: {}", e)))?;

        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Get embeddings
        let embedding_tensor = executor.get_embeddings(&token_ids)?;

        // Convert to vec
        let mut embedding = embedding_tensor
            .flatten_all()
            .map_err(|e| ferrum_types::FerrumError::model(format!("Flatten failed: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| ferrum_types::FerrumError::model(format!("to_vec1 failed: {}", e)))?;

        // Normalize if requested
        if cmd.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut embedding {
                    *v /= norm;
                }
            }
        }

        all_embeddings.push((text.clone(), embedding));
    }

    // Output embeddings
    match cmd.format.as_str() {
        "json" => {
            let output: Vec<serde_json::Value> = all_embeddings
                .iter()
                .map(|(text, emb)| {
                    serde_json::json!({
                        "text": text,
                        "embedding": emb,
                        "dimensions": emb.len()
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        "csv" => {
            if let Some((_, first_emb)) = all_embeddings.first() {
                let header: Vec<String> =
                    (0..first_emb.len()).map(|i| format!("dim_{}", i)).collect();
                println!("text,{}", header.join(","));
            }
            for (text, emb) in &all_embeddings {
                let emb_str: Vec<String> = emb.iter().map(|v| format!("{:.6}", v)).collect();
                println!("\"{}\",{}", text.replace("\"", "\\\""), emb_str.join(","));
            }
        }
        "raw" => {
            for (text, emb) in &all_embeddings {
                eprintln!("{}: {} dimensions", text.dimmed(), emb.len());
                let preview: Vec<String> =
                    emb.iter().take(5).map(|v| format!("{:.4}", v)).collect();
                println!("[{}, ...]", preview.join(", "));
            }
        }
        _ => {
            eprintln!("{}", format!("Unknown format: {}", cmd.format).red());
        }
    }

    Ok(())
}

fn find_cached_model(cache_dir: &PathBuf, model_id: &str) -> Option<ResolvedModelSource> {
    // HuggingFace cache structure: hub/models--Org--ModelName/snapshots/<hash>/
    let hub_dir = cache_dir.join("hub");
    let model_dir_name = format!("models--{}", model_id.replace("/", "--"));
    let model_dir = hub_dir.join(&model_dir_name);

    if model_dir.exists() {
        // Find the latest snapshot
        let snapshots_dir = model_dir.join("snapshots");
        if snapshots_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let snapshot_path = entry.path();
                    if snapshot_path.is_dir() && snapshot_path.join("config.json").exists() {
                        let format = detect_format(&snapshot_path);
                        if format != ModelFormat::Unknown {
                            return Some(ResolvedModelSource {
                                original: model_id.to_string(),
                                local_path: snapshot_path,
                                format,
                                from_cache: true,
                            });
                        }
                    }
                }
            }
        }
    }

    // Also check direct path (for models downloaded to custom locations)
    let direct = cache_dir.join(model_id);
    if direct.exists() && direct.join("config.json").exists() {
        let format = detect_format(&direct);
        if format != ModelFormat::Unknown {
            return Some(ResolvedModelSource {
                original: model_id.to_string(),
                local_path: direct,
                format,
                from_cache: true,
            });
        }
    }

    None
}

fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}

fn detect_format(path: &PathBuf) -> ModelFormat {
    if path.join("model.safetensors").exists() {
        ModelFormat::SafeTensors
    } else if std::fs::read_dir(path)
        .map(|d| {
            d.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext == "safetensors")
            })
        })
        .unwrap_or(false)
    {
        ModelFormat::SafeTensors
    } else if path.join("pytorch_model.bin").exists() {
        ModelFormat::PyTorchBin
    } else {
        ModelFormat::Unknown
    }
}
