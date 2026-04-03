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

/// Generate embeddings using BERT or CLIP models
#[derive(Args, Debug)]
pub struct EmbedCommand {
    /// Model name (e.g., google-bert/bert-base-chinese, OFA-Sys/chinese-clip-vit-base-patch16)
    #[arg(required = true)]
    pub model: String,

    /// Text to embed (if not provided, reads from stdin)
    #[arg(short, long)]
    pub text: Option<String>,

    /// Image path to embed (CLIP models only)
    #[arg(short, long)]
    pub image: Option<String>,

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

    // Load model definition to detect architecture
    let mut config_manager = ConfigManager::new();
    let model_def = config_manager.load_from_path(&source.local_path).await?;

    let device = CandleDevice::Cpu;
    let is_clip = model_def.architecture == ferrum_models::Architecture::Clip;

    let mut all_embeddings: Vec<(String, Vec<f32>)> = Vec::new();

    if is_clip {
        // CLIP path: supports both text and image
        let executor = ferrum_models::ClipModelExecutor::from_path(
            &model_path,
            device.clone(),
            candle_core::DType::F32,
        )?;
        eprintln!("{}", "CLIP model loaded.".green());

        if let Some(ref image_path) = cmd.image {
            let embedding_tensor = executor.embed_image_path(image_path)?;
            let embedding = tensor_to_vec(&embedding_tensor, cmd.normalize)?;
            all_embeddings.push((format!("[image] {image_path}"), embedding));
        }

        let texts = collect_texts(&cmd)?;
        if !texts.is_empty() {
            let tokenizer =
                tokenizers::Tokenizer::from_file(source.local_path.join("tokenizer.json"))
                    .map_err(|e| {
                        ferrum_types::FerrumError::model(format!("Load tokenizer: {e}"))
                    })?;

            for text in &texts {
                let encoding = tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| ferrum_types::FerrumError::model(format!("Tokenize: {e}")))?;
                let embedding_tensor = executor.embed_text(encoding.get_ids())?;
                let embedding = tensor_to_vec(&embedding_tensor, cmd.normalize)?;
                all_embeddings.push((text.clone(), embedding));
            }
        }

        if all_embeddings.is_empty() {
            eprintln!("{}", "No input provided. Use --text or --image.".yellow());
            return Ok(());
        }
    } else {
        // BERT path (existing)
        let executor = BertModelExecutor::from_path(&model_path, &model_def, device).await?;
        eprintln!("{}", "BERT model loaded.".green());

        let tokenizer = tokenizers::Tokenizer::from_file(source.local_path.join("tokenizer.json"))
            .map_err(|e| ferrum_types::FerrumError::model(format!("Load tokenizer: {e}")))?;

        let texts = collect_texts(&cmd)?;
        if texts.is_empty() {
            eprintln!("{}", "No text provided.".yellow());
            return Ok(());
        }

        for text in &texts {
            let encoding = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| ferrum_types::FerrumError::model(format!("Tokenize: {e}")))?;
            let embedding_tensor = executor.get_embeddings(encoding.get_ids())?;
            let embedding = tensor_to_vec(&embedding_tensor, cmd.normalize)?;
            all_embeddings.push((text.clone(), embedding));
        }
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

fn collect_texts(cmd: &EmbedCommand) -> Result<Vec<String>> {
    if let Some(ref text) = cmd.text {
        Ok(vec![text.clone()])
    } else if cmd.image.is_some() {
        // Image-only mode, no text needed
        Ok(vec![])
    } else {
        eprintln!(
            "{}",
            "Reading text from stdin (one per line, Ctrl+D to finish):".dimmed()
        );
        let stdin = io::stdin();
        Ok(stdin.lock().lines().filter_map(|l| l.ok()).collect())
    }
}

fn tensor_to_vec(tensor: &candle_core::Tensor, normalize: bool) -> Result<Vec<f32>> {
    let mut embedding = tensor
        .flatten_all()
        .map_err(|e| ferrum_types::FerrumError::model(format!("Flatten: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| ferrum_types::FerrumError::model(format!("to_vec1: {e}")))?;

    if normalize {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
    }
    Ok(embedding)
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
