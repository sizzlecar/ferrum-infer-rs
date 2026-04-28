//! Transcribe command - Speech-to-text using Whisper models

use crate::config::CliConfig;
use candle_core::{DType, Device as CandleDevice};
use clap::Args;
use colored::Colorize;
use ferrum_models::source::{ModelFormat, ResolvedModelSource};
use ferrum_models::{ConfigManager, HfDownloader, WhisperModelExecutor};
use ferrum_types::Result;
use std::path::PathBuf;

/// Transcribe audio files using Whisper models
#[derive(Args, Debug)]
pub struct TranscribeCommand {
    /// Whisper model name (e.g., whisper-tiny, openai/whisper-base)
    #[arg(required = true)]
    pub model: String,

    /// Audio file path (WAV format)
    #[arg(required = true)]
    pub audio: String,

    /// Language hint (e.g., en, zh, ja)
    #[arg(short, long)]
    pub language: Option<String>,

    /// Backend: auto, cpu, metal (default: auto)
    #[arg(short, long, default_value = "auto")]
    pub backend: String,
}

pub async fn execute(cmd: TranscribeCommand, config: CliConfig) -> Result<()> {
    let model_id = resolve_whisper_alias(&cmd.model);
    let cache_dir = get_hf_cache_dir(&config);

    eprintln!("{} {}", "Model:".dimmed(), model_id.cyan());

    // Find or download model
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

    // Verify architecture
    let mut config_manager = ConfigManager::new();
    let model_def = config_manager.load_from_path(&source.local_path).await?;
    if model_def.architecture != ferrum_models::Architecture::Whisper {
        return Err(ferrum_types::FerrumError::model(format!(
            "'{}' is not a Whisper model (detected: {:?})",
            model_id, model_def.architecture
        )));
    }

    let candle_device = select_candle_device(&cmd.backend);
    eprintln!("{} {:?}", "Device:".dimmed(), &candle_device);
    eprintln!("{}", "Loading Whisper model...".dimmed());
    let executor = WhisperModelExecutor::from_path(
        &source.local_path.to_string_lossy(),
        candle_device,
        DType::F32,
    )?;
    eprintln!("{}", "Model loaded.".green());

    // Transcribe
    eprintln!("{} {}", "Audio:".dimmed(), cmd.audio.cyan());
    let start = std::time::Instant::now();
    let text = executor.transcribe_file(&cmd.audio, cmd.language.as_deref())?;
    let elapsed = start.elapsed();

    println!("{}", text);
    eprintln!("\n{} {:.2}s", "Time:".dimmed(), elapsed.as_secs_f64());

    Ok(())
}

fn resolve_whisper_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "whisper-tiny" | "whisper:tiny" => "openai/whisper-tiny".to_string(),
        "whisper-base" | "whisper:base" => "openai/whisper-base".to_string(),
        "whisper-small" | "whisper:small" => "openai/whisper-small".to_string(),
        "whisper-medium" | "whisper:medium" => "openai/whisper-medium".to_string(),
        "whisper-large-v3" | "whisper:large-v3" => "openai/whisper-large-v3".to_string(),
        "whisper-turbo" | "whisper:turbo" | "whisper-large-v3-turbo" => {
            "openai/whisper-large-v3-turbo".to_string()
        }
        _ => name.to_string(),
    }
}

fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}

fn find_cached_model(cache_dir: &PathBuf, model_id: &str) -> Option<ResolvedModelSource> {
    let hub_dir = cache_dir.join("hub");
    let model_dir_name = format!("models--{}", model_id.replace("/", "--"));
    let model_dir = hub_dir.join(&model_dir_name);

    if model_dir.exists() {
        let snapshots_dir = model_dir.join("snapshots");
        if snapshots_dir.exists() {
            // Try refs/main first
            let ref_main = model_dir.join("refs").join("main");
            if let Ok(rev) = std::fs::read_to_string(&ref_main) {
                let rev = rev.trim();
                if !rev.is_empty() {
                    let snapshot = snapshots_dir.join(rev);
                    if snapshot.exists() {
                        let format = detect_format(&snapshot);
                        if format != ModelFormat::Unknown {
                            return Some(ResolvedModelSource {
                                original: model_id.to_string(),
                                local_path: snapshot,
                                format,
                                from_cache: true,
                            });
                        }
                    }
                }
            }
            // Fallback: first snapshot
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() && path.join("config.json").exists() {
                        let format = detect_format(&path);
                        if format != ModelFormat::Unknown {
                            return Some(ResolvedModelSource {
                                original: model_id.to_string(),
                                local_path: path,
                                format,
                                from_cache: true,
                            });
                        }
                    }
                }
            }
        }
    }

    None
}

pub fn select_candle_device(backend: &str) -> CandleDevice {
    match backend.to_lowercase().as_str() {
        "cpu" => CandleDevice::Cpu,
        "metal" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return CandleDevice::new_metal(0).unwrap_or(CandleDevice::Cpu);
            }
            #[allow(unreachable_code)]
            {
                eprintln!("Metal not available, falling back to CPU");
                CandleDevice::Cpu
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                return CandleDevice::new_cuda(0).unwrap_or_else(|e| {
                    eprintln!("CUDA unavailable ({e}), falling back to CPU");
                    CandleDevice::Cpu
                });
            }
            #[allow(unreachable_code)]
            {
                eprintln!("CUDA feature not compiled, falling back to CPU");
                CandleDevice::Cpu
            }
        }
        "auto" | _ => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(d) = CandleDevice::new_cuda(0) {
                    return d;
                }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return CandleDevice::new_metal(0).unwrap_or(CandleDevice::Cpu);
            }
            #[allow(unreachable_code)]
            CandleDevice::Cpu
        }
    }
}

fn detect_format(path: &PathBuf) -> ModelFormat {
    if path.join("model.safetensors").exists() {
        ModelFormat::SafeTensors
    } else if std::fs::read_dir(path)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
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
