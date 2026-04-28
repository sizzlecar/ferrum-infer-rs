//! TTS command - Text-to-speech using Qwen3-TTS models

use crate::config::CliConfig;
use candle_core::{DType, Device as CandleDevice};
use clap::Args;
use colored::Colorize;
use ferrum_models::source::{ModelFormat, ResolvedModelSource};
use ferrum_models::HfDownloader;
use ferrum_types::Result;
use std::path::PathBuf;

/// Synthesize speech from text using Qwen3-TTS models
#[derive(Args, Debug)]
pub struct TtsCommand {
    /// TTS model name (e.g., qwen3-tts, Qwen/Qwen3-TTS-12Hz-0.6B-Base)
    #[arg(required = true)]
    pub model: String,

    /// Text to synthesize
    #[arg(required = true)]
    pub text: String,

    /// Output WAV file path
    #[arg(short, long, default_value = "/tmp/ferrum_tts_output.wav")]
    pub output: String,

    /// Language (e.g., chinese, english, auto)
    #[arg(short, long, default_value = "auto")]
    pub language: String,

    /// Backend: auto, cpu, metal (default: auto)
    #[arg(short, long, default_value = "auto")]
    pub backend: String,

    /// Reference audio for voice cloning (WAV/M4A/MP3)
    #[arg(long)]
    pub ref_audio: Option<String>,

    /// Reference audio transcript (required for ICL voice cloning)
    #[arg(long)]
    pub ref_text: Option<String>,

    /// Enable streaming mode (generate audio in chunks)
    #[arg(long)]
    pub streaming: bool,

    /// Frames per streaming chunk (default: 10, ~800ms per chunk)
    #[arg(long, default_value = "10")]
    pub chunk_frames: usize,
}

pub async fn execute(cmd: TtsCommand, config: CliConfig) -> Result<()> {
    let model_id = resolve_tts_alias(&cmd.model);
    let cache_dir = get_hf_cache_dir(&config);

    eprintln!("{} {}", "Model:".dimmed(), model_id.cyan());

    // Find or download model
    let source = match find_cached_model(&cache_dir, &model_id) {
        Some(source) => source,
        None => {
            eprintln!(
                "{} Model '{}' not found locally, downloading...",
                ">>>".cyan(),
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

    // Verify architecture — Qwen3-TTS config.json has "talker_config" key
    let config_path = source.local_path.join("config.json");
    let config_data = std::fs::read_to_string(&config_path)
        .map_err(|e| ferrum_types::FerrumError::model(format!("read config.json: {e}")))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_data)
        .map_err(|e| ferrum_types::FerrumError::model(format!("parse config.json: {e}")))?;

    if config_json.get("talker_config").is_none() {
        return Err(ferrum_types::FerrumError::model(format!(
            "'{}' does not appear to be a Qwen3-TTS model (missing talker_config)",
            model_id
        )));
    }

    let candle_device = select_candle_device(&cmd.backend);
    eprintln!("{} {:?}", "Device:".dimmed(), &candle_device);
    eprintln!("{}", "Loading TTS model...".dimmed());

    let mut executor = ferrum_models::TtsModelExecutor::from_path(
        &source.local_path.to_string_lossy(),
        candle_device,
        DType::F32,
    )?;
    eprintln!("{}", "Model loaded.".green());

    // Synthesize
    eprintln!(
        "{} \"{}\"",
        "Text:".dimmed(),
        if cmd.text.chars().count() > 80 {
            let truncated: String = cmd.text.chars().take(77).collect();
            format!("{truncated}...")
        } else {
            cmd.text.clone()
        }
        .cyan()
    );

    let start = std::time::Instant::now();
    let sample_rate = executor.sample_rate();

    if cmd.streaming && cmd.ref_audio.is_none() {
        // Streaming mode: generate and save chunks incrementally
        eprintln!("{}", "Streaming mode enabled".yellow());
        let mut all_samples = Vec::new();
        let sr = sample_rate;
        let t0 = start;
        let chunks = executor.synthesize_streaming(
            &cmd.text,
            &cmd.language,
            cmd.chunk_frames,
            |idx, chunk| {
                let chunk_dur = chunk.len() as f64 / sr as f64;
                eprintln!(
                    "  {} chunk {} — {:.2}s audio (at {:.1}s)",
                    "▶".green(),
                    idx,
                    chunk_dur,
                    t0.elapsed().as_secs_f64(),
                );
            },
        )?;
        let elapsed = start.elapsed();
        for chunk in &chunks {
            all_samples.extend_from_slice(chunk);
        }

        let duration_secs = all_samples.len() as f64 / sample_rate as f64;
        save_wav(&cmd.output, &all_samples, sample_rate as u32)?;

        eprintln!("\n{} {}", "Output:".dimmed(), cmd.output.green());
        eprintln!(
            "{} {:.2}s audio, {:.2}s elapsed (RTF={:.2}x), {} chunks",
            "Stats:".dimmed(),
            duration_secs,
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / duration_secs.max(0.001),
            chunks.len(),
        );
    } else {
        // Batch mode
        let samples = if let Some(ref_audio) = &cmd.ref_audio {
            let ref_text = cmd.ref_text.as_deref().ok_or_else(|| {
                ferrum_types::FerrumError::model("--ref-text required for voice cloning")
            })?;
            eprintln!("{} {}", "Ref audio:".dimmed(), ref_audio.cyan());
            executor.synthesize_voice_clone(&cmd.text, &cmd.language, ref_audio, ref_text)?
        } else {
            executor.synthesize(&cmd.text, &cmd.language)?
        };
        let elapsed = start.elapsed();
        let duration_secs = samples.len() as f64 / sample_rate as f64;
        save_wav(&cmd.output, &samples, sample_rate as u32)?;

        eprintln!("\n{} {}", "Output:".dimmed(), cmd.output.green());
        eprintln!(
            "{} {:.2}s audio, {:.2}s elapsed (RTF={:.2}x)",
            "Stats:".dimmed(),
            duration_secs,
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / duration_secs.max(0.001),
        );
    }

    Ok(())
}

/// Save PCM samples as a 16-bit mono WAV file (no external dependency).
fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    use std::io::Write;

    let num_samples = samples.len() as u32;
    let bytes_per_sample: u16 = 2; // 16-bit
    let channels: u16 = 1;
    let data_size = num_samples * bytes_per_sample as u32;
    let file_size = 36 + data_size;

    let mut buf: Vec<u8> = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * bytes_per_sample;
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&(bytes_per_sample * 8).to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&value.to_le_bytes());
    }

    let mut file = std::fs::File::create(path)
        .map_err(|e| ferrum_types::FerrumError::model(format!("create WAV file: {e}")))?;
    file.write_all(&buf)
        .map_err(|e| ferrum_types::FerrumError::model(format!("write WAV file: {e}")))?;

    Ok(())
}

fn resolve_tts_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "qwen3-tts" | "qwen3:tts" | "qwen3-tts:0.6b" => "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
        "qwen3-tts:instruct" | "qwen3-tts-instruct" => {
            "Qwen/Qwen3-TTS-12Hz-0.6B-Instruct".to_string()
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

fn select_candle_device(backend: &str) -> CandleDevice {
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
