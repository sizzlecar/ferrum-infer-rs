//! Audio preprocessing for Whisper ASR.
//!
//! Load audio files → decode → resample to 16kHz mono → f32 PCM samples.
//! Supports WAV natively; M4A/MP3/FLAC/OGG via ffmpeg auto-conversion.

use ferrum_types::{FerrumError, Result};
use std::path::Path;

/// Whisper processes 30-second chunks. At 16kHz → 480,000 samples.
pub const CHUNK_SAMPLES: usize = 16000 * 30;

/// Load audio file and return 16kHz mono f32 PCM samples.
///
/// If the file is not WAV, tries ffmpeg conversion automatically.
pub fn load_audio(path: &str) -> Result<Vec<f32>> {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // WAV: direct load
    if ext == "wav" {
        return load_wav_file(path);
    }

    // Non-WAV: convert via ffmpeg
    convert_with_ffmpeg(path)
}

/// Load audio file and return mono f32 PCM samples at a configurable sample rate.
///
/// Similar to `load_audio` but resamples to `target_rate` instead of 16kHz.
/// Useful for TTS speaker encoder which expects 24kHz input.
pub fn load_audio_at_rate(path: &str, target_rate: u32) -> Result<Vec<f32>> {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // WAV: decode then resample to target_rate
    if ext == "wav" {
        return load_wav_file_at_rate(path, target_rate);
    }

    // Non-WAV: convert via ffmpeg to target_rate
    convert_with_ffmpeg_at_rate(path, target_rate)
}

/// Load audio from raw bytes.
/// Tries WAV first; if that fails and bytes look non-WAV, tries ffmpeg.
pub fn load_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    // Try WAV first
    match load_wav_bytes(data) {
        Ok(pcm) => return Ok(pcm),
        Err(_) => {}
    }

    // Fallback: write to temp file and convert via ffmpeg
    let tmp = std::env::temp_dir().join("ferrum_audio_tmp");
    std::fs::write(&tmp, data).map_err(|e| FerrumError::model(format!("write temp audio: {e}")))?;
    let result = convert_with_ffmpeg(tmp.to_str().unwrap_or(""));
    let _ = std::fs::remove_file(&tmp);
    result
}

/// Split PCM samples into 30-second chunks for Whisper processing.
pub fn chunk_pcm(pcm: &[f32]) -> Vec<&[f32]> {
    if pcm.len() <= CHUNK_SAMPLES {
        return vec![pcm];
    }
    pcm.chunks(CHUNK_SAMPLES).collect()
}

// ── WAV loading ─────────────────────────────────────────────────────────

fn load_wav_file(path: &str) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| FerrumError::model(format!("open audio {path}: {e}")))?;
    decode_wav(reader)
}

fn load_wav_bytes(data: &[u8]) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(data);
    let reader =
        hound::WavReader::new(cursor).map_err(|e| FerrumError::model(format!("decode: {e}")))?;
    decode_wav(reader)
}

fn decode_wav<R: std::io::Read>(reader: hound::WavReader<R>) -> Result<Vec<f32>> {
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Mix to mono
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Resample to 16kHz if needed
    if (sample_rate - 16000.0).abs() < 1.0 {
        Ok(mono)
    } else {
        Ok(resample(&mono, sample_rate, 16000.0))
    }
}

// ── ffmpeg conversion ───────────────────────────────────────────────────

fn convert_with_ffmpeg(input_path: &str) -> Result<Vec<f32>> {
    let output = std::env::temp_dir().join("ferrum_ffmpeg_out.wav");
    let output_str = output.to_string_lossy().to_string();

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            "-f",
            "wav",
            &output_str,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    match status {
        Ok(s) if s.success() => {
            let result = load_wav_file(&output_str);
            let _ = std::fs::remove_file(&output);
            result
        }
        Ok(s) => Err(FerrumError::model(format!(
            "ffmpeg exited with code {}. Is the audio file valid?",
            s.code().unwrap_or(-1)
        ))),
        Err(_) => Err(FerrumError::model(
            "ffmpeg not found. Install ffmpeg to process non-WAV audio (brew install ffmpeg)",
        )),
    }
}

// ── WAV loading at configurable rate ─────────────────────────────────────

fn load_wav_file_at_rate(path: &str, target_rate: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| FerrumError::model(format!("open audio {path}: {e}")))?;
    decode_wav_at_rate(reader, target_rate)
}

fn decode_wav_at_rate<R: std::io::Read>(
    reader: hound::WavReader<R>,
    target_rate: u32,
) -> Result<Vec<f32>> {
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Mix to mono
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Resample to target_rate if needed
    let target = target_rate as f64;
    if (sample_rate - target).abs() < 1.0 {
        Ok(mono)
    } else {
        Ok(resample(&mono, sample_rate, target))
    }
}

fn convert_with_ffmpeg_at_rate(input_path: &str, target_rate: u32) -> Result<Vec<f32>> {
    let output = std::env::temp_dir().join("ferrum_ffmpeg_out_rate.wav");
    let output_str = output.to_string_lossy().to_string();
    let rate_str = target_rate.to_string();

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input_path,
            "-ar",
            &rate_str,
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            "-f",
            "wav",
            &output_str,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    match status {
        Ok(s) if s.success() => {
            let result = load_wav_file_at_rate(&output_str, target_rate);
            let _ = std::fs::remove_file(&output);
            result
        }
        Ok(s) => Err(FerrumError::model(format!(
            "ffmpeg exited with code {}. Is the audio file valid?",
            s.code().unwrap_or(-1)
        ))),
        Err(_) => Err(FerrumError::model(
            "ffmpeg not found. Install ffmpeg to process non-WAV audio (brew install ffmpeg)",
        )),
    }
}

// ── Resampler ───────────────────────────────────────────────────────────

pub(crate) fn resample(input: &[f32], from_rate: f64, to_rate: f64) -> Vec<f32> {
    use rubato::{
        audioadapter::Adapter, Async, FixedAsync, Resampler as RubatoResampler,
        SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let ratio = to_rate / from_rate;
    let chunk_size = 1024;

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler =
        Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, 1, FixedAsync::Input)
            .expect("resample init");

    let mut output = Vec::new();
    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let chunk = &input[pos..end];
        let data: Vec<f32> = if chunk.len() < chunk_size {
            let mut p = chunk.to_vec();
            p.resize(chunk_size, 0.0);
            p
        } else {
            chunk.to_vec()
        };

        let input_vecs = vec![data];
        let input_adapter =
            audioadapter_buffers::direct::SequentialSliceOfVecs::new(&input_vecs, 1, chunk_size)
                .expect("input adapter");
        let result = resampler
            .process(&input_adapter, 0, None)
            .expect("resample");
        let frames = result.frames();
        for i in 0..frames {
            output.push(result.read_sample(0, i).unwrap_or(0.0));
        }
        pos += chunk_size;
    }
    output
}
