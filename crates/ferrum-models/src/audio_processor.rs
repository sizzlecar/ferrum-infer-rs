//! Audio preprocessing for Whisper ASR.
//!
//! Load audio files → decode → resample to 16kHz mono → f32 PCM samples.

use ferrum_types::{FerrumError, Result};

/// Load audio file and return 16kHz mono f32 PCM samples.
///
/// Supports WAV (via hound). For other formats, convert to WAV first.
pub fn load_audio(path: &str) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| FerrumError::model(format!("open audio {path}: {e}")))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    // Read all samples as f32
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

/// Load audio from raw bytes (WAV format).
pub fn load_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(data);
    let reader = hound::WavReader::new(cursor)
        .map_err(|e| FerrumError::model(format!("decode audio: {e}")))?;

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

    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if (sample_rate - 16000.0).abs() < 1.0 {
        Ok(mono)
    } else {
        Ok(resample(&mono, sample_rate, 16000.0))
    }
}

/// Simple linear interpolation resampler.
fn resample(input: &[f32], from_rate: f64, to_rate: f64) -> Vec<f32> {
    let ratio = from_rate / to_rate;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        let sample = if idx + 1 < input.len() {
            input[idx] * (1.0 - frac) + input[idx + 1] * frac
        } else if idx < input.len() {
            input[idx]
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}
