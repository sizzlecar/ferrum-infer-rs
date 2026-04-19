//! Audio processing tests — WAV read/write, resample, mel spectrogram.
//! Runs on CPU, no model needed.

use std::io::Write;

/// Generate a sine wave WAV file in memory.
fn gen_sine_wav(sample_rate: u32, freq: f32, duration_s: f32) -> Vec<u8> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let mut buf = Vec::new();
    // WAV header
    let data_size = (n * 2) as u32;
    let file_size = 44 + data_size;
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(file_size - 8).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * freq * t).sin();
        let i16_val = (sample * 32000.0) as i16;
        buf.extend_from_slice(&i16_val.to_le_bytes());
    }
    buf
}

#[test]
fn test_wav_roundtrip() {
    let wav_data = gen_sine_wav(16000, 440.0, 0.5);
    let tmp = std::env::temp_dir().join("ferrum_test_roundtrip.wav");
    std::fs::write(&tmp, &wav_data).unwrap();

    let pcm =
        ferrum_models::audio_processor::load_audio_at_rate(tmp.to_str().unwrap(), 16000).unwrap();

    assert!(pcm.len() > 7000, "too few samples: {}", pcm.len());
    assert!(pcm.len() < 9000, "too many samples: {}", pcm.len());
    assert!(pcm.iter().all(|x| x.is_finite()));
    // Sine wave should have energy
    let rms: f32 = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
    assert!(rms > 0.3, "RMS too low: {rms}");
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_resample_44100_to_16000() {
    let wav_data = gen_sine_wav(44100, 440.0, 1.0);
    let tmp = std::env::temp_dir().join("ferrum_test_resample.wav");
    std::fs::write(&tmp, &wav_data).unwrap();

    let pcm =
        ferrum_models::audio_processor::load_audio_at_rate(tmp.to_str().unwrap(), 16000).unwrap();

    // 44100 → 16000: ~16000 samples for 1s
    assert!(pcm.len() > 15000, "too few: {}", pcm.len());
    assert!(pcm.len() < 17000, "too many: {}", pcm.len());
    assert!(pcm.iter().all(|x| x.is_finite()));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_resample_44100_to_24000() {
    let wav_data = gen_sine_wav(44100, 440.0, 1.0);
    let tmp = std::env::temp_dir().join("ferrum_test_resample_24k.wav");
    std::fs::write(&tmp, &wav_data).unwrap();

    let pcm =
        ferrum_models::audio_processor::load_audio_at_rate(tmp.to_str().unwrap(), 24000).unwrap();

    // 44100 → 24000: ~24000 samples for 1s
    assert!(pcm.len() > 23000, "too few: {}", pcm.len());
    assert!(pcm.len() < 25000, "too many: {}", pcm.len());
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_no_resample_when_rate_matches() {
    let wav_data = gen_sine_wav(24000, 440.0, 0.5);
    let tmp = std::env::temp_dir().join("ferrum_test_noresample.wav");
    std::fs::write(&tmp, &wav_data).unwrap();

    let pcm =
        ferrum_models::audio_processor::load_audio_at_rate(tmp.to_str().unwrap(), 24000).unwrap();

    assert_eq!(pcm.len(), 12000); // exact: 24000 * 0.5
    std::fs::remove_file(&tmp).ok();
}
