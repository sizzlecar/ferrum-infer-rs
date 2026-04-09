//! Mel spectrogram computation matching Python whisper exactly.
//!
//! Uses rustfft for STFT (matching torch.stft with center=True),
//! pre-computed mel filterbank, and identical normalization.

use rustfft::{num_complex::Complex, FftPlanner};

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;

/// Compute log-mel spectrogram matching Python whisper.audio.log_mel_spectrogram.
///
/// - `pcm`: audio samples (f32, 16kHz mono)
/// - `n_mels`: 80 or 128
/// - `mel_filters`: pre-loaded filter bank, shape [n_mels, N_FFT/2 + 1], row-major
///
/// Returns flat Vec<f32> in [n_mels, n_frames] layout (row-major per mel bin).
pub fn log_mel_spectrogram(pcm: &[f32], n_mels: usize, mel_filters: &[f32]) -> Vec<f32> {
    let n_fft_half = N_FFT / 2 + 1; // 201

    // Step 1: Reflect-pad (center=True in torch.stft)
    let padded = reflect_pad(pcm, N_FFT / 2);

    // Step 2: STFT → magnitudes squared
    let magnitudes = stft_magnitudes_squared(&padded);
    // magnitudes: [n_fft_half, n_frames_raw]
    let n_frames_raw = magnitudes.len() / n_fft_half;
    // Drop last frame (matching Python: stft[..., :-1])
    let n_frames = n_frames_raw - 1;

    // Step 3: mel_filters[n_mels, n_fft_half] @ magnitudes[n_fft_half, n_frames]
    let mut mel_spec = vec![0f32; n_mels * n_frames];
    for m in 0..n_mels {
        for t in 0..n_frames {
            let mut sum = 0f32;
            for f in 0..n_fft_half {
                sum += mel_filters[m * n_fft_half + f] * magnitudes[f * n_frames_raw + t];
            }
            mel_spec[m * n_frames + t] = sum;
        }
    }

    // Step 4: log10(clamp(x, 1e-10))
    for v in &mut mel_spec {
        *v = v.max(1e-10).log10();
    }

    // Step 5: max(x, global_max - 8.0)
    let global_max = mel_spec
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let clamp_min = global_max - 8.0;
    for v in &mut mel_spec {
        *v = v.max(clamp_min);
    }

    // Step 6: (x + 4.0) / 4.0
    for v in &mut mel_spec {
        *v = (*v + 4.0) / 4.0;
    }

    mel_spec
}

/// Reflect-pad signal on both sides (matching torch center=True).
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // Left reflect: signal[pad], signal[pad-1], ..., signal[1]
    for i in (1..=pad).rev() {
        out.push(signal[i.min(n - 1)]);
    }
    out.extend_from_slice(signal);
    // Right reflect: signal[n-2], signal[n-3], ..., signal[n-1-pad]
    for i in 1..=pad {
        out.push(signal[(n - 1).saturating_sub(i)]);
    }
    out
}

/// STFT with Hann window, returning magnitudes squared.
/// Returns [n_fft_half, n_frames] in column-major (frequency × time).
fn stft_magnitudes_squared(padded: &[f32]) -> Vec<f32> {
    let n_fft_half = N_FFT / 2 + 1;
    let n_frames = (padded.len() - N_FFT) / HOP_LENGTH + 1;

    // Hann window
    let hann: Vec<f32> = (0..N_FFT)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos())
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // Output: [n_fft_half, n_frames] column-major
    let mut magnitudes = vec![0f32; n_fft_half * n_frames];

    let mut buffer = vec![Complex::new(0f32, 0f32); N_FFT];

    for t in 0..n_frames {
        let offset = t * HOP_LENGTH;
        // Apply window
        for i in 0..N_FFT {
            buffer[i] = Complex::new(padded[offset + i] * hann[i], 0.0);
        }
        // FFT
        fft.process(&mut buffer);
        // Magnitude squared for first n_fft_half bins
        for f in 0..n_fft_half {
            magnitudes[f * n_frames + t] = buffer[f].norm_sqr();
        }
    }

    magnitudes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflect_pad() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&signal, 2);
        // Left: signal[2], signal[1] = 3.0, 2.0
        // Right: signal[3], signal[2] = 4.0, 3.0
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_mel_shape() {
        // 1 second of silence at 16kHz
        let pcm = vec![0.0f32; 16000];
        let filters = vec![0.0f32; 80 * 201];
        let mel = log_mel_spectrogram(&pcm, 80, &filters);
        let n_frames = mel.len() / 80;
        assert_eq!(n_frames, 100); // 16000 / 160 = 100 frames
    }
}
