//! ECAPA-TDNN Speaker Encoder for Qwen3-TTS voice cloning.
//!
//! Takes a mel spectrogram [1, T, 128] and outputs a 1024-dim speaker embedding.
//! Used for zero-shot voice cloning by conditioning the TTS Talker on reference audio.
//!
//! Architecture: TDNN → 3x SE-Res2Net blocks → MFA → ASP → FC → [1024]
//!
//! Weight prefix in safetensors: `speaker_encoder.`

use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use ferrum_types::{FerrumError, Result};
use tracing::info;

// ── Mel filterbank for speaker encoder (128 x 513, row-major f32le) ──────

const MEL_FILTERS: &[u8] = include_bytes!("mel_filters_spkenc.bin");

// ── Reflect-pad Conv1d (Metal-compatible) ─────────────────────────────────
//
// candle's Conv1d only supports zero-padding. For "same" convolution with
// reflect padding mode, we manually reflect-pad the input tensor and run
// Conv1d with padding=0.

/// Reflect-pad a 3D tensor [B, C, T] along the time (last) dimension.
fn reflect_pad_1d(x: &Tensor, pad_left: usize, pad_right: usize) -> candle_core::Result<Tensor> {
    if pad_left == 0 && pad_right == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(2)?;
    let mut parts: Vec<Tensor> = Vec::new();

    let x = x.contiguous()?;

    // Left reflection: indices pad_left, pad_left-1, ..., 1
    if pad_left > 0 {
        let mut left_indices = Vec::with_capacity(pad_left);
        for i in (1..=pad_left).rev() {
            left_indices.push(i.min(t - 1) as u32);
        }
        let idx = Tensor::new(left_indices, x.device())?;
        parts.push(x.index_select(&idx, 2)?);
    }

    // Original
    parts.push(x.clone());

    // Right reflection: indices t-2, t-3, ..., t-1-pad_right
    if pad_right > 0 {
        let mut right_indices = Vec::with_capacity(pad_right);
        for i in 1..=pad_right {
            right_indices.push((t - 1).saturating_sub(i) as u32);
        }
        let idx = Tensor::new(right_indices, x.device())?;
        parts.push(x.index_select(&idx, 2)?);
    }

    Tensor::cat(&parts, 2)
}

/// Conv1d with reflect padding ("same" mode with padding_mode="reflect").
struct ReflectConv1d {
    conv: Conv1d,
    pad_left: usize,
    pad_right: usize,
}

impl ReflectConv1d {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let effective_kernel = dilation * (kernel_size - 1) + 1;
        let total_pad = effective_kernel - 1;
        let pad_left = total_pad / 2;
        let pad_right = total_pad - pad_left;

        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let w = vb.get((out_ch, in_ch / groups, kernel_size), "weight")?;
        let b = vb.get(out_ch, "bias").ok();
        Ok(Self {
            conv: Conv1d::new(w, b, cfg),
            pad_left,
            pad_right,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = reflect_pad_1d(x, self.pad_left, self.pad_right)?;
        self.conv.forward(&x)
    }
}

// ── TimeDelayNetBlock (TDNN) ──────────────────────────────────────────────
//
// Conv1d(in→out, kernel, dilation, padding="same", padding_mode="reflect") + ReLU

struct TimeDelayNetBlock {
    conv: ReflectConv1d,
}

impl TimeDelayNetBlock {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let conv = ReflectConv1d::load(in_ch, out_ch, kernel_size, dilation, 1, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.conv.forward(x)?.relu()
    }
}

// ── Res2NetBlock ──────────────────────────────────────────────────────────
//
// Splits input into 8 chunks along channel dim. First chunk passes through
// unchanged. Chunks 1..7 go through TDNN blocks with cumulative addition
// from the previous chunk's output.

struct Res2NetBlock {
    scale: usize, // 8
    chunk_size: usize,
    blocks: Vec<TimeDelayNetBlock>, // 7 blocks (indices 0..6 → chunks 1..7)
}

impl Res2NetBlock {
    fn load(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let chunk_size = channels / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for j in 0..(scale - 1) {
            let tdnn = TimeDelayNetBlock::load(
                chunk_size,
                chunk_size,
                kernel_size,
                dilation,
                vb.pp(format!("blocks.{j}")),
            )?;
            blocks.push(tdnn);
        }
        Ok(Self {
            scale,
            chunk_size,
            blocks,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [B, C, T] — split along channel dim into `scale` chunks
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);

        // chunk[0]: pass through (no conv)
        let chunk0 = x.narrow(1, 0, self.chunk_size)?;
        outputs.push(chunk0);

        for i in 1..self.scale {
            let chunk_i = x.narrow(1, i * self.chunk_size, self.chunk_size)?;
            // chunk[i>0] gets chunk[i] + output[i-1] as input
            let input_i = (chunk_i + outputs.last().unwrap())?;
            let out_i = self.blocks[i - 1].forward(&input_i)?;
            outputs.push(out_i);
        }

        Tensor::cat(&outputs, 1)
    }
}

// ── SqueezeExcitationBlock ────────────────────────────────────────────────
//
// Global avg pool → Conv1d(ch→se_ch, k=1) + ReLU → Conv1d(se_ch→ch, k=1) + Sigmoid → multiply

struct SqueezeExcitationBlock {
    conv1: ReflectConv1d,
    conv2: ReflectConv1d,
}

impl SqueezeExcitationBlock {
    fn load(channels: usize, se_channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv1 = ReflectConv1d::load(channels, se_channels, 1, 1, 1, vb.pp("conv1"))?;
        let conv2 = ReflectConv1d::load(se_channels, channels, 1, 1, 1, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Global average pooling over time: [B, C, T] → [B, C, 1]
        let s = x.mean_keepdim(2)?;
        let s = self.conv1.forward(&s)?.relu()?;
        let s = self.conv2.forward(&s)?;
        // Sigmoid
        let s = sigmoid(&s)?;
        // Channel-wise scale
        x.broadcast_mul(&s)
    }
}

/// Manual sigmoid: 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    let ones = x.ones_like()?;
    let neg = x.neg()?;
    ones.broadcast_div(&(neg.exp()? + 1.0)?)
}

// ── SqueezeExcitationRes2NetBlock ─────────────────────────────────────────
//
// TDNN1(in→out, k=1) + Res2Net(out, k=3, dilation) + TDNN2(out→out, k=1) + SE + residual

struct SERes2NetBlock {
    tdnn1: TimeDelayNetBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TimeDelayNetBlock,
    se_block: SqueezeExcitationBlock,
    shortcut: Option<ReflectConv1d>, // only if in_ch != out_ch
}

impl SERes2NetBlock {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        dilation: usize,
        se_channels: usize,
        res2net_scale: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let tdnn1 = TimeDelayNetBlock::load(in_ch, out_ch, 1, 1, vb.pp("tdnn1"))?;
        let res2net_block = Res2NetBlock::load(
            out_ch,
            kernel_size,
            dilation,
            res2net_scale,
            vb.pp("res2net_block"),
        )?;
        let tdnn2 = TimeDelayNetBlock::load(out_ch, out_ch, 1, 1, vb.pp("tdnn2"))?;
        let se_block = SqueezeExcitationBlock::load(out_ch, se_channels, vb.pp("se_block"))?;
        let shortcut = if in_ch != out_ch {
            Some(ReflectConv1d::load(
                in_ch,
                out_ch,
                1,
                1,
                1,
                vb.pp("shortcut.conv"),
            )?)
        } else {
            None
        };
        Ok(Self {
            tdnn1,
            res2net_block,
            tdnn2,
            se_block,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        let out = self.tdnn1.forward(x)?;
        let out = self.res2net_block.forward(&out)?;
        let out = self.tdnn2.forward(&out)?;
        let out = self.se_block.forward(&out)?;
        (out + residual)?.relu()
    }
}

// ── Attentive Statistics Pooling ──────────────────────────────────────────
//
// Compute attention-weighted mean and std over time dimension.
// Input: [B, C, T]  Output: [B, C*2, 1]

struct AttentiveStatisticsPooling {
    tdnn: TimeDelayNetBlock, // Conv1d(ch*3 → attention_ch, k=1)
    conv: ReflectConv1d,     // Conv1d(attention_ch → ch, k=1)
}

impl AttentiveStatisticsPooling {
    fn load(
        channels: usize,
        attention_channels: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let tdnn = TimeDelayNetBlock::load(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?;
        let conv = ReflectConv1d::load(attention_channels, channels, 1, 1, 1, vb.pp("conv"))?;
        Ok(Self { tdnn, conv })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [B, C, T]

        // Compute mean and std over time (full mask — no padding)
        let mean = x.mean_keepdim(2)?; // [B, C, 1]
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(2)?;
        let std = (var + 1e-9)?.sqrt()?; // [B, C, 1]

        // Expand mean/std to match time dim
        let mean_exp = mean.expand(x.dims())?; // [B, C, T]
        let std_exp = std.expand(x.dims())?; // [B, C, T]

        // Concat [x, mean, std] along channel dim → [B, C*3, T]
        let cat = Tensor::cat(&[x, &mean_exp, &std_exp], 1)?;

        // Attention weights
        let attn = self.tdnn.forward(&cat)?; // [B, attn_ch, T] (includes ReLU)
        let attn = attn.tanh()?;
        let attn = self.conv.forward(&attn)?; // [B, C, T]

        // Softmax over time dimension
        let attn = softmax_dim2(&attn)?; // [B, C, T]

        // Weighted statistics
        let weighted = (x * &attn)?;
        let w_mean = weighted.sum_keepdim(2)?; // [B, C, 1]

        let w_diff = x.broadcast_sub(&w_mean)?;
        let w_var = (w_diff.sqr()? * &attn)?.sum_keepdim(2)?;
        let w_std = (w_var + 1e-9)?.sqrt()?; // [B, C, 1]

        // Concat mean + std → [B, C*2, 1]
        Tensor::cat(&[&w_mean, &w_std], 1)
    }
}

/// Softmax over dim 2 (time dimension for [B, C, T] tensors).
fn softmax_dim2(x: &Tensor) -> candle_core::Result<Tensor> {
    let max = x.max_keepdim(2)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(2)?;
    exp.broadcast_div(&sum)
}

// ── SpeakerEncoder (full model) ───────────────────────────────────────────
//
// blocks[0]: TDNN(128→512, k=5, dilation=1)
// blocks[1]: SE-Res2Net(512→512, k=3, dilation=2)
// blocks[2]: SE-Res2Net(512→512, k=3, dilation=3)
// blocks[3]: SE-Res2Net(512→512, k=3, dilation=4)
// MFA: concat(blocks[1..3] outputs) → TDNN(1536→1536, k=1)
// ASP: AttentiveStatisticsPooling(1536, attention_channels=128)
// FC: Conv1d(3072→1024, k=1) with reflect padding

pub struct SpeakerEncoder {
    block0: TimeDelayNetBlock,
    se_blocks: Vec<SERes2NetBlock>, // 3 blocks
    mfa: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: ReflectConv1d,
}

impl SpeakerEncoder {
    /// Load speaker encoder weights from VarBuilder.
    /// Expects the VarBuilder to be scoped to the `speaker_encoder` prefix.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        info!("Loading ECAPA-TDNN speaker encoder");

        // blocks.0: TDNN(128→512, k=5, dilation=1)
        let block0 = TimeDelayNetBlock::load(128, 512, 5, 1, vb.pp("blocks.0"))
            .map_err(|e| FerrumError::model(format!("speaker_encoder blocks.0: {e}")))?;

        // blocks.1-3: SE-Res2Net(512→512, k=3, dilation=2,3,4)
        let mut se_blocks = Vec::with_capacity(3);
        for (i, dilation) in [(1usize, 2usize), (2, 3), (3, 4)] {
            let blk = SERes2NetBlock::load(
                512, // in_ch
                512, // out_ch
                3,   // kernel_size
                dilation,
                128, // se_channels
                8,   // res2net_scale
                vb.pp(format!("blocks.{i}")),
            )
            .map_err(|e| FerrumError::model(format!("speaker_encoder blocks.{i}: {e}")))?;
            se_blocks.push(blk);
        }

        // MFA: TDNN(1536→1536, k=1, dilation=1)
        let mfa = TimeDelayNetBlock::load(1536, 1536, 1, 1, vb.pp("mfa"))
            .map_err(|e| FerrumError::model(format!("speaker_encoder mfa: {e}")))?;

        // ASP: AttentiveStatisticsPooling(1536, attention_channels=128)
        let asp = AttentiveStatisticsPooling::load(1536, 128, vb.pp("asp"))
            .map_err(|e| FerrumError::model(format!("speaker_encoder asp: {e}")))?;

        // FC: Conv1d(3072→1024, k=1) with reflect padding
        let fc = ReflectConv1d::load(3072, 1024, 1, 1, 1, vb.pp("fc"))
            .map_err(|e| FerrumError::model(format!("speaker_encoder fc: {e}")))?;

        info!("Speaker encoder loaded (ECAPA-TDNN, 1024-dim output)");
        Ok(Self {
            block0,
            se_blocks,
            mfa,
            asp,
            fc,
        })
    }

    /// Forward pass: mel spectrogram → speaker embedding.
    ///
    /// - `mel`: [1, T, 128] mel spectrogram tensor
    /// - Returns: [1024] speaker embedding vector
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Transpose [1, T, 128] → [1, 128, T] for Conv1d processing
        let x = mel
            .transpose(1, 2)
            .map_err(|e| FerrumError::model(format!("speaker_encoder transpose: {e}")))?;

        // blocks[0]: initial TDNN
        let x = self
            .block0
            .forward(&x)
            .map_err(|e| FerrumError::model(format!("speaker_encoder block0: {e}")))?;

        // blocks[1..3]: SE-Res2Net blocks (collect outputs for MFA)
        let mut se_outputs = Vec::with_capacity(3);
        let mut x = x;
        for (i, blk) in self.se_blocks.iter().enumerate() {
            x = blk
                .forward(&x)
                .map_err(|e| FerrumError::model(format!("speaker_encoder se_block[{i}]: {e}")))?;
            se_outputs.push(x.clone());
        }

        // MFA: concat SE block outputs along channel dim → TDNN
        let mfa_in = Tensor::cat(&se_outputs, 1)
            .map_err(|e| FerrumError::model(format!("speaker_encoder mfa cat: {e}")))?;
        let mfa_out = self
            .mfa
            .forward(&mfa_in)
            .map_err(|e| FerrumError::model(format!("speaker_encoder mfa: {e}")))?;

        // ASP: [B, 1536, T] → [B, 3072, 1]
        let asp_out = self
            .asp
            .forward(&mfa_out)
            .map_err(|e| FerrumError::model(format!("speaker_encoder asp: {e}")))?;

        // FC: [B, 3072, 1] → [B, 1024, 1]
        let fc_out = self
            .fc
            .forward(&asp_out)
            .map_err(|e| FerrumError::model(format!("speaker_encoder fc: {e}")))?;

        // Squeeze: [B, 1024, 1] → [1024]
        let emb = fc_out
            .squeeze(2)
            .map_err(|e| FerrumError::model(format!("speaker_encoder squeeze(2): {e}")))?
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("speaker_encoder squeeze(0): {e}")))?;

        Ok(emb)
    }
}

// ── Mel spectrogram for speaker encoder ───────────────────────────────────
//
// Different from Whisper mel:
//   - 24kHz sample rate, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000
//   - Magnitude (NOT squared): sqrt(re^2 + im^2 + 1e-9)
//   - Log compression: log(clamp(x, 1e-5)) (NOT log10)
//   - No normalization (no max-8.0 clamp, no +4/4 scaling)
//   - Returns [1, T, 128] in row-major (time x mels)

/// Compute mel spectrogram for the speaker encoder.
///
/// - `pcm`: audio samples (f32, 24kHz mono)
/// - Returns flat `Vec<f32>` in [T, 128] layout (row-major, time-first)
///   suitable for creating a [1, T, 128] tensor.
pub fn mel_spectrogram_speaker_encoder(pcm: &[f32]) -> Vec<f32> {
    use rustfft::{num_complex::Complex, FftPlanner};

    const N_FFT: usize = 1024;
    const HOP_SIZE: usize = 256;
    const WIN_SIZE: usize = 1024;
    const N_MELS: usize = 128;
    const N_FFT_HALF: usize = N_FFT / 2 + 1; // 513

    // Parse mel filterbank from embedded binary (128 x 513, f32le row-major)
    let mel_filters = parse_mel_filters();

    // Step 1: Reflect-pad for center=False with (n_fft - hop_size) / 2 on each side
    let pad_size = (N_FFT - HOP_SIZE) / 2; // 384
    let padded = reflect_pad_pcm(pcm, pad_size);

    // Step 2: STFT with Hann window
    let n_frames = (padded.len() - N_FFT) / HOP_SIZE + 1;

    // Hann window (periodic: 2*pi*i / win_size, NOT win_size-1)
    let hann: Vec<f32> = (0..WIN_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / WIN_SIZE as f32).cos()))
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // Compute magnitude spectrogram: sqrt(re^2 + im^2 + 1e-9) (NOT squared)
    // Layout: [N_FFT_HALF, n_frames] column-major (freq x time)
    let mut magnitudes = vec![0f32; N_FFT_HALF * n_frames];
    let mut buffer = vec![Complex::new(0f32, 0f32); N_FFT];

    for t in 0..n_frames {
        let offset = t * HOP_SIZE;
        for i in 0..N_FFT {
            buffer[i] = Complex::new(padded[offset + i] * hann[i], 0.0);
        }
        fft.process(&mut buffer);
        for f in 0..N_FFT_HALF {
            let mag_sq = buffer[f].re * buffer[f].re + buffer[f].im * buffer[f].im;
            magnitudes[f * n_frames + t] = (mag_sq + 1e-9).sqrt();
        }
    }

    // Step 3: Mel projection: mel_filters[N_MELS, N_FFT_HALF] @ mag[N_FFT_HALF, n_frames]
    // Result: [N_MELS, n_frames]
    let mut mel_spec = vec![0f32; N_MELS * n_frames];
    for m in 0..N_MELS {
        for t in 0..n_frames {
            let mut sum = 0f32;
            for f in 0..N_FFT_HALF {
                sum += mel_filters[m * N_FFT_HALF + f] * magnitudes[f * n_frames + t];
            }
            mel_spec[m * n_frames + t] = sum;
        }
    }

    // Step 4: Log compression: log(clamp(x, min=1e-5))
    for v in &mut mel_spec {
        *v = v.max(1e-5).ln();
    }

    // Step 5: Transpose from [N_MELS, n_frames] to [n_frames, N_MELS] (time x mels)
    let mut output = vec![0f32; n_frames * N_MELS];
    for t in 0..n_frames {
        for m in 0..N_MELS {
            output[t * N_MELS + m] = mel_spec[m * n_frames + t];
        }
    }

    output
}

/// Parse mel filterbank from embedded binary data.
/// Shape: [128, 513], stored as f32 little-endian row-major.
fn parse_mel_filters() -> Vec<f32> {
    const N_MELS: usize = 128;
    const N_FFT_HALF: usize = 513;
    let expected = N_MELS * N_FFT_HALF;

    assert_eq!(
        MEL_FILTERS.len(),
        expected * 4,
        "mel_filters_spkenc.bin: expected {} bytes ({} x {} x 4), got {}",
        expected * 4,
        N_MELS,
        N_FFT_HALF,
        MEL_FILTERS.len()
    );

    let mut filters = vec![0f32; expected];
    for (i, chunk) in MEL_FILTERS.chunks_exact(4).enumerate() {
        filters[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    filters
}

/// Reflect-pad PCM signal on both sides.
fn reflect_pad_pcm(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // Left reflection: signal[pad], signal[pad-1], ..., signal[1]
    for i in (1..=pad).rev() {
        out.push(signal[i.min(n - 1)]);
    }
    out.extend_from_slice(signal);
    // Right reflection: signal[n-2], signal[n-3], ..., signal[n-1-pad]
    for i in 1..=pad {
        out.push(signal[(n - 1).saturating_sub(i)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflect_pad_pcm() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad_pcm(&signal, 2);
        // Left: signal[2], signal[1] = 3.0, 2.0
        // Right: signal[3], signal[2] = 4.0, 3.0
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_mel_filters_parse() {
        let filters = parse_mel_filters();
        assert_eq!(filters.len(), 128 * 513);
        // Should contain some non-zero values (it's a real filterbank)
        let nonzero = filters.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "mel filterbank should have non-zero entries");
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        // 1 second of silence at 24kHz
        let pcm = vec![0.0f32; 24000];
        let mel = mel_spectrogram_speaker_encoder(&pcm);
        // n_frames = (24000 + 2*384 - 1024) / 256 + 1
        //          = (24768 - 1024) / 256 + 1 = 23744 / 256 + 1 = 92 + 1 = 93
        let n_frames = mel.len() / 128;
        assert_eq!(mel.len() % 128, 0, "mel length should be multiple of 128");
        assert!(n_frames > 0, "should have at least 1 frame");
    }

    #[test]
    fn test_sigmoid() {
        let dev = candle_core::Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &dev).unwrap();
        let s = sigmoid(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!((s[0] - 0.5).abs() < 1e-5);
        assert!((s[1] - 0.7311).abs() < 1e-3);
        assert!((s[2] - 0.2689).abs() < 1e-3);
    }
}
