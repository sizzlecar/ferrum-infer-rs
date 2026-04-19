//! Audio preprocessing for Whisper ASR.
//!
//! Load audio files → decode → resample to 16kHz mono → f32 PCM samples.
//! Pure-Rust pipeline via `symphonia` — no ffmpeg runtime dependency.
//! Supports WAV / MP3 / FLAC / M4A (AAC) / OGG (Vorbis).

use ferrum_types::{FerrumError, Result};
use std::path::Path;

/// Whisper processes 30-second chunks. At 16kHz → 480,000 samples.
pub const CHUNK_SAMPLES: usize = 16000 * 30;

/// Load audio file and return 16kHz mono f32 PCM samples.
pub fn load_audio(path: &str) -> Result<Vec<f32>> {
    load_audio_at_rate(path, 16000)
}

/// Load audio file and return mono f32 PCM samples at a configurable sample rate.
///
/// Accepts WAV / MP3 / FLAC / M4A (AAC) / OGG (Vorbis). Non-mono sources are
/// downmixed by averaging channels; sample rate is converted via sinc resampling.
/// Useful for TTS speaker encoder which expects 24kHz input.
pub fn load_audio_at_rate(path: &str, target_rate: u32) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)
        .map_err(|e| FerrumError::model(format!("open audio {path}: {e}")))?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default());

    // Hint the decoder with the file extension — purely an optimisation; the
    // probe still content-sniffs if the hint is absent or wrong.
    let mut hint = symphonia::core::probe::Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(&ext.to_lowercase());
    }

    decode_with_symphonia(mss, &hint, target_rate)
}

/// Load audio from raw bytes (used by the HTTP multipart endpoint).
///
/// Content-sniffs the format; supports the same codec set as `load_audio`.
pub fn load_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(cursor), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    decode_with_symphonia(mss, &hint, 16000)
}

/// Split PCM samples into 30-second chunks for Whisper processing.
pub fn chunk_pcm(pcm: &[f32]) -> Vec<&[f32]> {
    if pcm.len() <= CHUNK_SAMPLES {
        return vec![pcm];
    }
    pcm.chunks(CHUNK_SAMPLES).collect()
}

// ── symphonia-based decoding ────────────────────────────────────────────

fn decode_with_symphonia(
    mss: symphonia::core::io::MediaSourceStream,
    hint: &symphonia::core::probe::Hint,
    target_rate: u32,
) -> Result<Vec<f32>> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::meta::MetadataOptions;

    let fmt_opts: FormatOptions = Default::default();
    let meta_opts: MetadataOptions = Default::default();
    let probed = symphonia::default::get_probe()
        .format(hint, mss, &fmt_opts, &meta_opts)
        .map_err(|e| FerrumError::model(format!("probe audio: {e}")))?;

    let mut format = probed.format;

    // First audio track wins. Whisper-friendly files are mono or stereo; we
    // downmix to mono below.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| FerrumError::model("no audio track in file"))?;
    let track_id = track.id;

    let source_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| FerrumError::model("audio track missing sample rate"))?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|e| FerrumError::model(format!("decoder init: {e}")))?;

    // Accumulate interleaved f32 samples across all decoded packets, then
    // downmix + resample once at the end. Cheaper in wall time than per-packet
    // resampling for short files and simpler to reason about.
    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(FerrumError::model(format!("next_packet: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => append_interleaved_f32(&decoded, &mut interleaved, channels),
            Err(SymError::DecodeError(_)) => continue, // skip corrupt packet
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FerrumError::model(format!("decode: {e}"))),
        }
    }

    // Downmix to mono (average channels)
    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Resample to target_rate if needed
    if source_rate == target_rate {
        Ok(mono)
    } else {
        Ok(resample(&mono, source_rate as f64, target_rate as f64))
    }
}

/// Convert any symphonia AudioBufferRef variant into interleaved f32 samples
/// and append to `out`. Handles U8 / S16 / S24 / S32 / F32 / F64 — which
/// covers every codec we compile in.
fn append_interleaved_f32(
    buf: &symphonia::core::audio::AudioBufferRef<'_>,
    out: &mut Vec<f32>,
    channels: usize,
) {
    use symphonia::core::audio::{AudioBuffer, AudioBufferRef, Signal};
    use symphonia::core::conv::IntoSample;

    fn push_interleaved<S>(buf: &AudioBuffer<S>, out: &mut Vec<f32>, channels: usize)
    where
        S: symphonia::core::sample::Sample + IntoSample<f32> + Copy,
    {
        let frames = buf.frames();
        out.reserve(frames * channels);
        for frame in 0..frames {
            for ch in 0..channels {
                let s: S = buf.chan(ch)[frame];
                out.push(s.into_sample());
            }
        }
    }

    match buf {
        AudioBufferRef::U8(b) => push_interleaved(b, out, channels),
        AudioBufferRef::S16(b) => push_interleaved(b, out, channels),
        AudioBufferRef::S24(b) => push_interleaved(b, out, channels),
        AudioBufferRef::S32(b) => push_interleaved(b, out, channels),
        AudioBufferRef::F32(b) => push_interleaved(b, out, channels),
        AudioBufferRef::F64(b) => push_interleaved(b, out, channels),
        _ => {
            // U16, S8 etc. not exposed by the codecs we compile in; silently
            // skip rather than add dead conversion code.
        }
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
