//! Mimi-based speech tokenizer encoder for Qwen3-TTS ICL voice cloning.
//!
//! Takes raw 24kHz audio and outputs codec token indices [T, 16]
//! (1 semantic + 15 acoustic codebooks).
//!
//! Uses candle-transformers' Mimi components directly (SeaNetEncoder,
//! ProjectedTransformer, ConvDownsample1d, SplitResidualVectorQuantizer).
//!
//! Loaded from speech_tokenizer/model.safetensors.

use candle_core::{DType, Device as CandleDevice, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use ferrum_types::{FerrumError, Result};
use tracing::info;

// ── Config ──────────────────────────────────────────────────────────────

const HIDDEN_SIZE: usize = 512;
const NUM_HEADS: usize = 8;
const NUM_TRANSFORMER_LAYERS: usize = 8;
const SEMANTIC_CODEBOOK_SIZE: usize = 2048;
const ACOUSTIC_CODEBOOK_SIZE: usize = 2048;
const CODEBOOK_DIM: usize = 256;
const NUM_ACOUSTIC_CODEBOOKS: usize = 31;
const NUM_OUTPUT_CODEBOOKS: usize = 16; // 1 semantic + 15 acoustic

// ── SpeechTokenizerEncoder ──────────────────────────────────────────────

/// Mimi-based speech tokenizer encoder: raw 24kHz PCM → codec tokens.
///
/// All components use candle-transformers' Mimi implementation for correctness.
/// This is a cold path (runs once per voice clone), so performance is secondary.
pub struct SpeechTokenizerEncoder {
    conv_stack: candle_transformers::models::mimi::seanet::SeaNetEncoder,
    transformer:
        parking_lot::Mutex<candle_transformers::models::mimi::transformer::ProjectedTransformer>,
    downsample: candle_transformers::models::mimi::conv::ConvDownsample1d,
    quantizer: candle_transformers::models::mimi::quantization::SplitResidualVectorQuantizer,
    device: CandleDevice,
}

impl SpeechTokenizerEncoder {
    /// Load from VarBuilder scoped to `encoder.` prefix.
    pub fn load(vb: VarBuilder, device: CandleDevice) -> Result<Self> {
        let mimi_cfg = candle_transformers::models::mimi::Config::v0_1(Some(NUM_OUTPUT_CODEBOOKS));

        let conv_stack = candle_transformers::models::mimi::seanet::SeaNetEncoder::new(
            &mimi_cfg.seanet,
            vb.pp("encoder"),
        )
        .map_err(|e| FerrumError::model(format!("encoder conv stack: {e}")))?;

        let transformer =
            candle_transformers::models::mimi::transformer::ProjectedTransformer::new(
                mimi_cfg.seanet.dimension,
                &[mimi_cfg.seanet.dimension],
                &mimi_cfg.transformer,
                vb.pp("encoder_transformer"),
            )
            .map_err(|e| FerrumError::model(format!("encoder transformer: {e}")))?;

        let downsample = candle_transformers::models::mimi::conv::ConvDownsample1d::new(
            2, // stride: 25Hz → 12.5Hz
            mimi_cfg.seanet.dimension,
            true, // causal
            true, // learnt
            vb.pp("downsample"),
        )
        .map_err(|e| FerrumError::model(format!("encoder downsample: {e}")))?;

        let quantizer =
            candle_transformers::models::mimi::quantization::SplitResidualVectorQuantizer::new(
                CODEBOOK_DIM,
                Some(HIDDEN_SIZE),
                Some(HIDDEN_SIZE),
                NUM_OUTPUT_CODEBOOKS,
                SEMANTIC_CODEBOOK_SIZE,
                vb.pp("quantizer"),
            )
            .map_err(|e| FerrumError::model(format!("encoder quantizer: {e}")))?;

        info!(
            "SpeechTokenizerEncoder loaded: conv=15 layers (960x ds) + 2x downsample, \
             transformer={} layers (h={}, heads={}), \
             RVQ=1x{}+{}x{} → {} codebooks",
            NUM_TRANSFORMER_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            SEMANTIC_CODEBOOK_SIZE,
            NUM_ACOUSTIC_CODEBOOKS,
            ACOUSTIC_CODEBOOK_SIZE,
            NUM_OUTPUT_CODEBOOKS,
        );

        Ok(Self {
            conv_stack,
            transformer: parking_lot::Mutex::new(transformer),
            downsample,
            quantizer,
            device,
        })
    }

    /// Encode 24kHz mono PCM → codec token indices `[T, 16]`.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        let num_samples = pcm.len();
        info!(
            "SpeechTokenizerEncoder: encoding {} samples ({:.2}s @ 24kHz)",
            num_samples,
            num_samples as f64 / 24000.0,
        );

        let input = Tensor::from_vec(pcm.to_vec(), (1, 1, num_samples), &self.device)
            .map_err(|e| FerrumError::model(format!("input tensor: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| FerrumError::model(format!("input dtype: {e}")))?;

        // Conv encoder → Transformer → Downsample → Quantize
        let conv_out = input
            .apply(&self.conv_stack)
            .map_err(|e| FerrumError::model(format!("conv encoder: {e}")))?;

        let mut transformer = self.transformer.lock();
        let hidden = transformer
            .forward(&conv_out)
            .map_err(|e| FerrumError::model(format!("encoder transformer: {e}")))?;
        let hidden = &hidden[0];

        let hidden = hidden
            .apply(&self.downsample)
            .map_err(|e| FerrumError::model(format!("encoder downsample: {e}")))?;

        let codes = self
            .quantizer
            .encode(&hidden)
            .map_err(|e| FerrumError::model(format!("quantizer encode: {e}")))?;

        // [1, 16, T] → Vec<Vec<u32>> as [T, 16]
        let codes = codes
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze: {e}")))?
            .transpose(0, 1)
            .map_err(|e| FerrumError::model(format!("transpose: {e}")))?
            .to_dtype(DType::U32)
            .map_err(|e| FerrumError::model(format!("to_u32: {e}")))?;

        let t = codes
            .dim(0)
            .map_err(|e| FerrumError::model(format!("dim: {e}")))?;
        let k = codes
            .dim(1)
            .map_err(|e| FerrumError::model(format!("dim1: {e}")))?;
        info!("SpeechTokenizerEncoder: {} frames, {} codebooks", t, k);

        let mut result = Vec::with_capacity(t);
        for ti in 0..t {
            let row: Vec<u32> = codes
                .i(ti)
                .and_then(|r| r.to_vec1())
                .map_err(|e| FerrumError::model(format!("codes row: {e}")))?;
            result.push(row);
        }
        Ok(result)
    }
}
