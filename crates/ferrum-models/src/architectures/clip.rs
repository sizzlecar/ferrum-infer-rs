//! CLIP model wrapper — supports OpenAI CLIP, Chinese-CLIP, and SigLIP.
//!
//! Wraps candle-transformers' ClipModel / ChineseClipModel / siglip::Model
//! with a unified interface for text and image embedding extraction.

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::chinese_clip::{ChineseClipConfig, ChineseClipModel};
use candle_transformers::models::clip::{self, ClipConfig, ClipModel};
use candle_transformers::models::siglip;
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use tracing::info;

/// Which CLIP variant is loaded.
enum ClipVariant {
    OpenAI(ClipModel),
    Chinese(ChineseClipModel),
    SigLIP(siglip::Model),
}

/// Unified CLIP model wrapper.
pub struct ClipModelWrapper {
    model: Mutex<ClipVariant>,
    device: CandleDevice,
    dtype: DType,
    image_size: usize,
    projection_dim: usize,
}

impl ClipModelWrapper {
    /// Load OpenAI CLIP from VarBuilder.
    pub fn new_openai(
        vb: VarBuilder,
        config: &ClipConfig,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("Loading OpenAI CLIP (image_size={})", config.image_size);
        let model = ClipModel::new(vb, config)
            .map_err(|e| FerrumError::model(format!("CLIP load: {e}")))?;
        Ok(Self {
            projection_dim: config.vision_config.projection_dim,
            image_size: config.image_size,
            model: Mutex::new(ClipVariant::OpenAI(model)),
            device,
            dtype,
        })
    }

    /// Load Chinese-CLIP from VarBuilder.
    pub fn new_chinese(
        vb: VarBuilder,
        config: &ChineseClipConfig,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!(
            "Loading Chinese-CLIP (image_size={}, projection_dim={})",
            config.image_size, config.projection_dim
        );
        let model = ChineseClipModel::new(vb, config)
            .map_err(|e| FerrumError::model(format!("Chinese-CLIP load: {e}")))?;
        Ok(Self {
            projection_dim: config.projection_dim,
            image_size: config.image_size,
            model: Mutex::new(ClipVariant::Chinese(model)),
            device,
            dtype,
        })
    }

    /// Load SigLIP from VarBuilder.
    pub fn new_siglip(
        vb: VarBuilder,
        config: &siglip::Config,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        let image_size = config.vision_config.image_size;
        let projection_dim = config.vision_config.hidden_size;
        info!(
            "Loading SigLIP (image_size={}, hidden_size={})",
            image_size, projection_dim
        );
        let model = siglip::Model::new(config, vb)
            .map_err(|e| FerrumError::model(format!("SigLIP load: {e}")))?;
        Ok(Self {
            projection_dim,
            image_size,
            model: Mutex::new(ClipVariant::SigLIP(model)),
            device,
            dtype,
        })
    }

    /// Load from config.json — auto-detects CLIP variant.
    ///
    /// candle's ClipConfig doesn't derive Deserialize, so we use preset configs
    /// and override image_size / projection_dim from the JSON when present.
    pub fn from_config_json(
        vb: VarBuilder,
        config_path: &std::path::Path,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(config_path)
                .map_err(|e| FerrumError::model(format!("read config: {e}")))?,
        )
        .map_err(|e| FerrumError::model(format!("parse config: {e}")))?;

        let model_type = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or("");

        if model_type == "siglip" {
            // SigLIP config derives Deserialize — parse directly
            let config: siglip::Config =
                serde_json::from_value(raw).unwrap_or_else(|_| siglip::Config::base_patch16_224());
            return Self::new_siglip(vb, &config, device, dtype);
        }

        if model_type == "chinese_clip" {
            let mut config = ChineseClipConfig::clip_vit_base_patch16();
            if let Some(v) = raw.get("projection_dim").and_then(|v| v.as_u64()) {
                config.projection_dim = v as usize;
            }
            if let Some(vc) = raw.get("vision_config") {
                if let Some(v) = vc.get("image_size").and_then(|v| v.as_u64()) {
                    config.vision_config.image_size = v as usize;
                    config.image_size = v as usize;
                }
            }
            Self::new_chinese(vb, &config, device, dtype)
        } else {
            let mut config = ClipConfig::vit_base_patch32();
            if let Some(vc) = raw.get("vision_config") {
                if let Some(v) = vc.get("image_size").and_then(|v| v.as_u64()) {
                    config.vision_config.image_size = v as usize;
                    config.image_size = v as usize;
                }
                if let Some(v) = vc.get("projection_dim").and_then(|v| v.as_u64()) {
                    config.vision_config.projection_dim = v as usize;
                }
            }
            Self::new_openai(vb, &config, device, dtype)
        }
    }

    /// Get text embedding (L2-normalized).
    pub fn get_text_features(&self, input_ids: &Tensor) -> Result<Tensor> {
        let model = self.model.lock();
        let features = match &*model {
            ClipVariant::OpenAI(m) => m
                .get_text_features(input_ids)
                .map_err(|e| FerrumError::model(format!("text features: {e}")))?,
            ClipVariant::Chinese(m) => m
                .get_text_features(input_ids, None, None)
                .map_err(|e| FerrumError::model(format!("text features: {e}")))?,
            ClipVariant::SigLIP(m) => m
                .get_text_features(input_ids)
                .map_err(|e| FerrumError::model(format!("text features: {e}")))?,
        };
        clip::div_l2_norm(&features).map_err(|e| FerrumError::model(format!("l2 norm: {e}")))
    }

    /// Get image embedding (L2-normalized).
    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let model = self.model.lock();
        let features = match &*model {
            ClipVariant::OpenAI(m) => m
                .get_image_features(pixel_values)
                .map_err(|e| FerrumError::model(format!("image features: {e}")))?,
            ClipVariant::Chinese(m) => m
                .get_image_features(pixel_values)
                .map_err(|e| FerrumError::model(format!("image features: {e}")))?,
            ClipVariant::SigLIP(m) => m
                .get_image_features(pixel_values)
                .map_err(|e| FerrumError::model(format!("image features: {e}")))?,
        };
        clip::div_l2_norm(&features).map_err(|e| FerrumError::model(format!("l2 norm: {e}")))
    }

    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn image_size(&self) -> usize {
        self.image_size
    }

    pub fn projection_dim(&self) -> usize {
        self.projection_dim
    }
}
