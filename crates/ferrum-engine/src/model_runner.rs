//! Model runner with Candle integration

use ferrum_core::{Model, ModelLoader, ModelConfig, ModelInfo, Result};
use candle_core::Device;
use std::sync::Arc;

/// Placeholder for model runner implementation
pub struct CandleModelRunner {
    device: Device,
}

impl CandleModelRunner {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}
