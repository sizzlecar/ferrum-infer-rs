//! 权重加载占位实现

use std::sync::Arc;

use ferrum_interfaces::WeightLoader;
use ferrum_types::Result;

#[derive(Debug, Clone, Copy, Default)]
pub enum WeightFormat {
    #[default]
    Unknown,
}

#[derive(Clone, Default)]
pub struct WeightLoaderHandle(pub Arc<dyn WeightLoader + Send + Sync>);

pub fn default_weight_loader(_format: WeightFormat) -> Result<WeightLoaderHandle> {
    Err(ferrum_types::FerrumError::not_implemented(
        "Weight loader placeholder",
    ))
}
