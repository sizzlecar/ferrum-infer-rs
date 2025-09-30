//! 模型来源解析占位实现

use ferrum_types::{ModelSource, Result};

#[derive(Debug, Clone, Default)]
pub struct ModelSourceConfig {
    pub cache_dir: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelSourceResolver {
    config: ModelSourceConfig,
}

impl ModelSourceResolver {
    pub fn new(config: ModelSourceConfig) -> Self {
        Self { config }
    }

    pub async fn resolve(&self, id: &str) -> Result<ResolvedModelSource> {
        Ok(ResolvedModelSource {
            original: id.to_string(),
            local_path: self.config.cache_dir.clone().unwrap_or_default(),
            format: ModelFormat::Unknown,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum ModelFormat {
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Default)]
pub struct ResolvedModelSource {
    pub original: String,
    pub local_path: std::path::PathBuf,
    pub format: ModelFormat,
}

impl From<ResolvedModelSource> for ModelSource {
    fn from(value: ResolvedModelSource) -> Self {
        ModelSource::Local(value.local_path.display().to_string())
    }
}
