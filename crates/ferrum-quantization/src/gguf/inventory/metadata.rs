//! Product source discovery independent of the tensor decoder's type support.

use std::collections::BTreeMap;
use std::io::{Read, Seek};

use candle_core::{Error, Result};

use super::header::{base_model_repository_index, Header};

/// Declared provenance used to locate independent semantic/tokenizer sources.
/// Reading this metadata does not validate or load the tensor payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufModelMetadata {
    pub architecture: String,
    pub source_repository_url: Option<String>,
    pub base_model_count: Option<u64>,
    /// Sparse because repository URLs are optional for each declared parent.
    pub base_model_repository_urls: BTreeMap<u64, String>,
}

impl GgufModelMetadata {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let header = Header::read(reader)?;
        let string = |key: &str| {
            header
                .metadata
                .get(key)
                .map(|v| v.to_string().cloned())
                .transpose()
        };
        let architecture = string("general.architecture")?
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Error::Msg("GGUF requires nonempty general.architecture".into()))?;
        let base_model_count = header
            .metadata
            .get("general.base_model.count")
            .map(super::metadata_integer)
            .transpose()?;
        let mut base_model_repository_urls = BTreeMap::new();
        for (key, value) in &header.metadata {
            if let Some(index) = base_model_repository_index(key) {
                if base_model_count.is_none_or(|count| index >= count) {
                    return Err(Error::Msg(format!(
                        "GGUF {key} is outside the declared general.base_model.count"
                    )));
                }
                base_model_repository_urls.insert(index, value.to_string()?.clone());
            }
        }
        Ok(Self {
            architecture,
            source_repository_url: string("general.source.repo_url")?,
            base_model_count,
            base_model_repository_urls,
        })
    }
}
