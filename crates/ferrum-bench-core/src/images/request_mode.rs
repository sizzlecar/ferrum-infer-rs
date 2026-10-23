use super::{invalid, ImageBenchError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One mode per dataset prevents a repetition or endpoint from silently using
/// different generation/edit workloads. Generate is the unchanged v1 default.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageRequestMode {
    #[default]
    Generate,
    Edit {
        image: ImageReference,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageReference {
    /// CLI paths are relative to the dataset file; library callers resolve
    /// paths relative to their own process directory. This is not an identity.
    pub path: PathBuf,
    /// Optional predeclared encoded-file identity. The runner always computes
    /// and reports the actual SHA-256, even when this check is omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_sha256: Option<String>,
}

impl ImageRequestMode {
    pub fn is_generate(&self) -> bool {
        matches!(self, Self::Generate)
    }

    pub fn validate(&self) -> Result<(), ImageBenchError> {
        if let Self::Edit { image } = self {
            if image.path.as_os_str().is_empty() {
                return Err(invalid("edit reference image path must be nonempty"));
            }
            if image.encoded_sha256.as_ref().is_some_and(|digest| {
                digest.len() != 64
                    || !digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            }) {
                return Err(invalid("reference SHA-256 must be 64 lowercase hex digits"));
            }
        }
        Ok(())
    }
}
