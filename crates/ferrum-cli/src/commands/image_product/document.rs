//! Bounded architecture dispatch for explicit product documents. Legacy Qwen
//! documents retain their original schema; H3 must opt into its typed schema.
use super::{p_h3::PH3ImageProductConfig, ImageProductConfig};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

pub(super) enum ImageProductDocument {
    QwenImage21(ImageProductConfig),
    PH3Image(PH3ImageProductConfig),
}
#[derive(Deserialize)]
struct ArchitectureDeclaration {
    architecture: Option<super::p_h3::PH3ImageProductArchitecture>,
}
impl ImageProductDocument {
    pub(super) fn load(path: &Path) -> Result<Self> {
        let path = std::fs::canonicalize(path)
            .with_context(|| format!("resolve image product document {}", path.display()))?;
        let bytes = super::source::read_bounded(&path, 1 << 20)?;
        // Parse the discriminator and the chosen schema from the same bounded
        // immutable bytes. A malformed H3 document never falls back to Qwen.
        let declaration: ArchitectureDeclaration = serde_json::from_slice(&bytes)
            .context("invalid image product architecture declaration")?;
        match declaration.architecture {
            Some(super::p_h3::PH3ImageProductArchitecture::PH3Image) => Ok(Self::PH3Image(
                PH3ImageProductConfig::from_bytes(&path, &bytes)?,
            )),
            None => Ok(Self::QwenImage21(ImageProductConfig::from_bytes(
                &path, &bytes,
            )?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_architecture_rejects_unknown_and_incomplete_documents_without_fallback() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("pipeline.json");
        std::fs::write(
            &path,
            br#"{"architecture":"unknown","model":"public-alias"}"#,
        )
        .unwrap();
        let error = format!("{:#}", ImageProductDocument::load(&path).err().unwrap());
        assert!(error.contains("architecture declaration"), "{error}");
        std::fs::write(
            &path,
            br#"{"architecture":"p_h3_image","model":"public-alias"}"#,
        )
        .unwrap();
        let error = format!("{:#}", ImageProductDocument::load(&path).err().unwrap());
        assert!(error.contains("typed H3 product document"), "{error}");
        assert!(!error.contains("execution_profile"), "{error}");
        std::fs::write(&path, br#"{"model":"public-alias"}"#).unwrap();
        let error = format!("{:#}", ImageProductDocument::load(&path).err().unwrap());
        assert!(error.contains("execution_profile"), "{error}");
        std::fs::write(
            &path,
            br#"{"architecture":"p_h3_image","architecture":"p_h3_image"}"#,
        )
        .unwrap();
        let error = format!("{:#}", ImageProductDocument::load(&path).err().unwrap());
        assert!(error.contains("duplicate field"), "{error}");
    }
}
