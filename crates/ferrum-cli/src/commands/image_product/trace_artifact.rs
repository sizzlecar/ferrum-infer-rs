//! An explicitly configured diagnostic sink. It is opened before native
//! construction, never overwrites a file, and consumes the original bounded
//! engine collector without a second payload buffer or inferred retirement.
#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
use anyhow::Context;
use anyhow::{ensure, Result};
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use ferrum_engine::image_generation::continuous_trace::ImageContinuousTraceHandle;
use ferrum_engine::image_generation::continuous_trace::ImageContinuousTraceLimits;
#[cfg(feature = "cuda")]
use ferrum_engine::image_generation::p_h3::PH3ImagePromptExpansionDiagnosticPolicy;
use serde::Deserialize;
#[cfg(any(test, feature = "cuda"))]
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
use std::{fs::File, io::Write};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageContinuousTraceConfig {
    pub path: PathBuf,
    pub maximum_bytes: usize,
}
impl PH3ImageContinuousTraceConfig {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            !self.path.as_os_str().is_empty(),
            "continuous trace path must be nonempty"
        );
        self.limits().validate()
    }
    pub(super) fn resolve_path(&mut self, base: &Path) {
        if self.path.is_relative() {
            self.path = base.join(&self.path);
        }
    }
    fn limits(&self) -> ImageContinuousTraceLimits {
        ImageContinuousTraceLimits {
            maximum_bytes: self.maximum_bytes,
        }
    }
}

#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) struct PreparedTraceArtifact {
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    path: PathBuf,
    file: File,
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    limits: ImageContinuousTraceLimits,
    #[cfg(feature = "cuda")]
    prompt_expansion_diagnostics: PH3ImagePromptExpansionDiagnosticPolicy,
}
#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
impl PreparedTraceArtifact {
    #[cfg(any(test, feature = "cuda"))]
    pub(super) fn open(config: &PH3ImageContinuousTraceConfig) -> Result<Self> {
        config.validate()?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&config.path)
            .with_context(|| format!("create new continuous trace {}", config.path.display()))?;
        Ok(Self {
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            path: config.path.clone(),
            file,
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            limits: config.limits(),
            #[cfg(feature = "cuda")]
            prompt_expansion_diagnostics: PH3ImagePromptExpansionDiagnosticPolicy::Disabled,
        })
    }
    #[cfg(feature = "cuda")]
    pub(super) fn with_prompt_expansion_diagnostics(
        mut self,
        policy: PH3ImagePromptExpansionDiagnosticPolicy,
    ) -> Self {
        self.prompt_expansion_diagnostics = policy;
        self
    }
    #[cfg(feature = "cuda")]
    pub(super) fn prompt_expansion_diagnostics(&self) -> PH3ImagePromptExpansionDiagnosticPolicy {
        self.prompt_expansion_diagnostics
    }
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    pub(super) fn limits(&self) -> ImageContinuousTraceLimits {
        self.limits
    }
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    pub(super) fn bind(self, handle: ImageContinuousTraceHandle) -> ActiveTraceArtifact {
        ActiveTraceArtifact {
            path: self.path,
            file: self.file,
            handle,
        }
    }
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) struct ActiveTraceArtifact {
    path: PathBuf,
    file: File,
    handle: ImageContinuousTraceHandle,
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
impl ActiveTraceArtifact {
    /// The caller has already obtained its original inference/shutdown result.
    /// Diagnostic I/O cannot replace that result or consume its recovery owner.
    /// An unsealed footer remains unsealed; exporting never creates an ACK.
    pub(super) fn export_best_effort(mut self) {
        let status = self.handle.status();
        match write_snapshot(&mut self.file, |file| self.handle.write_jsonl(file)) {
            Ok(status) => tracing::info!(
                path=%self.path.display(), session_uuid=%status.session_uuid,
                sealed=status.sealed, truncated=status.truncated,
                recorded=status.recorded, dropped=status.dropped,
                used_bytes=status.used_bytes, maximum_bytes=status.maximum_bytes,
                "continuous image trace exported"
            ),
            Err(error) => tracing::warn!(
                path=%self.path.display(), session_uuid=%status.session_uuid,
                sealed=status.sealed, truncated=status.truncated, error=%error,
                "continuous image trace export failed; inference result is unchanged"
            ),
        }
        // The handle retains real Host Output backing but no native/runtime
        // owner. Release it before callers inspect final Host refunds. A failed
        // pipeline's original recorder may correctly remain with its recovery.
    }
}

#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn write_snapshot<T>(file: &mut File, write: impl FnOnce(&mut File) -> Result<T>) -> Result<T> {
    let result = write(file)?;
    file.flush().context("flush continuous image trace")?;
    file.sync_all().context("sync continuous image trace")?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_trace_sink_is_preopened_once_without_overwriting_an_existing_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let config = PH3ImageContinuousTraceConfig {
            path: directory.path().join("trace.jsonl"),
            maximum_bytes: 256,
        };
        let mut prepared = PreparedTraceArtifact::open(&config).unwrap();
        assert!(config.path.is_file());
        assert!(PreparedTraceArtifact::open(&config).is_err());
        write_snapshot(&mut prepared.file, |file| {
            file.write_all(b"{\"event\":\"bounded test prefix\"}\n")?;
            Ok(())
        })
        .unwrap();
        let original = std::fs::read(&config.path).unwrap();
        assert!(PreparedTraceArtifact::open(&config).is_err());
        assert_eq!(std::fs::read(&config.path).unwrap(), original);
    }

    #[test]
    fn continuous_trace_export_surfaces_real_sink_errors_without_replacing_existing_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("read-only.jsonl");
        std::fs::write(&path, b"original\n").unwrap();
        let mut file = File::open(&path).unwrap();
        let result = write_snapshot(&mut file, |file| {
            file.write_all(b"replacement\n")?;
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"original\n");
    }
}
