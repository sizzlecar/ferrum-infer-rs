//! A live collector retains the immutable process producer identity. Product
//! runtimes prepare it before starting feedback consumption or accepting work.
use super::super::profile_export::ProducerIdentity;
use ferrum_types::FerrumError;
use std::sync::OnceLock;

#[derive(Default)]
pub(super) struct ProducerIdentityCache {
    identity: OnceLock<Result<ProducerIdentity, String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_producer_identity_preparation_failure_is_optional_and_sticky() {
        let cache = ProducerIdentityCache::default();
        cache
            .identity
            .set(Err("original executable read failed".to_owned()))
            .unwrap();
        // Runtime cold setup invokes this infallible preparation before its
        // worker is exposed. It cannot reject an imported inference snapshot.
        let original = cache.current().unwrap_err().to_string();
        cache.prepare();
        assert_eq!(cache.current().unwrap_err().to_string(), original);
        cache.prepare();
        assert!(
            matches!(cache.identity.get(), Some(Err(error)) if error == "original executable read failed")
        );
    }
}

impl ProducerIdentityCache {
    /// Every collector and generation in this process has the same executable origin.
    /// Preserve a failed read too: repeatedly hashing a changing/unreadable
    /// executable cannot supply a stable identity for this collector.
    pub(super) fn prepare(&self) {
        // Optional source metadata failure retains the original failed Result;
        // it does not reject construction of a runtime with usable imported costs.
        let _ = self.current();
    }

    pub fn current(&self) -> Result<&ProducerIdentity, FerrumError> {
        self.identity
            .get_or_init(|| {
                #[cfg(test)]
                let started = std::time::Instant::now();
                let result = ProducerIdentity::current().map_err(|error| error.to_string());
                #[cfg(test)]
                eprintln!("original producer identity initialization: elapsed_ns={} executable_bytes={:?} result={}", started.elapsed().as_nanos(), result.as_ref().ok().map(|value| value.executable_bytes), if result.is_ok() { "ready" } else { "failed" });
                result
            })
            .as_ref()
            .map_err(|error| {
                FerrumError::config(format!("live calibration producer identity: {error}"))
            })
    }
}
