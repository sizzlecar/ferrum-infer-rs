//! Convolution algorithm selection bounds; the chosen algorithm's actual
//! workspace is admitted with the rest of its compiled execution plan.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageCudaConvolutionWorkspacePolicy {
    /// Preserve the explicit cuDNN workspace limit, including a zero limit.
    #[default]
    ConfiguredLimit,
    /// Derive a selection ceiling from device storage remaining after declared
    /// opaque allowances and the largest component's execution weights.
    Auto,
}

#[cfg(any(feature = "cuda", test))]
impl ImageCudaConvolutionWorkspacePolicy {
    pub(crate) fn resolve(
        self,
        configured_limit: u64,
        device_capacity: u64,
        opaque_bytes: u64,
        largest_execution_weights: u64,
    ) -> anyhow::Result<u64> {
        use anyhow::Context;
        match self {
            Self::ConfiguredLimit => Ok(configured_limit),
            Self::Auto => device_capacity
                .checked_sub(opaque_bytes)
                .and_then(|storage| storage.checked_sub(largest_execution_weights))
                .filter(|&remaining| remaining > 0)
                .map(|remaining| remaining.min(isize::MAX as u64))
                .context("CUDA convolution has no storage headroom after opaque allowances and component weights"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ImageCudaConvolutionWorkspacePolicy as Policy;

    #[test]
    fn selection_bound_preserves_declared_storage_and_explicit_limits() {
        assert_eq!(Policy::Auto.resolve(16, 1000, 100, 600).unwrap(), 300);
        assert!(Policy::Auto.resolve(16, 700, 100, 600).is_err());
        assert!(Policy::Auto.resolve(16, 99, 100, 600).is_err());
        assert_eq!(
            Policy::ConfiguredLimit.resolve(0, 1000, 100, 600).unwrap(),
            0
        );
        assert_eq!(
            Policy::ConfiguredLimit.resolve(16, 1000, 100, 600).unwrap(),
            16
        );
    }
}
