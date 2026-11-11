//! Shared run/serve selection of the ConvRot GEMM output implementation.
#[cfg(any(feature = "cuda", test))]
use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageConvRotOutputPolicy {
    /// Use the locked fused implementation when available. Unsupported BF16
    /// geometries and all F32 outputs retain the separate epilogue.
    #[default]
    Auto,
    Separate,
    /// Require the native capability; geometry fallbacks remain explicit in
    /// each admitted provider implementation.
    CutlassFused,
}
impl PH3ImageConvRotOutputPolicy {
    /// Declared static native allowance inside the existing total ceiling.
    /// This is not a measurement of module size or process peak memory.
    pub fn additional_opaque_bytes(self) -> u64 {
        if cfg!(feature = "cuda-cutlass-int8-fused-output") && self != Self::Separate {
            64 << 20
        } else {
            0
        }
    }
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn resolve(self, available: bool) -> Result<bool> {
        match self {
            Self::Separate => Ok(false),
            Self::Auto => Ok(available),
            Self::CutlassFused => {
                ensure!(
                    available,
                    "fused ConvRot output requires its native CUDA capability"
                );
                Ok(true)
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_requires_capability_and_preserves_explicit_baseline() {
        assert!(!PH3ImageConvRotOutputPolicy::Auto.resolve(false).unwrap());
        assert!(PH3ImageConvRotOutputPolicy::Auto.resolve(true).unwrap());
        assert!(!PH3ImageConvRotOutputPolicy::Separate.resolve(true).unwrap());
        assert!(PH3ImageConvRotOutputPolicy::CutlassFused
            .resolve(false)
            .is_err());
        assert!(PH3ImageConvRotOutputPolicy::CutlassFused
            .resolve(true)
            .unwrap());
        assert_eq!(
            PH3ImageConvRotOutputPolicy::Separate.additional_opaque_bytes(),
            0
        );
        assert_eq!(
            PH3ImageConvRotOutputPolicy::Auto.additional_opaque_bytes(),
            if cfg!(feature = "cuda-cutlass-int8-fused-output") {
                64 << 20
            } else {
                0
            }
        );
    }
}
