//! Backend-independent wire identity for the CUDA attention choice. Merely
//! parsing this policy does not construct a CUDA device or load cuDNN.

/// Explicit attention implementation selection. The experimental SDPA profile
/// changes DiT numerics and requires target validation; it is not a throughput
/// or numerical qualification result. Omitted legacy fields remain sequential.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageCudaAttentionPolicy {
    #[default]
    Sequential,
    CudnnSdpaExperimentalV1,
}

impl ImageCudaAttentionPolicy {
    /// Additional declared native-plan/variant-pack allowance. This is an
    /// experimental capacity reservation, not a measured native-memory bound.
    pub const fn additional_opaque_bytes(self) -> u64 {
        match self {
            Self::Sequential => 0,
            Self::CudnnSdpaExperimentalV1 => 128 * 1024 * 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_policy_wire_is_strict_and_default_preserves_old_path() {
        assert_eq!(
            ImageCudaAttentionPolicy::default(),
            ImageCudaAttentionPolicy::Sequential
        );
        for (name, expected) in [
            ("sequential", ImageCudaAttentionPolicy::Sequential),
            (
                "cudnn_sdpa_experimental_v1",
                ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1,
            ),
        ] {
            let value = serde_json::Value::String(name.into());
            assert_eq!(
                serde_json::from_value::<ImageCudaAttentionPolicy>(value.clone()).unwrap(),
                expected
            );
            assert_eq!(serde_json::to_value(expected).unwrap(), value);
        }
        for invalid in [
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!("automatic"),
        ] {
            assert!(serde_json::from_value::<ImageCudaAttentionPolicy>(invalid).is_err());
        }
    }
}
