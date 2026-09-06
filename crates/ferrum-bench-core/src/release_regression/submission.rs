//! Narrow planning and fixture inputs for legacy backend submission checks.
//! These descriptors do not certify quantized operators or model performance.
use super::{Backend, Behavior, CheckDescriptor, EvidenceLayer, ExecutionTarget, ObligationScope};
use serde::{Deserialize, Serialize};

pub const SUBMISSION_CHECK_ID: &str = "backend-submission.metal-legacy-context";
pub const LEGACY_EXECUTION_PATH: &str = "legacy-model-executor";
pub const PROBE_EXECUTION_PATH: &str =
    "MetalContext::submit_and_wait (Backend F32 compute/blit/compute)";
pub const MAX_SUBMISSION_NMSE: f64 = 1e-7;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubmissionConfig {
    pub tokens: usize,
    pub intermediate: usize,
    pub k: usize,
    pub seed: u64,
    pub max_nmse: f64,
}

impl SubmissionConfig {
    /// Arithmetic bounds of the registered F32 GEMM/blit/SiLU fixture. This
    /// validates its inputs before reading a report; it does not claim memory
    /// availability or successful device execution.
    pub fn segment_len(&self) -> Result<usize, String> {
        if self.tokens == 0 || self.intermediate == 0 || self.k == 0 {
            return Err("submission dimensions must be nonzero".into());
        }
        if !self.max_nmse.is_finite() || self.max_nmse <= 0.0 || self.max_nmse > MAX_SUBMISSION_NMSE
        {
            return Err("submission tolerance must be positive and at most 1e-7".into());
        }
        let width = self
            .intermediate
            .checked_mul(2)
            .ok_or("submission width overflow")?;
        for (rows, cols) in [(self.tokens, self.k), (width, self.k), (self.tokens, width)] {
            let elements = rows
                .checked_mul(cols)
                .ok_or("submission matrix size overflow")?;
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or("submission matrix byte size overflow")?;
            if elements > i32::MAX as usize || bytes > isize::MAX as usize {
                return Err("submission matrix exceeds kernel or host indexing range".into());
            }
        }
        // The existing Metal GEMM launch pads these dimensions before signed
        // indexing; GEMV uses the same K range even when tokens == 1.
        for (dimension, tile) in [(self.tokens, 64), (width, 32), (self.k, 32)] {
            if dimension
                .div_ceil(tile)
                .checked_mul(tile)
                .is_none_or(|n| n > i32::MAX as usize)
            {
                return Err("submission tile exceeds signed kernel indexing range".into());
            }
        }
        let segment = self
            .tokens
            .checked_mul(self.intermediate)
            .ok_or("submission segment size overflow")?;
        let total = segment
            .checked_mul(super::numerics::SUBMISSION_PHASES.len())
            .ok_or("submission output size overflow")?;
        if total
            .checked_mul(std::mem::size_of::<f32>())
            .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err("submission output exceeds host indexing range".into());
        }
        Ok(segment)
    }

    pub fn kernel_entrypoint(&self) -> &'static str {
        if self.tokens == 1 {
            "gemv_f32 -> blit -> silu_mul_split_f32 (legacy MetalContext)"
        } else {
            "gemm_f32_v2 -> blit -> silu_mul_split_f32 (legacy MetalContext)"
        }
    }
}

pub fn submission_scope() -> ObligationScope {
    ObligationScope::ExecutionPath {
        backend: Backend::Metal,
        execution_path: LEGACY_EXECUTION_PATH.into(),
    }
}

/// Register only an execution route present in the required inventory. The
/// target anchors the descriptor to that route; its weight precision does not
/// expand the F32 submission fixture into a quantized-kernel correctness claim.
pub fn submission_check_descriptors(required_targets: &[ExecutionTarget]) -> Vec<CheckDescriptor> {
    let target = required_targets
        .iter()
        .filter(|target| {
            target.backend == Backend::Metal && target.execution_path == LEGACY_EXECUTION_PATH
        })
        .min_by(|a, b| {
            a.architecture
                .cmp(&b.architecture)
                .then_with(|| a.precision.cmp(&b.precision))
                .then_with(|| format!("{:?}", a.protocol).cmp(&format!("{:?}", b.protocol)))
        });
    target
        .map(|target| CheckDescriptor {
            id: SUBMISSION_CHECK_ID.into(),
            behavior: Behavior::SubmissionCompletion,
            layer: EvidenceLayer::BackendNumerics,
            entrypoints: Vec::new(),
            target: Some(target.clone()),
        })
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn target(backend: Backend, path: &str, precision: &str) -> ExecutionTarget {
        ExecutionTarget {
            architecture: "dense".into(),
            protocol: ferrum_types::ModelOutputProtocol::Text,
            precision: precision.into(),
            backend,
            execution_path: path.into(),
        }
    }
    #[test]
    fn descriptor_requires_declared_metal_legacy_route_and_cannot_cover_kernel_numerics() {
        let cuda = target(Backend::Cuda, LEGACY_EXECUTION_PATH, "safetensors-bf16");
        let vnext = target(Backend::Metal, "production-plan-runtime", "gguf-q4_k_m");
        assert!(submission_check_descriptors(&[]).is_empty());
        assert!(submission_check_descriptors(&[cuda.clone(), vnext.clone()]).is_empty());
        let legacy = target(Backend::Metal, LEGACY_EXECUTION_PATH, "gguf-q4_k_m");
        let descriptors = submission_check_descriptors(&[cuda, vnext, legacy.clone()]);
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].target, Some(legacy));
        assert_eq!(descriptors[0].behavior, Behavior::SubmissionCompletion);
        assert_eq!(descriptors[0].layer, EvidenceLayer::BackendNumerics);
        assert!(descriptors[0].entrypoints.is_empty());
        assert_eq!(
            submission_scope(),
            ObligationScope::ExecutionPath {
                backend: Backend::Metal,
                execution_path: LEGACY_EXECUTION_PATH.into()
            }
        );
    }
    #[test]
    fn shared_route_registration_does_not_expand_per_precision() {
        let first = target(Backend::Metal, LEGACY_EXECUTION_PATH, "gguf-q4_k_m");
        let second = target(Backend::Metal, LEGACY_EXECUTION_PATH, "gguf-q8_0");
        assert_eq!(
            submission_check_descriptors(&[first.clone(), second.clone()]),
            submission_check_descriptors(&[second, first])
        );
    }
    #[test]
    fn submission_shape_is_checked_without_fixing_sample_dimensions() {
        for (tokens, intermediate, k) in [(1, 3, 5), (3, 33, 35), (4, 64, 32)] {
            let config = SubmissionConfig {
                tokens,
                intermediate,
                k,
                seed: 7,
                max_nmse: 1e-7,
            };
            assert_eq!(config.segment_len().unwrap(), tokens * intermediate);
        }
        for (tokens, intermediate, k, max_nmse) in [
            (0, 3, 5, 1e-7),
            (1, 0, 5, 1e-7),
            (1, 3, 0, 1e-7),
            (usize::MAX, 3, 5, 1e-7),
            (1, usize::MAX, 5, 1e-7),
            (1, 3, usize::MAX, 1e-7),
            (1, 3, i32::MAX as usize, 1e-7),
            (1, 3, 5, 1e-6),
            (1, 3, 5, 0.0),
            (1, 3, 5, f64::NAN),
        ] {
            assert!(SubmissionConfig {
                tokens,
                intermediate,
                k,
                seed: 7,
                max_nmse
            }
            .segment_len()
            .is_err());
        }
    }
}
