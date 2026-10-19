//! Captured launch geometry, independent of model or cost-estimator policy.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PagedAttentionLaunchGeometry {
    SinglePass {
        score_capacity: usize,
    },
    SplitK {
        num_splits: usize,
        score_capacity: usize,
    },
}

impl PagedAttentionLaunchGeometry {
    /// The CUDA route/launch calculation used by both preparation and dispatch.
    /// `score_capacity` is the per-CTA score extent, not an observed KV frontier.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn cuda_varlen(
        policy: Option<bool>,
        num_seqs: usize,
        total_q_tokens: usize,
        max_kv_len: usize,
    ) -> Self {
        let split = policy.unwrap_or(total_q_tokens <= 64 && (num_seqs <= 4 || max_kv_len >= 768));
        if split {
            let num_splits = match max_kv_len {
                0..=384 => 2,
                385..=1024 => 4,
                1025..=2048 => 8,
                _ => 16,
            };
            Self::SplitK {
                num_splits,
                score_capacity: max_kv_len.div_ceil(num_splits).max(1),
            }
        } else {
            Self::SinglePass {
                score_capacity: super::attention_score_capacity_bucket(max_kv_len),
            }
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn shared_bytes(self) -> ferrum_types::Result<u32> {
        let capacity = match self {
            Self::SinglePass { score_capacity } | Self::SplitK { score_capacity, .. } => {
                score_capacity
            }
        };
        capacity
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| {
                ferrum_types::FerrumError::model("paged attention shared-memory extent overflow")
            })
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn split_scratch(
        self,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) -> ferrum_types::Result<Option<(usize, usize)>> {
        let Self::SplitK { num_splits, .. } = self else {
            return Ok(None);
        };
        let ml = tokens
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(num_splits))
            .ok_or_else(|| ferrum_types::FerrumError::model("split-K scratch extent overflow"))?;
        let out = ml
            .checked_mul(dim)
            .ok_or_else(|| ferrum_types::FerrumError::model("split-K output extent overflow"))?;
        Ok(Some((out, ml)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_geometry_controls_route_chunk_and_scratch() {
        let auto = PagedAttentionLaunchGeometry::cuda_varlen(None, 5, 5, 5);
        assert_eq!(
            auto,
            PagedAttentionLaunchGeometry::cuda_varlen(None, 5, 5, 7)
        );
        let bucket = PagedAttentionLaunchGeometry::cuda_varlen(None, 16, 16, 129);
        assert_eq!(
            bucket,
            PagedAttentionLaunchGeometry::cuda_varlen(None, 16, 16, 256)
        );
        assert_ne!(
            bucket,
            PagedAttentionLaunchGeometry::cuda_varlen(None, 16, 16, 257)
        );
        let five = PagedAttentionLaunchGeometry::cuda_varlen(Some(true), 5, 5, 5);
        let seven = PagedAttentionLaunchGeometry::cuda_varlen(Some(true), 5, 5, 7);
        assert_ne!(five, seven);
        assert_eq!(five.shared_bytes().unwrap(), 12);
        assert_eq!(seven.shared_bytes().unwrap(), 16);
        let before = PagedAttentionLaunchGeometry::cuda_varlen(None, 1, 1, 384);
        let after = PagedAttentionLaunchGeometry::cuda_varlen(None, 1, 1, 385);
        assert_eq!(before.split_scratch(1, 2, 64).unwrap(), Some((256, 4)));
        assert_eq!(after.split_scratch(1, 2, 64).unwrap(), Some((512, 8)));
        assert_ne!(before, after);
        assert!(after.split_scratch(usize::MAX, 2, 64).is_err());
        assert!(PagedAttentionLaunchGeometry::SinglePass {
            score_capacity: usize::MAX
        }
        .shared_bytes()
        .is_err());
    }
}
