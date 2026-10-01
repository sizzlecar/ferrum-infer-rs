//! Numeric-only future repetition does not construct a token history or an
//! executable LogitsReturnPolicy. The normal first wave retains real IDs.
use super::*;
use ferrum_interfaces::model_executor::TokenSelectionMask;

pub(super) enum ProjectedOutputRole<'a> {
    IntermediatePrefill,
    FinalPrefill,
    Decode(&'a LogitsReturnPolicy),
    Greedy {
        token_mask: Option<&'a TokenSelectionMask>,
        repetition_tokens: u64,
        repetition_penalty: f32,
    },
}
impl ProjectedOutputRole<'_> {
    pub fn requires_full_logits(&self) -> bool {
        match self {
            Self::FinalPrefill => true,
            Self::Decode(policy) => policy.requires_full_logits(),
            _ => false,
        }
    }
    pub fn repetition(&self, mode: VNextProductOutputMode) -> (u64, f32) {
        if mode != VNextProductOutputMode::GreedyToken {
            return (0, 1.);
        }
        match self {
            Self::Decode(policy) => {
                let repetition = product_repetition_input(Some(policy), mode);
                (repetition.token_ids.len() as u64, repetition.penalty)
            }
            Self::Greedy {
                repetition_tokens,
                repetition_penalty,
                ..
            } => (*repetition_tokens, *repetition_penalty),
            _ => (0, 1.),
        }
    }
    pub fn validate_real_ids(&self, mode: VNextProductOutputMode, vocabulary: usize) -> bool {
        match self {
            Self::Decode(policy) => product_repetition_input(Some(policy), mode)
                .token_ids
                .iter()
                .all(|id| usize::try_from(*id).is_ok_and(|id| id < vocabulary)),
            _ => true,
        }
    }
}
pub(super) fn output_mode(
    kind: VNextExecutionWaveKind,
    rows: &[ProjectedOutputRole<'_>],
) -> VNextProductOutputMode {
    if kind == VNextExecutionWaveKind::Prefill {
        return VNextProductOutputMode::FullLogits;
    }
    let mut has_decode = false;
    for row in rows {
        match row {
            ProjectedOutputRole::IntermediatePrefill if kind == VNextExecutionWaveKind::Mixed => {}
            ProjectedOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax { .. })
            | ProjectedOutputRole::Greedy { .. } => has_decode = true,
            _ => return VNextProductOutputMode::FullLogits,
        }
    }
    if has_decode {
        VNextProductOutputMode::GreedyToken
    } else {
        VNextProductOutputMode::FullLogits
    }
}
pub(super) fn mask_content(
    role: &ProjectedOutputRole<'_>,
    mode: VNextProductOutputMode,
    vocabulary_size: usize,
) -> std::result::Result<ProductTokenMaskContent, U> {
    let mask = if mode == VNextProductOutputMode::GreedyToken {
        match role {
            ProjectedOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
                token_mask, ..
            }) => token_mask.as_ref(),
            ProjectedOutputRole::Greedy { token_mask, .. } => *token_mask,
            _ => None,
        }
    } else {
        None
    };
    let vocabulary_size = u64::try_from(vocabulary_size).map_err(|_| U::Capacity)?;
    Ok(match mask {
        Some(mask) => ProductTokenMaskContent::selection(
            vocabulary_size,
            mask.fingerprint,
            &mask.valid_token_mask,
        ),
        None => ProductTokenMaskContent::AllValid { vocabulary_size },
    })
}
pub(super) fn validate_projected_repetition(
    host: Option<HostCostFeaturesV1>,
    count: u64,
    penalty: f32,
    vocabulary: usize,
    capacity: usize,
) -> std::result::Result<(), U> {
    let host = host
        .filter(HostCostFeaturesV1::supports_installed_plain_text_content)
        .ok_or(U::OutputBranch)?;
    let enabled = match host.policy.empirical_content_domain {
        Some(HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
            sampling: PlainTextSamplingRouteV2::Greedy { repetition_penalty },
            ..
        })) => repetition_penalty,
        Some(HostContentDomainV1::PlainTextGreedyV1) => false,
        _ => return Err(U::OutputBranch),
    };
    if !penalty.is_finite()
        || penalty <= 0.
        || count > vocabulary as u64
        || count > capacity as u64
        || count > host.state.sampling_history_tokens
        || count > u64::from(u32::MAX)
        || (count == 0 && penalty != 1.)
        || (!enabled && (count != 0 || penalty != 1.))
        || (enabled && host.state.sampling_history_tokens > 0 && (count == 0 || penalty == 1.))
    {
        return Err(U::InvalidInput);
    }
    Ok(())
}
