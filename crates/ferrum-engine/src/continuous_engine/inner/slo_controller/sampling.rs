//! Shared eligibility before any sampling-history vector is materialized.
//! This reads the installed identity; it never grants prediction or execution.
use super::*;
use crate::continuous_engine::SequenceSamplingHistoryScope;
use ferrum_interfaces::execution_cost::{HostContentDomainV1, PlainTextSamplingRouteV2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner) enum SamplingCapability {
    /// Preserve the existing exact first-wave path. Changed future content
    /// still needs an installed plain-text capability and qualified coverage.
    ExactOnly,
    InstalledPlainText(PlainTextSamplingRouteV2),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner) enum SamplingUnavailable {
    MissingIdentity,
    UnsupportedRoute,
    HistoryUnavailable,
}
impl SamplingUnavailable {
    pub(in crate::continuous_engine::inner) fn label(self) -> &'static str {
        match self {
            Self::MissingIdentity => "host_policy_unknown",
            Self::UnsupportedRoute => "sampling_route_unsupported",
            Self::HistoryUnavailable => "sampling_history_unavailable",
        }
    }
}

/// No decoder, policy hashing, token-history cloning or allocation. Reading
/// full-generation host features cannot enter the fallible sliced-history path.
pub(in crate::continuous_engine::inner) fn capability(
    sequence: &SequenceState,
) -> std::result::Result<SamplingCapability, SamplingUnavailable> {
    if sequence
        .cost_policy_signature
        .is_none_or(|identity| identity == [0; 32])
    {
        return Err(SamplingUnavailable::MissingIdentity);
    }
    let penalty = sequence.sampling_params.repetition_penalty;
    if !penalty.is_finite() || penalty <= 0.0 {
        return Err(SamplingUnavailable::UnsupportedRoute);
    }
    let installed = matches!(
        sequence.sampling_history.scope(),
        SequenceSamplingHistoryScope::FullGeneration
    )
    .then(|| super::super::cost_observation::participant_host_features(sequence))
    .flatten()
    .filter(|host| {
        host.policy.categorical_signature != [0; 32] && host.supports_installed_plain_text_content()
    });
    let route = installed.and_then(|host| match host.policy.empirical_content_domain {
        Some(HostContentDomainV1::PlainTextGreedyV1) => Some(PlainTextSamplingRouteV2::Greedy {
            repetition_penalty: false,
        }),
        Some(HostContentDomainV1::PlainTextInstalledV2(policy)) => Some(policy.sampling),
        None => None,
    });
    match route {
        Some(PlainTextSamplingRouteV2::Greedy { repetition_penalty })
            if repetition_penalty != (penalty != 1.0) =>
        {
            Err(SamplingUnavailable::UnsupportedRoute)
        }
        Some(route) => Ok(SamplingCapability::InstalledPlainText(route)),
        None if penalty == 1.0 => Ok(SamplingCapability::ExactOnly),
        None => Err(SamplingUnavailable::UnsupportedRoute),
    }
}

pub(in crate::continuous_engine::inner) struct FutureSamplingPolicy {
    pub policy: ferrum_interfaces::model_executor::LogitsReturnPolicy,
    pub repetition: Option<(u64, f32, u64)>,
}

/// The installed future policy, shared by controller snapshots and inventory.
pub(in crate::continuous_engine::inner) fn future_policy(
    sequence: &SequenceState,
    actual: &ferrum_interfaces::model_executor::LogitsReturnPolicy,
    vocabulary: u64,
) -> std::result::Result<Option<FutureSamplingPolicy>, SamplingUnavailable> {
    use ferrum_interfaces::model_executor::LogitsReturnPolicy;
    let mut repetition = None;
    let policy = match capability(sequence)? {
        SamplingCapability::ExactOnly => return Ok(None),
        SamplingCapability::InstalledPlainText(PlainTextSamplingRouteV2::FullLogits) => {
            LogitsReturnPolicy::FullLogits
        }
        SamplingCapability::InstalledPlainText(PlainTextSamplingRouteV2::Greedy {
            repetition_penalty: penalty_enabled,
        }) => {
            let repetition_penalty = match actual {
                LogitsReturnPolicy::GreedyArgmax {
                    repetition_penalty, ..
                } => repetition_penalty.clone(),
                _ => sequence.model_decode_repetition_penalty(),
            };
            if penalty_enabled {
                let unique = repetition_penalty
                    .as_ref()
                    .map_or(0, |r| r.token_ids().len()) as u64;
                ferrum_interfaces::vnext::FutureRepetitionRangeV3::from_observed_history(
                    unique,
                    sequence.generated_tokens.len() as u64,
                    sequence.generated_tokens.len() as u64,
                    vocabulary,
                )
                .map_err(|_| SamplingUnavailable::HistoryUnavailable)?;
                repetition = Some((
                    unique,
                    sequence.sampling_params.repetition_penalty,
                    vocabulary,
                ));
            }
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: sequence.argmax_token_mask.clone(),
                repetition_penalty,
            }
        }
    };
    Ok(Some(FutureSamplingPolicy { policy, repetition }))
}
