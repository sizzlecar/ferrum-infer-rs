use super::*;
use ferrum_types::{
    ModelOutputProtocol, ResponseCompletionBoundary, ResponseFormat, StructuredOutputStart,
};

/// Necessary declaration-only checks from the original InstalledPlainText
/// capability. Success is not a live capability: actual admission, decoder,
/// policy, masks and prefix installation retain their full original checks.
pub(super) fn select(
    input: &[AutomaticCostProbeTemplate],
    maximum_retained: usize,
) -> Result<(Vec<AutomaticCostProbeTemplate>, Vec<usize>)> {
    let retained = input
        .iter()
        .try_fold(0usize, |n, t| n.checked_add(t.retained_payload_bytes()?))
        .ok_or_else(|| error("probe template retained payload overflow"))?;
    if retained > maximum_retained {
        return Err(error(
            "probe templates exceed shared retained capacity before decoding",
        ));
    }
    let mut selected = Vec::new();
    let mut excluded = Vec::new();
    for (index, template) in input.iter().enumerate() {
        let request = template.resolved_request()?;
        let p = &request.sampling_params;
        if p.model_output_protocol != ModelOutputProtocol::Text
            || !matches!(p.response_format, ResponseFormat::Text)
            || !matches!(p.structured_output_start, StructuredOutputStart::Immediate)
            || !matches!(
                p.response_completion_boundary,
                ResponseCompletionBoundary::Immediate
            )
        {
            excluded.push(index);
        } else {
            selected.push(template.clone());
        }
    }
    if selected.is_empty() {
        return Err(error(format!(
            "all {} actual product templates require an unsupported installed output boundary",
            excluded.len()
        )));
    }
    Ok((selected, excluded))
}

pub(super) fn supports_sampling(
    template: &AutomaticCostProbeTemplate,
    preset: SloAutomaticCostProbeSamplingPresetV1,
) -> Result<bool> {
    if !template.supports_preset(preset) {
        return Ok(false);
    }
    let (request, _) = template.instantiate(NonZeroUsize::new(3).unwrap(), 0, preset)?;
    let p = request.sampling_params;
    Ok(p.tfs.is_none() && p.typical_p.is_none() && p.mirostat.is_none())
}

/// A tokenizer-valid prefix is not necessarily in an installed truncated
/// distribution. Inspect the real installed processor recipe before declaring
/// a prepared cohort; actual masks and processed-logit checks still apply.
pub(super) fn prepared_prefix_unavailable(
    template: &AutomaticCostProbeTemplate,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    vocabulary: usize,
) -> Result<Option<&'static str>> {
    use ferrum_interfaces::sampler::{BuiltinLogitsProcessorCostV1, SamplingConfig};
    let (request, _) = template.instantiate(NonZeroUsize::new(3).unwrap(), 0, preset)?;
    let identity = SamplingConfig::from_params(&request.sampling_params)
        .cost_identity()
        .ok_or_else(|| error("probe installed sampling recipe unavailable"))?;
    Ok(identity
        .processors
        .iter()
        .find_map(|processor| match processor {
            BuiltinLogitsProcessorCostV1::TopK { k } if *k < vocabulary => {
                Some("installed_top_k_truncation")
            }
            BuiltinLogitsProcessorCostV1::TopP { .. } => Some("installed_top_p_truncation"),
            BuiltinLogitsProcessorCostV1::MinP { .. } => Some("installed_min_p_truncation"),
            _ => None,
        }))
}

/// Only predicts which original input opportunity to declare. The ordinary
/// SequenceState and installed host capability remain the live authority.
/// Mirrors model-side argmax selection, including its real repetition support.
pub(super) fn declared_greedy_sampling(
    template: &AutomaticCostProbeTemplate,
    preset: SloAutomaticCostProbeSamplingPresetV1,
) -> Result<bool> {
    let (request, _) = template.instantiate(NonZeroUsize::new(3).unwrap(), 0, preset)?;
    let p = request.sampling_params;
    Ok(p.temperature == 0.0
        && p.top_p == 1.0
        && p.top_k.is_none()
        && p.repetition_penalty.is_finite()
        && p.repetition_penalty > 0.0
        && p.presence_penalty == 0.0
        && p.frequency_penalty == 0.0
        && p.min_p.is_none()
        && p.tfs.is_none()
        && p.typical_p.is_none()
        && p.mirostat.is_none())
}
