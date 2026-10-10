//! Explicit Q6 F16 projection arithmetic with physical-leaf geometry selection.
use super::*;
use ferrum_interfaces::vnext::UpstreamMarkerV2Profile;

#[cfg(test)]
mod tests;

pub(super) fn profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut result = attention_geometry::profile(master)?;
    result.id =
        NumericalProfileId::new(F32_MASTER_Q6_HEAD_ATTENTION_M8_Q6_F16_MMQ_V1_NUMERICAL_PROFILE_ID)
            .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    for (old, new) in [
        (
            UpstreamMarkerV2Profile::SwiGluExtraAllRows,
            UpstreamMarkerV2Profile::SwiGluQ6F16,
        ),
        (
            UpstreamMarkerV2Profile::GatedDeltaM8Geometry,
            UpstreamMarkerV2Profile::GatedDeltaQ6F16,
        ),
        (
            UpstreamMarkerV2Profile::CausalM8Geometry,
            UpstreamMarkerV2Profile::CausalQ6F16,
        ),
    ] {
        if let Some(operation) = result
            .operations
            .iter_mut()
            .find(|operation| operation.operation_id.as_str() == old.operation_id())
        {
            operation.operation_id = operation_id(new.operation_id())?;
            operation.composite_arithmetic = Some(new.arithmetic());
        }
    }
    result
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    result.validate()?;
    Ok(result)
}
