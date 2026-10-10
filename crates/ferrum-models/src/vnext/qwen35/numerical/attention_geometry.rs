//! Explicit opt-in attention arithmetic on the existing Q6-head/AA body.
//! Declaring the profile does not inherit output-quality or latency approval.
use super::*;
use ferrum_interfaces::vnext::UpstreamMarkerV2Profile;

#[cfg(test)]
mod tests;

pub(super) fn profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut result = q6_head::profile(master)?;
    result.id = NumericalProfileId::new(
        F32_MASTER_Q6_HEAD_ATTENTION_M8_MMQ_GEOMETRY_V1_NUMERICAL_PROFILE_ID,
    )
    .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    for (old, new) in [
        (
            UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
            UpstreamMarkerV2Profile::GatedDeltaM8Geometry,
        ),
        (
            UpstreamMarkerV2Profile::CausalExtraAllRows,
            UpstreamMarkerV2Profile::CausalM8Geometry,
        ),
    ] {
        if let Some(operation) = result
            .operations
            .iter_mut()
            .find(|op| op.operation_id.as_str() == old.operation_id())
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
