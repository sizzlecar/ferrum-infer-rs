//! Source5 has its own strict identity and preparation lifecycle. Only the
//! ordinary numerical replay machinery is shared with source4.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixPlanV5;
mod wire;
use wire::*;
mod lifecycle;
mod profile;
mod stages;
pub(super) use lifecycle::{Preparation, Progress};
pub use profile::{
    export_structured_profile_v12, load_structured_profile_v12, ImportedStructuredCatalogV12,
    StructuredProfileExportReceiptV12,
};

pub fn structured_prefix_source_header_v5(
    children: Vec<serde_json::Value>,
    maximum_file_bytes: u64,
    maximum_children: usize,
    maximum_retained_numeric_bytes: usize,
    maximum_retained_coordinates: usize,
    prefix_plan: StructuredPrefixPlanV5,
) -> Result<serde_json::Value, CostProfileError> {
    let common = shared_header(
        children,
        maximum_file_bytes,
        maximum_children,
        maximum_retained_numeric_bytes,
        maximum_retained_coordinates,
    )?;
    let prefix_plan_sha256 = prefix_plan
        .signature(&common.common.cohort_plan)
        .map_err(numeric_error)?;
    let mut h = HeaderV5::new(common, prefix_plan, prefix_plan_sha256);
    h.capture_protocol = h.signature()?;
    let value = serde_json::to_value(h)?;
    checked_header_record_bytes(&value, maximum_file_bytes)?;
    Ok(value)
}

pub(super) fn header(
    value: serde_json::Value,
) -> Result<(HeaderV4, StructuredPrefixPlanV5), CostProfileError> {
    let h: HeaderV5 = serde_json::from_value(value)?;
    if h.artifact_type != "ferrum.structured-prefix-live-source"
        || h.schema_version != 5
        || h.model_revision != MODEL_REVISION_V2
        || h.capture_protocol != h.signature()?
        || h.prefix_plan
            .signature(&h.common.cohort_plan)
            .map_err(numeric_error)?
            != h.prefix_plan_sha256
    {
        return Err(invalid("invalid source5 identity or prefix declaration"));
    }
    Ok(h.into_parts())
}

pub(super) fn is_preparation(value: &serde_json::Value) -> bool {
    matches!(
        value["kind"].as_str(),
        Some("preparation_offered" | "preparation_completed" | "preparation_released")
    )
}

#[cfg(test)]
mod tests;
