//! Discovery facts from the original typed receipt. Observed pending/Length
//! patterns do not authorize any future branch or enlarge a frozen population.
use ferrum_interfaces::execution_cost::StructuredHostRowV1;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredInputV2, StructuredOwnerKeyV2, StructuredUnknownV2, MODEL_REVISION_V2,
};
use ferrum_types::SloStructuredCostCapture;
use serde::{Serialize, Serializer};
mod summary;
pub(in crate::commands::calibrate_slo) use summary::DiscoverySummaryV2;
#[derive(Serialize)]
pub(in crate::commands::calibrate_slo) struct DiscoveryReportV2 {
    model_revision: &'static str,
    #[serde(flatten)]
    input: DiscoveryInputV2,
}
#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum DiscoveryInputV2 {
    Known {
        owner: StructuredOwnerKeyV2,
        domain_signature: [u8; 32],
        basis: Vec<f64>,
        support: Vec<u64>,
        physical_host_rows: Vec<StructuredHostRowV1>,
    },
    Unknown {
        #[serde(serialize_with = "unknown")]
        reason: StructuredUnknownV2,
    },
}
fn unknown<S: Serializer>(reason: &StructuredUnknownV2, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&format_args!("{reason:?}"))
}
pub(super) fn inspect(
    mode: SloStructuredCostCapture,
    input: impl FnOnce() -> Result<StructuredInputV2, StructuredUnknownV2>,
) -> Option<DiscoveryReportV2> {
    match mode {
        SloStructuredCostCapture::Disabled => None,
        SloStructuredCostCapture::HostSettledV1 => Some(DiscoveryReportV2 {
            model_revision: MODEL_REVISION_V2,
            input: match input() {
                Ok(input) => DiscoveryInputV2::Known {
                    owner: input.owner().clone(),
                    domain_signature: *input.domain_signature(),
                    basis: input.regression_axes().to_vec(),
                    support: input.joint_support_coordinates().to_vec(),
                    physical_host_rows: input.physical_host_rows().to_vec(),
                },
                Err(reason) => DiscoveryInputV2::Unknown { reason },
            },
        }),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_v2_discovery_disabled_is_lazy_and_unknown_is_typed() {
        assert!(inspect(SloStructuredCostCapture::Disabled, || panic!(
            "disabled capture cannot inspect typed input"
        ))
        .is_none());
        let value = inspect(SloStructuredCostCapture::HostSettledV1, || {
            Err(StructuredUnknownV2::MissingEvidence)
        })
        .unwrap();
        let wire = serde_json::to_value(value).unwrap();
        assert_eq!(wire["status"], "unknown");
        assert_eq!(wire["reason"], "MissingEvidence");
        assert!(wire.get("owner").is_none());
        assert!(wire.get("basis").is_none());
    }
}
