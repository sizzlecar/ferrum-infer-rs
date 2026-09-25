//! Bounded read-only input diagnostics, not an importable training artifact.
use ferrum_scheduler::implementations::continuous::cost_model::structured::{
    StructuredInputV1, StructuredScopeV1, StructuredUnknown, MODEL_REVISION,
};
use ferrum_types::SloStructuredCostCapture;
use serde::{Serialize, Serializer};

#[derive(Serialize)]
pub(super) struct DiscoveryReport {
    model_revision: &'static str,
    #[serde(flatten)]
    input: DiscoveryInput,
}
#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum DiscoveryInput {
    Known {
        domain_signature: [u8; 32],
        scope: DiscoveryScope,
        basis: Vec<f64>,
        support: Vec<u64>,
    },
    Unknown {
        #[serde(serialize_with = "unknown_reason")]
        reason: StructuredUnknown,
    },
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DiscoveryScope {
    OrdinaryDecodeSingleLength { rows: usize },
}
impl From<StructuredScopeV1> for DiscoveryScope {
    fn from(scope: StructuredScopeV1) -> Self {
        match scope {
            StructuredScopeV1::OrdinaryDecodeSingleLength { rows } => {
                Self::OrdinaryDecodeSingleLength { rows }
            }
        }
    }
}
fn unknown_reason<S: Serializer>(
    reason: &StructuredUnknown,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // A finite typed enum, not an unbounded error/source dump.
    serializer.collect_str(&format_args!("{reason:?}"))
}

pub(super) fn inspect(
    actual_mode: SloStructuredCostCapture,
    input: impl FnOnce() -> Result<StructuredInputV1, StructuredUnknown>,
) -> Option<DiscoveryReport> {
    match actual_mode {
        SloStructuredCostCapture::Disabled => None,
        SloStructuredCostCapture::HostSettledV1 => {
            let input = match input() {
                Ok(input) => DiscoveryInput::Known {
                    domain_signature: *input.domain_signature(),
                    scope: input.scope().into(),
                    // The engine helper enforces the core's 4096-axis ceiling.
                    basis: input.regression_axes().to_vec(),
                    support: input.joint_support_coordinates().to_vec(),
                },
                Err(reason) => DiscoveryInput::Unknown { reason },
            };
            Some(DiscoveryReport {
                model_revision: MODEL_REVISION,
                input,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_discovery_disabled_never_evaluates_input() {
        assert!(inspect(SloStructuredCostCapture::Disabled, || {
            panic!("disabled capture must not project diagnostic inputs")
        })
        .is_none());
    }

    #[test]
    fn structured_discovery_enabled_missing_and_invalid_inputs_remain_distinct_unknowns() {
        for reason in [
            StructuredUnknown::MissingEvidence,
            StructuredUnknown::InvalidSample,
            StructuredUnknown::UnsupportedScope,
        ] {
            let report = inspect(SloStructuredCostCapture::HostSettledV1, || Err(reason)).unwrap();
            let json = serde_json::to_value(report).unwrap();
            assert_eq!(json["status"], "unknown");
            assert_eq!(json["reason"], format!("{reason:?}"));
            assert_eq!(json["model_revision"], MODEL_REVISION);
            assert!(json.get("domain_signature").is_none());
            assert!(json.get("basis").is_none());
            assert!(json.get("support").is_none());
        }
    }
}
