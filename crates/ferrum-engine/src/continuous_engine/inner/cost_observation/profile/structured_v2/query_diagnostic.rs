//! Trace-only identities from the original failed lookup; no replay or retry.
use super::{ImportedStructuredModelV2, QueryFailure, StructuredQueryV2};

pub(super) fn failure(
    stage: &'static str,
    failure: QueryFailure,
    query: &StructuredQueryV2,
    child: Option<&ImportedStructuredModelV2>,
    local_now: u64,
    version: u64,
) -> QueryFailure {
    if !tracing::enabled!(target: "ferrum::structured_presubmit_audit", tracing::Level::TRACE) {
        return failure;
    }
    // All added evidence is borrowed or copied from checked, fixed-size facts.
    // In particular, do not hash a family or format the query's numeric vectors.
    let input = query.input();
    let numerical_family = input.numerical_family_key();
    tracing::trace!(target: "ferrum::structured_presubmit_audit",
        stage,
        ?failure,
        model_epoch = version,
        local_now_ns = local_now,
        query_owner = ?query.owner(),
        query_domain = ?query.domain_signature(),
        query_physical_domain = ?input.physical_domain_signature(),
        query_numerical_family = ?numerical_family,
        query_basis_axes = input.regression_axes().len(),
        query_support_axes = input.joint_support_coordinates().len(),
        selected_child_domain = ?child.map(|child| child.domain_signature()),
        selected_child_numerical_family = ?child.and_then(|child| child.numerical_family_key()),
        "original structured query failed");
    failure
}
