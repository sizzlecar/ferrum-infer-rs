//! Disposable readiness traversal. Only the final complete collection may
//! retain numerical input authority or establish a conditional member floor.
use super::*;

pub(in super::super) struct ReadinessProgress {
    pub next_target: usize,
    pub total_targets: usize,
    pub gap: Option<InventoryGap>,
}

pub(in super::super) async fn collect_charged(
    session: &mut CalibrationSession,
    cases: &[Case],
    templates: &[AutomaticCostProbeTemplate],
    prompts: &[usize],
    pair: &PrefixPair,
    policy: StructuredPopulationPolicyV1,
    decode_boundaries: &[u32],
    limits: InventoryLimits,
    charge: &mut ProbePreflightCharge,
) -> Result<CheckedCaseInventory> {
    collect_inner(
        session,
        cases,
        templates,
        prompts,
        pair,
        policy,
        decode_boundaries,
        limits,
        None,
        &mut None,
        charge,
        None,
    )
    .await
}

pub(in super::super) async fn probe_readiness(
    session: &mut CalibrationSession,
    cases: &[Case],
    templates: &[AutomaticCostProbeTemplate],
    prompts: &[usize],
    pair: &PrefixPair,
    policy: StructuredPopulationPolicyV1,
    decode_boundaries: &[u32],
    limits: InventoryLimits,
    first_target: usize,
    charge: &mut ProbePreflightCharge,
    can_act: &(dyn Fn(usize, &GeometryProjectionUnknown) -> bool + Sync),
) -> Result<ReadinessProgress> {
    let first = cases
        .first()
        .ok_or_else(|| error("readiness group is empty"))?;
    if cases.iter().any(|case| !same_group(first, case)) {
        return Err(error("readiness cursor cannot cross original owner groups"));
    }
    let mut progress = None;
    // The temporary inventory has no populated facts in readiness mode and is
    // dropped here. In particular, no Known from an older view is exported.
    let _ = collect_inner(
        session,
        cases,
        templates,
        prompts,
        pair,
        policy,
        decode_boundaries,
        limits,
        Some(first_target),
        &mut progress,
        charge,
        Some(can_act),
    )
    .await?;
    progress.ok_or_else(|| error("readiness traversal produced no receipt"))
}

pub(super) fn record_charge(
    accumulated: &mut ProbePreflightCharge,
    receipt: &mut ProbePreflightCharge,
    admitted: usize,
    projections: usize,
    limits: &InventoryLimits,
) -> Result<()> {
    accumulated.admitted_requests = accumulated
        .admitted_requests
        .checked_add(admitted)
        .filter(|n| *n <= limits.maximum_admitted_requests)
        .ok_or_else(|| error("inventory admitted request accounting overflow"))?;
    accumulated.planning_admitted_requests = accumulated.admitted_requests;
    accumulated.projection_attempts = accumulated
        .projection_attempts
        .checked_add(projections)
        .filter(|n| *n <= limits.maximum_projection_attempts)
        .ok_or_else(|| error("inventory projection accounting overflow"))?;
    // The caller debits this receipt even when a following typed outcome is an
    // error. Successful admissions and attempted queries are never refunded.
    *receipt = *accumulated;
    Ok(())
}

pub(super) fn require_nonfatal(
    reason: &GeometryProjectionUnknown,
    stage: &str,
    scenario: usize,
    target: GeometryInputTarget,
    projection_attempts: usize,
    maximum_projections: usize,
) -> Result<()> {
    if matches!(
        reason,
        GeometryProjectionUnknown::BudgetExhausted
            | GeometryProjectionUnknown::Capacity
            | GeometryProjectionUnknown::Admission(_)
    ) {
        return Err(error(format!(
            "{stage} exhausted original projection allowance: reason={reason:?}, scenario={scenario}, target={target:?}, projection_attempts={projection_attempts}, maximum_projections={maximum_projections}"
        )));
    }
    Ok(())
}
