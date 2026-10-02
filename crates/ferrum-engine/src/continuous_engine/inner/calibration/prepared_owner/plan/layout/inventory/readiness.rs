//! An uninterrupted first traversal may complete the inventory. After any
//! preparation, only a fresh final capture may retain checked input authority.
use super::*;

#[derive(Clone, Copy)]
pub(in super::super) struct ReadinessCursor {
    pub first_target: usize,
    pub retain_complete: bool,
}

pub(in super::super) struct ReadinessProgress {
    pub next_target: usize,
    pub total_targets: usize,
    pub gap: Option<InventoryGap>,
}

pub(in super::super) enum ReadinessCapture {
    Complete(CheckedCaseInventory),
    Progress(ReadinessProgress),
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
    match collect_inner(
        session,
        cases,
        templates,
        prompts,
        pair,
        policy,
        decode_boundaries,
        limits,
        None,
        charge,
        None,
    )
    .await?
    {
        ReadinessCapture::Complete(inventory) => Ok(inventory),
        ReadinessCapture::Progress(_) => {
            Err(error("complete inventory returned a readiness cursor"))
        }
    }
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
    cursor: ReadinessCursor,
    charge: &mut ProbePreflightCharge,
    can_act: &(dyn Fn(usize, &GeometryProjectionUnknown) -> bool + Sync),
) -> Result<ReadinessCapture> {
    let first = cases
        .first()
        .ok_or_else(|| error("readiness group is empty"))?;
    if cases.iter().any(|case| !same_group(first, case)) {
        return Err(error("readiness cursor cannot cross original owner groups"));
    }
    collect_inner(
        session,
        cases,
        templates,
        prompts,
        pair,
        policy,
        decode_boundaries,
        limits,
        Some(cursor),
        charge,
        Some(can_act),
    )
    .await
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
