use super::types::*;

/// Enforce checks around callbacks even if an implementation does not poll or
/// mistakenly ignores a failed poll. Callbacks must themselves be nonblocking.
fn checked<T>(
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    operation: impl FnOnce(
        &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<T, PlanningUnknownReason>,
) -> Result<T, PlanningUnknownReason> {
    poll()?;
    let mut failure = None;
    let result = operation(&mut || {
        if let Some(reason) = failure {
            return Err(reason);
        }
        let result = poll();
        if let Err(reason) = result {
            failure = Some(reason);
        }
        result
    });
    let after = poll();
    if let Some(reason) = failure {
        return Err(reason);
    }
    after?;
    result
}

pub(super) fn begin<'a>(
    resolver: &'a dyn PlanningResourceResolver,
    snapshot: &SchedulerSnapshot,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Box<dyn PlanningResourceProjection + 'a>, PlanningUnknownReason> {
    checked(poll, |poll| resolver.begin(snapshot, poll))
}

pub(super) fn apply(
    projection: &mut dyn PlanningResourceProjection,
    query: &PlanningResourceQuery<'_>,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    checked(poll, |poll| projection.apply(query, poll))
}
