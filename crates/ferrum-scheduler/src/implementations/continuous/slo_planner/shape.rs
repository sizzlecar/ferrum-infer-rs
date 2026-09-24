//! Legal logical work is resolved into a canonical physical route explicitly.
use super::{super::cost_model::*, cost_shape::canonical_cost_shape, types::*};
use ferrum_interfaces::execution_cost::{ActualRowWork, ActualWaveKind};
#[cfg(test)]
use std::cell::Cell;

/// Enforced total invocation bound derived from the configured search limits.
/// It includes raw candidate attempts and all complete-sequence replays.
#[cfg(test)]
pub(super) struct ResolutionSession<'a> {
    resolver: &'a dyn PlanningShapeResolver,
    remaining: Cell<usize>,
    max_alternatives: usize,
}
#[cfg(test)]
impl<'a> ResolutionSession<'a> {
    pub fn new(resolver: &'a dyn PlanningShapeResolver, settings: &BoundedPlannerSettings) -> Self {
        let depth = settings.search.lookahead_waves.get();
        // Validation caps the product at 65536 and depth at 16. This upper
        // bound is checked independently of callbacks cooperating with polling.
        let limit = settings.search.candidate_limit.get()
            * settings.search.beam_width.get()
            * depth
            // Each raw attempt can invoke both ordering and resolution.
            * (16 + 2 * depth);
        Self {
            resolver,
            remaining: Cell::new(limit),
            max_alternatives: settings.search.max_shape_alternatives.get(),
        }
    }
}
#[cfg(test)]
impl PlanningShapeResolver for ResolutionSession<'_> {
    fn resolve_domain(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<
        Option<PlanningShapeDomain<ferrum_interfaces::execution_cost::CanonicalWaveCostShape>>,
        PlanningUnknownReason,
    > {
        let remaining = self
            .remaining
            .get()
            .checked_sub(1)
            .ok_or(PlanningUnknownReason::SearchIncomplete)?;
        self.remaining.set(remaining);
        let result = self.resolver.resolve_domain(query, poll_budget)?;
        if result
            .as_ref()
            .is_some_and(|domain| domain.shapes().len() > self.max_alternatives)
        {
            return Err(PlanningUnknownReason::ShapeCapacity);
        }
        Ok(result)
    }
    fn order_work(
        &self,
        snapshot: &SchedulerSnapshot,
        work: &mut [CandidateWork],
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason> {
        let remaining = self
            .remaining
            .get()
            .checked_sub(1)
            .ok_or(PlanningUnknownReason::SearchIncomplete)?;
        self.remaining.set(remaining);
        self.resolver.order_work(snapshot, work, poll_budget)
    }
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<
        Option<ferrum_interfaces::execution_cost::CanonicalWaveCostShape>,
        PlanningUnknownReason,
    > {
        let remaining = self
            .remaining
            .get()
            .checked_sub(1)
            .ok_or(PlanningUnknownReason::SearchIncomplete)?;
        self.remaining.set(remaining);
        self.resolver.resolve(query, poll_budget)
    }
}

pub(super) fn order_work(
    snapshot: &SchedulerSnapshot,
    work: &mut [CandidateWork],
    resolver: &dyn PlanningShapeResolver,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    poll_budget()?;
    if work.len() > snapshot.capabilities.max_wave_rows.get() {
        return Err(PlanningUnknownReason::ShapeCapacity);
    }
    let original = work.to_vec();
    let mut failure = None;
    let result = resolver.order_work(snapshot, work, &mut || {
        if let Some(reason) = failure {
            return Err(reason);
        }
        let result = poll_budget();
        if let Err(reason) = result {
            failure = Some(reason);
        }
        result
    });
    let after = poll_budget();
    if let Some(reason) = failure {
        return Err(reason);
    }
    after?;
    result?;
    validate_permutation(&original, work, poll_budget)
}

pub(super) fn validate_permutation(
    original: &[CandidateWork],
    work: &[CandidateWork],
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    if work.len() != original.len() {
        return Err(PlanningUnknownReason::InvalidShapeEvidence);
    }
    // O(n²), with at most 256 rows. Count full entries rather than just IDs so
    // duplicates, offset changes and substituted generations cannot pass.
    for (index, entry) in work.iter().enumerate() {
        poll_budget()?;
        if work[..index].contains(entry) || original.iter().filter(|old| *old == entry).count() != 1
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn resolve(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    work: &[CandidateWork],
    resolver: &dyn PlanningShapeResolver,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<PlanningShapeDomain<WaveExecutionShape>>, PlanningUnknownReason> {
    // Both work input and returned projection are checked by the caller; no
    // resolver may smuggle additional rows or resource progress into a wave.
    if work.is_empty() || work.len() > snapshot.capabilities.max_wave_rows.get() {
        return Ok(None);
    }
    let mut failure = None;
    let legal = legal_rows(snapshot, requests, work, &mut || match poll_budget() {
        Ok(()) => true,
        Err(reason) => {
            failure = Some(reason);
            false
        }
    });
    if let Some(reason) = failure {
        return Err(reason);
    }
    let Some((kind, rows, recurrent_state_bytes)) = legal else {
        return Ok(None);
    };
    poll_budget()?;
    let mut callback_failure = None;
    let result = resolver.resolve_domain(
        &PlanningShapeQuery {
            snapshot,
            prior_waves: &[],
            kind,
            rows: &rows,
            recurrent_state_bytes,
        },
        &mut || {
            if let Some(reason) = callback_failure {
                return Err(reason);
            }
            let result = poll_budget();
            if let Err(reason) = result {
                callback_failure = Some(reason);
            }
            result
        },
    );
    // Enforced even if the implementation never polls. This detects an overrun
    // but cannot preempt a synchronous callback while it is running.
    let after = poll_budget();
    if let Some(reason) = callback_failure {
        return Err(reason);
    }
    after?;
    let domain = result?.ok_or(PlanningUnknownReason::ShapeUnavailable)?;
    validate_domain(
        snapshot,
        kind,
        &rows,
        recurrent_state_bytes,
        &domain,
        poll_budget,
    )
    .map(Some)
}

pub(super) fn validate_domain(
    snapshot: &SchedulerSnapshot,
    kind: ActualWaveKind,
    rows: &[PlanningShapeRow<'_>],
    recurrent_state_bytes: u64,
    domain: &PlanningShapeDomain<ferrum_interfaces::execution_cost::CanonicalWaveCostShape>,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<PlanningShapeDomain<WaveExecutionShape>, PlanningUnknownReason> {
    if domain.shapes().is_empty() || domain.shapes().len() > 256 {
        return Err(PlanningUnknownReason::ShapeCapacity);
    }
    let mut shapes = Vec::new();
    shapes
        .try_reserve_exact(domain.shapes().len())
        .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
    for canonical in domain.shapes() {
        poll_budget()?;
        if canonical.rows.len() != rows.len()
            || canonical.kind != kind
            || canonical.recurrent_state_bytes != recurrent_state_bytes
            || canonical
                .rows
                .iter()
                .zip(rows)
                .any(|(actual, expected)| *actual != expected.work)
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        if matches!(domain, PlanningShapeDomain::HostContentAlternatives(_))
            && canonical.host_content_features.is_none()
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let shape = canonical_cost_shape(canonical)?;
        if shape.path != snapshot.capabilities.path
            || shape.graph_state != snapshot.capabilities.graph_state
            || shape.order != snapshot.capabilities.order
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        shapes.push(shape);
    }
    Ok(match domain {
        PlanningShapeDomain::Exact(_) => PlanningShapeDomain::Exact(shapes.pop().unwrap()),
        PlanningShapeDomain::HostContentAlternatives(_) => {
            PlanningShapeDomain::HostContentAlternatives(shapes)
        }
    })
}

pub(super) fn legal_rows<'a>(
    snapshot: &SchedulerSnapshot,
    requests: &'a [RequestSchedulingView],
    work: &[CandidateWork],
    poll_budget: &mut dyn FnMut() -> bool,
) -> Option<(ActualWaveKind, Vec<PlanningShapeRow<'a>>, u64)> {
    let caps = &snapshot.capabilities;
    work.first()?;
    let mut rows = Vec::with_capacity(work.len());
    let mut decode_kv_tokens = Vec::new();
    let mut prefill_chunks = Vec::new();
    let mut prefill_tokens = 0_u64;
    let mut recurrent_state_bytes = 0_u64;
    for (position, entry) in work.iter().enumerate() {
        if !poll_budget() {
            return None;
        }
        if work[..position]
            .iter()
            .any(|other| other.key.request_id == entry.key.request_id)
        {
            return None;
        }
        let request = requests.iter().find(|request| request.key == entry.key)?;
        if request.readiness != RequestReadiness::Ready || request.timing.completed() {
            return None;
        }
        recurrent_state_bytes = recurrent_state_bytes.checked_add(request.recurrent_state_bytes)?;
        match (&entry.action, &request.phase) {
            (WaveAction::Decode, RequestPhaseView::Decode) => {
                decode_kv_tokens.push(request.context_tokens);
                rows.push(PlanningShapeRow {
                    request,
                    work: ActualRowWork::Decode {
                        kv_tokens: request.context_tokens,
                    },
                });
            }
            (WaveAction::Prefill { offset, count }, RequestPhaseView::Prefill(progress)) => {
                let end = offset.checked_add(count.get())?;
                let is_final = end == progress.total_prompt_tokens.get();
                if *offset != progress.offset
                    || end > progress.executable_until
                    || end > progress.total_prompt_tokens.get()
                    || *offset % caps.prefill_alignment.get() != 0
                    || (count.get() % caps.prefill_alignment.get() != 0
                        && !(is_final && caps.allow_final_short_chunk))
                    || (!caps.prefill_chunk_sizes.contains(count)
                        && !caps
                            .work_policy
                            .declared_prefill_chunks()
                            .any(|size| size == u64::from(count.get()))
                        && !(is_final
                            && caps.allow_final_short_chunk
                            && caps.prefill_chunk_sizes.iter().any(|size| size > count)))
                    || progress.reference.work_at(end).is_none()
                {
                    return None;
                }
                prefill_tokens = prefill_tokens.checked_add(u64::from(count.get()))?;
                rows.push(PlanningShapeRow {
                    request,
                    work: ActualRowWork::Prefill {
                        offset: *offset,
                        count: count.get(),
                        total_prompt_tokens: progress.total_prompt_tokens.get(),
                    },
                });
                prefill_chunks.push(PrefillShape {
                    offset: *offset,
                    count: *count,
                    total_prompt_tokens: progress.total_prompt_tokens,
                });
            }
            _ => return None,
        }
    }
    if work.len() > caps.max_wave_rows.get()
        || prefill_tokens > caps.max_prefill_tokens_per_wave.get()
        || (!decode_kv_tokens.is_empty()
            && !caps
                .decode_batch_sizes
                .iter()
                .any(|size| size.get() == decode_kv_tokens.len()))
        || (!prefill_chunks.is_empty()
            && !caps
                .prefill_batch_sizes
                .iter()
                .any(|size| size.get() == prefill_chunks.len()))
    {
        return None;
    }
    let kind = match (decode_kv_tokens.is_empty(), prefill_chunks.is_empty()) {
        (false, true) => ActualWaveKind::Decode,
        (true, false) => ActualWaveKind::Prefill,
        (false, false) if caps.native_mixed => ActualWaveKind::Mixed,
        _ => return None,
    };
    Some((kind, rows, recurrent_state_bytes))
}
