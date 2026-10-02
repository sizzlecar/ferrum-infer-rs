//! Cold allocator preparation for a current UnmaterializedCapacity gap.
//! It spends the original owner/time allowance but emits no model wave or
//! numerical sample. Only the next complete capture can establish a route.
use super::*;
use ferrum_interfaces::{
    execution_cost::ActualWaveKind,
    model_executor::{ExecutorResourcePreparationOutcome, ExecutorResourcePreparationRequest},
};
use std::sync::atomic::Ordering;

#[cfg(test)]
pub(super) fn requests(
    case: &Case,
    prompt: usize,
    whole_chunk: usize,
) -> Result<Vec<ExecutorResourcePreparationRequest>> {
    requests_with_row_ceiling(case, prompt, whole_chunk, None)
}

pub(super) fn requests_with_row_ceiling(
    case: &Case,
    prompt: usize,
    whole_chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
) -> Result<Vec<ExecutorResourcePreparationRequest>> {
    let participants = NonZeroUsize::new(case.width)
        .ok_or_else(|| error("resource readiness has no participants"))?;
    let frontier = prompt
        .checked_add(case.maximum_output.get() - 1)
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| error("resource readiness frontier overflow"))?;
    let whole_chunk = u32::try_from(whole_chunk)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            error("resource readiness whole-wave capacity exceeds typed token domain")
        })?;
    let chunk = usize::try_from(
        case.prefill_chunk(whole_chunk, prefill_row_ceiling, case.width)?
            .get(),
    )
    .ok()
    .map(|chunk| chunk.min(prompt))
    .and_then(NonZeroUsize::new)
    .ok_or_else(|| error("resource readiness exceeds whole-wave capacity"))?;
    let mut requests = vec![ExecutorResourcePreparationRequest::new(
        participants,
        frontier,
        chunk,
        ActualWaveKind::Prefill,
    )?];
    // The final prompt chunk can select a different real workspace bucket.
    // Prepare only shapes in this declaration, not every configured bucket.
    if let Some(tail) = NonZeroUsize::new(prompt % chunk.get()) {
        requests.push(ExecutorResourcePreparationRequest::new(
            participants,
            frontier,
            tail,
            ActualWaveKind::Prefill,
        )?);
    }
    if case.maximum_output.get() > 1 {
        requests.push(ExecutorResourcePreparationRequest::new(
            participants,
            frontier,
            NonZeroUsize::MIN,
            ActualWaveKind::Decode,
        )?);
    }
    Ok(requests)
}

pub(super) fn prepare(
    session: &CalibrationSession,
    input: &PreparedProbeInputs,
    case: &Case,
    budget: &mut ProbeExecutionBudget,
) -> Result<()> {
    session.completed_owner_boundary()?;
    let inner = &session.engine.inner;
    if !inner.manual_calibration_driver
        || inner.bg_loop_spawned.load(Ordering::Acquire)
        || inner.is_running.load(Ordering::Acquire)
        || inner.shutdown_started.load(Ordering::Acquire)
        || inner.scheduler.active_count() != 0
        || inner.scheduler.waiting_count() != 0
        || case.width > session.limits.maximum_requests().get()
    {
        return Err(error(
            "resource readiness requires an unused isolated session",
        ));
    }
    let requests = requests_with_row_ceiling(
        case,
        input.prompts[case.template],
        input.chunk.get() as usize,
        input.prefill_row_ceiling,
    )?;
    let mut reserved_participants = 0usize;
    for request in &requests {
        // Reserve before any disposable owner can be created. Failed/partial
        // preparation retains this debit; only complete groups count admitted.
        budget.claim_readiness_requests(request.participants())?;
        reserved_participants = reserved_participants
            .checked_add(request.participants())
            .ok_or_else(|| error("resource readiness owner count overflow"))?;
    }
    let deadline = budget.deadline();
    let receipt = inner
        .model_executor
        .prepare_execution_resources(&requests, &mut || Instant::now() < deadline)?;
    if receipt.prepared_participants > reserved_participants
        || (receipt.outcome == ExecutorResourcePreparationOutcome::Prepared
            && receipt.prepared_participants != reserved_participants)
    {
        return Err(error(
            "resource readiness receipt differs from reserved owner groups",
        ));
    }
    if receipt.prepared_participants != 0 {
        budget.record_readiness_admissions(receipt.prepared_participants)?;
    }
    tracing::info!(template = case.template, ?requests, ?receipt,
        preflight = ?budget.preflight_charge(),
        "Automatic allocator readiness without numerical execution");
    budget.require_selection_time()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readiness_prepares_the_same_row_ceiling_and_physical_width() {
        let case = Case {
            product: OpportunityProduct::Prefill,
            template: 0,
            width: 2,
            maximum_output: NonZeroUsize::new(2).unwrap(),
            release_generated: 0,
            suffix_tokens: 2,
            preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
            prefix: PrefixKind::Ordinary,
            route: CalibrationDecodeRoute::Actual,
            reset: false,
            acquisition: None,
        };
        let requests = requests_with_row_ceiling(&case, 3, 2, Some(NonZeroU32::MIN)).unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|request| request.participants() == 2));
        assert!(requests
            .iter()
            .all(|request| request.tokens_per_sequence() == 1));
        assert!(requests_with_row_ceiling(&case, 3, 1, Some(NonZeroU32::MIN)).is_err());
        let mut single = case;
        single.width = 1;
        let requests = requests_with_row_ceiling(&single, 3, 2, Some(NonZeroU32::MIN)).unwrap();
        assert!(requests
            .iter()
            .all(|request| request.tokens_per_sequence() == 1));
        let uncapped = requests_with_row_ceiling(&single, 3, 2, None).unwrap();
        assert_eq!(uncapped[0].tokens_per_sequence(), 2);
    }
    #[test]
    fn resource_readiness_shapes_preserve_original_rows_frontier_and_chunk() {
        for (width, prompt, output, whole_chunk, expected_chunk) in [
            (1, 4094, 2, 2048, 2048),
            (3, 4094, 2, 2048, 682),
            (8, 9, 3, 32, 4),
            (2, 1, 1, 8, 1),
        ] {
            let case = Case {
                product: OpportunityProduct::Prefill,
                template: 0,
                width,
                maximum_output: NonZeroUsize::new(output).unwrap(),
                release_generated: 0,
                suffix_tokens: output,
                preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
                prefix: PrefixKind::Ordinary,
                route: CalibrationDecodeRoute::Actual,
                reset: false,
                acquisition: None,
            };
            let shapes = requests(&case, prompt, whole_chunk).unwrap();
            let tail = prompt % expected_chunk;
            assert_eq!(
                shapes.len(),
                1 + usize::from(tail != 0) + usize::from(output > 1)
            );
            for shape in &shapes {
                assert_eq!(shape.participants(), width);
                assert_eq!(shape.sequence_tokens(), prompt + output - 1);
            }
            assert_eq!(shapes[0].tokens_per_sequence(), expected_chunk);
            assert_eq!(shapes[0].kind(), ActualWaveKind::Prefill);
            if tail != 0 {
                assert_eq!(shapes[1].tokens_per_sequence(), tail);
                assert_eq!(shapes[1].kind(), ActualWaveKind::Prefill);
            }
            if output > 1 {
                let decode = shapes.last().unwrap();
                assert_eq!(decode.tokens_per_sequence(), 1);
                assert_eq!(decode.kind(), ActualWaveKind::Decode);
            }
        }
    }
}
