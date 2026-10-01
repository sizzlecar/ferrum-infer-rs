//! A real submitted CPU call starts while the old source block is full. The
//! worker can open its successor while that call is still in flight; the next
//! first offer closes the original unoffered FIFO prefix without stalling work.
use super::*;

fn continuing(generated: u64) -> Wave {
    let mut host = wave(A).host;
    host.state.generated_tokens_before = generated;
    host.state.sampling_history_tokens = generated;
    host.state.maximum_output_tokens = 64;
    let mut w = wave_with_host_rows(A, 7 + generated as u32, host, 2);
    for (position, row) in w.actual.rows.iter_mut().enumerate() {
        row.request_id = RequestId(uuid::Uuid::from_u128(900 + position as u128));
        row.owner_incarnation = 90 + position as u64;
        row.work_generation = 10 + generated;
    }
    w
}

#[tokio::test]
async fn automatic_owner_block_first_offer_fifo_preserves_inflight_unticketed_progress() {
    for first in [
        FirstOffer::Completed,
        FirstOffer::Outside,
        FirstOffer::NotSubmitted,
    ] {
        original_inflight_progress(first).await;
    }
}

#[derive(Clone, Copy)]
enum FirstOffer {
    Completed,
    Outside,
    NotSubmitted,
}

fn not_submitted(f: &Families, generated: u64) {
    let at = f.clock.now_ns().unwrap() + 100;
    f.clock.set(at);
    let w = continuing(generated);
    let mut call = EngineCostCall::begin(
        &f.runtime.ids,
        f.clock.clone(),
        f.runtime.sink.clone(),
        EngineCostCallSpec {
            identity: identity(),
            participants: w
                .actual
                .rows
                .iter()
                .map(|row| CostObservationParticipant {
                    request_id: row.request_id.clone(),
                    owner_incarnation: row.owner_incarnation,
                    work_generation: row.work_generation,
                    input_index: row.input_index,
                    output_policy_signature: Some([6; 32]),
                    host_features: Some(w.host),
                })
                .collect(),
            prepare_started_at_ns: Some(at),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 8,
                max_retained_rows: 128,
            },
        },
    )
    .unwrap()
    .with_live_ticket(Some(f.runtime.reserve_live_ticket(Some(at)).unwrap()));
    {
        let mut context = call.context().unwrap();
        f.clock.set(at + 3);
        context.finish_capacity_deferred(&super::super::super::no_submission::capacity());
        f.clock.set(at + 5);
    }
    assert!(call.recorder.no_submission().is_some());
    f.clock.set(at + 8);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    f.clock.set(at + 20);
}

async fn original_inflight_progress(first: FirstOffer) {
    let f = Families::new();
    for generated in 1..=OFFERS as u64 {
        fixture::record_cohort_route(&f.runtime, &f.clock, continuing(generated)).unwrap();
        f.runtime.consume_samples();
    }
    assert_eq!(f.live().audit().population.phase, 1);
    assert!(f.live().audit().population.closed);
    // The hook runs after genuine CPU commands and host commits, but before
    // original call.finish assigns its accepted FIFO. It is not a no-submit.
    let stages = fixture::record_unticketed_cohort_with_hook(
        &f.runtime,
        &f.clock,
        continuing(OFFERS as u64 + 1),
        |_| {
            for _ in 0..2 {
                f.runtime.consume_samples();
                let audit = f.live().audit();
                assert_eq!(audit.population.phase, 2, "{audit:#?}");
                assert!(!audit.population.closed);
                assert_eq!(audit.population.issued, 0);
                assert_eq!(audit.failed_generations, 0);
            }
        },
    )
    .unwrap();
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    assert_eq!(f.runtime.audit_snapshot().sink.raw_pending, 1);
    assert_eq!(f.live().audit().population.phase, 2);
    f.runtime.consume_samples();
    let reopened = f.live().audit();
    assert_eq!(reopened.population.phase, 2, "{reopened:#?}");
    assert_eq!(reopened.population.issued, 0);
    let mut generated = OFFERS as u64 + 2;
    match first {
        FirstOffer::Completed => {
            fixture::record_cohort_route(&f.runtime, &f.clock, continuing(generated)).unwrap();
            generated += 1;
        }
        FirstOffer::Outside => {
            fixture::record_outside_cohort_route(&f.runtime, &f.clock, continuing(generated))
                .unwrap();
            generated += 1;
        }
        FirstOffer::NotSubmitted => not_submitted(&f, generated),
    }
    f.runtime.consume_samples();
    for _ in 1..OFFERS {
        fixture::record_cohort_route(&f.runtime, &f.clock, continuing(generated)).unwrap();
        f.runtime.consume_samples();
        generated += 1;
    }
    let completed = f.live().audit();
    assert!(completed.population.closed, "{completed:#?}");
    assert_eq!(completed.population.issued, OFFERS);
    assert_eq!(completed.population.retired, OFFERS);
    assert_eq!(completed.failed_generations, 0, "{completed:#?}");
    assert_eq!(
        completed.population.no_submission,
        usize::from(matches!(first, FirstOffer::NotSubmitted))
    );
    assert_eq!(
        completed.population.outside_declared_route,
        usize::from(matches!(first, FirstOffer::Outside))
    );
    assert_eq!(
        completed.automatic.unwrap().owner_blocks.unwrap().offered,
        2 * OFFERS as u64
    );
    let sink = f.runtime.audit_snapshot().sink;
    assert_eq!(sink.raw_resolved, 2 * OFFERS as u64 + 1);
    assert_eq!(sink.raw_lost, 0);
    assert_eq!(sink.raw_resolution_failed, 0);
    f.runtime.shutdown().await.unwrap();
}
