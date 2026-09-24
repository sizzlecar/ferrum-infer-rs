//! Actual engine owner/capture lifetime with the controlled executor's honest
//! Unsupported cost route. Successful native routes have separate product tests.
use super::*;

#[tokio::test]
async fn unified_execution_binds_capture_and_unsupported_never_creates_a_successor() {
    let (engine, _, executor) = fixture().await;
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    let captured = prefill::captured(&engine, &executor).await;
    let before = prefill::UnsubmittedState::capture(&engine, &executor);
    let context = shape::ExecutorShape {
        engine: &engine.inner,
        captured: &captured,
    };
    let copy = captured.snapshot.clone();
    assert!(matches!(
        context.begin(&copy, &mut || Ok(())),
        Err(PlanningUnknownReason::InvalidSnapshot)
    ));
    let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
    let request = &captured.snapshot.requests[0];
    let work = [CandidateWork {
        key: request.key.clone(),
        action: WaveAction::Prefill {
            offset: 0,
            count: NonZeroU32::new(2).unwrap(),
        },
    }];
    let rows = [PlanningShapeRow {
        request,
        work: ActualRowWork::Prefill {
            offset: 0,
            count: 2,
            total_prompt_tokens: 4,
        },
    }];
    let input = PlanningExecutionInput {
        work: &work,
        requests: &captured.snapshot.requests,
        kind: ActualWaveKind::Prefill,
        rows: &rows,
        recurrent_state_bytes: request.recurrent_state_bytes,
    };
    assert!(parent.project(&input, &mut || Ok(())).unwrap().is_none());
    // A failure cannot advance the private numeric parent or consume live work.
    assert!(parent.project(&input, &mut || Ok(())).unwrap().is_none());
    assert!(matches!(
        parent.project(&input, &mut || Err(
            PlanningUnknownReason::ComputeBudgetExhausted
        )),
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
    before.assert_unchanged(&engine, &executor);
    drop(parent);
    drop(context);
    drop(captured);
    cleanup(engine, session).await;
}
