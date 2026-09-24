//! Real scheduler/output owner completion; the device is the explicit
//! controlled protocol executor, not a Metal numerical or performance proof.
use super::*;
use crate::continuous_engine::inner::cost_observation::EngineCostRuntime;
use crate::continuous_engine::{HostStageCompleteness, HostStageEvidenceV1};

async fn observed(width: usize) -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut session, executor) = fixture(width).await;
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::new(identity, &inner.config.scheduler.slo.cost_observation, None)
            .unwrap(),
    ));
    executor
        .emit_cost_observations
        .store(true, Ordering::Release);
    executor
        .completion_work_known
        .store(true, Ordering::Release);
    (session, executor)
}

async fn add_max(
    session: &mut CalibrationSession,
    maximum: usize,
) -> (RequestId, CreditedOutputSession) {
    let mut request =
        ferrum_types::InferenceRequest::new("test", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.max_tokens = maximum;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let output = session
        .add_request(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(session, &id, false).await;
    admit(session).await;
    (id, output)
}

fn stages(report: &CalibrationWaveReport) -> &HostStageEvidenceV1 {
    assert!(
        matches!(&report.observation, CalibrationObservation::Rejected { reason } if reason == "Composite"),
        "{:?}",
        report.observation
    );
    report
        .host_stages
        .as_deref()
        .expect("actual call stage capture")
}

#[tokio::test]
async fn terminal_prefill_max_one_keeps_legacy_composite_and_hands_off_one_completion() {
    let (mut session, executor) = observed(1).await;
    let (id, mut output) = add_max(&mut session, 1).await;
    let row = frontier(&session, &id)
        .prefill_work(NonZeroU32::MIN)
        .unwrap();
    let report = wave(&mut session, &executor, vec![row]).await;
    assert!(report.error.is_none(), "{:?}", report.error);
    let stages = stages(&report);
    let queue = report.host_stage_queue.unwrap();
    assert!(queue.accepted_ordinal.is_some());
    assert_eq!(
        queue.disposition,
        crate::continuous_engine::HostStageQueueDisposition::Published
    );
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave,
        "{stages:?}"
    );
    assert!(stages.full_wall_ns.is_some_and(|ns| ns > 0));
    let row = &stages.rows[0];
    assert!(row.output_published_at_ns <= row.settled_at_ns);
    let terminal = row.terminal.as_ref().unwrap();
    assert_eq!(terminal.finish_reason, ferrum_types::FinishReason::Length);
    assert_eq!(terminal.generated_tokens, 1);
    assert!(
        terminal.terminal_handoff_succeeded
            && terminal.request_slot_closed
            && terminal.owner_matched
    );
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 1);
    assert!(session.frontiers().unwrap().is_empty());
    bounded(async {
        while let Some(frame) = output.frames.next().await {
            drop(frame);
        }
    })
    .await;
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn terminal_decode_mixed_owner_lengths_settle_in_real_host_order_without_replay() {
    let (mut session, executor) = observed(3).await;
    let mut outputs = Vec::new();
    let mut ids = Vec::new();
    for maximum in [2, 3, 2] {
        let (id, output) = add_max(&mut session, maximum).await;
        ids.push(id);
        outputs.push(output);
    }
    let prefills = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .prefill_work(NonZeroU32::MIN)
                .unwrap()
        })
        .collect();
    wave(&mut session, &executor, prefills).await;
    for (id, output) in ids.iter().zip(&mut outputs) {
        drop(bounded(output.frames.next()).await.unwrap());
        ready(&session, id, false).await;
    }
    let decodes = ids
        .iter()
        .map(|id| frontier(&session, id).decode_work().unwrap())
        .collect();
    let report = wave(&mut session, &executor, decodes).await;
    let stages = stages(&report);
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave,
        "{stages:?}"
    );
    assert_eq!(
        stages
            .rows
            .iter()
            .filter(|row| row.terminal.is_some())
            .count(),
        2
    );
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 2);
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    assert_eq!(session.frontiers().unwrap().len(), 1);
    assert_eq!(frontier(&session, &ids[1]).generated_tokens(), 2);
    for output in &mut outputs {
        drop(bounded(output.frames.next()).await.unwrap());
    }
    drop(outputs);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn terminal_output_and_cache_failures_remain_failed_even_when_public_completion_is_ok() {
    for output_failure in [true, false] {
        let (mut session, executor) = observed(1).await;
        let (id, output) = add_max(&mut session, 1).await;
        if output_failure {
            // The actual final projection cannot extend this published prefix.
            // It records an output failure internally; successful cache and
            // scheduler cleanup must not turn its receipt into success.
            session
                .engine
                .inner
                .sequences
                .write()
                .get_mut(&id)
                .unwrap()
                .streamed_text_len = usize::MAX;
        } else {
            executor.completion_fail.store(true, Ordering::Release);
        }
        let row = frontier(&session, &id)
            .prefill_work(NonZeroU32::MIN)
            .unwrap();
        let report = wave(&mut session, &executor, vec![row]).await;
        let stages = report.host_stages.as_ref().unwrap();
        assert_eq!(stages.full_wall_ns, None);
        let terminal = stages.rows[0].terminal.as_ref().unwrap();
        assert!(terminal.output_failed);
        if output_failure {
            assert!(report.error.is_none());
        } else {
            assert!(terminal.physical_failed);
        }
        assert!(session.frontiers().unwrap().is_empty());
        assert_eq!(executor.physical.load(Ordering::Acquire), 1);
        drop(output);
        session.shutdown().await.unwrap();
    }
}
