//! Production continuations and real Stored Qwen DiT. These preconditioned
//! fixtures isolate membership/solver/resource semantics; they do not pretend
//! Reference implements CUDA Philox or qualify the full image service.
use super::super::tests::{config, input, pipeline, trajectory};
use super::*;
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime as Runtime;

pub(super) fn request(
    p: &mut QwenImage21Pipeline<Runtime>,
    row: usize,
) -> ContinuousRequest<Runtime> {
    let config = config(p, &[row]);
    let stage = p.prepare_denoiser(&config, 1).unwrap();
    let decoder = p.execution.sources.assets.decoder_batch(1, 2, 2).unwrap();
    let family = p
        .prepare_family(
            ImageStage::Decoder,
            ImageStagePlanRole::WholeStage,
            1,
            QwenImage21VaeDecoderFamily::new().unwrap(),
            &decoder,
            &p.profiles.decoder,
        )
        .unwrap();
    let decode_stage = PreparedStage::compile(
        ImageStage::Decoder,
        1,
        family,
        &decoder.compile_options().unwrap(),
        p,
    )
    .unwrap();
    let original_sample = input(
        &p.execution.host,
        &config,
        QwenImage21TransformerFamily::LATENTS_ID,
        &[row],
    );
    let original_text = input(
        &p.execution.host,
        &config,
        QwenImage21TransformerFamily::TEXT_ID,
        &[row],
    );
    let persistent = p
        .execution
        .host
        .reserve_scope(&[
            (
                HostMemoryClass::InputPreprocess,
                2 * p
                    .execution
                    .host
                    .backing_bytes(original_sample.bytes().len())
                    .unwrap()
                    + p.execution
                        .host
                        .backing_bytes(original_text.bytes().len())
                        .unwrap(),
            ),
            (
                HostMemoryClass::Output,
                p.execution
                    .host
                    .backing_bytes(
                        decoder
                            .output_tensor()
                            .unwrap()
                            .minimum_storage_bytes()
                            .unwrap() as usize,
                    )
                    .unwrap(),
            ),
        ])
        .unwrap();
    let host = persistent.account();
    let sample = copy(&host, &original_sample).unwrap();
    let text = copy(&host, &original_text).unwrap();
    drop((original_sample, original_text));
    let solver = ManagedBf16Euler::new(
        p.execution.sources.assets.schedule.solver().clone(),
        host,
        sample.tensor().clone(),
        p.profiles.euler,
    )
    .unwrap();
    let ports = host_budget::ports(&stage).unwrap();
    let work_budget = ExecutionHostBudget {
        input: host_budget::full_generation_input_peak(&p.execution.host, ports, 1).unwrap(),
        output: host_budget::sum_ports(&p.execution.host, ports.outputs().iter()).unwrap(),
        staging: host_budget::staging_bytes(
            &p.execution.host,
            &stage,
            p.execution.config.static_initialization,
        )
        .unwrap()
        .max(
            host_budget::staging_bytes(
                &p.execution.host,
                &decode_stage,
                p.execution.config.static_initialization,
            )
            .unwrap(),
        ),
        weights: 0,
    };
    let minimum_work = Some(
        p.execution
            .host
            .reserve_scope(&work_budget.claims())
            .unwrap(),
    );
    let job = p.execution.next_job;
    p.execution.next_job += 1;
    ContinuousRequest {
        pipeline: p.execution.service_preparation_identity.clone(),
        state: RequestState {
            id: RequestId {
                job,
                caller_index: 0,
            },
            sample,
            solver,
        },
        text,
        transformer: config,
        decoder,
        decode_stage,
        latent_shape: QwenImage21LatentShape::new(1, 2, 2).unwrap(),
        persistent,
        minimum_work,
        work_budget,
        _backing: vec![],
        stages: vec![],
        steps: vec![],
        seed: row as u64,
        width: 32,
        height: 32,
        last_batch: (1, 0),
        decoded: false,
    }
}
fn refs(states: &mut [ContinuousRequest<Runtime>]) -> Vec<&mut ContinuousRequest<Runtime>> {
    states.iter_mut().collect()
}
fn prepare(
    p: &mut QwenImage21Pipeline<Runtime>,
    states: &mut [ContinuousRequest<Runtime>],
) -> ContinuousBatch<Runtime> {
    p.try_prepare_continuous_batch(&mut refs(states))
        .unwrap_or_else(|e| panic!("prepare actual cohort: {e}"))
        .expect("fixture cohort fits")
}
fn check(r: &ContinuousRequest<Runtime>, baseline: &[Vec<Vec<u8>>]) {
    assert_eq!(
        r.state.sample.bytes().bytes(),
        baseline[r.state.solver.step_index()][0]
    );
}
fn close(
    mut p: QwenImage21Pipeline<Runtime>,
    states: Vec<ContinuousRequest<Runtime>>,
    batch: ContinuousBatch<Runtime>,
) {
    assert!(batch.active.is_none());
    drop((batch, states));
    p.shutdown().unwrap_or_else(|e| panic!("shutdown: {e}"));
    assert_eq!(p.execution.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(p.execution.runtime.snapshot().live_allocations, 0);
}

#[test]
fn actual_dit_mixed_progress_join_and_cancellation_refill_match_independent_b1() {
    let a = trajectory(&[0], &[0]);
    let b = trajectory(&[1], &[0]);
    let (mut p, _directory) = pipeline();
    let mut states = vec![request(&mut p, 0)];
    let mut batch = prepare(&mut p, &mut states);
    batch
        .advance(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("first: {e}"));
    check(&states[0], &a);
    batch
        .pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("pause: {e}"));
    states.push(request(&mut p, 1));
    let next = prepare(&mut p, &mut states);
    drop(batch);
    batch = next;
    assert_eq!(
        states
            .iter()
            .map(|r| r.state.solver.step_index())
            .collect::<Vec<_>>(),
        [1, 0]
    );
    batch
        .advance(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("mixed: {e}"));
    check(&states[0], &a);
    check(&states[1], &b);
    batch
        .pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("pause mixed: {e}"));
    // Cancel the older request at its fence; a fresh job with the same physical
    // geometry must not inherit that request's old sample, solver or prepared ID.
    let removed = states.remove(0);
    let removed_id = removed.state.id;
    drop(removed);
    states.push(request(&mut p, 0));
    assert_ne!(states[1].state.id, removed_id);
    let before = p.execution.runtime.snapshot().submissions;
    assert!(batch.advance(&mut p, &mut refs(&mut states)).is_err());
    assert_eq!(
        p.execution.runtime.snapshot().submissions,
        before,
        "stale membership submitted a forward"
    );
    let next = prepare(&mut p, &mut states);
    drop(batch);
    batch = next;
    batch
        .advance(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("refill: {e}"));
    check(&states[0], &b);
    check(&states[1], &a);
    batch
        .pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("final pause: {e}"));
    close(p, states, batch);
}

#[test]
fn candidate_host_shortage_keeps_old_admitted_step_credit_usable() {
    let baseline = trajectory(&[0], &[0]);
    let (mut p, _directory) = pipeline();
    let mut states = vec![request(&mut p, 0)];
    let mut old = prepare(&mut p, &mut states);
    old.pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("pause: {e}"));
    let mut candidate = request(&mut p, 1);
    let free = p
        .execution
        .host
        .limits()
        .class_bytes(HostMemoryClass::InputPreprocess)
        - p.execution
            .host
            .usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::InputPreprocess);
    let alignment = p.execution.host.allocation_alignment();
    let logical = (free.saturating_sub(alignment - 1) / alignment) * alignment;
    let blocker = p
        .execution
        .host
        .reserve(HostMemoryClass::InputPreprocess, logical)
        .unwrap();
    let before = p.execution.runtime.snapshot().submissions;
    assert!(p
        .try_prepare_continuous_batch(&mut [&mut states[0], &mut candidate])
        .unwrap_or_else(|e| panic!("candidate: {e}"))
        .is_none());
    assert_eq!(p.execution.runtime.snapshot().submissions, before);
    old.advance(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("authorized active step: {e}"));
    check(&states[0], &baseline);
    old.pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("pause: {e}"));
    drop((blocker, candidate));
    close(p, states, old);
}

#[test]
fn failed_last_successor_preserves_all_request_samples_and_cursors() {
    let a = trajectory(&[0], &[0]);
    let b = trajectory(&[1], &[0]);
    let (mut p, _directory) = pipeline();
    let mut states = vec![request(&mut p, 0), request(&mut p, 1)];
    let mut batch = prepare(&mut p, &mut states);
    let snapshots = states
        .iter()
        .map(|r| {
            (
                r.state.solver.step_index(),
                r.state.sample.bytes().bytes().to_vec(),
            )
        })
        .collect::<Vec<_>>();
    let charge = states[1].state.sample.bytes().len();
    let blocker = states[1]
        .persistent
        .account()
        .reserve(HostMemoryClass::InputPreprocess, charge)
        .unwrap();
    let error = batch
        .advance(&mut p, &mut refs(&mut states))
        .err()
        .expect("last successor exceeds its own persistent grant");
    assert_eq!(
        states
            .iter()
            .map(|r| (
                r.state.solver.step_index(),
                r.state.sample.bytes().bytes().to_vec()
            ))
            .collect::<Vec<_>>(),
        snapshots
    );
    let ImageGenerationFailure::Processing {
        active: Some(active),
        ..
    } = error
    else {
        panic!("failure lost its actual stage owner")
    };
    p.finish_stage(active, &mut batch.stages)
        .unwrap_or_else(|e| panic!("retire: {e}"));
    drop(blocker);
    batch
        .advance(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("retry authorized step: {e}"));
    check(&states[0], &a);
    check(&states[1], &b);
    batch
        .pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("pause: {e}"));
    close(p, states, batch);
}

#[test]
fn real_survivor_compaction_exchanges_work_at_full_parent_capacity_before_publication() {
    let baseline = trajectory(&[0], &[0]);
    let (mut p, _directory) = pipeline();
    let mut states = vec![request(&mut p, 0), request(&mut p, 1)];
    let mut old = prepare(&mut p, &mut states);
    old.advance(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("first actual forward: {e}"));
    old.pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("pause actual stage: {e}"));
    let old_view = old.scope.as_ref().unwrap().account();
    let free = p
        .execution
        .host
        .limits()
        .class_bytes(HostMemoryClass::InputPreprocess)
        - p.execution
            .host
            .usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::InputPreprocess);
    let alignment = p.execution.host.allocation_alignment();
    let logical = (free.saturating_sub(alignment - 1) / alignment) * alignment;
    let blocker = p
        .execution
        .host
        .reserve(HostMemoryClass::InputPreprocess, logical)
        .unwrap();
    // Keep the exiting request and its immutable samples until the smaller
    // native plan is actually admitted. A duplicate fresh Host grant cannot
    // fit, while an atomic exchange can use existing unused step credit.
    let next = p
        .try_replace_continuous_batch(&mut old, &mut [&mut states[0]])
        .unwrap_or_else(|e| panic!("admit compact real native stage: {e}"))
        .expect("smaller actual stage fits old step authorization");
    assert!(old.scope.is_none());
    assert!(next.active.is_some());
    assert_eq!(next.prepared.plan.batch_size, 1);
    assert!(matches!(
        old_view.reserve(HostMemoryClass::InputPreprocess, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    let removed = states.pop().unwrap();
    drop(removed);
    drop(old);
    let mut next = next;
    if !states[0].is_complete() {
        next.advance(&mut p, &mut refs(&mut states))
            .unwrap_or_else(|e| panic!("survivor actual forward: {e}"));
        check(&states[0], &baseline);
    }
    next.pause(&mut p, &mut refs(&mut states))
        .unwrap_or_else(|e| panic!("pause survivor: {e}"));
    drop(blocker);
    close(p, states, next);
}
