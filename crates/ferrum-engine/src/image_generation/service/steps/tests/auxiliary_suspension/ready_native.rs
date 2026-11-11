//! Exercise the worker boundary with real Ready and native initialization owners.
//! No auxiliary completion or successful DiT execution is synthesized.
use super::super::super::{initialization as worker_initialization, restore_at_boundary};
use super::*;

fn queued(seed: u64) -> Queued<ImageGenerationOutput> {
    Queued {
        trace: trace::RequestTrace::unobserved(),
        request: QueuedRequest::Generate(request(seed)),
        adapter: None,
        submitted: Instant::now(),
        cancelled: Arc::new(AtomicBool::new(false)),
        response: oneshot::channel().0,
        output: None,
        _prompt_reservation: None,
        preprocessing_bytes: 0,
        preparation_duration: Duration::ZERO,
        preparation_group: None,
    }
}

#[test]
fn ready_fifo_waiter_allows_original_native_blocker_to_retire_and_restore_real_forward() {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let mut generator = generator(false, 3);
    let runtime = generator.inner.pipeline.execution.runtime.clone();
    let host = generator.host_account().clone();
    let trace = generator.inner.trace.clone();
    let control = generator.control.clone();
    generator.inner.boundary.release(usize::MAX);
    control.release();

    let mut survivor = generator
        .start_request(ImageBatchPreparation::new(0_u64), &request(0))
        .unwrap();
    let mut batch = generator
        .try_prepare_batch(&mut [&mut survivor], None)
        .unwrap()
        .unwrap();
    assert_eq!(
        generator
            .advance_batch(&mut batch, &mut [&mut survivor])
            .unwrap(),
        [false]
    );
    let ready = generator
        .start_request(ImageBatchPreparation::new(2_u64), &request(2))
        .unwrap();
    generator
        .pause_batch(&mut batch, &mut [&mut survivor])
        .unwrap();
    let mut prepared = ImageBatchPreparation::new(1_u64);
    assert!(generator
        .try_admit_request_at_boundary(
            &mut prepared,
            ImagePreparationResources::default(),
            Some(&mut batch),
            &mut [&mut survivor],
        )
        .unwrap());
    let ImageInitializationOutcome::Pending(native) = generator
        .begin_image_initialization(prepared, ImageRequestRef::Generate(&request(1)))
        .unwrap()
    else {
        panic!("original auxiliary owner must remain native-live");
    };
    assert!(native.get::<Phase>().unwrap().native.is_some());
    assert!(runtime.snapshot().live_allocations > 0);

    let mut running = vec![Running {
        job: queued(0),
        state: survivor,
        complete: false,
    }];
    let mut ready_waiter = Pending::new(queued(2));
    ready_waiter.state = Some(ready);
    let mut native_waiter = Pending::new(queued(1));
    native_waiter.initialization = Some(native);
    let mut pending = VecDeque::from([ready_waiter, native_waiter]);
    let shared = Shared {
        trace: None,
        state: Mutex::new(State {
            queue: VecDeque::new(),
            queue_backing: None,
            accepting: true,
            closing: false,
            active: running.len(),
            preparing: pending.len(),
            preparing_bytes: pending.iter().map(|p| p.job.pending_bytes()).sum(),
            rejected: 0,
            executed_batches: 1,
            executed_requests: 1,
            last_batch_size: 1,
            capacity_reductions: 0,
        }),
        changed: Condvar::new(),
        resources: Arc::new(ResourceSignal::default()),
        config: ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            maximum_pending: 4,
            maximum_pending_bytes: 1 << 18,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 3,
            request_batch_max_wait_ms: 0,
        },
        host: Some(host.clone()),
        adapter: None,
    };
    let mut worker = GeneratorAdapter(Box::new(generator));
    let mut batch = Some(batch);
    assert!(!restore_at_boundary(&mut worker, &mut batch, &mut running, &pending, None).unwrap());
    assert_eq!(trace.lock().unwrap().len(), 1);

    let tick = worker_initialization::advance(
        &mut worker,
        &shared,
        &mut batch,
        &mut running,
        &mut pending,
    )
    .unwrap();
    assert!(
        tick.made_progress,
        "Ready must not strand an existing native owner"
    );
    assert_eq!(pending.len(), 2);
    assert!(pending.iter().all(|p| p.initialization.is_none()));
    assert_eq!(
        pending
            .iter()
            .map(|p| p
                .state
                .as_ref()
                .unwrap()
                .get::<ContinuousRequest<Runtime>>()
                .unwrap()
                .fixture_snapshot()
                .0)
            .collect::<Vec<_>>(),
        [2, 1],
        "retirement must preserve the original Ready FIFO member"
    );
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert!(restore_at_boundary(&mut worker, &mut batch, &mut running, &pending, None).unwrap());
    assert_eq!(
        worker
            .denoising_generator()
            .unwrap()
            .advance_batch(batch.as_mut().unwrap(), &mut [&mut running[0].state])
            .unwrap(),
        [true]
    );
    assert_eq!(
        *control.events.lock().unwrap(),
        [
            "actual_forward",
            "auxiliary_native_live",
            "auxiliary_closed",
            "restored_native_cohort",
            "actual_forward"
        ]
    );
    let observed = trace.lock().unwrap();
    assert_eq!(observed.len(), 2);
    assert_eq!((observed[1][0].0, observed[1][0].1), (0, 2));
    check(&observed, &independent);
    drop(observed);

    pause(&mut worker, &mut batch, &mut running).unwrap();
    drop((batch, running, pending));
    worker.shutdown().unwrap();
    drop(worker);
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
