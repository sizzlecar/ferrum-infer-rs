use super::*;

struct Worker {
    capacity: usize,
    observed_capacity: usize,
    shape: Option<u8>,
    probes: Vec<usize>,
}
impl BatchWorker for Worker {
    type Output = u64;
    fn capacity_search_context(&self, _: &[&QueuedRequest]) -> Option<ImageCapacitySearchContext> {
        self.shape.map(|shape| ImageCapacitySearchContext {
            shape: [shape; 32],
            capacity_bytes: self.observed_capacity as u64,
        })
    }
    fn prepare(
        &mut self,
        requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.probes.push(requests.len());
        if requests.len() > self.capacity {
            let proof = super::super::super::ImageDeviceCapacityExceeded {
                report: super::super::super::ImageDeviceCapacityReport {
                    // A planner may reject a stage before the full plan exists.
                    complete_batch: false,
                    required_bytes: requests.len() as u64,
                    capacity_bytes: self.capacity as u64,
                    retained_static_bytes: 0,
                    active_stage_peak_bytes: requests.len() as u64,
                    retained_dynamic_allowance_bytes: 0,
                    native_allowance_bytes: 0,
                },
            };
            return Err(ImageBatchError::Capacity {
                diagnostic: proof.to_string(),
                proof,
            });
        }
        Ok(ImageBatchPreparation::new(
            requests
                .iter()
                .map(|request| request.generation().seed)
                .collect::<Vec<_>>(),
        ))
    }
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        unreachable!("only the real scheduler's preparation is exercised")
    }
}

fn shared() -> Shared<u64> {
    Shared {
        state: Mutex::new(State {
            queue: VecDeque::new(),
            queue_backing: None,
            accepting: true,
            closing: false,
            active: 0,
            preparing: 0,
            preparing_bytes: 0,
            rejected: 0,
            executed_batches: 0,
            executed_requests: 0,
            last_batch_size: 0,
            capacity_reductions: 0,
        }),
        changed: Condvar::new(),
        resources: Arc::new(ResourceSignal::default()),
        config: ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            maximum_pending: 64,
            maximum_pending_bytes: 64 * 1024,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 64,
            request_batch_max_wait_ms: 0,
        },
        host: None,
        adapter: None,
    }
}

fn prepare(
    worker: &mut Worker,
    hint: &mut CapacitySearchHint,
    count: usize,
    adapter: Option<ImageAdapterBatchIdentity>,
) -> (Vec<u64>, Vec<u64>) {
    let shared = shared();
    let adapter = adapter.map(Arc::new);
    let mut batch = (0..count)
        .map(|seed| Queued {
            request: QueuedRequest::Generate(ImageGenerationRequest {
                prompt: "a".into(),
                width: 32,
                height: 32,
                seed: seed as u64,
                n: 1,
            }),
            adapter: adapter.clone(),
            submitted: Instant::now(),
            cancelled: Arc::new(AtomicBool::new(false)),
            response: oneshot::channel().0,
            output: None,
            _prompt_reservation: None,
            preprocessing_bytes: 0,
            preparation_duration: Duration::ZERO,
        })
        .collect::<Vec<_>>();
    {
        let mut state = shared.lock();
        state.preparing = count;
        state.preparing_bytes = batch.iter().map(Queued::pending_bytes).sum();
    }
    let mut prepared = scheduler::prepare_capacity_prefix(
        worker,
        &mut batch,
        &shared,
        hint,
        &mut batch_efficiency::BatchEfficiency::default(),
    )
    .unwrap();
    let completed = prepared.1.take();
    let prepared = prepared.into_inner::<Vec<u64>>().unwrap();
    let selected = batch
        .iter()
        .map(|job| job.request.generation().seed)
        .collect::<Vec<_>>();
    assert_eq!(prepared, selected);
    let tail = shared
        .lock()
        .queue
        .iter()
        .map(|job| job.request.generation().seed)
        .collect::<Vec<_>>();
    assert_eq!(
        selected.iter().chain(&tail).copied().collect::<Vec<_>>(),
        (0..count as u64).collect::<Vec<_>>()
    );
    // Simulate the worker-loop hook after successful execution. Preparation
    // alone must not mark a batch completed, including cancelled preparations.
    if let Some(completed) = completed {
        hint.completed(
            completed.context,
            &completed.adapter,
            completed.batch_size,
            shared.config.max_num_seqs,
        );
    }
    (selected, tail)
}

fn worker(capacity: usize) -> Worker {
    Worker {
        capacity,
        observed_capacity: capacity,
        shape: Some(0),
        probes: Vec::new(),
    }
}

#[test]
fn completed_prefix_grows_geometrically_without_becoming_a_capacity_limit() {
    let mut worker = worker(16);
    let mut hint = CapacitySearchHint::default();
    assert_eq!(prepare(&mut worker, &mut hint, 1, None).0.len(), 1);
    for expected in [2, 4, 8, 16] {
        worker.probes.clear();
        assert_eq!(prepare(&mut worker, &mut hint, 32, None).0.len(), expected);
        assert_eq!(worker.probes, [expected]);
    }
    let context = worker.capacity_search_context(&[]);
    assert_eq!(
        hint.upper(context, &None, 64),
        64,
        "a successful batch is not proof that a larger one cannot fit"
    );
    prepare(&mut worker, &mut hint, 1, None);
    assert_eq!(hint.successful(context, &None), Some(16));
}

#[test]
fn one_failed_growth_probe_immediately_executes_the_fresh_completed_prefix() {
    let mut worker = worker(5);
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 4, None);
    // Do not repeat the full 15 -> 7 -> 3 -> 5 -> 6 search in front of
    // waiting requests. A failed expansion returns to the known FIFO prefix.
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 15, None).0.len(), 4);
    assert_eq!(worker.probes, [8, 4]);
    // Subsequent real dispatches narrow the bracket, one unknown size each.
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 15, None).0.len(), 4);
    assert_eq!(worker.probes, [6, 4]);
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 15, None).0.len(), 5);
    assert_eq!(worker.probes, [5]);
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 15, None).0.len(), 5);
    assert_eq!(
        worker.probes,
        [5],
        "the final known prefix is still prepared"
    );
}

#[test]
fn failed_fallback_descends_safely_and_capacity_recovery_allows_growth() {
    let mut worker = worker(8);
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 4, None);
    // Capacity can fall after the metadata snapshot. Both the growth probe
    // and the previously successful fallback must receive fresh checks.
    worker.capacity = 1;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 16, None).0.len(), 1);
    assert_eq!(worker.probes, [8, 4, 2, 1]);

    worker.capacity = 8;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 16, None).0.len(), 8);
    assert_eq!(worker.probes, [8]);
}

#[test]
fn changing_capacity_reuses_independent_failure_bounds() {
    let mut worker = worker(3);
    let mut hint = CapacitySearchHint::default();
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 2);
    assert_eq!(worker.probes, [9, 4, 2]);
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 3);
    worker.capacity = 2;
    worker.observed_capacity = 2;
    prepare(&mut worker, &mut hint, 9, None);

    worker.capacity = 3;
    worker.observed_capacity = 3;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 3);
    assert_eq!(worker.probes, [3]);

    worker.capacity = 2;
    worker.observed_capacity = 2;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 2);
    assert_eq!(worker.probes, [2]);

    // The old rejection of nine still brackets a recovered larger capacity.
    worker.capacity = 8;
    worker.observed_capacity = 8;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 6);
    assert_eq!(worker.probes, [6]);
}

#[test]
fn shape_and_adapter_changes_do_not_reuse_success_or_failure_boundaries() {
    let mut worker = worker(2);
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 2, None);
    prepare(&mut worker, &mut hint, 8, None);
    worker.capacity = 8;
    worker.observed_capacity = 8;
    worker.shape = Some(1);
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 8);
    assert_eq!(worker.probes, [8]);

    worker.capacity = 6;
    worker.observed_capacity = 6;
    let adapter = ImageAdapterBatchIdentity {
        effective_identity: "different adapter".into(),
        strength_bits: 1f32.to_bits(),
    };
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, Some(adapter)).0.len(), 4);
    assert_eq!(worker.probes, [8, 4]);
}

#[test]
fn first_fitting_context_prefix_executes_but_missing_context_keeps_full_search() {
    let mut contextual = worker(5);
    let mut hint = CapacitySearchHint::default();
    assert_eq!(prepare(&mut contextual, &mut hint, 15, None).0.len(), 3);
    assert_eq!(contextual.probes, [15, 7, 3]);

    let mut uncached = worker(5);
    uncached.shape = None;
    let mut hint = CapacitySearchHint::default();
    assert_eq!(prepare(&mut uncached, &mut hint, 15, None).0.len(), 5);
    assert_eq!(uncached.probes, [15, 7, 3, 5, 6]);
    assert!(hint.0.is_none(), "no incomplete context is retained");
}
