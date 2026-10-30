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
    let prepared = scheduler::prepare_capacity_prefix(
        worker,
        &mut batch,
        &shared,
        hint,
        &mut batch_efficiency::BatchEfficiency::default(),
    )
    .unwrap();
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
fn short_successes_do_not_limit_the_next_available_fifo_prefix() {
    let mut worker = worker(16);
    let mut hint = CapacitySearchHint::default();
    assert_eq!(prepare(&mut worker, &mut hint, 1, None).0.len(), 1);
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 16, None).0.len(), 16);
    assert_eq!(worker.probes, [16]);
    let context = worker.capacity_search_context(&[]);
    assert_eq!(
        hint.upper(context, &None, 64),
        64,
        "a successful batch is not proof that a larger one cannot fit"
    );
    assert!(hint.0.is_none(), "success does not create a capacity bound");
}

#[test]
fn failed_candidate_executes_the_first_fitting_prefix_without_a_small_history_fallback() {
    let mut worker = worker(5);
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 1, None);
    // A short prior queue neither caps the next candidate nor becomes the
    // preferred fallback. Stop at the first fresh fit without searching up.
    worker.probes.clear();
    assert_eq!(
        prepare(&mut worker, &mut hint, 7, None),
        (vec![0, 1, 2], vec![3, 4, 5, 6])
    );
    assert_eq!(worker.probes, [7, 3]);
}

#[test]
fn live_capacity_drop_descends_safely_and_recovery_reopens_the_candidate() {
    let mut worker = worker(8);
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 4, None);
    // Capacity can fall after the metadata snapshot. All smaller prefixes
    // must still receive fresh checks, even if they previously fitted.
    worker.capacity = 1;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 1);
    assert_eq!(worker.probes, [8, 4, 2, 1]);

    worker.capacity = 8;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 8);
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
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 8);
    assert_eq!(worker.probes, [8]);
}

#[test]
fn shape_and_adapter_changes_do_not_reuse_failure_boundaries() {
    let mut worker = worker(2);
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 2, None);
    prepare(&mut worker, &mut hint, 8, None);
    worker.capacity = 8;
    // Keep the old low snapshot so only the changed identity invalidates
    // the negative bounds; fresh preparation sees the recovered device.
    worker.shape = Some(1);
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 8);
    assert_eq!(worker.probes, [8]);

    worker.capacity = 6;
    worker.shape = Some(0);
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
