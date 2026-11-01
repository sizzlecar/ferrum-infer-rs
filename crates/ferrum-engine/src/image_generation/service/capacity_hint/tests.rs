use super::*;

#[test]
fn progressive_preparation_runs_fitting_work_then_continues_expansion() {
    let mut worker = Worker {
        capacity: 16,
        observed_capacity: 16,
        shape: 0,
        probes: Vec::new(),
    };
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 1, None);
    worker.probes.clear();
    assert_eq!(
        prepare_with_search(&mut worker, &mut hint, 7, None, true)
            .0
            .len(),
        4
    );
    assert_eq!(worker.probes, [4]);
    worker.probes.clear();
    assert_eq!(
        prepare_with_search(&mut worker, &mut hint, 15, None, true)
            .0
            .len(),
        9
    );
    assert_eq!(
        worker.probes,
        [9],
        "a successful small prefix is never a capacity ceiling"
    );
}

#[test]
fn progressive_capacity_failure_prepares_known_work_and_retains_fresh_checks() {
    let mut worker = Worker {
        capacity: 5,
        observed_capacity: 5,
        shape: 0,
        probes: Vec::new(),
    };
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 4, None);
    for (selected, probes) in [(4, vec![9, 4]), (4, vec![6, 4]), (5, vec![5])] {
        worker.probes.clear();
        assert_eq!(
            prepare_with_search(&mut worker, &mut hint, 15, None, true)
                .0
                .len(),
            selected
        );
        assert_eq!(worker.probes, probes);
    }
    worker.capacity = 2;
    worker.observed_capacity = 2;
    worker.probes.clear();
    assert_eq!(
        prepare_with_search(&mut worker, &mut hint, 15, None, true)
            .0
            .len(),
        2
    );
    assert_eq!(
        worker.probes,
        [5, 2],
        "the old fit does not grant physical capacity"
    );
}

#[test]
fn reduced_fifo_tail_does_not_wait_for_newer_requests_in_its_execution_batch() {
    let mut worker = Worker {
        capacity: 4,
        observed_capacity: 4,
        shape: 0,
        probes: Vec::new(),
    };
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 1, None);
    let shared = shared();
    let mut batch = (0..7).map(|seed| queued(seed, None)).collect::<Vec<_>>();
    {
        let mut state = shared.lock();
        state.preparing = batch.len();
        state.preparing_bytes = batch.iter().map(Queued::pending_bytes).sum();
    }
    scheduler::prepare_capacity_prefix_using(
        &mut worker,
        &mut batch,
        &shared,
        &mut hint,
        &mut batch_efficiency::BatchEfficiency::default(),
        true,
    )
    .unwrap();
    assert_eq!(batch.len(), 4);
    {
        let mut state = shared.lock();
        state.preparing = 0;
        state.preparing_bytes = 0;
        // Cancellation must not remove the surviving tail's boundary.
        state.queue[1].cancelled.store(true, Ordering::Release);
        state.queue.extend((7..11).map(|seed| queued(seed, None)));
    }
    let tail = scheduler::take_batch(&worker, &shared, None).unwrap();
    assert_eq!(
        tail.iter()
            .map(|j| j.request.generation().seed)
            .collect::<Vec<_>>(),
        [4, 6]
    );
    assert_eq!(
        shared.lock().preparing_bytes,
        tail.iter().map(Queued::pending_bytes).sum::<usize>()
    );
    let new = scheduler::take_batch(&worker, &shared, None).unwrap();
    assert_eq!(
        new.iter()
            .map(|j| j.request.generation().seed)
            .collect::<Vec<_>>(),
        [7, 8, 9, 10]
    );
}

struct Worker {
    capacity: usize,
    observed_capacity: usize,
    shape: u8,
    probes: Vec<usize>,
}
impl BatchWorker for Worker {
    type Output = u64;
    fn capacity_search_context(&self, _: &[&QueuedRequest]) -> Option<ImageCapacitySearchContext> {
        Some(ImageCapacitySearchContext {
            shape: [self.shape; 32],
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

fn queued(seed: u64, adapter: Option<Arc<ImageAdapterBatchIdentity>>) -> Queued<u64> {
    Queued {
        request: QueuedRequest::Generate(ImageGenerationRequest {
            prompt: "a".into(),
            width: 32,
            height: 32,
            seed,
            n: 1,
        }),
        adapter,
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

fn prepare(
    worker: &mut Worker,
    hint: &mut CapacitySearchHint,
    count: usize,
    adapter: Option<ImageAdapterBatchIdentity>,
) -> (Vec<u64>, Vec<u64>) {
    prepare_with_search(worker, hint, count, adapter, false)
}

fn prepare_with_search(
    worker: &mut dyn BatchWorker<Output = u64>,
    hint: &mut CapacitySearchHint,
    count: usize,
    adapter: Option<ImageAdapterBatchIdentity>,
    progressive: bool,
) -> (Vec<u64>, Vec<u64>) {
    let shared = shared();
    let adapter = adapter.map(Arc::new);
    let mut batch = (0..count)
        .map(|seed| queued(seed as u64, adapter.clone()))
        .collect::<Vec<_>>();
    {
        let mut state = shared.lock();
        state.preparing = count;
        state.preparing_bytes = batch.iter().map(Queued::pending_bytes).sum();
    }
    let prepared = scheduler::prepare_capacity_prefix_using(
        worker,
        &mut batch,
        &shared,
        hint,
        &mut batch_efficiency::BatchEfficiency::default(),
        progressive,
    )
    .unwrap()
    .into_inner::<Vec<u64>>()
    .unwrap();
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

struct ResidentWorker {
    inner: Worker,
    static_bytes: u64,
    preparation_capacity: Option<u64>,
}

impl BatchWorker for ResidentWorker {
    type Output = u64;
    fn capacity_search_context(
        &self,
        requests: &[&QueuedRequest],
    ) -> Option<ImageCapacitySearchContext> {
        self.inner.capacity_search_context(requests)
    }
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        self.inner.compatible(a, b)
    }
    fn prepare(
        &mut self,
        requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        let count = requests.len() as u64;
        self.inner.prepare(requests).map(|p| {
            p.with_resident_capacity(super::super::super::ImageDeviceCapacityReport {
                complete_batch: true,
                required_bytes: self.static_bytes + count,
                capacity_bytes: self
                    .preparation_capacity
                    .unwrap_or(self.inner.observed_capacity as u64),
                retained_static_bytes: self.static_bytes,
                active_stage_peak_bytes: count,
                retained_dynamic_allowance_bytes: 0,
                native_allowance_bytes: 0,
            })
        })
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        unreachable!("CPU planning fixture")
    }
}

#[test]
fn resident_preference_does_not_become_a_device_limit_or_block_required_swapping() {
    let mut worker = ResidentWorker {
        inner: Worker {
            capacity: 8,
            observed_capacity: 8,
            shape: 0,
            probes: Vec::new(),
        },
        static_bytes: 5,
        preparation_capacity: None,
    };
    let mut hint = CapacitySearchHint::default();
    let selected = prepare_with_search(&mut worker, &mut hint, 8, None, true)
        .0
        .len();
    assert_eq!(selected, 3);
    assert_eq!(hint.upper(worker.capacity_search_context(&[]), &None, 8), 8);
    assert_eq!(
        hint.resident_upper(worker.capacity_search_context(&[]), &None, 8),
        3
    );
    worker.inner.capacity = 16;
    worker.inner.observed_capacity = 16;
    assert!(
        prepare_with_search(&mut worker, &mut hint, 16, None, true)
            .0
            .len()
            > selected
    );
    // Required weights exceed the whole device. Staged work remains legal.
    worker.static_bytes = 17;
    worker.inner.shape = 1;
    let mut hint = CapacitySearchHint::default();
    assert_eq!(
        prepare_with_search(&mut worker, &mut hint, 8, None, true)
            .0
            .len(),
        8
    );
    // The fresh plan may observe less headroom than the preceding context.
    // A soft residency preference must never shrink a legal singleton to zero.
    worker.inner.shape = 2;
    worker.static_bytes = 5;
    worker.preparation_capacity = Some(4);
    assert_eq!(
        prepare_with_search(
            &mut worker,
            &mut CapacitySearchHint::default(),
            1,
            None,
            true
        )
        .0
        .len(),
        1
    );
}

#[test]
fn known_boundary_skips_rejected_shapes_without_turning_short_batches_into_limits() {
    let mut worker = Worker {
        capacity: 3,
        observed_capacity: 3,
        shape: 0,
        probes: Vec::new(),
    };
    let mut hint = CapacitySearchHint::default();
    assert_eq!(prepare(&mut worker, &mut hint, 1, None).0.len(), 1);
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 3);
    assert!(worker.probes.contains(&8), "one success is not a ceiling");
    // A short remaining tail must not erase the proven capacity boundary.
    assert_eq!(prepare(&mut worker, &mut hint, 1, None).0.len(), 1);
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 3);
    assert_eq!(
        worker.probes,
        [3],
        "one fresh preparation, no failed probes"
    );
}

#[test]
fn cached_boundary_still_checks_live_capacity_and_expands_when_capacity_recovers() {
    let mut worker = Worker {
        capacity: 3,
        observed_capacity: 3,
        shape: 0,
        probes: Vec::new(),
    };
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 8, None);
    // The physical state changes after the hint's metadata observation. A
    // cached batch still receives the real prepare failure and shrinks safely.
    worker.capacity = 2;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 2);
    assert!(worker.probes.contains(&3));
    worker.capacity = 6;
    worker.observed_capacity = 6;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 6);
    assert!(worker.probes.iter().any(|&batch| batch > 3));
}

#[test]
fn changing_capacity_reuses_each_known_failure_bound_without_reprobing_it() {
    let mut worker = Worker {
        capacity: 3,
        observed_capacity: 3,
        shape: 0,
        probes: Vec::new(),
    };
    let mut hint = CapacitySearchHint::default();
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 3);
    assert!(worker.probes.contains(&4));
    worker.capacity = 2;
    worker.observed_capacity = 2;
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 2);

    // Restoring capacity reuses the earlier batch-four failure instead of
    // forgetting it and searching again from the entire queued candidate.
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

    // A more distant failed probe remains useful too; bounds are neither a
    // fixed two-capacity table nor a permanent cap on a now-larger device.
    worker.capacity = 8;
    worker.observed_capacity = 8;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 9, None).0.len(), 8);
    assert_eq!(worker.probes, [8]);
}

#[test]
fn shape_and_adapter_changes_do_not_reuse_another_capacity_boundary() {
    let mut worker = Worker {
        capacity: 2,
        observed_capacity: 4,
        shape: 0,
        probes: Vec::new(),
    };
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 8, None);
    worker.capacity = 4;
    worker.shape = 1;
    assert_eq!(prepare(&mut worker, &mut hint, 8, None).0.len(), 4);
    worker.capacity = 6;
    let adapter = ImageAdapterBatchIdentity {
        effective_identity: "different adapter".into(),
        strength_bits: 1f32.to_bits(),
    };
    assert_eq!(prepare(&mut worker, &mut hint, 8, Some(adapter)).0.len(), 6);
}
