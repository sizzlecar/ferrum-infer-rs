use super::*;

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
    let prepared = scheduler::prepare_capacity_prefix(worker, &mut batch, &shared, hint)
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
