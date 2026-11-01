use super::*;

struct Worker {
    capacity: usize,
    probes: Vec<usize>,
}

impl BatchWorker for Worker {
    type Output = u64;

    fn capacity_search_context(&self, _: &[&QueuedRequest]) -> Option<ImageCapacitySearchContext> {
        Some(context())
    }

    fn prepare(
        &mut self,
        requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.probes.push(requests.len());
        if requests.len() > self.capacity {
            let proof = crate::image_generation::ImageDeviceCapacityExceeded {
                report: crate::image_generation::ImageDeviceCapacityReport {
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
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        unreachable!("this regression exercises preparation, before device work")
    }
}

fn context() -> ImageCapacitySearchContext {
    ImageCapacitySearchContext {
        shape: [7; 32],
        capacity_bytes: 4,
    }
}

fn queued(seed: u64) -> Queued<u64> {
    Queued {
        request: QueuedRequest::Generate(ImageGenerationRequest {
            prompt: "a".into(),
            width: 32,
            height: 32,
            seed,
            n: 1,
        }),
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

fn prepare(worker: &mut Worker, efficiency: &mut BatchEfficiency) -> (Vec<u64>, Vec<u64>) {
    let mut batch = (0..4).map(queued).collect::<Vec<_>>();
    let shared = Shared {
        state: Mutex::new(State {
            // A newer arrival must stay behind the returned original tail.
            queue: VecDeque::from([queued(4)]),
            queue_backing: None,
            accepting: true,
            closing: false,
            active: 0,
            preparing: batch.len(),
            preparing_bytes: batch.iter().map(Queued::pending_bytes).sum(),
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
            maximum_pending: 8,
            maximum_pending_bytes: 8192,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 4,
            request_batch_max_wait_ms: 0,
        },
        host: None,
        adapter: None,
    };
    let prepared = scheduler::prepare_capacity_prefix(
        worker,
        &mut batch,
        &shared,
        &mut capacity_hint::CapacitySearchHint::default(),
        efficiency,
    )
    .unwrap()
    .into_inner::<Vec<u64>>()
    .unwrap();
    let selected = batch
        .iter()
        .map(|job| job.request.generation().seed)
        .collect::<Vec<_>>();
    assert_eq!(prepared, selected);
    let state = shared.lock();
    assert_eq!(state.preparing, selected.len());
    assert_eq!(
        state.preparing_bytes,
        batch.iter().map(Queued::pending_bytes).sum::<usize>()
    );
    let tail = state
        .queue
        .iter()
        .map(|job| job.request.generation().seed)
        .collect();
    (selected, tail)
}

#[test]
fn efficient_prefix_still_prepares_and_shrinks_on_live_capacity_without_reordering() {
    let mut efficiency = BatchEfficiency::default();
    for (batch_size, milliseconds) in [(2, 2), (4, 8)] {
        efficiency.observe(
            ExecutionContext {
                context: context(),
                adapter: None,
                capacity_upper: 4,
                batch_size,
                exploration: false,
            },
            Duration::from_millis(milliseconds),
            ImageBatchExecutionFeedback {
                stages: vec![ImageBatchStageFeedback {
                    stage: ImageStage::Denoiser,
                    initialization_us: 0,
                    weights_retained: true,
                }],
            },
            4,
        );
    }
    let mut worker = Worker {
        capacity: 4,
        probes: Vec::new(),
    };
    assert_eq!(
        prepare(&mut worker, &mut efficiency),
        (vec![0, 1], vec![2, 3, 4])
    );
    assert_eq!(
        worker.probes,
        [2],
        "selection never replaces fresh preparation"
    );

    // Physical capacity falls after the optimistic observation. Performance
    // learning grants no admission: the selected batch must fail and shrink.
    worker.capacity = 1;
    worker.probes.clear();
    assert_eq!(
        prepare(&mut worker, &mut efficiency),
        (vec![0], vec![1, 2, 3, 4])
    );
    assert_eq!(worker.probes, [2, 1]);
}
