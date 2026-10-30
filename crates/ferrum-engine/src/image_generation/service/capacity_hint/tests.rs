use super::*;

struct Worker {
    capacity: usize,
    observed_capacity: usize,
    shape: u8,
    probes: Vec<usize>,
    report_complete_demand: bool,
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
        let prepared = ImageBatchPreparation::new(
            requests
                .iter()
                .map(|request| request.generation().seed)
                .collect::<Vec<_>>(),
        );
        Ok(if self.report_complete_demand {
            prepared.with_capacity_report(demand(requests.len() as u64, self.capacity as u64))
        } else {
            prepared
        })
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
        report_complete_demand: false,
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
        report_complete_demand: false,
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
        report_complete_demand: false,
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
        report_complete_demand: false,
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

fn demand(
    required_bytes: u64,
    capacity_bytes: u64,
) -> super::super::super::ImageDeviceCapacityReport {
    super::super::super::ImageDeviceCapacityReport {
        complete_batch: true,
        required_bytes,
        capacity_bytes,
        retained_static_bytes: 0,
        active_stage_peak_bytes: required_bytes,
        retained_dynamic_allowance_bytes: 0,
        native_allowance_bytes: 0,
    }
}

#[test]
fn complete_demand_skips_distant_probes_but_proves_the_maximum_and_preserves_fifo() {
    let mut worker = Worker {
        capacity: 5,
        observed_capacity: 5,
        shape: 0,
        probes: Vec::new(),
        report_complete_demand: true,
    };
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 1, None);
    prepare(&mut worker, &mut hint, 4, None);
    worker.probes.clear();
    let (selected, tail) = prepare(&mut worker, &mut hint, 15, None);
    assert_eq!(selected.len(), 5);
    assert_eq!(tail.len(), 10);
    assert_eq!(
        worker.probes,
        [5, 6],
        "the estimate is followed by a real boundary probe"
    );

    worker.capacity = 10;
    worker.observed_capacity = 10;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 15, None).0.len(), 10);
    assert_eq!(
        worker.probes,
        [10, 11],
        "more available memory expands the search"
    );
}

#[test]
fn overestimated_capacity_still_prepares_each_prefix_after_live_capacity_falls() {
    let mut worker = Worker {
        capacity: 5,
        observed_capacity: 5,
        shape: 0,
        probes: Vec::new(),
        report_complete_demand: true,
    };
    let mut hint = CapacitySearchHint::default();
    prepare(&mut worker, &mut hint, 1, None);
    prepare(&mut worker, &mut hint, 4, None);
    // Physical storage changes after the metadata observation, before prepare.
    worker.capacity = 2;
    worker.probes.clear();
    assert_eq!(prepare(&mut worker, &mut hint, 15, None).0.len(), 2);
    assert_eq!(worker.probes.first(), Some(&5));
    assert_eq!(worker.probes.last(), Some(&2));
}

#[test]
fn underestimated_batch_does_not_stop_at_the_first_fitting_batch() {
    let context = Some(ImageCapacitySearchContext {
        shape: [0; 32],
        capacity_bytes: 12,
    });
    let mut hint = CapacitySearchHint::default();
    // A previous workspace choice made the larger shape unusually expensive.
    hint.observe_demand(context, &None, 1, demand(1, 12), 64);
    hint.observe_demand(context, &None, 4, demand(8, 12), 64);
    let mut worker = Worker {
        capacity: 12,
        observed_capacity: 12,
        shape: 0,
        probes: Vec::new(),
        report_complete_demand: true,
    };
    assert_eq!(prepare(&mut worker, &mut hint, 15, None).0.len(), 12);
    assert_eq!(worker.probes.first(), Some(&5));
    assert!(
        worker.probes.contains(&13),
        "the fitting estimate is not a capacity bound"
    );
}

#[test]
fn demand_estimates_require_complete_matching_increasing_representable_points() {
    let context = Some(ImageCapacitySearchContext {
        shape: [0; 32],
        capacity_bytes: 9,
    });
    let mut hint = CapacitySearchHint::default();
    hint.observe_demand(context, &None, 1, demand(1, 9), 64);
    let mut partial = demand(4, 9);
    partial.complete_batch = false;
    hint.observe_demand(context, &None, 4, partial, 64);
    assert_eq!(hint.candidate(context, &None, 1, 16, 16), 16);
    for required in [1, 0] {
        hint.observe_demand(context, &None, 4, demand(required, 9), 64);
        assert_eq!(hint.candidate(context, &None, 1, 16, 16), 16);
    }
    hint.observe_demand(context, &None, 4, demand(4, 9), 64);
    assert_eq!(hint.candidate(context, &None, 1, 16, 16), 9);
    let other_shape = Some(ImageCapacitySearchContext {
        shape: [1; 32],
        capacity_bytes: 9,
    });
    let other_adapter = Some(Arc::new(ImageAdapterBatchIdentity {
        effective_identity: "another adapter".into(),
        strength_bits: 1f32.to_bits(),
    }));
    assert_eq!(hint.candidate(other_shape, &None, 1, 16, 16), 16);
    assert_eq!(hint.candidate(context, &other_adapter, 1, 16, 16), 16);

    let huge = Some(ImageCapacitySearchContext {
        shape: [2; 32],
        capacity_bytes: u64::MAX,
    });
    hint.observe_demand(huge, &None, 1, demand(1, u64::MAX), usize::MAX);
    hint.observe_demand(huge, &None, usize::MAX, demand(2, u64::MAX), usize::MAX);
    assert_eq!(hint.candidate(huge, &None, 1, 16, 16), 16);
}
