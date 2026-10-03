use super::*;
use parking_lot::Condvar;
use std::sync::mpsc;

#[derive(Default)]
struct Gate {
    pause: bool,
    entered: bool,
    release: bool,
}
#[derive(Default)]
struct Output {
    bytes: Mutex<Vec<u8>>,
    gate: Mutex<Gate>,
    changed: Condvar,
    fail_write: AtomicBool,
    fail_sync: AtomicBool,
}
impl Output {
    fn pause(&self) {
        let mut g = self.gate.lock();
        g.pause = true;
        g.release = false;
        g.entered = false;
    }
    fn wait_paused(&self) {
        let mut g = self.gate.lock();
        while !g.entered {
            assert!(
                !self
                    .changed
                    .wait_for(&mut g, std::time::Duration::from_secs(5))
                    .timed_out(),
                "writer did not enter controlled output"
            );
        }
    }
    fn release(&self) {
        self.gate.lock().release = true;
        self.changed.notify_all();
    }
    fn records(&self) -> Vec<Value> {
        self.bytes
            .lock()
            .split(|b| *b == b'\n')
            .filter(|b| !b.is_empty())
            .map(|b| serde_json::from_slice(b).unwrap())
            .collect()
    }
}
struct TestDestination(Arc<Output>);
impl Write for TestDestination {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut g = self.0.gate.lock();
        if g.pause && !g.release {
            g.entered = true;
            self.0.changed.notify_all();
            while !g.release {
                self.0.changed.wait(&mut g);
            }
        }
        drop(g);
        if self.0.fail_write.load(Ordering::Acquire) {
            // A real write may leave a prefix. Such a file must have no usable footer.
            self.0
                .bytes
                .lock()
                .extend_from_slice(&bytes[..bytes.len().min(7)]);
            return Err(io::Error::other("injected short device write"));
        }
        self.0.bytes.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Destination for TestDestination {
    fn sync(&mut self) -> io::Result<()> {
        if self.0.fail_sync.load(Ordering::Acquire) {
            Err(io::Error::other("injected sync failure"))
        } else {
            Ok(())
        }
    }
}
fn limits() -> SloRequiredQueryObservationLimits {
    SloRequiredQueryObservationLimits {
        queue_events: 256,
        queue_bytes: 1024 * 1024,
        max_event_bytes: 64 * 1024,
        max_transaction_events: 1024,
        max_events: 10000,
        max_file_bytes: 1024 * 1024,
    }
}
fn paused(limits: SloRequiredQueryObservationLimits) -> (Arc<Writer>, Arc<Output>) {
    let output = Arc::new(Output::default());
    let writer = Writer::start(limits, Box::new(TestDestination(output.clone()))).unwrap();
    output.pause();
    writer
        .bind_identity(&ferrum_types::SloConfig::default(), None)
        .unwrap();
    output.wait_paused();
    (writer, output)
}
fn finish(t: &Transaction) {
    t.finish(
        "observed",
        "unknown",
        "cost_unavailable",
        1234,
        2_000_000,
        false,
        false,
        &PlanningSearchStats::default(),
    );
}
fn lookup(t: &Transaction, key: PlanningQueryKey) {
    t.lookup(
        key,
        u64::MAX - 3,
        PlanningObservedCost {
            outcome: PlanningQueryOutcome::Known(PlanningCost {
                typical_ns: 31,
                planning_ns: 41,
                model_version: 17,
                valid_for_ns: 5,
            }),
            cost_now_ns: Some(u64::MAX - 9),
        },
    );
}

// Real canonical command/host construction and physical-domain validation.
// No serialized family key is used to create a checked query.
fn identity_query(algorithm: u8, context_limit: u32, policy: u8) -> StructuredQueryV2 {
    identity_query_algorithms(&[algorithm], context_limit, policy)
}

fn identity_query_algorithms(
    algorithms: &[u8],
    context_limit: u32,
    policy: u8,
) -> StructuredQueryV2 {
    use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
    use std::num::{NonZeroU32, NonZeroU64};
    let domain = CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(8).unwrap(),
            maximum_context_tokens: NonZeroU32::new(context_limit).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(8).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(4096).unwrap(),
            repetition_slot_capacity: 512,
            fixed_state_bytes_per_row: 32,
        },
    )
    .unwrap();
    let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    for &algorithm in algorithms {
        selected
            .kernel(
                SelectedAlgorithmClassV1::new(
                    "fixture.query-identity",
                    1,
                    [algorithm; 32],
                    [2; 32],
                )
                .unwrap(),
                KernelNumericWorkV1 {
                    logical_units: 8,
                    padded_units: 8,
                    inner_units_per_logical_unit: 2,
                    grid: [1, 1, 1],
                    scratch_bytes: 32,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
    }
    let selected = selected.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::GreedyToken);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.query-identity",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(CostProviderIdentity {
                provider_id: "numerical-fixture",
                implementation_fingerprint: "v1",
                operation_fingerprint: "v1",
            }),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            batching_form: "packed",
            compute_dispatch_count: algorithms.len() as u64,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&selected),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 64 },
            host_policy_signature: [3; 32],
            mask_upload_required: false,
            output: CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1f32.to_bits(),
            },
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: [policy; 32],
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 3,
                    maximum_output_tokens: 21,
                    sampling_history_tokens: 3,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    let wave = builder
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            32,
        )
        .unwrap();
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    StructuredQueryV2::from_future_with_domain(
        &wave.exact,
        selected,
        recipe,
        &HostContentForecastV2::Exact,
        &domain,
    )
    .unwrap()
}

#[test]
fn required_query_writer_preserves_checked_family_algorithm_and_physical_identity() {
    use ferrum_interfaces::execution_cost::{AlgorithmWorkKindV1, SelectedAlgorithmClassV1};
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;

    let mut queries = vec![
        identity_query(1, 1024, 4),
        identity_query(2, 1024, 4),
        identity_query(1, 2048, 4),
        identity_query(1, 1024, 5),
    ];
    let universe =
        DeclaredAlgorithmUniverseV1::from_inputs([queries[0].input(), queries[1].input()], 4096)
            .unwrap();
    queries.push(
        queries[0]
            .clone()
            .with_algorithm_universe(&universe)
            .unwrap(),
    );
    let original: Vec<_> = queries
        .iter()
        .map(|q| q.input().numerical_family_key().unwrap())
        .collect();
    let (writer, output) = paused(limits());
    let t = writer.begin();
    for (alternative, query) in queries.iter().enumerate() {
        t.constructed(
            PlanningQueryKey {
                attempt: 1,
                alternative,
            },
            Ok(query),
        );
    }
    finish(&t);
    // The producer finishes with the output thread blocked; all new identity
    // extraction/serialization occurs on that existing bounded writer.
    output.release();
    writer.close().unwrap();
    let records = output.records();
    let identities: Vec<_> = records
        .iter()
        .filter(|r| r["event"] == "query_constructed")
        .map(|r| &r["data"]["identity"])
        .collect();
    assert_eq!(identities.len(), queries.len());
    for (record, algorithm) in records
        .iter()
        .filter(|r| r["event"] == "query_constructed")
        .zip([1, 2, 1, 1, 1])
    {
        let selected =
            SelectedAlgorithmClassV1::new("fixture.query-identity", 1, [algorithm; 32], [2; 32])
                .unwrap();
        assert_eq!(
            record["data"]["algorithm_axes"],
            json!([{
                "signature": selected.signature(), "kind": AlgorithmWorkKindV1::Kernel as u8,
            }])
        );
        // Alignment to A+B must still report this query's actual A alone.
        let prebound = record["data"]["alternative"] == 4;
        assert_eq!(
            record["data"]["prebound_algorithm_universe"],
            if prebound {
                json!(universe.signature())
            } else {
                Value::Null
            }
        );
    }
    for ((identity, query), family) in identities.iter().zip(&queries).zip(&original) {
        assert_eq!(
            identity["numerical_family"],
            serde_json::to_value(family).unwrap()
        );
        assert!(identity["numerical_family_error"].is_null());
        assert_eq!(
            identity["physical_domain_signature"],
            serde_json::to_value(query.input().physical_domain_signature()).unwrap()
        );
        assert_eq!(
            identity["numerical_family"]["workload_domain"],
            identity["physical_domain_signature"]
        );
        assert_eq!(
            identity["numerical_family"]["algorithm_domain"],
            query.input().algorithm_universe_signature().map_or_else(
                || identity["owner"]["algorithm_domain"].clone(),
                |u| json!(u)
            )
        );
        for (name, policy) in [
            (
                "installed_algorithm_set_identity",
                StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
            ),
            (
                "ordered_identity",
                StructuredCostTemplatePolicyV1::OrderedV1,
            ),
        ] {
            let (owner, domain) = query.input().cost_template_identity(policy).unwrap();
            assert_eq!(
                identity[name]["owner"],
                serde_json::to_value(owner).unwrap()
            );
            assert_eq!(
                identity[name]["domain_signature"],
                serde_json::to_value(domain).unwrap()
            );
        }
        assert_eq!(query.input().numerical_family_key().unwrap(), *family);
    }
    let a = identities[0];
    assert_ne!(
        a["owner"]["algorithm_domain"],
        identities[1]["owner"]["algorithm_domain"]
    );
    assert_eq!(
        a["physical_domain_signature"],
        identities[1]["physical_domain_signature"]
    );
    assert_ne!(
        a["physical_domain_signature"],
        identities[2]["physical_domain_signature"]
    );
    assert_eq!(
        a["owner"]["algorithm_domain"],
        identities[2]["owner"]["algorithm_domain"]
    );
    assert_ne!(
        a["numerical_family"]["host_policy"],
        identities[3]["numerical_family"]["host_policy"]
    );
    assert_eq!(
        a["owner"]["algorithm_domain"],
        identities[3]["owner"]["algorithm_domain"]
    );
    assert_eq!(
        a["physical_domain_signature"],
        identities[3]["physical_domain_signature"]
    );
    assert_eq!(writer.snapshot()["recording_complete"], true);
}

#[test]
fn required_query_writer_algorithm_roster_respects_exact_event_byte_limit() {
    struct ReleaseOutput(Arc<Output>);
    impl Drop for ReleaseOutput {
        fn drop(&mut self) {
            self.0.release();
        }
    }
    // A real checked multi-command roster makes wire bytes, rather than the
    // retained query, the limiting resource. Clone once in the fixture to make
    // the producer's original Vec capacities compact too: admission estimates
    // those capacities BEFORE making its own compact queued clone.
    let original = identity_query_algorithms(&(1..=32).collect::<Vec<_>>(), 1024, 4);
    let query = original.clone();
    let key = PlanningQueryKey {
        attempt: 1,
        alternative: 0,
    };
    let entry = Entry {
        ordinal: 3,
        retained_ordinal: 3,
        transaction: 1,
        event: Event::Query(key, Ok(query.clone())),
        bytes: 0,
    };
    let retained = entry.event.retained_bytes().unwrap();
    let mut line = BoundedLine {
        bytes: Vec::new(),
        limit: limits().max_event_bytes,
    };
    entry.event.encode(&entry, &mut line).unwrap();
    line.write_all(b"\n").unwrap();
    let exact = line.bytes.len();
    let estimated = query.observation_retained_bytes().unwrap() + size_of::<Entry>();
    assert!(retained <= estimated);
    assert!(estimated < exact - 1);
    assert!(retained < exact - 1);
    assert!(query.observation_demand_scratch_bytes().unwrap() < exact - 1);
    for (limit, succeeds) in [(exact, true), (exact - 1, false)] {
        let mut l = limits();
        l.max_event_bytes = limit;
        let output = Arc::new(Output::default());
        let writer = Writer::start(l, Box::new(TestDestination(output.clone()))).unwrap();
        // Installed before any paused operation/assertion; unwinding releases
        // output before Writer::drop joins its worker.
        let _release = ReleaseOutput(output.clone());
        output.pause();
        writer
            .bind_identity(&ferrum_types::SloConfig::default(), None)
            .unwrap();
        output.wait_paused();
        let retained_before = writer.snapshot()["retained_bytes"].as_u64().unwrap();
        let t = writer.begin();
        let accepted_before = writer.snapshot()["statistics"]["accepted"]
            .as_u64()
            .unwrap();
        t.constructed(key, Ok(&query));
        assert_eq!(
            writer.snapshot()["statistics"]["accepted"]
                .as_u64()
                .unwrap(),
            accepted_before + 1
        );
        // No extra roster allocation accompanies the original queued clone.
        assert_eq!(
            writer.snapshot()["retained_bytes"].as_u64().unwrap() - retained_before,
            (retained - size_of::<Entry>()) as u64
        );
        finish(&t);
        output.release();
        assert_eq!(writer.close().is_ok(), succeeds);
        let h = writer.snapshot();
        assert_eq!(h["recording_complete"], succeeds);
        assert_eq!(h["retained_events"], 0);
        assert_eq!(h["statistics"]["lost"]["capacity"], 0);
        assert_eq!(h["statistics"]["lost"]["encoding"], u64::from(!succeeds));
        let records = output.records();
        let queries: Vec<_> = records
            .iter()
            .filter(|r| r["event"] == "query_constructed")
            .collect();
        assert_eq!(queries.len(), usize::from(succeeds));
        if succeeds {
            assert_eq!(
                queries[0]["data"]["algorithm_axes"]
                    .as_array()
                    .unwrap()
                    .len(),
                32
            );
        }
        assert_eq!(records.last().unwrap()["recording_complete"], succeeds);
    }
}

#[test]
fn controller_checkpoints_remain_nonblocking_and_keep_original_transaction_budget() {
    let (writer, output) = paused(limits());
    let t = writer.begin();
    let transaction = t.id();
    for (edge, elapsed_ns) in [("begin", 400_000), ("end", 730_000)] {
        t.checkpoint(ControllerCheckpoint {
            stage: "candidate_projection",
            edge,
            elapsed_ns: Some(elapsed_ns),
            hard_budget_ns: 2_000_000,
            optional_deadline_elapsed_ns: Some(1_250_000),
            completion_preparation_ns: Some(350_000),
            publication_reserve_ns: Some(400_000),
        });
    }
    finish(&t);
    // The producer returned while the real output writer was held blocked.
    output.release();
    writer.close().unwrap();
    let records = output.records();
    let points: Vec<_> = records
        .iter()
        .filter(|r| r["event"] == "controller_checkpoint")
        .collect();
    assert_eq!(points.len(), 2);
    assert!(points.iter().all(|r| r["transaction"] == transaction));
    assert_eq!(points[0]["data"]["edge"], "begin");
    assert_eq!(points[1]["data"]["elapsed_ns"], 730_000);
    assert_eq!(points[1]["data"]["hard_budget_ns"], 2_000_000);
    assert_eq!(points[1]["data"]["optional_deadline_elapsed_ns"], 1_250_000);
    assert_eq!(writer.snapshot()["recording_complete"], true);
}

#[test]
fn required_query_writer_preserves_original_clock_and_explicit_not_queried_and_replay_identity() {
    let (writer, output) = paused(limits());
    let t = writer.begin();
    let a = t.begin_attempt(PlanningQueryPhase::Search, 0, &[], &[]);
    t.constructed(
        PlanningQueryKey {
            attempt: a,
            alternative: 0,
        },
        Err(StructuredUnknownV2::MissingEvidence),
    );
    t.constructed(
        PlanningQueryKey {
            attempt: a,
            alternative: 1,
        },
        Err(StructuredUnknownV2::MissingEvidence),
    );
    lookup(
        &t,
        PlanningQueryKey {
            attempt: a,
            alternative: 0,
        },
    );
    t.end_attempt(
        a,
        2,
        1,
        PlanningQueryAttemptEnd::Unknown(PlanningUnknownReason::CostUnavailable),
    );
    let first = t.begin_replay(2);
    t.end_replay(
        first,
        PlanningQueryAttemptEnd::Unknown(PlanningUnknownReason::ComputeBudgetExhausted),
    );
    let second = t.begin_replay(1);
    let b = t.begin_attempt(
        PlanningQueryPhase::IndependentReplay { replay: second },
        0,
        &[],
        &[],
    );
    t.end_attempt(b, 0, 0, PlanningQueryAttemptEnd::Completed);
    t.end_replay(second, PlanningQueryAttemptEnd::Completed);
    t.selected_replay(second);
    finish(&t);
    drop(t);
    output.release();
    writer.close().unwrap();
    writer.close().unwrap();
    assert_eq!(writer.snapshot()["recording_complete"], true);
    let records = output.records();
    assert_eq!(records.first().unwrap()["event"], "run_header");
    assert_eq!(records[1]["event"], "run_identity");
    for (i, r) in records[1..records.len() - 1].iter().enumerate() {
        assert_eq!(r["retained_ordinal"].as_u64(), Some(i as u64 + 1));
    }
    let lookup = records
        .iter()
        .find(|r| r["event"] == "query_lookup")
        .unwrap();
    assert_eq!(
        lookup["data"]["planning_now_ns"].as_u64(),
        Some(u64::MAX - 3)
    );
    assert_eq!(lookup["data"]["cost_now_ns"].as_u64(), Some(u64::MAX - 9));
    assert_eq!(lookup["data"]["outcome"]["cost"]["model_version"], 17);
    assert_eq!(lookup["data"]["outcome"]["cost"]["valid_for_ns"], 5);
    assert_eq!(
        records
            .iter()
            .filter(|r| r["event"] == "query_constructed")
            .count(),
        2
    );
    for record in records.iter().filter(|r| r["event"] == "query_constructed") {
        assert!(record["data"]["algorithm_axes"].is_null());
        assert!(record["data"]["prebound_algorithm_universe"].is_null());
    }
    assert_eq!(
        records
            .iter()
            .filter(|r| r["event"] == "query_lookup")
            .count(),
        1
    );
    let end = records
        .iter()
        .find(|r| r["event"] == "attempt_end")
        .unwrap();
    assert_eq!(end["data"]["not_queried_start"], 1);
    assert_eq!(end["data"]["not_queried_end"], 2);
    let selected = records
        .iter()
        .find(|r| r["event"] == "selected_replay")
        .unwrap();
    assert_eq!(selected["data"]["replay"].as_u64(), Some(second));
    assert_ne!(first, second);
    assert_eq!(records.last().unwrap()["recording_complete"], true);
}

#[test]
fn required_query_writer_queue_inflight_and_byte_bounds_reject_without_constructing_payload() {
    let mut l = limits();
    l.queue_events = 1;
    let (writer, output) = paused(l);
    assert_eq!(
        writer.snapshot()["retained_events"],
        1,
        "in-flight identity still occupies capacity"
    );
    assert!(!writer.shared.offer(
        7,
        Some(1),
        Some(size_of::<Entry>()),
        || panic!("full queue must not build payload"),
        false
    ));
    assert!(!writer.shared.offer(
        7,
        Some(2),
        Some(usize::MAX),
        || panic!("oversized event must not build payload"),
        false
    ));
    output.release();
    assert!(writer.close().is_err());
    let health = writer.snapshot();
    assert_eq!(health["statistics"]["lost"]["capacity"], 2);
    assert_eq!(health["retained_events"], 0);
    assert_eq!(health["recording_complete"], false);
    assert_eq!(
        output.records().last().unwrap()["recording_complete"],
        false
    );
}

#[test]
fn required_query_writer_concurrent_producers_worker_and_health_preserve_all_admitted_events() {
    let (writer, output) = paused(limits());
    let start = Arc::new(std::sync::Barrier::new(5));
    let mut producers = Vec::new();
    for producer in 0..4 {
        let writer = writer.clone();
        let start = start.clone();
        producers.push(thread::spawn(move || {
            start.wait();
            for sequence in 0..32 {
                assert!(writer.shared.offer(
                    producer + 1,
                    Some(sequence + 1),
                    Some(size_of::<Entry>()),
                    || Some(Event::ReplayBegin {
                        replay: sequence + 1,
                        waves: 1,
                    }),
                    false
                ));
                assert!(
                    writer.snapshot()["retained_bytes"].as_u64().unwrap()
                        <= writer.shared.limits.queue_bytes as u64
                );
            }
        }));
    }
    start.wait();
    // Worker writes and releases concurrently with all producers and health
    // snapshots. Total offered population fits even if no consumption occurs.
    output.release();
    for producer in producers {
        producer.join().unwrap();
    }
    writer.close().unwrap();
    let records = output.records();
    let mut last = [0; 4];
    for (index, record) in records
        .iter()
        .filter(|r| r["retained_ordinal"].is_u64())
        .enumerate()
    {
        assert_eq!(record["retained_ordinal"].as_u64(), Some(index as u64 + 1));
        if record["event"] == "replay_begin" {
            let owner = record["transaction"].as_u64().unwrap() as usize - 1;
            last[owner] += 1;
            assert_eq!(record["data"]["replay"].as_u64(), Some(last[owner]));
        }
    }
    assert_eq!(last, [32; 4]);
    let health = writer.snapshot();
    assert_eq!(health["statistics"]["offered"], 129);
    assert_eq!(health["statistics"]["accepted"], 129);
    assert_eq!(health["statistics"]["written"], 129);
    assert_eq!(health["statistics"]["lost"]["contention"], 0);
    assert_eq!(health["retained_events"], 0);
    assert_eq!(health["recording_complete"], true);
}

#[test]
fn required_query_writer_close_waits_for_admitted_unpublished_producer() {
    let output = Arc::new(Output::default());
    let writer = Writer::start(limits(), Box::new(TestDestination(output.clone()))).unwrap();
    writer
        .bind_identity(&ferrum_types::SloConfig::default(), None)
        .unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let producer_writer = writer.clone();
    let producer = thread::spawn(move || {
        producer_writer.shared.offer(
            7,
            Some(1),
            Some(size_of::<Entry>()),
            || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Some(Event::Begin)
            },
            false,
        )
    });
    entered_rx.recv().unwrap();
    assert_eq!(writer.shared.gate.load(Ordering::Acquire) & !CLOSED, 1);
    let closing_writer = writer.clone();
    let (done_tx, done_rx) = mpsc::channel();
    let closing = thread::spawn(move || {
        done_tx
            .send(closing_writer.close().map_err(|e| e.to_string()))
            .unwrap();
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while writer.shared.gate.load(Ordering::Acquire) & CLOSED == 0 {
        assert!(std::time::Instant::now() < deadline);
        thread::yield_now();
    }
    assert!(matches!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    assert_eq!(writer.snapshot()["statistics"]["footer_written"], false);
    release_tx.send(()).unwrap();
    assert!(producer.join().unwrap());
    done_rx.recv().unwrap().unwrap();
    closing.join().unwrap();
    let rows = output.records();
    assert_eq!(
        rows.iter()
            .filter(|r| r["event"] == "transaction_begin")
            .count(),
        1
    );
    assert_eq!(rows.last().unwrap()["recording_complete"], true);
    assert_eq!(writer.snapshot()["retained_events"], 0);
    assert!(!writer.shared.offer(
        8,
        Some(1),
        Some(size_of::<Entry>()),
        || panic!("closed producer must not construct"),
        false
    ));
    assert_eq!(writer.snapshot()["statistics"]["lost"]["closed"], 1);
    assert!(writer.close().is_err());
}

#[test]
fn required_query_writer_panicking_payload_releases_reservation_and_close_gate() {
    let (writer, output) = paused(limits());
    let original_bytes = writer.snapshot()["retained_bytes"].as_u64().unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        writer.shared.offer(
            1,
            Some(1),
            Some(size_of::<Entry>() + 100),
            || panic!("payload construction panic"),
            false,
        )
    }));
    assert!(panic.is_err());
    assert_eq!(writer.shared.gate.load(Ordering::Acquire), 0);
    assert_eq!(writer.snapshot()["retained_events"], 1);
    assert_eq!(
        writer.snapshot()["retained_bytes"].as_u64(),
        Some(original_bytes)
    );
    output.release();
    assert!(writer.close().is_err(), "offered event was not written");
    assert_eq!(writer.snapshot()["retained_events"], 0);
}

#[test]
fn required_query_writer_close_waits_for_accepted_output_and_repeated_call_retains_failure() {
    let (writer, output) = paused(limits());
    let t = writer.begin();
    finish(&t);
    drop(t);
    output.fail_sync.store(true, Ordering::Release);
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let closing = writer.clone();
    let handle = thread::spawn(move || {
        started_tx.send(()).unwrap();
        done_tx.send(closing.close().is_err()).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(matches!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    output.release();
    assert!(done_rx.recv().unwrap());
    handle.join().unwrap();
    assert!(writer.close().is_err());
    let h = writer.snapshot();
    assert_eq!(h["recording_complete"], false);
    assert_eq!(h["statistics"]["writer_failed"], true);
    assert_eq!(h["statistics"]["flushed_and_synced"], false);
    assert!(h["statistics"]["first_error"]
        .as_str()
        .unwrap()
        .contains("sync"));
}

#[test]
fn required_query_writer_partial_write_cannot_publish_a_success_footer() {
    let (writer, output) = paused(limits());
    let t = writer.begin();
    finish(&t);
    drop(t);
    output.fail_write.store(true, Ordering::Release);
    output.release();
    assert!(writer.close().is_err());
    assert!(writer.close().is_err());
    let h = writer.snapshot();
    assert_eq!(h["recording_complete"], false);
    assert_eq!(h["statistics"]["footer_written"], false);
    assert_eq!(h["retained_events"], 0);
    assert!(h["statistics"]["lost"]["writer_failed"].as_u64().unwrap() > 0);
    assert!(!String::from_utf8(output.bytes.lock().clone())
        .unwrap()
        .contains("run_footer"));
}

#[test]
fn required_query_writer_transaction_and_file_limits_preserve_incomplete_footer() {
    let mut l = limits();
    l.max_transaction_events = 2;
    let (writer, output) = paused(l);
    let t = writer.begin();
    lookup(
        &t,
        PlanningQueryKey {
            attempt: 1,
            alternative: 0,
        },
    );
    finish(&t);
    drop(t);
    output.release();
    assert!(writer.close().is_err());
    assert_eq!(
        writer.snapshot()["statistics"]["lost"]["transaction_limit"],
        1
    );
    assert_eq!(
        output.records().last().unwrap()["recording_complete"],
        false
    );

    let mut l = limits();
    l.max_file_bytes = 32 * 1024;
    let file_limit = l.max_file_bytes;
    let (writer, output) = paused(l);
    let t = writer.begin();
    for i in 0..100 {
        lookup(
            &t,
            PlanningQueryKey {
                attempt: 1,
                alternative: i,
            },
        );
    }
    finish(&t);
    drop(t);
    output.release();
    assert!(writer.close().is_err());
    assert!(output.bytes.lock().len() as u64 <= file_limit);
    let footer = output.records().pop().unwrap();
    assert_eq!(footer["event"], "run_footer");
    assert_eq!(footer["recording_complete"], false);
    assert!(footer["statistics"]["lost"]["capacity"].as_u64().unwrap() > 0);
}

#[test]
fn required_query_writer_abandonment_and_post_finish_callbacks_are_not_silent() {
    let (writer, output) = paused(limits());
    let abandoned = writer.begin();
    drop(abandoned);
    let done = writer.begin();
    finish(&done);
    lookup(
        &done,
        PlanningQueryKey {
            attempt: 1,
            alternative: 0,
        },
    );
    drop(done);
    output.release();
    assert!(writer.close().is_err());
    let h = writer.snapshot();
    assert_eq!(h["statistics"]["abandoned_transactions"], 1);
    assert_eq!(h["statistics"]["active_transactions"], 0);
    assert_eq!(h["statistics"]["lost"]["closed"], 1);
    assert_eq!(h["recording_complete"], false);
}

#[test]
fn required_query_writer_disabled_and_create_new_preserve_original_file() {
    assert!(Writer::open(&SloRequiredQueryObservationConfig::Disabled)
        .unwrap()
        .is_none());
    let path = std::env::temp_dir().join(format!(
        "ferrum-query-writer-{}.jsonl",
        uuid::Uuid::new_v4()
    ));
    std::fs::write(&path, b"original evidence\n").unwrap();
    let config = SloRequiredQueryObservationConfig::StructuredRequiredV1 {
        path: path.clone(),
        limits: limits(),
    };
    assert!(Writer::open(&config).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"original evidence\n");
    std::fs::remove_file(&path).unwrap();
    let writer = Writer::open(&config).unwrap().unwrap();
    writer
        .bind_identity(&ferrum_types::SloConfig::default(), None)
        .unwrap();
    writer.close().unwrap();
    let records = std::fs::read_to_string(&path).unwrap();
    assert!(records.contains("run_identity") && records.contains("run_footer"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn uncalibrated_required_query_writer_labels_absent_model_and_keeps_complete_footer() {
    let path = std::env::temp_dir().join(format!(
        "ferrum-uncalibrated-query-{}.jsonl",
        uuid::Uuid::new_v4()
    ));
    let mut config = ferrum_types::SloConfig::default();
    config.mode = ferrum_types::SloMode::Observe;
    config.default_service_class = Some("fixture".into());
    config.services.push(ferrum_types::ServiceSloConfig {
        id: "fixture".into(),
        server_token_commit: ferrum_types::SloLatencyBudgets {
            ttft_ms: std::num::NonZeroU64::new(500).unwrap(),
            tpot_ms: std::num::NonZeroU64::new(40).unwrap(),
            itl_ms: std::num::NonZeroU64::new(80).unwrap(),
        },
        attainment: Default::default(),
        client_visible: None,
    });
    config.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    config.required_query_observation =
        SloRequiredQueryObservationConfig::StructuredUncalibratedV1 {
            path: path.clone(),
            limits: limits(),
        };
    config.prefill_reference = Some(ferrum_types::SloPrefillReferenceConfig {
        artifact_path: "fixture-reference.json".into(),
        expected_protocol_sha256: [1; 32],
        limits: Default::default(),
    });
    config.validate().unwrap();
    let writer = Writer::open(&config.required_query_observation)
        .unwrap()
        .unwrap();
    writer.bind_identity(&config, None).unwrap();
    writer.close().unwrap();
    let records: Vec<Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let identity = records
        .iter()
        .find(|row| row["event"] == "run_identity")
        .unwrap();
    assert!(identity["data"]["cost_profile_receipt"].is_null());
    assert!(identity["data"]["uncalibrated_scope"]
        .as_str()
        .unwrap()
        .contains("actual_root_candidates_only"));
    assert!(identity["data"]["uncalibrated_scope"]
        .as_str()
        .unwrap()
        .contains("unknown_stops_each_edge"));
    assert_eq!(records.last().unwrap()["recording_complete"], true);
    // A complete empty recording is not query-coverage evidence.
    assert_eq!(writer.snapshot()["statistics"]["accepted"], 1);
    std::fs::remove_file(path).unwrap();
}
