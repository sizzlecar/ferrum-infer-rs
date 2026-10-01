use super::*;

#[test]
#[ignore = "requires the original source8 checkpoint plus runtime source via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_original_source8_through_production_joint_bank() {
    let spec: AuditSpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").unwrap()).unwrap(),
    )
    .unwrap();
    let limits = CostProfileLoadLimits::default();
    let mut bytes = Vec::new();
    std::fs::File::open(&spec.checkpoint_source)
        .unwrap()
        .take(spec.checkpoint_bytes as u64)
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes.len(), spec.checkpoint_bytes);
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&bytes)),
        spec.checkpoint_sha256
    );
    let original = header(&bytes);
    let original_checkpoint = replay_structured_source_v8(&bytes, &limits).unwrap();
    let original_children = original_checkpoint.qualified_children();
    let mut declaration = original.declaration.clone();
    declaration
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1;
    let mut digest = Sha256::new();
    digest.update(b"ferrum.offline.source8-joint-bank.v1\0");
    digest.update(original.capture_identity);
    let h = StructuredPreparedOwnerBlockHeaderV8::new(
        digest.finalize().into(),
        original.generation,
        original.fingerprint.clone(),
        original.producer.clone(),
        original.opening,
        declaration,
        original.maximum_file_bytes,
    )
    .unwrap();
    let h = match &original.monotonic_domain {
        Some(d) => h.with_monotonic_domain(d.clone()).unwrap(),
        None => h,
    };
    let budget = NonZeroU64::new(h.maximum_file_bytes).unwrap();
    let mut candidate =
        StructuredPreparedOwnerBlockCollectorV8::new_streaming(h.clone(), limits.clone(), budget)
            .unwrap();
    let mut replay =
        StructuredPreparedOwnerBlockCollectorV8::new_streaming(h.clone(), limits.clone(), budget)
            .unwrap();
    let mut phase_counts = Vec::new();
    let mut checkpoint_verified = false;
    use StructuredPreparedOwnerBlockRecordV8 as R;
    for line in bytes.split_inclusive(|b| *b == b'\n').skip(1) {
        let record: R = serde_json::from_slice(line).unwrap();
        let generated = match &record {
            R::Population(StructuredServiceRecordV7::BlockOpen {
                opened_at_ns,
                fifo_cutoff,
                ..
            }) => candidate.open_block(*opened_at_ns, *fifo_cutoff).unwrap(),
            R::Population(StructuredServiceRecordV7::BlockClose { block, closing, .. }) => {
                let before: Vec<_> = candidate
                    .population
                    .owners
                    .iter()
                    .filter_map(|o| {
                        let counts = match &o.state {
                            OwnerState::Fitted(m) => m.diagnose_joint_cell_inputs(&o.samples),
                            OwnerState::Calibrated(m) => m.diagnose_joint_cell_inputs(&o.samples),
                            _ => None,
                        }?;
                        Some(json!({"owner":o.contract.owner_attempt_id,"counts":counts}))
                    })
                    .collect();
                let generated = candidate.close_block(*closing).unwrap();
                if let R::Population(StructuredServiceRecordV7::BlockClose { freezes, .. }) =
                    &generated
                {
                    for freeze in freezes {
                        let owner = candidate
                            .population
                            .owners
                            .iter()
                            .find(|o| o.contract.owner_attempt_id == freeze.owner_attempt_id)
                            .unwrap();
                        let bank = match &owner.state {
                            OwnerState::Calibrated(m) => m.diagnose_joint_cell_bank(),
                            OwnerState::Qualified(m) => m.diagnose_joint_cell_bank(),
                            _ => None,
                        };
                        phase_counts.push(json!({"block":block,"owner":freeze.owner_attempt_id,
                            "phase":freeze.close.phase,"members":freeze.close.member_count,
                            "failure":freeze.failure,"bank":bank,"input_counts_before_close":before}));
                    }
                }
                generated
            }
            R::Population(StructuredServiceRecordV7::Checkpoint { closing, .. }) => {
                let (generated, checkpoint) = candidate.checkpoint(*closing).unwrap();
                replay.push(&generated).unwrap();
                assert_eq!(candidate.source_receipt(), replay.source_receipt());
                if checkpoint.qualified_children() > 0 {
                    let independent = StructuredPreparedOwnerBlockCheckpointV8 {
                        header: h.clone(),
                        population: StructuredServiceCheckpointV7::from_collector(
                            &replay.population,
                            *closing,
                        )
                        .unwrap(),
                    };
                    let a = checkpoint
                        .activate_same_process_memory_streaming(*closing, &limits, budget)
                        .unwrap();
                    let b = independent
                        .activate_same_process_memory_streaming(*closing, &limits, budget)
                        .unwrap();
                    assert_eq!(
                        a.children
                            .iter()
                            .map(|m| m.parameters_signature())
                            .collect::<Vec<_>>(),
                        b.children
                            .iter()
                            .map(|m| m.parameters_signature())
                            .collect::<Vec<_>>()
                    );
                    checkpoint_verified = true;
                }
                continue;
            }
            R::Tail(StructuredPreparedTailRecordV8::PartialTailClosed { tail }) => candidate
                .seal_complete_cohorts_with_partial_tail(tail.closing)
                .unwrap()
                .unwrap(),
            R::Population(StructuredServiceRecordV7::Failed {
                ticket,
                fifo,
                at_ns,
                reason,
                ..
            }) => candidate
                .fail(*ticket, *fifo, *at_ns, reason.clone())
                .unwrap(),
            R::Population(StructuredServiceRecordV7::Footer { closing, .. }) => {
                candidate.stop(*closing).unwrap()
            }
            _ => {
                candidate.push(&record).unwrap();
                record.clone()
            }
        };
        replay.push(&generated).unwrap();
        assert_eq!(candidate.source_receipt(), replay.source_receipt());
    }
    let runtime = seeded::verify_source_file(&spec.actual_source, spec.actual_source_sha256);
    let mut contract = runtime.declaration.nonnegative_envelope.clone().unwrap();
    if runtime.declaration.schedule.algorithm_universe
        == Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
    {
        contract.algorithm_universe = None;
    }
    let models: Vec<_> = candidate
        .population
        .owners
        .iter()
        .filter_map(|o| match &o.state {
            OwnerState::Qualified(m) => Some(m.clone()),
            _ => None,
        })
        .collect();
    let mut offered = 0usize;
    let mut known = 0usize;
    let mut misses = 0usize;
    let mut paired_known = Vec::new();
    for line in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        let StructuredServiceRecordV7::Completed { wave } = record else {
            continue;
        };
        let (prepared, offered_rows) =
            physical::original_prepared(&wave.host_stages, wave.independent.as_ref()).unwrap();
        let query = StructuredQueryV2::exact(
            input_replay::project_service_actual_with_domain(
                &prepared,
                &offered_rows,
                &contract.workload_domain,
            )
            .unwrap(),
        );
        if query.input().owner().rows != 1 {
            continue;
        }
        offered += 1;
        let prediction = models.iter().find_map(|m| {
            m.predict_query(
                &runtime.fingerprint.clone().into(),
                &query,
                wave.issued_at_ns,
            )
            .ok()
        });
        if let Some(p) = prediction {
            known += 1;
            let (_, wall, _) = physical::validate_parts(
                &runtime.fingerprint,
                runtime.opening.monotonic_ns,
                Some(&contract),
                runtime.opening.monotonic_ns,
                wave.ticket,
                wave.fifo,
                wave.issued_at_ns,
                &wave.host_stages,
                wave.independent.as_ref(),
                &mut physical::Frontiers::default(),
            )
            .unwrap();
            misses += usize::from(p.planning_ns < wall);
            if spec
                .pairs
                .iter()
                .any(|pair| pair.call_id == wave.host_stages.call_id)
            {
                paired_known.push(wave.host_stages.call_id);
            }
        }
    }
    eprintln!(
        "PRODUCTION_JOINT_BANK_SOURCE8 {}",
        json!({"scope":"same original source8 cohort plan/cuts, typed numerical candidate and independent replay; no live adoption",
        "original_qualified_children":original_children,"candidate_qualified_children":models.len(),
        "checkpoint_import_verified":checkpoint_verified,"phase_counts":phase_counts,
        "later_runtime_b1_offers":offered,"later_runtime_known":known,"later_runtime_misses":misses,
        "paired_call_known":paired_known,"prepared_audit":candidate.prepared_audit()})
    );
    // The result is valid evidence even if fixed cohorts cannot populate a cell.
    // A zero-child candidate is reported explicitly and cannot be published.
    assert_eq!(models.is_empty(), !checkpoint_verified);
}
