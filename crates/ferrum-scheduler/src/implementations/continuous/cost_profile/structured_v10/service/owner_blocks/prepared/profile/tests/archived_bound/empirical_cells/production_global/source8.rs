use super::*;

#[test]
#[ignore = "requires the original source8 checkpoint plus runtime source via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_original_source8_through_production_global_residual() {
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
        .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1;
    let mut digest = Sha256::new();
    digest.update(b"ferrum.offline.source8-global-residual.v1\0");
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
    let mut original_fit_certificates_verified = 0usize;
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
                let before_bytes = candidate.population.audit().retained_numeric_bytes;
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
                        if let OwnerState::Fitted(model) = &owner.state {
                            let R::Population(StructuredServiceRecordV7::BlockClose {
                                freezes: original_freezes,
                                ..
                            }) = &record
                            else {
                                unreachable!()
                            };
                            let original_freeze = original_freezes
                                .iter()
                                .find(|o| o.owner_attempt_id == freeze.owner_attempt_id)
                                .unwrap();
                            assert_eq!(
                                serde_json::to_value(model.nonnegative_fit_certificate()).unwrap(),
                                serde_json::to_value(
                                    original_freeze.nonnegative_fit_certificate.as_ref()
                                )
                                .unwrap()
                            );
                            original_fit_certificates_verified += 1;
                        }
                        assert!(owner.samples.is_empty(), "closed phase samples released");
                        phase_counts.push(json!({"block":block,"owner":freeze.owner_attempt_id,
                            "phase":freeze.close.phase,"members":freeze.close.member_count,"failure":freeze.failure,
                            "before_close_numeric_bytes":before_bytes,"after_close_numeric_bytes":candidate.population.audit().retained_numeric_bytes}));
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
    let mut rows = BTreeMap::<u32, RowStats>::new();
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
        offered += 1;
        let stats = rows.entry(query.input().owner().rows).or_default();
        stats.total += 1;
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
            stats.known += 1;
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
            stats.misses += usize::from(p.planning_ns < wall);
            stats.maximum_under_ns = stats
                .maximum_under_ns
                .max(wall.saturating_sub(p.planning_ns));
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
        "PRODUCTION_GLOBAL_RESIDUAL_SOURCE8 {}",
        json!({"scope":"same original source8 cohort plan/cuts, typed numerical candidate and independent replay; no live adoption",
        "original_qualified_children":original_children,"candidate_qualified_children":models.len(),
        "checkpoint_import_verified":checkpoint_verified,"phase_counts":phase_counts,
        "original_identified_fit_certificates_verified":original_fit_certificates_verified,
        "future_by_rows":rows.iter().map(|(r,c)|json!({"rows":r,"total":c.total,"known":c.known,"misses":c.misses,"max_under_ns":c.maximum_under_ns})).collect::<Vec<_>>(),
        "later_runtime_all_rows_offers":offered,"later_runtime_known":known,"later_runtime_misses":misses,
        "paired_call_known":paired_known,"prepared_audit":candidate.prepared_audit()})
    );
    assert!(
        checkpoint_verified && !models.is_empty(),
        "must actually qualify and import"
    );
    assert!(original_fit_certificates_verified > 0);
    assert_eq!(misses, 0);
}
