//! Original timestamps and rows under the production collector/catalog partition.
//! Counterfactual membership is declared before BlockOpen, not a live ticket claim.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacitySpec {
    source: PathBuf,
    sha256: [u8; 32],
    production: Production,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Production {
    settings: ferrum_types::SloAutomaticCalibrationSettingsV1,
    capacity: Capacity,
}
#[derive(Deserialize, serde::Serialize)]
struct Capacity {
    maximum_source_slots: usize,
    source_slot_bytes: usize,
    collector_bytes_per_source: usize,
    catalog_bytes_per_source: usize,
    source_metadata_bytes: usize,
}
struct Attempt {
    collector: StructuredServiceCollectorV7,
    original_start_block: u64,
    original_offset: u64,
    seed_bytes: usize,
    qualified_at_ns: Option<u64>,
    imported_children: usize,
    catalog_peak: usize,
    failure: Option<String>,
    capacity_isolated: bool,
    failure_boundary: Option<Value>,
}
impl Attempt {
    fn open(
        original: &StructuredServiceHeaderV7,
        capacity: &Capacity,
        seed: DeclaredAlgorithmUniverseV1,
        opening: StructuredServiceClockV7,
        start_block: u64,
        offset: u64,
        global_residual: bool,
    ) -> Self {
        let mut declaration = original.declaration.clone();
        let envelope = declaration.nonnegative_envelope.as_mut().unwrap();
        let old_seed = envelope
            .algorithm_universe
            .as_ref()
            .unwrap()
            .retained_payload_bytes()
            .unwrap();
        let seed_bytes = seed.retained_payload_bytes().unwrap();
        declaration.maximum_discovery_bytes = declaration
            .maximum_discovery_bytes
            .checked_add(old_seed)
            .unwrap()
            .checked_sub(seed_bytes)
            .unwrap();
        declaration.maximum_retained_numeric_bytes = capacity
            .collector_bytes_per_source
            .checked_sub(seed_bytes)
            .filter(|v| *v > 0)
            .unwrap();
        declaration.schedule.phase_support =
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2);
        envelope.algorithm_universe = Some(seed);
        if global_residual {
            envelope.planning_estimator =
                NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1;
            let readiness = declaration.schedule.input_readiness.as_ref().unwrap();
            declaration.schedule.input_readiness = Some(
                OwnerInputReadinessV1::new_fit_target_v4(
                    readiness.maximum_phase_blocks,
                    readiness.maximum_geometry_visits,
                )
                .unwrap(),
            );
        }
        let mut identity = Sha256::new();
        identity.update(b"ferrum.offline.original-capacity-attempt.v1\0");
        identity.update(original.capture_identity);
        identity.update(start_block.to_le_bytes());
        let header = StructuredServiceHeaderV7::new(
            identity.finalize().into(),
            original.generation,
            original.fingerprint.clone(),
            original.producer.clone(),
            opening,
            declaration,
            original.maximum_file_bytes,
        )
        .unwrap();
        let header = match &original.monotonic_domain {
            Some(domain) => header.with_monotonic_domain(domain.clone()).unwrap(),
            None => header,
        };
        Self {
            collector: StructuredServiceCollectorV7::new_streaming(
                header,
                CostProfileLoadLimits::default(),
                NonZeroU64::new(original.maximum_file_bytes).unwrap(),
            )
            .unwrap(),
            original_start_block: start_block,
            original_offset: offset,
            seed_bytes,
            qualified_at_ns: None,
            imported_children: 0,
            catalog_peak: 0,
            failure: None,
            capacity_isolated: false,
            failure_boundary: None,
        }
    }
    fn feed(&mut self, record: &StructuredServiceRecordV7, capacity: &Capacity) {
        if self.failure.is_some() {
            return;
        }
        let before_receipt = self.collector.source_receipt();
        let before_offered = self.collector.offered();
        let before_qualified = self.collector.qualified_children();
        let before_samples: Vec<_> = self
            .collector
            .owners
            .iter()
            .map(|o| (o.samples.len(), o.samples.capacity()))
            .collect();
        let result = match record {
            StructuredServiceRecordV7::BlockOpen {
                opened_at_ns,
                fifo_cutoff,
                ..
            } => self
                .collector
                .open_block(*opened_at_ns, *fifo_cutoff)
                .map(|_| ()),
            StructuredServiceRecordV7::Completed { wave } => {
                let mut wave = wave.clone();
                wave.ticket = wave.ticket.checked_sub(self.original_offset).unwrap();
                self.collector
                    .push(&StructuredServiceRecordV7::Completed { wave })
            }
            StructuredServiceRecordV7::OutsideDeclaredRoute { wave } => {
                let mut wave = wave.clone();
                wave.ticket = wave.ticket.checked_sub(self.original_offset).unwrap();
                self.collector
                    .push(&StructuredServiceRecordV7::OutsideDeclaredRoute { wave })
            }
            StructuredServiceRecordV7::NotSubmitted { attempt } => {
                let ticket = attempt.ticket.checked_sub(self.original_offset).unwrap();
                self.collector
                    .push(&StructuredServiceRecordV7::NotSubmitted {
                        attempt: attempt.clone().with_source_position(ticket),
                    })
            }
            StructuredServiceRecordV7::BlockClose { block, closing, .. } => {
                self.close(*block, *closing, capacity)
            }
            _ => Ok(()),
        };
        if let Err(error) = result {
            if matches!(
                &error,
                CostProfileError::Limit("source7 shared workspace capacity")
            ) {
                // This refusal happens before reserve_physical_sample grows the
                // population Vec or appends the original completed record.
                assert_eq!(
                    before_samples,
                    self.collector
                        .owners
                        .iter()
                        .map(|o| (o.samples.len(), o.samples.capacity()))
                        .collect::<Vec<_>>()
                );
                assert_eq!(before_offered, self.collector.offered());
                assert_eq!(before_receipt, self.collector.source_receipt());
                assert_eq!(before_qualified, self.collector.qualified_children());
                assert!(self.collector.audit().poisoned);
                let closing = self
                    .collector
                    .last_close
                    .unwrap_or(self.collector.header.opening);
                assert!(
                    self.collector.checkpoint(closing).is_err(),
                    "poisoned source cannot publish"
                );
                assert!(
                    self.collector.push(record).is_err(),
                    "poisoned source cannot resume training"
                );
                assert_eq!(before_receipt, self.collector.source_receipt());
                self.capacity_isolated = true;
                self.failure_boundary = Some(json!({"offered":before_offered,
                    "unchanged_sample_lengths_and_capacities":true,"unappended_record":true,
                    "qualified_before":before_qualified,"qualified_after":self.collector.qualified_children(),
                    "poisoned":true,"checkpoint_rejected":true,"further_ingest_rejected":true}));
            }
            self.failure = Some(format!("{error:?}"));
        }
    }
    fn close(
        &mut self,
        block: u64,
        closing: StructuredServiceClockV7,
        capacity: &Capacity,
    ) -> Result<(), CostProfileError> {
        let record = self.collector.close_block(closing)?;
        let StructuredServiceRecordV7::BlockClose { freezes, .. } = record else {
            unreachable!()
        };
        eprintln!(
            "ROLLING_CAPACITY_BLOCK {}",
            json!({
                "source_start_block":self.original_start_block,"original_block":block,
                "closing":closing,"audit":self.collector.audit(),
                "requested_reservation_peak_bytes":self.collector.reserved_peak_for_tests(),
            "accepted_reservation_peak_bytes":self.collector.accepted_reservation_peak_for_tests(),
                "freezes":freezes.iter().map(|f|json!({
                    "attempt":f.owner_attempt_id,"phase":f.close.phase,
                    "members":f.close.member_count,"failure":f.failure,
                })).collect::<Vec<_>>(),
            })
        );
        let qualified = self.collector.qualified_children();
        if qualified != self.imported_children {
            let (_, checkpoint) = self.collector.checkpoint(closing)?;
            let limits = CostProfileLoadLimits::default();
            let imported = match &self.collector.header.monotonic_domain {
                Some(domain) => checkpoint.activate_same_boot_memory_streaming(
                    closing,
                    domain,
                    &limits,
                    NonZeroU64::new(self.collector.header.maximum_file_bytes).unwrap(),
                ),
                None => checkpoint.activate_same_process_memory_streaming(
                    closing,
                    &limits,
                    NonZeroU64::new(self.collector.header.maximum_file_bytes).unwrap(),
                ),
            }?;
            let retained = imported
                .children
                .iter()
                .try_fold(0usize, |sum, child| {
                    sum.checked_add(child.retained_payload_bytes()?)
                })
                .expect("catalog payload size");
            assert!(
                retained <= capacity.catalog_bytes_per_source,
                "catalog {retained} exceeds {}",
                capacity.catalog_bytes_per_source
            );
            assert_eq!(imported.children.len(), qualified);
            self.catalog_peak = self.catalog_peak.max(retained);
            self.imported_children = qualified;
            self.qualified_at_ns.get_or_insert(closing.monotonic_ns);
        }
        Ok(())
    }
    fn summary(&self) -> Value {
        json!({
            "original_start_block":self.original_start_block,"original_offset":self.original_offset,
            "opening":self.collector.header.opening,
            "seed_bytes":self.seed_bytes,
            "declared_numeric_bytes":self.collector.header.declaration.maximum_retained_numeric_bytes,
            "reserved_peak_bytes":self.collector.reserved_peak_for_tests(),
            "collector_plus_seed_accepted_peak_bytes":self.collector.accepted_reservation_peak_for_tests()+self.seed_bytes,
            "catalog_peak_bytes":self.catalog_peak,"imported_children":self.imported_children,
            "qualified_at_ns":self.qualified_at_ns,"failure":self.failure,
            "capacity_refusal_isolated":self.capacity_isolated,"failure_boundary":self.failure_boundary,
            "final_audit":self.collector.audit(),
        })
    }
}

#[test]
#[ignore = "requires immutable original source and production capacity via FERRUM_ROLLING_ORIGINAL_SPEC"]
fn archived_source7_rolling_production_capacity_preserves_original_two_attempt_qualification() {
    run_capacity(false);
}

#[test]
#[ignore = "requires immutable original source and production capacity via FERRUM_ROLLING_ORIGINAL_SPEC"]
fn archived_source7_global_residual_rolling_production_capacity() {
    run_capacity(true);
}

fn run_capacity(global_residual: bool) {
    let spec: CapacitySpec = serde_json::from_slice(
        &std::fs::read(
            std::env::var_os("FERRUM_ROLLING_ORIGINAL_SPEC").expect("capacity source spec"),
        )
        .unwrap(),
    )
    .unwrap();
    let header = seeded::verify_source_file(&spec.source, spec.sha256);
    let settings = &spec.production.settings;
    settings.validate().unwrap();
    assert_eq!(
        settings.population_schedule,
        ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2
    );
    let capacity = &spec.production.capacity;
    let generations = settings.maximum_retained_generations.get();
    let bytes = settings.maximum_retained_numeric_bytes.get();
    let share = bytes
        .checked_mul(generations)
        .unwrap()
        .checked_div(generations.checked_add(3).unwrap())
        .unwrap()
        .min(bytes / 2);
    assert_eq!(capacity.maximum_source_slots, generations + 1);
    assert_eq!(capacity.source_slot_bytes, share);
    assert_eq!(
        capacity.collector_bytes_per_source
            + capacity.catalog_bytes_per_source
            + capacity.source_metadata_bytes,
        share
    );
    assert!(capacity.maximum_source_slots >= 2 && capacity.source_metadata_bytes > 0);
    assert_eq!(
        header.declaration.maximum_owners,
        settings.maximum_owners.get()
    );
    assert_eq!(
        header.declaration.maximum_window_ns,
        settings.maximum_window_ns.get()
    );
    assert_eq!(
        header.declaration.schedule.block_offered,
        settings.discovery_offered_waves.get()
    );
    let seed = header
        .declaration
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .algorithm_universe
        .as_ref()
        .unwrap()
        .clone();
    let seed_bytes = seed.retained_payload_bytes().unwrap();
    assert_eq!(
        header.declaration.maximum_retained_numeric_bytes + seed_bytes,
        share,
        "declared runtime settings must reproduce original source slot before rolling partition"
    );
    assert_eq!(
        header.declaration.maximum_discovery_bytes + seed_bytes,
        settings.maximum_discovery_bytes.get()
    );
    assert_eq!(
        header.maximum_file_bytes,
        settings.maximum_encoded_source_bytes.get()
    );
    let mut original = StructuredServiceCollectorV7::new_streaming(
        header.clone(),
        CostProfileLoadLimits::default(),
        NonZeroU64::new(header.maximum_file_bytes).unwrap(),
    )
    .unwrap();
    let mut attempts = vec![Attempt::open(
        &header,
        capacity,
        seed,
        header.opening,
        1,
        0,
        global_residual,
    )];
    let maximum_attempts = if global_residual {
        capacity.maximum_source_slots
    } else {
        2
    };
    let mut aggregate_live_peak = 0usize;
    let mut aggregate_reserved_peak = 0usize;
    let mut aggregate_accepted_peak = 0usize;
    let mut other_source_offers_after_capacity_refusal = 0u64;
    let mut other_source_members_after_capacity_refusal = 0usize;
    let mut pending: Option<(u64, StructuredServiceClockV7, DeclaredAlgorithmUniverseV1)> = None;
    let mut original_rejection = None;
    for line in BufReader::new(std::fs::File::open(&spec.source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        if let Err(error) = original.push(&record) {
            let StructuredServiceRecordV7::Completed { wave } = &record else {
                panic!("unexpected original rejection: {error:?}")
            };
            assert!(matches!(&error, CostProfileError::Metadata(reason)
                if *reason == "source7 observation clock differs"));
            let deadline = header.opening.monotonic_ns + header.declaration.maximum_window_ns;
            assert!(
                wave.issued_at_ns <= deadline
                    && wave.host_stages.finalized_at_ns.unwrap() > deadline
            );
            original_rejection = Some(json!({
                "ticket":wave.ticket,"fifo":wave.fifo,"deadline_ns":deadline,
                "finalized_at_ns":wave.host_stages.finalized_at_ns,"error":format!("{error:?}"),
            }));
            break; // No synthetic close or following-generation training splice.
        }
        if let StructuredServiceRecordV7::BlockOpen {
            block,
            opened_at_ns,
            ..
        } = &record
        {
            if let Some((parent_block, closing, seed)) = pending.take() {
                assert_eq!(*block, parent_block + 1);
                assert!(*opened_at_ns >= closing.monotonic_ns);
                attempts.push(Attempt::open(
                    &header,
                    capacity,
                    seed,
                    closing,
                    *block,
                    original.offered(),
                    global_residual,
                ));
            }
        }
        let already_isolated = attempts.iter().any(|a| a.capacity_isolated);
        for attempt in &mut attempts {
            let before = attempt.collector.offered();
            let before_members: usize = attempt
                .collector
                .owners
                .iter()
                .map(|o| o.samples.len())
                .sum();
            attempt.feed(&record, capacity);
            if already_isolated && !attempt.capacity_isolated {
                other_source_offers_after_capacity_refusal += attempt.collector.offered() - before;
                let after_members: usize = attempt
                    .collector
                    .owners
                    .iter()
                    .map(|o| o.samples.len())
                    .sum();
                other_source_members_after_capacity_refusal +=
                    after_members.saturating_sub(before_members);
            }
        }
        let source = attempts.last().unwrap();
        if attempts.len() < maximum_attempts && pending.is_none() && source.failure.is_none() {
            if let StructuredServiceRecordV7::BlockClose { block, closing, .. } = &record {
                // The first actual Discovery freeze, irrespective of later Fit
                // errors, is the only launch trigger. Never search launch blocks.
                if let Some(seed) = source.collector.frozen_algorithm_universe() {
                    pending = Some((*block, *closing, seed.clone()));
                }
            }
        }
        let live = attempts
            .iter()
            .map(|a| {
                a.collector.audit().retained_numeric_bytes
                    + a.seed_bytes
                    + a.catalog_peak
                    + capacity.source_metadata_bytes
            })
            .sum::<usize>();
        let reserved = attempts
            .iter()
            .map(|a| {
                a.collector.reserved_peak_for_tests()
                    + a.seed_bytes
                    + a.catalog_peak
                    + capacity.source_metadata_bytes
            })
            .sum::<usize>();
        let accepted = attempts
            .iter()
            .map(|a| {
                a.collector.accepted_reservation_peak_for_tests()
                    + a.seed_bytes
                    + a.catalog_peak
                    + capacity.source_metadata_bytes
            })
            .sum::<usize>();
        aggregate_accepted_peak = aggregate_accepted_peak.max(accepted);
        aggregate_live_peak = aggregate_live_peak.max(live);
        aggregate_reserved_peak = aggregate_reserved_peak.max(reserved);
        assert!(
            live <= attempts.len() * capacity.source_slot_bytes,
            "actual retained collector/catalog/metadata exceeded admitted slots"
        );
    }
    let report = json!({
        "counterfactual_only":true,"production":capacity,
        "strategy":if global_residual { "identified_fit_global_residual_v1" } else { "original_identified_envelope_v2" },
        "configured_source_slots":capacity.maximum_source_slots,"actually_admitted_sources":attempts.len(),
        "aggregate_live_accounted_peak_bytes":aggregate_live_peak,
        "sum_individual_requested_peaks_bytes":aggregate_reserved_peak,
        "sum_individual_accepted_peaks_bytes":aggregate_accepted_peak,
        "capacity_refusal_count":attempts.iter().filter(|a|a.capacity_isolated).count(),
        "qualified_source_count":attempts.iter().filter(|a|a.imported_children>0).count(),
        "incomplete_source_count":attempts.iter().filter(|a|a.failure.is_none()&&a.imported_children==0).count(),
        "other_source_offers_after_capacity_refusal":other_source_offers_after_capacity_refusal,
        "other_source_members_after_capacity_refusal":other_source_members_after_capacity_refusal,
        "admitted_slot_allowance_bytes":attempts.len()*capacity.source_slot_bytes,
        "scope":"original collector/catalog partitions and discovery-driven overlapping original-input attempts; no live producer enrollment/ACK or throughput claim",
        "attempts":attempts.iter().map(Attempt::summary).collect::<Vec<_>>(),
        "original_control_reserved_peak_bytes":original.reserved_peak_for_tests(),
        "original_rejection":original_rejection,
    });
    eprintln!("ROLLING_ORIGINAL_CAPACITY {report}");
    if global_residual {
        assert!(
            attempts.iter().any(|a| a.imported_children > 0),
            "candidate must qualify at least one original-input source: {report}"
        );
    }
    if global_residual {
        assert!(
            attempts.len() >= 2 && attempts.len() <= capacity.maximum_source_slots,
            "{report}"
        );
    } else {
        assert_eq!(attempts.len(), 2, "{report}");
    }
    if attempts.iter().any(|a| a.capacity_isolated) {
        assert!(
            other_source_offers_after_capacity_refusal > 0,
            "other sources must keep progressing: {report}"
        );
    }
    for attempt in &attempts {
        if global_residual && attempt.capacity_isolated {
            assert!(
                attempt.failure.is_some() && attempt.collector.audit().poisoned,
                "{report}"
            );
            assert!(attempt.failure_boundary.is_some(), "{report}");
        } else {
            assert!(attempt.failure.is_none(), "{report}");
        }
        assert!(
            attempt.collector.accepted_reservation_peak_for_tests() + attempt.seed_bytes
                <= capacity.collector_bytes_per_source,
            "{report}"
        );
        if !global_residual {
            assert!(attempt.imported_children > 0, "{report}");
        }
    }
}
