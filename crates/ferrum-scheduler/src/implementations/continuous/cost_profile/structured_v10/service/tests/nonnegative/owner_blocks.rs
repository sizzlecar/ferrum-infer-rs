//! Canonical CPU producer -> original source7 stream -> independently replayed
//! physical-envelope certificate. Engine tests separately authenticate tickets.
use super::*;
mod algorithm_universe;
mod diagnostics;
mod fifo_gap;
mod input_readiness;
mod joint_bank;
mod numerical_family;
mod prediction_validity;
mod prospective_completion_binding;
mod replay_timing;
mod subprocess;

#[test]
fn source7_streaming_budget_is_independent_of_file_cap_and_still_exhaustible() {
    let budget = std::num::NonZeroU64::new(1024 * 1024 * 1024).unwrap();
    let original = block_header();
    let with_budget = |maximum| {
        StructuredServiceHeaderV7::new(
            original.capture_identity,
            original.generation,
            original.fingerprint.clone(),
            original.producer.clone(),
            original.opening,
            original.declaration.clone(),
            maximum,
        )
        .unwrap()
    };
    let limits = CostProfileLoadLimits {
        max_file_bytes: std::num::NonZeroUsize::new(1024).unwrap(),
        ..Default::default()
    };
    let h = with_budget(budget.get());
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new_streaming(h, limits.clone(), budget).unwrap();
    let mut closing = paired(1);
    for block in 1..=4 {
        closing = complete_block(&mut c, &mut bytes, block);
    }
    let (record, checkpoint) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &record);
    assert_eq!(
        checkpoint.source_receipt(),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    assert!(bytes.len() > limits.max_file_bytes.get());
    assert!(replay_structured_source_v7(&bytes, &limits).is_err());
    let memory = checkpoint
        .activate_same_process_memory_streaming(paired(70_000), &limits, budget)
        .unwrap();
    let replay = replay_structured_source_v7(&bytes, &CostProfileLoadLimits::default())
        .unwrap()
        .activate_same_process_memory(paired(70_000), &CostProfileLoadLimits::default())
        .unwrap();
    assert_eq!(memory.source_sha256, replay.source_sha256);
    assert_eq!(
        memory.children[0].parameters_signature(),
        replay.children[0].parameters_signature()
    );
    assert_eq!(
        memory.children[0]
            .predict_query_local(&old::fingerprint(), &query(), 70_000)
            .unwrap()
            .planning_ns,
        replay.children[0]
            .predict_query_local(&old::fingerprint(), &query(), 70_000)
            .unwrap()
            .planning_ns,
    );

    // A smaller original declaration still fails at its real byte boundary.
    // No ticket is resampled or removed to get below the work limit.
    let small = std::num::NonZeroU64::new((bytes.len() / 2) as u64).unwrap();
    let mut bounded =
        StructuredServiceCollectorV7::new_streaming(with_budget(small.get()), limits, small)
            .unwrap();
    let result = (|| -> Result<(), CostProfileError> {
        for block in 1..=4 {
            let first = (block - 1) * 8 + 1;
            bounded.open_block(first * 2_000 - 1, (first - 1) * 3)?;
            for ticket in first..first + 8 {
                bounded.push(&block_wave(ticket))?;
            }
            bounded.close_block(paired((first + 7) * 2_000 + 1_101))?;
        }
        Ok(())
    })();
    assert!(matches!(
        result,
        Err(CostProfileError::Limit("source7 byte capacity"))
    ));
    assert!(bounded.source_receipt().0 <= small.get());
    assert_eq!(bounded.qualified_children(), 0);
    assert!(bounded.checkpoint(paired(70_000)).is_err());
}

// V7 has a feature-independent wire contract. Do not use the V6 parent's
// typed serializer here: it intentionally preserves that older protocol.
fn append(bytes: &mut Vec<u8>, record: &impl Serialize) {
    bytes.extend(record_bytes_v7(record).unwrap());
}

fn block_header() -> StructuredServiceHeaderV7 {
    let mut h = physical_header();
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .template_policy = StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1;
    StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        StructuredServiceDeclarationV7 {
            schedule: OwnerBlockScheduleV1::new(8, [8; 3], [8; 3]).unwrap(),
            route_population: h.declaration.route_population,
            domain_policy: h.declaration.domain_policy,
            nonnegative_envelope: h.declaration.nonnegative_envelope,
            settings: h.declaration.settings,
            maximum_window_ns: h.declaration.maximum_window_ns,
            maximum_owners: 8,
            maximum_discovery_bytes: 1024 * 1024,
            maximum_retained_numeric_bytes: h.declaration.maximum_retained_numeric_bytes,
        },
        h.maximum_file_bytes,
    )
    .unwrap()
}
fn block_wave(ticket: u64) -> StructuredServiceRecordV7 {
    let w = continuous_wave(ticket, StructuredPhaseV2::Fit);
    StructuredServiceRecordV7::Completed {
        wave: StructuredServiceWaveV7::from_diagnostic(
            w.ticket,
            w.issued_at_ns,
            w.fifo,
            serde_json::to_value(&w.host_stages).unwrap(),
            w.independent,
        )
        .unwrap(),
    }
}
fn paired(monotonic_ns: u64) -> StructuredServiceClockV7 {
    StructuredServiceClockV7 {
        monotonic_ns,
        wall_unix_ns: 1_000_000 + monotonic_ns - 1,
    }
}
fn complete_block(
    c: &mut StructuredServiceCollectorV7,
    bytes: &mut Vec<u8>,
    block: u64,
) -> StructuredServiceClockV7 {
    let first = (block - 1) * 8 + 1;
    append(
        bytes,
        &c.open_block(first * 2_000 - 1, (first - 1) * 3).unwrap(),
    );
    for ticket in first..first + 8 {
        let record = block_wave(ticket);
        c.push(&record).unwrap();
        append(bytes, &record);
    }
    let closing = paired((first + 7) * 2_000 + 1_101);
    let record = c.close_block(closing).unwrap();
    if let StructuredServiceRecordV7::BlockClose {
        freezes,
        discoveries,
        ..
    } = &record
    {
        if block == 1 {
            assert_eq!(discoveries.len(), 1);
            assert!(freezes.is_empty());
        } else {
            assert!(discoveries.is_empty());
            assert_eq!(freezes.len(), 1);
            let frozen = &freezes[0];
            assert_eq!(frozen.close.member_count, 8);
            assert_eq!(frozen.domain.owner_offered, 8);
            assert_eq!(frozen.domain.eligible, 8);
            assert_eq!(frozen.failure, None);
            assert_eq!(frozen.nonnegative_fit_certificate.is_some(), block == 2);
        }
    } else {
        panic!("expected a complete block record");
    }
    append(bytes, &record);
    closing
}
pub(in super::super::super) fn collected() -> (
    Vec<u8>,
    StructuredServiceCollectorV7,
    StructuredServiceCheckpointV7,
    StructuredServiceClockV7,
) {
    collected_with_header(block_header())
}

fn collected_with_header(
    h: StructuredServiceHeaderV7,
) -> (
    Vec<u8>,
    StructuredServiceCollectorV7,
    StructuredServiceCheckpointV7,
    StructuredServiceClockV7,
) {
    let mut bytes = Vec::new();
    append(&mut bytes, &h);
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    let mut closing = paired(1);
    for block in 1..=4 {
        closing = complete_block(&mut c, &mut bytes, block);
    }
    assert_eq!(c.qualified_children(), 1);
    let (record, checkpoint) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &record);
    assert_eq!(
        checkpoint.source_receipt(),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    (bytes, c, checkpoint, closing)
}
fn query() -> StructuredQueryV2 {
    let (p, offered, _) = old::prepared("owner-block-query", 2, 1);
    StructuredQueryV2::exact(
        prepared::project_service_actual_with_domain(&p, &offered, &domain())
            .unwrap()
            .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
            .unwrap()
            .with_settled_terminal_causes(&[])
            .unwrap(),
    )
}
#[test]
fn source7_profile14_replays_checkpoint_before_later_partial_failed_block() {
    let (mut bytes, mut collector, checkpoint, closing) = collected();
    let limits = CostProfileLoadLimits::default();
    let cutoff = bytes.len() as u64;
    let prefix_sha: [u8; 32] = Sha256::digest(&bytes).into();
    let memory = checkpoint
        .activate_same_process_memory(paired(70_000), &limits)
        .unwrap();
    let replayed = replay_structured_source_v7(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(paired(70_000), &limits)
        .unwrap();
    assert_eq!(
        memory.children[0].parameters_signature(),
        replayed.children[0].parameters_signature()
    );
    assert_eq!(memory.children[0].workload_domain(), Some(&domain()));
    // Later journal evidence is retained and hash-bound, but never added to the
    // already frozen three populations or used to renew their clocks.
    append(&mut bytes, &collector.open_block(66_000 - 1, 96).unwrap());
    let record = block_wave(33);
    collector.push(&record).unwrap();
    append(&mut bytes, &record);
    append(
        &mut bytes,
        &collector
            .fail(34, 102, 68_000, "original ticket lost")
            .unwrap(),
    );
    append(&mut bytes, &collector.stop(paired(68_001)).unwrap());
    assert!(replay_structured_source_v7(&bytes, &limits).is_err());
    let files = Files::new(&bytes);
    let receipt = export_structured_profile_v14(
        &files.source,
        Sha256::digest(&bytes).into(),
        cutoff,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    assert_eq!(receipt.schema_version, 14);
    let imported = load_structured_profile_v14(
        &files.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(paired(70_000).wall_unix_ns),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 70_000,
        },
    )
    .unwrap();
    assert_eq!(imported.source_bytes, cutoff);
    assert_eq!(imported.journal_bytes, bytes.len() as u64);
    assert_eq!(imported.source_sha256, prefix_sha);
    assert_eq!(
        (imported.offered_attempts, imported.total_shape_rows),
        (32, 32)
    );
    assert_eq!(
        imported.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [8; 3]
    );
    let expected = memory.children[0]
        .predict_query_local(&old::fingerprint(), &query(), 70_000)
        .unwrap();
    let actual = imported.children[0]
        .predict_query_local(&old::fingerprint(), &query(), 70_000)
        .unwrap();
    assert_eq!(actual.planning_ns, expected.planning_ns);
    assert_eq!(actual.valid_until_ns, expected.valid_until_ns);
    assert!(load_structured_profile_v13(
        &files.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(1_100_000),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 100
        }
    )
    .is_err());
}
#[test]
fn source7_original_ticket_and_first_ready_block_freeze_cannot_be_removed_or_changed() {
    let (bytes, _, _, _) = collected();
    let limits = CostProfileLoadLimits::default();
    for mutation in 0..4 {
        let mut modified = Vec::new();
        let mut changed = false;
        for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
            if index == 0 {
                modified.extend_from_slice(line);
                continue;
            }
            let mut r: StructuredServiceRecordV7 = serde_json::from_slice(line).unwrap();
            match &mut r {
                StructuredServiceRecordV7::Completed { wave }
                    if mutation == 0 && wave.ticket == 8 =>
                {
                    changed = true;
                    continue;
                }
                StructuredServiceRecordV7::BlockClose {
                    block: 2, freezes, ..
                } => match mutation {
                    1 => {
                        freezes.clear();
                        changed = true;
                    }
                    2 => {
                        freezes[0].parameters_sha256 = Some([99; 32]);
                        changed = true;
                    }
                    3 => {
                        freezes[0].nonnegative_fit_certificate = None;
                        changed = true;
                    }
                    _ => {}
                },
                _ => {}
            }
            append(&mut modified, &r);
        }
        assert!(changed);
        assert!(replay_structured_source_v7(&modified, &limits).is_err());
    }
}
#[test]
fn source7_partial_fit_checkpoint_and_original_epoch_expiry_cannot_publish() {
    let h = block_header();
    let deadline = h.opening.monotonic_ns + h.declaration.maximum_window_ns;
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    let mut bytes = Vec::new();
    complete_block(&mut c, &mut bytes, 1);
    c.open_block(18_000 - 1, 24).unwrap();
    for ticket in 9..16 {
        c.push(&block_wave(ticket)).unwrap();
    }
    assert!(c.close_block(paired(33_101)).is_err());
    assert_eq!(c.qualified_children(), 0);
    assert!(c.checkpoint(paired(33_101)).is_err());
    let mut c = StructuredServiceCollectorV7::new(block_header(), CostProfileLoadLimits::default())
        .unwrap();
    assert!(c.open_block(deadline + 1, 0).is_err());
}

mod same_boot;
