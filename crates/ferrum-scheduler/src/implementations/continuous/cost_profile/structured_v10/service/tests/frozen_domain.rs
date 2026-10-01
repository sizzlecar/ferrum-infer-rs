//! Input-domain selection still replays every original offered service wave.
use super::*;

#[test]
fn first_outside_fit_diagnostics_leave_original_source_replay_unchanged() {
    let (bytes, _, _) = domain_source(StructuredServiceDomainPolicyV1::FrozenFitSupportV1, false);
    let mut lines = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty());
    let header = serde_json::from_slice(lines.next().unwrap()).unwrap();
    let mut collector =
        StructuredServiceCollectorV6::new(header, CostProfileLoadLimits::default()).unwrap();
    for line in lines {
        let record: StructuredServiceRecordV6 = serde_json::from_slice(line).unwrap();
        if let StructuredServiceRecordV6::PhaseFreeze {
            phase, children, ..
        } = &record
        {
            let diagnostics = collector.first_outside_fit_diagnostics();
            // Calling twice borrows the same original population and does not
            // consume/relabel it, or affect the expected source freeze bytes.
            assert_eq!(collector.first_outside_fit_diagnostics(), diagnostics);
            if *phase == StructuredPhaseV2::Fit {
                assert!(diagnostics.is_empty());
            } else {
                assert_eq!(children[0].domain.as_ref().unwrap().outside_fit_support, 8);
                assert_eq!(diagnostics.len(), 1);
                assert_eq!(diagnostics[0].0, 0);
                assert!(matches!(
                    diagnostics[0].1.reason,
                    StructuredFitSupportReasonV1::AboveAllFitMax { .. }
                ));
            }
        }
        // The unchanged original freeze/digest/population is still mandatory.
        collector.push(&record).unwrap();
    }
    assert_eq!(collector.qualified_children(), 1);
    assert!(collector.first_outside_fit_diagnostics().is_empty());
}

#[test]
fn failed_owner_samples_do_not_masquerade_as_fit_support_diagnostics() {
    let header = domain_header(StructuredServiceDomainPolicyV1::FrozenFitSupportV1);
    let mut declaration = header.declaration;
    declaration.settings.min_phase_samples = 17;
    let header = StructuredServiceHeaderV6::new(
        header.capture_identity,
        header.generation,
        header.fingerprint,
        header.producer,
        header.opening,
        declaration,
        header.maximum_file_bytes,
    )
    .unwrap();
    let mut collector =
        StructuredServiceCollectorV6::new(header, CostProfileLoadLimits::default()).unwrap();
    for index in 0..2 {
        let phase = phase_at(index);
        let first = index as u64 * 16 + 1;
        collector
            .push(&StructuredServiceRecordV6::PhaseOpen {
                phase,
                opened_at_ns: first * 2_000 - 1,
                fifo_cutoff: (first - 1) * 3,
            })
            .unwrap();
        for ticket in first..first + 16 {
            collector
                .push(&StructuredServiceRecordV6::Completed {
                    wave: domain_wave(ticket, phase, if index == 0 { 1 } else { 2 }, 1_000),
                })
                .unwrap();
        }
        assert!(collector.first_outside_fit_diagnostics().is_empty());
        let StructuredServiceRecordV6::PhaseFreeze { children, .. } =
            collector.freeze((first + 15) * 2_000 + 1_601).unwrap()
        else {
            unreachable!()
        };
        assert!(children[0].failure.is_some());
        if index == 1 {
            let domain = children[0].domain.as_ref().unwrap();
            assert_eq!(domain.unclassified_failed_owner, 16);
            assert_eq!(domain.outside_fit_support, 0);
        }
    }
}

fn domain_header(policy: StructuredServiceDomainPolicyV1) -> StructuredServiceHeaderV6 {
    let h = header();
    let mut declaration = h.declaration;
    declaration.domain_policy = policy;
    declaration.phase_offered_waves = [16; 3];
    StructuredServiceHeaderV6::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        declaration,
        h.maximum_file_bytes,
    )
    .unwrap()
}

fn domain_wave(
    ticket: u64,
    phase: StructuredPhaseV2,
    generated: u64,
    wall: u64,
) -> StructuredServiceWaveV6 {
    let (prepared, _, _) = old::prepared(&format!("domain-request-{ticket}"), 2, generated);
    StructuredServiceWaveV6 {
        ticket,
        phase,
        issued_at_ns: ticket * 2_000,
        fifo: ticket * 3,
        prepared_route: None,
        host_stages: old::stages(&old::header(), &prepared, ticket, wall),
        independent: None,
    }
}

fn domain_source(
    policy: StructuredServiceDomainPolicyV1,
    slow_eligible_qualification: bool,
) -> (
    Vec<u8>,
    StructuredServiceCollectorV6,
    Vec<StructuredServiceChildFreezeV6>,
) {
    let header = domain_header(policy);
    let mut bytes = Vec::new();
    append(&mut bytes, &header);
    let mut collector =
        StructuredServiceCollectorV6::new(header, CostProfileLoadLimits::default()).unwrap();
    let mut freezes = Vec::new();
    for index in 0..3 {
        let phase = phase_at(index);
        let first = index as u64 * 16 + 1;
        let open = StructuredServiceRecordV6::PhaseOpen {
            phase,
            opened_at_ns: first * 2_000 - 1,
            fifo_cutoff: (first - 1) * 3,
        };
        collector.push(&open).unwrap();
        append(&mut bytes, &open);
        for ticket in first..first + 16 {
            // Same actual owner. Later generated/Length inputs have no Fit
            // support; the early input recurs across independent requests.
            let generated = if index > 0 && ticket % 2 == 0 { 2 } else { 1 };
            let wall = if index == 2 && generated == 1 && slow_eligible_qualification {
                1_500
            } else {
                1_000
            };
            let complete = StructuredServiceRecordV6::Completed {
                wave: domain_wave(ticket, phase, generated, wall),
            };
            collector.push(&complete).unwrap();
            append(&mut bytes, &complete);
        }
        let freeze = collector.freeze((first + 15) * 2_000 + 1_601).unwrap();
        let StructuredServiceRecordV6::PhaseFreeze { children, .. } = &freeze else {
            unreachable!()
        };
        freezes.push(children[0].clone());
        append(&mut bytes, &freeze);
    }
    let close_mono = 97_701;
    let footer = StructuredServiceRecordV6::Footer {
        offered: 48,
        accepted_fifo_cutoff: 144,
        closing: StructuredServiceClockV6 {
            monotonic_ns: close_mono,
            wall_unix_ns: 1_000_000 + close_mono - 1,
        },
        failure: None,
    };
    collector.push(&footer).unwrap();
    append(&mut bytes, &footer);
    (bytes, collector, freezes)
}

#[test]
fn frozen_domain_keeps_all_raw_offered_waves_and_replays_eligible_counts() {
    let (bytes, collector, freezes) =
        domain_source(StructuredServiceDomainPolicyV1::FrozenFitSupportV1, false);
    assert_eq!(collector.qualified_children(), 1);
    assert_eq!(
        (
            collector.offered(),
            collector.total_rows,
            collector.last_fifo()
        ),
        (48, 48, 144)
    );
    let mut tickets = Vec::new();
    for line in bytes
        .split(|b| *b == b'\n')
        .skip(1)
        .filter(|line| !line.is_empty())
    {
        if let StructuredServiceRecordV6::Completed { wave } = serde_json::from_slice(line).unwrap()
        {
            tickets.push(wave.ticket);
        }
    }
    assert_eq!(tickets, (1..=48).collect::<Vec<_>>());
    for (index, freeze) in freezes.iter().enumerate() {
        let audit = freeze.domain.as_ref().unwrap();
        assert_eq!(audit.owner_offered, 16);
        assert_eq!(audit.eligible, if index == 0 { 16 } else { 8 });
        assert_eq!(freeze.members, audit.eligible);
        assert_eq!(audit.outside_fit_support, if index == 0 { 0 } else { 8 });
        assert_eq!(
            (
                audit.outside_residual_support,
                audit.unclassified_failed_owner
            ),
            (0, 0)
        );
        assert_eq!(
            audit.frozen_domain_parameters_sha256,
            index
                .checked_sub(1)
                .map(|previous| freezes[previous].parameters_sha256.unwrap())
        );
        assert!(freeze.failure.is_none());
    }
    let replayed = replay::replay_source(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(replayed.qualified_children(), 1);
    assert_eq!(
        replayed.phases[0]
            .iter()
            .map(|p| p.members)
            .collect::<Vec<_>>(),
        vec![16, 8, 8]
    );
    assert_eq!(
        replayed.phases[0]
            .iter()
            .map(|p| p.member_cutoff)
            .collect::<Vec<_>>(),
        vec![16, 24, 32]
    );

    // A fresh profile importer must derive the same eligible child from all
    // source bytes. This also exercises policy/parameters binding across IO.
    let files = Files::new(&bytes);
    let exported = export_structured_profile_v13(
        &files.source,
        Sha256::digest(&bytes).into(),
        &files.profile,
        0,
        &CostProfileLoadLimits::default(),
    )
    .unwrap();
    assert_eq!(exported.children.len(), 1);
    let imported = load_structured_profile_v13(
        &files.profile,
        &old::fingerprint(),
        &CostProfileLoadLimits::default(),
        ProfileLoadClock {
            wall_unix_ns: Some(1_097_800),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 500,
        },
    )
    .unwrap();
    assert_eq!(
        (imported.offered_attempts, imported.total_shape_rows),
        (48, 48)
    );
    let (_, _, supported) = old::prepared("supported-query", 2, 1);
    assert!(imported.children[0]
        .predict_query_local(
            &old::fingerprint(),
            &StructuredQueryV2::exact(supported),
            500
        )
        .is_ok());
    let (_, _, outside) = old::prepared("outside-query", 2, 2);
    assert!(imported.children[0]
        .predict_query_local(&old::fingerprint(), &StructuredQueryV2::exact(outside), 500)
        .is_err());
}

#[test]
fn legacy_all_offered_semantics_and_wire_are_preserved() {
    let (bytes, collector, freezes) =
        domain_source(StructuredServiceDomainPolicyV1::AllOffered, false);
    assert_eq!(collector.qualified_children(), 0);
    assert_eq!(freezes[1].members, 16);
    assert!(freezes[1].failure.is_some());
    assert!(freezes.iter().all(|f| f.domain.is_none()));
    let header_wire: serde_json::Value =
        serde_json::from_slice(bytes.split(|b| *b == b'\n').next().unwrap()).unwrap();
    assert!(header_wire["declaration"].get("domain_policy").is_none());
    assert!(serde_json::to_value(&freezes[0])
        .unwrap()
        .get("domain")
        .is_none());
    assert_eq!(
        replay::replay_source(&bytes, &CostProfileLoadLimits::default())
            .unwrap()
            .qualified_children(),
        0
    );
}

#[test]
fn selected_heldout_low_bound_fails_owner_without_discarding_outside_evidence() {
    let (bytes, collector, freezes) =
        domain_source(StructuredServiceDomainPolicyV1::FrozenFitSupportV1, true);
    assert_eq!(collector.qualified_children(), 0);
    let qual = &freezes[2];
    assert_eq!(qual.failure.as_deref(), Some("QualificationUnderestimate"));
    let audit = qual.domain.as_ref().unwrap();
    assert_eq!(
        (
            audit.owner_offered,
            audit.eligible,
            audit.outside_fit_support
        ),
        (16, 8, 8)
    );
    assert_eq!(collector.offered(), 48);
    assert_eq!(
        replay::replay_source(&bytes, &CostProfileLoadLimits::default())
            .unwrap()
            .qualified_children(),
        0
    );
}

#[test]
fn frozen_domain_selection_counters_policy_and_original_settlement_are_bound() {
    let (bytes, _, _) = domain_source(StructuredServiceDomainPolicyV1::FrozenFitSupportV1, false);
    for mutation in 0..4 {
        let mut out = Vec::new();
        let mut changed = false;
        for (index, line) in bytes
            .split(|b| *b == b'\n')
            .filter(|l| !l.is_empty())
            .enumerate()
        {
            if index == 0 {
                let mut header: StructuredServiceHeaderV6 = serde_json::from_slice(line).unwrap();
                if mutation == 0 {
                    header.declaration.domain_policy = StructuredServiceDomainPolicyV1::AllOffered;
                    changed = true;
                }
                append(&mut out, &header);
                continue;
            }
            let mut record: StructuredServiceRecordV6 = serde_json::from_slice(line).unwrap();
            match &mut record {
                StructuredServiceRecordV6::PhaseFreeze {
                    phase: StructuredPhaseV2::Residual,
                    children,
                    ..
                } if mutation == 1 => {
                    let audit = children[0].domain.as_mut().unwrap();
                    audit.eligible += 1;
                    audit.outside_fit_support -= 1;
                    changed = true;
                }
                StructuredServiceRecordV6::Completed { wave }
                    if wave.ticket == 18 && mutation == 2 =>
                {
                    // This is outside input support, but its physical evidence
                    // must still validate before numerical selection happens.
                    wave.host_stages
                        .structured_evidence
                        .as_mut()
                        .unwrap()
                        .as_mut()
                        .unwrap()
                        .stage_binding = [91; 32];
                    changed = true;
                }
                StructuredServiceRecordV6::PhaseFreeze {
                    phase: StructuredPhaseV2::Residual,
                    children,
                    ..
                } if mutation == 3 => {
                    children[0]
                        .domain
                        .as_mut()
                        .unwrap()
                        .frozen_domain_parameters_sha256 = Some([91; 32]);
                    changed = true;
                }
                _ => {}
            }
            append(&mut out, &record);
        }
        assert!(changed);
        assert!(replay::replay_source(&out, &CostProfileLoadLimits::default()).is_err());
    }
    let mut unknown = serde_json::to_value(domain_header(
        StructuredServiceDomainPolicyV1::FrozenFitSupportV1,
    ))
    .unwrap();
    unknown["declaration"]["domain_policy"] = serde_json::json!("unknown_policy");
    assert!(serde_json::from_value::<StructuredServiceHeaderV6>(unknown).is_err());
}
