//! Source protocol tests use real canonical CPU fixtures and synthetic wall
//! times. Installed EOS/Stop branch challenges are tested by model lifecycle
//! tests; this fixture covers the legacy plain-text, no-terminal work domain.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    StructuredCostTemplatePolicyV1, StructuredPopulationPolicyV1,
};
use ferrum_interfaces::execution_cost::{
    CostWorkloadDomainV1, CostWorkloadLimitsV1, ExecutorCostIdentity, EXECUTOR_COST_IDENTITY_SCHEMA,
};
use std::num::{NonZeroU32, NonZeroU64};

pub(in super::super) mod owner_blocks;
mod retained_memory;

fn domain() -> CostWorkloadDomainV1 {
    let f = old::fingerprint();
    CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: f.model_weights,
            numerical_policy: f.numerical_policy,
            device_runtime: f.device_runtime,
            execution_config: f.execution_config,
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(1).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(64).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(16).unwrap(),
            repetition_slot_capacity: 3,
            fixed_state_bytes_per_row: 0,
        },
    )
    .unwrap()
}
fn physical_header() -> StructuredServiceHeaderV6 {
    let mut h = header();
    h.declaration.domain_policy = StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1;
    h.declaration.nonnegative_envelope = Some(NonNegativeEnvelopeContractV1 {
        algorithm_universe: None,
        planning_estimator: Default::default(),
        population_policy: Default::default(),
        template_policy: Default::default(),
        workload_domain: domain(),
        settings: EnvelopeSettings::default(),
        challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
    });
    StructuredServiceHeaderV6::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap()
}
fn continuous_wave(ticket: u64, phase: StructuredPhaseV2) -> StructuredServiceWaveV6 {
    let (p, _, _) = old::prepared(&format!("physical-{ticket}"), 2, 1);
    StructuredServiceWaveV6 {
        ticket,
        phase,
        issued_at_ns: ticket * 2_000,
        fifo: ticket * 3,
        prepared_route: None,
        host_stages: old::stages(&old::header(), &p, ticket, 1_000),
        independent: None,
    }
}
fn physical_source() -> (Vec<u8>, StructuredInputV2, StructuredServiceCollectorV6) {
    physical_source_with_policy(StructuredCostTemplatePolicyV1::OrderedV1)
}
fn physical_source_with_policy(
    policy: StructuredCostTemplatePolicyV1,
) -> (Vec<u8>, StructuredInputV2, StructuredServiceCollectorV6) {
    let mut h = physical_header();
    if !policy.is_ordered() {
        let (p, offered, _) = old::prepared("numerical-family-declaration", 2, 1);
        let input = prepared::project_service_actual_with_domain(&p, &offered, &domain())
            .unwrap()
            .with_cost_template_policy(policy)
            .unwrap();
        h.declaration.scopes[0].owner = input.owner().clone();
        h.declaration
            .nonnegative_envelope
            .as_mut()
            .unwrap()
            .template_policy = policy;
        h = StructuredServiceHeaderV6::new(
            h.capture_identity,
            h.generation,
            h.fingerprint,
            h.producer,
            h.opening,
            h.declaration,
            h.maximum_file_bytes,
        )
        .unwrap();
    }
    let mut bytes = Vec::new();
    append(&mut bytes, &h);
    let mut collector =
        StructuredServiceCollectorV6::new(h, CostProfileLoadLimits::default()).unwrap();
    for phase_index in 0..3 {
        let phase = phase_at(phase_index);
        let first = phase_index as u64 * 8 + 1;
        let open = StructuredServiceRecordV6::PhaseOpen {
            phase,
            opened_at_ns: first * 2_000 - 1,
            fifo_cutoff: (first - 1) * 3,
        };
        collector.push(&open).unwrap();
        append(&mut bytes, &open);
        for ticket in first..first + 8 {
            let completed = StructuredServiceRecordV6::Completed {
                wave: continuous_wave(ticket, phase),
            };
            collector.push(&completed).unwrap();
            append(&mut bytes, &completed);
        }
        let frozen = collector.freeze((first + 7) * 2_000 + 1_101).unwrap();
        let StructuredServiceRecordV6::PhaseFreeze { children, .. } = &frozen else {
            unreachable!()
        };
        assert_eq!(children[0].members, 8);
        assert_eq!(children[0].failure, None);
        assert_eq!(
            children[0].nonnegative_fit_certificate.is_some(),
            phase == StructuredPhaseV2::Fit
        );
        append(&mut bytes, &frozen);
    }
    assert_eq!(collector.qualified_children(), 1);
    let footer = StructuredServiceRecordV6::Footer {
        offered: 24,
        accepted_fifo_cutoff: 72,
        closing: StructuredServiceClockV6 {
            monotonic_ns: 49_111,
            wall_unix_ns: 1_049_110,
        },
        failure: None,
    };
    collector.push(&footer).unwrap();
    append(&mut bytes, &footer);
    let (p, offered, _) = old::prepared("query", 2, 1);
    let input = prepared::project_service_actual_with_domain(&p, &offered, &domain())
        .unwrap()
        .with_settled_terminal_causes(&[])
        .unwrap();
    (bytes, input, collector)
}

#[test]
fn nonnegative_source6_certificate_replay_matches_memory_and_profile13() {
    check_physical_roundtrip(StructuredCostTemplatePolicyV1::OrderedV1);
}
#[test]
fn numerical_family_source6_certificate_replay_matches_memory_and_profile13() {
    check_physical_roundtrip(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1);
}
fn check_physical_roundtrip(policy: StructuredCostTemplatePolicyV1) {
    let (bytes, input, collector) = physical_source_with_policy(policy);
    let limits = CostProfileLoadLimits::default();
    let expected_certificate = collector
        .models()
        .next()
        .unwrap()
        .1
        .nonnegative_fit_certificate()
        .unwrap()
        .clone();
    let replayed = replay::replay_source(&bytes, &limits).unwrap();
    assert_eq!(
        replayed
            .models()
            .next()
            .unwrap()
            .1
            .nonnegative_fit_certificate(),
        Some(&expected_certificate)
    );
    let memory = collector
        .activate_same_process_memory(
            StructuredServiceClockV6 {
                monotonic_ns: 50_000,
                wall_unix_ns: 1,
            },
            &limits,
        )
        .unwrap();
    let files = Files::new(&bytes);
    export_structured_profile_v13(
        &files.source,
        Sha256::digest(&bytes).into(),
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    let imported = load_structured_profile_v13(
        &files.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(1_049_999),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 50_000,
        },
    )
    .unwrap();
    let query = StructuredQueryV2::exact(input);
    for catalog in [&memory, &imported] {
        assert_eq!(catalog.workload_domain(), Some(&domain()));
        assert_eq!(catalog.children[0].owner().cost_template_policy(), policy);
        assert_eq!(
            (catalog.offered_attempts, catalog.total_shape_rows),
            (24, 24)
        );
        assert_eq!(
            catalog.source_sha256,
            <[u8; 32]>::from(Sha256::digest(&bytes))
        );
        assert_eq!(
            catalog.children[0]
                .provenance()
                .phases
                .each_ref()
                .map(|p| p.members),
            [8; 3]
        );
    }
    let a = memory.children[0]
        .predict_query_local(&old::fingerprint(), &query, 50_000)
        .unwrap();
    let b = imported.children[0]
        .predict_query_local(&old::fingerprint(), &query, 50_000)
        .unwrap();
    assert_eq!(
        memory.children[0].parameters_signature(),
        imported.children[0].parameters_signature()
    );
    assert_eq!(a.planning_ns, b.planning_ns);
    assert_eq!(a.valid_until_ns, b.valid_until_ns);
}

#[test]
fn nonnegative_source6_rejects_certificate_tamper_and_wrong_phase() {
    let (bytes, _, _) = physical_source();
    for mutation in 0..6 {
        let mut changed = false;
        let mut out = Vec::new();
        for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
            if index == 0 {
                out.extend_from_slice(line);
                continue;
            }
            let mut record: StructuredServiceRecordV6 = serde_json::from_slice(line).unwrap();
            if let StructuredServiceRecordV6::PhaseFreeze {
                phase: StructuredPhaseV2::Fit,
                children,
                ..
            } = &mut record
            {
                let child = &mut children[0];
                match mutation {
                    0 => {
                        child
                            .nonnegative_fit_certificate
                            .as_mut()
                            .unwrap()
                            .input_digest[0] ^= 1
                    }
                    1 => {
                        child
                            .nonnegative_fit_certificate
                            .as_mut()
                            .unwrap()
                            .coefficient_words[0][0] ^= 1
                    }
                    2 => {
                        child
                            .nonnegative_fit_certificate
                            .as_mut()
                            .unwrap()
                            .epsilon_ns += 1
                    }
                    3 => child.nonnegative_fit_certificate = None,
                    4 => {
                        child.failure = Some("InvalidInput".into());
                        child.parameters_sha256 = None;
                        child.nonnegative_fit_certificate = None;
                    }
                    5 => {
                        child
                            .nonnegative_fit_certificate
                            .as_mut()
                            .unwrap()
                            .column_maxima[0] += 1
                    }
                    _ => unreachable!(),
                }
                changed = true;
            }
            append(&mut out, &record);
        }
        assert!(changed);
        assert!(
            replay::replay_source(&out, &CostProfileLoadLimits::default()).is_err(),
            "mutation {mutation}"
        );
    }
    // A valid certificate is still invalid in a heldout freeze, including when
    // it was copied verbatim from this stream's original Fit phase.
    let mut certificate = None;
    let mut out = Vec::new();
    for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
        if index == 0 {
            out.extend_from_slice(line);
            continue;
        }
        let mut record: StructuredServiceRecordV6 = serde_json::from_slice(line).unwrap();
        if let StructuredServiceRecordV6::PhaseFreeze {
            phase, children, ..
        } = &mut record
        {
            if *phase == StructuredPhaseV2::Fit {
                certificate = children[0].nonnegative_fit_certificate.clone();
            }
            if *phase == StructuredPhaseV2::Residual {
                children[0].nonnegative_fit_certificate = certificate.clone();
            }
        }
        append(&mut out, &record);
    }
    assert!(replay::replay_source(&out, &CostProfileLoadLimits::default()).is_err());
}

#[test]
fn nonnegative_source6_requires_policy_payload_identity_and_real_terminal_causes() {
    let good = physical_header();
    for mutation in 0..4 {
        let mut h = good.clone();
        match mutation {
            0 => h.declaration.nonnegative_envelope = None,
            1 => h.declaration.domain_policy = StructuredServiceDomainPolicyV1::AllOffered,
            2 => h.declaration.domain_policy = StructuredServiceDomainPolicyV1::FrozenFitSupportV1,
            3 => h.fingerprint.device_runtime[0] ^= 1,
            _ => unreachable!(),
        }
        assert!(StructuredServiceHeaderV6::new(
            h.capture_identity,
            h.generation,
            h.fingerprint,
            h.producer,
            h.opening,
            h.declaration,
            h.maximum_file_bytes
        )
        .is_err());
    }
    let mut collector =
        StructuredServiceCollectorV6::new(good, CostProfileLoadLimits::default()).unwrap();
    collector
        .push(&StructuredServiceRecordV6::PhaseOpen {
            phase: StructuredPhaseV2::Fit,
            opened_at_ns: 1_999,
            fifo_cutoff: 0,
        })
        .unwrap();
    // The legacy PlainTextGreedyV1 recipe has no installed EOS/Stop permission.
    // A structurally complete terminal DTO cannot grant it by changing causes.
    assert!(collector
        .push(&StructuredServiceRecordV6::Completed {
            wave: wave(1, StructuredPhaseV2::Fit)
        })
        .is_err());
}

#[test]
fn source6_legacy_wire_omits_new_contract_and_certificate_bytes() {
    #[derive(Serialize)]
    struct LegacyDeclaration<'a> {
        phase_offered_waves: [usize; 3],
        maximum_window_ns: u64,
        settings: &'a StructuredSettingsV2,
        scopes: &'a [StructuredScopeV2],
        maximum_retained_numeric_bytes: usize,
    }
    let h = header();
    let d = &h.declaration;
    let legacy = LegacyDeclaration {
        phase_offered_waves: d.phase_offered_waves,
        maximum_window_ns: d.maximum_window_ns,
        settings: &d.settings,
        scopes: &d.scopes,
        maximum_retained_numeric_bytes: d.maximum_retained_numeric_bytes,
    };
    let old_bytes = serde_json::to_vec(&legacy).unwrap();
    assert_eq!(serde_json::to_vec(d).unwrap(), old_bytes);
    assert_eq!(
        h.declaration_sha256,
        <[u8; 32]>::from(Sha256::digest(&old_bytes))
    );
    let (bytes, _, _) = source();
    let mut output = Vec::new();
    for (i, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
        let value: serde_json::Value = serde_json::from_slice(line).unwrap();
        if i == 0 {
            assert!(value["declaration"].get("nonnegative_envelope").is_none());
            append(
                &mut output,
                &serde_json::from_slice::<StructuredServiceHeaderV6>(line).unwrap(),
            );
        } else {
            if let Some(children) = value.get("children").and_then(serde_json::Value::as_array) {
                for child in children {
                    assert!(child.get("nonnegative_fit_certificate").is_none());
                }
            }
            append(
                &mut output,
                &serde_json::from_slice::<StructuredServiceRecordV6>(line).unwrap(),
            );
        }
    }
    assert_eq!(output, bytes);
    assert_eq!(
        replay::replay_source(&bytes, &CostProfileLoadLimits::default())
            .unwrap()
            .qualified_children(),
        1
    );
    let mut unknown = serde_json::to_value(h.declaration).unwrap();
    unknown["domain_policy"] = serde_json::json!("unknown_numerical_policy_v99");
    assert!(serde_json::from_value::<StructuredServiceDeclarationV6>(unknown).is_err());
}

#[test]
fn nonnegative_source6_failed_fit_keeps_full_population_and_revalidates_failure() {
    let mut h = physical_header();
    let count = h.declaration.settings.min_phase_samples - 1;
    assert!(count > 0);
    h.declaration.phase_offered_waves = [count; 3];
    h = StructuredServiceHeaderV6::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap();
    let mut bytes = Vec::new();
    append(&mut bytes, &h);
    let mut collector =
        StructuredServiceCollectorV6::new(h, CostProfileLoadLimits::default()).unwrap();
    for i in 0..3 {
        let phase = phase_at(i);
        let first = (i * count) as u64 + 1;
        let open = StructuredServiceRecordV6::PhaseOpen {
            phase,
            opened_at_ns: first * 2_000 - 1,
            fifo_cutoff: (first - 1) * 3,
        };
        collector.push(&open).unwrap();
        append(&mut bytes, &open);
        for ticket in first..first + count as u64 {
            let record = StructuredServiceRecordV6::Completed {
                wave: continuous_wave(ticket, phase),
            };
            collector.push(&record).unwrap();
            append(&mut bytes, &record);
        }
        let frozen = collector
            .freeze((first + count as u64 - 1) * 2_000 + 1_101)
            .unwrap();
        let StructuredServiceRecordV6::PhaseFreeze { children, .. } = &frozen else {
            unreachable!()
        };
        assert_eq!(children[0].members, count);
        assert!(children[0].failure.is_some());
        assert!(children[0].parameters_sha256.is_none());
        assert!(children[0].nonnegative_fit_certificate.is_none());
        append(&mut bytes, &frozen);
    }
    let final_ticket = 3 * count as u64;
    let close = final_ticket * 2_000 + 1_111;
    let footer = StructuredServiceRecordV6::Footer {
        offered: final_ticket,
        accepted_fifo_cutoff: final_ticket * 3,
        closing: StructuredServiceClockV6 {
            monotonic_ns: close,
            wall_unix_ns: 999_999 + close,
        },
        failure: None,
    };
    collector.push(&footer).unwrap();
    append(&mut bytes, &footer);
    let replayed = replay::replay_source(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(replayed.offered(), final_ticket);
    assert_eq!(replayed.qualified_children(), 0);
    assert!(replayed
        .activate_same_process_memory(
            StructuredServiceClockV6 {
                monotonic_ns: close + 1,
                wall_unix_ns: 1
            },
            &CostProfileLoadLimits::default()
        )
        .is_err());
}

#[test]
fn source6_rejects_family_population_even_with_self_consistent_signatures() {
    let (prepared, offered, _) = old::prepared("source6-family-declaration", 2, 1);
    let family = prepared::project_service_actual_with_domain(&prepared, &offered, &domain())
        .unwrap()
        .numerical_family_key()
        .unwrap();
    let exact = physical_header();
    exact.validate().unwrap();
    for (declare_family, attach_key) in [(true, false), (false, true), (true, true)] {
        let mut h = exact.clone();
        if declare_family {
            h.declaration
                .nonnegative_envelope
                .as_mut()
                .unwrap()
                .population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
        }
        if attach_key {
            h.declaration.scopes[0].numerical_family = Some(family);
        }

        // The constructor recomputes both signatures before validating the
        // declared population. Rejecting this is not a stale-signature check.
        assert!(matches!(
            StructuredServiceHeaderV6::new(
                h.capture_identity,
                h.generation,
                h.fingerprint.clone(),
                h.producer.clone(),
                h.opening,
                h.declaration.clone(),
                h.maximum_file_bytes,
            ),
            Err(CostProfileError::Metadata(
                "source6 population policy requires exact owner"
            ))
        ));

        // Re-sign the wire DTO independently so replay exercises the same
        // population boundary on an internally consistent serialized header.
        h.declaration_sha256 = Sha256::digest(serde_json::to_vec(&h.declaration).unwrap()).into();
        let mut protocol = Sha256::new();
        protocol.update(SERVICE_SOURCE_PROTOCOL_V6.as_bytes());
        protocol.update(MODEL_REVISION_V2.as_bytes());
        protocol.update(h.declaration_sha256);
        protocol.update(serde_json::to_vec(&h.fingerprint).unwrap());
        protocol.update(h.maximum_file_bytes.to_le_bytes());
        h.protocol = protocol.finalize().into();
        let mut bytes = Vec::new();
        append(&mut bytes, &h);
        assert!(matches!(
            replay::replay_source(&bytes, &CostProfileLoadLimits::default()),
            Err(CostProfileError::Metadata(
                "source6 population policy requires exact owner"
            ))
        ));
    }
}

#[test]
fn numerical_family_source6_rejects_retagging_original_ordered_declaration() {
    let mut old = physical_header();
    old.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .template_policy = StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1;
    assert!(StructuredServiceHeaderV6::new(
        old.capture_identity,
        old.generation,
        old.fingerprint,
        old.producer,
        old.opening,
        old.declaration,
        old.maximum_file_bytes
    )
    .is_err());
    let (bytes, _, _) =
        physical_source_with_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1);
    let mut header: StructuredServiceHeaderV6 =
        serde_json::from_slice(bytes.split_inclusive(|b| *b == b'\n').next().unwrap()).unwrap();
    header
        .declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .template_policy = StructuredCostTemplatePolicyV1::OrderedV1;
    assert!(StructuredServiceHeaderV6::new(
        header.capture_identity,
        header.generation,
        header.fingerprint,
        header.producer,
        header.opening,
        header.declaration,
        header.maximum_file_bytes
    )
    .is_err());
}

#[test]
fn numerical_family_diagnostic_keeps_original_source_validation_and_only_projects_identity() {
    let (bytes, _, _) = physical_source();
    let before = bytes.clone();
    let report =
        super::super::owner_diagnostic::diagnose_bytes(&bytes, &CostProfileLoadLimits::default())
            .unwrap();
    assert!(report.complete_source_verified);
    assert!(!report.numerical_family_projection_truncated);
    assert_eq!(report.numerical_family_projections.len(), 1);
    let pair = &report.numerical_family_projections[0];
    assert_eq!(
        pair.original_owner.cost_template_policy(),
        StructuredCostTemplatePolicyV1::OrderedV1
    );
    assert_eq!(
        pair.projected_owner.cost_template_policy(),
        StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
    );
    assert_eq!(
        pair.original_owner.algorithm_domain,
        pair.projected_owner.algorithm_domain
    );
    assert_eq!(pair.verified_waves, 24);
    assert_eq!(bytes, before);
    // The original declaration/model stays Ordered even though a diagnostic
    // can report how the same validated input would be grouped by new policy.
    let replayed = replay::replay_source(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(
        replayed
            .models()
            .next()
            .unwrap()
            .1
            .owner()
            .cost_template_policy(),
        StructuredCostTemplatePolicyV1::OrderedV1
    );
}
