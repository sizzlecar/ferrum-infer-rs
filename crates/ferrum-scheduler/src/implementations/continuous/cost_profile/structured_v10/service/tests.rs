//! Real canonical producer and strict source replay, synthetic CPU timings.
//! Engine private-ticket integration supplies separate live-authority tests.
use super::super::tests::fixture as old;
use super::*;
mod frozen_domain;
mod live;
pub(super) mod nonnegative;
mod owner_diagnostic;
mod route_population;

fn header() -> StructuredServiceHeaderV6 {
    let h = old::header();
    StructuredServiceHeaderV6::new(
        [47; 32],
        1,
        h.fingerprint,
        h.producer,
        StructuredServiceClockV6 {
            wall_unix_ns: 1_000_000,
            monotonic_ns: 1,
        },
        StructuredServiceDeclarationV6 {
            route_population: ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
            domain_policy: StructuredServiceDomainPolicyV1::AllOffered,
            nonnegative_envelope: None,
            phase_offered_waves: [8; 3],
            maximum_window_ns: 1_000_000_000,
            settings: h.settings.native(),
            scopes: vec![h.scope],
            maximum_retained_numeric_bytes: 32 * 1024 * 1024,
        },
        16 * 1024 * 1024,
    )
    .unwrap()
}
fn wave(ticket: u64, phase: StructuredPhaseV2) -> StructuredServiceWaveV6 {
    let h = old::header();
    let (p, _, _) = old::prepared(&format!("request-{ticket}"), 2, 1);
    let mut stages = old::stages(&h, &p, ticket, 1_000);
    let (last, _, _) = old::prepared("terminal-shape", 3, 2);
    let mut terminal = old::stages(&h, &last, ticket, 1_000)
        .rows
        .remove(0)
        .terminal
        .unwrap();
    terminal.finish_reason = if ticket % 2 == 0 {
        ferrum_types::FinishReason::EOS
    } else {
        ferrum_types::FinishReason::Stop
    };
    terminal.generated_tokens = 2;
    terminal.through_output_ordinal = 2;
    stages.rows[0].terminal = Some(terminal);
    stages.rows[0].completion_started_at_ns = Some(ticket * 2_000 + 900);
    let binding = observation::stage_binding(&stages, None).unwrap();
    stages
        .structured_evidence
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .stage_binding = binding;
    StructuredServiceWaveV6 {
        ticket,
        phase,
        issued_at_ns: ticket * 2_000,
        fifo: ticket * 3,
        prepared_route: None,
        host_stages: stages,
        independent: None,
    }
}
fn append(bytes: &mut Vec<u8>, record: &impl Serialize) {
    bytes.extend(replay::record_bytes(record).unwrap());
}
fn source() -> (Vec<u8>, StructuredInputV2, u64) {
    let (bytes, input, wall, _) = collected_source();
    (bytes, input, wall)
}
fn collected_source() -> (
    Vec<u8>,
    StructuredInputV2,
    u64,
    StructuredServiceCollectorV6,
) {
    let header = header();
    let mut bytes = Vec::new();
    append(&mut bytes, &header);
    let mut collector =
        StructuredServiceCollectorV6::new(header, CostProfileLoadLimits::default()).unwrap();
    for i in 0..3 {
        let phase = phase_at(i);
        let first = i as u64 * 8 + 1;
        let open = StructuredServiceRecordV6::PhaseOpen {
            phase,
            opened_at_ns: first * 2_000 - 1,
            fifo_cutoff: (first - 1) * 3,
        };
        collector.push(&open).unwrap();
        append(&mut bytes, &open);
        for ticket in first..first + 8 {
            let completed = StructuredServiceRecordV6::Completed {
                wave: wave(ticket, phase),
            };
            collector.push(&completed).unwrap();
            append(&mut bytes, &completed);
        }
        let freeze = collector.freeze((first + 7) * 2_000 + 1_101).unwrap();
        append(&mut bytes, &freeze);
    }
    assert_eq!(collector.qualified_children(), 1);
    let close_mono = 49_111;
    let footer = StructuredServiceRecordV6::Footer {
        offered: 24,
        accepted_fifo_cutoff: 72,
        closing: StructuredServiceClockV6 {
            monotonic_ns: close_mono,
            wall_unix_ns: 1_000_000 + close_mono - 1,
        },
        failure: None,
    };
    collector.push(&footer).unwrap();
    append(&mut bytes, &footer);
    let (_, _, input) = old::prepared("query", 2, 1);
    (bytes, input, 1_000_000 + close_mono - 1, collector)
}
struct Files {
    dir: PathBuf,
    source: PathBuf,
    profile: PathBuf,
}
impl Files {
    fn new(bytes: &[u8]) -> Self {
        let dir = std::env::temp_dir().join(format!("ferrum-source6-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("source6.jsonl");
        std::fs::write(&source, bytes).unwrap();
        let profile = dir.join("profile13.json");
        Self {
            dir,
            source,
            profile,
        }
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn source6_profile13_roundtrip_natural_terminals_and_unchanged_original_age() {
    let (bytes, input, close_wall) = source();
    let f = Files::new(&bytes);
    let limits = CostProfileLoadLimits::default();
    let exported = export_structured_profile_v13(
        &f.source,
        Sha256::digest(&bytes).into(),
        &f.profile,
        0,
        &limits,
    )
    .unwrap();
    assert_eq!(exported.children.len(), 1);
    let load = ProfileLoadClock {
        wall_unix_ns: Some(close_wall + 100),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 500,
    };
    let imported =
        load_structured_profile_v13(&f.profile, &old::fingerprint(), &limits, load).unwrap();
    assert_eq!(
        (imported.offered_attempts, imported.total_shape_rows),
        (24, 24)
    );
    let child = &imported.children[0];
    let query = StructuredQueryV2::exact(input);
    let first = child
        .predict_query_local(&old::fingerprint(), &query, 500)
        .unwrap();
    assert_eq!(first.valid_until_ns, 1_000_003_100);
    assert_eq!(child.provenance().schema_version, 13);
    assert_eq!(
        child.provenance().phases.each_ref().map(|p| p.members),
        [8; 3]
    );
    let later = load_structured_profile_v13(
        &f.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(close_wall + 1_100),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 50,
        },
    )
    .unwrap();
    assert!(
        later.children[0].provenance().oldest_imported_age_ns
            > child.provenance().oldest_imported_age_ns
    );
    assert!(later.children[0]
        .predict_query_local(&old::fingerprint(), &query, 1_000_000_050)
        .is_err());
    assert!(export_structured_profile_v10(
        &f.source,
        Sha256::digest(&bytes).into(),
        &f.dir.join("old10"),
        0,
        &limits
    )
    .is_err());
    assert!(export_structured_profile_v11(
        &f.source,
        Sha256::digest(&bytes).into(),
        &f.dir.join("old11"),
        &[0],
        &limits
    )
    .is_err());
    assert!(export_structured_profile_v12(
        &f.source,
        Sha256::digest(&bytes).into(),
        &f.dir.join("old12"),
        &[0],
        &limits
    )
    .is_err());
}
#[test]
fn source6_missing_last_ticket_duplicate_and_mutated_phase_freeze_fail_closed() {
    let (bytes, _, _) = source();
    let limits = CostProfileLoadLimits::default();
    for mutation in 0..3 {
        let mut out = Vec::new();
        let mut changed = false;
        for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
            if index == 0 {
                out.extend_from_slice(line);
                continue;
            }
            let mut r: StructuredServiceRecordV6 = serde_json::from_slice(line).unwrap();
            match &mut r {
                StructuredServiceRecordV6::Completed { wave } if !changed && wave.ticket == 8 => {
                    changed = true;
                    if mutation == 0 {
                        continue;
                    }
                    if mutation == 1 {
                        wave.ticket = 7;
                    }
                }
                StructuredServiceRecordV6::PhaseFreeze { children, .. }
                    if mutation == 2 && children[0].parameters_sha256.is_some() =>
                {
                    children[0].parameters_sha256 = Some([91; 32]);
                    changed = true;
                }
                _ => {}
            }
            append(&mut out, &r);
        }
        assert!(replay::replay_source(&out, &limits).is_err());
    }
}
#[test]
fn source6_natural_terminal_requires_original_pending_and_cleanup_contract() {
    let h = header();
    let mut w = wave(1, StructuredPhaseV2::Fit);
    assert!(physical::validate(&h, 1, &w, &mut Default::default()).is_ok());
    for reason in [
        ferrum_types::FinishReason::Cancelled,
        ferrum_types::FinishReason::Length,
    ] {
        w = wave(1, StructuredPhaseV2::Fit);
        w.host_stages.rows[0]
            .terminal
            .as_mut()
            .unwrap()
            .finish_reason = reason;
        let hash = observation::stage_binding(&w.host_stages, None).unwrap();
        w.host_stages
            .structured_evidence
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap()
            .stage_binding = hash;
        assert!(physical::validate(&h, 1, &w, &mut Default::default()).is_err());
    }
    w = wave(1, StructuredPhaseV2::Fit);
    w.host_stages.rows[0]
        .terminal
        .as_mut()
        .unwrap()
        .admission_cancellation_work = serde_json::json!("unknown");
    let hash = observation::stage_binding(&w.host_stages, None).unwrap();
    w.host_stages
        .structured_evidence
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .stage_binding = hash;
    assert!(physical::validate(&h, 1, &w, &mut Default::default()).is_err());
    let (p, _, _) = old::prepared("request-1", 2, 1);
    let good = wave(1, StructuredPhaseV2::Fit);
    assert!(observation::validate(
        &old::header(),
        &p,
        &good.host_stages,
        None,
        good.host_stages
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .stage_binding
    )
    .is_err());
}
