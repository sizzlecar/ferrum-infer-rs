use super::super::owner_diagnostic::diagnose_bytes;
use super::*;

#[test]
fn source6_owner_diagnostic_validates_complete_original_source() {
    let (bytes, _, _) = source();
    let report = diagnose_bytes(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert!(report.complete_source_verified);
    assert!(report.stop.is_none());
    assert!(report.first_outside_by_phase.iter().all(Option::is_none));
    assert_eq!(report.verified_records, 31);
    assert_eq!(
        report.source_sha256,
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    for phase in report.route_population {
        assert_eq!(phase.attempted, 8);
        assert_eq!(phase.eligible_route, 8);
    }
}

fn different_scope_source(invalid_receipt: bool) -> Vec<u8> {
    let original = header();
    let mut declaration = original.declaration.clone();
    // This changes only the declared training scope. The actual sample still
    // comes from the original canonical producer and complete settlement.
    declaration.scopes[0].owner.installed_policy[0] ^= 1;
    let header = StructuredServiceHeaderV6::new(
        original.capture_identity,
        original.generation,
        original.fingerprint,
        original.producer,
        original.opening,
        declaration,
        original.maximum_file_bytes,
    )
    .unwrap();
    let mut bytes = Vec::new();
    append(&mut bytes, &header);
    append(
        &mut bytes,
        &StructuredServiceRecordV6::PhaseOpen {
            phase: StructuredPhaseV2::Fit,
            opened_at_ns: 1_999,
            fifo_cutoff: 0,
        },
    );
    let mut actual = wave(1, StructuredPhaseV2::Fit);
    if invalid_receipt {
        actual.host_stages.prepare_started_at_ns = Some(2_001);
    }
    append(
        &mut bytes,
        &StructuredServiceRecordV6::Completed { wave: actual },
    );
    append(
        &mut bytes,
        &StructuredServiceRecordV6::Footer {
            offered: 1,
            accepted_fifo_cutoff: 3,
            closing: StructuredServiceClockV6 {
                monotonic_ns: 4_000,
                wall_unix_ns: 1_003_999,
            },
            failure: Some("original generation stopped".into()),
        },
    );
    bytes
}

#[test]
fn source6_owner_diagnostic_reports_valid_prefix_without_importing_failed_source() {
    let bytes = different_scope_source(false);
    let report = diagnose_bytes(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert!(!report.complete_source_verified);
    assert_eq!(report.verified_records, 2);
    assert_eq!(report.stop.as_ref().unwrap().record, 3);
    let first = report.first_outside_by_phase[0].as_ref().unwrap();
    assert_eq!(first.ticket, 1);
    assert_eq!(first.nearest_declaration_index, 0);
    assert_eq!(
        first.differences,
        vec![StructuredOwnerDifferenceV1::InstalledPolicy]
    );
    assert_eq!(first.actual_owner, header().declaration.scopes[0].owner);
    assert!(report.first_outside_by_phase[1..]
        .iter()
        .all(Option::is_none));
}

#[test]
fn source6_owner_diagnostic_never_reports_owner_from_invalid_settlement() {
    let bytes = different_scope_source(true);
    let report = diagnose_bytes(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert!(!report.complete_source_verified);
    assert_eq!(report.verified_records, 1);
    assert_eq!(report.stop.as_ref().unwrap().record, 2);
    assert!(report.first_outside_by_phase.iter().all(Option::is_none));
}
