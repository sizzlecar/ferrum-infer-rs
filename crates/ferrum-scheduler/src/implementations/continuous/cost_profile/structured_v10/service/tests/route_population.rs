use super::*;
use ferrum_types::SloCalibrationRoutePopulationV1 as P;

#[test]
fn service_route_population_old_declaration_bytes_stay_all_attempts() {
    let h = header();
    let value = serde_json::to_value(&h.declaration).unwrap();
    assert!(value.get("route_population").is_none());
    let decoded: StructuredServiceDeclarationV6 = serde_json::from_value(value).unwrap();
    assert_eq!(decoded.route_population, P::AllAttempts);
    let old = h.protocol;
    let mut declaration = h.declaration;
    declaration.route_population = P::WarmOrGraphDisabledV1;
    let warm = StructuredServiceHeaderV6::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        declaration,
        h.maximum_file_bytes,
    )
    .unwrap();
    assert_ne!(
        warm.protocol, old,
        "new population must be frozen into source/profile binding"
    );
}
#[test]
fn service_declared_route_requires_original_presubmit_proof_and_cannot_supply_extra_offers() {
    let old = header();
    let mut d = old.declaration;
    d.route_population = P::WarmOrGraphDisabledV1;
    let h = StructuredServiceHeaderV6::new(
        old.capture_identity,
        old.generation,
        old.fingerprint,
        old.producer,
        old.opening,
        d,
        old.maximum_file_bytes,
    )
    .unwrap();
    let mut c = StructuredServiceCollectorV6::new(h, CostProfileLoadLimits::default()).unwrap();
    c.push(&StructuredServiceRecordV6::PhaseOpen {
        phase: StructuredPhaseV2::Fit,
        opened_at_ns: 2,
        fifo_cutoff: 0,
    })
    .unwrap();
    assert!(c
        .push(&StructuredServiceRecordV6::Completed {
            wave: wave(1, StructuredPhaseV2::Fit)
        })
        .is_err());
    assert_eq!(
        c.route_population_counts()[0],
        StructuredServiceRouteCountsV1::default()
    );
    assert!(
        c.freeze(30_000).is_err(),
        "missing proof poisons the window instead of being replaced"
    );
}
#[test]
fn service_legacy_qualified_source_replays_without_route_claims() {
    let (bytes, _, _, collected) = collected_source();
    assert_eq!(
        collected.route_population_counts(),
        [StructuredServiceRouteCountsV1 {
            attempted: 8,
            eligible_route: 8,
            outside_declared_route: 0,
            no_submission: 0,
        }; 3]
    );
    let replayed = replay::replay_source(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(replayed.qualified_children(), 1);
    assert_eq!(
        replayed.route_population_counts(),
        collected.route_population_counts()
    );
    assert!(!std::str::from_utf8(&bytes)
        .unwrap()
        .contains("route_population"));
}
