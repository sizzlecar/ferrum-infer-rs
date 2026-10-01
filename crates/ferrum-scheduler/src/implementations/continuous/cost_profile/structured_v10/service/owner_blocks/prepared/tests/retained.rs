use super::*;

#[test]
fn source8_driver_and_collector_share_one_exact_capacity_boundary() {
    let h = header();
    let maximum = h.declaration.population.maximum_retained_numeric_bytes;
    let mut c =
        StructuredPreparedOwnerBlockCollectorV8::new(h, CostProfileLoadLimits::default()).unwrap();
    let baseline = c.audit().retained_numeric_bytes;
    assert!(baseline > 0 && baseline < maximum);
    c.retain_external(maximum - baseline).unwrap();
    assert_eq!(c.audit().retained_numeric_bytes, maximum);
    c.retain_external(0).unwrap();
    assert_eq!(c.audit().retained_numeric_bytes, baseline);
    assert!(c.retain_external(maximum - baseline + 1).is_err());
    assert!(c.audit().poisoned);
    assert!(
        c.retain_external(0).is_err(),
        "a failed capacity gate cannot renew the original population"
    );
    assert_eq!(c.qualified_children(), 0);
}

#[test]
fn source8_driver_capacity_overflow_cannot_open_or_publish() {
    let h = header();
    let mut c =
        StructuredPreparedOwnerBlockCollectorV8::new(h, CostProfileLoadLimits::default()).unwrap();
    assert!(c.retain_external(usize::MAX).is_err());
    assert!(c.open_block(2, 0).is_err());
    assert!(c
        .checkpoint(StructuredServiceClockV7 {
            monotonic_ns: 2,
            wall_unix_ns: 1_000_001
        })
        .is_err());
}

#[test]
fn source8_declaration_retains_frozen_universe_within_the_same_payload_limit() {
    use crate::implementations::continuous::cost_model::structured_v2::{
        DeclaredAlgorithmUniverseV1, StructuredPopulationPolicyV1,
    };
    use crate::implementations::continuous::cost_profile::structured_v10::prepared;
    let mut declaration = header().declaration;
    let domain = &declaration
        .population
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .workload_domain;
    let (original, offered, _) = old::prepared("universe-retained", 2, 1);
    let input = prepared::project_service_actual_with_domain(&original, &offered, domain).unwrap();
    let universe = DeclaredAlgorithmUniverseV1::from_inputs([&input], 4096).unwrap();
    let before = declaration.retained_payload_bytes().unwrap();
    let envelope = declaration
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap();
    envelope.population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    envelope.algorithm_universe = Some(universe.clone());
    envelope.validate().unwrap();
    let charged = declaration.retained_payload_bytes().unwrap();
    assert_eq!(charged, before + universe.retained_payload_bytes().unwrap());
    declaration.population.maximum_retained_numeric_bytes = charged;
    declaration.validate().unwrap();
    declaration.population.maximum_retained_numeric_bytes = charged - 1;
    assert!(matches!(
        declaration.validate(),
        Err(CostProfileError::Limit(_))
    ));
    declaration
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .algorithm_universe = None;
    assert_eq!(declaration.retained_payload_bytes(), Some(before));
    declaration.population.maximum_retained_numeric_bytes = before;
    declaration.validate().unwrap();
}

#[test]
fn source8_rejects_source7_discovery_universe_even_with_self_consistent_header() {
    use crate::implementations::continuous::cost_model::structured_v2::{
        OwnerAlgorithmUniversePolicyV1, StructuredPopulationPolicyV1,
    };
    let mut h = header();
    h.declaration.population.schedule.algorithm_universe =
        Some(OwnerAlgorithmUniversePolicyV1::FirstOrdinaryDiscoveryBlockSubsetV1);
    h.declaration
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    assert!(StructuredPreparedOwnerBlockHeaderV8::new(
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
