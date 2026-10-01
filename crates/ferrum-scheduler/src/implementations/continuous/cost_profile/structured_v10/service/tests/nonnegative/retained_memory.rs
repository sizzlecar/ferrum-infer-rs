use super::*;
use crate::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1;

// The original phase workspace charge deliberately stays conservative. This
// budget can hold two complete eight-member workspaces, but not the cumulative
// allocation traffic from three phases whose original samples were released.
#[test]
fn nonnegative_source6_releases_completed_phase_workspace_and_replays() {
    releases_completed_phase_workspace_and_replays(
        NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1,
    );
}

#[test]
fn nonnegative_source6_identified_releases_completed_phase_workspace_and_replays() {
    releases_completed_phase_workspace_and_replays(
        NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2,
    );
}

fn releases_completed_phase_workspace_and_replays(estimator: NonNegativePlanningEstimatorV1) {
    let (p, offered, _) = old::prepared("memory-budget", 2, 1);
    let input = prepared::project_service_actual_with_domain(&p, &offered, &domain())
        .unwrap()
        .with_settled_terminal_causes(&[])
        .unwrap();
    let per_sample = input.retained_numeric_bytes().unwrap() * 12
        + std::mem::size_of::<StructuredNumericObservationV2>() * 4;
    let mut h = estimator_header(estimator);
    let budget = per_sample * 16;
    h.declaration.maximum_retained_numeric_bytes = budget;
    let h = StructuredServiceHeaderV6::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap();
    let limits = CostProfileLoadLimits::default();
    let mut bytes = Vec::new();
    append(&mut bytes, &h);
    let mut collector = StructuredServiceCollectorV6::new(h, limits.clone()).unwrap();
    let empty_charge = collector.retained_numeric_bytes().unwrap();
    for index in 0..3 {
        let phase = phase_at(index);
        let first = index as u64 * 8 + 1;
        let open = StructuredServiceRecordV6::PhaseOpen {
            phase,
            opened_at_ns: first * 2_000 - 1,
            fifo_cutoff: (first - 1) * 3,
        };
        collector.push(&open).unwrap();
        append(&mut bytes, &open);
        for ticket in first..first + 8 {
            let record = StructuredServiceRecordV6::Completed {
                wave: continuous_wave(ticket, phase),
            };
            collector.push(&record).unwrap_or_else(|error| {
                panic!("phase {phase:?}, original ticket {ticket}: {error}")
            });
            append(&mut bytes, &record);
        }
        let open_charge = collector.retained_numeric_bytes().unwrap();
        let frozen = collector.freeze((first + 7) * 2_000 + 1_101).unwrap();
        let frozen_charge = collector.retained_numeric_bytes().unwrap();
        assert!(open_charge <= budget && frozen_charge <= budget);
        assert!(
            frozen_charge < open_charge,
            "released phase workspace must return capacity"
        );
        assert!(
            frozen_charge > empty_charge,
            "surviving model and population metadata stay charged"
        );
        let StructuredServiceRecordV6::PhaseFreeze { children, .. } = &frozen else {
            unreachable!()
        };
        assert_eq!(children[0].members, 8);
        assert_eq!(children[0].failure, None);
        assert_eq!(
            children[0].nonnegative_fit_certificate.is_some(),
            index == 0
        );
        if let Some(certificate) = &children[0].nonnegative_fit_certificate {
            assert_eq!(
                certificate.signed_basis.is_some(),
                estimator == NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2,
                "the original Fit freeze must carry the selected estimator's certificate"
            );
            if let Some(signed) = &certificate.signed_basis {
                assert_eq!(signed.anchor_indices.len(), certificate.geometry_rank);
                assert_eq!(
                    signed.normalized_query_to_anchor_bits.len(),
                    certificate.geometry_rank * certificate.column_maxima.len()
                );
            }
        }
        append(&mut bytes, &frozen);
    }
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
    let replayed = replay::replay_source(&bytes, &limits).unwrap();
    assert_eq!(collector.qualified_children(), 1);
    assert_eq!(replayed.qualified_children(), 1);
    let live = collector.models().next().unwrap().1;
    let imported = replayed.models().next().unwrap().1;
    assert_eq!(live.parameters_signature(), imported.parameters_signature());
    assert_eq!(
        live.nonnegative_fit_certificate(),
        imported.nonnegative_fit_certificate()
    );
    assert_eq!(
        live.nonnegative_fit_certificate()
            .unwrap()
            .signed_basis
            .is_some(),
        estimator == NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
    );
    let query = StructuredQueryV2::exact(input);
    assert_eq!(
        live.predict_query(&old::fingerprint().into(), &query, 50_000)
            .unwrap()
            .planning_ns,
        imported
            .predict_query(&old::fingerprint().into(), &query, 50_000)
            .unwrap()
            .planning_ns,
    );
}

fn estimator_header(estimator: NonNegativePlanningEstimatorV1) -> StructuredServiceHeaderV6 {
    let mut header = physical_header();
    header
        .declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = estimator;
    // Rebind the actual declaration, rather than editing an already signed
    // source header or changing the model after qualification.
    let budget = header.declaration.maximum_retained_numeric_bytes;
    with_budget(header, budget)
}

fn with_budget(mut h: StructuredServiceHeaderV6, limit: usize) -> StructuredServiceHeaderV6 {
    h.declaration.maximum_retained_numeric_bytes = limit;
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

#[test]
fn nonnegative_source6_actual_retained_boundary_is_not_cleared_or_unbounded() {
    actual_retained_boundary(NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1);
}

#[test]
fn nonnegative_source6_identified_actual_retained_boundary_is_not_cleared_or_unbounded() {
    actual_retained_boundary(NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2);
}

fn actual_retained_boundary(estimator: NonNegativePlanningEstimatorV1) {
    let open = StructuredServiceRecordV6::PhaseOpen {
        phase: StructuredPhaseV2::Fit,
        opened_at_ns: 1_999,
        fifo_cutoff: 0,
    };
    let first = StructuredServiceRecordV6::Completed {
        wave: continuous_wave(1, StructuredPhaseV2::Fit),
    };
    let mut probe = StructuredServiceCollectorV6::new(
        estimator_header(estimator),
        CostProfileLoadLimits::default(),
    )
    .unwrap();
    probe.push(&open).unwrap();
    probe.push(&first).unwrap();
    let exact = probe.retained_numeric_bytes().unwrap();
    for (budget, expected) in [(exact, true), (exact - 1, false)] {
        let h = with_budget(estimator_header(estimator), budget);
        let mut collector =
            StructuredServiceCollectorV6::new(h, CostProfileLoadLimits::default()).unwrap();
        collector.push(&open).unwrap();
        let result = collector.push(&first);
        assert_eq!(result.is_ok(), expected, "one-byte boundary: {result:?}");
        if expected {
            assert_eq!(collector.retained_numeric_bytes(), Some(exact));
            let second = StructuredServiceRecordV6::Completed {
                wave: continuous_wave(2, StructuredPhaseV2::Fit),
            };
            assert!(matches!(
                collector.push(&second),
                Err(CostProfileError::Limit("source6 numeric capacity"))
            ));
        }
        assert_eq!(collector.qualified_children(), 0);
        assert!(collector.freeze(17_101).is_err());
    }
}

#[test]
fn rowspace_source6_keeps_qualified_support_and_population_metadata_charged() {
    let (bytes, input, _, collector) = super::super::collected_source();
    let charge = collector.retained_numeric_bytes().unwrap();
    let model = collector.models().next().unwrap().1;
    let payload = model.retained_payload_bytes().unwrap();
    assert!(payload > std::mem::size_of::<QualifiedStructuredModelV2>());
    assert!(charge >= payload);
    let replayed = replay::replay_source(&bytes, &CostProfileLoadLimits::default()).unwrap();
    let replay_model = replayed.models().next().unwrap().1;
    // Wire decoding may reserve a different Vec capacity than the live typed
    // builder. Each ledger must count its own allocation, not agree in bytes.
    assert!(
        replayed.retained_numeric_bytes().unwrap()
            >= replay_model.retained_payload_bytes().unwrap()
    );
    assert_eq!(
        model.parameters_signature(),
        replay_model.parameters_signature()
    );
    let query = StructuredQueryV2::exact(input);
    assert_eq!(
        model
            .predict_query(&old::fingerprint().into(), &query, 50_000)
            .unwrap()
            .planning_ns,
        replay_model
            .predict_query(&old::fingerprint().into(), &query, 50_000)
            .unwrap()
            .planning_ns,
    );
}
