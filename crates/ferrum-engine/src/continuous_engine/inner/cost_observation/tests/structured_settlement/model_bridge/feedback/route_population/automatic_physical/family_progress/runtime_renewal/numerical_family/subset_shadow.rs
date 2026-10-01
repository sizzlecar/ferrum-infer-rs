//! Catalog selection counterexample, using original CPU source7 qualification
//! for both donors. The first child enters through real startup activation;
//! this does not substitute for the separately tested source8 cohort protocol.
use super::*;
use crate::continuous_engine::inner::cost_observation::live_calibration::Publication;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    DeclaredAlgorithmUniverseV1, OwnerPhaseSupportPolicyV1,
};

fn selected_wave(algorithm: &'static str, rows: u32, pending: bool, length: bool) -> Wave {
    let original = ordinary_wave(rows, pending, length, 7);
    wave_with_host_rows(algorithm, 7, original.host, rows)
}

fn selected_query(f: &Families, algorithm: &'static str, rows: u32) -> StructuredQueryV2 {
    let w = selected_wave(algorithm, rows, false, false);
    StructuredQueryV2::from_future_with_domain(
        &w.prepared.exact,
        &w.prepared.selected,
        &w.prepared.recipe,
        &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
        &f.domain,
    )
    .unwrap()
}

fn selected_block(f: &mut Families, algorithm: &'static str, rows: u32) {
    selected_algorithms_block(f, &[algorithm], rows);
}

fn selected_algorithms_block(f: &mut Families, algorithms: &[&'static str], rows: u32) {
    f.runtime.consume_samples();
    for ordinal in 0..BLOCK_OFFERS {
        let algorithm = algorithms[ordinal % algorithms.len()];
        let corner = ordinal / algorithms.len();
        let pending = corner % 2 == 0;
        let length = corner / 2 % 2 == 0;
        let w = selected_wave(algorithm, rows, pending, length);
        let stages = fixture::record_cohort_route(&f.runtime, &f.clock, w)
            .expect("original selected CPU commands and complete host settlement");
        assert_eq!(stages.rows.len(), rows as usize);
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_some_and(Result::is_ok));
        assert!(stages.route_evidence.is_some());
        assert!(stages.rows.iter().all(|r| r.terminal.is_some() == length));
        f.runtime.consume_samples();
        f.recorded += 1;
        assert_population_healthy(f);
    }
    let sink = f.runtime.audit_snapshot().sink;
    assert_eq!(sink.raw_offered, f.recorded);
    assert_eq!(sink.raw_resolved, f.recorded);
    assert_eq!(sink.raw_lost, 0);
    assert_eq!(sink.raw_resolution_failed, 0);
}

#[tokio::test]
async fn automatic_catalog_phase_support_dispatch_preserves_retained_known_input_support() {
    // Cold donor: its entire selected algorithm roster was actually executed
    // in Discovery/Fit/Residual/Qualification, all at B1.
    let mut donor = families(SloAutomaticCalibrationDiagnosticsV1::MemoryOnly);
    for _ in 0..4 {
        selected_block(&mut donor, A, 1);
    }
    assert_eq!(donor.live().audit().qualified_publications, 1);
    let original = only_child(&donor);
    let cold_query = selected_query(&donor, A, 1);
    let cold_at = donor.clock.now_ns().unwrap();
    original
        .predict_query_local(original.fingerprint(), &cold_query, cold_at)
        .unwrap();
    let original_expiry = expires_at(&original, &cold_query, cold_at);
    let original_provenance = serde_json::to_value(original.provenance()).unwrap();
    let publication = Publication {
        children: vec![original.clone()],
        receipt: donor.runtime.training.published_catalog_receipt().unwrap(),
    };
    donor.runtime.shutdown().await.unwrap();

    let mut target =
        families_before_or_after_start(SloAutomaticCalibrationDiagnosticsV1::MemoryOnly, false);
    target.clock.set(cold_at);
    let other_query = selected_query(&target, B, 2);
    let seed = DeclaredAlgorithmUniverseV1::from_inputs(
        [cold_query.input(), other_query.input()],
        StructuredSettingsV2::default().max_axes,
    )
    .unwrap();
    assert!(seed.algorithm_count() > original.algorithm_universe().unwrap().algorithm_count());
    assert!(seed
        .contains_checked_algorithms(cold_query.input())
        .unwrap());
    target
        .runtime
        .install_startup_algorithm_seed(seed.clone())
        .unwrap();
    let waiter = target
        .runtime
        .sink
        .request_catalog_activation(0, publication)
        .unwrap();
    target.runtime.consume_samples();
    waiter
        .wait()
        .await
        .unwrap()
        .catalog_activation
        .unwrap()
        .unwrap();
    let before = target.runtime.snapshot().unwrap();
    let before_identity = before.prospective_identity().unwrap();
    let before_prediction = before
        .audit_structured_query_v2(&cold_query, cold_at)
        .unwrap();
    assert_eq!(
        before.prospective_source(&cold_query).unwrap().domain,
        *original.domain_signature()
    );
    let before_feedback = serde_json::to_value(target.runtime.audit_snapshot()).unwrap()
        ["structured_feedback"]
        .clone();
    assert!(before_feedback["revoked"].is_null());

    // The next independent source declares A+B in advance, then receives only
    // actual B2/B work. No old Fit member or fabricated counter fills an axis.
    target.runtime.begin_automatic_calibration().unwrap();
    target.runtime.consume_samples();
    let header = target.runtime.automatic_original_header_for_test().unwrap();
    assert_eq!(
        header.declaration.schedule.phase_support,
        Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1)
    );
    for phase in 0..4 {
        selected_block(&mut target, B, 2);
        assert_eq!(
            target.live().audit().qualified_publications,
            1 + u64::from(phase == 3)
        );
        assert_eq!(before_identity.current(), phase != 3);
    }
    let now = target.clock.now_ns().unwrap();
    let children = target.runtime.training.live_catalog_children(now).unwrap();
    assert_eq!(
        children.len(),
        2,
        "distinct declared universes remain distinct populations"
    );
    let retained = children
        .iter()
        .find(|c| c.domain_signature() == original.domain_signature())
        .unwrap();
    let new = children
        .iter()
        .find(|c| c.domain_signature() != original.domain_signature())
        .unwrap();
    assert_eq!(new.algorithm_universe(), Some(&seed));
    assert!(!new.same_population(retained));
    assert_eq!(
        serde_json::to_value(retained.provenance()).unwrap(),
        original_provenance
    );
    assert_eq!(expires_at(retained, &cold_query, now), original_expiry);
    assert_eq!(
        retained
            .predict_query_local(retained.fingerprint(), &cold_query, now)
            .unwrap()
            .planning_ns,
        before_prediction.planning_ns
    );
    assert_eq!(
        new.predict_query_local(new.fingerprint(), &cold_query, now)
            .unwrap_err(),
        StructuredUnknownV2::QualificationCoverage
    );
    assert_eq!(new.catalog_input_membership(&cold_query), Ok(Some(false)));
    assert_eq!(
        retained.catalog_input_membership(&cold_query),
        Ok(Some(true))
    );
    new.predict_query_local(new.fingerprint(), &other_query, now)
        .unwrap();
    for child in [&original, new] {
        assert_eq!(
            child.provenance().phases.each_ref().map(|p| p.members),
            [BLOCK_OFFERS; 3]
        );
    }

    let merged = target.runtime.snapshot().unwrap();
    assert!(merged.model_version() > before.model_version());
    assert_eq!(
        merged.prospective_source(&cold_query).unwrap().domain,
        *retained.domain_signature()
    );
    // Dispatch excludes the new child's unsupported input before prediction.
    // Original provenance/TTL and the retained child's actual prediction stay
    // unchanged; no failing prediction is used to search another model.
    assert_eq!(
        merged
            .audit_structured_query_v2(&cold_query, now)
            .unwrap()
            .planning_ns,
        before_prediction.planning_ns
    );
    merged.audit_structured_query_v2(&other_query, now).unwrap();
    assert_eq!(
        before
            .audit_structured_query_v2(&cold_query, now)
            .unwrap_err(),
        StructuredUnknownV2::RuntimeValidity
    );
    let after_feedback = serde_json::to_value(target.runtime.audit_snapshot()).unwrap()
        ["structured_feedback"]
        .clone();
    assert!(after_feedback["revoked"].is_null());
    for scope in before_feedback["scopes"].as_array().unwrap() {
        let after = after_feedback["scopes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["signature"] == scope["signature"])
            .unwrap();
        assert_eq!(
            after, scope,
            "the old child retains its original feedback state"
        );
    }
    assert_eq!(
        retained
            .predict_query_local(retained.fingerprint(), &cold_query, original_expiry + 1)
            .unwrap_err(),
        StructuredUnknownV2::Stale
    );
    target.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_catalog_phase_support_expired_chosen_child_never_falls_back() {
    // The older large-U child actually qualifies both A and B. A later
    // independent small-U child also qualifies A, and remains current longer.
    let mut donor = families(SloAutomaticCalibrationDiagnosticsV1::MemoryOnly);
    for _ in 0..4 {
        selected_algorithms_block(&mut donor, &[A, B], 1);
    }
    assert_eq!(donor.live().audit().qualified_publications, 1);
    let chosen = only_child(&donor);
    let query = selected_query(&donor, A, 1);
    let donor_now = donor.clock.now_ns().unwrap();
    let chosen_expiry = expires_at(&chosen, &query, donor_now);
    let publication = Publication {
        children: vec![chosen.clone()],
        receipt: donor.runtime.training.published_catalog_receipt().unwrap(),
    };
    donor.runtime.shutdown().await.unwrap();

    let mut target =
        families_before_or_after_start(SloAutomaticCalibrationDiagnosticsV1::MemoryOnly, false);
    target.clock.set(donor_now);
    let waiter = target
        .runtime
        .sink
        .request_catalog_activation(0, publication)
        .unwrap();
    target.runtime.consume_samples();
    waiter
        .wait()
        .await
        .unwrap()
        .catalog_activation
        .unwrap()
        .unwrap();
    target.runtime.begin_automatic_calibration().unwrap();
    target.runtime.consume_samples();
    for _ in 0..4 {
        selected_block(&mut target, A, 1);
    }
    assert_eq!(target.live().audit().qualified_publications, 2);
    let now = target.clock.now_ns().unwrap();
    let children = target.runtime.training.live_catalog_children(now).unwrap();
    assert_eq!(children.len(), 2);
    let smaller = children
        .iter()
        .find(|c| c.domain_signature() != chosen.domain_signature())
        .unwrap();
    assert!(
        chosen.algorithm_universe().unwrap().algorithm_count()
            > smaller.algorithm_universe().unwrap().algorithm_count()
    );
    assert_eq!(chosen.catalog_input_membership(&query), Ok(Some(true)));
    assert_eq!(smaller.catalog_input_membership(&query), Ok(Some(true)));
    assert!(expires_at(smaller, &query, now) > chosen_expiry);
    let snapshot = target.runtime.snapshot().unwrap();
    assert_eq!(
        snapshot.prospective_source(&query).unwrap().domain,
        *chosen.domain_signature()
    );
    assert_eq!(
        snapshot
            .audit_structured_query_v2(&query, now)
            .unwrap()
            .planning_ns,
        chosen
            .predict_query_local(chosen.fingerprint(), &query, now)
            .unwrap()
            .planning_ns
    );

    let expired_at = chosen_expiry.checked_add(1).unwrap();
    smaller
        .predict_query_local(smaller.fingerprint(), &query, expired_at)
        .unwrap();
    assert_eq!(chosen.catalog_input_membership(&query), Ok(Some(true)));
    assert_eq!(
        snapshot.prospective_source(&query).unwrap().domain,
        *chosen.domain_signature()
    );
    assert_eq!(
        snapshot
            .audit_structured_query_v2(&query, expired_at)
            .unwrap_err(),
        StructuredUnknownV2::Stale,
        "expired chosen model cannot retry a smaller current model"
    );
    target.runtime.shutdown().await.unwrap();
}
