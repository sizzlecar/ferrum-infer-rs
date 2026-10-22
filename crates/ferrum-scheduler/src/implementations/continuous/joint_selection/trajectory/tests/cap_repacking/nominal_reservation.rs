use super::*;

#[test]
fn nominal_reservation_prices_only_capped_work_without_refilling_saved_debit() {
    let (requests, adapter) = fixture();
    let limit = NonZeroUsize::new(32).unwrap();
    let active = adapter.admission.active_mask(&requests).unwrap();
    let reserved = adapter
        .natural_members_from_active(
            &requests,
            Some((
                limit,
                &adapter.cap_commit_targets,
                PrefillPackingMode::ReservePolicyChunk,
            )),
            &active,
        )
        .unwrap();
    assert_eq!(
        geometry(&reserved),
        vec![
            (0, 0, 32, 256),
            (1, 128, 32, 384),
            (2, 256, 32, 512),
            (3, 384, 32, 640),
        ]
    );
    assert_eq!(
        reserved
            .iter()
            .map(|m| m.work.requested.as_range().len())
            .sum::<usize>(),
        128
    );
    assert_eq!(preview(&adapter, &requests, 32).len(), 7);
    // This proves this wave's input, not a fixed width bound for later waves.
    let domain = WidthDomain::new(Instant::now(), reserved.len());
    let works = reserved.iter().map(|m| m.work).collect::<Vec<_>>();
    assert!(matches!(
        domain.predict(
            &CostQuery {
                work: &works,
                work_is_exact: true,
                execution: CostExecution::SingleInvocation,
            },
            domain.at
        ),
        CostPrediction::Known { .. }
    ));

    // A real executor ceiling shrinks the policy quantum as well as work;
    // this is different from merely reaching the prompt's final short tail.
    let (requests, mut limited) = fixture();
    limited.shapes[0].execution_ceiling = Some(16);
    let wave = limited
        .natural_members_from_active(
            &requests,
            Some((
                limit,
                &limited.cap_commit_targets,
                PrefillPackingMode::ReservePolicyChunk,
            )),
            &active,
        )
        .unwrap();
    assert_eq!(
        geometry(&wave),
        vec![
            (0, 0, 16, 256),
            (1, 128, 32, 384),
            (2, 256, 32, 512),
            (3, 384, 32, 640),
            (4, 512, 112, 768),
        ]
    );
}

#[test]
fn nominal_reservation_preview_matches_installed_baseline_through_real_endpoints() {
    let (requests, adapter) = fixture();
    let limit = NonZeroUsize::new(32).unwrap();
    // The same declared six-prefiller domain that rejected the old quarter
    // action now supports the entire new action, not only its first wave.
    let supported_width = preview(&adapter, &requests, 64).len();
    let snapshot = WidthDomain::new(Instant::now(), supported_width);
    let preview = evaluate(
        &requests,
        &adapter,
        &snapshot,
        FirstAction::ReservedPrefillCapCommit(limit),
        &[],
    );
    assert_endpoints(preview.static_result.complete().unwrap(), &requests);
    let preview_queries = snapshot.queries.lock().unwrap().clone();

    let (_, mut installed) = fixture();
    for target in &adapter.cap_commit_targets {
        installed.shapes[target.request].phase_cap = Some(limit);
        installed.shapes[target.request].phase_packing =
            PrefillPacking::ReservePolicyChunk { prior_limit: None };
    }
    installed.cap_commit_targets.clear();
    let installed_snapshot = WidthDomain::new(Instant::now(), supported_width);
    let actual = evaluate(
        &requests,
        &installed,
        &installed_snapshot,
        FirstAction::Static,
        &[],
    );
    assert_endpoints(actual.static_result.complete().unwrap(), &requests);
    assert_eq!(preview_queries, *installed_snapshot.queries.lock().unwrap());
    assert!(installed.shapes[4..]
        .iter()
        .all(|shape| shape.phase_cap.is_none()));
    // Even the final positive chunks retain the live policy quantum. After
    // that completed wave the three untargeted prefills use ordinary packing.
    let final_target_wave = preview_queries
        .iter()
        .find(|work| {
            work[0].kind == AttemptWorkKind::Prefill && work[0].requested.as_range() == (224..256)
        })
        .unwrap();
    assert_eq!(final_target_wave.len(), adapter.cap_commit_targets.len());
    assert!(final_target_wave
        .iter()
        .all(|work| work.requested.as_range().len() == 32));
    assert!(preview_queries
        .iter()
        .filter(|work| work[0].kind == AttemptWorkKind::Prefill)
        .all(|work| work.len() <= adapter.cap_commit_targets.len()));
}
