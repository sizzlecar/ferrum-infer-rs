//! Real capacity evidence changes, not injected Known/Unknown route answers.
use super::*;
use crate::continuous_engine::inner::slo_controller::tests::prefill;

fn resource_view(evidence: &CoreEvidence) -> ResourcePlanningView {
    let sessions = evidence
        .sessions
        .iter()
        .map(Arc::as_ref)
        .collect::<Vec<_>>();
    match evidence
        .fixture
        .as_ref()
        .unwrap()
        .plan_resources
        .execution_cost_route_view(
            &sessions,
            &vec![1; sessions.len()],
            evidence.lane.as_ref().unwrap().as_ref(),
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
        ExecutionCostRouteAvailability::Known(route) => route.resource_view().clone(),
        other => panic!("isolated fixture resource view: {other:?}"),
    }
}

fn close_peer(peer: contract::Fixture) {
    drop(peer.registry);
    drop(peer.impostor_registry);
    drop(peer.runtime);
    assert!(matches!(
        vnext::PlanRuntimeResources::close(peer.plan_resources),
        Ok(vnext::PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn controller_fixture_accounts_isolate_unrelated_plans_but_preserve_explicit_sharing() {
    let first = CoreEvidence::new(1);
    let before = resource_view(&first);
    let second = CoreEvidence::new(1);
    let first_device = first
        .fixture
        .as_ref()
        .unwrap()
        .runtime
        .descriptor
        .id
        .clone();
    let second_device = &second.fixture.as_ref().unwrap().runtime.descriptor.id;
    assert_ne!(&first_device, second_device);
    assert!(before.same_live_evidence(&resource_view(&first)));

    // A deliberately shared identity still uses the real process-wide account.
    // Provisioning a second plan changes its live capacity evidence.
    let peer = contract::fixture_with_device_id(first_device);
    assert!(!before.same_live_evidence(&resource_view(&first)));
    close_peer(peer);
}

#[tokio::test]
async fn missing_reference_retries_real_changed_evidence_without_a_time_promise() {
    let (mut engine, _, executor) = fixture_with_width(1).await;
    {
        let inner = Arc::get_mut(&mut engine.inner).unwrap();
        inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
        inner.prefill_reference_runtime = None;
    }
    let (id, session) = prefill::request(&engine, 1, 1).await;
    prefill::admit(&engine, 1).await;
    let shared_peer = Arc::new(Mutex::new(None));
    let device = executor
        .evidence
        .fixture
        .as_ref()
        .unwrap()
        .runtime
        .descriptor
        .id
        .clone();
    *executor.before_resource_revalidation.lock() = Some(Box::new({
        let shared_peer = Arc::clone(&shared_peer);
        move || {
            *shared_peer.lock() = Some(contract::fixture_with_device_id(device));
        }
    }));
    let changed_comparisons = Arc::new(AtomicUsize::new(0));
    *executor.after_resource_revalidation.lock() = Some(Box::new({
        let executor = Arc::clone(&executor);
        let inner = Arc::clone(&engine.inner);
        let changed_comparisons = Arc::clone(&changed_comparisons);
        move || {
            // The actual before/after resource comparison is Known(false).
            // No read failure or fabricated route caused this publication retry.
            assert!(executor
                .resource_revalidation_changed
                .load(Ordering::Acquire));
            assert!(executor.resource_planning_unknown.lock().is_none());
            assert!(executor.cost_route_unknown.lock().is_none());
            assert_eq!(executor.physical.load(Ordering::Acquire), 0);
            let state = inner.slo_controller.lock();
            let observation = state.last_observation.unwrap();
            assert_eq!(observation.disposition, "unknown");
            assert_eq!(observation.reason, "missing_reference_work");
            changed_comparisons.fetch_add(1, Ordering::AcqRel);
        }
    }));

    // The helper must follow the real independent retry timer despite the
    // earlier observation. The unchanged-state checks remain active each turn.
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
    })
    .await;
    assert_eq!(changed_comparisons.load(Ordering::Acquire), 1);
    assert!(!executor
        .resource_revalidation_changed
        .load(Ordering::Acquire));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(!engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .has_started_witness());
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert!(!engine.inner.sequences.read().contains_key(&id));
    cleanup(engine, session).await;
    close_peer(shared_peer.lock().take().unwrap());
}
