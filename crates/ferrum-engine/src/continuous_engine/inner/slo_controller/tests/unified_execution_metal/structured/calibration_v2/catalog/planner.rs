//! Real native projection + scheduler-owned evidence; no receipt constructor.
use super::*;
use ferrum_interfaces::execution_cost::HostContentForecastV2;
use ferrum_scheduler::implementations::continuous::{
    cost_model::{ExecutionFingerprint, WaveExecutionShape},
    slo_planner::{
        AnchoredPlanningCostModel, PlanningCost, PlanningCostClockAnchor, PlanningCostEvidence,
        PlanningCostEvidenceRequirement, PlanningCostModel, PlanningExecutionContext,
    },
};
use std::cell::Cell;
#[path = "planner/query_metrics.rs"]
mod query_metrics;
#[path = "planner/required_audit.rs"]
mod required_audit;

struct Compare<'a> {
    direct: AnchoredPlanningCostModel<'a>,
    catalog: AnchoredPlanningCostModel<'a>,
    original_exact: WaveExecutionShape,
    known_original: Cell<usize>,
    queries: Cell<usize>,
    direct_anchor: PlanningCostClockAnchor,
    catalog_anchor: PlanningCostClockAnchor,
    ttl: u64,
}
impl PlanningCostModel for Compare<'_> {
    fn model_version(&self) -> u64 {
        self.direct.model_version()
    }
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        PlanningCostEvidenceRequirement::StructuredV2
    }
    fn supports_empirical_host_content(&self) -> bool {
        true
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        _: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        panic!("the real V2 planner must provide its bound evidence")
    }
    fn predict_with_evidence(
        &self,
        fp: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now: u64,
    ) -> Option<PlanningCost> {
        // Every alternative the planner presents goes through both original
        // imported snapshots, including unsupported future scopes.
        let ordinal = self.queries.get() + 1;
        self.queries.set(ordinal);
        if ordinal <= 8 {
            eprintln!("native planner query={ordinal} original_exact={} now={now} direct_clock={:?} catalog_clock={:?}", shape == &self.original_exact, self.direct_anchor.cost_time_ns(now), self.catalog_anchor.cost_time_ns(now));
            match evidence.map(|value| value.structured_query_v2_for(shape)) {
                Some(Ok(query)) => eprintln!(
                    "bound V2 owner={:?} domain={:?}",
                    query.owner(),
                    query.domain_signature()
                ),
                other => eprintln!("bound V2 input unavailable: {other:?}"),
            }
        }
        let a = self.direct.predict_with_evidence(fp, shape, evidence, now);
        let b = self.catalog.predict_with_evidence(fp, shape, evidence, now);
        assert_eq!(
            a.is_some(),
            b.is_some(),
            "catalog changed scope availability"
        );
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a.typical_ns, b.typical_ns);
            assert_eq!(a.planning_ns, b.planning_ns);
            assert_eq!(a.model_version, b.model_version);
            // Different real import clocks legitimately change remaining TTL.
            assert!(a.valid_for_ns > 0 && b.valid_for_ns > 0);
            let expired = now.checked_add(self.ttl).unwrap();
            assert!(self
                .direct
                .predict_with_evidence(fp, shape, evidence, expired)
                .is_none());
            assert!(self
                .catalog
                .predict_with_evidence(fp, shape, evidence, expired)
                .is_none());
            if shape == &self.original_exact {
                let query = evidence.unwrap().structured_query_v2_for(shape).unwrap();
                assert_eq!(query.owner().role, StructuredWaveRoleV2::OrdinaryDecode);
                self.known_original.set(self.known_original.get() + 1);
            }
        }
        a
    }
}

pub(super) async fn compare_real_queries(
    session: &mut CalibrationSession,
    direct: &EngineCostRuntime,
    catalog: &EngineCostRuntime,
    ttl: u64,
) {
    // The original collector is already finished. This is one extra complete
    // request, never another training population or a synthetic pending state.
    let (id, consume) = add_native_request(session).await;
    let inner = session.test_engine_inner();
    ready(&inner, &id).await;
    admit(session, &id).await;
    while let Some(_) = frontier(session, &id).prefill_progress() {
        ready(&inner, &id).await;
        let work = frontier(session, &id).prefill_work(n32(8)).unwrap();
        wave(session, vec![work]).await;
    }
    ready(&inner, &id).await;
    assert_eq!(frontier(session, &id).generated_tokens(), 1);
    {
        let mut captured = capture(&inner);
        let direct_snapshot = direct.snapshot().unwrap();
        let catalog_snapshot = catalog.snapshot().unwrap();
        assert_eq!(
            direct_snapshot.fingerprint(),
            &captured.snapshot.fingerprint
        );
        assert_eq!(
            catalog_snapshot.fingerprint(),
            &captured.snapshot.fingerprint
        );
        assert_eq!(
            direct_snapshot.model_version(),
            catalog_snapshot.model_version()
        );
        let anchor = |runtime: &EngineCostRuntime| {
            let before = slo_clock_now();
            let at = runtime.clock.now_ns().unwrap();
            let after = slo_clock_now();
            PlanningCostClockAnchor::conservative_read(&captured.origin, before, at, after).unwrap()
        };
        let direct_anchor = anchor(direct);
        let catalog_anchor = anchor(catalog);
        // Only the local read-only query snapshot switches model. Engine state,
        // resources, real request policies and publication authority stay owned
        // by the unchanged session; this plan is never submitted.
        captured.snapshot.cost_model_version = direct_snapshot.model_version();
        captured.model = direct_snapshot.clone();
        captured.anchor = direct_anchor;
        let context = shape::ExecutorShape {
            engine: &inner,
            captured: &captured,
        };
        let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
        let work = [CandidateWork {
            key: captured.snapshot.requests[0].key.clone(),
            action: WaveAction::Decode,
        }];
        let projected = project(parent.as_ref(), &captured.snapshot.requests, &work).unwrap();
        assert!(matches!(
            projected.host_content_forecasts.as_ref().unwrap().exact(),
            Some(HostContentForecastV2::Exact)
        ));
        let original_exact =
            canonical_cost_shape(projected.canonical_domain.exact().unwrap()).unwrap();
        let compare = Compare {
            direct: AnchoredPlanningCostModel::new(direct_snapshot.as_ref(), direct_anchor),
            catalog: AnchoredPlanningCostModel::new(catalog_snapshot.as_ref(), catalog_anchor),
            original_exact,
            known_original: Cell::new(0),
            queries: Cell::new(0),
            direct_anchor,
            catalog_anchor,
            ttl,
        };
        let planner = BoundedSloPlanner {
            settings: BoundedPlannerSettings {
                search: inner.config.scheduler.slo.planner.clone(),
                ..Default::default()
            },
        };
        let counters = inner.model_executor.cache_metrics_snapshot().unwrap()["counters"].clone();
        let window = captured.budget.planning_window(&captured.origin).unwrap();
        let metrics = query_metrics::QueryMetrics::default();
        let decision = metrics::with_local_recorder(&metrics, || {
            captured
                .origin
                .propose_scoped_with_execution_budget_window(
                    &planner,
                    &captured.snapshot,
                    &compare,
                    &context,
                    None,
                    window,
                    slo_clock_now,
                )
                .unwrap()
        });
        eprintln!("actual runtime V2 reasons: {:?}", metrics.snapshot());
        eprintln!(
            "native planner decision={decision:?}; queries={} known_original={}",
            compare.queries.get(),
            compare.known_original.get()
        );
        assert!(
            compare.known_original.get() > 0,
            "must compare a Known real ordinary exact query, not two Unknown results"
        );
        required_audit::check_native_paths(&context, &captured, &mut |query| {
            use crate::continuous_engine::RequiredFutureAuditCostV2;
            let now = direct.clock.now_ns().unwrap();
            match direct_snapshot.audit_structured_query_v2(query, now) {
                Ok(value) => RequiredFutureAuditCostV2::KnownAtRead {
                    local_now_ns: now,
                    planning_ns: value.planning_ns,
                    valid_for_ns: value.valid_for_ns,
                },
                Err(reason) => RequiredFutureAuditCostV2::Unknown {
                    local_now_ns: Some(now),
                    reason: format!("{reason:?}"),
                },
            }
        });
        assert_no_live_effects(&inner, &captured, &counters);
    }
    while !session.frontiers().unwrap().is_empty() {
        ready(&inner, &id).await;
        let work = frontier(session, &id).decode_work().unwrap();
        wave(session, vec![work]).await;
    }
    // The shared consumer requires original terminal frame, Length and all five
    // completion tokens; a read-only planner result never terminates the request.
    tokio::time::timeout(Duration::from_secs(30), consume)
        .await
        .unwrap()
        .unwrap();
}
