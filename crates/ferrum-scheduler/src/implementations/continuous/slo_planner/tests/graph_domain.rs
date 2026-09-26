use super::*;
use ferrum_interfaces::execution_cost::*;

#[test]
fn configured_eager_domain_does_not_authorize_cold_or_resident_only_work() {
    assert_eq!(
        validate(
            ActualWaveGraphState::Cold,
            PlanningGraphDomain::ConfiguredPerWave
        ),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    );
    assert_eq!(
        validate(
            ActualWaveGraphState::ConfiguredEager,
            PlanningGraphDomain::SnapshotExact
        ),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    );
    assert_eq!(
        validate(
            ActualWaveGraphState::ConfiguredEager,
            PlanningGraphDomain::ResidentReplayOnly
        ),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    );
}

fn validate(
    graph: ActualWaveGraphState,
    domain: PlanningGraphDomain,
) -> Result<PlanningShapeDomain<WaveExecutionShape>, PlanningUnknownReason> {
    let mut snapshot = snapshot(vec![decode(1)]);
    snapshot.capabilities.graph_state = WaveGraphState::Disabled;
    let rows = [PlanningShapeRow {
        request: &snapshot.requests[0],
        work: ActualRowWork::Decode { kv_tokens: 10 },
    }];
    let mut canonical = TestResolver
        .resolve(
            &PlanningShapeQuery {
                snapshot: &snapshot,
                prior_waves: &[],
                kind: ActualWaveKind::Decode,
                rows: &rows,
                recurrent_state_bytes: 0,
            },
            &mut || Ok(()),
        )
        .unwrap()
        .unwrap();
    canonical.graph = graph;
    shape::validate_domain(
        &snapshot,
        ActualWaveKind::Decode,
        &rows,
        0,
        &PlanningShapeDomain::Exact(canonical),
        domain,
        &mut || Ok(()),
    )
}

#[test]
fn graph_domain_default_exact_rejects_cold_and_warm_mismatch() {
    assert!(validate(
        ActualWaveGraphState::Disabled,
        PlanningGraphDomain::SnapshotExact
    )
    .is_ok());
    for graph in [ActualWaveGraphState::Cold, ActualWaveGraphState::Warm] {
        assert_eq!(
            validate(graph, PlanningGraphDomain::SnapshotExact),
            Err(PlanningUnknownReason::InvalidShapeEvidence)
        );
    }
}

#[test]
fn graph_domain_configured_accepts_real_per_wave_labels_and_rejects_disabled() {
    for graph in [
        ActualWaveGraphState::ConfiguredEager,
        ActualWaveGraphState::Warm,
    ] {
        let result = validate(graph, PlanningGraphDomain::ConfiguredPerWave).unwrap();
        assert_eq!(
            result.exact().unwrap().graph_state,
            match graph {
                ActualWaveGraphState::ConfiguredEager => WaveGraphState::ConfiguredEager,
                _ => WaveGraphState::Warm,
            }
        );
    }
    assert_eq!(
        validate(
            ActualWaveGraphState::Disabled,
            PlanningGraphDomain::ConfiguredPerWave
        ),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    );
}

struct ConfiguredResolver {
    graph: ActualWaveGraphState,
    stale: std::cell::Cell<bool>,
}
impl PlanningShapeResolver for ConfiguredResolver {
    fn graph_domain(&self) -> Result<PlanningGraphDomain, PlanningUnknownReason> {
        if self.stale.get() {
            Err(PlanningUnknownReason::UnknownResourceEvidence)
        } else {
            Ok(PlanningGraphDomain::ConfiguredPerWave)
        }
    }
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        let mut canonical = TestResolver.resolve(query, poll)?.unwrap();
        canonical.graph = self.graph;
        Ok(Some(canonical))
    }
}

#[test]
fn graph_domain_survives_execution_wrappers_and_preserves_selected_canonical() {
    for graph in [
        ActualWaveGraphState::ConfiguredEager,
        ActualWaveGraphState::Warm,
    ] {
        let snapshot = snapshot(vec![decode(1)]);
        let resolver = ConfiguredResolver {
            graph,
            stale: std::cell::Cell::new(false),
        };
        let expected = if graph == ActualWaveGraphState::ConfiguredEager {
            WaveGraphState::ConfiguredEager
        } else {
            WaveGraphState::Warm
        };
        let seen = std::cell::Cell::new(0);
        let model = Model(|shape: &WaveExecutionShape| {
            assert_eq!(shape.graph_state, expected);
            seen.set(seen.get() + 1);
            Some(5)
        });
        let decision = planner(1).propose(&snapshot, &model, &resolver, &mut Clock(100));
        assert!(
            matches!(decision, PlanningDecision::FeasibleWithinHorizon { .. }),
            "{decision:?}"
        );
        assert!(seen.get() > 0);
    }
}

#[test]
fn graph_domain_revoked_before_final_replay_cannot_reuse_a_saved_witness() {
    let snapshot = snapshot(vec![decode(1)]);
    let resolver = ConfiguredResolver {
        graph: ActualWaveGraphState::Warm,
        stale: std::cell::Cell::new(false),
    };
    let model = Model(|_: &WaveExecutionShape| {
        resolver.stale.set(true);
        Some(5)
    });
    let decision = planner(1).propose(&snapshot, &model, &resolver, &mut Clock(100));
    assert!(
        matches!(
            decision,
            PlanningDecision::Unknown {
                reason: PlanningUnknownReason::UnknownResourceEvidence,
                ..
            }
        ),
        "{decision:?}"
    );
}

#[test]
fn graph_domain_startup_ready_requires_warm_and_never_infers_configured_eager() {
    assert!(validate(
        ActualWaveGraphState::Warm,
        PlanningGraphDomain::ResidentReplayOnly
    )
    .is_ok());
    for graph in [ActualWaveGraphState::Disabled, ActualWaveGraphState::Cold] {
        assert_eq!(
            validate(graph, PlanningGraphDomain::ResidentReplayOnly),
            Err(PlanningUnknownReason::InvalidShapeEvidence)
        );
    }
}
