use super::*;

fn rejected(
    snapshot: &snapshot::Snapshot,
    work: &[CostWork],
    execution: CostExecution,
    now: Instant,
    reason: CostUnknownReason,
) {
    let query = CostQuery {
        work,
        work_is_exact: true,
        execution,
    };
    assert_eq!(
        snapshot.predict(&query, now),
        CostPrediction::Unknown(reason)
    );
    assert_eq!(
        snapshot.cost_scenarios(&query, now, 4),
        CostScenarioPrediction::Unknown(reason)
    );
}

#[test]
fn scalar_and_scenarios_report_same_precise_rejection_without_relaxing_support() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let config = config();
        let shared = shared(&config);
        let mut learner = Learner::new(config, identity(), family);
        let end = fill(&mut learner, &shared, Instant::now(), 0.01, 0.02);
        let mut snapshot = learner.snapshot(end).unwrap();
        let known = CostQuery {
            work: &[work()],
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        };
        let expected = snapshot.predict(&known, end);
        assert!(
            matches!(expected, CostPrediction::Known { central_service_time, upper_service_time, .. } if central_service_time == Duration::from_millis(10) && upper_service_time == Duration::from_millis(20))
        );
        let original_model = serde_json::to_value(&*snapshot.domains[0].model).unwrap();
        let detail = |value| CostUnknownReason::UnsupportedWithDetail(value);
        for invalid in [
            CostWork {
                context_tokens: 3,
                ..work()
            },
            CostWork {
                requested: ExecutionRange::new(2, 2).unwrap(),
                ..work()
            },
            CostWork {
                prefix_restore: AttemptPrefixRestore::Restored { tokens: 4 },
                ..work()
            },
            CostWork {
                internally_reused_tokens: Some(5),
                ..work()
            },
        ] {
            rejected(
                &snapshot,
                &[invalid],
                CostExecution::SingleInvocation,
                end,
                detail(CostUnsupportedDetail::InvalidWorkProjection),
            );
        }
        let prefill = CostWork {
            kind: AttemptWorkKind::Prefill,
            requested: ExecutionRange::new(0, 4).unwrap(),
            ..work()
        };
        rejected(
            &snapshot,
            &[prefill, work()],
            CostExecution::SequentialPrefillThenDecode,
            end,
            detail(CostUnsupportedDetail::ExecutionComposition),
        );
        rejected(
            &snapshot,
            &[prefill],
            CostExecution::SingleInvocation,
            end,
            CostUnknownReason::NoCoverage,
        );
        let future = CostWork {
            requested: ExecutionRange::new(4, 1).unwrap(),
            context_tokens: 5,
            ..work()
        };
        rejected(
            &snapshot,
            &[future],
            CostExecution::SingleInvocation,
            end,
            detail(CostUnsupportedDetail::OutsideObservedSupport),
        );
        assert_eq!(snapshot.predict(&known, end), expected);
        assert_eq!(
            serde_json::to_value(&*snapshot.domains[0].model).unwrap(),
            original_model
        );
        snapshot.identity = Arc::new(ExecutionIdentity::new("another-executor".into()).unwrap());
        rejected(
            &snapshot,
            &[work()],
            CostExecution::SingleInvocation,
            end,
            detail(CostUnsupportedDetail::ExecutionIdentityMismatch),
        );
        snapshot.identity = Arc::new(identity());
        // Staleness retains its original class and clock boundary, independent
        // of the new unsupported subcounts.
        rejected(
            &snapshot,
            &[work()],
            CostExecution::SingleInvocation,
            end + Duration::from_secs(30),
            CostUnknownReason::Stale,
        );
        snapshot.domains[0].model_expires_at = end;
        rejected(
            &snapshot,
            &[work()],
            CostExecution::SingleInvocation,
            end,
            CostUnknownReason::Stale,
        );
    }
}
