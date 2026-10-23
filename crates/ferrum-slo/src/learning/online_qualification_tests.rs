use super::*;

fn observed(
    width: usize,
    context: usize,
    completed_at: Instant,
    millis: u64,
) -> ExecutionAttemptSummary {
    ExecutionAttemptSummary {
        started_at: completed_at - Duration::from_millis(millis),
        completed_at,
        eligibility: ExecutionAttemptEligibility::Completed,
        participants: (0..width)
            .map(|_| ExecutionAttemptParticipant {
                request_id: RequestId::new(),
                kind: AttemptWorkKind::Decode,
                requested: ExecutionRange::new(context - 1, 1).unwrap(),
                context_tokens: context,
                prefix_restore: AttemptPrefixRestore::NoneObserved,
                internally_reused_tokens: None,
                product_failed: false,
                outcome: AttemptParticipantOutcome::Completed {
                    work: ExecutedTokenWork::Decode {
                        position: context - 1,
                    },
                    frontier_committed: true,
                },
            })
            .collect(),
    }
}

fn trained(family: ModelFamily) -> (Learner, Shared, Instant) {
    let config = config();
    let shared = shared(&config);
    let mut learner = Learner::new(config, identity(), family);
    let start = Instant::now();
    for i in 0..8 {
        let at = start + Duration::from_millis(i * 10);
        learner.apply(&shared, &observed(4, 16, at, 2), at);
    }
    assert_eq!(shared.stats().model_fits, 1);
    learner.config.retrain_every = usize::MAX;
    (learner, shared, start + Duration::from_millis(70))
}

fn predict_work(snapshot: &snapshot::Snapshot, work: &[CostWork], at: Instant) -> CostPrediction {
    snapshot.predict(
        &CostQuery {
            work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        },
        at,
    )
}

fn parameters(model: &FittedModel) -> serde_json::Value {
    let mut value = serde_json::to_value(model).unwrap();
    value.as_object_mut().unwrap().remove("support");
    value
}

#[test]
fn real_new_point_qualifies_fixed_model_without_changing_old_snapshot_or_unseen_bounds() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let (mut learner, shared, end) = trained(family);
        let old = learner.snapshot(end).unwrap();
        let old_model = old.domains[0].model.clone();
        let old_parameters = parameters(&old_model);
        let expiry = old.domains[0].model_expires_at;
        let before = shared.stats();
        let at = end + Duration::from_millis(100);
        let observation = observed(1, 16, at, 50);
        let work: Vec<_> = observation.participants.iter().map(sample_work).collect();
        assert_eq!(
            predict_work(&old, &work, at),
            CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport
            ))
        );

        assert!(learner.apply(&shared, &observation, at));
        let new = learner.snapshot(at).unwrap();
        assert!(new.revision > old.revision);
        assert_eq!(new.domains[0].model_revision, old.domains[0].model_revision);
        assert_eq!(new.domains[0].model_expires_at, expiry);
        assert_eq!(parameters(&new.domains[0].model), old_parameters);
        assert!(!Arc::ptr_eq(&old_model, &new.domains[0].model));
        assert!(
            matches!(predict_work(&new, &work, at), CostPrediction::Known { observed_at, .. } if observed_at == at)
        );
        assert_eq!(
            predict_work(&old, &work, at),
            CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport
            ))
        );
        let future = observed(1, 17, at, 50);
        let future_work: Vec<_> = future.participants.iter().map(sample_work).collect();
        assert_eq!(
            predict_work(&new, &future_work, at),
            CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport
            ))
        );
        assert_eq!(
            shared.stats().worker_base_prediction_known,
            before.worker_base_prediction_known
        );
        assert_eq!(
            shared.stats().worker_base_prediction_unknown,
            before.worker_base_prediction_unknown + 1
        );
        assert_eq!(
            shared.stats().calibration_updates,
            before.calibration_updates + 1
        );

        // Repeated in-envelope updates retain the no-copy fast path.
        let model = new.domains[0].model.clone();
        let later = at + Duration::from_millis(100);
        assert!(learner.apply(&shared, &observed(1, 16, later, 40), later));
        assert!(Arc::ptr_eq(
            &model,
            &learner.domains[0].active.as_ref().unwrap().model
        ));
        assert_eq!(
            learner.domains[0].active.as_ref().unwrap().model_expires_at,
            expiry
        );
    }
}

#[test]
fn observed_labels_change_only_signed_residuals_not_fitted_parameters_or_base() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let mut results = Vec::new();
        for millis in [50, 100] {
            let (mut learner, shared, end) = trained(family);
            let at = end + Duration::from_millis(100);
            let observation = observed(1, 16, at, millis);
            let work: Vec<_> = observation.participants.iter().map(sample_work).collect();
            assert!(learner.apply(&shared, &observation, at));
            let snapshot = learner.snapshot(at).unwrap();
            let CostPrediction::Known {
                upper_service_time, ..
            } = predict_work(&snapshot, &work, at)
            else {
                panic!("qualified measured point");
            };
            results.push((
                parameters(&snapshot.domains[0].model),
                snapshot.domains[0]
                    .model
                    .predict(&identity(), &WorkFeatures::extract(&work, 4).unwrap())
                    .unwrap(),
                upper_service_time,
            ));
        }
        assert_eq!(results[0].0, results[1].0);
        assert_eq!(results[0].1, results[1].1);
        assert!(results[0].2 < results[1].2);
    }
}

#[test]
fn invalid_late_and_expired_facts_do_not_expand_support() {
    #[derive(Clone, Copy, Debug)]
    enum Rejection {
        Interrupted,
        Uncommitted,
        Partial,
        Future,
        OutOfOrder,
        ExpiredModel,
        ExpiredObservation,
    }
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        for case in [
            Rejection::Interrupted,
            Rejection::Uncommitted,
            Rejection::Partial,
            Rejection::Future,
            Rejection::OutOfOrder,
            Rejection::ExpiredModel,
            Rejection::ExpiredObservation,
        ] {
            let (mut learner, shared, end) = trained(family);
            let old = learner.domains[0].active.as_ref().unwrap().model.clone();
            let expiry = learner.domains[0].active.as_ref().unwrap().model_expires_at;
            let at = end + Duration::from_millis(100);
            let mut observation = observed(1, 16, at, 50);
            let mut now = at;
            match case {
                Rejection::Interrupted => {
                    observation.eligibility = ExecutionAttemptEligibility::Interrupted
                }
                Rejection::Uncommitted => {
                    observation.participants[0].outcome = AttemptParticipantOutcome::Completed {
                        work: ExecutedTokenWork::Decode { position: 15 },
                        frontier_committed: false,
                    }
                }
                Rejection::Partial => {
                    observation.participants[0].outcome = AttemptParticipantOutcome::Completed {
                        work: ExecutedTokenWork::Decode { position: 14 },
                        frontier_committed: true,
                    }
                }
                Rejection::Future => now = at - Duration::from_millis(1),
                Rejection::OutOfOrder => {
                    observation = observed(1, 16, end - Duration::from_millis(1), 2)
                }
                Rejection::ExpiredModel => now = expiry,
                Rejection::ExpiredObservation => {
                    now = at + Duration::from_millis(learner.config.calibration_ttl_ms)
                }
            }
            assert!(
                !learner.apply(&shared, &observation, now),
                "{family:?} case {case:?}"
            );
            let active = learner.domains[0].active.as_ref().unwrap();
            assert!(Arc::ptr_eq(&old, &active.model), "{family:?} case {case:?}");
            assert_eq!(active.model_expires_at, expiry);
        }
    }
}

#[test]
fn restored_model_qualifies_only_real_observations_after_its_load_cutoff() {
    use crate::learning::state::{RestoredDomain, RestoredState, StateLoadReport, StateLoadStatus};
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let (mut learner, shared, end) = trained(family);
        let active = learner.domains.pop().unwrap().active.unwrap();
        let original_model = active.model.clone();
        let expiry = active.model_expires_at;
        let loaded_at = end + Duration::from_millis(100);
        let restored = RestoredState {
            family,
            revision: 1,
            domains: vec![RestoredDomain {
                model: active.model,
                model_revision: active.model_revision,
                model_expires_at: expiry,
                calibration: active.calibration,
            }],
            report: StateLoadReport {
                status: StateLoadStatus::Loaded,
                loaded_domains: 1,
                expired_domains: 0,
                expired_residual_entries: 0,
            },
            loaded_at,
        };
        let mut restored = Learner::restore(learner.config, identity(), family, restored);
        let prior = observed(1, 16, loaded_at - Duration::from_millis(1), 2);
        assert!(!restored.apply(&shared, &prior, loaded_at));
        assert!(Arc::ptr_eq(
            &original_model,
            &restored.domains[0].active.as_ref().unwrap().model
        ));
        let fresh_at = loaded_at + Duration::from_millis(100);
        assert!(restored.apply(&shared, &observed(1, 16, fresh_at, 50), fresh_at));
        let active = restored.domains[0].active.as_ref().unwrap();
        assert!(!Arc::ptr_eq(&original_model, &active.model));
        assert_eq!(active.model_expires_at, expiry);
        assert_eq!(active.model_revision, 1);
    }
}

#[test]
fn identity_domain_invalid_features_and_rejected_residual_leave_envelope_unchanged() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let (mut learner, shared, end) = trained(family);
        let old = learner.domains[0].active.as_ref().unwrap().model.clone();
        let observation = observed(1, 16, end, 50);
        let work: Vec<_> = observation.participants.iter().map(sample_work).collect();
        let features = WorkFeatures::extract(&work, 4).unwrap();
        let other_identity = ExecutionIdentity::new("other-executor".into()).unwrap();
        assert!(old
            .qualify_observed_copy(&other_identity, &features)
            .is_err());
        let prefill = CostWork {
            kind: AttemptWorkKind::Prefill,
            ..work[0]
        };
        assert!(old
            .qualify_observed_copy(&identity(), &WorkFeatures::extract(&[prefill], 4).unwrap())
            .is_err());
        let mut invalid = serde_json::to_value(features).unwrap();
        invalid["values"][1] = (-1.0).into();
        let invalid: WorkFeatures = serde_json::from_value(invalid).unwrap();
        assert!(old.qualify_observed_copy(&identity(), &invalid).is_err());

        // Rejected labels cannot publish geometry without a residual.
        let at = end + Duration::from_millis(100);
        assert!(!learner.observe(
            &shared,
            TrainingSample {
                features,
                service_seconds: f64::NAN
            },
            at,
            at
        ));
        assert!(Arc::ptr_eq(
            &old,
            &learner.domains[0].active.as_ref().unwrap().model
        ));
        assert!(old.predict(&identity(), &features).is_err());
    }
}
