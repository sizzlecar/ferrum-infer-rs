use super::*;

fn fitted(family: ModelFamily) -> FittedModel {
    FittedModel::train(
        family,
        identity(),
        &[sample(1, 0.001), sample(4, 0.004)],
        &[sample(2, 0.002)],
        &compact_budget(),
    )
    .unwrap()
}

#[test]
fn detailed_rejection_preserves_predictions_support_and_legacy_error_classes() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let fit = fitted(family);
        let original = serde_json::to_value(&fit).unwrap();
        let known = prefill(2);
        assert_eq!(
            fit.predict_detailed(&identity(), &known).unwrap(),
            fit.predict(&identity(), &known).unwrap()
        );
        let other = ExecutionIdentity::new("another-executor".into()).unwrap();
        assert_eq!(
            fit.predict_detailed(&other, &prefill(5)),
            Err(PredictionRejection::ExecutionIdentityMismatch)
        );
        let other_domain = WorkFeatures::extract(
            &[CostWork {
                kind: AttemptWorkKind::Decode,
                requested: ExecutionRange::new(1, 1).unwrap(),
                context_tokens: 2,
                prefix_restore: AttemptPrefixRestore::NoneObserved,
                internally_reused_tokens: None,
            }],
            1,
        )
        .unwrap();
        assert_eq!(
            fit.predict_detailed(&identity(), &other_domain),
            Err(PredictionRejection::FeatureDomainMismatch)
        );
        assert_eq!(
            fit.predict_detailed(&identity(), &prefill(5)),
            Err(PredictionRejection::OutsideObservedSupport)
        );
        assert_eq!(
            fit.predict(&other, &prefill(5)),
            Err(ModelError::UnsupportedDomain)
        );
        assert_eq!(
            fit.predict(&identity(), &other_domain),
            Err(ModelError::UnsupportedDomain)
        );
        assert_eq!(
            fit.predict(&identity(), &prefill(5)),
            Err(ModelError::UnsupportedDomain)
        );
        assert_eq!(
            serde_json::to_value(&fit).unwrap(),
            original,
            "diagnosis must not qualify an unobserved point or mutate coefficients"
        );
    }
}

#[test]
fn detailed_rejection_separates_invalid_numeric_model_output_from_support() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let mut fit = fitted(family);
        // Deliberately inject invalid in-memory coefficients in this test. Normal
        // persisted-model loading rejects these; the query guard must also fail
        // closed if finite operands overflow during fixed-model evaluation.
        let mut predictor = serde_json::to_value(&fit.predictor).unwrap();
        match family {
            ModelFamily::Physical => {
                predictor["Physical"]["coefficients"] =
                    serde_json::to_value([f64::MAX; FEATURE_COUNT + 1]).unwrap();
            }
            ModelFamily::Gbdt => {
                predictor["Gbdt"]["learning_rate"] = serde_json::json!(1.0);
                predictor["Gbdt"]["center"]["initial"] = serde_json::json!(f64::MAX);
                predictor["Gbdt"]["center"]["trees"] =
                    serde_json::json!([{"nodes":[{"value":f64::MAX,"split":null}]}]);
            }
        }
        fit.predictor = serde_json::from_value(predictor).unwrap();
        let expected = ModelError::InvalidData("invalid model prediction");
        assert_eq!(
            fit.predict_detailed(&identity(), &prefill(2)),
            Err(PredictionRejection::InvalidBasePrediction(expected.clone()))
        );
        assert_eq!(fit.predict(&identity(), &prefill(2)), Err(expected));
    }
}
