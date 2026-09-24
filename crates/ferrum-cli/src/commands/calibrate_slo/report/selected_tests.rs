use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::statistical::model::{
    HeldoutEvaluationV1, ModelUnknown, WholeWavePredictionV1,
};

#[test]
fn selected_heldout_report_counts_actual_underestimates_without_legacy_fallback() {
    let mut totals = Summary::default();
    let evidence = selected_prediction(
        Ok(HeldoutEvaluationV1 {
            prediction: Ok(WholeWavePredictionV1 {
                fitted_ns: 70,
                residual_ns: 15,
                static_margin_ns: 5,
                planning_ns: 90,
                valid_until_ns: 1000,
                fit_samples: 12,
                residual_samples: 9,
            }),
            actual_ns: 105,
            underestimate_ns: Some(15),
        }),
        &mut totals,
    );
    assert_eq!(evidence["underestimate_ns"], 15);
    assert_eq!(totals.selected_validation_offered, 1);
    assert_eq!(totals.selected_validation_known, 1);
    assert_eq!(totals.selected_validation_underestimates, 1);
    assert_eq!(totals.validation_known, 0);
    assert_eq!(totals.host_content_validation_known, 0);
}

#[test]
fn selected_heldout_report_retains_missing_receipts_and_unsupported_families() {
    let mut totals = Summary::default();
    let absent = selected_prediction(Err(ModelUnknown::WrongSource), &mut totals);
    let unsupported = selected_prediction(
        Ok(HeldoutEvaluationV1 {
            prediction: Err(ModelUnknown::InsufficientResidual),
            actual_ns: 123,
            underestimate_ns: None,
        }),
        &mut totals,
    );
    assert!(absent["actual_ns"].is_null());
    assert_eq!(unsupported["actual_ns"], 123);
    assert_eq!(totals.selected_validation_offered, 2);
    assert_eq!(totals.selected_validation_unknown, 2);
    assert_eq!(
        totals
            .selected_validation_unknown_reasons
            .get("WrongSource"),
        Some(&1)
    );
    assert_eq!(
        totals
            .selected_validation_unknown_reasons
            .get("InsufficientResidual"),
        Some(&1)
    );
}
