use super::*;

#[test]
fn nonnegative_residual_diagnostic_reports_real_phase_branch_and_keeps_transition_failure() {
    let fit = population(StructuredPhaseV2::Fit, true);
    let residual = population(StructuredPhaseV2::Residual, false);
    let fitted = fit_with(&fit);
    let diagnostic = fitted.diagnose_nonnegative_residual(&residual).unwrap();
    assert_eq!(diagnostic["gate"], "phase_branches");
    assert_eq!(diagnostic["reason"], "QualificationCoverage");
    assert_eq!(diagnostic["early"], false);
    assert_eq!(diagnostic["continuation"], true);
    assert!(matches!(
        fitted.calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        ),
        Err(StructuredUnknown::QualificationCoverage)
    ));
}

#[test]
fn nonnegative_residual_diagnostic_reports_original_pending_axis_without_refitting() {
    let mut fit = population(StructuredPhaseV2::Fit, true);
    for (i, sample) in fit.iter_mut().enumerate() {
        let terminal = [9, 0, 1, 2][i / 4];
        let causes = sample.input.settled_terminal_causes().unwrap().to_vec();
        sample.input = input(terminal, [false, false], 64, 2 + (i % 2) as u64)
            .with_settled_terminal_causes(&causes)
            .unwrap();
    }
    let residual = population(StructuredPhaseV2::Residual, true);
    let fitted = fit_with(&fit);
    let signature = fitted.parameters_signature();
    let diagnostic = fitted.diagnose_nonnegative_residual(&residual).unwrap();
    assert_eq!(diagnostic["gate"], "membership_axes");
    assert_eq!(diagnostic["reason"], "UnidentifiedDirection");
    assert_eq!(
        diagnostic["first_unseen_axis"]["axis_label"],
        "pending.count"
    );
    assert_eq!(diagnostic["first_unseen_axis"]["fit_maximum"], 0);
    assert_eq!(diagnostic["first_unseen_axis"]["actual_value"], 1);
    assert_eq!(
        diagnostic["original_sample"]["ticket"],
        residual[1].membership.offered_ordinal
    );
    assert_eq!(fitted.parameters_signature(), signature);
    assert!(matches!(
        fitted.calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        ),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
}
