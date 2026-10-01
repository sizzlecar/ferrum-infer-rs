use super::*;

#[test]
fn detailed_query_finite_wide_prediction_and_corrupt_arithmetic_are_distinct() {
    let axes: Vec<_> = (1..=12).map(|x| vec![1, x, 0]).collect();
    let rows: Vec<_> = axes
        .iter()
        .map(|axes| FitSample {
            axes,
            wall_ns: axes[1] * 1_000_000,
        })
        .collect();
    let mut fit = NonNegativeFit::fit(
        &rows,
        &StructuredSettingsV2 {
            max_wave_ns: 100_000_000,
            static_margin_ns: 0,
            ..Default::default()
        },
        EnvelopeSettings::default(),
    )
    .unwrap();
    assert!(fit.predict_fitted_detailed(&[1, 13, 0]).is_ok());
    for (input, expected) in [
        (
            [1, 1_000, 0],
            StructuredQueryFailureV2::OutsidePredictionRange(StructuredUnknown::Capacity),
        ),
        (
            [1, EXACT_INTEGER_LIMIT, 0],
            StructuredQueryFailureV2::OutsidePredictionRange(StructuredUnknown::Numerical),
        ),
        (
            [1, 13, 1],
            StructuredQueryFailureV2::OutsideSupport(StructuredUnknown::UnidentifiedDirection),
        ),
        (
            [1, EXACT_INTEGER_LIMIT + 1, 0],
            StructuredQueryFailureV2::Invalid(StructuredUnknown::InvalidInput),
        ),
    ] {
        assert_eq!(fit.predict_fitted_detailed(&input).unwrap_err(), expected);
        assert_eq!(fit.predict_fitted(&input).unwrap_err(), expected.reason());
    }
    // Negative control: corrupt the private arithmetic operand after a real
    // fit. No source or importer accepts this as a qualified model. Overflow
    // must not acquire the ordinary finite-range exclusion classification.
    fit.coefficients[1] = u128::MAX;
    let input = [1, EXACT_INTEGER_LIMIT, 0];
    assert_eq!(
        fit.predict_fitted_detailed(&input).unwrap_err(),
        StructuredQueryFailureV2::Invalid(StructuredUnknown::Numerical)
    );
    assert_eq!(
        fit.predict_fitted(&input),
        Err(StructuredUnknown::Numerical)
    );
}
