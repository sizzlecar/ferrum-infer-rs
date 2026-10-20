//! Shared full-vocabulary engineering qualification for actual model tests.

/// Preserve the existing managed/qualification test envelope. This is not a
/// correctly-rounded or cross-shape equivalence guarantee: callers supply an
/// independent reference and separately enforce their complete vocabulary size.
pub(crate) fn assert_full_logits_qualification(actual: &[f32], expected: &[f32]) {
    assert!(!expected.is_empty(), "a full vocabulary oracle is required");
    assert_eq!(actual.len(), expected.len());
    let mut magnitude = 0.0_f32;
    let mut maximum_error = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        magnitude = magnitude.max(expected.abs());
        maximum_error = maximum_error.max((actual - expected).abs());
    }
    assert!(
        maximum_error <= 0.02 * magnitude.max(1e-6),
        "full-logits error={maximum_error}, reference magnitude={magnitude}"
    );
}
