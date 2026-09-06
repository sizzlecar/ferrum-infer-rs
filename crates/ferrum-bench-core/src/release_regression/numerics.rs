//! Pure numerical replay shared by live backend probes and release consumers.
//! No hardware probing, dispatch or execution status lives here. Raw arrays and
//! recomputed metrics require separately bound execution provenance.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Exact IEEE-754 output representation, including non-finite and signed-zero
/// values. Unlike JSON float arrays this does not silently turn NaN into null.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawOutput {
    pub f32_bits: Vec<u32>,
}

impl RawOutput {
    pub fn from_f32(values: &[f32]) -> Self {
        Self {
            f32_bits: values.iter().map(|value| value.to_bits()).collect(),
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.f32_bits
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalMetrics {
    pub element_count: usize,
    pub nmse: f64,
    pub max_abs: f64,
    pub reference_mse: f64,
    /// Matches the existing op-diff definition: for reference MSE < 1e-30,
    /// the reported error is absolute MSE rather than a normalized ratio.
    pub uses_absolute_mse: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalFailure {
    pub reason: String,
    pub metrics: Option<NumericalMetrics>,
}

impl fmt::Display for NumericalFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for NumericalFailure {}

/// Normalized mean-squared error.
///
/// NMSE = mse(a, b) / mse(a, 0). Returns the raw `mse(a, b)` when the
/// reference is degenerate (all zeros) — falls back gracefully so tests
/// for ops that legitimately output zero don't divide by zero.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
pub fn nmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "nmse: length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let mse_ab: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum::<f64>()
        / n;
    let mse_a0: f64 = a
        .iter()
        .map(|x| {
            let d = *x as f64;
            d * d
        })
        .sum::<f64>()
        / n;
    if mse_a0 < 1e-30 {
        return mse_ab;
    }
    mse_ab / mse_a0
}

fn failure(reason: impl Into<String>) -> NumericalFailure {
    NumericalFailure {
        reason: reason.into(),
        metrics: None,
    }
}

/// Reject invalid tolerances before probing or dispatching a backend.
pub fn validate_tolerance(tolerance: f64) -> Result<(), NumericalFailure> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(failure("tolerance must be finite and strictly positive"));
    }
    Ok(())
}

/// Require an actual nonempty finite CPU reference before backend execution.
pub fn validate_reference(reference: &[f32]) -> Result<(), NumericalFailure> {
    if reference.is_empty() {
        return Err(failure("CPU reference output is empty"));
    }
    if let Some(index) = reference.iter().position(|value| !value.is_finite()) {
        return Err(failure(format!(
            "CPU reference contains a non-finite value at {index}"
        )));
    }
    Ok(())
}

/// Shared pure comparator for live execution and raw-output replay. A finite,
/// positive tolerance is required before observing results; equality to the
/// threshold fails, matching the existing op-diff strict inequality.
pub fn compare_outputs(
    reference: &[f32],
    actual: &[f32],
    tolerance: f64,
) -> Result<NumericalMetrics, NumericalFailure> {
    validate_tolerance(tolerance)?;
    validate_reference(reference)?;
    if actual.is_empty() {
        return Err(failure("required backend output is empty"));
    }
    if actual.len() != reference.len() {
        return Err(failure(format!(
            "output length mismatch: reference={} actual={}",
            reference.len(),
            actual.len()
        )));
    }
    if let Some(index) = actual.iter().position(|value| !value.is_finite()) {
        return Err(failure(format!(
            "required backend contains a non-finite value at {index}"
        )));
    }
    let nmse = nmse(reference, actual);
    let reference_mse = reference
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        / reference.len() as f64;
    let max_abs = reference
        .iter()
        .zip(actual)
        .map(|(expected, observed)| (f64::from(*expected) - f64::from(*observed)).abs())
        .fold(0.0, f64::max);
    if !nmse.is_finite() || nmse < 0.0 || !reference_mse.is_finite() || !max_abs.is_finite() {
        return Err(failure("numerical measurement is non-finite or invalid"));
    }
    let metrics = NumericalMetrics {
        element_count: reference.len(),
        nmse,
        max_abs,
        reference_mse,
        uses_absolute_mse: reference_mse < 1e-30,
    };
    if nmse >= tolerance {
        return Err(NumericalFailure {
            reason: format!("NMSE {nmse} is not below tolerance {tolerance}; max_abs={max_abs}"),
            metrics: Some(metrics),
        });
    }
    Ok(metrics)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionPhase {
    Initial,
    ReusedContext,
    IndependentContext,
    DropFlush,
}

pub const SUBMISSION_PHASES: [SubmissionPhase; 4] = [
    SubmissionPhase::Initial,
    SubmissionPhase::ReusedContext,
    SubmissionPhase::IndependentContext,
    SubmissionPhase::DropFlush,
];

/// Replay each declared legacy-context lifecycle segment independently.
/// The caller binds the segment length and raw arrays to the actual fixture,
/// backend and candidate. This comparison cannot attest that a GPU executed.
pub fn compare_submission_segments(
    segment_len: usize,
    reference: &[f32],
    actual: &[f32],
    tolerance: f64,
) -> Result<Vec<NumericalMetrics>, String> {
    if segment_len == 0 {
        return Err("context segment length must be nonzero".into());
    }
    let expected = segment_len
        .checked_mul(SUBMISSION_PHASES.len())
        .ok_or("context output element count overflow")?;
    if reference.len() != expected || actual.len() != expected {
        return Err(format!(
            "context output shape: expected {expected}, reference {}, actual {}",
            reference.len(),
            actual.len()
        ));
    }
    SUBMISSION_PHASES
        .iter()
        .enumerate()
        .map(|(index, phase)| {
            let range = index * segment_len..(index + 1) * segment_len;
            compare_outputs(&reference[range.clone()], &actual[range], tolerance)
                .map_err(|error| format!("{phase:?}: {error}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_equal_outputs_pass_and_zero_reference_uses_absolute_mse() {
        let metric = compare_outputs(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0], 1e-7).unwrap();
        assert_eq!(metric.nmse, 0.0);
        assert_eq!(metric.element_count, 3);
        assert!(!metric.uses_absolute_mse);
        let zero = compare_outputs(&[0.0, 0.0], &[0.0, 0.0], 1e-7).unwrap();
        assert!(zero.uses_absolute_mse);
        assert_eq!(zero.max_abs, 0.0);
    }

    #[test]
    fn invalid_arrays_and_tolerances_never_produce_a_numerical_pass() {
        for (reference, actual) in [
            (vec![], vec![]),
            (vec![1.0], vec![]),
            (vec![1.0], vec![1.0, 2.0]),
            (vec![f32::NAN], vec![1.0]),
            (vec![1.0], vec![f32::NAN]),
            (vec![f32::INFINITY], vec![1.0]),
            (vec![1.0], vec![f32::NEG_INFINITY]),
        ] {
            assert!(compare_outputs(&reference, &actual, 1e-7).is_err());
        }
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(compare_outputs(&[1.0], &[1.0], tolerance).is_err());
        }
    }

    #[test]
    fn perturbed_output_fails_and_retains_the_measured_error() {
        let failure = compare_outputs(&[1.0, 2.0], &[2.0, 4.0], 1e-7).unwrap_err();
        let metrics = failure.metrics.unwrap();
        assert_eq!(metrics.nmse, 1.0);
        assert_eq!(metrics.max_abs, 2.0);
        assert!(compare_outputs(&[1.0], &[2.0], 1.0).is_err());
    }

    #[test]
    fn raw_json_roundtrip_preserves_non_finite_payloads_and_signed_zero() {
        let bits = vec![
            0x7fc0_1234,
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            (-0.0_f32).to_bits(),
            1.0_f32.to_bits(),
        ];
        let raw = RawOutput {
            f32_bits: bits.clone(),
        };
        let json = serde_json::to_string(&raw).unwrap();
        let decoded: RawOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.f32_bits, bits);
        assert_eq!(RawOutput::from_f32(&decoded.to_f32()), raw);
    }

    #[test]
    fn nmse_identical_is_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(nmse(&a, &a) < 1e-30);
    }

    #[test]
    fn nmse_scaled_b_proportional() {
        // b = 1.01 * a → relative error 0.01, NMSE ≈ 0.0001
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = a.iter().map(|x| x * 1.01).collect();
        let n = nmse(&a, &b);
        // NMSE = mse(0.01*a, 0) / mse(a, 0) = 0.0001
        assert!((n - 1e-4).abs() < 1e-5);
    }

    #[test]
    fn nmse_zero_reference_falls_back() {
        // a all-zero: NMSE returns raw MSE.
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.1, 0.1, 0.1];
        let n = nmse(&a, &b);
        assert!((n - 0.01).abs() < 1e-9);
    }

    #[test]
    fn submission_segments_reject_hidden_failure_missing_data_and_invalid_lengths() {
        let reference = [1e8, 1.0, 2.0, 3.0];
        let actual = [1e8, 0.0, 2.0, 3.0];
        assert!(compare_outputs(&reference, &actual, 1e-7).is_ok());
        let error = compare_submission_segments(1, &reference, &actual, 1e-7).unwrap_err();
        assert!(error.contains("ReusedContext"), "{error}");
        let metrics = compare_submission_segments(1, &reference, &reference, 1e-7).unwrap();
        assert_eq!(metrics.len(), SUBMISSION_PHASES.len());
        assert!(metrics.iter().all(|metric| metric.nmse == 0.0));
        for len in 0..reference.len() {
            assert!(
                compare_submission_segments(1, &reference[..len], &reference[..len], 1e-7).is_err()
            );
        }
        assert!(compare_submission_segments(0, &[], &[], 1e-7).is_err());
        assert!(compare_submission_segments(usize::MAX, &[], &[], 1e-7).is_err());
        assert!(compare_submission_segments(1, &reference, &reference, f64::NAN).is_err());
        let mut nonfinite = reference;
        nonfinite[3] = f32::NAN;
        assert!(compare_submission_segments(1, &reference, &nonfinite, 1e-7).is_err());
    }

    #[test]
    fn raw_nmse_keeps_empty_and_length_mismatch_semantics() {
        assert_eq!(nmse(&[], &[]), 0.0);
        assert!(std::panic::catch_unwind(|| nmse(&[1.0], &[])).is_err());
        // The raw primitive remains permissive; strict consumption uses compare_outputs.
        assert!(nmse(&[f32::NAN], &[1.0]).is_nan());
        assert!(compare_outputs(&[], &[], 1e-7).is_err());
    }

    #[test]
    fn submission_phase_wire_names_roundtrip_and_unknown_is_rejected() {
        let json = serde_json::to_string(&SUBMISSION_PHASES).unwrap();
        let phases: Vec<SubmissionPhase> = serde_json::from_str(&json).unwrap();
        assert_eq!(phases, SUBMISSION_PHASES);
        assert!(serde_json::from_str::<SubmissionPhase>("\"unexecuted\"").is_err());
    }
}
