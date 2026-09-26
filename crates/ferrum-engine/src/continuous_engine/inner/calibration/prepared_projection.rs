//! Calibration-only diagnostic time bound. This grants no execution time or
//! cost-model authority and never changes the original observation clocks.
use super::*;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;
use std::time::Duration;

/// Explicit wall allowance for one manual V2 membership projection. The hard
/// ceiling matches the existing bounded required-future diagnostic allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u64", into = "u64")]
pub struct StructuredPreparedProjectionBudgetV2(NonZeroU64);

impl StructuredPreparedProjectionBudgetV2 {
    pub const MAX_MICROSECONDS: u64 = 30_000_000;

    pub fn new(microseconds: u64) -> Result<Self> {
        let value = NonZeroU64::new(microseconds)
            .filter(|value| value.get() <= Self::MAX_MICROSECONDS)
            .ok_or_else(|| {
                FerrumError::config(
                    "structured Prepared diagnostic budget must be in 1..=30000000 microseconds",
                )
            })?;
        Ok(Self(value))
    }

    pub const fn microseconds(self) -> u64 {
        self.0.get()
    }

    pub fn duration(self) -> Duration {
        Duration::from_micros(self.microseconds())
    }
}

impl TryFrom<u64> for StructuredPreparedProjectionBudgetV2 {
    type Error = FerrumError;
    fn try_from(value: u64) -> Result<Self> {
        Self::new(value)
    }
}
impl From<StructuredPreparedProjectionBudgetV2> for u64 {
    fn from(value: StructuredPreparedProjectionBudgetV2) -> Self {
        value.microseconds()
    }
}

/// A diagnostic for every attempted read, including failed reads. None in
/// `configured_budget_us` means the legacy planner allowance was inherited.
/// These values are never added to execution/host targets or used for sampling.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredPreparedProjectionReportV2 {
    pub configured_budget_us: Option<u64>,
    pub budget_ns: u64,
    pub wall_ns: Option<u64>,
    pub budget_exhausted: bool,
    pub error: Option<String>,
}

impl StructuredPreparedProjectionReportV2 {
    pub(in crate::continuous_engine::inner) fn capture<T>(
        configured: Option<StructuredPreparedProjectionBudgetV2>,
        inherited: Duration,
        mut now: impl FnMut() -> Instant,
        operation: impl FnOnce(Instant, Duration) -> Result<T>,
    ) -> (Result<T>, Self) {
        let allowance = configured.map_or(inherited, |value| value.duration());
        let started = now();
        let mut result = operation(started, allowance);
        let wall_ns = now()
            .checked_duration_since(started)
            .and_then(|elapsed| u64::try_from(elapsed.as_nanos()).ok());
        let budget_ns = u64::try_from(allowance.as_nanos()).unwrap_or(u64::MAX);
        if wall_ns.is_none() && result.is_ok() {
            result = Err(FerrumError::unsupported(
                "structured Prepared diagnostic clock moved backwards",
            ));
        }
        let budget_exhausted = wall_ns.is_some_and(|elapsed| elapsed >= budget_ns);
        if budget_exhausted && result.is_ok() {
            result = Err(FerrumError::unsupported(
                "structured Prepared diagnostic budget exhausted at completion",
            ));
        }
        let report = Self {
            configured_budget_us: configured.map(|value| value.microseconds()),
            budget_ns,
            wall_ns,
            budget_exhausted,
            error: result.as_ref().err().map(ToString::to_string),
        };
        (result, report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_budget_wire_checks_real_positive_and_hard_bounds() {
        for value in [1, StructuredPreparedProjectionBudgetV2::MAX_MICROSECONDS] {
            let budget: StructuredPreparedProjectionBudgetV2 =
                serde_json::from_value(value.into()).unwrap();
            assert_eq!(budget.microseconds(), value);
            assert_eq!(serde_json::to_value(budget).unwrap(), value);
        }
        for value in [
            0,
            StructuredPreparedProjectionBudgetV2::MAX_MICROSECONDS + 1,
            u64::MAX,
        ] {
            assert!(StructuredPreparedProjectionBudgetV2::new(value).is_err());
            assert!(
                serde_json::from_value::<StructuredPreparedProjectionBudgetV2>(value.into())
                    .is_err()
            );
        }
    }

    #[test]
    fn diagnostic_reports_original_attempt_without_extending_inherited_planner_time() {
        let start = Instant::now();
        let explicit = StructuredPreparedProjectionBudgetV2::new(1_000_000).unwrap();
        for configured in [None, Some(explicit)] {
            let mut times = [start, start + Duration::from_micros(3_000)].into_iter();
            let (result, report) = StructuredPreparedProjectionReportV2::capture(
                configured,
                Duration::from_micros(2_000),
                || times.next().unwrap(),
                |observed_start, allowance| {
                    assert_eq!(observed_start, start);
                    assert_eq!(
                        allowance,
                        configured.map_or(Duration::from_micros(2_000), |v| v.duration())
                    );
                    Err::<(), _>(FerrumError::unsupported("original projection failure"))
                },
            );
            assert!(result.is_err());
            assert_eq!(report.wall_ns, Some(3_000_000));
            assert_eq!(report.budget_exhausted, configured.is_none());
            assert!(report
                .error
                .unwrap()
                .contains("original projection failure"));
        }
    }

    #[test]
    fn diagnostic_completion_deadline_is_exclusive_and_fails_closed() {
        let start = Instant::now();
        let allowance = Duration::from_micros(2_000);
        for elapsed_ns in [1_999_999, 2_000_000, 2_000_001] {
            let mut times = [start, start + Duration::from_nanos(elapsed_ns)].into_iter();
            let mut calls = 0;
            let (result, report) = StructuredPreparedProjectionReportV2::capture(
                None,
                allowance,
                || times.next().unwrap(),
                |_, _| {
                    calls += 1;
                    Ok(())
                },
            );
            assert_eq!(calls, 1, "an expired attempt must not be retried");
            assert_eq!(report.wall_ns, Some(elapsed_ns));
            assert_eq!(report.budget_exhausted, elapsed_ns >= 2_000_000);
            assert_eq!(result.is_ok(), elapsed_ns < 2_000_000);
            assert_eq!(report.error.is_none(), elapsed_ns < 2_000_000);
        }
    }

    #[test]
    fn diagnostic_clock_failure_cannot_become_success() {
        let start = Instant::now();
        let mut times = [start, start - Duration::from_nanos(1)].into_iter();
        let (result, report) = StructuredPreparedProjectionReportV2::capture(
            None,
            Duration::from_millis(2),
            || times.next().unwrap(),
            |_, _| Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(report.wall_ns, None);
        assert!(report.error.is_some());
    }
}
