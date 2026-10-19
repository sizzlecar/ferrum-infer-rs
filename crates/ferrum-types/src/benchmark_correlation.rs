//! Typed HTTP correlation shared by the serving endpoint and benchmark clients.
//! This wire identity carries no timing aggregation or benchmark success claim.

use serde::{Deserialize, Serialize};

pub const BENCHMARK_RUN_ID_HEADER: &str = "x-ferrum-benchmark-run-id";
pub const BENCHMARK_CELL_ID_HEADER: &str = "x-ferrum-benchmark-cell-id";
pub const BENCHMARK_REPEAT_INDEX_HEADER: &str = "x-ferrum-benchmark-repeat-index";
pub const BENCHMARK_PHASE_HEADER: &str = "x-ferrum-benchmark-phase";
pub const BENCHMARK_REQUEST_INDEX_HEADER: &str = "x-ferrum-benchmark-request-index";
pub const MAX_BENCHMARK_CORRELATION_ID_LEN: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPhase {
    Warmup,
    Measured,
}

impl BenchmarkPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Measured => "measured",
        }
    }
}

impl std::str::FromStr for BenchmarkPhase {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "warmup" => Ok(Self::Warmup),
            "measured" => Ok(Self::Measured),
            _ => Err("benchmark phase must be 'warmup' or 'measured'".to_string()),
        }
    }
}

/// Typed identity carried from `bench-serve` to request-scoped server events.
/// Indices are zero-based and local to their enclosing cell/repeat/phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkRequestCorrelation {
    pub benchmark_run_id: String,
    pub cell_id: String,
    pub repeat_index: u32,
    pub phase: BenchmarkPhase,
    pub request_index: u32,
}

impl BenchmarkRequestCorrelation {
    pub fn new(
        benchmark_run_id: String,
        cell_id: String,
        repeat_index: u32,
        phase: BenchmarkPhase,
        request_index: u32,
    ) -> Result<Self, String> {
        validate_benchmark_correlation_id("benchmark_run_id", &benchmark_run_id)?;
        validate_benchmark_correlation_id("cell_id", &cell_id)?;
        Ok(Self {
            benchmark_run_id,
            cell_id,
            repeat_index,
            phase,
            request_index,
        })
    }

    pub fn from_header_values(
        benchmark_run_id: Option<&str>,
        cell_id: Option<&str>,
        repeat_index: Option<&str>,
        phase: Option<&str>,
        request_index: Option<&str>,
    ) -> Result<Option<Self>, String> {
        let present = [
            benchmark_run_id.is_some(),
            cell_id.is_some(),
            repeat_index.is_some(),
            phase.is_some(),
            request_index.is_some(),
        ];
        if !present.into_iter().any(|value| value) {
            return Ok(None);
        }
        if !present.into_iter().all(|value| value) {
            return Err("benchmark correlation headers must be provided together".to_string());
        }
        let parse_index = |field: &str, value: &str| {
            value
                .parse::<u32>()
                .map_err(|_| format!("{field} must be an unsigned 32-bit integer"))
        };
        Self::new(
            benchmark_run_id.expect("presence checked").to_string(),
            cell_id.expect("presence checked").to_string(),
            parse_index("repeat_index", repeat_index.expect("presence checked"))?,
            phase.expect("presence checked").parse()?,
            parse_index("request_index", request_index.expect("presence checked"))?,
        )
        .map(Some)
    }
}

fn validate_benchmark_correlation_id(field: &str, value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > MAX_BENCHMARK_CORRELATION_ID_LEN {
        return Err(format!(
            "{field} must contain 1..={MAX_BENCHMARK_CORRELATION_ID_LEN} ASCII characters"
        ));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(format!(
            "{field} may contain only ASCII letters, digits, '-', '_', '.', and ':'"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_correlation_headers_require_complete_valid_identity() {
        let correlation = BenchmarkRequestCorrelation::from_header_values(
            Some("bench-123"),
            Some("cell-0-closed-c8"),
            Some("2"),
            Some("measured"),
            Some("17"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(correlation.benchmark_run_id, "bench-123");
        assert_eq!(correlation.cell_id, "cell-0-closed-c8");
        assert_eq!(correlation.repeat_index, 2);
        assert_eq!(correlation.phase, BenchmarkPhase::Measured);
        assert_eq!(correlation.request_index, 17);

        assert!(BenchmarkRequestCorrelation::from_header_values(
            Some("bench-123"),
            None,
            None,
            None,
            None,
        )
        .is_err());
        assert!(BenchmarkRequestCorrelation::from_header_values(
            Some("bad id"),
            Some("cell-0"),
            Some("0"),
            Some("warmup"),
            Some("0"),
        )
        .is_err());
    }
}
